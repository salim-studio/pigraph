"""Checkpoint — MemorySaver + SqliteSaver (مطابق لـ langgraph.checkpoint)."""
from __future__ import annotations

import copy
import pickle
import sqlite3
import threading
import time
from threading import RLock

__all__ = ["Checkpoint", "MemorySaver", "InMemorySaver", "MemorySaverCheckpoint",
           "SqliteSaver", "BaseCheckpointer"]


class Checkpoint:
    __slots__ = ("thread_id", "step", "state", "next_nodes")

    def __init__(self, thread_id: str, step: int, state: dict, next_nodes: tuple = ()):
        self.thread_id = thread_id
        self.step = step
        self.state = state
        self.next_nodes = next_nodes


class BaseCheckpointer:
    def get(self, thread_id: str): raise NotImplementedError
    def put(self, thread_id: str, state: dict, step: int): raise NotImplementedError


class MemorySaver(BaseCheckpointer):
    """حفظ محلي سريع thread-safe: thread_id -> قائمة لقطات."""

    def __init__(self):
        self._store: dict[str, list[dict]] = {}
        self._lock = RLock()

    def put(self, thread_id: str, state: dict, step: int):
        snap = copy.deepcopy(state)
        with self._lock:
            self._store.setdefault(thread_id, []).append({"step": step, "state": snap})

    def get(self, thread_id: str) -> dict | None:
        with self._lock:
            hist = self._store.get(thread_id)
            if not hist:
                return None
            return copy.deepcopy(hist[-1]["state"])

    def get_tuple(self, thread_id: str):
        with self._lock:
            hist = self._store.get(thread_id)
            if not hist:
                return None
            return copy.deepcopy(hist[-1])

    def history(self, thread_id: str) -> list[dict]:
        with self._lock:
            return copy.deepcopy(self._store.get(thread_id, []))

    def clear(self, thread_id: str | None = None):
        with self._lock:
            if thread_id is None:
                self._store.clear()
            else:
                self._store.pop(thread_id, None)


class SqliteSaver(BaseCheckpointer):
    """حفظ دائم عبر sqlite3 (مطابق لـ langgraph.checkpoint.sqlite.SqliteSaver).

    مثال:
        with SqliteSaver("checkpoints.db") as saver:
            app = g.compile(checkpointer=saver)
    """

    def __init__(self, path: str = ":memory:"):
        self.path = path
        self._lock = RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        with self._lock, self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS checkpoints "
                "(thread_id TEXT, step INTEGER, ts REAL, state BLOB, "
                "PRIMARY KEY (thread_id, step))"
            )

    def put(self, thread_id: str, state: dict, step: int):
        blob = pickle.dumps(copy.deepcopy(state), protocol=4)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO checkpoints VALUES (?,?,?,?)",
                (thread_id, step, time.time(), blob),
            )

    def get(self, thread_id: str) -> dict | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT state FROM checkpoints WHERE thread_id=? ORDER BY step DESC LIMIT 1",
                (thread_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return pickle.loads(row[0])

    def history(self, thread_id: str) -> list[dict]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT step, state FROM checkpoints WHERE thread_id=? ORDER BY step",
                (thread_id,),
            )
            return [{"step": s, "state": pickle.loads(b)} for s, b in cur.fetchall()]

    def clear(self, thread_id: str | None = None):
        with self._lock, self._conn:
            if thread_id is None:
                self._conn.execute("DELETE FROM checkpoints")
            else:
                self._conn.execute("DELETE FROM checkpoints WHERE thread_id=?", (thread_id,))

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


# توافق اسمي
MemorySaverCheckpoint = MemorySaver
InMemorySaver = MemorySaver
