"""Database — unified access for analysts & developers (zero hard deps).

يدعم sqlite3 دائماً (stdlib)، و duckdb / sqlalchemy / pandas عند توفرها —
بدون كسر الاستيراد إن غابت.

أمثلة:
    from pigraph.db import Database
    db = Database("sqlite:///my.db")   # أو ":memory:" أو "data.db"
    db.execute("CREATE TABLE t (id INTEGER, name TEXT)")
    db.executemany("INSERT INTO t VALUES (?,?)", [(1,"a"),(2,"b")])
    print(db.query("SELECT * FROM t"))
    df = db.query_df("SELECT * FROM t")   # يحتاج pandas

    # داخل رسم بياني:
    from pigraph import StateGraph, START, END
    g = StateGraph(dict)
    g.add_node("load", db.read_node("SELECT * FROM t", out_key="rows"))
    g.add_edge(START, "load"); g.add_edge("load", END)
"""
from __future__ import annotations

import csv
import json
import os
import sqlite3
import threading

__all__ = ["Database", "sql_node", "csv_loader_node", "JsonFileSaver"]


def _has(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


class Database:
    """موحّد قواعد البيانات: sqlite افتراضياً + duckdb/sqlalchemy اختيارياً.

    Parameters:
        url: ":memory:" | "file.db" | "sqlite:///file.db" |
             "duckdb:///file.ddb" | "sqlite:///:memory:" |
             أي URL تدعمه sqlalchemy (postgresql://... إلخ)
    """

    def __init__(self, url: str = ":memory:"):
        self.url = url or ":memory:"
        self._lock = threading.RLock()
        self._kind, self._path = self._parse(url)
        self._sqlite: sqlite3.Connection | None = None
        self._duck = None
        self._sa_engine = None
        if self._kind == "sqlite":
            self._sqlite = sqlite3.connect(self._path, check_same_thread=False)
            self._sqlite.row_factory = sqlite3.Row
        elif self._kind == "duckdb":
            import duckdb  # type: ignore  # optional dep
            self._duck = duckdb.connect(self._path or ":memory:")
        elif self._kind == "sqlalchemy":
            import sqlalchemy as _sa  # type: ignore  # optional dep
            self._sa_engine = _sa.create_engine(self.url)

    # -- parsing --
    @staticmethod
    def _parse(url: str):
        u = url.strip()
        if u in (":memory:", "sqlite:///:memory:"):
            return "sqlite", ":memory:"
        if u.startswith("duckdb://"):
            path = u[len("duckdb://"):] or ":memory:"
            return "duckdb", path
        if u.startswith("sqlite:///"):
            return "sqlite", u[len("sqlite:///"):] or ":memory:"
        if u.startswith("sqlite://"):
            return "sqlite", u[len("sqlite://"):] or ":memory:"
        if u.endswith((".db", ".sqlite", ".sqlite3")) or u == ":memory:" or "/" not in u and "://" not in u:
            return "sqlite", u
        return "sqlalchemy", u

    # -- core ops --
    def execute(self, sql: str, params=()) -> int:
        """تنفيذ INSERT/CREATE/... — يعيد عدد الصفوف المتأثرة."""
        with self._lock:
            if self._kind == "sqlite":
                assert self._sqlite is not None
                with self._sqlite:
                    cur = self._sqlite.execute(sql, params or ())
                    return cur.rowcount
            if self._kind == "duckdb":
                self._duck.execute(sql, params or None)  # type: ignore
                try:
                    return self._duck.fetchall().__len__()  # type: ignore
                except Exception:
                    return 0
            # sqlalchemy
            from sqlalchemy import text as _text  # type: ignore
            with self._sa_engine.begin() as c:  # type: ignore
                r = c.execute(_text(sql), params or {})
                try:
                    return r.rowcount
                except Exception:
                    return 0

    def executemany(self, sql: str, rows) -> int:
        rows = list(rows)
        if not rows:
            return 0
        with self._lock:
            if self._kind == "sqlite":
                assert self._sqlite is not None
                with self._sqlite:
                    cur = self._sqlite.executemany(sql, rows)
                    return cur.rowcount
            if self._kind == "duckdb":
                self._duck.executemany(sql, rows)  # type: ignore
                return len(rows)
            from sqlalchemy import text as _text  # type: ignore
            with self._sa_engine.begin() as c:  # type: ignore
                # sqlalchemy executemany عبر loop بسيط
                n = 0
                for r in rows:
                    c.execute(_text(sql), r if isinstance(r, dict) else {"p%d" % i: v for i, v in enumerate(r)})
                    n += 1
                return n

    def query(self, sql: str, params=()) -> list[dict]:
        """استعلام يعيد list[dict] — يعمل دائماً بدون pandas."""
        with self._lock:
            if self._kind == "sqlite":
                assert self._sqlite is not None
                cur = self._sqlite.execute(sql, params or ())
                cols = [d[0] for d in cur.description] if cur.description else []
                return [dict(zip(cols, row)) for row in cur.fetchall()]
            if self._kind == "duckdb":
                cur = self._duck.execute(sql, params or None)  # type: ignore
                try:
                    cols = [d[0] for d in cur.description] if cur.description else []
                    return [dict(zip(cols, r)) for r in cur.fetchall()]
                except Exception:
                    return []
            from sqlalchemy import text as _text  # type: ignore
            with self._sa_engine.connect() as c:  # type: ignore
                r = c.execute(_text(sql), params or {})
                cols = list(r.keys())
                return [dict(zip(cols, row)) for row in r.fetchall()]

    # -- pandas helpers (اختيارية) --
    def query_df(self, sql: str, params=None):
        """يعيد pandas.DataFrame — يتطلب pandas."""
        try:
            import pandas as _pd  # type: ignore
        except ImportError as e:
            raise ImportError("query_df يحتاج pandas: pip install pandas") from e
        if self._kind == "sqlite":
            return _pd.read_sql_query(sql, self._sqlite, params=params)
        if self._kind == "duckdb":
            return self._duck.execute(sql).fetchdf()  # type: ignore
        return _pd.read_sql_query(sql, self._sa_engine, params=params)

    def save_df(self, df, table: str, if_exists: str = "replace"):
        """حفظ DataFrame في جدول — يتطلب pandas (+sqlalchemy لغير sqlite)."""
        try:
            import pandas as _pd  # noqa
        except ImportError as e:
            raise ImportError("save_df يحتاج pandas") from e
        with self._lock:
            if self._kind == "sqlite":
                df.to_sql(table, self._sqlite, if_exists=if_exists, index=False)
                return
            if self._kind == "duckdb":
                self._duck.register("_tmp_df", df)  # type: ignore
                if if_exists == "replace":
                    self._duck.execute(f"DROP TABLE IF EXISTS {table}")  # type: ignore
                self._duck.execute(f"CREATE TABLE {table} AS SELECT * FROM _tmp_df")  # type: ignore
                return
            df.to_sql(table, self._sa_engine, if_exists=if_exists, index=False)

    def load_csv(self, path: str, table: str | None = None, delimiter: str = ",") -> list[dict]:
        """تحميل CSV (stdlib) — اختيارياً ينشئ جدولاً ويعيده كـ list[dict]."""
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f, delimiter=delimiter))
        if table:
            if rows:
                cols = list(rows[0].keys())
                coldef = ", ".join(f'"{c}" TEXT' for c in cols)
                self.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({coldef})')
                self.execute(f'DELETE FROM "{table}"')
                ph = ", ".join(["?"] * len(cols))
                self.executemany(
                    f'INSERT INTO "{table}" VALUES ({ph})',
                    [[r.get(c) for c in cols] for r in rows],
                )
        return rows

    def tables(self) -> list[str]:
        if self._kind == "duckdb":
            rows = self.query("SHOW TABLES")
            return [list(r.values())[0] for r in rows]
        if self._kind == "sqlalchemy":
            from sqlalchemy import inspect as _insp  # type: ignore
            return _insp(self._sa_engine).get_table_names()  # type: ignore
        rows = self.query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        return [r["name"] for r in rows]

    def schema(self, table: str) -> list[dict]:
        if self._kind == "duckdb":
            return self.query(f"DESCRIBE {table}")
        if self._kind == "sqlalchemy":
            from sqlalchemy import inspect as _insp  # type: ignore
            return [{"name": c["name"], "type": str(c["type"])} for c in _insp(self._sa_engine).get_columns(table)]  # type: ignore
        return self.query(f'PRAGMA table_info("{table}")')

    # -- graph nodes --
    def read_node(self, sql: str, out_key: str = "rows", params=()):
        """مصنع عقدة رسم: تنفذ SQL وتضع النتيجة في state[out_key]."""
        def _node(state: dict) -> dict:
            return {out_key: self.query(sql, params)}
        _node.__name__ = f"db_read_{out_key}"
        return _node

    def write_node(self, table: str, in_key: str = "rows"):
        """مصنع عقدة كتابة: تأخذ list[dict] من state[in_key] وتدخلها في جدول."""
        def _node(state: dict) -> dict:
            rows = state.get(in_key, []) or []
            if not rows:
                return {"inserted": 0}
            cols = list(rows[0].keys())
            ph = ", ".join(["?"] * len(cols))
            colnames = ", ".join(f'"{c}"' for c in cols)
            self.execute(
                f'CREATE TABLE IF NOT EXISTS "{table}" ('
                + ", ".join(f'"{c}" TEXT' for c in cols) + ")"
            )
            n = self.executemany(
                f'INSERT INTO "{table}" ({colnames}) VALUES ({ph})',
                [[r.get(c) for c in cols] for r in rows],
            )
            return {"inserted": n}
        _node.__name__ = f"db_write_{table}"
        return _node

    def close(self):
        try:
            if self._sqlite is not None:
                self._sqlite.close()
        except Exception:
            pass
        try:
            if self._duck is not None:
                self._duck.close()
        except Exception:
            pass
        try:
            if self._sa_engine is not None:
                self._sa_engine.dispose()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


def sql_node(db: Database, sql: str, out_key: str = "rows", params=()):
    """اختصار: عقدة قراءة SQL جاهزة للرسم."""
    return db.read_node(sql, out_key=out_key, params=params)


def csv_loader_node(path: str, out_key: str = "rows", delimiter: str = ","):
    """عقدة تحميل CSV داخل الرسم (بدون قاعدة بيانات)."""
    def _node(state: dict) -> dict:
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f, delimiter=delimiter))
        return {out_key: rows}
    _node.__name__ = "csv_loader"
    return _node


class JsonFileSaver:
    """checkpointer بسيط قائم على ملف JSON — مفيد للنماذج الأولية.

    مثال:
        sv = JsonFileSaver("ckpts.json")
        app = g.compile(checkpointer=sv)
    """

    def __init__(self, path: str = "checkpoints.json"):
        self.path = path
        self._lock = threading.RLock()
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def _load(self) -> dict:
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save(self, data: dict):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, default=str)
        os.replace(tmp, self.path)

    def put(self, thread_id: str, state: dict, step: int):
        with self._lock:
            data = self._load()
            hist = data.setdefault(thread_id, [])
            try:
                snap = json.loads(json.dumps(state, default=str))
            except Exception:
                snap = {"_repr": repr(state)}
            hist.append({"step": step, "state": snap})
            self._save(data)

    def get(self, thread_id: str):
        with self._lock:
            hist = self._load().get(thread_id, [])
            return dict(hist[-1]["state"]) if hist else None

    def history(self, thread_id: str) -> list[dict]:
        with self._lock:
            return list(self._load().get(thread_id, []))
