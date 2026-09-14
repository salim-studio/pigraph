"""Utils — retry / cache / timing decorators for nodes (stdlib only).

أمثلة:
    from pigraph.utils import retry, cached, timed, parallel_map, batch_apply

    @retry(tries=3, delay=0.1)
    def flaky_node(state): ...

    @cached(ttl=60)
    def expensive_node(state): ...
"""
from __future__ import annotations

import functools
import time

__all__ = ["retry", "cached", "timed", "parallel_map", "batch_apply"]


def retry(tries: int = 3, delay: float = 0.0, exceptions=(Exception,)):
    """إعادة المحاولة عند الفشل — للعقد غير المستقرة (API calls)."""
    def _dec(fn):
        @functools.wraps(fn)
        def _w(*a, **k):
            err = None
            for i in range(max(1, tries)):
                try:
                    return fn(*a, **k)
                except exceptions as e:
                    err = e
                    if delay and i < tries - 1:
                        time.sleep(delay)
            raise err  # type: ignore
        return _w
    return _dec


def cached(ttl: float | None = None, maxsize: int = 1024):
    """تخزين مؤقت لنتائج العقد حسب state (JSON-serializable)."""
    def _dec(fn):
        store: dict = {}
        order: list = []

        @functools.wraps(fn)
        def _w(state: dict):
            import json as _j
            try:
                key = _j.dumps(state, sort_keys=True, default=str)
            except Exception:
                return fn(state)
            now = time.time()
            if key in store:
                val, ts = store[key]
                if ttl is None or now - ts < ttl:
                    return val
            out = fn(state)
            store[key] = (out, now)
            order.append(key)
            while len(order) > maxsize:
                store.pop(order.pop(0), None)
            return out

        _w.cache_clear = store.clear  # type: ignore
        return _w
    return _dec


def timed(fn):
    """يطبع زمن تنفيذ العقدة (للتشخيص السريع)."""
    @functools.wraps(fn)
    def _w(state: dict):
        t0 = time.perf_counter()
        try:
            return fn(state)
        finally:
            dt = (time.perf_counter() - t0) * 1000
            print(f"[pigraph] {getattr(fn, '__name__', 'node')}: {dt:.2f}ms")
    return _w


def parallel_map(fn, items: list, max_workers: int = 8) -> list:
    """تطبيق متوازٍ لدالة على قائمة (ThreadPool)."""
    from concurrent.futures import ThreadPoolExecutor
    items = list(items or [])
    if not items:
        return []
    if len(items) == 1:
        return [fn(items[0])]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as ex:
        return list(ex.map(fn, items))


def batch_apply(fn, states: list[dict], max_workers: int = 8) -> list[dict]:
    """تطبيق عقدة واحدة على عدة states متوازياً."""
    return parallel_map(fn, states, max_workers)
