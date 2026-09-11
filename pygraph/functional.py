"""Functional API — @entrypoint / @task (مطابق لـ langgraph.func)."""
from __future__ import annotations

import asyncio
import functools
import inspect
from concurrent.futures import ThreadPoolExecutor

__all__ = ["entrypoint", "task"]


class _Task:
    def __init__(self, fn, max_workers: int = 8):
        functools.update_wrapper(self, fn)
        self.fn = fn
        self.is_async = inspect.iscoroutinefunction(fn)

    def __call__(self, *a, **k):
        return self.fn(*a, **k)

    def result(self, *a, **k):
        return self.fn(*a, **k)


def task(fn=None, **kw):
    """مطابق لـ langgraph.func.task: يغلف دالة لتعمل داخل entrypoint (متوازية عند الإمكان)."""
    def _wrap(f):
        return _Task(f, **kw)
    if fn is None:
        return _wrap
    if callable(fn):
        return _Task(fn)
    return _wrap


class EntrypointRunner:
    """كائن يشبه CompiledGraph يعيد @entrypoint."""

    def __init__(self, fn, checkpointer=None):
        self.fn = fn
        self.checkpointer = checkpointer
        self.is_async = inspect.iscoroutinefunction(fn)

    def _tid(self, config):
        return ((config or {}).get("configurable") or {}).get("thread_id", "default")

    def invoke(self, inputs, config=None, **kw):
        tid = self._tid(config)
        if self.checkpointer is not None and inputs is None:
            saved = self.checkpointer.get(tid)
            if saved and isinstance(saved, dict) and "return" in saved:
                return saved["return"]
        if self.is_async:
            import asyncio as _a
            try:
                loop = _a.get_event_loop()
                if not loop.is_running():
                    out = loop.run_until_complete(self.fn(inputs))
                else:
                    out = _a.run(self.fn(inputs))
            except RuntimeError:
                out = _a.run(self.fn(inputs))
        else:
            out = self.fn(inputs)
        if self.checkpointer is not None:
            try:
                self.checkpointer.put(tid, {"return": out, "input": inputs}, 1)
            except Exception:
                pass
        return out

    async def ainvoke(self, inputs, config=None, **kw):
        if self.is_async:
            out = await self.fn(inputs)
        else:
            out = await asyncio.to_thread(self.fn, inputs)
        return out

    def stream(self, inputs, config=None, **kw):
        yield self.invoke(inputs, config)

    async def astream(self, inputs, config=None, **kw):
        yield await self.ainvoke(inputs, config)

    def batch(self, inputs_list, config=None, max_workers: int = 8, **kw):
        with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(inputs_list)))) as ex:
            return list(ex.map(lambda x: self.invoke(x, config), inputs_list))


def entrypoint(fn=None, checkpointer=None, **kw):
    """مطابق لـ langgraph.func.entrypoint.

    مثال:
        @entrypoint(checkpointer=MemorySaver())
        def flow(inputs):
            a = my_task(inputs["x"])
            return {"y": a * 2}
        flow.invoke({"x": 3})
    """
    def _wrap(f):
        return EntrypointRunner(f, checkpointer=checkpointer)
    if fn is None:
        return _wrap
    if callable(fn):
        return EntrypointRunner(fn, checkpointer=checkpointer)
    return _wrap
