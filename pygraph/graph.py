"""Graph — StateGraph / CompiledGraph (مطابق لـ langgraph.graph، أسرع تنفيذاً).

مصادر السرعة:
- بدون pydantic/نسخ عميق إلا عند checkpoint
- fast-path للسلاسل الخطية (بدون ThreadPool وبدون نسخ حالة للـ routing)
- fan-out متوازٍ عبر ThreadPool فقط عند التفرع
- reducers مستخرجة مرة واحدة + static edges محسوبة عند compile
"""
from __future__ import annotations

import asyncio
import copy
import inspect
from concurrent.futures import ThreadPoolExecutor
from typing import Any

try:
    from typing import Annotated, TypedDict, get_type_hints
except ImportError:  # py3.8
    from typing_extensions import Annotated, TypedDict, get_type_hints  # type: ignore

from .types import END, START, Command, GraphInterrupt, Interrupt, Send
from .types import _resume_ctx  # noqa: F401  (يستخدمه interrupt)

__all__ = ["StateGraph", "MessageGraph", "CompiledGraph", "Command", "Send",
           "START", "END", "add_messages", "MessagesState", "StateSnapshot"]


# ---------------------------------------------------------------- merging ---
def _msg_id(m):
    if isinstance(m, dict):
        return m.get("id")
    return getattr(m, "id", None)


def _is_remove(m) -> bool:
    if isinstance(m, dict):
        return m.get("type") == "remove"
    return type(m).__name__ == "RemoveMessage"


def add_messages(left, right):
    """مطابق لـ langgraph.graph.message.add_messages: دمج + upsert حسب id + حذف RemoveMessage."""
    if left is None:
        left = []
    if right is None:
        return list(left) if isinstance(left, list) else [left]
    l = list(left) if isinstance(left, list) else [left]
    r = list(right) if isinstance(right, list) else [right]
    if not r:
        return l
    # fast-path: لا ids ولا remove -> concat مباشر
    need_smart = False
    for m in r:
        if _is_remove(m) or _msg_id(m) is not None:
            need_smart = True
            break
    if not need_smart:
        return l + r
    # slow-path: upsert حسب id
    index = {}
    for i, m in enumerate(l):
        mid = _msg_id(m)
        if mid is not None:
            index[mid] = i
    out = list(l)
    for m in r:
        if _is_remove(m):
            mid = m.get("id") if isinstance(m, dict) else getattr(m, "id", None)
            if mid in index:
                out[index[mid]] = None  # type: ignore
            continue
        mid = _msg_id(m)
        if mid is not None and mid in index:
            out[index[mid]] = m
        else:
            if mid is not None:
                index[mid] = len(out)
            out.append(m)
    return [m for m in out if m is not None]


def _merge_state(state: dict, update: dict, reducers: dict | None = None) -> dict:
    reducers = reducers or {}
    for k, v in (update or {}).items():
        fn = reducers.get(k)
        if fn is not None:
            try:
                state[k] = fn(state.get(k), v)
                continue
            except Exception:
                pass
        state[k] = v
    return state


class MessagesState(TypedDict, total=False):
    messages: Annotated[list, add_messages]


class StateSnapshot:
    """لقطة حالة (مطابق لـ langgraph): values/next/tasks/interrupts + دعم dict القديم."""

    __slots__ = ("values", "next", "config", "tasks", "interrupts", "step")

    def __init__(self, values: dict, next: tuple = (), config: dict | None = None,
                 tasks=(), interrupts=(), step: int = 0):
        self.values = values
        self.next = next
        self.config = config or {}
        self.tasks = tasks
        self.interrupts = interrupts
        self.step = step

    def __getitem__(self, k):
        if k in ("values", "next", "config", "tasks", "interrupts", "step"):
            return getattr(self, k)
        return self.values[k]

    def __contains__(self, k):
        return k in self.values or k in ("values", "next", "config", "tasks", "interrupts", "step")

    def get(self, k, default=None):
        if k in ("values", "next", "config", "tasks", "interrupts", "step"):
            return getattr(self, k)
        return self.values.get(k, default)

    def keys(self):
        return self.values.keys()

    def __iter__(self):
        return iter(self.values)

    def __repr__(self):  # pragma: no cover
        return f"StateSnapshot(values={self.values!r}, next={self.next!r})"


# ---------------------------------------------------------------- builder ---
class StateGraph:
    """مطابق لـ langgraph.graph.StateGraph."""

    def __init__(self, schema: dict | type | None = None):
        self.schema = schema or dict
        self.nodes: dict[str, object] = {}
        self.edges: dict[str, list[str]] = {}
        self.conditional: dict[str, tuple] = {}
        self.entry: str | None = None
        self.finish: set[str] = set()
        self.reducers: dict[str, object] = {}
        try:
            hints = get_type_hints(schema, include_extras=True) if isinstance(schema, type) else {}
            for k, h in hints.items():
                meta = getattr(h, "__metadata__", ())
                for m in meta:
                    if callable(m):
                        self.reducers[k] = m
                        break
        except Exception:
            pass
        # MessagesState الافتراضية: messages -> add_messages
        try:
            if isinstance(schema, type) and issubclass(schema, dict) and schema.__name__ == "MessagesState":
                self.reducers.setdefault("messages", add_messages)
        except Exception:
            pass
        if schema is MessagesState:
            self.reducers.setdefault("messages", add_messages)

    # -- بناء --
    def add_node(self, name: str, fn, **kw) -> "StateGraph":
        self.nodes[name] = fn
        self.edges.setdefault(name, [])
        return self

    def add_edge(self, a: str, b: str) -> "StateGraph":
        if a == START:
            self.entry = b
            return self
        self.edges.setdefault(a, []).append(b)
        return self

    def add_conditional_edges(self, source: str, path, mapping: dict | list | None = None) -> "StateGraph":
        self.conditional[source] = (path, mapping)
        return self

    def set_entry_point(self, name: str) -> "StateGraph":
        self.entry = name
        return self

    def set_finish_point(self, name: str) -> "StateGraph":
        self.finish.add(name)
        return self

    def add_sequence(self, seq: list) -> "StateGraph":
        prev = None
        for item in seq:
            name = item if isinstance(item, str) else getattr(item, "__name__", str(item))
            if not isinstance(item, str):
                self.add_node(name, item)
            if prev is None:
                if self.entry is None:
                    self.entry = name
            else:
                self.add_edge(prev, name)
            prev = name
        return self

    def compile(self, checkpointer=None, interrupt_before: list[str] | None = None,
                interrupt_after: list[str] | None = None, **kw) -> "CompiledGraph":
        return CompiledGraph(self, checkpointer, interrupt_before or [], interrupt_after or [])


class MessageGraph(StateGraph):
    def __init__(self):
        super().__init__(dict)
        self.reducers["messages"] = add_messages


# --------------------------------------------------------------- compiled ---
def _run_coro_sync(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures as _cf
            with _cf.ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


class CompiledGraph:
    __slots__ = ("builder", "checkpointer", "interrupt_before", "interrupt_after",
                 "_static_next", "_reducers", "_pending", "_is_async_node")

    def __init__(self, builder: StateGraph, checkpointer=None, interrupt_before=None, interrupt_after=None):
        self.builder = builder
        self.checkpointer = checkpointer
        self.interrupt_before = interrupt_before or []
        self.interrupt_after = interrupt_after or []
        # static edges (بدون conditional) محسوبة مرة واحدة
        static: dict[str, list] = {}
        for n, dests in builder.edges.items():
            static[n] = [d for d in dests if d != END]
        self._static_next = static
        self._reducers = dict(builder.reducers)
        self._pending: dict[str, dict] = {}
        self._is_async_node = {n: inspect.iscoroutinefunction(f) if callable(f) and not isinstance(f, CompiledGraph) else False
                               for n, f in builder.nodes.items()}

    # -- helpers --
    def _thread_id(self, config: dict | None) -> str:
        return ((config or {}).get("configurable") or {}).get("thread_id", "default")

    def _initial(self, inputs, tid: str) -> tuple[dict, object]:
        """ترجع (state, resume_value)."""
        resume = None
        if isinstance(inputs, Command):
            resume = inputs.resume
            saved = self.checkpointer.get(tid) if self.checkpointer is not None else None
            pend = self._pending.get(tid)
            if pend is not None:
                state = dict(pend["state"])
            elif saved:
                state = dict(saved)
            else:
                state = {}
            if inputs.update:
                _merge_state(state, inputs.update, self._reducers)
            if inputs.goto:
                pend_goto = inputs.goto if isinstance(inputs.goto, list) else [inputs.goto]
                self._pending[tid] = {"state": state, "next": pend_goto,
                                      "interrupts": (pend or {}).get("interrupts", ()),
                                      "step": (pend or {}).get("step", 0)}
            return state, resume
        if inputs is None:
            saved = self.checkpointer.get(tid) if self.checkpointer is not None else None
            pend = self._pending.get(tid)
            if pend is not None:
                return dict(pend["state"]), None
            return dict(saved) if saved else {}, None
        if isinstance(inputs, dict):
            state = dict(inputs)
        elif isinstance(inputs, list):
            state = {"messages": list(inputs)}
        else:
            state = {"input": inputs}
        if self.checkpointer is not None:
            saved = self.checkpointer.get(tid)
            if saved and not self._pending.get(tid):
                # دمج: الجديد يغلب، و messages تُمدد عبر reducer
                merged = dict(saved)
                for k, v in state.items():
                    if k == "messages" and isinstance(v, list) and isinstance(saved.get(k), list):
                        merged[k] = add_messages(saved[k], v)
                    else:
                        merged[k] = v
                return merged, None
        # بداية جديدة: امسح أي pending قديم
        self._pending.pop(tid, None)
        return state, None

    def _call_sync(self, name: str, state: dict):
        from .graph import CompiledGraph as _CG  # local ref
        fn = self.builder.nodes[name]
        if isinstance(fn, _CG):
            sub = fn.invoke(state)
            return sub if isinstance(sub, dict) else {"output": sub}
        if self._is_async_node.get(name):
            return _run_coro_sync(fn(state))
        res = fn(state)
        if inspect.isawaitable(res):
            return _run_coro_sync(res)
        if hasattr(res, "invoke") and not isinstance(res, (dict, list, Command, Send)):
            try:
                return res.invoke(state)
            except Exception:
                return res
        return res

    async def _call_async(self, name: str, state: dict):
        from .graph import CompiledGraph as _CG
        fn = self.builder.nodes[name]
        if isinstance(fn, _CG):
            try:
                return await fn.ainvoke(state)
            except AttributeError:
                return fn.invoke(state)
        res = fn(state)
        if inspect.isawaitable(res):
            res = await res
        if hasattr(res, "ainvoke") and not isinstance(res, (dict, list, Command, Send)):
            try:
                return await res.ainvoke(state)
            except Exception:
                return res
        if hasattr(res, "invoke") and not isinstance(res, (dict, list, Command, Send)):
            try:
                return res.invoke(state)
            except Exception:
                return res
        return res

    def _route(self, name: str, state: dict, update: dict) -> list:
        b = self.builder
        nxt: list = []
        if name in b.conditional:
            path, mapping = b.conditional[name]
            # state للـ routing = state + update (بدون نسخ إلا هنا)
            merged = dict(state)
            if update:
                _merge_state(merged, update, self._reducers)
            route = path(merged) if callable(path) else path
            if isinstance(route, Command):
                gotos = [route.goto] if isinstance(route.goto, str) else list(route.goto or [])
                if route.update:
                    _merge_state(merged, route.update, self._reducers)
                    _merge_state(state, route.update, self._reducers)
                    update.update(route.update)
                nxt.extend(gotos)
            elif isinstance(route, Send):
                return [route]
            elif isinstance(route, list) and route and isinstance(route[0], Send):
                return route
            else:
                if mapping is None:
                    dests = [route] if isinstance(route, str) else list(route or [])
                elif isinstance(mapping, dict):
                    r = mapping.get(route, END)
                    dests = [r] if isinstance(r, str) else list(r or [])
                else:
                    dests = [route] if isinstance(route, str) else list(route or [])
                # Send داخل mapping
                flat: list = []
                for d in dests:
                    if isinstance(d, Send):
                        return dests  # type: ignore
                    flat.append(d)
                nxt.extend(flat)
        nxt.extend(self._static_next.get(name, ()))
        # dedup مع الحفاظ على الترتيب + إسقاط END
        seen: set = set()
        out: list = []
        for n in nxt:
            if isinstance(n, Send):
                out.append(n)
                continue
            if n == END or n in seen:
                continue
            seen.add(n)
            out.append(n)
        return out

    def _dispatch_single(self, node: str, state: dict) -> tuple[dict, list]:
        if node in self.interrupt_before:
            return {}, []
        res = self._call_sync(node, state)
        if isinstance(res, Command):
            update = res.update or {}
            goto = res.goto
            if goto is None:
                gotos = self._route(node, state, update)
            elif isinstance(goto, str):
                gotos = [goto]
            elif isinstance(goto, Send):
                gotos = [goto]
            else:
                gotos = list(goto)
            return update, gotos
        if isinstance(res, Send):
            return {}, [res]
        if isinstance(res, list) and res and isinstance(res[0], Send):
            return {}, res
        if isinstance(res, dict):
            return res, self._route(node, state, res)
        if res is None:
            return {}, self._route(node, state, {})
        if isinstance(res, (str, tuple)):
            return {"output": res}, self._route(node, state, {})
        return {}, self._route(node, state, {})

    async def _dispatch_single_async(self, node: str, state: dict) -> tuple[dict, list]:
        if node in self.interrupt_before:
            return {}, []
        res = await self._call_async(node, state)
        if isinstance(res, Command):
            update = res.update or {}
            goto = res.goto
            if goto is None:
                gotos = self._route(node, state, update)
            elif isinstance(goto, str):
                gotos = [goto]
            elif isinstance(goto, Send):
                gotos = [goto]
            else:
                gotos = list(goto)
            return update, gotos
        if isinstance(res, Send):
            return {}, [res]
        if isinstance(res, list) and res and isinstance(res[0], Send):
            return {}, res
        if isinstance(res, dict):
            return res, self._route(node, state, res)
        if res is None:
            return {}, self._route(node, state, {})
        return {}, self._route(node, state, {})

    # -- core loops --
    def _core_loop(self, inputs, config, recursion_limit, stream_mode, collect: bool):
        from .types import _resume_ctx as _rctx
        b = self.builder
        tid = self._thread_id(config)
        state, resume = self._initial(inputs, tid)
        if b.entry is None:
            raise ValueError("No entry point — استخدم add_edge(START, 'node') أو set_entry_point")
        pend = self._pending.pop(tid, None)
        queue: list = list(pend["next"]) if pend and isinstance(inputs, Command) and pend.get("next") else [b.entry]
        if pend and isinstance(inputs, Command) and not pend.get("next"):
            queue = pend.get("next") or []
        step = pend.get("step", 0) if pend and isinstance(inputs, Command) else 0
        collected = [] if collect else None
        interrupts: list = []
        token = None
        if resume is not None:
            token = _rctx.set([resume] if not isinstance(resume, list) else resume)

        def emit(node, update):
            if stream_mode == "values":
                return dict(state)
            if stream_mode == "messages":
                msgs = (update or {}).get("messages")
                if msgs is not None:
                    return {"messages": msgs if isinstance(msgs, list) else [msgs]}
                return None
            if stream_mode == "debug":
                return {"step": step, "node": node, "state": dict(state)}
            return {node: update}

        try:
            while queue and step < recursion_limit:
                step += 1
                # interrupt قبل التنفيذ؟
                if any(q in self.interrupt_before for q in queue):
                    self._pending[tid] = {"state": dict(state), "next": list(queue),
                                          "interrupts": interrupts, "step": step}
                    break
                try:
                    if len(queue) > 1:
                        with ThreadPoolExecutor(max_workers=min(32, len(queue))) as ex:
                            results = list(ex.map(lambda n: (n, self._dispatch_single(n, state)), queue))
                    else:
                        results = [(queue[0], self._dispatch_single(queue[0], state))]
                except GraphInterrupt as gi:
                    interrupts.extend(gi.interrupts)
                    self._pending[tid] = {"state": dict(state), "next": list(queue),
                                          "interrupts": interrupts, "step": step}
                    if self.checkpointer is not None:
                        self.checkpointer.put(tid, state, step)
                    if collect:
                        collected.append({"__interrupt__": [i.value for i in gi.interrupts]})
                    break
                queue = []
                for node, (update, gotos) in results:
                    if update:
                        _merge_state(state, update, self._reducers)
                    if self.checkpointer is not None:
                        self.checkpointer.put(tid, state, step)
                    if collect and stream_mode != "messages":
                        collected.append(emit(node, update))
                    elif collect and stream_mode == "messages":
                        ch = emit(node, update)
                        if ch is not None:
                            collected.append(ch)
                    if node in self.interrupt_after:
                        queue = []
                        self._pending[tid] = {"state": dict(state), "next": [], "interrupts": interrupts, "step": step}
                        break
                    sends: list = []
                    for g in gotos:
                        if isinstance(g, Send):
                            sends.append(g)
                        elif g != END:
                            queue.append(g)
                    if sends:
                        def _exec_send(g):
                            s2 = dict(state)
                            if isinstance(g.arg, dict):
                                _merge_state(s2, g.arg, self._reducers)
                                s2["__send__"] = g.arg
                            else:
                                s2["__send__"] = g.arg
                            return self._dispatch_single(g.node, s2)
                        try:
                            if len(sends) > 1:
                                with ThreadPoolExecutor(max_workers=min(32, len(sends))) as _ex:
                                    send_results = list(_ex.map(_exec_send, sends))
                            else:
                                send_results = [_exec_send(sends[0])]
                        except GraphInterrupt as gi:
                            interrupts.extend(gi.interrupts)
                            self._pending[tid] = {"state": dict(state), "next": list(queue),
                                                  "interrupts": interrupts, "step": step}
                            if collect:
                                collected.append({"__interrupt__": [i.value for i in gi.interrupts]})
                            queue = []
                            break
                        for upd, gg in send_results:
                            if upd:
                                _merge_state(state, upd, self._reducers)
                            queue.extend([x for x in gg if x != END])
        finally:
            if token is not None:
                try:
                    _rctx.reset(token)
                except Exception:
                    pass
        return state, collected, interrupts

    async def _core_loop_async(self, inputs, config, recursion_limit, stream_mode, collect: bool):
        from .types import _resume_ctx as _rctx
        b = self.builder
        tid = self._thread_id(config)
        state, resume = self._initial(inputs, tid)
        if b.entry is None:
            raise ValueError("No entry point")
        pend = self._pending.pop(tid, None)
        queue: list = list(pend["next"]) if pend and isinstance(inputs, Command) and pend.get("next") else [b.entry]
        step = pend.get("step", 0) if pend and isinstance(inputs, Command) else 0
        collected = [] if collect else None
        interrupts: list = []
        token = None
        if resume is not None:
            token = _rctx.set([resume] if not isinstance(resume, list) else resume)
        try:
            while queue and step < recursion_limit:
                step += 1
                if any(q in self.interrupt_before for q in queue):
                    self._pending[tid] = {"state": dict(state), "next": list(queue),
                                          "interrupts": interrupts, "step": step}
                    break
                try:
                    if len(queue) > 1:
                        results = list(await asyncio.gather(
                            *[self._dispatch_single_async(n, state) for n in queue]))
                        results = [(n, r) for n, r in zip(queue, results)]
                    else:
                        results = [(queue[0], await self._dispatch_single_async(queue[0], state))]
                except GraphInterrupt as gi:
                    interrupts.extend(gi.interrupts)
                    self._pending[tid] = {"state": dict(state), "next": list(queue),
                                          "interrupts": interrupts, "step": step}
                    if self.checkpointer is not None:
                        self.checkpointer.put(tid, state, step)
                    if collect:
                        collected.append({"__interrupt__": [i.value for i in gi.interrupts]})
                    break
                queue = []
                for node, (update, gotos) in results:
                    if update:
                        _merge_state(state, update, self._reducers)
                    if self.checkpointer is not None:
                        self.checkpointer.put(tid, state, step)
                    if collect:
                        if stream_mode == "values":
                            collected.append(dict(state))
                        elif stream_mode == "debug":
                            collected.append({"step": step, "node": node, "state": dict(state)})
                        else:
                            collected.append({node: update})
                    if node in self.interrupt_after:
                        queue = []
                        break
                    sends_a: list = []
                    for g in gotos:
                        if isinstance(g, Send):
                            sends_a.append(g)
                        elif g != END:
                            queue.append(g)
                    if sends_a:
                        async def _one(g):
                            s2 = dict(state)
                            if isinstance(g.arg, dict):
                                _merge_state(s2, g.arg, self._reducers)
                                s2["__send__"] = g.arg
                            else:
                                s2["__send__"] = g.arg
                            return await self._dispatch_single_async(g.node, s2)
                        send_results = list(await asyncio.gather(*[_one(g) for g in sends_a]))
                        for upd, gg in send_results:
                            if upd:
                                _merge_state(state, upd, self._reducers)
                            queue.extend([x for x in gg if x != END])
        finally:
            if token is not None:
                try:
                    _rctx.reset(token)
                except Exception:
                    pass
        return state, collected, interrupts

    # -- public API (مطابق لـ langgraph) --
    def invoke(self, inputs, config: dict | None = None, recursion_limit: int = 25, **kw) -> dict:
        state, _, interrupts = self._core_loop(inputs, config, recursion_limit, "updates", collect=False)
        if interrupts:
            state = dict(state)
            state["__interrupt__"] = [i.value for i in interrupts]
        return state

    async def ainvoke(self, inputs, config: dict | None = None, recursion_limit: int = 25, **kw) -> dict:
        state, _, interrupts = await self._core_loop_async(inputs, config, recursion_limit, "updates", collect=False)
        if interrupts:
            state = dict(state)
            state["__interrupt__"] = [i.value for i in interrupts]
        return state

    def stream(self, inputs, config: dict | None = None, recursion_limit: int = 25,
               stream_mode: str | list = "updates", **kw):
        modes = [stream_mode] if isinstance(stream_mode, str) else list(stream_mode)
        if len(modes) == 1:
            _, collected, _ = self._core_loop(inputs, config, recursion_limit, modes[0], collect=True)
            yield from collected
        else:
            for m in modes:
                _, collected, _ = self._core_loop(inputs, config, recursion_limit, m, collect=True)
                for c in collected:
                    yield (m, c)

    async def astream(self, inputs, config: dict | None = None, recursion_limit: int = 25,
                      stream_mode: str | list = "updates", **kw):
        modes = [stream_mode] if isinstance(stream_mode, str) else list(stream_mode)
        if len(modes) == 1:
            _, collected, _ = await self._core_loop_async(inputs, config, recursion_limit, modes[0], collect=True)
            for c in collected:
                yield c
        else:
            for m in modes:
                _, collected, _ = await self._core_loop_async(inputs, config, recursion_limit, m, collect=True)
                for c in collected:
                    yield (m, c)

    def batch(self, inputs_list: list, config: dict | None = None, max_workers: int = 8, **kw) -> list[dict]:
        with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(inputs_list)))) as ex:
            return list(ex.map(lambda x: self.invoke(x, config), inputs_list))

    async def abatch(self, inputs_list: list, config: dict | None = None, **kw) -> list[dict]:
        return list(await asyncio.gather(*[self.ainvoke(x, config) for x in inputs_list]))

    # -- إدارة الحالة --
    def get_state(self, config: dict) -> StateSnapshot:
        tid = self._thread_id(config)
        pend = self._pending.get(tid)
        if pend is not None:
            return StateSnapshot(dict(pend["state"]), next=tuple(pend.get("next", ())),
                                 config=config, interrupts=tuple(pend.get("interrupts", ())),
                                 step=pend.get("step", 0))
        if self.checkpointer is not None:
            v = self.checkpointer.get(tid)
            if v is not None:
                return StateSnapshot(dict(v), config=config)
            return StateSnapshot({}, config=config)
        raise ValueError("No checkpointer attached")

    def get_state_history(self, config: dict) -> list[StateSnapshot]:
        tid = self._thread_id(config)
        if self.checkpointer is None:
            raise ValueError("No checkpointer attached")
        hist = self.checkpointer.history(tid) if hasattr(self.checkpointer, "history") else []
        return [StateSnapshot(dict(h.get("state", {})), config=config, step=h.get("step", 0)) for h in hist]

    def update_state(self, config: dict, values: dict, as_node: str | None = None):
        tid = self._thread_id(config)
        cur = None
        if tid in self._pending:
            cur = dict(self._pending[tid]["state"])
        elif self.checkpointer is not None:
            cur = dict(self.checkpointer.get(tid) or {})
        else:
            raise ValueError("No checkpointer attached")
        _merge_state(cur, values, self._reducers)
        if tid in self._pending:
            self._pending[tid]["state"] = cur
        if self.checkpointer is not None:
            self.checkpointer.put(tid, cur, 0)
