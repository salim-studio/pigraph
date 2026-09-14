"""Prebuilt — ToolNode / tools_condition / create_react_agent (مطابق لـ langgraph.prebuilt)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .graph import END, START, StateGraph
from .types import Command

__all__ = ["ToolNode", "tools_condition", "create_react_agent", "ToolExecutor"]


def _tool_calls_from_message(msg):
    if isinstance(msg, dict):
        tc = msg.get("tool_calls") or []
        if tc:
            return tc
        # openai-style: {"function_call": ...}؟ نتجاهل
        return []
    tc = getattr(msg, "tool_calls", None) or getattr(msg, "tool_call", None)
    if tc is None:
        return []
    return tc if isinstance(tc, list) else [tc]


def _tool_call_parts(tc):
    if isinstance(tc, dict):
        return (tc.get("id") or tc.get("tool_call_id") or "call_0",
                tc.get("name") or tc.get("function", {}).get("name") if isinstance(tc.get("function"), dict) else tc.get("name"),
                tc.get("args") or tc.get("arguments") or {})
    name = getattr(tc, "name", None) or getattr(getattr(tc, "function", None), "name", None)
    cid = getattr(tc, "id", None) or getattr(tc, "tool_call_id", None) or "call_0"
    args = getattr(tc, "args", None) or getattr(tc, "arguments", None) or {}
    if isinstance(args, str):
        import json as _j
        try:
            args = _j.loads(args)
        except Exception:
            args = {"input": args}
    return cid, name, args


def _make_tool_message(content, tool_call_id, name=None):
    try:
        from pychain import ToolMessage as _TM
        return _TM(content=str(content), tool_call_id=tool_call_id, name=name or "")
    except Exception:
        return {"role": "tool", "content": str(content), "tool_call_id": tool_call_id, "name": name or ""}


class ToolNode:
    """عقدة تنفيذ أدوات متوازية (مطابق لـ langgraph.prebuilt.ToolNode).

    مثال:
        node = ToolNode([my_tool1, my_tool2])
        g.add_node("tools", node)
    """

    def __init__(self, tools, max_workers: int = 8):
        if isinstance(tools, dict):
            self._tools = dict(tools)
        else:
            self._tools = {}
            for t in tools:
                name = getattr(t, "name", None) or getattr(t, "__name__", None) or str(t)
                self._tools[name] = t
        self.max_workers = max_workers

    def _run_one(self, tool, args):
        try:
            if hasattr(tool, "invoke"):
                return tool.invoke(args if isinstance(args, dict) else {"input": args})
            if callable(tool):
                if isinstance(args, dict):
                    try:
                        return tool(**args)
                    except TypeError:
                        return tool(args)
                return tool(args)
            return f"unknown tool: {tool}"
        except Exception as e:
            return f"Error: {e}"

    def __call__(self, state: dict) -> dict:
        msgs = state.get("messages", [])
        if not msgs:
            return {"messages": []}
        last = msgs[-1]
        calls = _tool_calls_from_message(last)
        if not calls:
            return {"messages": []}
        jobs = []
        for tc in calls:
            cid, name, args = _tool_call_parts(tc)
            tool = self._tools.get(name)
            if tool is None:
                jobs.append((cid, name, None))
            else:
                jobs.append((cid, name, (tool, args)))

        def _exec(job):
            cid, name, payload = job
            if payload is None:
                return _make_tool_message(f"Tool '{name}' not found", cid, name)
            tool, args = payload
            return _make_tool_message(self._run_one(tool, args), cid, name)

        if len(jobs) > 1:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(jobs))) as ex:
                out = list(ex.map(_exec, jobs))
        else:
            out = [_exec(jobs[0])]
        return {"messages": out}


ToolExecutor = ToolNode


def tools_condition(state: dict, messages_key: str = "messages") -> str:
    """مطابق لـ langgraph.prebuilt.tools_condition: 'tools' أو END."""
    msgs = state.get(messages_key, [])
    if not msgs:
        return END
    last = msgs[-1]
    if _tool_calls_from_message(last):
        return "tools"
    return END


def create_react_agent(model, tools, prompt: str | None = None, checkpointer=None,
                       messages_key: str = "messages", **kw):
    """وكيل ReAct جاهز (مطابق لـ langgraph.prebuilt.create_react_agent).

    model: كائن chat (له invoke) أو دالة (state)->dict.
    tools: قائمة أدوات.
    """
    from .graph import add_messages
    try:
        from typing import Annotated as _A
        from typing import TypedDict as _TD
        class _State(_TD, total=False):
            messages: _A[list, add_messages]  # type: ignore
        schema = _State
    except Exception:
        schema = dict

    g = StateGraph(schema)

    def _agent(state: dict) -> dict:
        msgs = state.get(messages_key, [])
        if callable(model) and not hasattr(model, "invoke"):
            r = model(state)
            if isinstance(r, dict):
                return r
            return {messages_key: [r]}
        # chat model
        try:
            if prompt and hasattr(model, "invoke"):
                res = model.invoke(msgs)
            else:
                res = model.invoke(msgs)
        except Exception:
            res = model(msgs) if callable(model) else {"content": ""}
        if isinstance(res, dict) and messages_key in res:
            return res
        return {messages_key: [res]}

    _agent.__name__ = "agent"
    g.add_node("agent", _agent)
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=checkpointer)
