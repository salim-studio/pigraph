"""pygraph — بديل langgraph الأسرع (نفس الواجهة).

مثال:
    from pygraph import StateGraph, START, END, MemorySaver
    g = StateGraph(dict)
    g.add_node("a", lambda s: {"x": s.get("x", 0) + 1})
    g.add_node("b", lambda s: {"y": s["x"] * 2})
    g.add_edge(START, "a"); g.add_edge("a", "b"); g.add_edge("b", END)
    app = g.compile(checkpointer=MemorySaver())
    print(app.invoke({"x": 0}))

مصادر السرعة:
- حالة dict مباشرة بدون نسخ عميق إلا عند checkpoint
- fan-out متوازٍ عبر ThreadPool و fast-path للسلاسل الخطية
- reducers اختيارية بدون pydantic
"""
from __future__ import annotations

__version__ = "0.2.0"

from .types import END, START, Command, GraphInterrupt, Interrupt, Send, interrupt
from .graph import (CompiledGraph, MessageGraph, MessagesState, StateGraph,
                    StateSnapshot, add_messages)
from .checkpoint import (BaseCheckpointer, Checkpoint, InMemorySaver,
                         MemorySaver, MemorySaverCheckpoint, SqliteSaver)
from .prebuilt import ToolExecutor, ToolNode, create_react_agent, tools_condition
from .functional import entrypoint, task

__all__ = ["StateGraph", "MessageGraph", "CompiledGraph", "StateSnapshot", "MessagesState",
           "Command", "Send", "Interrupt", "GraphInterrupt", "interrupt",
           "START", "END", "add_messages",
           "MemorySaver", "InMemorySaver", "MemorySaverCheckpoint", "SqliteSaver",
           "BaseCheckpointer", "Checkpoint",
           "ToolNode", "ToolExecutor", "tools_condition", "create_react_agent",
           "entrypoint", "task"]
