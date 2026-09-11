"""Types — Command / Send / interrupt (مطابق لـ langgraph.types)."""
from __future__ import annotations

import uuid
from contextvars import ContextVar

__all__ = ["Command", "Send", "Interrupt", "interrupt", "GraphInterrupt", "START", "END"]

START = "__start__"
END = "__end__"


class Command:
    """أمر تحكم: تحديث الحالة + قفزة + استئناف بعد interrupt.

    أمثلة:
        Command(update={"x": 1}, goto="other")
        Command(goto="a")
        Command(resume="user input")  # لاستئناف بعد interrupt()
    """
    __slots__ = ("update", "goto", "resume", "graph")

    def __init__(self, update: dict | None = None, goto=None, resume=None, graph: str | None = None):
        self.update = update or {}
        self.goto = goto
        self.resume = resume
        self.graph = graph

    def __repr__(self):  # pragma: no cover
        return f"Command(update={self.update!r}, goto={self.goto!r}, resume={self.resume!r})"


class Send:
    """استدعاء متوازٍ لعقدة: Send("worker", {"i": 1})."""
    __slots__ = ("node", "arg")

    def __init__(self, node: str, arg):
        self.node = node
        self.arg = arg

    def __repr__(self):  # pragma: no cover
        return f"Send({self.node!r}, {self.arg!r})"


class Interrupt:
    """قيمة مقاطعة واحدة (مطابق لـ langgraph.types.Interrupt)."""
    __slots__ = ("value", "id", "resumable", "ns")

    def __init__(self, value=None, id: str | None = None, resumable: bool = True, ns=None):
        self.value = value
        self.id = id or f"interrupt-{uuid.uuid4().hex[:8]}"
        self.resumable = resumable
        self.ns = ns

    def __repr__(self):  # pragma: no cover
        return f"Interrupt({self.value!r}, id={self.id!r})"


class GraphInterrupt(Exception):
    """إشارة داخلية: تُرفع من داخل عقدة عند استدعاء interrupt()."""
    def __init__(self, interrupts: list[Interrupt]):
        super().__init__("Graph interrupted")
        self.interrupts = interrupts


# سياق الاستئناف: القيمة التي أعادها المستخدم عبر Command(resume=...)
_resume_ctx: ContextVar = ContextVar("pygraph_resume", default=None)


def interrupt(value=None) -> object:
    """أوقف التنفيذ واطلب إدخال بشري (مطابق لـ langgraph.types.interrupt).

    داخل عقدة:
        answer = interrupt({"question": "هل أوافق؟"})
    ثم يستأنف المستخدم عبر:
        app.invoke(Command(resume="نعم"))
    """
    resume = _resume_ctx.get()
    if resume is not None:
        # وضع الاستئناف: أعد القيمة مباشرة بدون توقف ثانٍ
        _resume_ctx.set(None)
        return resume[0] if isinstance(resume, list) and len(resume) == 1 else resume
    raise GraphInterrupt([Interrupt(value=value)])
