"""اختبارات pygraph — توافق langgraph + سرعة."""
import asyncio
import time

from pygraph import (END, START, Command, MemorySaver, MessageGraph, Send,
                     SqliteSaver, StateGraph, StateSnapshot, ToolNode,
                     add_messages, create_react_agent, entrypoint, interrupt,
                     task, tools_condition)


def test_linear():
    g = StateGraph(dict)
    g.add_node("a", lambda s: {"x": s.get("x", 0) + 1})
    g.add_node("b", lambda s: {"y": s["x"] * 2})
    g.add_edge(START, "a"); g.add_edge("a", "b"); g.add_edge("b", END)
    app = g.compile()
    assert app.invoke({"x": 1}) == {"x": 2, "y": 4}


def test_conditional_mapping():
    g = StateGraph(dict)
    g.add_node("s", lambda s: {"n": 5})
    g.add_node("even", lambda s: {"r": "even"})
    g.add_node("odd", lambda s: {"r": "odd"})
    g.set_entry_point("s")
    g.add_conditional_edges("s", lambda s: "E" if s["n"] % 2 == 0 else "O",
                            {"E": "even", "O": "odd"})
    g.add_edge("even", END); g.add_edge("odd", END)
    assert g.compile().invoke({})["r"] == "odd"


def test_command_goto():
    g = StateGraph(dict)
    g.add_node("a", lambda s: Command(update={"x": 1}, goto="c"))
    g.add_node("b", lambda s: {"x": 999})
    g.add_node("c", lambda s: {"y": s["x"] + 1})
    g.add_edge(START, "a"); g.add_edge("b", "c"); g.add_edge("c", END)
    out = g.compile().invoke({})
    assert out == {"x": 1, "y": 2}


def test_send_fanout():
    g = StateGraph(dict)
    g.add_node("fan", lambda s: [Send("w", {"i": i}) for i in range(4)])
    g.add_node("w", lambda s: {"vals": [s["__send__"]["i"] * 10]})
    g.set_entry_point("fan")
    g.add_edge("w", END)
    # reducer ضمني؟ نستخدم Annotated
    from typing import Annotated
    from typing import TypedDict
    class S(TypedDict, total=False):
        vals: Annotated[list, lambda a, b: (a or []) + b]
    g2 = StateGraph(S)
    g2.add_node("fan", lambda s: [Send("w", {"i": i}) for i in range(4)])
    g2.add_node("w", lambda s: {"vals": [s["__send__"]["i"]]})
    g2.set_entry_point("fan"); g2.add_edge("w", END)
    out = g2.compile().invoke({})
    assert sorted(out["vals"]) == [0, 1, 2, 3]


def test_add_messages_upsert():
    a = [{"id": "1", "content": "hi"}, {"id": "2", "content": "old"}]
    b = [{"id": "2", "content": "new"}, {"id": "3", "content": "x"}]
    out = add_messages(a, b)
    assert [m["content"] for m in out] == ["hi", "new", "x"]
    # remove
    out2 = add_messages(out, [{"type": "remove", "id": "1"}])
    assert all((m.get("id") if isinstance(m, dict) else None) != "1" for m in out2)


def test_message_graph():
    mg = MessageGraph()
    mg.add_node("a", lambda s: {"messages": [{"role": "user", "content": "hi"}]})
    mg.add_edge(START, "a"); mg.add_edge("a", END)
    out = mg.compile().invoke({})
    assert out["messages"][0]["content"] == "hi"
    out2 = mg.compile().invoke({"messages": [{"role": "user", "content": "yo"}]})
    assert len(out2["messages"]) >= 1


def test_interrupt_resume():
    def node(s):
        ans = interrupt({"q": "continue?"})
        return {"ans": ans}
    g = StateGraph(dict)
    g.add_node("n", node)
    g.add_edge(START, "n"); g.add_edge("n", END)
    app = g.compile(checkpointer=MemorySaver())
    tid = {"configurable": {"thread_id": "h1"}}
    out = app.invoke({"x": 1}, config=tid)
    assert "__interrupt__" in out
    out2 = app.invoke(Command(resume="yes"), config=tid)
    assert out2.get("ans") == "yes"


def test_interrupt_before_after():
    g = StateGraph(dict)
    g.add_node("a", lambda s: {"x": 1})
    g.add_node("b", lambda s: {"y": 2})
    g.add_edge(START, "a"); g.add_edge("a", "b"); g.add_edge("b", END)
    app = g.compile(checkpointer=MemorySaver(), interrupt_before=["b"])
    tid = {"configurable": {"thread_id": "ib1"}}
    app.invoke({"z": 0}, config=tid)
    st = app.get_state(tid)
    assert isinstance(st, StateSnapshot) and st["x"] == 1
    # update + resume يدوي عبر تحديث الحالة
    app.update_state(tid, {"extra": 5})
    assert app.get_state(tid)["extra"] == 5


def test_subgraph():
    sub = StateGraph(dict)
    sub.add_node("s1", lambda s: {"inner": s.get("v", 0) + 100})
    sub.add_edge(START, "s1"); sub.add_edge("s1", END)
    sub_app = sub.compile()
    g = StateGraph(dict)
    g.add_node("parent", sub_app)
    g.add_edge(START, "parent"); g.add_edge("parent", END)
    out = g.compile().invoke({"v": 1})
    assert out["inner"] == 101


def test_stream_modes():
    g = StateGraph(dict)
    g.add_node("a", lambda s: {"x": 1})
    g.add_node("b", lambda s: {"y": 2})
    g.add_edge(START, "a"); g.add_edge("a", "b"); g.add_edge("b", END)
    app = g.compile()
    ups = list(app.stream({"x": 0}, stream_mode="updates"))
    assert ups == [{"a": {"x": 1}}, {"b": {"y": 2}}]
    vals = list(app.stream({"x": 0}, stream_mode="values"))
    assert vals[-1] == {"x": 1, "y": 2}


def test_batch_and_history():
    g = StateGraph(dict)
    g.add_node("a", lambda s: {"x": s.get("x", 0) + 1})
    g.add_edge(START, "a"); g.add_edge("a", END)
    app = g.compile(checkpointer=MemorySaver())
    outs = app.batch([{"x": i} for i in range(4)])
    assert [o["x"] for o in outs] == [1, 2, 3, 4]
    tid = {"configurable": {"thread_id": "t9"}}
    app.invoke({"x": 1}, config=tid)
    app.invoke({"x": 2}, config=tid)
    hist = app.get_state_history(tid)
    assert len(hist) >= 2


def test_sqlite_saver(tmp_path):
    db = str(tmp_path / "c.db")
    with SqliteSaver(db) as sv:
        g = StateGraph(dict)
        g.add_node("a", lambda s: {"x": s.get("x", 0) + 5})
        g.add_edge(START, "a"); g.add_edge("a", END)
        app = g.compile(checkpointer=sv)
        tid = {"configurable": {"thread_id": "s1"}}
        assert app.invoke({"x": 1}, config=tid)["x"] == 6
        assert app.get_state(tid)["x"] == 6


def test_async():
    async def anode(s):
        await asyncio.sleep(0)
        return {"x": s.get("x", 0) + 1}
    g = StateGraph(dict)
    g.add_node("a", anode)
    g.add_edge(START, "a"); g.add_edge("a", END)
    app = g.compile()
    assert asyncio.run(app.ainvoke({"x": 1}))["x"] == 2
    async def _collect():
        return [c async for c in app.astream({"x": 0})]
    assert asyncio.run(_collect()) == [{"a": {"x": 1}}]


def test_toolnode_and_condition():
    def add(a: int, b: int) -> int:
        return a + b
    add.name = "add"
    node = ToolNode([add])
    st = {"messages": [{"role": "ai", "content": "",
                        "tool_calls": [{"id": "c1", "name": "add", "args": {"a": 2, "b": 3}}]}]}
    out = node(st)
    m0 = out["messages"][0]
    c0 = m0["content"] if isinstance(m0, dict) else getattr(m0, "content", str(m0))
    assert "5" in c0
    assert tools_condition(st) == "tools"
    assert tools_condition({"messages": [{"role": "user", "content": "hi"}]}) == END


def test_react_agent():
    def fake_model(msgs):
        # أول مرة: اطلب أداة، ثاني مرة: أجب
        last = msgs[-1] if msgs else {}
        c = last.get("content", "") if isinstance(last, dict) else getattr(last, "content", "")
        if "done" in str(c):
            return {"role": "ai", "content": "final"}
        return {"role": "ai", "content": "call tool",
                "tool_calls": [{"id": "1", "name": "echo", "args": {"x": "done"}}]}
    def echo(x: str = "") -> str:
        return f"echo:{x}"
    echo.name = "echo"
    # نموذج بسيط ينهي بعد أداة واحدة
    calls = {"n": 0}
    def model2(state):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"messages": [{"role": "ai", "content": "go",
                                   "tool_calls": [{"id": "1", "name": "echo", "args": {"x": "hi"}}]}]}
        return {"messages": [{"role": "ai", "content": "final"}]}
    app = create_react_agent(model2, [echo])
    out = app.invoke({"messages": [{"role": "user", "content": "start"}]}, config=None)
    assert any("final" in str(m.get("content", "")) for m in out["messages"] if isinstance(m, dict))


def test_functional():
    @task
    def double(x: int) -> int:
        return x * 2

    @entrypoint()
    def flow(inputs):
        return {"y": double(inputs["x"]) + 1}

    assert flow.invoke({"x": 3}) == {"y": 7}
    assert flow.batch([{"x": 1}, {"x": 2}]) == [{"y": 3}, {"y": 5}]


def test_speed_smoke():
    g = StateGraph(dict)
    for i in range(30):
        g.add_node(f"n{i}", (lambda s, i=i: {"x": s.get("x", 0) + 1}))
        if i == 0:
            g.add_edge(START, "n0")
        elif i > 0:
            g.add_edge(f"n{i-1}", f"n{i}")
    g.add_edge("n29", END)
    app = g.compile()
    t0 = time.perf_counter()
    for _ in range(200):
        app.invoke({"x": 0})
    dt = time.perf_counter() - t0
    # 200 * 30 عقدة يجب أن تنتهي بسرعة (< 5 ثوانٍ على أي جهاز حديث)
    assert dt < 5.0, f"too slow: {dt:.2f}s"
