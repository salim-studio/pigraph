"""مقارنة سرعة pygraph: (1) مقابل baseline ثقيل بأسلوب langgraph (deepcopy+تحقق كل خطوة)،
(2) تسريع fan-out المتوازي، (3) langgraph مباشرة إن توفرت."""
import copy
import time

from pygraph import END, START, StateGraph


def build_chain(n=50):
    g = StateGraph(dict)
    for i in range(n):
        g.add_node(f"n{i}", (lambda s, i=i: {"x": s.get("x", 0) + 1}))
        g.add_edge(START if i == 0 else f"n{i-1}", f"n{i}")
    g.add_edge(f"n{n-1}", END)
    return g.compile()


def build_heavy_baseline(n=50):
    """يحاكي سلوك langgraph: نسخ عميق + سجل تاريخ + تحقق مخطط بعد كل عقدة."""
    nodes = [(lambda s: {"x": s.get("x", 0) + 1}) for _ in range(n)]

    def run(inputs):
        state = dict(inputs)
        history = []
        for fn in nodes:
            snap = copy.deepcopy(state)          # لقطة قناة كما في Pregel
            history.append(copy.deepcopy(state))  # سجل
            assert isinstance(state, dict)        # تحقق مخطط
            upd = fn(dict(snap))
            assert isinstance(upd, dict)
            state.update(upd)
        return state
    return run


def bench(fn, iters=300):
    t0 = time.perf_counter()
    for _ in range(iters):
        fn({"x": 0})
    return time.perf_counter() - t0


def main():
    app = build_chain(50)
    heavy = build_heavy_baseline(50)
    app.invoke({"x": 0}); heavy({"x": 0})
    t_fast = bench(app.invoke, 300)
    t_heavy = bench(heavy, 300)
    print(f"pygraph  50-node x300: {t_fast:.3f}s  ({300*50/t_fast:,.0f} nodes/s)")
    print(f"heavy-baseline (deepcopy+history+validate) x300: {t_heavy:.3f}s")
    print(f"speedup vs heavy baseline: {t_heavy/t_fast:.2f}x")

    # fan-out متوازٍ مع I/O: 8 عمال × sleep(2ms) -> المتسلسل 16ms، المتوازي ~3ms
    import time as _t
    import pygraph as _pg
    from typing import Annotated
    from typing import TypedDict
    class S(TypedDict, total=False):
        vals: Annotated[list, lambda a, b: (a or []) + b]
    g2 = StateGraph(S)

    def _w(s):
        _t.sleep(0.002)
        return {"vals": [s.get("i", 0)]}

    g2.add_node("fan", lambda s: [_pg.Send("w", {"i": i}) for i in range(8)])
    g2.add_node("w", _w)
    g2.set_entry_point("fan"); g2.add_edge("w", END)
    app2 = g2.compile()
    t0 = time.perf_counter()
    for _ in range(20):
        app2.invoke({})
    t_par = (time.perf_counter() - t0) / 20 * 1000
    print(f"pygraph fan-out x8 (2ms I/O each): {t_par:.1f}ms/invoke متوازية (المتسلسل كان سيحتاج ~16ms)")

    try:
        from langgraph.graph import StateGraph as LSG, START as LS, END as LE
        lg = LSG(dict)
        for i in range(50):
            lg.add_node(f"n{i}", (lambda s, i=i: {"x": s.get("x", 0) + 1}))
            lg.add_edge(LS if i == 0 else f"n{i-1}", f"n{i}")
        lg.add_edge("n49", LE)
        lapp = lg.compile()
        lapp.invoke({"x": 0})
        t0 = time.perf_counter()
        for _ in range(100):
            lapp.invoke({"x": 0})
        t_lg = time.perf_counter() - t0
        t_pg100 = bench(app.invoke, 100)
        print(f"langgraph 50-node x100: {t_lg:.3f}s | pygraph: {t_pg100:.3f}s | speedup: {t_lg/t_pg100:.2f}x")
    except Exception as e:
        print(f"(langgraph غير مثبت — تخطي المقارنة المباشرة: {e})")


if __name__ == "__main__":
    main()
