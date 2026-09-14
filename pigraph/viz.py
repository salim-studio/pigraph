"""Viz & observability — mermaid / ascii / plots (matplotlib optional).

أمثلة:
    from pigraph.viz import graph_to_mermaid, graph_to_ascii, plot_history
    print(graph_to_mermaid(app))
    print(graph_to_ascii(app))
    plot_history([0.9, 0.5, 0.2], save="loss.png")
"""
from __future__ import annotations

__all__ = ["graph_to_mermaid", "graph_to_ascii", "save_mermaid",
           "plot_history", "TimerCallback"]


def _edges_of(app) -> tuple[str | None, dict, dict]:
    b = getattr(app, "builder", None)
    if b is None:
        return None, {}, {}
    return getattr(b, "entry", None), dict(getattr(b, "edges", {})), dict(getattr(b, "conditional", {}))


def graph_to_mermaid(app, title: str = "pigraph") -> str:
    """تحويل الرسم إلى مخطط Mermaid (يُلصق في GitHub/markdown)."""
    entry, edges, cond = _edges_of(app)
    lines = ["```mermaid", "flowchart TD"]
    nodes: set = set()
    for a, dests in edges.items():
        nodes.add(a)
        for d in dests:
            nodes.add(d)
            lines.append(f"    {a} --> {d}")
    for src, (path, mapping) in cond.items():
        nodes.add(src)
        name = getattr(path, "__name__", "cond")
        if isinstance(mapping, dict):
            for route, dest in mapping.items():
                dests = [dest] if isinstance(dest, str) else list(dest or [])
                for d in dests:
                    nodes.add(d)
                    lines.append(f'    {src} -. "{route}" .-> {d}')
        else:
            lines.append(f"    {src} -. {name} .-> cond_{src}")
    if entry:
        lines.append(f"    START(({entry} : entry)) --> {entry}")
    lines.append("```")
    return "\n".join(lines)


def graph_to_ascii(app) -> str:
    entry, edges, cond = _edges_of(app)
    out = [f"entry: {entry or '?'}"]
    for a, dests in (edges or {}).items():
        mark = " ?(cond)" if a in cond else ""
        out.append(f"[{a}]{mark} -> " + (", ".join(dests) if dests else "(end)"))
    for src, (path, mapping) in (cond or {}).items():
        out.append(f"  cond({src}): {getattr(path, '__name__', path)} -> {mapping}")
    return "\n".join(out)


def save_mermaid(app, path: str = "graph.md") -> str:
    md = graph_to_mermaid(app)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Graph\n\n" + md + "\n")
    return path


def plot_history(values: list[float], labels=None, title: str = "history",
                 save: str | None = None, show: bool = False):
    """رسم منحنى (matplotlib إن وُجد وإلا ASCII) — مفيد لخسارة التدريب."""
    values = list(values or [])
    if not values:
        return None
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        # ASCII fallback
        lo, hi = min(values), max(values)
        span = (hi - lo) or 1.0
        print(f"== {title} ==")
        for i, v in enumerate(values):
            bar = "#" * max(1, int(40 * (v - lo) / span))
            print(f"{i:3d} {v:10.4f} {bar}")
        return None
    import matplotlib.pyplot as plt  # type: ignore
    if labels is not None:
        plt.plot(labels, values, marker="o")
    else:
        plt.plot(values, marker="o")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=120)
    if show:
        plt.show()
    plt.close()
    return save


class TimerCallback:
    """جمّع أزمنة العقد: غلّف أي عقدة بـ timed() ثم اعرض التقرير."""

    def __init__(self):
        self.records: dict[str, list[float]] = {}

    def timed(self, fn, name: str | None = None):
        import time as _t
        nm = name or getattr(fn, "__name__", "node")
        rec = self.records.setdefault(nm, [])

        def _w(state: dict):
            t0 = _t.perf_counter()
            try:
                return fn(state)
            finally:
                rec.append((_t.perf_counter() - t0) * 1000)
        _w.__name__ = nm
        return _w

    def report(self) -> dict:
        out = {}
        for k, vs in self.records.items():
            out[k] = {"n": len(vs), "mean_ms": sum(vs) / len(vs) if vs else 0,
                      "total_ms": sum(vs)}
        return out
