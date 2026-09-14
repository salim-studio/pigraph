"""Data — ETL & preprocessing for analysts (pandas optional, stdlib fallback).

أمثلة:
    from pigraph.data import load_csv, clean_missing, normalize, train_test_split, DataPipeline

    rows = load_csv("data.csv")                       # list[dict] بدون pandas
    rows = clean_missing(rows, strategy="mean")
    X_train, X_test = train_test_split(rows, test_size=0.2, seed=42)

    # Pipeline + رسم بياني:
    pipe = DataPipeline()
    pipe.add_step(lambda rows: clean_missing(rows))
    pipe.add_step(lambda rows: normalize(rows, columns=["age"]))
    app = pipe.to_graph(out_key="dataset")  # CompiledGraph جاهز
"""
from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter

__all__ = [
    "load_csv", "save_csv", "load_json", "save_json",
    "clean_missing", "fill_missing", "drop_duplicates",
    "normalize", "standardize", "one_hot_encode", "label_encode",
    "train_test_split", "describe", "group_by_agg",
    "DataPipeline", "etl_node",
]


# ---------------------------------------------------------------- IO ---
def load_csv(path: str, delimiter: str = ",") -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def save_csv(rows: list[dict], path: str):
    rows = list(rows or [])
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def to_dataframe(rows: list[dict]):
    """تحويل إلى pandas.DataFrame — يتطلب pandas."""
    try:
        import pandas as _pd  # type: ignore
    except ImportError as e:
        raise ImportError("to_dataframe يحتاج pandas: pip install pandas") from e
    return _pd.DataFrame(list(rows or []))


def from_dataframe(df) -> list[dict]:
    return df.to_dict(orient="records")


# ------------------------------------------------------- cleaning ---
def _is_missing(v) -> bool:
    return v is None or v == "" or (isinstance(v, float) and math.isnan(v)) or str(v).lower() in ("na", "n/a", "null", "none")


def clean_missing(rows: list[dict], strategy: str = "drop", fill_value=None) -> list[dict]:
    """strategy: drop | mean | median | mode | value."""
    rows = [dict(r) for r in (rows or [])]
    if not rows:
        return rows
    if strategy == "drop":
        return [r for r in rows if not any(_is_missing(v) for v in r.values())]
    return fill_missing(rows, strategy=strategy, fill_value=fill_value)


def fill_missing(rows: list[dict], strategy: str = "mean", fill_value=None, columns=None) -> list[dict]:
    rows = [dict(r) for r in (rows or [])]
    if not rows:
        return rows
    cols = columns or list(rows[0].keys())
    stats: dict = {}
    for c in cols:
        vals = [r[c] for r in rows if not _is_missing(r.get(c))]
        nums: list[float] = []
        for v in vals:
            try:
                nums.append(float(v))
            except Exception:
                pass
        if strategy == "mean" and nums:
            stats[c] = sum(nums) / len(nums)
        elif strategy == "median" and nums:
            s = sorted(nums)
            stats[c] = s[len(s) // 2]
        elif strategy == "mode" and vals:
            stats[c] = Counter(vals).most_common(1)[0][0]
        elif strategy == "value":
            stats[c] = fill_value
        else:
            stats[c] = fill_value if fill_value is not None else (Counter(vals).most_common(1)[0][0] if vals else None)
    out = []
    for r in rows:
        r = dict(r)
        for c in cols:
            if _is_missing(r.get(c)):
                r[c] = stats[c]
        out.append(r)
    return out


def drop_duplicates(rows: list[dict], keys=None) -> list[dict]:
    seen: set = set()
    out = []
    for r in rows or []:
        k = tuple(r.get(c) for c in (keys or sorted(r.keys())))
        key = json.dumps(k, default=str, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


# ------------------------------------------------------- encoding/scaling ---
def _to_float(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def normalize(rows: list[dict], columns=None) -> list[dict]:
    """min-max إلى [0,1] للأعمدة الرقمية."""
    rows = [dict(r) for r in (rows or [])]
    if not rows:
        return rows
    cols = columns or list(rows[0].keys())
    bounds = {}
    for c in cols:
        nums = [_to_float(r.get(c)) for r in rows]
        lo, hi = min(nums), max(nums)
        bounds[c] = (lo, hi)
    out = []
    for r in rows:
        r = dict(r)
        for c in cols:
            lo, hi = bounds[c]
            r[c] = 0.0 if hi == lo else (_to_float(r.get(c)) - lo) / (hi - lo)
        out.append(r)
    return out


def standardize(rows: list[dict], columns=None) -> list[dict]:
    """z-score (mean=0, std=1)."""
    rows = [dict(r) for r in (rows or [])]
    if not rows:
        return rows
    cols = columns or list(rows[0].keys())
    stats = {}
    for c in cols:
        nums = [_to_float(r.get(c)) for r in rows]
        m = sum(nums) / len(nums) if nums else 0.0
        var = sum((x - m) ** 2 for x in nums) / len(nums) if nums else 0.0
        stats[c] = (m, math.sqrt(var) or 1.0)
    out = []
    for r in rows:
        r = dict(r)
        for c in cols:
            m, s = stats[c]
            r[c] = (_to_float(r.get(c)) - m) / s
        out.append(r)
    return out


def one_hot_encode(rows: list[dict], column: str, prefix: str | None = None) -> list[dict]:
    vals = sorted({str(r.get(column)) for r in (rows or [])})
    px = prefix or column
    out = []
    for r in (rows or []):
        r = dict(r)
        v = str(r.pop(column, None))
        for c in vals:
            r[f"{px}_{c}"] = 1 if v == c else 0
        out.append(r)
    return out


def label_encode(rows: list[dict], column: str):
    """يعيد (rows, mapping)."""
    vals = sorted({str(r.get(column)) for r in (rows or [])})
    mp = {v: i for i, v in enumerate(vals)}
    out = []
    for r in (rows or []):
        r = dict(r)
        r[column] = mp[str(r.get(column))]
        out.append(r)
    return out, mp


def train_test_split(rows, test_size: float = 0.2, seed: int = 42):
    """تقسيم list أو DataFrame — يستخدم sklearn عند توفرها وإلا خلطاً نقياً."""
    try:
        from sklearn.model_selection import train_test_split as _sk  # type: ignore
        return _sk(rows, test_size=test_size, random_state=seed)
    except Exception:
        pass
    data = list(rows)
    rnd = random.Random(seed)
    idx = list(range(len(data)))
    rnd.shuffle(idx)
    n_test = int(len(data) * test_size)
    test_idx = set(idx[:n_test])
    test = [data[i] for i in sorted(test_idx)]
    train = [data[i] for i in range(len(data)) if i not in test_idx]
    return train, test


def describe(rows: list[dict]) -> dict:
    """إحصاءات وصفية سريعة لكل عمود رقمي."""
    rows = list(rows or [])
    if not rows:
        return {}
    out: dict = {"n": len(rows), "columns": {}}
    for c in rows[0].keys():
        nums = []
        for r in rows:
            try:
                nums.append(float(r[c]))
            except Exception:
                pass
        if nums:
            nums.sort()
            m = sum(nums) / len(nums)
            out["columns"][c] = {
                "count": len(nums), "mean": m,
                "min": nums[0], "max": nums[-1],
                "median": nums[len(nums) // 2],
                "std": math.sqrt(sum((x - m) ** 2 for x in nums) / len(nums)),
            }
        else:
            out["columns"][c] = {"count": len(rows), "unique": len({str(r.get(c)) for r in rows})}
    return out


def group_by_agg(rows: list[dict], by: str, agg: str = "count", target: str | None = None) -> list[dict]:
    groups: dict = {}
    for r in (rows or []):
        groups.setdefault(str(r.get(by)), []).append(r)
    out = []
    for k, g in groups.items():
        if agg == "count":
            out.append({by: k, "count": len(g)})
        elif target:
            nums = [_to_float(r.get(target)) for r in g]
            if agg == "sum":
                out.append({by: k, "sum": sum(nums)})
            elif agg == "mean":
                out.append({by: k, "mean": sum(nums) / len(nums) if nums else 0})
            elif agg == "max":
                out.append({by: k, "max": max(nums) if nums else None})
            elif agg == "min":
                out.append({by: k, "min": min(nums) if nums else None})
            else:
                out.append({by: k, "count": len(g)})
    return out


class DataPipeline:
    """سلسلة خطوات ETL تُشغَّل مباشرة أو تُحوَّل إلى StateGraph.

    مثال:
        pipe = DataPipeline([lambda r: clean_missing(r), lambda r: normalize(r, ["age"])])
        clean = pipe.run(rows)
        app = pipe.to_graph(in_key="rows", out_key="dataset")
    """

    def __init__(self, steps=None):
        self.steps: list = list(steps or [])

    def add_step(self, fn, name: str | None = None):
        fn.__name__ = name or getattr(fn, "__name__", f"step{len(self.steps)}")
        self.steps.append(fn)
        return self

    def run(self, rows: list[dict]) -> list[dict]:
        data = list(rows or [])
        for fn in self.steps:
            data = fn(data)
        return data

    def to_graph(self, in_key: str = "rows", out_key: str = "dataset", checkpointer=None):
        from .graph import END, START, StateGraph
        steps = list(self.steps)

        def _run(state: dict) -> dict:
            data = state.get(in_key, [])
            for fn in steps:
                data = fn(data)
            return {out_key: data}

        _run.__name__ = "etl_pipeline"
        g = StateGraph(dict)
        g.add_node("etl", _run)
        g.add_edge(START, "etl")
        g.add_edge("etl", END)
        return g.compile(checkpointer=checkpointer)


def etl_node(fn, in_key: str = "rows", out_key: str = "rows"):
    """غلّف دالة ETL كعقدة رسم: state[in_key] -> state[out_key]."""
    def _node(state: dict) -> dict:
        return {out_key: fn(state.get(in_key, []))}
    _node.__name__ = getattr(fn, "__name__", "etl_node")
    return _node
