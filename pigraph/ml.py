"""ML — machine learning helpers as graph nodes (sklearn optional).

يعمل بدون sklearn عبر نماذج أساسية مدمجة، ويتسارع مع sklearn عند توفرها.

أمثلة:
    from pigraph.ml import make_train_node, make_predict_node, make_ml_graph, accuracy, mse

    g = make_ml_graph(model="logreg", target="label")  # sklearn إن وُجد وإلا baseline
    out = g.invoke({"dataset": [{"x": 1, "label": 0}, {"x": 2, "label": 1}]})
    # out = {"model": ..., "preds": [...], "metrics": {"accuracy": ...}}

    # يدوياً:
    g2 = StateGraph(dict)
    g2.add_node("train", make_train_node(MyModel(), features=["x"], target="label"))
    g2.add_node("predict", make_predict_node(features=["x"]))
"""
from __future__ import annotations

import math
from collections import Counter

__all__ = [
    "accuracy", "precision", "recall", "f1",
    "mse", "rmse", "mae", "r2",
    "MajorityClassifier", "MeanRegressor",
    "make_train_node", "make_predict_node", "make_eval_node",
    "make_ml_graph", "cross_validate",
]


# ------------------------------------------------------------- metrics ---
def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = list(y_true), list(y_pred)
    if not y_true:
        return 0.0
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


def _bin(y_true, y_pred, pos=1):
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == pos and b == pos)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a != pos and b == pos)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == pos and b != pos)
    return tp, fp, fn


def precision(y_true, y_pred, pos=1) -> float:
    tp, fp, _ = _bin(list(y_true), list(y_pred), pos)
    return tp / (tp + fp) if (tp + fp) else 0.0


def recall(y_true, y_pred, pos=1) -> float:
    tp, _, fn = _bin(list(y_true), list(y_pred), pos)
    return tp / (tp + fn) if (tp + fn) else 0.0


def f1(y_true, y_pred, pos=1) -> float:
    p, r = precision(y_true, y_pred, pos), recall(y_true, y_pred, pos)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def mse(y_true, y_pred) -> float:
    a, b = list(y_true), list(y_pred)
    if not a:
        return 0.0
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)) / len(a)


def rmse(y_true, y_pred) -> float:
    return math.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred) -> float:
    a, b = list(y_true), list(y_pred)
    if not a:
        return 0.0
    return sum(abs(float(x) - float(y)) for x, y in zip(a, b)) / len(a)


def r2(y_true, y_pred) -> float:
    a = [float(x) for x in y_true]
    b = [float(x) for x in y_pred]
    if not a:
        return 0.0
    m = sum(a) / len(a)
    ss_tot = sum((x - m) ** 2 for x in a)
    ss_res = sum((x - y) ** 2 for x, y in zip(a, b))
    return 1 - ss_res / ss_tot if ss_tot else 0.0


# ------------------------------------------------------- baseline models ---
class MajorityClassifier:
    """مصنّف أساسي: يتوقع الفئة الأكثر تكراراً (يعمل بدون sklearn)."""

    def __init__(self):
        self.majority = 0

    def fit(self, X, y):
        self.majority = Counter(list(y)).most_common(1)[0][0] if len(list(y)) else 0
        return self

    def predict(self, X):
        return [self.majority for _ in X]


class MeanRegressor:
    """منحدر أساسي: يتوقع متوسط y."""

    def __init__(self):
        self.mean = 0.0

    def fit(self, X, y):
        vals = [float(v) for v in y]
        self.mean = sum(vals) / len(vals) if vals else 0.0
        return self

    def predict(self, X):
        return [self.mean for _ in X]


def _default_model(kind: str = "auto"):
    """نموذج افتراضي: sklearn إن وُجد وإلا baseline."""
    try:
        if kind in ("logreg", "auto", "clf"):
            from sklearn.linear_model import LogisticRegression  # type: ignore
            return LogisticRegression(max_iter=500)
        if kind == "reg":
            from sklearn.linear_model import LinearRegression  # type: ignore
            return LinearRegression()
    except Exception:
        pass
    return MajorityClassifier() if kind != "reg" else MeanRegressor()


def _rows_to_xy(dataset: list[dict], features, target):
    X = [[r.get(f) for f in features] if features else [v for k, v in r.items() if k != target] for r in dataset]
    # حوّل القيم لعددية قدر الإمكان
    Xn = []
    for row in X:
        nr = []
        for v in row:
            try:
                nr.append(float(v))
            except Exception:
                nr.append(float(hash(str(v)) % 1000) / 1000.0)
        Xn.append(nr)
    y = [r.get(target) for r in dataset]
    return Xn, y


# ------------------------------------------------------- graph nodes ---
def make_train_node(model=None, features=None, target: str = "label",
                    in_key: str = "dataset", out_key: str = "model", kind: str = "auto"):
    """عقدة تدريب: state[in_key] (list[dict]) -> state[out_key] (model مدرّب)."""
    mdl = model if model is not None else _default_model(kind)

    def _node(state: dict) -> dict:
        data = state.get(in_key, []) or []
        feats = features or ([c for c in data[0].keys() if c != target] if data else [])
        X, y = _rows_to_xy(data, feats, target)
        try:
            mdl.fit(X, y)
        except Exception:
            fb = MajorityClassifier()
            try:
                fb.fit(X, y)
                return {out_key: fb, "features": feats, "target": target}
            except Exception as e:
                return {"train_error": str(e)}
        return {out_key: mdl, "features": feats, "target": target}

    _node.__name__ = "ml_train"
    return _node


def make_predict_node(model=None, features=None, in_key: str = "dataset",
                      model_key: str = "model", out_key: str = "preds"):
    """عقدة تنبؤ: تستخدم state[model_key] أو model المعطى."""
    def _node(state: dict) -> dict:
        mdl = model if model is not None else state.get(model_key)
        data = state.get(in_key, []) or []
        feats = features or state.get("features") or ([c for c in data[0].keys() if c != state.get("target", "label")] if data else [])
        target = state.get("target", "label")
        X, _ = _rows_to_xy(data, feats, target)
        if mdl is None:
            return {out_key: []}
        try:
            preds = list(mdl.predict(X))
        except Exception as e:
            return {"predict_error": str(e)}
        return {out_key: preds}

    _node.__name__ = "ml_predict"
    return _node


def make_eval_node(metrics=("accuracy",), target: str = "label",
                   preds_key: str = "preds", in_key: str = "dataset", out_key: str = "metrics"):
    """عقدة تقييم: تحسب المقاييس المطلوبة."""
    _fns = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

    def _node(state: dict) -> dict:
        data = state.get(in_key, []) or []
        preds = state.get(preds_key, []) or []
        tgt = state.get("target", target)
        y_true = [r.get(tgt) for r in data][:len(preds)]
        out = {}
        for m in metrics:
            fn = _fns.get(m)
            if fn is None:
                continue
            try:
                out[m] = float(fn(y_true, preds))
            except Exception:
                out[m] = 0.0
        return {out_key: out}

    _node.__name__ = "ml_eval"
    return _node


def make_ml_graph(model=None, features=None, target: str = "label",
                  metrics=("accuracy",), in_key: str = "dataset", kind: str = "auto",
                  checkpointer=None):
    """رسم ML كامل: train -> predict -> evaluate في 3 عقد."""
    from .graph import END, START, StateGraph
    g = StateGraph(dict)
    g.add_node("train", make_train_node(model, features, target, in_key, "model", kind))
    g.add_node("predict", make_predict_node(None, features, in_key, "model", "preds"))
    g.add_node("evaluate", make_eval_node(metrics, target, "preds", in_key, "metrics"))
    g.add_edge(START, "train")
    g.add_edge("train", "predict")
    g.add_edge("predict", "evaluate")
    g.add_edge("evaluate", END)
    return g.compile(checkpointer=checkpointer)


def cross_validate(model_fn, dataset: list[dict], features=None, target: str = "label",
                   k: int = 5, metric: str = "accuracy", seed: int = 42) -> dict:
    """تحقق متقاطع K-Fold نقي (يعمل بدون sklearn)."""
    import random as _rnd
    data = list(dataset or [])
    rnd = _rnd.Random(seed)
    idx = list(range(len(data)))
    rnd.shuffle(idx)
    folds = [[] for _ in range(max(1, k))]
    for j, i in enumerate(idx):
        folds[j % len(folds)].append(i)
    _fns = {"accuracy": accuracy, "f1": f1, "mse": mse, "mae": mae, "r2": r2}
    fn = _fns.get(metric, accuracy)
    scores = []
    feats = features or ([c for c in data[0].keys() if c != target] if data else [])
    for f in range(len(folds)):
        te = set(folds[f])
        tr_rows = [data[i] for i in range(len(data)) if i not in te]
        te_rows = [data[i] for i in folds[f]]
        if not tr_rows or not te_rows:
            continue
        Xtr, ytr = _rows_to_xy(tr_rows, feats, target)
        Xte, yte = _rows_to_xy(te_rows, feats, target)
        mdl = model_fn() if callable(model_fn) else model_fn
        try:
            mdl.fit(Xtr, ytr)
            preds = list(mdl.predict(Xte))
            scores.append(float(fn(yte, preds)))
        except Exception:
            scores.append(0.0)
    return {"metric": metric, "scores": scores,
            "mean": sum(scores) / len(scores) if scores else 0.0}
