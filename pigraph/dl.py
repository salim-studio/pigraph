"""DL — deep learning helpers (numpy only by default, torch/keras optional).

- SimpleMLP: شبكة MLP خالصة بـ numpy (تصنيف/انحدار) — بدون أي اعتماد ثقيل.
- Numericalizer: تحويل نصوص/فئات إلى متجهات (hashing trick + TF-IDF بسيط).
- torch_train_node / keras adapters عند توفر المكتبات.

أمثلة:
    from pigraph.dl import SimpleMLP, make_dl_node
    mlp = SimpleMLP(input_dim=2, hidden=[8], output_dim=1, task="regression")
    mlp.fit(X, y, epochs=50)
    preds = mlp.predict(X)
"""
from __future__ import annotations

import math
import random

__all__ = ["SimpleMLP", "Numericalizer", "tfidf", "chunk_text",
           "make_dl_node", "torch_train_node", "has_torch", "has_tf"]


def has_torch() -> bool:
    try:
        import torch  # noqa  # type: ignore
        return True
    except Exception:
        return False


def has_tf() -> bool:
    try:
        import tensorflow  # noqa  # type: ignore
        return True
    except Exception:
        return False


# ------------------------------------------------------- text utils ---
def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """تقسيم نص إلى مقاطع — مفيد لـ RAG."""
    text = text or ""
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += max(1, size - overlap)
    return out


def tfidf(docs: list[str]):
    """TF-IDF بسيط خالص (بدون sklearn) — يعيد (vectors, vocab)."""
    import re as _re
    toks = [[w for w in _re.findall(r"\w+", (d or "").lower())] for d in docs]
    vocab = sorted({w for t in toks for w in t})
    idx = {w: i for i, w in enumerate(vocab)}
    n = len(docs)
    df = [0] * len(vocab)
    for t in toks:
        for w in set(t):
            df[idx[w]] += 1
    vecs = []
    for t in toks:
        v = [0.0] * len(vocab)
        c = len(t) or 1
        for w in t:
            i = idx[w]
            tf = t.count(w) / c
            idf = math.log((1 + n) / (1 + df[i])) + 1.0
            v[i] = tf * idf
        vecs.append(v)
    return vecs, vocab


class Numericalizer:
    """تحويل list[dict] إلى X,y رقمية (hashing للفئات + floats للأرقام)."""

    def __init__(self, features=None, target: str = "label", dim: int = 16):
        self.features = features
        self.target = target
        self.dim = dim

    def transform(self, rows: list[dict]):
        feats = self.features or ([c for c in rows[0].keys() if c != self.target] if rows else [])
        X = []
        for r in rows:
            v = []
            for f in feats:
                x = r.get(f)
                try:
                    v.append(float(x))
                except Exception:
                    v.append(float(hash(str(x)) % 10000) / 10000.0)
            # pad/truncate إلى dim
            if len(v) < self.dim:
                v = v + [0.0] * (self.dim - len(v))
            X.append(v[:self.dim] if self.dim else v)
        y = [r.get(self.target) for r in rows]
        return X, y, feats


# ------------------------------------------------------- SimpleMLP (numpy-optional) ---
class SimpleMLP:
    """MLP صغير: يعمل بـ numpy إن وُجد وإلا بقوائم بايثون خالصة.

    task: "regression" | "binary" | "multiclass"
    """

    def __init__(self, input_dim: int = 8, hidden=(16,), output_dim: int = 1,
                 task: str = "regression", lr: float = 0.05, seed: int = 42):
        self.input_dim = input_dim
        self.hidden = tuple(hidden or ())
        self.output_dim = output_dim
        self.task = task
        self.lr = lr
        self._np = None
        try:
            import numpy as _np  # type: ignore
            self._np = _np
        except Exception:
            self._np = None
        rnd = random.Random(seed)
        layers = [input_dim, *self.hidden, output_dim]
        if self._np is not None:
            np = self._np
            rs = np.random.RandomState(seed)
            self.W = [rs.randn(a, b) * math.sqrt(2.0 / max(1, a)) for a, b in zip(layers[:-1], layers[1:])]
            self.b = [np.zeros(b) for b in layers[1:]]
        else:
            self.W = [[[rnd.gauss(0, 0.5) for _ in range(b)] for _ in range(a)] for a, b in zip(layers[:-1], layers[1:])]
            self.b = [[0.0] * b for b in layers[1:]]

    # -- numpy path --
    def _fit_numpy(self, X, y, epochs: int):
        np = self._np
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        if y.ndim == 1 and self.output_dim == 1:
            y = y.reshape(-1, 1)
        for _ in range(epochs):
            # forward
            acts, pres = [X], []
            a = X
            for li, (W, b) in enumerate(zip(self.W, self.b)):
                z = a @ W + b
                pres.append(z)
                a = np.maximum(z, 0) if li < len(self.W) - 1 else z
                if li == len(self.W) - 1:
                    if self.task == "binary":
                        a = 1 / (1 + np.exp(-z))
                    elif self.task == "multiclass":
                        e = np.exp(z - z.max(axis=1, keepdims=True))
                        a = e / e.sum(axis=1, keepdims=True)
                acts.append(a)
            # grad output
            if self.task == "regression":
                d = (acts[-1] - y) / len(X)
            elif self.task == "binary":
                d = (acts[-1] - y) / len(X)
            else:
                d = acts[-1].copy()
                for i, yi in enumerate(y.astype(int).ravel() if y.size == len(X) else y.argmax(axis=1)):
                    d[i, int(yi)] -= 1
                d /= len(X)
            # backward
            for li in reversed(range(len(self.W))):
                gW = acts[li].T @ d
                gb = d.sum(axis=0)
                if li > 0:
                    d = (d @ self.W[li].T) * (acts[li] > 0)
                self.W[li] -= self.lr * gW
                self.b[li] -= self.lr * gb
        return self

    def _predict_numpy(self, X):
        np = self._np
        a = np.array(X, dtype=float)
        for li, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            if li < len(self.W) - 1:
                a = np.maximum(z, 0)
            else:
                if self.task == "binary":
                    a = (1 / (1 + np.exp(-z)) > 0.5).astype(int).ravel().tolist()
                    return a
                if self.task == "multiclass":
                    e = np.exp(z - z.max(axis=1, keepdims=True))
                    return e.argmax(axis=1).tolist()
                a = z
        return a.ravel().tolist() if a.size else []

    # -- pure python path (شبكة بطبقة خفية واحدة كحد أقصى للسرعة) --
    def _fit_pure(self, X, y, epochs: int):
        # انحدار لوجستي/خطي مبسط على الميزات مباشرة (تجاهل hidden للسرعة)
        dim = self.input_dim
        w = [0.0] * dim
        b = 0.0
        Y = [float(v) for v in y]
        for _ in range(epochs):
            for xi, yi in zip(X, Y):
                xi = (list(xi) + [0.0] * dim)[:dim]
                z = sum(a * c for a, c in zip(w, xi)) + b
                if self.task == "binary":
                    p = 1 / (1 + math.exp(-max(-30, min(30, z))))
                    err = p - yi
                else:
                    err = z - yi
                for j in range(dim):
                    w[j] -= self.lr * err * xi[j]
                b -= self.lr * err
        self._pure_w, self._pure_b = w, b
        return self

    def _predict_pure(self, X):
        w = getattr(self, "_pure_w", [0.0] * self.input_dim)
        b = getattr(self, "_pure_b", 0.0)
        out = []
        for xi in X:
            xi = (list(xi) + [0.0] * self.input_dim)[: self.input_dim]
            z = sum(a * c for a, c in zip(w, xi)) + b
            if self.task == "binary":
                out.append(1 if z > 0 else 0)
            else:
                out.append(z)
        return out

    # -- public --
    def fit(self, X, y, epochs: int = 50):
        X = [list(map(float, row)) if isinstance(row, (list, tuple)) else [float(row)] for row in X]
        if self._np is not None:
            return self._fit_numpy(X, y, epochs)
        return self._fit_pure(X, y, epochs)

    def predict(self, X):
        X = [list(map(float, row)) if isinstance(row, (list, tuple)) else [float(row)] for row in X]
        if self._np is not None:
            return self._predict_numpy(X)
        return self._predict_pure(X)


def make_dl_node(features=None, target: str = "label", hidden=(16,), task: str = "regression",
                 epochs: int = 50, in_key: str = "dataset", out_key: str = "dl_model"):
    """عقدة تدريب DL خالصة (SimpleMLP) داخل الرسم."""
    def _node(state: dict) -> dict:
        rows = state.get(in_key, []) or []
        num = Numericalizer(features, target)
        X, y, feats = num.transform(rows)
        dim = len(X[0]) if X else (len(feats) or 4)
        y_vals = [0 if v is None else v for v in y]
        if task == "binary":
            y_vals = [1 if str(v) in ("1", "True", "true", "yes") or v == 1 else 0 for v in y_vals]
            out_dim = 1
        elif task == "multiclass":
            labels = sorted(set(map(str, y_vals)))
            mp = {v: i for i, v in enumerate(labels)}
            y_vals = [mp[str(v)] for v in y_vals]
            out_dim = max(2, len(labels))
        else:
            y_vals = [float(v or 0) for v in y_vals]
            out_dim = 1
        mlp = SimpleMLP(input_dim=dim, hidden=hidden, output_dim=out_dim, task=task)
        mlp.fit(X, y_vals, epochs=epochs)
        return {out_key: mlp, "dl_features": feats or [f"f{i}" for i in range(dim)]}

    _node.__name__ = "dl_train"
    return _node


def torch_train_node(model_fn, loss_fn=None, optimizer_fn=None, epochs: int = 3,
                     in_key: str = "dataloader", out_key: str = "torch_model"):
    """عقدة تدريب PyTorch — تتطلب torch (وإلا ترفع ImportError واضحاً)."""
    def _node(state: dict) -> dict:
        try:
            import torch  # type: ignore
        except ImportError as e:
            raise ImportError("torch_train_node يحتاج torch: pip install torch") from e
        model = model_fn() if callable(model_fn) else model_fn
        loader = state.get(in_key, [])
        opt = optimizer_fn(model.parameters()) if optimizer_fn else torch.optim.Adam(model.parameters())
        loss_fn_ = loss_fn or torch.nn.MSELoss()
        model.train()
        for _ in range(epochs):
            for batch in loader:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    xb, yb = batch
                elif isinstance(batch, dict):
                    xb, yb = batch["x"], batch["y"]
                else:
                    continue
                opt.zero_grad()
                loss = loss_fn_(model(xb), yb)
                loss.backward()
                opt.step()
        return {out_key: model}

    _node.__name__ = "torch_train"
    return _node
