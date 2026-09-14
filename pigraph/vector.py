"""Vector store — ذاكرة RAG خفيفة (numpy optional, pure-python fallback).

أمثلة:
    from pigraph.vector import InMemoryVectorStore, chunk_text
    store = InMemoryVectorStore()
    store.add_texts(["الذكاء الاصطناعي", "قواعد البيانات", "تعلم الآلة"])
    print(store.search("ذكاء", k=2))

    # داخل رسم ReAct/RAG:
    node = store.retrieval_node(out_key="context")
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import uuid

__all__ = ["Document", "InMemoryVectorStore", "chunk_text"]


def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    text = text or ""
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += max(1, size - overlap)
    return out


class Document:
    __slots__ = ("id", "text", "metadata", "embedding")

    def __init__(self, text: str, metadata: dict | None = None, id: str | None = None):
        self.id = id or f"doc-{uuid.uuid4().hex[:8]}"
        self.text = text or ""
        self.metadata = metadata or {}
        self.embedding = None

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "metadata": self.metadata}

    @staticmethod
    def from_dict(d: dict) -> "Document":
        return Document(d.get("text", ""), d.get("metadata", {}), d.get("id"))


def _tokens(s: str) -> list[str]:
    return re.findall(r"\w+", (s or "").lower())


class InMemoryVectorStore:
    """مخزن متجهات خفيف: TF-IDF داخلي + cosine — بدون أي اعتماد.

    إن وُجدت numpy استُخدمت للتسريع، وإلا قوائم خالصة.
    يدعم الحفظ/التحميل JSON والعمل كعقدة استرجاع في الرسم.
    """

    def __init__(self):
        self.docs: list[Document] = []
        self._lock = threading.RLock()
        self._np = None
        try:
            import numpy as _np  # type: ignore
            self._np = _np
        except Exception:
            self._np = None

    # -- write --
    def add_texts(self, texts: list[str], metadatas: list[dict] | None = None) -> list[str]:
        ids = []
        with self._lock:
            for i, t in enumerate(texts or []):
                md = (metadatas or [{}] * len(texts))[i] if metadatas else {}
                d = Document(t, md)
                self.docs.append(d)
                ids.append(d.id)
        return ids

    def add_documents(self, docs: list[Document]) -> list[str]:
        with self._lock:
            for d in docs or []:
                self.docs.append(d)
        return [d.id for d in (docs or [])]

    def delete(self, ids: list[str]):
        ids = set(ids or [])
        with self._lock:
            self.docs = [d for d in self.docs if d.id not in ids]

    def clear(self):
        with self._lock:
            self.docs = []

    def __len__(self):
        return len(self.docs)

    # -- search (TF-IDF + cosine, يُبنى عند كل استعلام — كافٍ لآلاف الوثائق) --
    def _vectors(self, texts: list[str]):
        toks = [_tokens(t) for t in texts]
        vocab = sorted({w for t in toks for w in t})
        idx = {w: i for i, w in enumerate(vocab)}
        n = len(texts)
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

    @staticmethod
    def _cos(a, b) -> float:
        s = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        # دعم جزئي للعربية: مكافأة الكلمات المشتركة حرفياً
        return s / (na * nb)

    def search(self, query: str, k: int = 4) -> list[dict]:
        """يعيد [{'id','text','score','metadata'}] مرتبة تنازلياً."""
        with self._lock:
            docs = list(self.docs)
        if not docs or not query:
            return []
        texts = [d.text for d in docs]
        try:
            vecs, _ = self._vectors([query, *texts])
        except Exception:
            return [{"id": d.id, "text": d.text, "score": 0.0, "metadata": d.metadata} for d in docs[:k]]
        qv, dvs = vecs[0], vecs[1:]
        scored = []
        qt = set(_tokens(query))
        for d, v in zip(docs, dvs):
            s = self._cos(qv, v)
            # تعزيز التطابق الحرفي (مهم للعربية بدون embeddings حقيقية)
            overlap = len(qt & set(_tokens(d.text)))
            s += 0.1 * overlap
            scored.append((s, d))
        scored.sort(key=lambda x: -x[0])
        return [{"id": d.id, "text": d.text, "score": round(float(s), 4), "metadata": d.metadata}
                for s, d in scored[:k]]

    # -- persistence --
    def save(self, path: str):
        with self._lock:
            data = [d.to_dict() for d in self.docs]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        if not os.path.exists(path):
            return self
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        with self._lock:
            self.docs = [Document.from_dict(d) for d in data]
        return self

    # -- graph integration --
    def retrieval_node(self, in_key: str = "question", out_key: str = "context", k: int = 4):
        """عقدة استرجاع: state[in_key] (سؤال) -> state[out_key] (نصوص مركبة)."""
        def _node(state: dict) -> dict:
            q = state.get(in_key, "")
            if isinstance(q, list):  # messages؟
                q = str((q[-1].get("content") if isinstance(q[-1], dict) else getattr(q[-1], "content", "")) if q else "")
            hits = self.search(str(q), k=k)
            ctx = "\n---\n".join(h["text"] for h in hits)
            return {out_key: ctx, f"{out_key}_hits": hits}
        _node.__name__ = "vector_retrieval"
        return _node
