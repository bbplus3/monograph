"""Local, on-machine retrieval — the privacy-preserving default. Ported from
build_index.py (embed + store) and search_diary.py (query).

One deliberate change from the originals: FAISS `IndexFlatIP` is an *exact* inner-
product search, which on L2-normalized vectors is identical, number for number, to
`embeddings @ query`. For a diary (thousands of entries) that numpy dot product is
instant, so this uses it directly and drops the faiss dependency — which is also
one of the harder packages to install on Windows. Same results, nothing to build.

On-disk format is portable and light: `<base>.npy` (the embedding matrix) and
`<base>.meta.json` (the aligned entries). No pickle, no faiss needed to reload.
"""
from __future__ import annotations
import json
from datetime import date
from pathlib import Path

import numpy as np

from .base import Index
from ..types import DiaryEntry


class LocalIndex(Index):
    def __init__(self, cfg=None, embedder=None):
        self.cfg = cfg
        self._embedder = embedder          # a callable list[str] -> np.ndarray, or None
        self.entries: list[DiaryEntry] = []
        self.E: np.ndarray | None = None

    # shared embedder (lazily created so injecting a fake needs no model)
    def _embed(self, texts):
        if self._embedder is None:
            from .embedding import LocalEmbedder
            model = (self.cfg.options.get("embed_model")
                     if self.cfg and getattr(self.cfg, "options", None) else None) or "all-MiniLM-L6-v2"
            self._embedder = LocalEmbedder(model)
        return self._embedder(texts)

    # --- Index interface -----------------------------------------------------
    def build(self, entries: list[DiaryEntry]) -> "LocalIndex":
        self.entries = [e for e in entries if (e.text or "").strip()]
        self.E = self._embed([e.text for e in self.entries])
        return self

    def search(self, query: str, k: int = 8) -> list[tuple[DiaryEntry, float]]:
        if self.E is None or not len(self.entries):
            raise RuntimeError("Index is empty — call build(entries) or load(path) first.")
        q = self._embed([query])[0]
        scores = self.E @ q                 # cosine (vectors are normalized)
        order = np.argsort(scores)[::-1][:k]
        return [(self.entries[i], float(scores[i])) for i in order]

    def save(self, path: str) -> None:
        base = str(path)
        if self.E is None:
            raise RuntimeError("Nothing to save — build the index first.")
        np.save(base + ".npy", self.E)
        meta = [{"date": e.entry_date.isoformat() if e.entry_date else None,
                 "text": e.text, "source": e.source, "meta": e.meta}
                for e in self.entries]
        Path(base + ".meta.json").write_text(json.dumps(meta, ensure_ascii=False),
                                             encoding="utf-8")

    @classmethod
    def load(cls, path: str, cfg=None, embedder=None) -> "LocalIndex":
        base = str(path)
        inst = cls(cfg, embedder)
        inst.E = np.load(base + ".npy")
        meta = json.loads(Path(base + ".meta.json").read_text(encoding="utf-8"))
        inst.entries = [
            DiaryEntry(text=m["text"],
                       entry_date=date.fromisoformat(m["date"]) if m.get("date") else None,
                       source=m.get("source"), meta=m.get("meta") or {})
            for m in meta
        ]
        return inst
