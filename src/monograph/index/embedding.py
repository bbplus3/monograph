"""Local sentence embeddings — shared by `index` (retrieval) and `arc` (analysis).
Runs entirely on-machine (all-MiniLM-L6-v2 by default); nothing is sent anywhere.

An embedder is just a callable `embed(list[str]) -> np.ndarray` of L2-normalized
float32 vectors, so anything with that shape can be swapped in. `embed_entries`
adds an on-disk cache keyed by entry count (ported from the .npy caching in your
memoir scripts).
"""
from __future__ import annotations
import os
import numpy as np


class LocalEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _lazy(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:  # pragma: no cover
                raise ImportError("Embeddings need: pip install 'monograph[index]'") from e
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def __call__(self, texts) -> np.ndarray:
        m = self._lazy()
        return m.encode(list(texts), batch_size=64, normalize_embeddings=True,
                        show_progress_bar=False).astype("float32")

    def embed_entries(self, texts, cache_path: str | None = None,
                      progress: bool = True) -> np.ndarray:
        """Embed a whole corpus, using/refreshing an .npy cache if given."""
        texts = list(texts)
        if cache_path and os.path.exists(cache_path):
            cached = np.load(cache_path)
            if cached.shape[0] == len(texts):
                return cached
        m = self._lazy()
        E = m.encode(texts, batch_size=64, normalize_embeddings=True,
                     show_progress_bar=progress).astype("float32")
        if cache_path:
            np.save(cache_path, E)
        return E
