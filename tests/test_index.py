import hashlib
from datetime import date

import numpy as np

from monograph.index.local import LocalIndex
from monograph.types import DiaryEntry


class FakeEmbedder:
    dim = 64

    def __call__(self, texts):
        out = []
        for t in texts:
            v = np.zeros(self.dim, dtype="float32")
            for w in str(t).lower().split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
            n = np.linalg.norm(v)
            out.append(v / n if n else v)
        return np.asarray(out, dtype="float32")


def _entries():
    return [
        DiaryEntry("the garden was full of rain and mud", entry_date=date(2020, 6, 2)),
        DiaryEntry("a long hard day at the office, exhausted", entry_date=date(2020, 6, 9)),
        DiaryEntry("planting seedlings in the wet soil", entry_date=date(2020, 6, 16)),
        DiaryEntry("", entry_date=date(2020, 6, 20)),   # empty — should be dropped
    ]


def test_build_drops_empty_and_search_ranks_by_meaning():
    idx = LocalIndex(embedder=FakeEmbedder()).build(_entries())
    assert len(idx.entries) == 3                          # empty entry dropped
    results = idx.search("rain soil garden", k=3)
    assert results[0][0].text != "a long hard day at the office, exhausted"
    assert all(isinstance(e, DiaryEntry) for e, _ in results)


def test_save_load_roundtrip(tmp_path):
    base = str(tmp_path / "idx")
    LocalIndex(embedder=FakeEmbedder()).build(_entries()).save(base)
    reloaded = LocalIndex.load(base, embedder=FakeEmbedder())
    assert len(reloaded.entries) == 3
    assert reloaded.entries[0].entry_date == date(2020, 6, 2)
    # search works after reload
    assert reloaded.search("seedlings", k=1)[0][0].text.startswith("planting")
