import hashlib
import json
from datetime import date

import numpy as np
import pytest

pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from monograph.types import DiaryEntry, Lenses, Evidence, Scene
from monograph.llm.base import LLM
from monograph.arc import (build_evidence, recon_report, propose_spines,
                           scenes_by_aspect, scenes_report)


class FakeEmbedder:
    """Deterministic bag-of-words embeddings — no model download needed."""
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


class FakeLLM(LLM):
    def complete(self, prompt, *, system=None, temperature=0.7, max_tokens=1024, json_mode=False):
        return json.dumps({"spines": [{
            "id": "s1", "title": "Learning to Stay", "throughline": "Can she break the pattern?",
            "arc": "from reacting to repairing", "draws_on": {"aspects": ["joy"], "people": [], "period": "2020-2021"},
            "scene_ids": [0, 1], "strengths": "clear change", "missing_to_author": "childhood"}]})


def _entries():
    words = ["joy with my child today", "a fight and real conflict again",
             "grief and loss this week", "work was steady and good",
             "my mother and my childhood", "a tender happy moment of love"]
    out = []
    for i in range(24):
        out.append(DiaryEntry(text=words[i % len(words)] + f" entry {i}",
                              entry_date=date(2020 + i // 12, (i % 12) + 1, 1)))
    return out


def test_build_evidence_shape():
    ev = build_evidence(_entries(), FakeEmbedder(), lenses=Lenses.default())
    assert isinstance(ev, Evidence)
    assert ev.n_valid > 0
    assert ev.aspects and all("share_pct" in a for a in ev.aspects)
    assert ev.timeline and ev.snippets


def test_propose_spines_parses_into_spine_objects():
    ev = build_evidence(_entries(), FakeEmbedder(), lenses=Lenses.default())
    spines = propose_spines(ev, FakeLLM())
    assert len(spines) == 1
    s = spines[0]
    assert s.title == "Learning to Stay"
    assert s.premise.endswith("?")                      # throughline is a question
    assert s.meta["missing_to_author"] == "childhood"


def test_recon_report_is_markdown():
    report = recon_report(_entries(), FakeEmbedder(), lenses=Lenses.default())
    assert report.startswith("# Memoir reconnaissance")
    assert "Aspects across the spine" in report


def test_scenes_by_aspect_returns_ranked_scenes():
    by = scenes_by_aspect(_entries(), FakeEmbedder(), lenses=Lenses.default(), per_aspect=3)
    assert set(by) == set(Lenses.default().aspects)
    some = next(v for v in by.values() if v)
    assert all(isinstance(s, Scene) for s in some)
    # ranked by score, descending
    scores = [s.meta["score"] for s in some]
    assert scores == sorted(scores, reverse=True)


def test_scenes_report_is_markdown():
    report = scenes_report(_entries(), FakeEmbedder(), lenses=Lenses.default(), per_aspect=3)
    assert report.startswith("# Memoir scenes")
    assert "Aspect mix" in report
