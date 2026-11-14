from datetime import date

from monograph.types import DiaryEntry, StyleProfile
from monograph.llm.base import LLM
from monograph.write.engine import MemoirEngine


class CapturingLLM(LLM):
    """Records the prompt it was given and returns a fixed passage."""
    def __init__(self):
        self.prompt = None
    def complete(self, prompt, *, system=None, temperature=0.7, max_tokens=1024, json_mode=False):
        self.prompt = prompt
        return "A drafted passage."


def _entries():
    return [
        DiaryEntry("Long reflective entry about patience and bedtime. " * 10,
                   entry_date=date(2019, 3, 1), meta={"weekday": "Friday"}),
        DiaryEntry("Short log: Costco, laundry.", entry_date=date(2020, 7, 4),
                   meta={"weekday": "Saturday"}),
        DiaryEntry("Another longer entry about losing my temper and the guilt after. " * 10,
                   entry_date=date(2019, 9, 10), meta={"weekday": "Tuesday"}),
    ]


def test_write_passage_by_dates_builds_chapter_and_prompt():
    llm = CapturingLLM()
    people = {"Kid": date(2010, 1, 1)}
    eng = MemoirEngine(llm, style=StyleProfile(notes="dry and wry"), entries=_entries(),
                       people=people, voice_window=("2018-01-01", "2021-12-31"))
    ch = eng.write_passage(date_from="2019-01-01", date_to="2019-12-31")
    assert ch.text == "A drafted passage."
    assert ch.meta["source_dates"]                      # entries were selected
    # prompt carries voice samples, the style notes, strict rules, and age context
    assert "dry and wry" in llm.prompt
    assert "STRICT RULES" in llm.prompt
    assert "Kid was 9" in llm.prompt                    # age computed at 2019 entry


def test_age_computation():
    eng = MemoirEngine(CapturingLLM(), people={"A": date(2000, 6, 15)})
    assert eng._ages_line(date(2020, 6, 14)) == "(A was 19)"
    assert eng._ages_line(date(2020, 6, 15)) == "(A was 20)"


def test_theme_without_index_errors():
    eng = MemoirEngine(CapturingLLM(), entries=_entries())
    try:
        eng.write_passage("patience")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_write_memoir_one_chapter_per_spine_aspect():
    import hashlib
    import numpy as np
    from monograph.index.local import LocalIndex
    from monograph.types import Spine

    class FakeEmbedder:
        dim = 32
        def __call__(self, texts):
            out = []
            for t in texts:
                v = np.zeros(self.dim, dtype="float32")
                for w in str(t).lower().split():
                    v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
                n = np.linalg.norm(v)
                out.append(v / n if n else v)
            return np.asarray(out, dtype="float32")

    idx = LocalIndex(embedder=FakeEmbedder()).build(_entries())
    eng = MemoirEngine(CapturingLLM(), index=idx)
    spine = Spine(title="T", premise="Can she?", arc="arc",
                  meta={"draws_on": {"aspects": ["temper", "repair"]}})
    chapters = eng.write_memoir(spine)
    assert [c.title for c in chapters] == ["temper", "repair"]
    assert [c.order for c in chapters] == [0, 1]
    assert all(c.text == "A drafted passage." for c in chapters)
