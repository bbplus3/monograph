"""Shared data structures passed between stages. Keeping these here (not inside
any one submodule) is what lets the stages stay decoupled: ingest produces
DiaryEntry, arc consumes them and produces Spine/Scene, write consumes those and
produces Chapter, narrate consumes Chapter.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass
class DiaryEntry:
    """One dated unit of a diary."""
    text: str
    entry_date: Optional[date] = None
    source: Optional[str] = None            # file/path it came from
    meta: dict = field(default_factory=dict)


@dataclass
class Scene:
    """A concrete moment distilled from one or more entries — the raw material of
    memoir (a place, a time, people, something that happened)."""
    title: str
    summary: str
    entries: list[DiaryEntry] = field(default_factory=list)
    when: Optional[str] = None
    people: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


@dataclass
class Spine:
    """A candidate narrative arc through the diary — the backbone a memoir could
    hang on (a question, a change, a through-line), plus the scenes that serve it."""
    title: str
    premise: str
    arc: str                                # one-paragraph shape of the story
    scenes: list[Scene] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


@dataclass
class Chapter:
    """A written unit ready to read or narrate."""
    title: str
    text: str
    order: int = 0
    meta: dict = field(default_factory=dict)


@dataclass
class StyleProfile:
    """How the author writes — supplied by the author, applied by the write stage."""
    name: str = "author"
    sample: str = ""                        # a chunk of their own prose
    notes: str = ""                         # optional guidance ("spare, present tense")
    meta: dict = field(default_factory=dict)


@dataclass
class VoiceRef:
    """A reference voice for narration."""
    wav_path: str
    ref_text: str                           # EXACT transcript of wav_path
    meta: dict = field(default_factory=dict)


@dataclass
class Lenses:
    """The *only* domain-specific input to the arc stage. A writer tunes these to
    their memoir's subject; nothing about any one book is baked into the library.

    - aspects: name -> a phrase describing that emotional/thematic lens. Each entry
      is scored against every aspect; its dominant aspect is the best match.
    - emo_pos / emo_neg: phrases whose mean similarity gives an entry's valence.
    - strands: optional named seed-groups for a compare/contrast memoir
      (e.g. {"present": [...], "past": [...]}). Empty = single-strand.
    - name_stop: capitalized tokens to ignore when guessing recurring people.
    """
    aspects: dict
    emo_pos: list
    emo_neg: list
    strands: dict = field(default_factory=dict)
    name_stop: set = field(default_factory=set)

    @classmethod
    def default(cls) -> "Lenses":
        """A generic memoir lens set. Override it for your subject (see
        examples/lenses_parenting.py for a worked, domain-specific example)."""
        return cls(
            aspects={
                "turning_point": "a moment everything changed, a decision or event that redirected my life",
                "relationship": "a close relationship, a bond or connection with another person",
                "conflict": "conflict, tension, a fight or a falling out",
                "loss": "loss, grief, something or someone gone",
                "growth": "learning, changing, becoming who I am, personal growth",
                "struggle": "a hardship, a struggle, something difficult I was going through",
                "joy": "joy, delight, a happy or tender moment",
                "place": "a place that mattered, where something happened",
                "work": "work, vocation, what I do and how I feel about it",
                "belief": "my beliefs, my values, my faith, what steadies or guides me",
            },
            emo_pos=["joy and happiness", "love and tenderness", "pride",
                     "gratitude and thankfulness", "calm and peace"],
            emo_neg=["anger and frustration", "fear, anxiety and worry",
                     "sadness and grief", "guilt and shame", "exhaustion and being overwhelmed"],
            strands={},
            name_stop={
                "Today", "Tomorrow", "Yesterday", "Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday", "Sunday", "January", "February",
                "March", "April", "May", "June", "July", "August", "September",
                "October", "November", "December", "I", "We", "My", "The", "Then",
                "It", "He", "She", "They", "That", "This", "And", "But", "So",
            },
        )


@dataclass
class Evidence:
    """Compact, structured signals distilled from a diary — the only thing handed
    to the LLM when proposing spines (never the whole journal). Ported from the
    'evidence pack' in spine_proposer.py."""
    n_entries: int
    n_valid: int
    aspects: list = field(default_factory=list)       # [{name, share_pct, trend, top_ids}]
    timeline: dict = field(default_factory=dict)      # year -> count
    turning_points: list = field(default_factory=list)
    entities: list = field(default_factory=list)      # [{name, count}]
    snippets: dict = field(default_factory=dict)       # id -> "date — text"
