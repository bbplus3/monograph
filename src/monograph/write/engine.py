"""Draft memoir prose from real diary entries, in the author's own voice. Ported
from memoir.py. The two things that were hardcoded there are now inputs:

  - the author's *voice* — sampled from their own older entries and described in a
    StyleProfile (StyleProfile.notes carries the characterization that used to be
    a hardcoded "dry and sarcastic ..." paragraph);
  - the *age context* — a `people` map of name -> birthdate, so the model never
    gets a person's developmental stage wrong at a given entry's date. (Your kids'
    birthdates live in examples/lenses_parenting.py, not in the library.)

Retrieval reuses the index (theme search) and the entry list (date ranges). The
local LLM is injected via the LLM interface, so Ollama is one choice, not baked in.

write_passage() is implemented. write_memoir() — turning a chosen spine into an
ordered set of chapters — waits on the scene/chapter segmentation from
memoir_scenes.py.
"""
from __future__ import annotations
from datetime import date

from ..types import Chapter, StyleProfile, Spine, DiaryEntry
from ..llm import LLM

_STRICT_RULES = (
    "STRICT RULES:\n"
    "- Use ONLY what is in the journal entries below. Do NOT invent events, people, "
    "places, or feelings that aren't there.\n"
    "- Only use details relevant to the theme. If an entry mentions the theme only in "
    "passing, or contains unrelated details (errands, chores, other people, workouts), "
    "IGNORE those parts.\n"
    "- Do NOT inflate mundane logged activities (going to the store, laundry, a workout) "
    "into meaningful or tender moments. Routine is routine; find the real emotional "
    "content, which is often understated or hidden behind a joke — not in the errands."
)


class MemoirEngine:
    def __init__(self, llm: LLM, style: StyleProfile | None = None, index=None,
                 entries: list[DiaryEntry] | None = None, people: dict | None = None,
                 voice_window: tuple[str, str] | None = None, temperature: float = 0.4):
        self.llm = llm
        self.style = style or StyleProfile()
        self.index = index
        self.entries = entries if entries is not None else (list(index.entries) if index else [])
        self.people = people or {}                 # name -> datetime.date (birthdate)
        self.voice_window = voice_window           # (iso_lo, iso_hi) to sample voice from
        self.temperature = temperature

    # --- age context ---------------------------------------------------------
    def _age_at(self, birth: date, d: date) -> int:
        return d.year - birth.year - ((d.month, d.day) < (birth.month, birth.day))

    def _ages_line(self, d: date | None) -> str:
        if not self.people or d is None:
            return ""
        return "(" + ", ".join(f"{n} was {self._age_at(b, d)}" for n, b in self.people.items()) + ")"

    # --- voice --------------------------------------------------------------
    def _voice_samples(self, n: int = 6, min_len: int = 200) -> str:
        if self.style.sample:                      # an explicit sample wins
            return self.style.sample
        pool = [e for e in self.entries if e.entry_date and len(e.text or "") > min_len]
        if self.voice_window:
            lo, hi = (date.fromisoformat(self.voice_window[0]),
                      date.fromisoformat(self.voice_window[1]))
            windowed = [e for e in pool if lo <= e.entry_date <= hi]
            pool = windowed or pool
        pool.sort(key=lambda e: e.entry_date)
        if len(pool) > n:                          # spread across the window
            step = len(pool) / n
            pool = [pool[int(i * step)] for i in range(n)]
        return "\n\n".join(f"[{e.entry_date}]\n{e.text}" for e in pool)

    # --- retrieval ----------------------------------------------------------
    def _by_theme(self, theme: str, k: int) -> list[DiaryEntry]:
        if self.index is None:
            raise RuntimeError("Theme retrieval needs an index; pass index= to MemoirEngine.")
        return [e for e, _ in self.index.search(theme, k=k)]

    def _by_dates(self, date_from: str, date_to: str, max_entries: int = 40) -> list[DiaryEntry]:
        lo, hi = date.fromisoformat(date_from), date.fromisoformat(date_to)
        picked = sorted((e for e in self.entries if e.entry_date and lo <= e.entry_date <= hi),
                        key=lambda e: e.entry_date)
        if len(picked) > max_entries:
            step = len(picked) / max_entries
            picked = [picked[int(i * step)] for i in range(max_entries)]
        return picked

    # --- prompt + generation ------------------------------------------------
    def _weekday(self, e: DiaryEntry) -> str:
        return e.meta.get("weekday") if e.meta.get("weekday") else (
            e.entry_date.strftime("%A") if e.entry_date else "")

    def _build_prompt(self, entries: list[DiaryEntry], instruction: str) -> str:
        samples = self._voice_samples()
        voice_notes = (self.style.notes + "\n\n") if self.style.notes else ""
        ages_rule = ("\n- Ages are given in brackets with each entry — respect them. "
                     "The entries span years, so people were different ages at different "
                     "dates. Never describe someone in a way that contradicts their age "
                     "at that entry's date.") if self.people else ""
        blocks = ""
        for e in entries:
            tag = f"{e.entry_date} - {self._weekday(e)} {self._ages_line(e.entry_date)}".strip()
            blocks += f"\n[{tag}]\n{e.text}\n"
        return (
            "You are helping an author turn their own real journal entries into a "
            "reflective memoir passage, in first person, in THEIR voice.\n\n"
            "THE AUTHOR'S VOICE — study these real entries to capture how they actually write:\n"
            f"{samples}\n\n"
            f"{voice_notes}"
            "Keep their voice, but give real events their real weight: match how they "
            "sound, without letting the tone erase the genuine impact of significant "
            "events. Serious things can be serious, even told in their voice.\n\n"
            f"{_STRICT_RULES}{ages_rule}\n\n"
            f"TASK: {instruction}\n\n"
            f"JOURNAL ENTRIES (each tagged with its date"
            f"{', weekday, and ages then' if self.people else ' and weekday'}):\n{blocks}\n"
            "Write the memoir passage now — their voice, faithful to the entries, real "
            "events given real weight:"
        )

    def write_passage(self, theme: str | None = None, *, date_from: str | None = None,
                      date_to: str | None = None, k: int = 6) -> Chapter:
        """Draft one passage from a theme (semantic retrieval) or a date range."""
        if date_from and date_to:
            entries = self._by_dates(date_from, date_to)
            instruction = (f"Write a reflective memoir passage covering {date_from} to "
                           f"{date_to}, drawing on these entries.")
            title = f"{date_from} — {date_to}"
        elif theme:
            entries = self._by_theme(theme, k)
            instruction = (f"Write a reflective memoir passage on the theme: '{theme}', "
                           f"drawing on these related entries.")
            title = theme
        else:
            raise ValueError("Give a theme or a date_from/date_to range.")
        if not entries:
            raise RuntimeError("No entries found for that theme or range.")
        text = self.llm.complete(self._build_prompt(entries, instruction),
                                 temperature=self.temperature, max_tokens=2048)
        return Chapter(title=title, text=text,
                       meta={"source_dates": [str(e.entry_date) for e in entries]})

    def write_chapter(self, spine: Spine, section: str, order: int = 0, k: int = 6) -> Chapter:
        """Draft one chapter for `section` (an aspect name or theme the spine draws
        on), retrieving the entries that best match it."""
        ch = self.write_passage(theme=section, k=k)
        ch.order = order
        ch.title = section
        return ch

    def write_memoir(self, spine: Spine, sections: list[str] | None = None,
                     k: int = 6) -> list[Chapter]:
        """Draft a whole memoir from a chosen spine: one chapter per section, in the
        spine's order. Sections default to the aspects the spine draws on (from
        propose_spines); pass your own list to override. Each chapter is drafted by
        write_passage from the entries that best match that section, so it needs an
        index. This is a sensible default assembly — reorder or merge to taste."""
        if sections is None:
            sections = spine.meta.get("draws_on", {}).get("aspects") or []
        if not sections:
            raise ValueError("No sections to write: the spine lists no aspects; pass "
                             "sections=[...] explicitly.")
        return [self.write_chapter(spine, s, order=i, k=k) for i, s in enumerate(sections)]
