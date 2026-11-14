"""Style is an input, never a built-in. The author supplies a sample of their own
prose (or notes), and the write engine drafts to match it."""
from __future__ import annotations
from pathlib import Path
from ..types import StyleProfile


def load_style(path: str | None, name: str = "author", notes: str = "") -> StyleProfile:
    """Load a style profile from a text sample of the author's own writing."""
    sample = ""
    if path:
        p = Path(path)
        if p.exists():
            sample = p.read_text(encoding="utf-8")
    return StyleProfile(name=name, sample=sample, notes=notes)
