"""An optional, user-supplied map that rewrites words a TTS model mispronounces
into phonetic respellings before synthesis. Format is one `word = sounds-like`
per line; '#' starts a comment. This is per-user *data*, never shipped with the
library — the author points Config.pronunciation_path at their own file."""
from __future__ import annotations
import os
import re


def load(path: str | None) -> list[tuple[str, str]]:
    """Load a pronunciation file into (word, respelling) pairs, longest-first so
    multi-word keys win. Missing path -> empty list (a no-op)."""
    pairs: list[tuple[str, str]] = []
    if not path or not os.path.exists(path):
        return pairs
    for ln in open(path, encoding="utf-8"):
        ln = ln.rstrip("\n")
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        if "=" in ln:
            a, b = ln.split("=", 1)
            a = a.strip()
            b = b.split("#", 1)[0].strip()   # allow trailing comment
            if a:
                pairs.append((a, b))
    pairs.sort(key=lambda p: -len(p[0]))
    return pairs


def apply(text: str, pairs: list[tuple[str, str]]) -> str:
    """Replace whole-word occurrences (case-sensitive) of each key."""
    for a, b in pairs:
        text = re.sub(r"\b" + re.escape(a) + r"\b", b, text)
    return text
