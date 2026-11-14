"""narrate — turn finished text into audio in the author's own voice.

    from monograph import Config
    from monograph.narrate import get_narrator
    n = get_narrator("f5", Config(voice_ref="voice.wav", voice_ref_text="..."))
    n.narrate_dir("chapters/", "audio/")

Backends live in .backends and are imported lazily.
"""
from .base import Narrator
from . import chapterize, pronunciation, voice, audio


def get_narrator(name: str, cfg) -> Narrator:
    """Resolve a backend name ('f5' | 'xtts') to a ready Narrator."""
    name = (name or cfg.narrate_backend or "f5").lower()
    if name == "f5":
        from .backends.f5 import F5Narrator
        return F5Narrator(cfg)
    if name in ("xtts", "coqui"):
        from .backends.xtts import XTTSNarrator
        return XTTSNarrator(cfg)
    raise ValueError(f"Unknown narrate backend: {name!r} (expected 'f5' or 'xtts')")


__all__ = ["Narrator", "get_narrator", "chapterize", "pronunciation", "voice", "audio"]
