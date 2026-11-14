"""Central configuration. One Config object flows through every stage, so there
are no hardcoded paths, names, voices, or models anywhere in the library — the
caller supplies all of it here or in a TOML file.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None


@dataclass
class Config:
    # --- where things live (all caller-supplied; nothing personal baked in) ---
    diary_path: Optional[str] = None        # file or directory of diary entries
    work_dir: str = "monograph_out"         # where the library writes its artifacts

    # --- pluggable backend choices (names resolved by each submodule) ---
    index_backend: str = "local"            # see index.get_index
    llm_backend: str = "null"               # see llm.get_llm  ("null" = offline stub)
    narrate_backend: str = "f5"             # see narrate.get_narrator  ("f5" | "xtts")

    # --- narration inputs (per-user data, passed in, never shipped) ---
    voice_ref: Optional[str] = None         # short clean WAV of the author's voice
    voice_ref_text: Optional[str] = None    # EXACT transcript of voice_ref (F5 needs it)
    pronunciation_path: Optional[str] = None  # optional word -> respelling map
    cross_fade: float = 0.30                # seconds; higher = smoother seams
    chunk_char_limit: int = 220

    # --- writing ---
    style_profile: Optional[str] = None     # path to a sample of the author's prose

    # --- privacy: the load-bearing contract (see README) ---
    local_only: bool = True                 # backends must refuse network calls when True

    # --- free-form backend options ---
    options: dict = field(default_factory=dict)

    @classmethod
    def from_toml(cls, path: str | Path) -> "Config":
        if tomllib is None:  # pragma: no cover
            raise RuntimeError("tomllib requires Python 3.11+; pass fields directly instead.")
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}
        fields = {k: v for k, v in data.items() if k in known}
        fields["options"] = {k: v for k, v in data.items() if k not in known}
        return cls(**fields)

    def to_dict(self) -> dict:
        return asdict(self)

    def ensure_work_dir(self) -> Path:
        p = Path(self.work_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def require_local_ok(self, backend_name: str) -> None:
        """A backend that is about to make a network call calls this first."""
        if self.local_only:
            raise PermissionError(
                f"{backend_name} needs a network call, but Config.local_only is True. "
                f"Set local_only=False to opt in to sending data off this machine."
            )
