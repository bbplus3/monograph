"""The one method every text model must provide. Keep this interface tiny so a
local model, a cloud API, or a fake for tests are all interchangeable."""
from __future__ import annotations
import abc
from typing import Optional


class LLM(abc.ABC):
    @abc.abstractmethod
    def complete(self, prompt: str, *, system: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: int = 1024,
                 json_mode: bool = False) -> str:
        """Return the model's completion of `prompt`. When `json_mode` is set, the
        backend should coax the model into returning JSON (callers still parse
        defensively)."""
        raise NotImplementedError


class NullLLM(LLM):
    """Offline placeholder. Lets the pipeline import and run structurally without a
    model wired up; any stage that truly needs generation will raise clearly."""

    def complete(self, prompt: str, *, system=None, temperature=0.7,
                 max_tokens=1024, json_mode=False) -> str:
        raise RuntimeError(
            "No LLM configured (llm_backend='null'). Set Config.llm_backend to a real "
            "backend (e.g. 'openai', or your own local model) to use arc/ and write/."
        )
