"""Pluggable text model used by `arc` and `write`. Local or cloud — the caller
chooses, and cloud is opt-in via Config.local_only."""
from .base import LLM, NullLLM


def get_llm(name: str, cfg=None) -> LLM:
    """Resolve a backend name to an LLM. Register your own here."""
    name = (name or "null").lower()
    if name == "null":
        return NullLLM()
    if name == "ollama":
        from .ollama_backend import OllamaLLM     # local LLM, no data leaves the machine
        return OllamaLLM(cfg)
    if name == "openai":
        if cfg is not None:
            cfg.require_local_ok("openai llm")     # cloud call — must opt out of local_only
        from .openai_backend import OpenAILLM
        return OpenAILLM(cfg)
    raise ValueError(f"Unknown llm backend: {name!r}")


__all__ = ["LLM", "NullLLM", "get_llm"]
