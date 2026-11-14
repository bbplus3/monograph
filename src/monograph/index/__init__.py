"""index — build a searchable index over diary entries and query it, plus the
shared local embedder used by both retrieval and the arc stage."""
from .base import Index


def get_index(name: str, cfg=None) -> Index:
    """Resolve a backend name to an Index. Register your own here."""
    name = (name or "local").lower()
    if name == "local":
        from .local import LocalIndex
        return LocalIndex(cfg)
    raise ValueError(f"Unknown index backend: {name!r}")


def get_embedder(name: str = "local", cfg=None):
    """Return an embedder: a callable embed(list[str]) -> np.ndarray."""
    name = (name or "local").lower()
    if name == "local":
        from .embedding import LocalEmbedder
        model = (cfg.options.get("embed_model") if cfg and cfg.options else None) or "all-MiniLM-L6-v2"
        return LocalEmbedder(model)
    raise ValueError(f"Unknown embedder: {name!r}")


__all__ = ["Index", "get_index", "get_embedder"]
