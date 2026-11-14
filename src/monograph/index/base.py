"""The retrieval interface. Any store (local embeddings, FAISS, Chroma, sqlite)
implements this so the rest of the library doesn't care which one is used."""
from __future__ import annotations
import abc
from ..types import DiaryEntry


class Index(abc.ABC):
    @abc.abstractmethod
    def build(self, entries: list[DiaryEntry]) -> None:
        """Index the entries (embed + store)."""

    @abc.abstractmethod
    def search(self, query: str, k: int = 8) -> list[tuple[DiaryEntry, float]]:
        """Return the k most relevant entries with similarity scores."""

    @abc.abstractmethod
    def save(self, path: str) -> None: ...

    @classmethod
    @abc.abstractmethod
    def load(cls, path: str) -> "Index": ...
