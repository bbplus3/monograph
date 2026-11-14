"""ingest — parse a diary (any format) into structured DiaryEntry objects."""
from .parser import parse_diary

__all__ = ["parse_diary"]
