"""Turn a body of writing into narration-ready units. Backend-agnostic: paragraph
and sentence splitting, punctuation normalization that TTS models prefer, and
reading a folder of chapter files. No assumptions about any particular book."""
from __future__ import annotations
import os
import re
import glob

_CLEAN = {"—": ", ", "–": ", ", "…": ". ",
          "’": "'", "‘": "'", "“": '"', "”": '"'}


def clean_text(t: str) -> str:
    """Normalize punctuation that TTS tends to mishandle (em/en dashes, ellipses,
    curly quotes) into plainer equivalents."""
    for a, b in _CLEAN.items():
        t = t.replace(a, b)
    t = re.sub(r"\s+-\s+", ", ", t)          # ' - ' used as a dash -> comma pause
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)   # no space before punctuation
    t = re.sub(r"\s+", " ", t)               # one utterance: collapse all whitespace
    return t.strip()


def split_document(text: str, title_max_chars: int = 80) -> tuple[str, list[str]]:
    """Split a chapter into (title, [paragraphs]), paragraphs separated by blank
    lines. A short first block is treated as the title; a long one means the
    chapter has no title and is all body."""
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    if not blocks:
        return "", []
    first = clean_text(blocks[0])
    if len(first) <= title_max_chars and len(blocks) > 1:
        return first, [clean_text(b) for b in blocks[1:]]
    return "", [clean_text(b) for b in blocks]


def split_chunks(text: str, limit: int = 220) -> list[str]:
    """Split one paragraph into sentence-sized pieces under `limit` characters,
    for backends (like XTTS) that need short inputs. Sentence boundaries first,
    then word-packing for any over-long sentence."""
    sents = re.split(r'(?<=[.!?…"”])\s+', text)
    out, cur = [], ""
    for s in sents:
        if len(cur) + len(s) + 1 <= limit:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                out.append(cur); cur = ""
            if len(s) <= limit:
                cur = s
            else:
                for w in s.split(" "):
                    if len(cur) + len(w) + 1 <= limit:
                        cur = (cur + " " + w).strip()
                    else:
                        if cur:
                            out.append(cur)
                        cur = w
    if cur:
        out.append(cur)
    return out


def read_documents(text_dir: str) -> list[tuple[str, str]]:
    """Read a folder of .txt chapter files, sorted by name. Returns (stem, text)."""
    files = sorted(glob.glob(os.path.join(text_dir, "*.txt")))
    docs = []
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        docs.append((stem, open(f, encoding="utf-8").read()))
    return docs


