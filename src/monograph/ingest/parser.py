"""Parse a diary into DiaryEntry objects — one front door for several formats.

Ported from diary_parser.py (the typo-tolerant .docx header splitter) plus the
load_entries/_coerce logic that the arc scripts each carried, so loading lives in
one place now. Everything runs locally; nothing leaves the machine.

Supported inputs (auto-detected by extension, or a directory of them):
  .docx                 dated-header diary ("Weekday M/D/YY", typo-tolerant)
  .txt / .md            same dated-header splitter over plain text
  .json / .jsonl        records with date/text fields (various key names)
  .csv / .tsv           a table with date/text columns
  .pkl / .pickle        a pickled list/dict/DataFrame of entries

parse_diary(path) -> list[DiaryEntry]. Pass return_flagged=True to also get the
lines that looked header-ish but couldn't be parsed (for manual review).
"""
from __future__ import annotations
import csv
import json
import pickle
import re
from datetime import datetime, date as date_cls
from pathlib import Path

from ..types import DiaryEntry

# Header = optional leading word (weekday, possibly misspelled) + a date. Only the
# date matters. Date: M/D/YY or M/D/YYYY, and we salvage a stray 3-digit year.
HEADER_RE = re.compile(r"^\s*(?:[A-Za-z]+\s+)?(\d{1,2})/(\d{1,2})/(\d{2,4})\s*$")
DATE_TOKEN = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}")
_RECORD_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv", ".pkl", ".pickle"}
_TEXT_SUFFIXES = {".txt", ".md"}


# ----------------------------------------------------------------- helpers ---
def _resolve_year(y) -> int:
    y = int(y)
    if y < 100:                    # 2-digit: 26 -> 2026, 18 -> 2018
        return 2000 + y
    if 100 <= y < 1000:            # malformed 3-digit like 219 -> 2019
        return 2000 + (y % 100)
    return y


def _coerce_date(v):
    if v in (None, ""):
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date_cls):
        return v
    s = str(v).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%Y/%m/%d",
                "%B %d, %Y", "%b %d, %Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(s).date()
    except Exception:
        return None


def _first(d, *keys):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


# --------------------------------------------------- dated-header splitter ---
def _finalize(entry, source) -> DiaryEntry:
    return DiaryEntry(text="\n".join(entry["_lines"]).strip(),
                      entry_date=entry["date"].date(),
                      source=source, meta={"weekday": entry["weekday"]})


def _entries_from_lines(lines, source):
    """Split a sequence of text lines into entries on dated headers."""
    entries, flagged, current = [], [], None
    for raw in lines:
        line = (raw or "").strip()
        if not line:
            if current is not None:
                current["_lines"].append("")
            continue
        m = HEADER_RE.match(line)
        if m:
            month, day, year_raw = m.groups()
            try:
                dt = datetime(_resolve_year(year_raw), int(month), int(day))
                if current is not None:
                    entries.append(_finalize(current, source))
                current = {"date": dt, "weekday": dt.strftime("%A"), "_lines": []}
            except ValueError:                      # header-shaped but impossible date
                flagged.append(line)
                if current is not None:
                    current["_lines"].append(line)
        else:
            if DATE_TOKEN.search(line) and len(line) < 40 and current is None:
                flagged.append(line)
            elif current is not None:
                current["_lines"].append(line)
            else:
                flagged.append(f"[pre-first-entry] {line[:60]}")
    if current is not None:
        entries.append(_finalize(current, source))
    return entries, flagged


def _parse_docx(path):
    try:
        from docx import Document
    except ImportError as e:
        raise ImportError("Reading .docx needs python-docx: "
                          "pip install 'monograph[ingest]'") from e
    doc = Document(str(path))
    return _entries_from_lines([p.text for p in doc.paragraphs], str(path))


def _parse_text(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    entries, flagged = _entries_from_lines(lines, str(path))
    if not entries:                                 # no headers: treat file as one entry
        d = _date_from_name(Path(path).name)
        text = "\n".join(lines).strip()
        if text:
            entries = [DiaryEntry(text=text, entry_date=d, source=str(path))]
            flagged = []
    return entries, flagged


def _date_from_name(name):
    m = re.search(r"(\d{4})[-_/.](\d{1,2})[-_/.](\d{1,2})", name)
    if m:
        try:
            return date_cls(int(m[1]), int(m[2]), int(m[3]))
        except ValueError:
            pass
    m = DATE_TOKEN.search(name)
    if m:
        try:
            mo, da, yr = re.split(r"/", m.group(0))
            return date_cls(_resolve_year(yr), int(mo), int(da))
        except ValueError:
            pass
    return None


# ------------------------------------------------------- record formats ---
def _coerce_records(obj):
    """Turn whatever was loaded (list/dict/tuple/DataFrame) into flat records."""
    if hasattr(obj, "to_dict"):                      # pandas DataFrame, without importing pandas
        try:
            return obj.to_dict("records")
        except Exception:
            pass
    if isinstance(obj, tuple):
        obj = list(obj)
    if isinstance(obj, dict):
        if isinstance(obj.get("entries"), list):
            return obj["entries"]
        vals = list(obj.values())
        if vals and all(isinstance(v, str) for v in vals[:25]):     # {date: text}
            return [{"date": k, "text": v} for k, v in obj.items()]
        for v in vals:
            if isinstance(v, list):
                return v
        return []
    if isinstance(obj, list):
        if obj and len(obj) <= 3 and isinstance(obj[0], list) and len(obj[0]) > 10:
            return obj[0]                            # unwrap (entries, embeddings, ...)
        return obj
    return []


def _records_to_entries(raw, source):
    out = []
    for r in raw:
        if isinstance(r, dict):
            text = _first(r, "text", "content", "body", "entry", "Entry", "Text")
            d = _first(r, "date", "Date", "entry_date", "dt", "day", "timestamp")
        elif isinstance(r, (list, tuple)) and len(r) >= 2:
            d, text = r[0], r[1]
        elif isinstance(r, str):
            d, text = None, r
        else:
            continue
        if text:
            out.append(DiaryEntry(text=str(text).strip(),
                                  entry_date=_coerce_date(d), source=source))
    return out


def _parse_records(path):
    p = Path(path)
    suf = p.suffix.lower()
    if suf in (".pkl", ".pickle"):
        with open(p, "rb") as f:
            raw = _coerce_records(pickle.load(f))
    elif suf == ".jsonl":
        raw = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    elif suf == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        raw = _coerce_records(data) if isinstance(data, dict) else data
    elif suf in (".csv", ".tsv"):
        with open(p, newline="", encoding="utf-8") as f:
            raw = list(csv.DictReader(f, delimiter="\t" if suf == ".tsv" else ","))
    else:
        raise ValueError(f"Not a record format: {suf}")
    return _records_to_entries(raw, str(p)), []


# --------------------------------------------------------------- dispatch ---
def _parse_one(path):
    suf = Path(path).suffix.lower()
    if suf in (".docx", ".doc"):
        return _parse_docx(path)
    if suf in _RECORD_SUFFIXES:
        return _parse_records(path)
    if suf in _TEXT_SUFFIXES:
        return _parse_text(path)
    raise ValueError(f"Don't know how to read {suf}; use .docx/.txt/.md/.json/.jsonl/.csv/.pkl")


def parse_diary(path: str, *, return_flagged: bool = False, **opts):
    """Parse a diary file (or a directory of them) into DiaryEntry objects."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    entries, flagged = [], []
    if p.is_dir():
        files = sorted(f for f in p.rglob("*")
                       if f.suffix.lower() in _RECORD_SUFFIXES | _TEXT_SUFFIXES | {".docx"})
        for f in files:
            e, fl = _parse_one(f)
            entries += e
            flagged += fl
    else:
        entries, flagged = _parse_one(p)

    entries.sort(key=lambda e: (e.entry_date is None, e.entry_date or date_cls.min))
    if return_flagged:
        return entries, flagged
    if flagged:
        print(f"[ingest] parsed {len(entries)} entries; {len(flagged)} line(s) "
              f"couldn't be parsed and were skipped. "
              f"Call parse_diary(..., return_flagged=True) to inspect them.")
    return entries
