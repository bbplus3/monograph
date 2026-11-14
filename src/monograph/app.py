"""Diary Companion — a local, private Streamlit app for exploring a journal:
find forgotten entries, follow a thread across the years, see your story freshly.
Ported from diary_app.py. Retrieval goes through monograph's LocalIndex (no faiss
/ pickle), and the family age context is optional config, not hardcoded.

Launch it with:   monograph app

It reads an index built by `monograph index` (default base name "diary_index").
Age labels appear if a monograph.toml with a [people] table sits in the folder:

    [people]
    Cash = "2007-01-10"
    Camille = "2008-01-31"
"""
from __future__ import annotations
import os
from datetime import date
from pathlib import Path

import streamlit as st


@st.cache_resource
def _load_index(base: str):
    from monograph.index.local import LocalIndex
    return LocalIndex.load(base)


def _load_people() -> dict:
    try:
        import tomllib
        p = Path("monograph.toml")
        if p.exists():
            data = tomllib.loads(p.read_text(encoding="utf-8"))
            return {k: date.fromisoformat(v) for k, v in data.get("people", {}).items()}
    except Exception:
        pass
    return {}


def _age_at(birth: date, d: date) -> int:
    return d.year - birth.year - ((d.month, d.day) < (birth.month, birth.day))


def run():
    st.set_page_config(page_title="Diary Companion", page_icon="📖", layout="wide")
    st.title("📖 Diary Companion")
    st.caption("A private, local space to find, revisit, and reflect on your own "
               "entries. Nothing leaves this machine.")

    base = st.sidebar.text_input("Index path",
                                 value=os.environ.get("MONOGRAPH_INDEX", "diary_index"))
    try:
        idx = _load_index(base)
    except Exception as e:
        st.error(f"Couldn't load an index at '{base}'. Build one first:\n\n"
                 f"`monograph index --diary <your diary> --out {base}`\n\n({e})")
        st.stop()

    people = _load_people()
    dated = [e for e in idx.entries if e.entry_date]
    span = (f"{min(e.entry_date for e in dated)} → {max(e.entry_date for e in dated)}"
            if dated else "no dates")
    st.sidebar.success(f"{len(idx.entries)} entries loaded ({span})")

    st.sidebar.header("Find a thread")
    theme = st.sidebar.text_input("Search by theme or feeling",
                                  placeholder="e.g. a recovery, feeling alone, hope")
    k = st.sidebar.slider("How many entries to surface", 5, 100, 30, step=5)
    min_score = st.sidebar.slider("Relevance threshold", 0.0, 0.6, 0.20, step=0.02,
                                  help="Higher = only stronger matches.")
    order = st.sidebar.radio("Show in order of", ["Chronological", "Relevance"])

    if not theme:
        st.info("Enter a theme or feeling in the sidebar to surface related entries "
                "from across your journal, shown with their dates so you can follow "
                "how a thread developed over time.")
        st.stop()

    hits = [(e, s) for e, s in idx.search(theme, k=k) if s >= min_score]
    if not hits:
        st.warning("No entries matched. Lower the relevance threshold or rephrase.")
        st.stop()

    if order == "Chronological":
        hits.sort(key=lambda es: es[0].entry_date or date.min)
    else:
        hits.sort(key=lambda es: es[1], reverse=True)

    st.subheader(f"{len(hits)} entries related to “{theme}”")
    years = [e.entry_date.year for e, _ in hits if e.entry_date]
    if years:
        counts = {y: years.count(y) for y in range(min(years), max(years) + 1)}
        st.caption("When these entries fall (a first glimpse of the pattern):")
        st.bar_chart({"entries": counts})
    st.divider()

    for e, score in hits:
        d = e.entry_date
        weekday = e.meta.get("weekday") or (d.strftime("%A") if d else "")
        cols = st.columns([3, 1])
        cols[0].markdown(f"**{d or '?'} · {weekday}**")
        if people and d:
            ages = " · ".join(f"{n} {_age_at(b, d)}" for n, b in people.items())
            cols[1].markdown(f"<div style='text-align:right;color:gray;font-size:0.8em'>"
                             f"{ages}</div>", unsafe_allow_html=True)
        st.write(e.text)
        st.caption(f"relevance {score:.2f}")
        st.divider()


if __name__ == "__main__":
    run()
