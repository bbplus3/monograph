"""Distill a diary into an Evidence pack — aspect prevalence and trend, a
timeline, turning points, recurring people, and a capped set of representative
snippets. This is the structured, privacy-preserving summary handed to the LLM
in propose_spines (the whole journal is never sent). Ported from
spine_proposer.build_evidence; all domain content comes from Lenses.
"""
from __future__ import annotations
import numpy as np

from ..types import DiaryEntry, Lenses, Evidence
from .scenes import build_frame, extract_entities, turning_points, snippet


def build_evidence(entries: list[DiaryEntry], embed, lenses: Lenses | None = None,
                   min_year: int | None = None, max_year: int | None = None,
                   per_aspect: int = 3, entry_embeddings=None) -> Evidence:
    lenses = lenses or Lenses.default()
    df, E = build_frame(entries, lenses, embed, entry_embeddings)

    yr = df["dt"].dt.year
    if min_year is None and max_year is None:
        df["valid"] = df["dt"].notna()
    else:
        lo = min_year if min_year is not None else -(10 ** 9)
        hi = max_year if max_year is not None else 10 ** 9
        df["valid"] = df["dt"].notna() & yr.between(lo, hi)
    valid = df[df["valid"]]
    vidx = valid.index.to_numpy()

    order = valid.sort_values("dt").index.to_numpy()
    half = len(order) // 2
    early, late = set(order[:half]), set(order[half:])

    aspects, ref_ids = [], set()
    for a in lenses.aspects:
        share = (valid["top_aspect"] == a).mean() * 100 if len(valid) else 0.0
        e_share = np.mean([df.at[i, "top_aspect"] == a for i in early]) if early else 0
        l_share = np.mean([df.at[i, "top_aspect"] == a for i in late]) if late else 0
        trend = ("rising" if l_share > e_share * 1.25
                 else "falling" if l_share < e_share * 0.8 else "steady")
        s = df.loc[vidx, "asp_" + a].to_numpy() if len(vidx) else np.array([])
        top = vidx[s.argsort()[::-1][:per_aspect]] if len(vidx) else np.array([], dtype=int)
        ref_ids.update(int(i) for i in top)
        aspects.append({"name": a, "share_pct": round(float(share), 1),
                        "trend": trend, "top_ids": [int(i) for i in top]})

    tps, _ = turning_points(valid)
    turning = []
    for t, z in tps:
        mask = valid["dt"].dt.to_period("M") == t.to_period("M")
        ids = valid.index[mask].to_numpy()[:2]
        ref_ids.update(int(i) for i in ids)
        turning.append({"month": t.strftime("%Y-%m"), "z": round(z, 2),
                        "ids": [int(i) for i in ids]})

    entities = extract_entities(valid["text"], lenses.name_stop)
    timeline = {int(y): int(c)
                for y, c in valid["dt"].dt.year.value_counts().sort_index().items()}
    snippets = {int(i): f"{df.at[i, 'dt'].date().isoformat()} — {snippet(df.at[i, 'text'])}"
                for i in sorted(ref_ids)}

    return Evidence(n_entries=int(len(df)), n_valid=int(len(valid)), aspects=aspects,
                    timeline=timeline, turning_points=turning, entities=entities,
                    snippets=snippets)
