"""Reconnaissance report over a diary: split into optional strands, cluster each
into recurring situation-types, score aspects, trace the emotional trajectory,
flag turning points, and (for a compare/contrast memoir) pair one strand's
situations with their nearest echoes in another. Returns Markdown.

Ported from memoir_recon.py. The present/past split there is now just a two-entry
`Lenses.strands`; with no strands it degrades to a single-strand report.
"""
from __future__ import annotations
import numpy as np

from ..types import DiaryEntry, Lenses
from .scenes import build_frame, centroid, top_terms, kmeans, snippet, turning_points


def recon_report(entries: list[DiaryEntry], embed, lenses: Lenses | None = None,
                 spine_percentile: float = 55, min_year: int | None = None,
                 max_year: int | None = None, entry_embeddings=None) -> str:
    import pandas as pd
    lenses = lenses or Lenses.default()
    df, E = build_frame(entries, lenses, embed, entry_embeddings)

    strand_names = list(lenses.strands.keys())
    if strand_names:
        S = np.stack([E @ centroid(embed(seeds)) for seeds in lenses.strands.values()], 1)
        df["strand"] = [strand_names[i] for i in S.argmax(1)]
        relevance = S.max(1)
    else:
        df["strand"] = "all"
        relevance = df[["asp_" + a for a in lenses.aspects]].to_numpy().max(1)
    df["spine"] = relevance >= np.quantile(relevance, spine_percentile / 100.0)

    out = ["# Memoir reconnaissance", "",
           f"{len(df)} entries · {int(df['dt'].notna().sum())} dated · "
           f"spine subset: {int(df['spine'].sum())} entries "
           f"(top {100 - spine_percentile:.0f}% by relevance).", ""]
    if strand_names:
        out += ["Strands: " + " vs. ".join(strand_names), ""]

    # strands over time
    if strand_names:
        dd = df[df["spine"] & df["dt"].notna()].copy()
        dd["year"] = dd["dt"].dt.year
        pivot = dd.pivot_table(index="year", columns="strand", values="text",
                               aggfunc="count", fill_value=0)
        out += ["## Strands over time (spine entries per year)", "",
                "| year | " + " | ".join(strand_names) + " |",
                "|---|" + "---|" * len(strand_names)]
        for y, row in pivot.iterrows():
            out.append("| " + str(y) + " | "
                       + " | ".join(str(int(row.get(n, 0))) for n in strand_names) + " |")
        out.append("")

    # clusters per strand
    def cluster_block(strand, title):
        idx = df.index[df["spine"] & (df["strand"] == strand)].to_numpy()
        if len(idx) < 6:
            return [f"## {title}", "", "_(too few entries to cluster)_", ""]
        labels, k = kmeans(E[idx], k=len(idx) // 30 + 3)
        block = [f"## {title}  ({len(idx)} entries, {k} themes)", ""]
        for c in range(k):
            members = idx[labels == c]
            if len(members) == 0:
                continue
            sub_c = centroid(E[members])
            order = members[(E[members] @ sub_c).argsort()[::-1]]
            terms = top_terms([df.at[i, "text"] for i in members])
            yrs = df.loc[members, "dt"].dt.year.dropna()
            span = f"{int(yrs.min())}-{int(yrs.max())}" if len(yrs) else "undated"
            block.append(f"**Theme {c + 1}** ({len(members)} entries, {span}) — {terms}")
            for i in order[:2]:
                d = df.at[i, "dt"]
                ds = d.date().isoformat() if pd.notna(d) else "?"
                block.append(f"  - *{ds}*: {snippet(df.at[i, 'text'])}")
            block.append("")
        return block

    for strand in (strand_names or ["all"]):
        label = strand if strand != "all" else "recurring situations"
        out += cluster_block(strand, f"{label} — recurring situations")

    # aspect distribution
    out += ["## Aspects across the spine", "", "| aspect | entries |", "|---|---|"]
    for a, n in df[df["spine"]]["top_aspect"].value_counts().items():
        out.append(f"| {a} | {int(n)} |")
    out.append("")

    # compare / contrast between the first two strands
    if len(strand_names) >= 2:
        a_name, b_name = strand_names[0], strand_names[1]
        a_idx = df.index[df["spine"] & (df["strand"] == a_name)].to_numpy()
        b_idx = df.index[df["spine"] & (df["strand"] == b_name)].to_numpy()
        out += [f"## Compare / contrast — {a_name} situations and their {b_name} echoes", ""]
        if len(a_idx) >= 6 and len(b_idx) >= 1:
            labels, k = kmeans(E[a_idx], k=len(a_idx) // 30 + 3)
            for c in range(k):
                members = a_idx[labels == c]
                if len(members) == 0:
                    continue
                cc = centroid(E[members])
                rep = members[(E[members] @ cc).argmax()]
                echoes = b_idx[(E[b_idx] @ cc).argsort()[::-1][:2]]
                drep = df.at[rep, "dt"]
                out.append(f"**{a_name}:** *{drep.date().isoformat() if pd.notna(drep) else '?'}*: "
                           f"{snippet(df.at[rep, 'text'])}")
                for e in echoes:
                    de = df.at[e, "dt"]
                    out.append(f"  - **{b_name} echo** "
                               f"*{de.date().isoformat() if pd.notna(de) else '?'}*: "
                               f"{snippet(df.at[e, 'text'])}")
                out.append("")
        else:
            out.append("_(not enough paired material — lower spine_percentile)_\n")

    # emotional trajectory + turning points
    md = df[df["spine"] & df["dt"].notna()].copy()
    tps, monthly = turning_points(md)
    out += ["## Emotional trajectory (spine entries, monthly mean valence)", "",
            "Valence > 0 leans warm/positive, < 0 leans hard/negative.", ""]
    if len(monthly) >= 6:
        yr = monthly.resample("YS").mean()
        out += ["| year | mean valence |", "|---|---|"]
        for d, v in yr.items():
            out.append(f"| {d.year} | {v:+.3f} |")
        out.append("")
        out += ["### Candidate turning points (largest month-to-month swings)", ""]
        mi = md.set_index("dt").sort_index()
        for t, z in tps:
            near = mi[(mi.index.year == t.year) & (mi.index.month == t.month)]
            out.append(f"**{t.strftime('%B %Y')}** (Δvalence z={z:+.2f})")
            for i in range(min(2, len(near))):
                out.append(f"  - {snippet(near.iloc[i]['text'])}")
            out.append("")
    else:
        out.append("_(not enough dated spine entries for a trajectory)_\n")

    return "\n".join(out)
