"""Signal extraction shared by the arc stage: build the scored frame of entries,
cluster them into recurring situation-types (scenes/themes), score aspects and
valence, find turning points, and spot recurring people. Generalized from
memoir_recon.py / spine_proposer.py — all domain content comes in via Lenses.
"""
from __future__ import annotations
import re
import numpy as np

from ..types import DiaryEntry, Lenses, Scene


def snippet(t: str, n: int = 260) -> str:
    t = re.sub(r"\s+", " ", t or "").strip()
    return t[:n] + ("…" if len(t) > n else "")


def centroid(vecs: np.ndarray) -> np.ndarray:
    c = vecs.mean(axis=0)
    nrm = np.linalg.norm(c)
    return c / nrm if nrm else c


def top_terms(texts, k: int = 6) -> str:
    from sklearn.feature_extraction.text import TfidfVectorizer
    try:
        v = TfidfVectorizer(stop_words="english", token_pattern=r"[A-Za-z][A-Za-z]{2,}",
                            max_features=4000)
        X = v.fit_transform(texts)
        scores = np.asarray(X.mean(axis=0)).ravel()
        vocab = np.array(v.get_feature_names_out())
        return ", ".join(vocab[scores.argsort()[::-1][:k]])
    except ValueError:
        return "(not enough text)"


def kmeans(sub: np.ndarray, k: int):
    from sklearn.cluster import KMeans
    k = max(2, min(k, max(2, len(sub) // 15)))
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(sub)
    return km.labels_, k


def build_frame(entries: list[DiaryEntry], lenses: Lenses, embed,
                entry_embeddings: np.ndarray | None = None):
    """Return (df, E): a pandas frame of entries scored against the lenses, plus
    the embedding matrix. `embed` is any callable list[str] -> np.ndarray."""
    import pandas as pd
    df = pd.DataFrame([{"date": e.entry_date, "text": e.text} for e in entries])
    df["dt"] = pd.to_datetime(df["date"], errors="coerce")
    E = entry_embeddings if entry_embeddings is not None else embed(df["text"].tolist())

    for name, phrase in lenses.aspects.items():
        df["asp_" + name] = E @ embed([phrase])[0]
    df["top_aspect"] = df[["asp_" + a for a in lenses.aspects]].idxmax(axis=1).str.replace(
        "asp_", "", regex=False)

    pos = np.stack([E @ embed([p])[0] for p in lenses.emo_pos], 1).mean(1)
    neg = np.stack([E @ embed([p])[0] for p in lenses.emo_neg], 1).mean(1)
    df["valence"] = pos - neg
    return df, E


def extract_entities(texts, name_stop, min_count: int = 8, limit: int = 20):
    counts: dict[str, int] = {}
    for t in texts:
        for tok in re.findall(r"\b[A-Z][a-z]+\b", t or ""):
            if tok not in name_stop:
                counts[tok] = counts.get(tok, 0) + 1
    return [{"name": k, "count": v}
            for k, v in sorted(counts.items(), key=lambda x: -x[1]) if v >= min_count][:limit]


def turning_points(valid_df, top_n: int = 6):
    """Months with the biggest valence swings. Returns [(period_ts, zscore)]."""
    md = valid_df.copy().set_index("dt").sort_index()
    monthly = md["valence"].resample("MS").mean().dropna()
    if len(monthly) < 6:
        return [], monthly
    chg = monthly.diff()
    z = (chg - chg.mean()) / (chg.std() or 1)
    picks = sorted(z.abs().sort_values(ascending=False).head(top_n).index)
    return [(t, float(z.loc[t])) for t in picks], monthly


def _valid_mask(df, min_year, max_year):
    if min_year is None and max_year is None:
        return df["dt"].notna()
    yr = df["dt"].dt.year
    lo = min_year if min_year is not None else -(10 ** 9)
    hi = max_year if max_year is not None else 10 ** 9
    return df["dt"].notna() & yr.between(lo, hi)


def scenes_by_aspect(entries: list[DiaryEntry], embed, lenses: Lenses | None = None,
                     per_aspect: int = 15, min_year: int | None = None,
                     max_year: int | None = None, entry_embeddings=None) -> dict[str, list[Scene]]:
    """For each aspect, the entries that read most strongly as it — candidate
    scenes, ranked. Ported from memoir_scenes.py: rank, don't cluster, because
    logistics-style daily logs are too similar to cluster well."""
    import pandas as pd
    lenses = lenses or Lenses.default()
    df, _ = build_frame(entries, lenses, embed, entry_embeddings)
    valid_idx = df.index[_valid_mask(df, min_year, max_year)].to_numpy()
    result: dict[str, list[Scene]] = {}
    for a in lenses.aspects:
        if not len(valid_idx):
            result[a] = []
            continue
        s = df.loc[valid_idx, "asp_" + a].to_numpy()
        order = valid_idx[s.argsort()[::-1][:per_aspect]]
        scenes = []
        for i in order:
            d = df.at[i, "dt"]
            when = d.date().isoformat() if pd.notna(d) else None
            scenes.append(Scene(
                title=f"{a} — {when or '?'}",
                summary=snippet(df.at[i, "text"]),
                entries=[entries[i]] if i < len(entries) else [],
                when=when,
                meta={"aspect": a, "score": float(df.at[i, "asp_" + a])},
            ))
        result[a] = scenes
    return result


def scenes_report(entries: list[DiaryEntry], embed, lenses: Lenses | None = None,
                  per_aspect: int = 15, min_year: int | None = None,
                  max_year: int | None = None, entry_embeddings=None) -> str:
    """Markdown shortlist of candidate scenes by aspect, plus the aspect mix,
    turning points, and out-of-range dates. Ported from memoir_scenes.py."""
    import pandas as pd
    lenses = lenses or Lenses.default()
    df, _ = build_frame(entries, lenses, embed, entry_embeddings)
    df["valid"] = _valid_mask(df, min_year, max_year)
    out_of_range = df[df["dt"].notna() & ~df["valid"]]

    out = ["# Memoir scenes — candidate material by emotional aspect", "",
           f"{len(df)} entries · {int(df['valid'].sum())} in range. Each section lists the "
           "entries that read most strongly as that feeling — your shortlist of scenes, "
           "not errand-logs.", "",
           "## Aspect mix (each in-range entry's dominant feeling)", "",
           "| aspect | entries |", "|---|---|"]
    for a, n in df[df["valid"]]["top_aspect"].value_counts().items():
        out.append(f"| {a} | {int(n)} |")
    out.append("")

    by_aspect = scenes_by_aspect(entries, embed, lenses, per_aspect, min_year, max_year,
                                 entry_embeddings=None)
    for a, scenes in by_aspect.items():
        out += [f"## {a.upper()} — top {per_aspect} candidate scenes", ""]
        for sc in scenes:
            out.append(f"- *{sc.when or '?'}* (score {sc.meta['score']:.2f}): {sc.summary}")
        out.append("")

    tps, _ = turning_points(df[df["valid"]])
    if tps:
        mi = df[df["valid"]].set_index("dt").sort_index()
        out += ["## Candidate turning points (biggest month-to-month swings)", ""]
        for t, z in tps:
            near = mi[(mi.index.year == t.year) & (mi.index.month == t.month)]
            out.append(f"**{t.strftime('%B %Y')}** (z={z:+.2f})")
            for i in range(min(2, len(near))):
                out.append(f"  - {snippet(near.iloc[i]['text'])}")
            out.append("")

    if len(out_of_range):
        out += ["## Entries with out-of-range dates (likely parser typos to fix)", ""]
        for _, r in out_of_range.iterrows():
            out.append(f"- parsed as {r['dt'].date().isoformat()}: {snippet(r['text'], 160)}")
        out.append("")
    return "\n".join(out)
