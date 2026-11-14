"""Propose candidate memoir spines from an Evidence pack, via any LLM. Ported
from spine_proposer.py: the system prompt and evidence formatting are already
domain-agnostic; only the Ollama call was specific, and that now lives behind the
LLM interface.

Each spine is one possible book — a throughline question, the arc it implies, the
scenes it draws on, and (crucially) what the journal is *missing* that the author
would have to write from memory.
"""
from __future__ import annotations

from ..types import Evidence, Spine
from ..llm import LLM
from ..llm.ollama_backend import parse_json

SYSTEM = (
    "You are a literary memoir editor doing structural reconnaissance on a writer's "
    "multi-year personal journal. You are given only aggregate signals and a capped "
    "set of representative entry snippets — never the whole journal. "
    "Propose 3 to 6 candidate SPINES the journal could become: distinct memoirs, each "
    "organized around one transformation or question.\n"
    "Hard rules:\n"
    "- Ground every spine ONLY in the evidence provided. Do not invent events, people, "
    "or feelings.\n"
    "- A diary records logistics and the present; it lacks interiority, retrospection, "
    "and anything before it starts. For EACH spine, state plainly in "
    "'missing_to_author' what is NOT in the journal and would have to be written from "
    "memory (e.g. childhood, how the writer was parented, what a moment felt like "
    "inside).\n"
    "- 'throughline' must be a QUESTION the reader follows.\n"
    "- 'scene_ids' must be chosen only from the provided snippet ids.\n"
    "Return ONLY JSON of the form: {\"spines\": [ {\"id\": str, \"title\": str, "
    "\"throughline\": str, \"arc\": str, \"draws_on\": {\"aspects\": [str], "
    "\"people\": [str], \"period\": str}, \"scene_ids\": [int], \"strengths\": str, "
    "\"missing_to_author\": str} ] }"
)


def _build_user(ev: Evidence) -> str:
    lines = ["EVIDENCE PACK", "", f"{ev.n_valid} dated entries in range.", "",
             "Aspect lenses (share of entries where this is the dominant feeling, and "
             "how its prevalence trends early->late):"]
    for a in ev.aspects:
        lines.append(f"- {a['name']}: {a['share_pct']}% ({a['trend']}); example ids {a['top_ids']}")
    lines += ["", "Entries per year: " + ", ".join(f"{y}:{c}" for y, c in ev.timeline.items())]
    lines += ["", "Biggest emotional swings (candidate turning points):"]
    for t in ev.turning_points:
        lines.append(f"- {t['month']} (z={t['z']}); ids {t['ids']}")
    lines += ["", "Recurring people: " + ", ".join(f"{e['name']}({e['count']})" for e in ev.entities)]
    lines += ["", "REPRESENTATIVE SNIPPETS (cite these ids in scene_ids):"]
    for i, text in ev.snippets.items():
        lines.append(f"[{i}] {text}")
    lines += ["", "Now propose the candidate spines as JSON."]
    return "\n".join(lines)


def propose_spines(evidence: Evidence, llm: LLM, temperature: float = 0.4) -> list[Spine]:
    """Ask the LLM for candidate spines and return them as Spine objects. The raw
    fields (throughline, draws_on, scene_ids, strengths, missing_to_author) are
    kept in Spine.meta, with resolved snippet texts for the chosen scene_ids."""
    content = llm.complete(_build_user(evidence), system=SYSTEM,
                           temperature=temperature, json_mode=True)
    result = parse_json(content)
    raw = result.get("spines", result if isinstance(result, list) else [])
    spines = []
    for sp in raw:
        scene_ids = sp.get("scene_ids", [])
        scene_texts = {int(i): evidence.snippets.get(int(i))
                       for i in scene_ids if str(i).lstrip("-").isdigit()}
        spines.append(Spine(
            title=sp.get("title", "(untitled)"),
            premise=sp.get("throughline", ""),
            arc=sp.get("arc", ""),
            scenes=[],
            meta={"id": sp.get("id"), "draws_on": sp.get("draws_on", {}),
                  "scene_ids": scene_ids, "scene_texts": scene_texts,
                  "strengths": sp.get("strengths", ""),
                  "missing_to_author": sp.get("missing_to_author", "")},
        ))
    return spines
