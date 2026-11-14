"""arc — the memoir-structure engine: distill a diary into signals, write a
reconnaissance report, and propose candidate spines. This is the heart of
monograph, ported from memoir_recon.py and spine_proposer.py and generalized so
all domain content lives in a `Lenses` object the writer supplies.

    from monograph.index import get_embedder
    from monograph.llm import get_llm
    from monograph.types import Lenses

    embed = get_embedder("local")
    report = recon_report(entries, embed, lenses=Lenses.default())   # markdown
    evidence = build_evidence(entries, embed, lenses=Lenses.default())
    spines = propose_spines(evidence, get_llm("ollama", cfg))        # candidate books
"""
from .analyze import build_evidence
from .recon import recon_report
from .spine import propose_spines
from .scenes import scenes_by_aspect, scenes_report

__all__ = ["build_evidence", "recon_report", "propose_spines",
           "scenes_by_aspect", "scenes_report"]
