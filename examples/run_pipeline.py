"""End-to-end sketch: diary -> memoir -> audio in the author's voice.

The `arc` and `narrate` stages are implemented; `ingest`/`index`/`write` run once
you've ported your own logic into their stubs. This file *uses* the library — it's
the general version of your project-specific driver scripts.
"""
from monograph import Config
from monograph.narrate import get_narrator


def narrate_only():
    """Works today: a folder of chapter .txt -> one .wav each, resumable."""
    cfg = Config(
        voice_ref="voice.wav",
        voice_ref_text="I am recording my voice so I can read my own book aloud, "
                       "in a calm and natural tone, the way I would read a story to someone I love.",
        narrate_backend="f5",
        cross_fade=0.30,
    )
    get_narrator(cfg.narrate_backend, cfg).narrate_dir("chapters/", "audio/")


def find_the_arc():
    """Works today (needs the `arc` + `index` extras and a local LLM via Ollama).
    Turns parsed entries into a recon report and candidate spines."""
    from monograph.ingest import parse_diary
    from monograph.index import get_embedder
    from monograph.arc import recon_report, build_evidence, propose_spines
    from monograph.llm import get_llm
    from monograph.types import Lenses

    # Bring your own lenses; see examples/lenses_parenting.py, or Lenses.default().
    lenses = Lenses.default()

    entries = parse_diary("./my-diary")          # once ingest is ported
    embed = get_embedder("local")

    open("recon_report.md", "w").write(recon_report(entries, embed, lenses=lenses))

    cfg = Config(llm_backend="ollama", options={"llm_model": "llama3.1:8b"})
    evidence = build_evidence(entries, embed, lenses=lenses)
    spines = propose_spines(evidence, get_llm(cfg.llm_backend, cfg))
    for s in spines:
        print(s.title, "—", s.premise)


if __name__ == "__main__":
    narrate_only()
