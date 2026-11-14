"""Thin command line over the pipeline. Each subcommand maps to one stage; the
`narrate` command works today, the others wire into their stage once you've
ported your logic in.

    monograph narrate --voice voice.wav --ref-text "..." --text chapters/ --out audio/
    monograph prepare-voice --src recording.m4a --out voice.wav
    monograph pipeline --diary ./my-diary --voice voice.wav --out ./book
"""
from __future__ import annotations
import argparse
import sys

from .config import Config


def _cfg(args) -> Config:
    cfg = Config.from_toml(args.config) if getattr(args, "config", None) else Config()
    for k in ("voice_ref", "voice_ref_text", "pronunciation_path",
              "narrate_backend", "work_dir", "diary_path"):
        v = getattr(args, k, None)
        if v is not None:
            setattr(cfg, k, v)
    return cfg


def cmd_prepare_voice(args):
    from .narrate.voice import prepare_reference
    out = prepare_reference(args.src, args.out)
    print("wrote", out)


def cmd_narrate(args):
    from .narrate import get_narrator
    cfg = _cfg(args)
    cfg.voice_ref = args.voice or cfg.voice_ref
    cfg.voice_ref_text = args.ref_text or cfg.voice_ref_text
    cfg.narrate_backend = args.backend or cfg.narrate_backend
    narrator = get_narrator(cfg.narrate_backend, cfg)
    outs = narrator.narrate_dir(args.text, args.out, only=args.only)
    print(f"done: {len(outs)} chapter(s) in {args.out}")


def cmd_pipeline(args):
    print("Full diary->memoir->audio pipeline is assembled from the stages once "
          "ingest/index/arc/write are ported in. See examples/run_pipeline.py.")


def cmd_index(args):
    """Parse a diary and build + save a searchable index."""
    from .ingest import parse_diary
    from .index.local import LocalIndex
    cfg = _cfg(args)
    entries = parse_diary(args.diary)
    idx = LocalIndex(cfg).build(entries)
    idx.save(args.out)
    print(f"indexed {len(idx.entries)} entries -> {args.out}.npy / {args.out}.meta.json")


def cmd_search(args):
    """Search a saved index by meaning (one-shot, or interactive)."""
    from .index.local import LocalIndex
    idx = LocalIndex.load(args.index)
    print(f"{len(idx.entries)} entries indexed.")

    def show(query):
        print(f"\n=== Top matches for: {query!r} ===\n")
        for entry, score in idx.search(query, k=args.k):
            d = entry.entry_date.isoformat() if entry.entry_date else "?"
            preview = (entry.text or "").replace("\n", " ")
            preview = preview[:300] + ("..." if len(preview) > 300 else "")
            print(f"[{d}]  (similarity {score:.3f})\n  {preview}\n")

    if args.query:
        show(" ".join(args.query))
        return
    print("Type a theme, feeling, or topic (empty line or 'quit' to exit).\n")
    while True:
        try:
            q = input("search> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("quit", "exit"):
            break
        show(q)


def cmd_scenes(args):
    """Write the candidate-scenes-by-aspect report (memoir_scenes)."""
    from .ingest import parse_diary
    from .index import get_embedder
    from .arc import scenes_report
    cfg = _cfg(args)
    entries = parse_diary(args.diary)
    report = scenes_report(entries, get_embedder("local", cfg), per_aspect=args.per_aspect)
    open(args.out, "w", encoding="utf-8").write(report)
    print(f"wrote {args.out} ({len(entries)} entries)")


def cmd_write(args):
    """Draft a memoir passage from a theme or a date range (local LLM)."""
    from .index.local import LocalIndex
    from .write.engine import MemoirEngine
    from .llm import get_llm
    cfg = _cfg(args)
    cfg.llm_backend = args.llm or cfg.llm_backend or "ollama"
    idx = LocalIndex.load(args.index)
    engine = MemoirEngine(get_llm(cfg.llm_backend, cfg), index=idx)
    if args.date_from and args.date_to:
        ch = engine.write_passage(date_from=args.date_from, date_to=args.date_to)
    else:
        ch = engine.write_passage(" ".join(args.theme), k=args.k)
    print("=" * 70 + f"\n{ch.text}\n" + "=" * 70)
    print("source dates:", ", ".join(ch.meta.get("source_dates", [])))


def cmd_app(args):
    """Launch the Streamlit front-end with one command."""
    import importlib.util
    import subprocess
    import sys
    spec = importlib.util.find_spec("monograph.app")
    if spec is None or not spec.origin:
        raise SystemExit("Could not locate monograph.app")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", spec.origin], check=True)
    except FileNotFoundError:
        raise SystemExit("Streamlit isn't installed. Run:  pip install 'monograph[app]'")


def main(argv=None):
    p = argparse.ArgumentParser(prog="monograph", description="diary -> memoir -> narrated audio")
    p.add_argument("--config", help="optional TOML config")
    sub = p.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("prepare-voice", help="clean/level a recording into a reference WAV")
    pv.add_argument("--src", required=True)
    pv.add_argument("--out", default="voice.wav")
    pv.set_defaults(func=cmd_prepare_voice)

    n = sub.add_parser("narrate", help="narrate a folder of .txt chapters")
    n.add_argument("--text", required=True, help="folder of chapter .txt files")
    n.add_argument("--out", default="audio", help="output folder for .wav files")
    n.add_argument("--voice", help="reference voice WAV")
    n.add_argument("--ref-text", dest="ref_text", help="exact transcript of the reference (F5)")
    n.add_argument("--backend", default=None, choices=["f5", "xtts"])
    n.add_argument("--pronunciation-path", dest="pronunciation_path", default=None)
    n.add_argument("--only", default=None, help="only chapters whose name contains this")
    n.set_defaults(func=cmd_narrate)

    pl = sub.add_parser("pipeline", help="run the whole thing (once stages are ported)")
    pl.add_argument("--diary"); pl.add_argument("--voice"); pl.add_argument("--out", default="book")
    pl.set_defaults(func=cmd_pipeline, diary_path=None)

    ix = sub.add_parser("index", help="parse a diary and build a searchable index")
    ix.add_argument("--diary", required=True, help="diary file or folder")
    ix.add_argument("--out", default="diary_index", help="output base path")
    ix.set_defaults(func=cmd_index)

    se = sub.add_parser("search", help="search a saved index by meaning")
    se.add_argument("--index", default="diary_index", help="index base path (from `index`)")
    se.add_argument("-k", type=int, default=5, help="how many matches to return")
    se.add_argument("query", nargs="*", help="query text; omit for interactive mode")
    se.set_defaults(func=cmd_search)

    sc = sub.add_parser("scenes", help="candidate scenes by aspect (report)")
    sc.add_argument("--diary", required=True)
    sc.add_argument("--out", default="memoir_scenes.md")
    sc.add_argument("--per-aspect", dest="per_aspect", type=int, default=15)
    sc.set_defaults(func=cmd_scenes)

    wr = sub.add_parser("write", help="draft a memoir passage (theme or date range)")
    wr.add_argument("--index", default="diary_index")
    wr.add_argument("--from", dest="date_from"); wr.add_argument("--to", dest="date_to")
    wr.add_argument("--llm", default=None, help="llm backend (default ollama)")
    wr.add_argument("-k", type=int, default=6)
    wr.add_argument("theme", nargs="*", help="theme text; or use --from/--to")
    wr.set_defaults(func=cmd_write)

    ap_app = sub.add_parser("app", help="launch the Streamlit front-end")
    ap_app.set_defaults(func=cmd_app)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
