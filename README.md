# monograph

Turn a personal diary into a memoir: **explore** it, find its **arcs**, **write**
it, and **narrate** it in your own voice. Local-first by design.

`monograph` is a pipeline with five stages, each usable on its own:

1. **ingest** — parse a diary (any format) into structured entries.
2. **index** — build a searchable index and query it (retrieval over your own writing).
3. **arc** — find scenes, propose narrative spines, and reconstruct a memoir structure.
4. **write** — draft prose from the chosen spine, in a style you supply.
5. **narrate** — turn the finished text into audio read in the author's own cloned voice.

A thin `cli` wires the stages together, and an optional `app` gives a GUI.

## Privacy, first

A diary is the most sensitive text someone owns. `monograph` runs **local by
default** — local parsing, local embeddings, local files, local TTS — and never
sends your words anywhere unless you explicitly choose a cloud backend and pass a
key. `Config.local\_only` defaults to `True`, and any backend that would make a
network call must check it and refuse when it's set. Treat that flag as a
load-bearing contract, not a suggestion.

## Getting started (step by step)

If you don't write code, follow this once and you'll have a working setup. You need
**Python 3.10 or newer** installed first (python.org → Downloads; on Windows, tick
"Add Python to PATH" during install).

**1. Open a terminal.**

* macOS: open the **Terminal** app (Cmd-Space, type "Terminal").
* Windows: open **PowerShell** (Start menu, type "PowerShell").

**2. Go to the folder that holds monograph.** Use `cd` ("change directory"):

```bash
cd path/to/monograph          # e.g.  cd Desktop/monograph
```

Tip: type `cd ` (with a space), then drag the folder onto the terminal window — it
fills in the path for you.

**3. Create a virtual environment** (a private sandbox for this project's packages),
and **activate** it:

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\\Scripts\\Activate.ps1
```

When it's active you'll see `(.venv)` at the start of the line. Do this (step 3,
activate) every time you open a new terminal to work on the project.

**4. Install monograph** and the pieces you want:

```bash
pip install -e ".\[app,arc,index]"      # GUI + arc engine + local search
pip install -e ".\[narrate-f5]"         # add the F5 narrator (big download)
pip install -e ".\[all]"                # everything at once
```

Narration and model backends are heavy and optional — install only what you'll use.

**5. Launch the app:**

```bash
monograph app
```

That opens the GUI in your web browser. Close it with Ctrl-C in the terminal.

## Using it from the command line

You can also drive each stage directly (handy for long narration runs):

```bash
monograph prepare-voice --src recording.m4a --out voice.wav
monograph narrate --text chapters/ --out audio/ --voice voice.wav --ref-text "I am recording my voice ..."
```

## Using it from Python

```python
from monograph import Config
from monograph.narrate import get\_narrator

cfg = Config(voice\_ref="voice.wav", voice\_ref\_text="I am recording my voice ...")
get\_narrator("f5", cfg).narrate\_dir("chapters/", "audio/")   # one wav per .txt, resumable
```

## File names: how the stages connect

You never have to rename your diary to match the code — every path is an argument
(the original scripts hardcoded names like `journal\_all.docx`; this doesn't). Two
things matter:

* **Your diary** can have any name and live anywhere. Point `--diary` (or the app's
sidebar, or `Config.diary\_path`) at it — `.docx`, `.txt`/`.md`, `.json`, `.csv`,
`.pkl`, or a folder of them.
* **The index** is saved under a base name you choose with `--out` (default
`diary\_index`); building it writes `<base>.npy` and `<base>.meta.json`. Every stage
that *reads* the index — `search`, `write`, and `app` — takes that same base name
via `--index` (the app via its sidebar or the `MONOGRAPH\_INDEX` env var). The one
rule: **use the same base name to build and to read.**

A consistent run:

```bash
monograph index  --diary my\_journal.docx --out diary\_index
monograph scenes --diary my\_journal.docx
monograph search --index diary\_index "feeling alone"
monograph write  --index diary\_index "breaking the cycle"
monograph app                      # reads diary\_index by default
```

Change `diary\_index` everywhere together if you want a different name — nothing is
hardcoded, and no file needs renaming. (Age labels in the app are optional: add a
`monograph.toml` with a `\[people]` table of `Name = "YYYY-MM-DD"` birthdates.)

## Package layout

```
monograph/
  config.py        Config + privacy defaults
  types.py         shared dataclasses (DiaryEntry, Scene, Spine, Chapter, ...)
  ingest/          parse diaries -> entries
  index/           build + search (pluggable: base.Index, local.LocalIndex)
  arc/             scenes, spine, recon  (the memoir-structure engine)
  write/           memoir engine + style profiles
  narrate/         Narrator backends (F5, XTTS) + chapterize + pronunciation + voice prep
  llm/             LLM interface (pluggable; local or cloud, opt-in)
  cli.py           thin command line
  app.py           optional Streamlit front-end
```

Each backend sits behind a small interface (`index.Index`, `llm.LLM`,
`narrate.Narrator`) so anyone can swap in their own without touching the core.

## Status

All five stages are implemented and tested — `ingest`, `index`, `arc`, `write`,
`narrate` — so the full chain runs end to end: parse the diary, build the index,
run recon / scene / spine analysis, draft the memoir, and narrate it in the
author's voice. The `write` stage drafts a passage from a theme or date range, and
`write\_memoir()` assembles one chapter per aspect a chosen spine draws on; treat
that assembly as a sensible default to reorder or merge to taste.

