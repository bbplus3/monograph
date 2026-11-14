"""The Narrator interface and the shared narration flow. A backend only has to
implement `synth(text) -> (audio, sr)`; the base class handles pronunciation,
cleaning, paragraph structure, the pause after a chapter title, silence cleanup,
writing, and resumable folder runs."""
from __future__ import annotations
import abc
import os
import numpy as np

from . import chapterize, pronunciation, audio as A


class Narrator(abc.ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self._pron = pronunciation.load(getattr(cfg, "pronunciation_path", None))
        self.sr = 24000

    # --- the one thing each backend implements -------------------------------
    @abc.abstractmethod
    def synth(self, text: str) -> tuple[np.ndarray, int]:
        """Turn a block of text into (float32 audio, sample_rate)."""
        raise NotImplementedError

    # --- shared pipeline -----------------------------------------------------
    def _prep(self, text: str) -> str:
        return pronunciation.apply(chapterize.clean_text(text), self._pron)

    def narrate_text(self, text: str, out_path: str,
                     title_pause: float = 0.7, para_gap: float = 0.4) -> str:
        """Narrate one chapter to a WAV: title, a beat, then each paragraph."""
        title, paras = chapterize.split_document(text)
        parts, sr = [], self.sr
        if title:
            a, sr = self.synth(self._prep(title))
            parts += [A.trim_silence(a, sr), A.silence(title_pause, sr)]
        for para in paras:
            if not para.strip():
                continue
            a, sr = self.synth(self._prep(para))
            parts += [A.compress_silences(A.trim_silence(a, sr), sr), A.silence(para_gap, sr)]
        out = np.concatenate(parts) if parts else A.silence(0.1, sr)
        A.write_wav(out_path, out, sr)
        return out_path

    def narrate_dir(self, text_dir: str, out_dir: str, only: str | None = None,
                    verbose: bool = True) -> list[str]:
        """Narrate every .txt in `text_dir` to a WAV in `out_dir`. Resumable:
        skips chapters already done. Returns the list of output paths."""
        os.makedirs(out_dir, exist_ok=True)
        results = []
        for stem, text in chapterize.read_documents(text_dir):
            if only and only not in stem:
                continue
            out = os.path.join(out_dir, stem + ".wav")
            if os.path.exists(out):
                if verbose:
                    print("skip (done):", stem)
                results.append(out)
                continue
            if verbose:
                print(f"[{stem}] narrating ...")
            try:
                self.narrate_text(text, out)
                results.append(out)
            except Exception as e:                       # one bad chapter never halts the run
                if verbose:
                    print(f"   !! failed on {stem}: {e} (skipped — re-run to retry)")
        return results
