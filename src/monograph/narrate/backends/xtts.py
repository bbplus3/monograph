"""XTTS-v2 (Coqui) narrator. Smoother prosody than F5 but more prone to
mispronunciation; pair it with a pronunciation map. Takes short inputs, so this
backend splits each paragraph into sentence-sized chunks internally.

Loads the reference via soundfile (not torchaudio) to sidestep the torchcodec/
FFmpeg requirement on newer PyTorch.
"""
from __future__ import annotations
import os
import numpy as np
from ..base import Narrator
from .. import chapterize, audio as A


class XTTSNarrator(Narrator):
    def __init__(self, cfg):
        super().__init__(cfg)
        if not cfg.voice_ref:
            raise ValueError("XTTS needs Config.voice_ref (a WAV of the author's voice).")
        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        try:
            import torch, soundfile as sf
            import torchaudio
            from TTS.api import TTS
        except ImportError as e:
            raise ImportError("XTTS backend needs coqui-tts: pip install 'monograph[narrate-xtts]'") from e

        # read audio via soundfile, not torchaudio's torchcodec/FFmpeg path
        _orig = torchaudio.load
        def _sf_load(path, *a, **k):
            try:
                data, srate = sf.read(path, dtype="float32", always_2d=True)
                return torch.from_numpy(data.T.copy()), srate
            except Exception:
                return _orig(path, *a, **k)
        torchaudio.load = _sf_load

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        self._limit = int(getattr(cfg, "chunk_char_limit", 220))

    def synth(self, text: str) -> tuple[np.ndarray, int]:
        sr = self.sr
        pieces, gap = [], A.silence(0.12, sr)
        for chunk in chapterize.split_chunks(text, self._limit):
            wav = self._tts.tts(text=chunk, speaker_wav=self.cfg.voice_ref, language="en")
            pieces.append(A.trim_silence(np.asarray(wav, dtype=np.float32), sr))
            pieces.append(gap)
        out = np.concatenate(pieces) if pieces else A.silence(0.05, sr)
        return out, sr
