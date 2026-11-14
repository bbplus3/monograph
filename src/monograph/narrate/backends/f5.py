"""F5-TTS narrator. Accurate pronunciation; feeds whole paragraphs and lets F5
cross-fade its internal segments (set via Config.cross_fade). Needs a short,
accurately-transcribed reference clip — see monograph.narrate.voice.prepare_reference
and Config.voice_ref / voice_ref_text."""
from __future__ import annotations
import numpy as np
from ..base import Narrator


class F5Narrator(Narrator):
    def __init__(self, cfg):
        super().__init__(cfg)
        if not cfg.voice_ref or not cfg.voice_ref_text:
            raise ValueError("F5 needs Config.voice_ref (a WAV) and voice_ref_text "
                             "(its exact transcript).")
        try:
            from f5_tts.api import F5TTS
        except ImportError as e:
            raise ImportError("F5 backend needs f5-tts: pip install 'monograph[narrate-f5]'") from e
        self._model = F5TTS()                       # default model, auto device
        self.cross_fade = float(getattr(cfg, "cross_fade", 0.30))

    def synth(self, text: str) -> tuple[np.ndarray, int]:
        wav, sr, _ = self._model.infer(
            ref_file=self.cfg.voice_ref,
            ref_text=self.cfg.voice_ref_text,
            gen_text=text,
            cross_fade_duration=self.cross_fade,
            remove_silence=False,                   # we do our own silence cleanup
            show_info=lambda *a, **k: None,
        )
        return np.asarray(wav, dtype=np.float32), int(sr)
