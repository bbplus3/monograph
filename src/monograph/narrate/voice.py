"""Prepare a reference voice clip for cloning: convert to mono, resample, and
normalize the level. Uses ffmpeg when available (handles .m4a and friends), and
falls back to soundfile for WAV input so the common case needs no ffmpeg."""
from __future__ import annotations
import os
import shutil
import subprocess
import numpy as np


def prepare_reference(src: str, out_wav: str, sr: int = 24000,
                      target_peak_dbfs: float = -3.0, max_seconds: float | None = 15.0) -> str:
    """Produce a clean mono WAV from any recording, leveled to `target_peak_dbfs`.

    - Converts/decodes with ffmpeg if present; otherwise reads WAV via soundfile.
    - Trims leading/trailing silence and caps length (short clips clone best;
      F5-TTS in particular wants a reference under ~15 seconds).
    Returns the output path.
    """
    audio, in_sr = _load_any(src, sr)
    # trim silent edges
    idx = np.where(np.abs(audio) > 0.02)[0]
    if len(idx):
        s = max(0, idx[0] - int(0.05 * sr))
        e = min(len(audio), idx[-1] + int(0.12 * sr))
        audio = audio[s:e]
    # cap length
    if max_seconds is not None and len(audio) > int(max_seconds * sr):
        audio = audio[: int(max_seconds * sr)]
    # normalize peak
    peak = float(np.max(np.abs(audio)) + 1e-9)
    gain = 10 ** (target_peak_dbfs / 20) / peak
    audio = np.clip(audio * gain, -1.0, 1.0).astype(np.float32)

    import soundfile as sf
    sf.write(out_wav, audio, sr)
    return out_wav


def _load_any(src: str, sr: int):
    """Decode `src` to a mono float32 array at `sr`."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg and not src.lower().endswith(".wav"):
        tmp = out_tmp = src + ".tmp.wav"
        subprocess.run([ffmpeg, "-y", "-i", src, "-ac", "1", "-ar", str(sr), tmp],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        try:
            import soundfile as sf
            a, _ = sf.read(tmp, dtype="float32", always_2d=False)
        finally:
            if os.path.exists(out_tmp):
                os.remove(out_tmp)
        return np.asarray(a, dtype=np.float32), sr
    # WAV path (or no ffmpeg): soundfile handles it, resample if needed
    import soundfile as sf
    a, in_sr = sf.read(src, dtype="float32", always_2d=False)
    if getattr(a, "ndim", 1) > 1:
        a = a.mean(axis=1)
    a = np.asarray(a, dtype=np.float32)
    if in_sr != sr:
        n = int(len(a) * sr / in_sr)
        a = np.interp(np.linspace(0, len(a), n, endpoint=False), np.arange(len(a)), a).astype(np.float32)
    return a, sr
