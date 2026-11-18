"""Audio helpers shared by every narration backend: edge-trimming, compressing
the long internal silences TTS models emit, and reading/writing WAVs. These are
backend-agnostic and operate on plain float32 numpy arrays."""
from __future__ import annotations
import glob
import os
import re

import numpy as np


def trim_silence(a: np.ndarray, sr: int, thr: float = 0.015, pad: float = 0.03) -> np.ndarray:
    """Trim near-silence from both ends so segments butt together tightly."""
    idx = np.where(np.abs(a) > thr)[0]
    if len(idx) == 0:
        return a
    p = int(pad * sr)
    return a[max(0, idx[0] - p): min(len(a), idx[-1] + p)]


def compress_silences(a: np.ndarray, sr: int, max_sil: float = 0.32,
                      keep: float = 0.18, thr: float = 0.025) -> np.ndarray:
    """Crush any internal silence/breath longer than `max_sil` down to `keep`
    seconds. Removes the long mid-sentence breathing gaps TTS models produce."""
    win = int(0.02 * sr)
    n = len(a) // win
    if n == 0:
        return a
    e = np.sqrt(np.mean(a[:n * win].reshape(n, win) ** 2, axis=1)) + 1e-9
    silent = e < thr
    keepn = int(keep * sr)
    out, i = [], 0
    while i < n:
        j = i
        while j < n and silent[j] == silent[i]:
            j += 1
        seg = a[i * win:j * win]
        out.append(np.zeros(keepn, dtype=np.float32) if (silent[i] and len(seg) > max_sil * sr) else seg)
        i = j
    if n * win < len(a):
        out.append(a[n * win:])
    return np.concatenate(out) if out else a


def silence(seconds: float, sr: int) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


def write_wav(path: str, audio: np.ndarray, sr: int) -> None:
    import soundfile as sf
    sf.write(path, np.asarray(audio, dtype=np.float32), sr)


def read_wav(path: str):
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)            # downmix to mono
    return np.asarray(data, dtype=np.float32), sr


def _natkey(s: str):
    """Sort key so chapter_2 orders before chapter_10, not after it."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _ordered_paths(audio, order=None):
    if order is not None:
        return list(order)
    paths = list(audio) if isinstance(audio, (list, tuple)) else glob.glob(
        os.path.join(audio, "*.wav"))
    return sorted(paths, key=lambda p: _natkey(os.path.basename(p)))


def _to_pcm16(a: np.ndarray) -> bytes:
    return (np.clip(a, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _open_mp3(path: str, sr: int, bitrate: int):
    try:
        import lameenc
    except ImportError:
        raise SystemExit("MP3 output needs lameenc (no ffmpeg required).  "
                         "Install it with:  pip install 'monograph[mp3]'")
    enc = lameenc.Encoder()
    enc.set_bit_rate(bitrate)
    enc.set_in_sample_rate(sr)
    enc.set_channels(1)
    enc.set_quality(2)                       # 2 = high quality
    return enc, open(path, "wb")


def bind(audio, out: str, gap: float = 1.0, bitrate: int = 64, order=None) -> str:
    """Stitch per-chapter WAVs into one listenable file — the end of `narrate`.

    `audio`   a folder of .wav files, or an explicit list of wav paths.
    `out`     output path; ending in .mp3 encodes via lameenc (no ffmpeg),
              anything else writes a lossless WAV.
    `gap`     seconds of silence inserted between chapters.
    `order`   optional explicit list of paths; otherwise files are joined in
              natural-sorted filename order (chapter_2 before chapter_10).

    Streams chapter-by-chapter, so peak memory stays near one chapter even for a
    full-length book. Returns the output path.
    """
    paths = _ordered_paths(audio, order)
    if not paths:
        raise FileNotFoundError(f"No .wav files to bind in {audio!r}")

    is_mp3 = out.lower().endswith(".mp3")
    enc = mp3_fh = sf_out = sr = None
    try:
        for i, p in enumerate(paths):
            a, s = read_wav(p)
            if sr is None:
                sr = s
                if is_mp3:
                    enc, mp3_fh = _open_mp3(out, sr, bitrate)
                else:
                    import soundfile as sf
                    sf_out = sf.SoundFile(out, "w", samplerate=sr,
                                          channels=1, subtype="PCM_16")
            elif s != sr:
                raise ValueError(f"{os.path.basename(p)} is {s} Hz but the book is "
                                 f"{sr} Hz — every chapter must share one sample rate.")
            block = a
            if gap and i != len(paths) - 1:
                block = np.concatenate([a, silence(gap, sr)])
            if is_mp3:
                mp3_fh.write(enc.encode(_to_pcm16(block)))
            else:
                sf_out.write(block)
        if is_mp3 and enc is not None:
            mp3_fh.write(enc.flush())
    finally:
        if sf_out is not None:
            sf_out.close()
        if mp3_fh is not None:
            mp3_fh.close()
    return out
