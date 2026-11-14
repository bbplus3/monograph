"""Audio helpers shared by every narration backend: edge-trimming, compressing
the long internal silences TTS models emit, and reading/writing WAVs. These are
backend-agnostic and operate on plain float32 numpy arrays."""
from __future__ import annotations
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
