"""Tests for narrate.audio.bind — stitching per-chapter WAVs into one file."""
import os

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")

from monograph.narrate.audio import bind, _ordered_paths


def _wav(path, sr, secs, freq=220.0):
    t = np.linspace(0, secs, int(sr * secs), endpoint=False)
    sf.write(str(path), (0.1 * np.sin(2 * np.pi * freq * t)).astype("float32"), sr)


def test_natural_order(tmp_path):
    # lexical sort would put ch_10 before ch_2; natural sort must not.
    for name in ("ch_1.wav", "ch_2.wav", "ch_10.wav"):
        _wav(tmp_path / name, 8000, 0.02)
    names = [os.path.basename(p) for p in _ordered_paths(str(tmp_path))]
    assert names == ["ch_1.wav", "ch_2.wav", "ch_10.wav"]


def test_bind_wav_duration_and_rate(tmp_path):
    sr = 8000
    _wav(tmp_path / "a_1.wav", sr, 0.20)
    _wav(tmp_path / "a_2.wav", sr, 0.30)
    out = tmp_path / "book.wav"
    bind(str(tmp_path), str(out), gap=0.10)
    data, osr = sf.read(str(out))
    assert osr == sr
    # 0.20 + one 0.10 gap + 0.30 = 0.60s (one gap, not two)
    assert abs(len(data) / sr - 0.60) < 0.02


def test_explicit_order_overrides(tmp_path):
    sr = 8000
    _wav(tmp_path / "first.wav", sr, 0.10)
    _wav(tmp_path / "second.wav", sr, 0.10)
    out = tmp_path / "book.wav"
    bind(None, str(out), gap=0.0,
         order=[str(tmp_path / "second.wav"), str(tmp_path / "first.wav")])
    data, osr = sf.read(str(out))
    assert abs(len(data) / sr - 0.20) < 0.02


def test_mismatched_rate_raises(tmp_path):
    _wav(tmp_path / "a_1.wav", 8000, 0.1)
    _wav(tmp_path / "a_2.wav", 16000, 0.1)
    with pytest.raises(ValueError):
        bind(str(tmp_path), str(tmp_path / "book.wav"))


def test_empty_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        bind(str(tmp_path), str(tmp_path / "book.wav"))


def test_bind_mp3(tmp_path):
    pytest.importorskip("lameenc")
    sr = 24000
    _wav(tmp_path / "a_1.wav", sr, 0.20)
    _wav(tmp_path / "a_2.wav", sr, 0.20)
    out = tmp_path / "book.mp3"
    bind(str(tmp_path), str(out), gap=0.05, bitrate=64)
    assert out.exists() and out.stat().st_size > 0
    with open(out, "rb") as f:
        head = f.read(3)
    # MP3 frame sync (0xFFEx) or an ID3 tag
    assert head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xfa") or head == b"ID3"
