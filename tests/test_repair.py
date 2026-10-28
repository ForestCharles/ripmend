"""End-to-end check on a synthetic track with the kinds of damage seen in real rips."""

import subprocess

import numpy as np
import pytest

from ripmend.audio import RATE, decode, write_wav
from ripmend.repair import repair


def _music(seconds, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    x = np.zeros((len(t), 2))
    for _ in range(40):
        f = rng.uniform(80, 6000)
        env = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.1, 2) * t + rng.uniform(0, 6))
        x += env[:, None] * np.sin(2 * np.pi * f * t[:, None] + rng.uniform(0, 6, 2)) * rng.uniform(100, 600)
    return np.round(x).astype(np.int16)


def _damage(clean, rng):
    """Drop short runs of samples, repeat one stretch, and freeze a channel, like a failing drive."""
    pieces, pos = [], 0
    while pos < len(clean):
        step = int(rng.uniform(0.6, 1.4) * RATE)
        pieces.append(clean[pos:pos + step].copy())
        pos += step + int(rng.integers(110, 140))
    damaged = np.concatenate(pieces)
    damaged = np.concatenate([damaged[:3 * RATE], damaged[3 * RATE - 5000:]])  # repeat 5000 samples
    for start in rng.integers(RATE, len(damaged) - RATE, 6):
        damaged[start:start + 600, 0] = damaged[start, 0]
    # Rippers pad or trim to the expected track length.
    return np.concatenate([damaged, np.zeros((len(clean), 2), np.int16)])[:len(clean)]


@pytest.fixture(scope="module")
def rips(tmp_path_factory):
    d = tmp_path_factory.mktemp("rips")
    clean = _music(12)
    write_wav(str(d / "clean.wav"), clean)
    write_wav(str(d / "damaged.wav"), _damage(clean, np.random.default_rng(1)))
    subprocess.run(["ffmpeg", "-v", "error", "-i", d / "clean.wav", "-c:a", "libvorbis", "-q:a", "5",
                    d / "lossy.ogg"], check=True)
    return d, clean


def test_repair_matches_clean(rips):
    d, clean = rips
    out, report = repair([str(d / "damaged.wav"), str(d / "lossy.ogg")], log=lambda *_: None)
    # The output follows the lossy rip's timeline; find where the clean audio sits on it.
    ogg = decode(str(d / "lossy.ogg"))
    win = ogg[RATE:RATE + 4096, 0]
    shift = min(range(-2000, 2001), key=lambda s: ((clean[RATE - s:RATE - s + 4096, 0] - win) ** 2).sum())
    n = min(len(out), len(clean)) - abs(shift) - 2 * RATE
    a = out[RATE:RATE + n].astype(np.float64)
    b = clean[RATE - shift:RATE - shift + n].astype(np.float64)
    snr = 10 * np.log10((b ** 2).sum() / ((a - b) ** 2).sum())
    wav = report["sources"][0]
    assert wav["path"].endswith("damaged.wav")
    assert wav["timing_jumps"] >= 8
    assert 0.85 < wav["share_of_output"] < 0.999
    assert snr > 30, snr
    # Most of the output should be the lossless samples, untouched.
    assert (a == b).all(axis=1).mean() > 0.85


def test_silent_rip_is_ignored(rips, tmp_path):
    d, clean = rips
    write_wav(str(tmp_path / "silent.wav"), np.zeros_like(clean))
    _, report = repair([str(tmp_path / "silent.wav"), str(d / "lossy.ogg")], log=lambda *_: None)
    assert [s["path"] for s in report["sources"]] == [str(d / "lossy.ogg")]


def _conceal(clean, rng):
    """Bursts where every other sample is made up by the drive (held or averaged)."""
    v = clean.astype(np.int64)
    for ch in range(2):
        for start in rng.integers(1, len(v) - 3000, 60):
            for i in range(start, start + int(rng.integers(500, 3000)), 2):
                v[i, ch] = v[i - 1, ch] if rng.random() < 0.5 else (v[i - 1, ch] + v[i + 1, ch]) // 2
    return v.astype(np.int16)


def test_concealed_samples_are_replaced(rips, tmp_path):
    d, clean = rips
    concealed = _conceal(clean, np.random.default_rng(2))
    write_wav(str(tmp_path / "concealed.wav"), concealed)
    out, report = repair([str(tmp_path / "concealed.wav"), str(d / "lossy.ogg")], log=lambda *_: None)
    n = min(len(out), len(clean))
    # High-frequency error is what the made-up samples sound like (hiss bursts).
    hf_err = lambda x: (np.diff(x[:n].astype(np.float64) - clean[:n], axis=0) ** 2).sum()
    assert report["sources"][0]["concealed_replaced"] > 0.05
    assert hf_err(out) < hf_err(concealed) / 4
