"""Learn how a lossy rip's tonal balance differs from the lossless one, and undo it.

On stretches where both rips are known good, compare their average spectra
and build a smooth zero-phase EQ curve that makes the lossy audio's spectrum
match the lossless audio's. This captures level and EQ tilt (lossy encoders
often shift the top end). It deliberately matches magnitude only: a
least-squares waveform fit would just dull the treble, because codec noise up
there doesn't correlate with the original.
"""

import numpy as np

FFT = 4096
TAPS = 1024
MAX_FRAMES = 3000
MAX_BOOST_DB = 6.0
MAX_CUT_DB = 12.0
SMOOTH_OCT = 1 / 6
RATE = 44100


def _octave_smooth(power):
    freqs = np.fft.rfftfreq(FFT, 1 / RATE)
    csum = np.concatenate(([0.0], np.cumsum(power)))
    half = 2 ** (SMOOTH_OCT / 2)
    lo = np.searchsorted(freqs, freqs / half)
    hi = np.maximum(np.searchsorted(freqs, freqs * half, side="right"), lo + 1)
    return (csum[hi] - csum[lo]) / (hi - lo)


def fit(target, fill, mask):
    """Return (FIR taps, spectral mismatch in dB) mapping `fill` toward `target`.

    Returns None if the two don't share enough known-good audio.
    """
    hop = FFT // 2
    bad = np.concatenate(([0], np.cumsum(~mask)))
    starts = [s for s in range(0, len(mask) - FFT, hop) if bad[s + FFT] == bad[s]]
    if len(starts) < 20:
        return None
    if len(starts) > MAX_FRAMES:
        starts = [starts[i] for i in np.linspace(0, len(starts) - 1, MAX_FRAMES).astype(int)]
    win = np.hanning(FFT)[:, None]
    sxx = np.zeros(FFT // 2 + 1)
    syy = np.zeros(FFT // 2 + 1)
    for s in starts:
        sxx += (np.abs(np.fft.rfft(target[s:s + FFT] * win, axis=0)) ** 2).sum(1)
        syy += (np.abs(np.fft.rfft(fill[s:s + FFT] * win, axis=0)) ** 2).sum(1)
    sxx, syy = _octave_smooth(sxx), _octave_smooth(syy)
    floor = 1e-9 * max(sxx.max(), syy.max())
    gain_db = 10 * np.log10((sxx + floor) / (syy + floor))
    # Above a lossy encoder's lowpass there is nothing to restore; boosting
    # would only amplify codec noise, so leave those bins alone.
    gain_db[gain_db > 30] = 0.0
    gain_db = np.clip(gain_db, -MAX_CUT_DB, MAX_BOOST_DB)
    h = np.roll(np.fft.irfft(10 ** (gain_db / 20), FFT), TAPS // 2)[:TAPS] * np.hanning(TAPS)
    return h, _band_mismatch(sxx, syy)


def _band_mismatch(sxx, syy):
    """RMS level difference over audible third-octave bands, in dB."""
    freqs = np.fft.rfftfreq(FFT, 1 / RATE)
    centers = 1000 * 2 ** (np.arange(-17, 13) / 3)  # 20 Hz .. 16 kHz
    diffs = []
    for c in centers:
        k = (freqs >= c / 2 ** (1 / 6)) & (freqs < c * 2 ** (1 / 6))
        if k.any():
            diffs.append(10 * np.log10(sxx[k].sum() / syy[k].sum()))
    return float(np.sqrt(np.mean(np.square(diffs))))


def apply(fill, h, block=1 << 16):
    """Filter `fill` with zero-phase taps `h` (overlap-add, compensating the delay)."""
    n = 1 << int(np.ceil(np.log2(block + TAPS - 1)))
    hf = np.fft.rfft(h, n)[:, None]
    out = np.zeros((len(fill) + n, fill.shape[1]))
    for s in range(0, len(fill), block):
        seg = np.fft.irfft(np.fft.rfft(fill[s:s + block], n, axis=0) * hf, n, axis=0)
        out[s:s + n] += seg
    return out[TAPS // 2:TAPS // 2 + len(fill)].astype(np.float32)


def response_db(h, bands=((20, 200), (200, 2000), (2000, 8000), (8000, 14000), (14000, 20000))):
    """Average gain of the filter in a few bands, for the report."""
    spec = np.abs(np.fft.rfft(h, 8192)) ** 2
    freqs = np.fft.rfftfreq(8192, 1 / RATE)
    return {f"{lo}-{hi} Hz": round(float(10 * np.log10(spec[(freqs >= lo) & (freqs < hi)].mean() + 1e-12)), 2)
            for lo, hi in bands}
