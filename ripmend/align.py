"""Piecewise time alignment of a source against the reference timeline.

An offset `off` means source sample `i - off` lines up with reference
sample `i`. Damaged rips skip and repeat audio, so the offset is tracked
block by block instead of being assumed constant.
"""

import numpy as np

BLOCK = 4096
HOP = 2048
SILENCE_RMS = 30.0  # int16 units; blocks quieter than this carry no timing info
MIN_CORR = 0.6


def _next_pow2(n):
    return 1 << int(np.ceil(np.log2(n)))


def _best_lag(y, x):
    """Lag k maximizing the normalized correlation of x against y[k:k+len(x)]."""
    n = _next_pow2(len(y) + len(x))
    c = np.fft.irfft(np.fft.rfft(y, n) * np.conj(np.fft.rfft(x, n)), n)[: len(y) - len(x) + 1]
    csum = np.concatenate(([0.0], np.cumsum(y.astype(np.float64) ** 2)))
    ey = np.sqrt(np.maximum(csum[len(x):] - csum[: -len(x)], 1e-9))
    nc = c / (ey * np.sqrt((x.astype(np.float64) ** 2).sum()) + 1e-9)
    k = int(np.argmax(nc))
    return k, float(nc[k])


def _block_error(ref, src, r, off, length):
    a = r - off
    if a < 0 or a + length > len(src):
        return np.inf
    d = src[a:a + length] - ref[r:r + length]
    return float((d * d).mean())


def track(ref, src, search=8192, wide_search=5 * 44100):
    """Estimate the offset for each analysis block.

    ref, src: (n, 2) float arrays. Returns an array of offsets per block,
    NaN where the block is silent or no confident match was found.
    """
    ref_m = ref.mean(1)
    src_m = src.mean(1)
    n_blocks = max(0, (len(ref_m) - BLOCK) // HOP + 1)
    offs = np.full(n_blocks, np.nan)
    cur = 0
    lost = 10**9  # start with a wide search
    for b in range(n_blocks):
        r = b * HOP
        x = ref_m[r:r + BLOCK]
        if np.sqrt((x * x).mean()) < SILENCE_RMS:
            continue
        span = search if lost < 4 else wide_search
        lo = max(0, r - cur - span)
        hi = min(len(src_m), r - cur + span + BLOCK)
        if hi - lo < BLOCK:
            lost += 1
            continue
        k, corr = _best_lag(src_m[lo:hi], x)
        if corr < MIN_CORR:
            lost += 1
            continue
        off = r - (lo + k)
        # Correlation against a lossy reference can land a sample off; settle
        # on the integer offset with the smallest error.
        errs = {o: _block_error(ref, src, r, o, BLOCK) for o in range(off - 2, off + 3)}
        off = min(errs, key=errs.get)
        offs[b] = off
        cur = off
        lost = 0
    return offs


def candidates_for(offs, pos):
    """Offsets from analysis blocks near reference sample `pos`."""
    first = max(0, (pos - BLOCK) // HOP)
    last = min(len(offs) - 1, pos // HOP + 1)
    found = offs[first:last + 1]
    return [int(o) for o in np.unique(found[~np.isnan(found)])]


def map_source(src, sub_offs, sub, length):
    """Resample `src` onto the reference timeline using per-sub-block offsets.

    Returns the mapped signal and a mask of samples that came from inside src.
    """
    idx = np.arange(length) - np.repeat(sub_offs, sub)[:length]
    valid = (idx >= 0) & (idx < len(src))
    mapped = np.zeros((length, 2), np.float32)
    mapped[valid] = src[idx[valid]]
    return mapped, valid
