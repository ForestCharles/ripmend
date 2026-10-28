"""Detection of drive error concealment in a rip."""

import numpy as np


def held_mask(x, min_run=16, min_level=64, margin=32):
    """Mark samples where a channel is frozen at one nonzero value.

    When a drive can't read a frame it often repeats the last good sample,
    which shows up as a flat line in one or both channels. Real audio almost
    never holds a nonzero value for this long. Returns a bool mask over x.
    """
    bad = np.zeros(len(x), bool)
    for ch in range(x.shape[1]):
        v = x[:, ch]
        same = np.concatenate(([0], (v[1:] == v[:-1]).astype(np.int8), [0]))
        edges = np.diff(same)
        # A run of equal samples covers v[start] .. v[end] inclusive.
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        lengths = ends - starts + 1
        keep = (lengths >= min_run) & (np.abs(v[starts]) >= min_level)
        for s, e in zip(starts[keep], ends[keep]):
            bad[max(0, s - margin):e + 1 + margin] = True
    return bad


def concealed_mask(x, min_level=64):
    """Mark samples the drive most likely made up rather than read.

    On an unreadable sample a drive repeats the previous value or puts the
    average of its neighbours there. On a failing disc this happens to a
    large share of samples, scattered everywhere, and sounds like bursts of
    hiss. Returns an (n, channels) bool mask. A real sample matches by
    chance a few percent of the time; replacing those costs very little.
    """
    v = x.astype(np.int64)
    loud = np.abs(v) >= min_level
    mask = np.zeros(v.shape, bool)
    mask[1:] |= (v[1:] == v[:-1]) & loud[1:]
    mid = v[:-2] + v[2:]
    mask[1:-1] |= (np.abs(2 * v[1:-1] - mid) <= 1) & (v[:-2] != v[2:]) & loud[1:-1]
    return mask
