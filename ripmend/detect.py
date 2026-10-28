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
