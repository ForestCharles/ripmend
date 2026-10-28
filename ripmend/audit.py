"""Sweep a repaired track for glitches by comparing it with a lossy rip.

Everything here is relative to the reference: a glitch that the lossy rip
shares (a drumstick click, say) cancels out, and nothing can be found where
the lossy rip is itself wrong. What remains are moments where the repaired
track departs from the reference far more than codec noise does.
"""

import numpy as np

from . import align, detect, eqmatch
from .audio import RATE

WINDOW = 220          # 5 ms
CONTEXT = RATE        # local baseline: +/- 1 s
DIP_DB = 10.0         # a window this much worse than its surroundings is suspect
SPIKE_K = 8.0         # a sample this many times the local residual RMS is suspect
SPIKE_MIN = 1000      # int16 units
SPLICE_SPAN = 441     # 10 ms each side of a splice


def _global_offset(ref, out):
    """Offset of `out` against `ref`, from a confident block near the middle."""
    offs = align.track(ref, out)
    found = offs[~np.isnan(offs)]
    return int(np.median(found)) if len(found) else 0


def _running_median(x, half):
    """Median over +/- `half` entries, computed on a coarse grid for speed."""
    step = max(1, half // 8)
    centers = np.arange(0, len(x), step)
    meds = np.array([np.median(x[max(0, c - half):c + half + 1]) for c in centers])
    return np.interp(np.arange(len(x)), centers, meds)


def _residual_dips(out, ref):
    n = len(out) // WINDOW * WINDOW
    err = ((out[:n] - ref[:n]) ** 2).reshape(-1, WINDOW, 2).mean((1, 2))
    power = (ref[:n] ** 2).reshape(-1, WINDOW, 2).mean((1, 2))
    snr = 10 * np.log10((power + 1) / (err + 1))
    loud = power > 100 ** 2
    dip = _running_median(snr, CONTEXT // WINDOW) - snr
    hits = np.where(loud & (dip > DIP_DB))[0]
    return [(int(i * WINDOW), float(dip[i])) for i in hits]


def _spikes(out, ref):
    hits = []
    for ch in range(out.shape[1]):
        r = (out[:, ch] - ref[:, ch]).astype(np.float64)
        dev = r - np.convolve(r, np.ones(9) / 9, "same")
        rms = np.sqrt(np.convolve(dev * dev, np.ones(441) / 441, "same")) + 1
        idx = np.where((np.abs(dev) > SPIKE_K * rms) & (np.abs(dev) > SPIKE_MIN))[0]
        hits += [(int(i), float(abs(dev[i]))) for i in idx]
    return hits


def _cluster(events, gap):
    """Merge events closer than `gap` samples, keeping the strongest score."""
    merged = []
    for pos, score in sorted(events):
        if merged and pos - merged[-1][1] < gap:
            merged[-1][1] = pos
            merged[-1][2] = max(merged[-1][2], score)
            merged[-1][3] += 1
        else:
            merged.append([pos, pos, score, 1])
    return merged


def _treble_pulsing(out, ref, block=441):
    """Look for a few-per-second pulse in the treble that the reference doesn't have."""
    n = len(out) // block * block

    def treble(x):
        spec = np.fft.rfft(x[:n].mean(1).reshape(-1, block) * np.hanning(block), axis=1)
        freqs = np.fft.rfftfreq(block, 1 / RATE)
        return (np.abs(spec[:, freqs >= 8000]) ** 2).sum(1) + 1

    ratio = 10 * np.log10(treble(out) / treble(ref))
    ratio -= np.convolve(ratio, np.ones(200) / 200, "same")
    spec = np.abs(np.fft.rfft(ratio * np.hanning(len(ratio)))) ** 2
    freqs = np.fft.rfftfreq(len(ratio), block / RATE)
    band = (freqs >= 1) & (freqs <= 12)
    peak = int(np.argmax(spec[band]))
    return {
        "peak_hz": round(float(freqs[band][peak]), 2),
        "peak_over_median": round(float(spec[band][peak] / np.median(spec[band])), 1),
        "swing_db": round(float(np.std(ratio)), 2),
    }


def _splices(out, ref, report):
    if not report or "filled_from_lossy" not in report:
        return None
    err = (out - ref) ** 2
    power = ref ** 2
    jumps = []
    for start, end in report["filled_from_lossy"]:
        for t in (start, end):
            i = int(t * RATE)
            if SPLICE_SPAN <= i < len(out) - SPLICE_SPAN:
                e = err[i - SPLICE_SPAN:i + SPLICE_SPAN].mean()
                p = power[i - SPLICE_SPAN:i + SPLICE_SPAN].mean()
                jumps.append(10 * np.log10((p + 1) / (e + 1)))
    if not jumps:
        return None
    jumps = np.array(jumps)
    return {"count": len(jumps), "median_snr_db": round(float(np.median(jumps)), 1),
            "worst_snr_db": round(float(jumps.min()), 1)}


def _onto(out, other):
    """Line `other` up with `out` and EQ-match it, so only real differences remain."""
    off = _global_offset(out, other)
    other = other[max(0, -off):] if off < 0 else np.concatenate([np.zeros((off, 2), np.float32), other])
    other = np.concatenate([other[:len(out)], np.zeros((max(0, len(out) - len(other)), 2), np.float32)])
    fitted = eqmatch.fit(out, other, np.ones(len(out), bool))
    return eqmatch.apply(other, fitted[0]) if fitted is not None else other, off


def _verdict(out, ref, witness, start, end):
    """Is the repaired track the odd one out here, with both lossy rips agreeing?

    Checked over the suspect itself and over 20 ms around it, so a single bad
    sample isn't drowned out by the witness's own codec noise.
    """
    for lo, hi in ((start - 8, end + 8), (start - 441, end + 441)):
        s = slice(max(0, lo), hi)
        err = lambda a, b: float(((a[s] - b[s]) ** 2).mean()) + 1
        ours_ref, ours_wit, ref_wit = err(out, ref), err(out, witness), err(ref, witness)
        if ours_ref > 2 * ref_wit and ours_wit > 2 * ref_wit:
            return "repaired track differs from both lossy rips"
    if ref_wit > 2 * ours_wit:
        return "reference is the odd one out"
    return "unclear"


def audit(out, ref, report=None, top=20, witness=None):
    """Return (findings dict, suspect positions in samples, reference offset).

    `witness` is an optional second lossy rip (e.g. the MP3) used to tell
    glitches in the repaired track apart from quirks of the reference.
    """
    ref, off = _onto(out, ref)

    dips = _residual_dips(out, ref)
    spikes = _spikes(out, ref)
    # Scale both kinds of evidence to "dB above normal" so they rank together.
    events = dips + [(p, 20 * np.log10(s / SPIKE_MIN) + DIP_DB) for p, s in spikes]
    suspects = sorted(_cluster(events, RATE // 20), key=lambda c: -c[2])
    if witness is not None:
        witness, _ = _onto(out, witness)
        verdicts = [_verdict(out, ref, witness, a, b) for a, b, *_ in suspects]
        # Show the ones both lossy rips confirm first.
        order = sorted(range(len(suspects)), key=lambda i: (not verdicts[i].startswith("repaired"), -suspects[i][2]))
        suspects = [suspects[i] + [verdicts[i]] for i in order]

    concealed_out = float(detect.concealed_mask(out).mean())
    concealed_ref = float(detect.concealed_mask(np.round(ref)).mean())
    findings = {
        "offset_vs_reference": off,
        "suspects": [dict({"time_s": round(c[0] / RATE, 3), "length_ms": round((c[1] - c[0]) / RATE * 1000 + 5, 1),
                           "score_db": round(float(c[2]), 1), "events": c[3]},
                          **({"verdict": c[4]} if len(c) > 4 else {}))
                     for c in suspects[:top]],
        "suspect_count": len(suspects),
        "spike_samples": len(spikes),
        "residual_dips": len(dips),
        "treble_pulsing": _treble_pulsing(out, ref),
        # Two lossy rips pulse against each other at the music's beat too, so
        # this is the baseline the line above should be judged against.
        "treble_pulsing_witness_vs_reference": _treble_pulsing(witness, ref) if witness is not None else None,
        "concealment_signatures": {"output": round(concealed_out, 4), "reference_chance": round(concealed_ref, 4)},
        "splices": _splices(out, ref, report),
    }
    if witness is not None:
        findings["confirmed_by_witness"] = sum(c[4].startswith("repaired") for c in suspects)
    return findings, [c[0] for c in suspects[:top]], off


def suspects_reel(out, ref, positions, off, pad=0.75, gap=0.3):
    """For each suspect: the reference, a short gap, then the repaired track."""
    ref = ref[max(0, -off):] if off < 0 else np.concatenate([np.zeros((off, 2), np.float32), ref])
    silence = np.zeros((int(gap * RATE), 2), np.float32)
    pieces = []
    for p in sorted(positions):
        a, b = max(0, p - int(pad * RATE)), p + int(pad * RATE)
        pieces += [ref[a:b], silence, out[a:b], silence, silence]
    return np.concatenate(pieces) if pieces else silence
