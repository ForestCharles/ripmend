"""Combine several rips of one track into the best-sounding version.

The highest-bitrate lossy rip defines the timeline and serves as the
reference that every other rip is checked against. Lossless rips are mapped
onto that timeline piece by piece and used wherever they agree with the
reference. Everything else is filled from the lossy rips, EQ-matched to the
lossless audio, with short crossfades at every splice.
"""

from dataclasses import dataclass, field

import numpy as np

from . import align, detect, eqmatch
from .audio import RATE, load

SUB = 256          # granularity of the keep/fill decision, in samples
NOISE_FLOOR = 64.0 # int16 units; errors below this always pass
SILENT_PEAK = 100  # sources quieter than this everywhere are failed extracts


@dataclass
class Layer:
    source: object
    mapped: np.ndarray = field(repr=False)
    good: np.ndarray = field(repr=False)
    sub_offs: np.ndarray = field(repr=False)
    concealed: np.ndarray = field(default=None, repr=False)
    eq: dict = None


def _assign(ref, src, offs, nsub):
    """Pick one offset per sub-block and measure how well it matches."""
    sub_offs = np.zeros(nsub, np.int64)
    err = np.full(nsub, np.inf)
    power = np.zeros(nsub)
    known = np.zeros(nsub, bool)
    prev = None
    for j in range(nsub):
        p = j * SUB
        n = min(SUB, len(ref) - p)
        cands = align.candidates_for(offs, p)
        if prev is not None and prev not in cands:
            cands.append(prev)
        if not cands:
            continue
        errs = [align._block_error(ref, src, p, o, n) for o in cands]
        best = int(np.argmin(errs))
        # Stay on the current offset unless another is clearly better, so
        # lossy jitter doesn't create needless splices.
        if prev is not None and errs[cands.index(prev)] <= 1.5 * errs[best]:
            best = cands.index(prev)
        sub_offs[j] = prev = cands[best]
        err[j] = errs[best]
        r = ref[p:p + n]
        power[j] = float((r * r).mean())
        known[j] = True
    return sub_offs, err, power, known


def _sample_mask(sub_mask, length):
    return np.repeat(sub_mask, SUB)[:length]


def _break_at_offset_changes(good, sub_offs, length):
    changes = np.where(np.diff(sub_offs) != 0)[0] + 1
    good[np.minimum(changes * SUB, length - 1)] = False


def _map(ref, src, nsub, length, threshold_db, check_ref=True):
    offs = align.track(ref, src.samples)
    sub_offs, err, power, known = _assign(ref, src.samples, offs, nsub)
    mapped, valid = align.map_source(src.samples, sub_offs, SUB, length)
    held = detect.held_mask(src.samples)
    idx = np.clip(np.arange(length) - np.repeat(sub_offs, SUB)[:length], 0, len(src.samples) - 1)
    good = valid & ~held[idx] & _sample_mask(known, length)
    if check_ref:
        ok = err <= power * 10 ** (-threshold_db / 10) + NOISE_FLOOR ** 2
        good &= _sample_mask(ok, length)
    _break_at_offset_changes(good, sub_offs, length)
    concealed = detect.concealed_mask(src.samples)[idx] & valid[:, None]
    return Layer(src, mapped, good, sub_offs, concealed)


def _patch(layer, fallback):
    """Replace samples the drive made up with the fallback's, anchored to real ones.

    The fallback (lossy) supplies the fine detail; its offset from the real
    samples on either side is interpolated across the patch, so the low end
    stays exactly as the lossless rip has it.
    """
    out = layer.mapped.copy()
    pos = np.arange(len(out))
    for ch in range(out.shape[1]):
        bad = layer.concealed[:, ch]
        real = ~bad
        if not bad.any() or not real.any():
            continue
        err = layer.mapped[real, ch] - fallback[real, ch]
        out[bad, ch] = fallback[bad, ch] + np.interp(pos[bad], pos[real], err)
    return out


def _moving_sum(x, size):
    """Sum of x over a centered window of `size` samples, clamped at the ends."""
    csum = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    i = np.arange(len(x))
    lo = np.clip(i - size // 2, 0, len(x))
    hi = np.clip(i - size // 2 + size, 0, len(x))
    return csum[hi] - csum[lo], hi - lo


def _soft(good, crossfade):
    """Turn a keep mask into weights that ramp down inside the kept region."""
    if crossfade <= 1:
        return good.astype(np.float32)
    width = 2 * (crossfade // 2 + 1) + 1
    kept, n = _moving_sum(good, width)
    core = (kept == n).astype(np.float32)
    total, n = _moving_sum(core, crossfade)
    w = (total / n).astype(np.float32)
    w[w > 1 - 1e-6] = 1.0
    w[w < 1e-6] = 0.0
    return w


def _ranges(mask, merge_gap=0.05):
    """Contiguous True regions of a per-sample mask, in seconds."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    out = []
    for s, e in zip(starts / RATE, ends / RATE):
        if out and s - out[-1][1] < merge_gap:
            out[-1][1] = e
        else:
            out.append([s, e])
    return [(round(s, 3), round(e, 3)) for s, e in out]


def repair(paths, threshold_db=10.0, crossfade=128, eq=True, log=print):
    """Build the repaired track from `paths`. Returns (samples, report)."""
    sources = []
    for p in paths:
        s = load(p)
        if len(s.samples) == 0 or np.abs(s.samples).max() < SILENT_PEAK:
            log(f"  skipping {s.name}: silent (failed extract?)")
            continue
        sources.append(s)
    lossy = sorted((s for s in sources if not s.lossless), key=lambda s: -s.bitrate)
    lossless = [s for s in sources if s.lossless]
    if not lossy:
        raise ValueError("need at least one lossy rip to check the others against")

    ref = lossy[0]
    length = len(ref.samples)
    nsub = -(-length // SUB)
    log(f"  reference timeline: {ref.name} ({length / RATE:.1f} s)")

    ref_layer = Layer(ref, ref.samples, ~detect.held_mask(ref.samples), np.zeros(nsub, np.int64))
    other_lossy = []
    for s in lossy[1:]:
        log(f"  aligning {s.name}")
        other_lossy.append(_map(ref.samples, s, nsub, length, threshold_db, check_ref=False))
    primaries = []
    for s in lossless:
        log(f"  aligning and checking {s.name}")
        primaries.append(_map(ref.samples, s, nsub, length, threshold_db))
    primaries.sort(key=lambda l: -l.good.mean())

    if eq and primaries:
        target = _patch(primaries[0], ref_layer.mapped)
        for layer in [ref_layer] + other_lossy:
            mask = primaries[0].good & layer.good
            fitted = eqmatch.fit(target, layer.mapped, mask)
            if fitted is None:
                continue
            h, before = fitted
            matched = eqmatch.apply(layer.mapped, h)
            after = eqmatch.fit(target, matched, mask)[1]
            if after < before - 0.05:
                layer.mapped = matched
                layer.eq = {"mismatch_before_db": round(before, 2), "mismatch_after_db": round(after, 2),
                            "response_db": eqmatch.response_db(h)}
                log(f"  EQ-matched {layer.source.name}: spectral mismatch {before:.2f} -> {after:.2f} dB")
        # Recheck the lossless rips against the matched reference: the
        # tighter match lets fewer good stretches be mistaken for damage.
        if ref_layer.eq:
            for i, layer in enumerate(primaries):
                primaries[i] = _map(ref_layer.mapped, layer.source, nsub, length, threshold_db)
            primaries.sort(key=lambda l: -l.good.mean())

    for layer in primaries:
        layer.mapped = _patch(layer, ref_layer.mapped)
        log(f"  replaced {layer.concealed[layer.good].mean():.1%} of samples in {layer.source.name}"
            " that the drive had concealed")

    layers = primaries + [ref_layer] + other_lossy
    out = np.zeros((length, 2), np.float32)
    remaining = np.ones(length, np.float32)
    used = []
    for layer in layers:
        take = remaining * _soft(layer.good, crossfade)
        out += take[:, None] * layer.mapped
        remaining -= take
        used.append(take)
    out += remaining[:, None] * ref_layer.mapped
    used[len(primaries)] = used[len(primaries)] + remaining

    report = {"reference": ref.path, "duration_s": round(length / RATE, 3), "sources": []}
    for layer, take in zip(layers, used):
        entry = {
            "path": layer.source.path,
            "codec": layer.source.codec,
            "share_of_output": round(float(take.mean()), 4),
            "verified_good": round(float(layer.good.mean()), 4),
        }
        if layer.source.lossless:
            entry["timing_jumps"] = int((np.diff(layer.sub_offs) != 0).sum())
            entry["concealed_replaced"] = round(float(layer.concealed[layer.good].mean()), 4)
            entry["bit_exact"] = round(float(((take == 1) & ~layer.concealed.any(1)).mean()), 4)
        if layer.eq:
            entry["eq"] = layer.eq
        report["sources"].append(entry)
    if primaries:
        lossless_share = sum(used[:len(primaries)])
        report["filled_from_lossy"] = _ranges(lossless_share < 1)
    return out, report
