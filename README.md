# ripmend

Repair a damaged CD rip by combining it with other rips of the same disc.

You have a lossless rip (WAV/FLAC) that came off a failing disc with skips and
dropouts, plus one or more lossy rips (OGG, MP3, ...) that are complete but
lower quality. ripmend keeps every stretch of the lossless rip that it can
verify, and fills the rest from the lossy rips, so the result is mostly
bit-exact lossless audio with no gaps.

## How it works

1. **Decode everything** with ffmpeg to 44.1 kHz stereo.
2. **Pick a reference.** The highest-bitrate lossy rip defines the output
   timeline and is what everything else is checked against. Files that are
   silent throughout (failed extracts) are skipped.
3. **Re-align piece by piece.** Damaged rips don't just have holes: the drive
   drops a few milliseconds here and there and sometimes repeats a stretch,
   so the rip drifts out of sync. ripmend tracks the offset every 46 ms and
   follows each jump.
4. **Verify.** Each 6 ms chunk of the lossless rip is kept only if it matches
   the reference to within lossy-codec noise (10 dB by default) and isn't a
   drive's error concealment (a channel frozen on one value).
5. **Learn the difference.** On the stretches where both are known good,
   ripmend compares the spectra of the lossy and lossless rips and builds an
   EQ curve that makes the lossy fill match the lossless tonal balance and
   level.
6. **Splice.** Lossless where verified, otherwise the EQ-matched reference,
   otherwise the next lossy rip, with a 3 ms crossfade at each splice.

## Install

Needs Python 3.9+ and ffmpeg on your PATH.

```
git clone https://github.com/ForestCharles/ripmend.git
cd ripmend
pip install .
```

## Use

Repair a whole folder. Files are grouped by the track number in their names
(`02 Track02.wav`, `Artist - 02 - Title.ogg` and `Track  2.mp3` all count as
track 2):

```
ripmend batch path/to/rips path/to/output
ripmend batch -r path/to/rips path/to/output   # also look in subfolders
```

Or repair one track from files you name:

```
ripmend repair "02 Track02.wav" "Artist - 02 - Title.ogg" "Track 2.mp3" -o 02-fixed.wav
```

Each output WAV gets a `.report.json` next to it listing how much came from
each rip, how many timing jumps were found, what share is bit-exact lossless,
the EQ that was applied, and every time range that was filled from lossy audio.

Options (put them before `repair`/`batch`):

- `--threshold DB`: how closely lossless audio must match the reference to be
  kept. Lower keeps more of the lossless rip but risks keeping damage.
- `--crossfade N`: splice crossfade in samples (default 128).
- `--no-eq`: use the lossy fill as is.

## Limits

- At least one lossy rip is required; it is what the lossless rip is checked
  against. The output follows that rip's timing and length.
- If every rip is damaged in the same spot, the reference is used there anyway.
- All rips must be of the same master. A remaster or different edit won't
  line up and will simply be ignored where it doesn't match.

## Tests

```
pip install '.[test]'
pytest
```
