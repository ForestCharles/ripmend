"""ripmend: repair damaged CD rips using other rips of the same disc."""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

from .audio import write_wav
from .repair import repair

AUDIO_EXTS = {".wav", ".flac", ".aif", ".aiff", ".ogg", ".opus", ".mp3", ".m4a", ".aac", ".wma", ".ape", ".wv"}
TRACK_RE = re.compile(r"(?<!\d)(\d{1,2})(?!\d)")


def track_number(path):
    m = TRACK_RE.search(os.path.splitext(os.path.basename(path))[0])
    return int(m.group(1)) if m else None


def find_tracks(directory, recursive):
    groups = defaultdict(list)
    walker = os.walk(directory) if recursive else [(directory, [], os.listdir(directory))]
    for root, _, files in walker:
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in AUDIO_EXTS:
                n = track_number(f)
                if n is not None:
                    groups[n].append(os.path.join(root, f))
    return dict(sorted(groups.items()))


def run_one(paths, out_path, args):
    samples, report = repair(paths, threshold_db=args.threshold, crossfade=args.crossfade, eq=not args.no_eq)
    write_wav(out_path, samples)
    report_path = os.path.splitext(out_path)[0] + ".report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  wrote {out_path}")
    for s in report["sources"]:
        line = f"    {s['share_of_output']:6.1%} from {os.path.basename(s['path'])}"
        if "bit_exact" in s:
            line += f" (verified good {s['verified_good']:.1%}, {s['timing_jumps']} timing jumps)"
        print(line)


def main(argv=None):
    p = argparse.ArgumentParser(prog="ripmend", description=__doc__)
    p.add_argument("--threshold", type=float, default=10.0,
                   help="dB a lossless rip must match the reference by to be kept (default 10)")
    p.add_argument("--crossfade", type=int, default=128, help="splice crossfade length in samples (default 128)")
    p.add_argument("--no-eq", action="store_true", help="don't EQ-match lossy fill to the lossless rip")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("repair", help="repair one track from several rips of it")
    r.add_argument("sources", nargs="+", help="rips of the same track (any mix of WAV/FLAC/OGG/MP3/...)")
    r.add_argument("-o", "--output", required=True, help="output WAV path")

    b = sub.add_parser("batch", help="repair every track in a folder, grouping files by track number")
    b.add_argument("input_dir")
    b.add_argument("output_dir")
    b.add_argument("-r", "--recursive", action="store_true", help="also look in subfolders")

    args = p.parse_args(argv)
    if args.cmd == "repair":
        run_one(args.sources, args.output, args)
        return

    os.makedirs(args.output_dir, exist_ok=True)
    failed = False
    for n, paths in find_tracks(args.input_dir, args.recursive).items():
        print(f"track {n:02d}: {len(paths)} rips")
        try:
            run_one(paths, os.path.join(args.output_dir, f"{n:02d}.wav"), args)
        except Exception as e:  # keep going with the other tracks
            print(f"  failed: {e}", file=sys.stderr)
            failed = True
    sys.exit(1 if failed else 0)
