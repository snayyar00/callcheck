"""Usage: report_cli.py <dir_or_files...> --out report.html [--target-latency 0.4]"""

import argparse
import sys
from pathlib import Path

from callcheck.report import analyze_file, write_report
from callcheck.vad import SileroVAD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="report.html")
    ap.add_argument("--target-latency", type=float, default=0.4, help="target response latency in seconds")
    ap.add_argument("--title")
    args = ap.parse_args()
    paths = []
    for p in map(Path, args.inputs):
        paths += sorted(p.glob("*.wav")) if p.is_dir() else [p]
    if not paths:
        sys.exit("no WAV files found")
    vad = SileroVAD()
    calls = []
    for i, p in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] {p}", file=sys.stderr)
        calls.append(analyze_file(p, vad))
    title = args.title or f"callcheck report: {Path(args.inputs[0]).name}"
    write_report(calls, args.out, title, args.target_latency)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
