"""Recall/precision per fault type, broken down by hardening condition.

    python score_by_condition.py data/hard wideband
Imports match() from score.py and analyze() from callcheck; modifies neither.
"""
import json
import sys
from pathlib import Path

from callcheck.analyze import analyze, load
from callcheck.vad import SileroVAD
from score import match

TYPES = ["slow_response", "dead_air", "cutoff", "false_barge_in"]


def conditions(h):
    c = ["all"]
    if "noise" in h:
        c += [f"noise:{h['noise']['type']}", "noise:snr<10" if h["noise"]["snr_db"] < 10 else "noise:snr>=10"]
    else:
        c.append("no_noise")
    if "echo" in h: c.append("echo")
    if "quiet_caller_db" in h: c.append("quiet_caller")
    if "codec" in h: c.append("codec_opus")
    if "packet_loss_pct" in h: c.append("packet_loss")
    if "clipping" in h: c.append("clipping")
    if len(h) == 1: c.append("pristine")  # only "seed"
    return c


def main():
    root, band = Path(sys.argv[1]), sys.argv[2]
    vad = SileroVAD()
    st = {}
    ncalls = {}
    for tp_ in sorted((root / "truth").glob("*.json")):
        truth = json.loads(tp_.read_text())
        caller, agent = load(root / band / f"{truth['call_id']}.wav")
        res = analyze(caller, agent, vad=vad)
        for cond in conditions(truth.get("hardening", {"seed": 0})):
            ncalls[cond] = ncalls.get(cond, 0) + 1
            for t in TYPES:
                tp, miss, extra = match([e for e in truth["events"] if e["type"] == t],
                                        [e for e in res.events if e["type"] == t])
                s = st.setdefault((cond, t), [0, 0, 0])
                s[0] += len(tp); s[1] += len(miss); s[2] += len(extra)
    print(f"{band}: by condition (tp/fn/fp -> recall/precision)")
    print(f"{'condition':16} {'calls':>5} " + " ".join(f"{t:>22}" for t in TYPES))
    for cond in sorted(ncalls, key=lambda c: (c != "all", c)):
        cells = []
        for t in TYPES:
            tp, fn, fp = st[(cond, t)]
            r = f"{tp/(tp+fn):.0%}" if tp + fn else "-"
            p = f"{tp/(tp+fp):.0%}" if tp + fp else "-"
            cells.append(f"{tp}/{fn}/{fp} {r:>4}/{p:>4}".rjust(22))
        print(f"{cond:16} {ncalls[cond]:5} " + " ".join(cells))


if __name__ == "__main__":
    main()
