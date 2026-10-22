"""Score callcheck against the synthetic ground truth.

An event counts as found when the analyzer reports the same type within
TOL_S of the planted time. Everything else it reports is a false alarm.

    python score.py data/synthetic wideband
    python score.py data/synthetic phone
"""

import json
import sys
from pathlib import Path

from callcheck.analyze import analyze, load
from callcheck.vad import SileroVAD

TOL_S = 0.5
TYPES = ["slow_response", "dead_air", "cutoff", "false_barge_in", "wrong_readback"]


def match(truth_events, found_events):
    tp, used = [], set()
    for t in truth_events:
        for i, f in enumerate(found_events):
            if i not in used and f["type"] == t["type"] and abs(f["time"] - t["time"]) <= TOL_S:
                tp.append((t, f))
                used.add(i)
                break
    missed = [t for t in truth_events if not any(t is x for x, _ in tp)]
    extra = [f for i, f in enumerate(found_events) if i not in used]
    return tp, missed, extra


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic")
    band = sys.argv[2] if len(sys.argv) > 2 else "wideband"
    only = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else set(TYPES) - {"wrong_readback"}
    vad = SileroVAD()
    stats = {t: {"tp": 0, "fn": 0, "fp": 0} for t in only}
    lat_err = []
    for truth_path in sorted((root / "truth").glob("*.json")):
        truth = json.loads(truth_path.read_text())
        name = truth["call_id"]
        caller, agent = load(root / band / f"{name}.wav")
        res = analyze(caller, agent, vad=vad)
        t_ev = [e for e in truth["events"] if e["type"] in only]
        f_ev = [e for e in res.events if e["type"] in only]
        tp, missed, extra = match(t_ev, f_ev)
        for t, _ in tp:
            stats[t["type"]]["tp"] += 1
        for t in missed:
            stats[t["type"]]["fn"] += 1
        for f in extra:
            stats[f["type"]]["fp"] += 1
        # Latency accuracy on every turn the truth has a latency for.
        for item in truth["agent_latencies"]:
            start = truth["turns"][item["turn"]]["start"]
            near = [x for x in res.latencies if abs(x["agent_start"] - start) <= TOL_S]
            if near:
                lat_err.append(near[0]["latency_s"] - item["latency_s"])
        note = "".join(
            [f"  MISSED {m['type']}@{m['time']}" for m in missed] + [f"  EXTRA {x['type']}@{x['time']}" for x in extra])
        print(f"{name}{note or '  ok'}")

    print(f"\n{band}: per fault type (tolerance {TOL_S}s)")
    print(f"{'type':16} {'found':>7} {'missed':>7} {'false':>7} {'recall':>7} {'precision':>9}")
    for t in TYPES:
        if t not in stats:
            continue
        s = stats[t]
        rec = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else float("nan")
        pre = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else float("nan")
        print(f"{t:16} {s['tp']:7} {s['fn']:7} {s['fp']:7} {rec:7.0%} {pre:9.0%}")
    if lat_err:
        import numpy as np
        e = np.array(lat_err)
        print(f"latency: {len(e)} turns matched, mean error {e.mean()*1000:+.0f} ms, "
              f"median |error| {np.median(abs(e))*1000:.0f} ms, max |error| {abs(e).max()*1000:.0f} ms")


if __name__ == "__main__":
    main()
