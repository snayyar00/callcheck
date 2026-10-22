"""Score phone-number read-back checks against the synthetic ground truth.

    python score_entities.py data/synthetic wideband
    python score_entities.py data/heldout phone
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from callcheck.analyze import load
from callcheck.entities import DIGITS, extract
from callcheck.stt import transcribe
from callcheck.vad import SileroVAD
from score import TOL_S, match


def digits_of(text):
    """Digits in a truth turn: digit words in order."""
    out = ""
    for w in text.lower().replace(",", " ").replace(".", " ").replace("?", " ").split():
        out += DIGITS.get(w, "")
    return out


def truth_time_to_info(truth):
    """End of the caller's turn right after the first correct agent read-back."""
    said = truth["entities"]["phone_said"]
    turns = truth["turns"]
    for i, t in enumerate(turns):
        if t["speaker"] == "agent" and digits_of(t["text"]) == said:
            for u in turns[i + 1:]:
                if u["speaker"] == "caller":
                    return u["end"]
    return None


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic")
    band = sys.argv[2] if len(sys.argv) > 2 else "wideband"
    t0 = time.time()
    vad = SileroVAD()
    tp = fn = fp = 0
    tti_err = []
    tti_missing = 0
    for truth_path in sorted((root / "truth").glob("*.json")):
        truth = json.loads(truth_path.read_text())
        name = truth["call_id"]
        caller, agent = load(root / band / f"{name}.wav")
        res = extract(transcribe(caller), transcribe(agent), vad.segments(caller), vad.segments(agent))
        t_ev = [e for e in truth["events"] if e["type"] == "wrong_readback"]
        m, missed, extra = match(t_ev, res["events"])
        tp += len(m)
        fn += len(missed)
        fp += len(extra)
        notes = [f"  MISSED wrong_readback@{x['time']} said={x['said']} read={x['read_back']} "
                 f"(found said={res['said']} read={res['read_back']})" for x in missed]
        notes += [f"  EXTRA wrong_readback@{x['time']} said={x['said']} read={x['read_back']}" for x in extra]
        want = truth_time_to_info(truth)
        got = res["time_to_info_s"]
        if want is not None and got is not None:
            tti_err.append(got - want)
            if abs(got - want) > 1.0:
                notes.append(f"  TTI off by {got - want:+.2f}s (truth {want}, found {got})")
        elif want is not None:
            tti_missing += 1
            notes.append(f"  TTI missing (truth {want})")
        print(f"{name}{''.join(notes) or '  ok'}")

    print(f"\n{band}: wrong_readback (tolerance {TOL_S}s)")
    print(f"{'type':16} {'found':>7} {'missed':>7} {'false':>7} {'recall':>7} {'precision':>9}")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    pre = tp / (tp + fp) if tp + fp else float("nan")
    print(f"{'wrong_readback':16} {tp:7} {fn:7} {fp:7} {rec:7.0%} {pre:9.0%}")
    if tti_err:
        e = np.abs(np.array(tti_err))
        print(f"time_to_info_s: {len(e)} calls matched, {tti_missing} missing, "
              f"mean |error| {e.mean():.3f} s, max |error| {e.max():.3f} s")
    print(f"run time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
