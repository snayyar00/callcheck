"""Score callcheck against human-annotated SpokenWOZ calls (data/real_tel).

    uv run -q --python 3.12 --with numpy --with onnxruntime python realdata/telephone/score_real.py [--worst 5] [--recompute]

Truth: data/real_tel/truth/*.json. "cal" = annotation with the constant 0.34/0.35 s segment pad removed
(primary); "raw" = annotation as published. Match tolerance 0.5 s on the agent-start time.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from callcheck.analyze import analyze, load  # noqa: E402

# `score_real.py holdout` scores data/real_tel_holdout instead of the dev set.
SET = "holdout" if "holdout" in sys.argv else "dev"
DATA = ROOT / ("data/real_tel_holdout" if SET == "holdout" else "data/real_tel")
CACHE = Path(__file__).parent / "out" / f"callcheck_results{'_holdout' if SET == 'holdout' else ''}.json"
TOL = 0.5


def run_all(recompute):
    if CACHE.exists() and not recompute:
        return json.load(open(CACHE))
    out = {}
    for w in sorted((DATA / "wav").glob("*.wav")):
        c, a = load(w, 0)
        r = analyze(c, a)
        out[w.stem] = dict(latencies=r.latencies, events=r.events,
                           segments={k: [list(map(float, s)) for s in v] for k, v in r.segments.items()})
        print("analyzed", w.stem, file=sys.stderr)
    CACHE.parent.mkdir(exist_ok=True)
    json.dump(out, open(CACHE, "w"))
    return out


def match(truth_t, det_t):
    """Greedy 1:1 nearest match within TOL. Returns pairs (ti, di), unmatched truth, unmatched det."""
    pairs, used = [], set()
    for i, t in enumerate(truth_t):
        best = min((abs(d - t), j) for j, d in enumerate(det_t) if j not in used) if len(used) < len(det_t) else None
        if best and best[0] <= TOL:
            pairs.append((i, best[1])); used.add(best[1])
    mt = {i for i, _ in pairs}
    return pairs, [i for i in range(len(truth_t)) if i not in mt], [j for j in range(len(det_t)) if j not in used]


def pr(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float("nan"); r = tp / (tp + fn) if tp + fn else float("nan")
    return f"TP={tp} FP={fp} FN={fn} precision={p:.2f} recall={r:.2f}"


def cutoff_ci(per_call, n=2000, seed=0):
    """Resample whole calls, so calls with many overlaps carry their weight the way they would in a new sample."""
    rng = np.random.default_rng(seed); arr = np.array([c[1:] for c in per_call]); ps, rs = [], []
    for _ in range(n):
        tp, fp, fn = arr[rng.integers(0, len(arr), len(arr))].sum(0)
        ps.append(tp / (tp + fp) if tp + fp else np.nan); rs.append(tp / (tp + fn) if tp + fn else np.nan)
    q = lambda v: f"{np.nanpercentile(v, 2.5):.2f}-{np.nanpercentile(v, 97.5):.2f}"
    return f"precision {q(ps)}, recall {q(rs)}"


def stats(v):
    v = np.array(v)
    return f"n={len(v)} median={np.median(v):+.3f} p90_abs={np.percentile(np.abs(v), 90):.3f} max_abs={np.abs(v).max():.3f} mean_abs={np.abs(v).mean():.3f}"


def main():
    worst_n = int(sys.argv[sys.argv.index("--worst") + 1]) if "--worst" in sys.argv else 5
    res = run_all("--recompute" in sys.argv)
    truth = {p.stem: json.load(open(p)) for p in sorted((DATA / "truth").glob("*.json"))}
    print(f"calls={len(truth)}  hard={sum(t['group']=='hard' for t in truth.values())}  total_audio_min={sum(t['duration_s'] for t in truth.values())/60:.1f}")
    disagreements = []  # (abs_err_or_inf, call, kind, detail)
    for variant, sfx in (("CAL (pad-corrected, primary)", ""), ("RAW (annotation as published)", "_raw")):
        print(f"\n===== truth variant: {variant} =====")
        errs, miss_lat, extra_lat = [], 0, 0
        ov = {"all": [0, 0, 0], "nonbc": [0, 0, 0]}
        g15, g3 = [0, 0, 0], [0, 0, 0]
        sil3 = [0, 0]
        per_call = []  # (group, tp, fp, fn) for non-backchannel cut-offs, for the bootstrap and the group split
        for k, t in truth.items():
            r = res[k]
            tl = t["latencies" + sfx]
            # --- latency
            pairs, mt, md = match([x["agent_start"] for x in tl], [x["agent_start"] for x in r["latencies"]])
            for i, j in pairs:
                e = r["latencies"][j]["latency_s"] - tl[i]["latency_s"]
                errs.append(e)
                if sfx == "":
                    disagreements.append((abs(e), k, "latency", dict(t=tl[i]["agent_start"], truth=tl[i]["latency_s"], det=r["latencies"][j]["latency_s"])))
            miss_lat += len(mt); extra_lat += len(md)
            if sfx == "":
                for i in mt:
                    disagreements.append((0.9, k, "latency_missed", dict(t=tl[i]["agent_start"], truth=tl[i]["latency_s"])))
            # --- overlaps / cutoffs
            tov = t["overlaps" + sfx]
            det_c = [e["time"] for e in r["events"] if e["type"] == "cutoff"]
            p, mt, md = match([o["agent_start"] for o in tov], det_c)
            ov["all"][0] += len(p); ov["all"][1] += len(md); ov["all"][2] += len(mt)
            nb = [o for o in tov if not o["backchannel"]]
            p2, mt2, md2 = match([o["agent_start"] for o in nb], det_c)
            ov["nonbc"][0] += len(p2); ov["nonbc"][1] += len(md2); ov["nonbc"][2] += len(mt2)
            per_call.append((t["group"], len(p2), len(md2), len(mt2)))
            if sfx == "":
                for i in mt2:
                    disagreements.append((1.0, k, "cutoff_missed", dict(t=nb[i]["agent_start"], words=len(nb[i]["agent_text"].split()))))
                for j in md2:
                    disagreements.append((1.0, k, "cutoff_false", dict(t=det_c[j])))
            # --- gaps (response latencies truth) vs slow/dead events
            for thr, acc, types in ((1.5, g15, ("slow_response", "dead_air")), (3.0, g3, ("dead_air",))):
                tg = [x["agent_start"] for x in tl if x["latency_s"] > thr]
                dg = [e["time"] for e in r["events"] if e["type"] in types]
                p, mt, md = match(tg, dg)
                acc[0] += len(p); acc[1] += len(md); acc[2] += len(mt)
            sil3[0] += len(t["silences_gt3s" + sfx])
        print("LATENCY error (detected - annotated), seconds, over matched agent turns:")
        print("  ", stats(errs), f"| truth latencies missed by callcheck={miss_lat}, callcheck latencies with no truth={extra_lat}")
        print("  abs error <=0.25s:", f"{np.mean(np.abs(errs) <= .25):.0%}", " <=0.5s:", f"{np.mean(np.abs(errs) <= .5):.0%}")
        print("OVERLAP/cutoff (truth: agent starts >=0.3s into a caller turn with >=0.3s left; detected: 'cutoff'):")
        print("   vs all overlaps        ", pr(*ov["all"]))
        print("   vs non-backchannel only", pr(*ov["nonbc"]), "(backchannel = agent turn <=2 words)")
        print("   95% CI, bootstrap over calls (2000 resamples):", cutoff_ci(per_call))
        for g in ("hard", "random"):
            sub = [c for c in per_call if c[0] == g]
            print(f"   {g:6s} calls ({len(sub)}):", pr(*np.sum([c[1:] for c in sub], axis=0)))
        print(f"GAPS >1.5 s (slow_response|dead_air): {pr(*g15)}")
        print(f"GAPS >3.0 s (dead_air):              {pr(*g3)}")
        print(f"Truth >3 s mutual-silences in corpus: {sil3[0]}")
        if sfx == "":
            print("  truth counts: latencies=%d overlaps=%d (non-bc %d) gaps>1.5=%d gaps>3=%d" % (
                sum(len(t["latencies"]) for t in truth.values()), sum(len(t["overlaps"]) for t in truth.values()),
                sum(not o["backchannel"] for t in truth.values() for o in t["overlaps"]),
                sum(x["latency_s"] > 1.5 for t in truth.values() for x in t["latencies"]),
                sum(x["latency_s"] > 3 for t in truth.values() for x in t["latencies"])))
    print("\n===== FALSE BARGE-IN events (no truth; listen at these times) =====")
    n = 0
    for k, r in res.items():
        for e in r["events"]:
            if e["type"] == "false_barge_in":
                n += 1
                near = [f"{x['who']}:{x['start']:.1f}-{x['end']:.1f}" for x in truth[k]["turns"] if x["start"] - 1 <= e["time"] <= x["end"] + 1]
                print(f"  {k} t={e['time']:.2f}s agent_stopped_at={e['agent_stopped_at']:.2f}s annotated turns near: {near}")
    print("  total:", n, "in", f"{sum(t['duration_s'] for t in truth.values())/60:.1f} min of real speech")
    print("\n===== event totals =====")
    tot = {}
    for r in res.values():
        for e in r["events"]:
            tot[e["type"]] = tot.get(e["type"], 0) + 1
    print(" ", tot)

    if worst_n:
        print(f"\n===== {worst_n} WORST disagreements (CAL truth) =====")
    disagreements.sort(key=lambda x: -x[0])
    seen, shown = set(), 0
    for _, k, kind, d in disagreements:
        if shown >= worst_n:
            break
        if (k, round(d["t"])) in seen:
            continue
        seen.add((k, round(d["t"]))); shown += 1
        print(f"\n--- {k} {kind} {d}")
        t0 = d["t"]; lo, hi = t0 - 8, t0 + 3
        sg = res[k]["segments"]
        for who in ("caller", "agent"):
            print(f"  VAD {who} segs :", [(round(a, 2), round(b, 2)) for a, b in sg[who] if b > lo and a < hi])
        print("  VAD caller_turns:", [(round(a, 2), round(b, 2)) for a, b in sg["caller_turns"] if b > lo and a < hi])
        for x in truth[k]["turns"]:
            if x["end"] > lo and x["start"] < hi:
                # Transcript text only on request: SpokenWOZ is CC BY-NC, so committed outputs carry times only.
                text = f"  {x['text'][:70]}" if "--show-text" in sys.argv else ""
                print(f"  ANN {x['who']:6s} {x['start']:.2f}-{x['end']:.2f}{text}")


if __name__ == "__main__":
    main()
