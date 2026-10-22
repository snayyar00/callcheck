"""Score callcheck on AMI 2-party clips against human word-time truth.

Usage: uv run -q --python 3.12 --with numpy --with onnxruntime python realdata/ami/score_ami.py [--worst N] [--cache]
"""
import json
from collections import Counter
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from callcheck.analyze import Config, analyze, load  # noqa: E402
from callcheck.vad import SileroVAD  # noqa: E402

PAIRS = ROOT / "data/ami_pairs"
TOL = 0.5


def match(truth_times, pred_times, tol=TOL):
    """Greedy nearest one-to-one match. Returns list of (ti, pi) pairs."""
    cand = sorted((abs(t - p), i, j) for i, t in enumerate(truth_times) for j, p in enumerate(pred_times) if abs(t - p) <= tol)
    ut, up, out = set(), set(), []
    for d, i, j in cand:
        if i in ut or j in up:
            continue
        ut.add(i); up.add(j); out.append((i, j))
    return out


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def bleed(caller, agent, tr):
    """Median amplitude ratio agent-channel/caller-channel while only the caller speaks (0 = isolated, 1 = same signal)."""
    def rms(x, s, e):
        return float(np.sqrt(np.mean(x[int(s * 16000):int(e * 16000)] ** 2))) + 1e-9
    r = [rms(agent, s, e) / rms(caller, s, e) for s, e, _ in tr["A_words"]
         if e - s > 0.25 and not any(b[0] < e + 0.3 and b[1] > s - 0.3 for b in tr["B_words"])]
    return float(np.median(r)) if r else 0.0


def cls(lat, cfg):
    return "dead" if lat > cfg.dead_s else "slow" if lat > cfg.slow_s else "ok"


def main():
    cfg = Config()
    vad = SileroVAD()
    worst_n = int(sys.argv[sys.argv.index("--worst") + 1]) if "--worst" in sys.argv else 5
    clips = sorted(p.stem for p in (PAIRS / "wav").glob("*.wav"))
    bmax = float(sys.argv[sys.argv.index("--bleed-max") + 1]) if "--bleed-max" in sys.argv else 9
    bmin = float(sys.argv[sys.argv.index("--bleed-min") + 1]) if "--bleed-min" in sys.argv else -1
    bleeds = {}
    kept = []
    for c in clips:
        tr0 = json.load(open(PAIRS / f"truth/{c}.json")); cl, ag = load(PAIRS / f"wav/{c}.wav")
        bleeds[c] = bleed(cl, ag, tr0)
        if bmin <= bleeds[c] <= bmax:
            kept.append(c)
    print("bleed (agent-ch / caller-ch level while only caller speaks):", {c: round(v, 2) for c, v in bleeds.items()})
    clips = kept
    lat_err, lat_err_all, lat_rows = [], [], []
    tl_n = tl_hit = cc_lat_n = cc_lat_unmatched = 0
    cp_tp = cp_fn = cp_n_cc = 0
    cut_tp = cut_fp = cut_fn = 0
    cut_fp_bc = cut_fp_rows = 0
    cut_fp_list, cut_fn_list = [], []
    inter_n = inter_hit = 0
    bc_n = bc_flag = 0
    cls_agree = cls_total = 0
    cls_conf = {}
    slow_truth = slow_hit = 0
    fb_rows = []
    fb_cand_rows = []
    per_clip = []
    for c in clips:
        tr = json.load(open(PAIRS / f"truth/{c}.json"))
        caller, agent = load(PAIRS / f"wav/{c}.wav")
        res = analyze(caller, agent, cfg, vad)
        # ---- latency ----
        lat_truth = tr["latencies"]
        for scope, rows in (("nonbc", [l for l in lat_truth if not l["backchannel"]]), ("all", lat_truth)):
            m = match([l["b_start"] for l in rows], [l["agent_start"] for l in res.latencies])
            if scope == "nonbc":
                tl_n += len(rows); tl_hit += len(m)
                for i, j in m:
                    e = res.latencies[j]["latency_s"] - rows[i]["latency_s"]
                    lat_err.append(e)
                    lat_rows.append({"clip": c, "b_start": rows[i]["b_start"], "truth": rows[i]["latency_s"],
                                     "cc": res.latencies[j]["latency_s"], "err": round(e, 3), "words": rows[i]["b_words"]})
                    cls_total += 1
                    ct, cp = cls(rows[i]["latency_s"], cfg), cls(res.latencies[j]["latency_s"], cfg)
                    cls_agree += ct == cp
                    cls_conf[(ct, cp)] = cls_conf.get((ct, cp), 0) + 1
                for i, r in enumerate(rows):
                    if r["latency_s"] > cfg.slow_s:
                        slow_truth += 1
                        slow_hit += any(i == a and cls(res.latencies[b]["latency_s"], cfg) != "ok" for a, b in m)
            else:
                for i, j in m:
                    lat_err_all.append(res.latencies[j]["latency_s"] - rows[i]["latency_s"])
        cc_lat_n += len(res.latencies)
        mm = match([l["b_start"] for l in lat_truth], [l["agent_start"] for l in res.latencies])
        cc_lat_unmatched += len(res.latencies) - len(mm)
        # ---- cutoff ----
        cc_cut = [e for e in res.events if e["type"] == "cutoff"]
        ct_times = [x["b_start"] for x in tr["cutoffs"]]
        m = match(ct_times, [e["time"] for e in cc_cut])
        cut_tp += len(m); cut_fn += len(ct_times) - len(m); cut_fp += len(cc_cut) - len(m)
        mp = {j for _, j in m}; mt = {i for i, _ in m}
        for i, t in enumerate(ct_times):
            if i not in mt:
                st = [a for a, _ in res.segments["agent_turns"]]
                kind = ("agent_turn_start_near_but_no_cutoff_event" if any(abs(a - t) <= TOL for a in st) else
                        "agent_turn_fused_with_earlier_turn_or_late_start" if any(a < t - TOL < b for a, b in res.segments["agent_turns"]) else
                        "agent_onset_missed_by_vad")
                cut_fn_list.append((c, t, kind))
        cp_times = [x["b_start"] for x in tr["cutoffs_pause"]]
        mpz = match(cp_times, [e["time"] for e in cc_cut])
        cp_tp += len(mpz); cp_fn += len(cp_times) - len(mpz); cp_n_cc += len(cc_cut)
        # interruptions (overlap, non backchannel): did cc flag a cutoff within tol?
        inter = [o for o in tr["overlaps"] if o["kind"] == "interruption"]
        inter_n += len(inter)
        inter_hit += len(match([o["b_start"] for o in inter], [e["time"] for e in cc_cut]))
        # backchannels (any B backchannel turn, overlapping or not): flagged as cutoff?
        bcs = [t["start"] for t in tr["B_agent_turns"] if t["backchannel"]] + \
              [t["start"] for t in tr["B_turns"] if t["backchannel"] and not any(abs(t["start"] - u["start"]) < 1e-6 for u in tr["B_agent_turns"])]
        bcs = sorted(set(round(x, 3) for x in bcs))
        bc_n += len(bcs)
        mb = match(bcs, [e["time"] for e in cc_cut])
        bc_flag += len(mb)
        for j, e in enumerate(cc_cut):
            if j not in mp:
                near_bc = any(abs(e["time"] - b) <= TOL for b in bcs)
                cut_fp_list.append((c, e["time"], "backchannel" if near_bc else "other"))
        # ---- false barge-in ----
        va = tr["vocal_sounds"]["A"]
        for e in (e for e in res.events if e["type"] == "false_barge_in"):
            near = [v for v in va if v["start"] - 1.0 <= e["time"] <= v["end"] + 1.0]
            wnear = [w for w in tr["A_words"] if w[0] - 0.3 <= e["time"] <= w[1] + 0.3]
            fb_rows.append((c, e["time"], e["agent_stopped_at"], near, [w[2] for w in wnear][:4]))
        fbc = tr["false_barge_candidates"]
        fbm = [x for x in fbc]
        cc_fb = [e["time"] for e in res.events if e["type"] == "false_barge_in"]
        for x in fbc:
            s0 = x["start"]
            ends = [w[1] for w in tr["B_words"] if s0 - 0.3 <= w[1] <= s0 + 1.0]
            stopped = any(not any(e0 < w[0] <= e0 + 0.4 for w in tr["B_words"]) for e0 in ends)
            fb_cand_rows.append((c, s0, x["end"], x["type"], stopped, any(x["start"] - 0.5 <= t <= x["end"] + 0.5 for t in cc_fb)))
        per_clip.append({"clip": c, "bleed": round(bleeds[c], 2), "n_lat_truth": len(lat_truth), "n_cc_lat": len(res.latencies),
                         "cutoff_truth": len(ct_times), "cutoff_cc": len(cc_cut),
                         "fb_cc": sum(e["type"] == "false_barge_in" for e in res.events), "fb_cand": len(fbm)})
        # cache segments for worst-case inspection
        (PAIRS / "cc_cache").mkdir(exist_ok=True)
        json.dump({"segments": {k: [list(map(float, s)) for s in v] for k, v in res.segments.items()},
                   "latencies": res.latencies, "events": res.events}, open(PAIRS / f"cc_cache/{c}.json", "w"))

    e = np.array(lat_err); ea = np.array(lat_err_all)
    print(f"clips: {len(clips)}")
    print("\n== LATENCY (truth = non-backchannel B turn onset after A turn end; match start within 0.5 s) ==")
    print(f"truth latencies {tl_n}, matched by callcheck {tl_hit} (recall {tl_hit/max(tl_n,1):.2f}); callcheck latencies {cc_lat_n}, unmatched to any truth turn {cc_lat_unmatched}")
    print(f"signed error cc-truth (s): median {np.median(e):+.3f}  mean {e.mean():+.3f}")
    ae = np.abs(e)
    print(f"abs error (s): median {np.median(ae):.3f}  p90 {pct(ae,90):.3f}  max {ae.max():.3f}; within 0.25 s: {(ae<=0.25).mean():.2f}; within 0.5 s: {(ae<=0.5).mean():.2f}")
    if len(ea):
        print(f"incl. backchannel turns: n={len(ea)} abs err median {np.median(np.abs(ea)):.3f} p90 {pct(np.abs(ea),90):.3f} max {np.abs(ea).max():.3f}")
    print("\n== SLOW / DEAD AIR (class by cc thresholds 1.5 / 3.0 s) on matched latencies ==")
    print(f"class agreement {cls_agree}/{cls_total}; confusion (truth,cc): {dict(sorted(cls_conf.items()))}")
    print(f"truth slow/dead latencies {slow_truth}, cc flagged non-ok {slow_hit}")
    print("\n== CUT-OFF ==")
    p = cut_tp / max(cut_tp + cut_fp, 1); r = cut_tp / max(cut_tp + cut_fn, 1)
    print(f"vs cutoff truth (non-backchannel B turn within 2 s of A speech and A resumes): TP {cut_tp} FP {cut_fp} FN {cut_fn} precision {p:.2f} recall {r:.2f}")
    print(f"vs pause-only cutoff truth (B starts in a real gap between A words and A resumes): TP {cp_tp} FN {cp_fn} cc events {cp_n_cc} -> precision {cp_tp/max(cp_n_cc,1):.2f} recall {cp_tp/max(cp_tp+cp_fn,1):.2f}")
    print(f"vs real interruptions (B overlap, non-backchannel): {inter_hit}/{inter_n} flagged as cutoff within 0.5 s")
    print(f"backchannel turns by B: {bc_n}; wrongly flagged as cutoff (within 0.5 s of a cc cutoff event): {bc_flag}")
    fp_bc = sum(1 for x in cut_fp_list if x[2] == "backchannel")
    print(f"cc cutoff FPs: {len(cut_fp_list)} ({fp_bc} within 0.5 s of a backchannel onset, {len(cut_fp_list)-fp_bc} other)")
    print("\n== FALSE BARGE-IN: callcheck events vs annotated vocal sounds on caller (+-1 s) ==")
    print(f"callcheck false_barge_in events: {len(fb_rows)}; with an annotated A vocalsound nearby: {sum(1 for r in fb_rows if r[3])}; with A words within 0.3 s: {sum(1 for r in fb_rows if r[4])}")
    for c, t, st, near, w in fb_rows:
        print(f"  {c} t={t:.2f} agent_stopped={st:.2f} nearby_vocal={[(v['type'], v['start'], v['end']) for v in near] or 'none'} A_words={w or 'none'}")
    print(f"truth false-barge candidates (A vocalsound while B talks, no A word within 0.3 s): {sum(p['fb_cand'] for p in per_clip)}; cc events total {sum(p['fb_cc'] for p in per_clip)}")
    print("truth candidates where B (human words) actually went quiet within 1 s (true false-barge):")
    stopped = [r for r in fb_cand_rows if r[4]]
    print(f"  {len(stopped)}/{len(fb_cand_rows)} candidates; callcheck flagged {sum(1 for r in stopped if r[5])} of them; types: {dict(Counter(r[3] for r in stopped))}")
    for r in stopped:
        print(f"  {r[0]} t={r[1]:.2f}-{r[2]:.2f} type={r[3]} cc_flagged={r[5]}")
    print("\n== per clip ==")
    for p in per_clip:
        print(p)
    print("\n== cc cutoff FPs ==")
    for x in cut_fp_list:
        print(" ", x)
    print("== cutoff FN causes ==", dict(Counter(x[2] for x in cut_fn_list)))
    print("== cutoff FNs ==")
    for x in cut_fn_list:
        print(" ", x)
    print("\n== worst latency errors ==")
    for r in sorted(lat_rows, key=lambda r: -abs(r["err"]))[:worst_n * 2]:
        print(" ", r)


if __name__ == "__main__":
    main()
