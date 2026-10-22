"""Cause analysis helper: classify cutoff FPs/FNs and large latency errors against annotation (reads score_real cache)."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
res = json.load(open(Path(__file__).parent / "out/callcheck_results.json"))
truth = {p.stem: json.load(open(p)) for p in sorted((ROOT / "data/real_tel/truth").glob("*.json"))}
def ann_at(k, t, who=None):
    return [x for x in truth[k]["turns"] if x["start"] - 0.2 <= t <= x["end"] + 0.2 and (who is None or x["who"] == who)]
print("== callcheck cutoffs: what does the annotation say at the caller 'resume' time?")
cls = {}
for k, r in res.items():
    for e in r["events"]:
        if e["type"] != "cutoff": continue
        a, ra = e["time"], e["caller_resumed_at"]
        tov = [o for o in truth[k]["overlaps"] if abs(o["agent_start"] - a) <= .5]
        c = ann_at(k, ra, "caller")
        # agent turn containing a
        at = [x for x in truth[k]["turns"] if x["who"] == "agent" and x["start"] - .5 <= a <= x["end"]]
        # next caller segment length (vad)
        seg = [s for s in r["segments"]["caller"] if abs(s[0] - ra) < .05]
        d = seg[0][1] - seg[0][0] if seg else 0
        if tov: lab = "TP"
        elif c and len(c[0]["text"].split()) <= 3: lab = "caller_short_reply_during_agent(<=3w: backchannel/ack)"
        elif c: lab = "caller_real_turn_after_agent_start(annot. caller speaks over agent but agent start was outside caller turn)"
        else: lab = "no_annotated_caller_speech_at_resume (noise/echo/unannotated)"
        cls[lab] = cls.get(lab, 0) + 1
        print(f"  {k} agent_start={a:.1f} resume={ra:.1f} vad_len={d:.2f} -> {lab}  | ann@resume: {[(x['text'][:30]) for x in c]}")
print(cls)
print("\n== latency errors >0.5 s: was there VAD caller speech between annotated caller end and agent start?")
c2 = {}
for k, r in res.items():
    lat = truth[k]["latencies"]
    for L in lat:
        m = [x for x in r["latencies"] if abs(x["agent_start"] - L["agent_start"]) <= .5]
        if not m: continue
        err = m[0]["latency_s"] - L["latency_s"]
        if abs(err) <= .5: continue
        extra = [s for s in r["segments"]["caller"] if s[1] > L["caller_end"] + .05 and s[0] < L["agent_start"] - .0]
        lab = "VAD hears caller sound after annotated end (annotation truncated / breath / unannotated)" if (err < 0 and extra) else ("VAD ended caller early" if err > 0 else "other")
        c2[lab] = c2.get(lab, 0) + 1
        print(f"  {k} t={L['agent_start']:.1f} truth={L['latency_s']:.2f} det={m[0]['latency_s']:.2f} extra_vad_caller={[(round(a,2),round(b,2)) for a,b in extra][:3]}")
print(c2)
print("\n== missed truth overlaps (agent starts inside caller turn)")
for k, r in res.items():
    det = [e["time"] for e in r["events"] if e["type"] == "cutoff"]
    for o in truth[k]["overlaps"]:
        if all(abs(o["agent_start"] - d) > .5 for d in det):
            s = [(round(a, 1), round(b, 1)) for a, b in r["segments"]["caller"] if b > o["agent_start"] - 1 and a < o["agent_start"] + 3]
            print(f"  {k} agent_start={o['agent_start']:.1f} caller_turn={[round(v,1) for v in o['caller_turn']]} bc={o['backchannel']} agent='{o['agent_text'][:30]}' vad_caller={s}")
