"""Print VAD segments vs truth words around a time. usage: inspect_clip.py CLIP T0 T1"""
import json, sys
from pathlib import Path
P = Path(__file__).resolve().parents[2] / "data/ami_pairs"
c, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
tr = json.load(open(P / f"truth/{c}.json")); cc = json.load(open(P / f"cc_cache/{c}.json"))
def win(xs, f=lambda x: x): return [x for x in xs if t1 >= f(x)[0] and t0 <= f(x)[1]]
def runs(words, gap):
    out = []
    for s, e, w in words:
        if out and s - out[-1][1] < gap: out[-1][1] = e; out[-1][2].append(w)
        else: out.append([s, e, [w]])
    return out
for who, key in (("A(caller)", "A_words"), ("B(agent)", "B_words")):
    print(who, "truth phrases:")
    for s, e, w in win(runs([tuple(x) for x in tr[key]], 0.5), lambda x: x): print(f"   {s:7.2f}-{e:7.2f} {' '.join(w)[:70]}")
for k in ("caller", "agent", "caller_turns", "agent_turns"):
    print("cc", k, [(round(s,2), round(e,2)) for s, e in win(cc["segments"][k], lambda x: x)])
print("cc events", win(cc["events"], lambda x: (x["time"], x["time"])))
print("cc latencies", win(cc["latencies"], lambda x: (x["agent_start"], x["agent_start"])))
print("A vocal", win(tr["vocal_sounds"]["A"], lambda x: (x["start"], x["end"])))
