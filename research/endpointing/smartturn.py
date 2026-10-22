"""Baseline: open smart-turn v3.2 (pipecat-ai, ONNX int8) at the same decision points. Needs run.py's cache.
uv run -q --python 3.12 --with numpy --with onnxruntime --with scikit-learn --with transformers --with matplotlib research/endpointing/smartturn.py
Model file: research/endpointing/models/smart-turn-v3.2-cpu.onnx (https://huggingface.co/pipecat-ai/smart-turn-v3), BSD-2.
"""
import json, pickle, sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run
from transformers import WhisperFeatureExtractor

fe = WhisperFeatureExtractor(chunk_length=8)
sess = ort.InferenceSession(str(HERE / "models/smart-turn-v3.2-cpu.onnx"), providers=["CPUExecutionProvider"])
d, tr_ids, te_ids, _ = run.choose_split()
te = [pickle.load(open(run.CACHE / f"{k}.pkl", "rb")) for k in te_ids]
te = [r for r in te if r["ok"]]
Xte, yte, silte, didte, gapte = run.build_arrays(te)
P = np.full(Xte.shape[:2], np.nan)
idx_dec = [run.DS.index(v) for v in run.SPEC4 + [1.2]]
n = 0
CP = HERE / "smartturn_P.npy"
if CP.exists():
    P = np.load(CP); te_loop = []
else:
    te_loop = list(enumerate(te))
for gi, r in te_loop:
    x = np.load(run.CACHE / f"{r['id']}.audio.npy").astype(np.float32) / 32767
    base = int(np.sum(didte < gi))
    for ri, row in enumerate(r["rows"]):
        for di in idx_dec:
            dd = run.DS[di]
            if dd > row["sil"]:
                continue
            t_end = int((row["ps"] + dd) * run.SR)
            a = x[max(0, t_end - 8 * run.SR):t_end]
            feats = fe(a, sampling_rate=16000, return_tensors="np", padding="max_length", max_length=8 * 16000,
                       truncation=True, do_normalize=True)["input_features"].astype(np.float32)
            lg = sess.run(None, {"input_features": feats})[0].reshape(-1)[0]
            P[base + ri, di] = 1 / (1 + np.exp(-lg))
            n += 1
print("inferences:", n)
np.save(HERE / "smartturn_P.npy", P)
from sklearn.metrics import roc_auc_score
out = {"auc_by_d": {}, "curve": [], "n_pauses": int(len(yte))}
for di in idx_dec:
    m = ~np.isnan(P[:, di])
    out["auc_by_d"][str(run.DS[di])] = round(float(roc_auc_score(yte[m], P[m, di])), 3)
print("smart-turn AUC by delay:", out["auc_by_d"])
ndlg = len(te)
for F in (0.8, 1.2):
    for th in np.round(np.concatenate([np.linspace(0.02, 0.5, 5), np.linspace(0.52, 0.8, 29), [0.85, 0.9, 0.98]]), 3):
        f = run.policy_fire(P, silte, run.DS, [run.DS.index(v) for v in run.SPEC4], th, F)
        pt, bs = run.boot_stats(didte, ndlg, f, yte, F, n_boot=300)
        out["curve"].append(dict(F=F, thr=float(th), cut=pt[0], mean_lat=pt[1], med_lat=pt[2]))
f8 = run.timeout_fire(silte, 0.8)
pt8, bs8 = run.boot_stats(didte, ndlg, f8, yte, 0.8, n_boot=300)
for F in (0.8, 1.2):
    pts = [p for p in out["curve"] if p["F"] == F]
    print(f"F={F}: at cut<=timeout0.8 ({pt8[0]:.3f}): best mean lat", min([p["mean_lat"] for p in pts if p["cut"] <= pt8[0]], default=None),
          "| at lat<=0.8: best cut", min([p["cut"] for p in pts if p["mean_lat"] <= pt8[1]], default=None))
    for p in pts[::2]:
        print("  thr", p["thr"], "cut %.1f%% lat %.2f" % (p["cut"] * 100, p["mean_lat"]))
json.dump(out, open(HERE / "smartturn_results.json", "w"), indent=1, default=float)

# ---- redraw the main plot with smart-turn added, and print a latency-matched table
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = json.load(open(HERE / "results.json"))
fig, ax = plt.subplots(figsize=(7.5, 5))
tc = R["curves"]["timeout"]
ax.plot([p["cut"] * 100 for p in tc], [p["mean_lat"] for p in tc], "ko-", label="silence timeout (0.2-1.5 s)")
for p in tc:
    if p["param"] in (0.3, 0.5, 0.8, 1.0, 1.5):
        ax.annotate(f'{p["param"]}s', (p["cut"] * 100, p["mean_lat"]), textcoords="offset points", xytext=(5, 4), fontsize=8)
for key, c in (("gboost|spec4|F1.2", "tab:red"), ("logreg|spec4|F1.2", "tab:blue")):
    pts = sorted(R["curves"][key], key=lambda p: p["cut"])
    ax.plot([p["cut"] * 100 for p in pts], [p["mean_lat"] for p in pts], "-", color=c, label=key.replace("|", " "))
pts = sorted([p for p in out["curve"] if p["F"] == 1.2 and p["cut"] < 0.995], key=lambda p: p["cut"])
ax.plot([p["cut"] * 100 for p in pts], [p["mean_lat"] for p in pts], "-", color="tab:green", label="smart-turn v3.2 (spec4, F1.2)")
ax.set_xlabel("cut-off rate: % of HOLD pauses where the agent would interrupt")
ax.set_ylabel("mean added wait on END pauses (s)")
ax.set_title(f"Endpointing on SpokenWOZ test split ({len(te)} calls, {len(yte)} pauses)")
ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_ylim(0, 1.6)
fig.tight_layout(); fig.savefig(HERE / "curve.png", dpi=130)
print("\nlatency-matched table: min cut-off % reachable at mean added wait <= L (point estimates, test)")
def best(pts_, L): 
    v = [p["cut"] for p in pts_ if p["mean_lat"] <= L + 1e-9]; return min(v) * 100 if v else float("nan")
rows = {"timeout": tc, "logreg": R["curves"]["logreg|spec4|F1.2"], "gboost": R["curves"]["gboost|spec4|F1.2"],
        "gboost dense": R["curves"]["gboost|dense|F1.2"], "smart-turn": [p for p in out["curve"] if p["F"] == 1.2]}
print("L(s)      " + "  ".join(f"{k:>12}" for k in rows))
for L in (0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
    print(f"{L:<9} " + "  ".join(f"{best(v, L):12.1f}" for v in rows.values()))
