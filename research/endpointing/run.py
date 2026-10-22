"""Endpointing study on SpokenWOZ test calls: silence timeout vs a small causal model.

Run:  uv run -q --python 3.12 --with numpy --with onnxruntime --with scikit-learn --with matplotlib \
          research/endpointing/run.py [--rebuild] [--smartturn]
Stages: (1) per-dialogue VAD + features, cached in research/endpointing/cache/  (2) train/eval  (3) plot + results json.
Local only: SpokenWOZ is CC BY-NC 4.0.  Nothing here is committed or published.
"""
import json, os, random, subprocess, sys, zipfile, pickle, io
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
TXT = ROOT / "data/.corpora/spokenwoz_text/data.json"
ZIP = ROOT / "data/.corpora/spokenwoz_test/audio_5700_test.zip"
SR = 16000
PAD_START, PAD_END = 0.34, 0.35          # annotation pad, from realdata/telephone/build_real_tel.py
DS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5]   # decision delays after pause start
SPEC4 = [0.2, 0.3, 0.5, 0.8]
N_TRAIN, N_TEST, SEED = 200, 100, 11
MAX_DUR = 400.0
FEATS = ["d", "last_seg_dur", "f0_slope", "f0_rel", "f0_voiced", "e_slope", "e_rel", "rate",
         "turn_len", "n_pauses", "mean_prev_pause", "pause_e_rel"]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------- annotation
def agent_turns(log):
    """Agent turns (merged consecutive system entries), pad-corrected. Returns [(start,end,n_words)]."""
    out, last = [], None
    for e in log:
        if not e["words"]:
            continue
        who = e["tag"]
        s, en = e["words"][0]["BeginTime"] / 1000 + PAD_START, e["words"][-1]["EndTime"] / 1000 - PAD_END
        if en <= s:
            s, en = (s + en) / 2 - 0.1, (s + en) / 2 + 0.1
        if who == "system":
            if last == "system":
                out[-1][1] = max(out[-1][1], en); out[-1][2] += len(e["words"])
            else:
                out.append([s, en, len(e["words"])])
        last = who
    return out


def excluded_ids():
    ex = set()
    for d in ("data/real_tel/truth", "data/real_tel_holdout/truth"):
        ex |= {f.stem for f in (ROOT / d).glob("*.json")}
    return ex


def choose_split():
    d = json.load(open(TXT))
    with zipfile.ZipFile(ZIP) as z:
        have = {Path(n).stem for n in z.namelist() if n.endswith(".wav") and "__MACOSX" not in n}
    ex = excluded_ids()
    ids = sorted(k for k in d if k in have and k not in ex and any(e["words"] for e in d[k]["log"]))
    ids = [k for k in ids if max(w["EndTime"] for e in d[k]["log"] for w in e["words"]) / 1000 <= MAX_DUR]
    random.Random(SEED).shuffle(ids)
    return d, ids[:N_TRAIN], ids[N_TRAIN:N_TRAIN + N_TEST], len(ex)


# ---------------------------------------------------------------- audio
def load_audio(k):
    with zipfile.ZipFile(ZIP) as z:
        raw = z.read(f"audio_5700_test/{k}.wav")
    tmp = CACHE / f"_{k}.wav"
    tmp.write_bytes(raw)
    try:
        r = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(tmp), "-ar", str(SR), "-f", "f32le", "-ac", "2", "-"],
                           capture_output=True, check=True)
    finally:
        tmp.unlink(missing_ok=True)
    return np.frombuffer(r.stdout, np.float32).reshape(-1, 2)


def frame_db(x, hop=320):
    n = len(x) // hop
    r = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-10)
    return 20 * np.log10(r)


def f0_track(x, t0, t1, hop=0.01, win=0.04, floor_db=None):
    """Simple autocorrelation f0 (semitones re 100 Hz) over [t0,t1]; returns (times, semitones) of voiced frames."""
    ts, st = [], []
    w = int(win * SR)
    for t in np.arange(t0, t1 - win + 1e-9, hop):
        a = int(t * SR)
        f = x[a:a + w]
        if len(f) < w:
            break
        f = f - f.mean()
        e = (f ** 2).sum()
        if e < 1e-6:
            continue
        if floor_db is not None and 10 * np.log10(e / w + 1e-10) < floor_db:
            continue
        n = 2 * w
        sp = np.fft.rfft(f, n)
        r = np.fft.irfft(sp * np.conj(sp), n)[:w]
        r = r / r[0]
        lo, hi = SR // 400, SR // 75
        lag = lo + int(np.argmax(r[lo:hi]))
        if r[lag] > 0.35:
            ts.append(t + win / 2); st.append(12 * np.log2((SR / lag) / 100))
    return np.array(ts), np.array(st)


def pause_static_feats(x, edb, segs, k, atrn):
    """Features that depend only on audio up to the pause start (end of segment k)."""
    s0, s1 = segs[k]
    # turn start = first caller segment starting after the last (non-backchannel) agent turn that ended before s1
    prev_end = max([a[1] for a in atrn if a[2] > 2 and a[1] <= s1], default=-1.0)
    j = k
    while j > 0 and segs[j - 1][0] >= prev_end - 0.1:
        j -= 1
    turn_start = segs[j][0]
    gaps = [segs[i + 1][0] - segs[i][1] for i in range(j, k)]
    gaps = [g for g in gaps if g >= 0.2]
    # energy over last 300 ms of the segment
    fl = lambda t: int(t / 0.02)
    a, b = fl(max(s0, s1 - 0.3)), max(fl(s1), fl(max(s0, s1 - 0.3)) + 2)
    seg_db = edb[fl(s0):max(fl(s1), fl(s0) + 1)]
    ref = seg_db.mean() if len(seg_db) else -60
    e = edb[a:b]
    e_slope = float(np.polyfit(np.arange(len(e)) * 0.02, e, 1)[0]) if len(e) >= 3 else 0.0
    e_rel = float(e.mean() - ref) if len(e) else 0.0
    # f0 slope over last 300 ms, level relative to the segment median
    peak = edb[fl(s0):max(fl(s1), fl(s0) + 1)].max() if len(seg_db) else -40
    t_, s_ = f0_track(x, max(s0, s1 - 0.3), s1, floor_db=peak - 35)
    if len(t_) >= 4:
        f0_slope = float(np.polyfit(t_, s_, 1)[0]); f0_last = float(np.median(s_))
        t2, s2 = f0_track(x, max(s0, s1 - 1.5), s1, hop=0.02, floor_db=peak - 35)
        f0_rel = f0_last - float(np.median(s2)) if len(s2) >= 4 else 0.0
        f0_v = min(1.0, len(t_) / 20)
    else:
        f0_slope, f0_rel, f0_v = 0.0, 0.0, 0.0
    # speech-rate proxy: envelope peaks per second over last <=1 s of the segment
    w0 = max(s0, s1 - 1.0)
    en = edb[fl(w0):fl(s1)]
    if len(en) >= 5:
        sm = np.convolve(en, np.ones(3) / 3, "same")
        pk = (sm[2:-2] > sm[1:-3]) & (sm[2:-2] > sm[:-4]) & (sm[2:-2] >= sm[3:-1]) & (sm[2:-2] >= sm[4:]) & (sm[2:-2] > sm.max() - 15)
        rate = float(pk.sum() / max(s1 - w0, 0.2))
    else:
        rate = 0.0
    return dict(last_seg_dur=s1 - s0, f0_slope=f0_slope, f0_rel=f0_rel, f0_voiced=f0_v, e_slope=e_slope, e_rel=e_rel,
                rate=rate, turn_len=s1 - turn_start, n_pauses=len(gaps), mean_prev_pause=float(np.mean(gaps)) if gaps else 0.0,
                ref_db=float(ref), turn_start=turn_start)


def process(k):
    """Stage 1 for one dialogue. Returns dict of arrays."""
    out_f = CACHE / f"{k}.pkl"
    if out_f.exists():
        return pickle.load(open(out_f, "rb"))
    from callcheck.vad import SileroVAD
    d = json.load(open(TXT))[k]
    st = load_audio(k)
    x = st[:, 0].astype(np.float32)
    # channel sanity (agent turns louder on ch1 than ch0)
    atrn = agent_turns(d["log"])
    big = [a for a in atrn if a[2] > 2 and a[1] * SR < len(st)]
    ok = True
    if big:
        r0 = np.mean([np.sqrt((st[int(a[0] * SR):int(a[1] * SR), 0] ** 2).mean() + 1e-12) for a in big])
        r1 = np.mean([np.sqrt((st[int(a[0] * SR):int(a[1] * SR), 1] ** 2).mean() + 1e-12) for a in big])
        ok = r1 > 2 * r0
    res = dict(id=k, ok=bool(ok), dur=len(x) / SR, rows=[])
    if not ok:
        pickle.dump(res, open(out_f, "wb")); return res
    peak = np.abs(x).max()
    if peak > 0:
        x = x / peak * 0.5
    segs = SileroVAD().segments(x)
    edb = frame_db(x)
    big_turns = [a for a in atrn if a[2] > 2]
    total = len(x) / SR
    for i, (s0, s1) in enumerate(segs):
        nxt = segs[i + 1][0] if i + 1 < len(segs) else total
        sil = nxt - s1
        if sil < 0.2:
            continue
        # agent holds the floor at pause start -> not a caller decision point
        if any(a[0] <= s1 <= a[1] for a in big_turns):
            continue
        fut = [a[0] for a in big_turns if a[0] > s1 - 0.0]
        A = fut[0] if fut else np.inf
        if A < s1 + 0.2:
            continue                         # agent already started: no decision to make
        label = 1 if (A < nxt or i + 1 == len(segs)) else 0   # END: agent speaks before caller resumes (or call ends)
        if i + 1 == len(segs) and not fut:
            continue                         # trailing silence with no agent turn: unlabeled
        sf = pause_static_feats(x, edb, segs, i, atrn)
        row = dict(seg=i, ps=s1, sil=sil, label=label, gap_to_agent=(A - s1) if np.isfinite(A) else np.nan,
                   vad_end=(i + 1 == len(segs)), X=np.full((len(DS), len(FEATS)), np.nan, np.float32))
        for di, dd in enumerate(DS):
            if dd > sil:
                continue
            a, b = int(s1 / 0.02), int((s1 + dd) / 0.02)
            pe = float(edb[a:b].mean() - sf["ref_db"]) if b > a else 0.0
            row["X"][di] = [dd, sf["last_seg_dur"], sf["f0_slope"], sf["f0_rel"], sf["f0_voiced"], sf["e_slope"], sf["e_rel"],
                            sf["rate"], sf["turn_len"], sf["n_pauses"], sf["mean_prev_pause"], pe]
        res["rows"].append(row)
    # keep what smart-turn needs (8 s of caller audio) lazily: store int16 audio for later
    np.save(CACHE / f"{k}.audio.npy", (x * 32767).astype(np.int16))
    pickle.dump(res, open(out_f, "wb"))
    return res


# ---------------------------------------------------------------- policies / metrics
def build_arrays(results):
    """Flatten rows -> arrays. X: [n, nD, nF]."""
    X, y, sil, did, gap = [], [], [], [], []
    for di, r in enumerate(results):
        for row in r["rows"]:
            X.append(row["X"]); y.append(row["label"]); sil.append(row["sil"]); did.append(di); gap.append(row["gap_to_agent"])
    return np.array(X), np.array(y), np.array(sil), np.array(did), np.array(gap)


def policy_fire(P, sil, ds, decision_idx, thr, F):
    """First decision delay (<= F, <= sil) where P>=thr, else forced fire at F. Returns fire time per pause (np.inf when
    pause is shorter than F and nothing fired = caller resumed first)."""
    fire = np.full(len(sil), np.inf)
    done = np.zeros(len(sil), bool)
    for di in decision_idx:
        dd = ds[di]
        if dd > F:
            continue
        m = (~done) & (dd <= sil) & (P[:, di] >= thr)
        fire[m] = dd; done |= m
    m = (~done) & (F <= sil)
    fire[m] = F
    return fire


def timeout_fire(sil, T):
    return np.where(sil >= T, T, np.inf)


def eval_fire(fire, y, T_fallback):
    """cut = HOLD pauses where agent fired before caller resumed; latency of END = fire (if never fired, fallback)."""
    hold = y == 0
    cut = hold & np.isfinite(fire)
    lat = np.where(np.isfinite(fire), fire, T_fallback)[y == 1]
    return cut, lat


def boot_stats(did, ndlg, fire, y, fb, n_boot=1000, rng=None):
    """Bootstrap by dialogue. Returns arrays (cut_rate, mean_lat, med_lat) over resamples + point values."""
    rng = rng or np.random.default_rng(0)
    cut_d = np.zeros(ndlg); hold_d = np.zeros(ndlg)
    vals = np.unique(np.where(np.isfinite(fire), fire, fb)[y == 1])
    lat_h = np.zeros((ndlg, len(vals)))
    for g in range(ndlg):
        m = did == g
        hold_d[g] = (m & (y == 0)).sum()
        cut_d[g] = (m & (y == 0) & np.isfinite(fire)).sum()
        lv = np.where(np.isfinite(fire[m & (y == 1)]), fire[m & (y == 1)], fb)
        for j, v in enumerate(vals):
            lat_h[g, j] = (lv == v).sum()
    def stats(w):
        cr = (w @ cut_d) / max(w @ hold_d, 1)
        h = w @ lat_h
        n = h.sum()
        mean = (h * vals).sum() / n
        med = vals[np.searchsorted(np.cumsum(h), n / 2)]
        return cr, mean, med
    point = stats(np.ones(ndlg))
    bs = np.array([stats(np.bincount(rng.integers(0, ndlg, ndlg), minlength=ndlg).astype(float)) for _ in range(n_boot)])
    return point, bs


def main():
    CACHE.mkdir(exist_ok=True)
    rebuild = "--rebuild" in sys.argv
    d, tr_ids, te_ids, nex = choose_split()
    print(f"excluded dialogues (real_tel + holdout): {nex}; train {len(tr_ids)}, test {len(te_ids)}")
    if rebuild:
        for f in CACHE.glob("*.pkl"):
            f.unlink()
    todo = tr_ids + te_ids
    with Pool(8) as p:
        res = p.map(process, todo, chunksize=2)
    byid = {r["id"]: r for r in res}
    bad = [k for k in todo if not byid[k]["ok"]]
    print("rejected on channel check:", bad)
    tr = [byid[k] for k in tr_ids if byid[k]["ok"]]
    te = [byid[k] for k in te_ids if byid[k]["ok"]]
    Xtr, ytr, siltr, didtr, _ = build_arrays(tr)
    Xte, yte, silte, didte, gapte = build_arrays(te)
    print(f"train pauses {len(ytr)} (END {ytr.mean():.1%}) in {len(tr)} dialogues; test pauses {len(yte)} (END {yte.mean():.1%}) in {len(te)}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    def flat(X, y):
        m = ~np.isnan(X[:, :, 0])
        return X[m], np.repeat(y[:, None], X.shape[1], 1)[m]
    Ftr, Ltr = flat(Xtr, ytr)
    sc = StandardScaler().fit(Ftr)
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Ftr), Ltr)
    gb = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.06, min_samples_leaf=40, random_state=0).fit(Ftr, Ltr)

    def predict(model, X, scale):
        P = np.full(X.shape[:2], np.nan)
        m = ~np.isnan(X[:, :, 0])
        Z = X[m]
        P[m] = model.predict_proba(sc.transform(Z) if scale else Z)[:, 1]
        return P
    models = {"logreg": predict(lr, Xte, True), "gboost": predict(gb, Xte, False)}
    Fte, Lte = flat(Xte, yte)
    out = {"n_train_pauses": int(len(ytr)), "n_test_pauses": int(len(yte)), "test_end_frac": float(yte.mean()),
           "n_train_dialogues": len(tr), "n_test_dialogues": len(te), "excluded": nex, "auc_by_d": {}}
    for name, P in models.items():
        out["auc_by_d"][name] = {}
        for di, dd in enumerate(DS):
            m = ~np.isnan(P[:, di])
            if len(set(yte[m])) > 1:
                out["auc_by_d"][name][str(dd)] = round(float(roc_auc_score(yte[m], P[m, di])), 3)
        print(name, "AUC by delay:", out["auc_by_d"][name])
    # simple baseline model: pause length alone is constant in d -> AUC 0.5 by construction (d only); report duration-only LR
    gbc = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.06, min_samples_leaf=40, random_state=0)
    imp = {}
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(gb, Fte[:4000], Lte[:4000], n_repeats=3, random_state=0, scoring="roc_auc")
    imp = {f: round(float(v), 4) for f, v in zip(FEATS, pi.importances_mean)}
    out["gboost_perm_importance_auc"] = imp
    print("gboost permutation importance (AUC drop):", imp)

    ndlg = len(te)
    curves = {}
    fb = 1.5
    # silence timeouts
    for T in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
        f = timeout_fire(silte, T)
        pt, bs = boot_stats(didte, ndlg, f, yte, T)
        curves.setdefault("timeout", []).append(dict(param=T, cut=pt[0], mean_lat=pt[1], med_lat=pt[2],
                                                       cut_ci=np.percentile(bs[:, 0], [2.5, 97.5]).tolist(),
                                                       mean_ci=np.percentile(bs[:, 1], [2.5, 97.5]).tolist()))
    thrs = np.round(np.concatenate([np.linspace(0.05, 0.95, 37), [0.97, 0.99]]), 3)
    variants = {"spec4": [DS.index(v) for v in SPEC4], "dense": list(range(len(DS)))}
    boots = {}
    for name, P in models.items():
        for vname, idx in variants.items():
            for F in (0.8, 1.2, 1.5):
                key = f"{name}|{vname}|F{F}"
                pts = []
                for th in thrs:
                    f = policy_fire(P, silte, DS, idx, th, F)
                    pt, bs = boot_stats(didte, ndlg, f, yte, F, n_boot=300)
                    pts.append(dict(param=float(th), cut=pt[0], mean_lat=pt[1], med_lat=pt[2], boot=bs[:, :2].tolist()))
                curves[key] = pts
    # headline: at equal cut-off rate to the 0.8 s timeout, and at equal mean latency
    rng = np.random.default_rng(1)
    f8 = timeout_fire(silte, 0.8)
    pt8, bs8 = boot_stats(didte, ndlg, f8, yte, 0.8, n_boot=300)  # same rng(0) draws as model curves -> paired
    head = {"timeout_0.8": dict(cut=pt8[0], lat=pt8[1])}
    for key, pts in curves.items():
        if key == "timeout":
            continue
        # point estimate
        def best_lat_at_cut(c, pts_=pts):
            ok = [p["mean_lat"] for p in pts_ if p["cut"] <= c]
            return min(ok) if ok else np.nan
        def best_cut_at_lat(l, pts_=pts):
            ok = [p["cut"] for p in pts_ if p["mean_lat"] <= l]
            return min(ok) if ok else np.nan
        lat_at = best_lat_at_cut(pt8[0]); cut_at = best_cut_at_lat(pt8[1])
        # bootstrap: use paired resample index b (same dialogue draws are NOT shared across policies; draws independent, fine for CI width)
        nb = min(len(pts[0]["boot"]), len(bs8))
        dl, dc = [], []
        for b in range(nb):
            c8, l8 = bs8[b, 0], bs8[b, 1]
            ok = [p["boot"][b][1] for p in pts if p["boot"][b][0] <= c8]
            dl.append(l8 - min(ok) if ok else np.nan)
            ok = [p["boot"][b][0] for p in pts if p["boot"][b][1] <= l8]
            dc.append(c8 - min(ok) if ok else np.nan)
        head[key] = dict(lat_at_equal_cut=lat_at, cut_at_equal_lat=cut_at,
                         lat_saving_ci=np.nanpercentile(dl, [2.5, 50, 97.5]).tolist(),
                         cut_saving_ci=np.nanpercentile(dc, [2.5, 50, 97.5]).tolist())
    out["headline"] = head
    out["human_gap_median_s"] = float(np.nanmedian(gapte[yte == 1]))
    out["human_gap_quartiles"] = np.nanpercentile(gapte[yte == 1], [25, 75]).tolist()
    for k_ in curves:
        for p in curves[k_]:
            p.pop("boot", None)
    out["curves"] = curves
    json.dump(out, open(HERE / "results.json", "w"), indent=1, default=float)
    print(json.dumps(head, indent=1, default=float))
    print("human (annotated agent) gap after caller pause start, END pauses: median", out["human_gap_median_s"],
          "IQR", out["human_gap_quartiles"])

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5))
    tc = curves["timeout"]
    ax.plot([p["cut"] * 100 for p in tc], [p["mean_lat"] for p in tc], "ko-", label="silence timeout (0.2-1.5 s)")
    for p in tc:
        if p["param"] in (0.3, 0.5, 0.8, 1.0, 1.5):
            ax.annotate(f'{p["param"]}s', (p["cut"] * 100, p["mean_lat"]), textcoords="offset points", xytext=(5, 4), fontsize=8)
    for key, c in (("gboost|spec4|F1.2", "tab:red"), ("logreg|spec4|F1.2", "tab:blue"), ("gboost|dense|F1.2", "tab:orange")):
        pts = sorted(curves[key], key=lambda p: p["cut"])
        ax.plot([p["cut"] * 100 for p in pts], [p["mean_lat"] for p in pts], "-", color=c, label=key.replace("|", " "))
    ax.set_xlabel("cut-off rate: % of HOLD pauses where the agent would interrupt")
    ax.set_ylabel("mean added wait on END pauses (s)")
    ax.set_title(f"Endpointing on SpokenWOZ test split ({len(te)} calls, {len(yte)} pauses)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_ylim(0, 1.6)
    fig.tight_layout(); fig.savefig(HERE / "curve.png", dpi=130)
    print("wrote", HERE / "results.json", HERE / "curve.png")


if __name__ == "__main__":
    main()
