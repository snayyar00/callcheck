"""Build data/real_tel/{wav,truth} from SpokenWOZ test set (CC BY-NC 4.0).

Inputs (already downloaded, untouched):
  data/.corpora/spokenwoz_text/data.json              word-timestamped transcripts (BeginTime/EndTime ms)
  data/.corpora/spokenwoz_test/audio_5700_test.zip    8 kHz stereo, ch0 = customer ("user"), ch1 = service ("system")
Usage: python realdata/telephone/build_real_tel.py
"""
import json, random, subprocess, sys, zipfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TXT = ROOT / "data/.corpora/spokenwoz_text/data.json"
ZIP = ROOT / "data/.corpora/spokenwoz_test/audio_5700_test.zip"
RAW = ROOT / "data/.corpora/spokenwoz_wav"
OUT = ROOT / "data/real_tel"
N_RANDOM, N_HARD, SEED = 22, 10, 7
# Holdout build: `build_real_tel.py holdout` writes data/real_tel_holdout, never reusing a dev-set call,
# and picks its hard calls by real overlaps so the cut-off detector has enough to be scored on.
HOLDOUT = len(sys.argv) > 1 and sys.argv[1] == "holdout"
if HOLDOUT:
    OUT, N_RANDOM, N_HARD, SEED = ROOT / "data/real_tel_holdout", 20, 20, 2026
BACKCHANNEL_MAX_WORDS = 2
# Annotated utterance boundaries are padded vs. the audio: measured on own-channel energy over 1090 turns,
# annotated start is a median 0.34 s BEFORE voice onset (IQR 0.29-0.35) and annotated end is 0.35 s AFTER
# voice offset (IQR 0.32-0.39). We remove that constant pad to get the "cal" truth; "raw" keeps annotation as-is.
PAD_START, PAD_END = 0.34, 0.35


def turns_of(log, pad=True):
    """One turn per annotated log entry; consecutive same-speaker entries are merged."""
    out = []
    for e in log:
        if not e["words"]:
            continue
        who = "caller" if e["tag"] == "user" else "agent"
        s, en = e["words"][0]["BeginTime"] / 1000, e["words"][-1]["EndTime"] / 1000
        if pad:
            s, en = s + PAD_START, en - PAD_END
            if en <= s:
                s, en = (s + en) / 2 - 0.1, (s + en) / 2 + 0.1
        if out and out[-1]["who"] == who:
            out[-1]["end"] = max(out[-1]["end"], en); out[-1]["text"] += " " + e["text"]
            out[-1]["n_words"] += len(e["words"])
        else:
            out.append(dict(who=who, start=s, end=en, text=e["text"], n_words=len(e["words"])))
    return out


def derive(turns, dur):
    lat, ov, sil = [], [], []
    for i, t in enumerate(turns):
        if t["who"] != "agent":
            continue
        # response latency: previous turn in the log is the caller and ended before the agent started
        if i and turns[i - 1]["who"] == "caller" and turns[i - 1]["end"] <= t["start"]:
            c = turns[i - 1]
            lat.append(dict(agent_start=t["start"], caller_end=c["end"], latency_s=round(t["start"] - c["end"], 3)))
        # overlap: agent starts while a caller turn is still going (>=0.3 s in, >=0.3 s left)
        for c in turns:
            if c["who"] == "caller" and c["start"] + 0.3 <= t["start"] <= c["end"] - 0.3:
                ov.append(dict(agent_start=t["start"], caller_turn=[c["start"], c["end"]],
                               caller_remaining_s=round(c["end"] - t["start"], 3),
                               agent_end=t["end"], agent_text=t["text"],
                               backchannel=t["n_words"] <= BACKCHANNEL_MAX_WORDS))
    # silences: gaps > 3 s with nobody talking (union of all turns)
    cur = 0.0
    for s, e in sorted((t["start"], t["end"]) for t in turns):
        if s - cur > 3.0:
            sil.append(dict(start=cur, end=s, dur_s=round(s - cur, 3)))
        cur = max(cur, e)
    return lat, ov, sil


def onset_error(x, sr, turns, who_idx):
    """Annotation precision probe: |annotated start - first energy onset| on the speaker's own channel."""
    fr = int(0.02 * sr); n = len(x) // fr
    rms = np.sqrt((x[: n * fr].reshape(n, fr) ** 2).mean(1)); thr = max(60.0, 0.1 * np.percentile(rms, 95))
    errs = []
    for t in turns:
        if t["who"] != who_idx:
            continue
        a, b = max(0, int((t["start"] - 1.0) / 0.02)), min(n, int((t["start"] + 1.0) / 0.02))
        on = [k for k in range(a, b) if rms[k] > thr]
        if on:
            errs.append(on[0] * 0.02 - t["start"])
    return errs


def main():
    d = json.load(open(TXT))
    info = {}
    for k, v in d.items():
        tr = turns_of(v["log"])
        if len(tr) < 12:
            continue
        dur = tr[-1]["end"]
        if not 60 <= dur <= 300:
            continue
        lat, ov, sil = derive(tr, dur)
        info[k] = dict(dur=dur, n_ov=sum(not o["backchannel"] for o in ov), n_lat=len(lat), n_sil=len(sil),
                       n_slow=sum(l["latency_s"] > 1.5 for l in lat))
    keys = sorted(info)
    if HOLDOUT:
        dev = {f.stem for f in (ROOT / "data/real_tel/truth").glob("*.json")}
        keys = [k for k in keys if k not in dev]
        hard = sorted(keys, key=lambda k: -info[k]["n_ov"])[:N_HARD + 10]
    else:
        hard = sorted(keys, key=lambda k: -(info[k]["n_ov"] + info[k]["n_sil"] + info[k]["n_slow"]))[:N_HARD]
    rest = [k for k in keys if k not in hard]
    random.Random(SEED).shuffle(rest)
    chosen = hard + rest[: N_RANDOM + 8]  # extras in case of channel-check rejects
    RAW.mkdir(exist_ok=True, parents=True)
    with zipfile.ZipFile(ZIP) as z:
        for k in chosen:
            if not (RAW / "audio_5700_test" / f"{k}.wav").exists():
                z.extract(f"audio_5700_test/{k}.wav", RAW)
    (OUT / "wav").mkdir(parents=True, exist_ok=True); (OUT / "truth").mkdir(exist_ok=True)
    kept, rej, errs_all = [], [], {"caller": [], "agent": []}
    for k in chosen:
        if len(kept) >= N_HARD + N_RANDOM:
            break
        p = RAW / "audio_5700_test" / f"{k}.wav"
        # ffmpeg, not the wave module: a few SpokenWOZ files are 32-bit float WAV, which wave cannot read.
        sr = 8000
        x = np.frombuffer(subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(p), "-f", "s16le", "-ac", "2",
                                          "-ar", str(sr), "-"], capture_output=True, check=True).stdout,
                          np.int16).reshape(-1, 2).astype(float)
        tr = turns_of(d[k]["log"], pad=False)
        # channel sanity: caller turns must be louder on ch0 and agent turns on ch1 (else swapped/mono: reject)
        def rms(who, ch):
            return np.mean([np.sqrt((x[int(t["start"] * sr):int(t["end"] * sr), ch] ** 2).mean() + 1e-9)
                            for t in tr if t["who"] == who and t["end"] * sr <= len(x)])
        if not (rms("caller", 0) > 3 * rms("caller", 1) and rms("agent", 1) > 3 * rms("agent", 0)):
            rej.append(k); continue
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(p), "-ar", "16000", "-ac", "2",
                        "-c:a", "pcm_s16le", str(OUT / "wav" / f"{k}.wav")], check=True)
        lat, ov, sil = derive(tr, info[k]["dur"])
        tr_raw = turns_of(d[k]["log"], pad=False)
        lat_r, ov_r, sil_r = derive(tr_raw, info[k]["dur"])
        tr_cal = turns_of(d[k]["log"])
        lat, ov, sil = derive(tr_cal, info[k]["dur"])
        for who, ch in (("caller", 0), ("agent", 1)):
            errs_all[who] += onset_error(x[:, ch], sr, tr_cal, who)
        json.dump(dict(id=k, source="SpokenWOZ test (CC BY-NC 4.0)", caller_channel=0,
                       group="hard" if k in hard else "random", duration_s=len(x) / sr,
                       turns=[dict(who=t["who"], start=t["start"], end=t["end"], text=t["text"], n_words=t["n_words"]) for t in tr_cal],
                       truth_note="primary fields use pad-corrected turns (see build_real_tel.py); *_raw use annotation as-is",
                       latencies=lat, overlaps=ov, silences_gt3s=sil,
                       turns_raw=[dict(who=t["who"], start=t["start"], end=t["end"]) for t in tr_raw],
                       latencies_raw=lat_r, overlaps_raw=ov_r, silences_gt3s_raw=sil_r),
                  open(OUT / "truth" / f"{k}.json", "w"), indent=1)
        kept.append(k)
    pr = {w: dict(n=len(e), median_abs_s=float(np.median(np.abs(e))), p90_abs_s=float(np.percentile(np.abs(e), 90)),
                  median_signed_s=float(np.median(e))) for w, e in errs_all.items()}
    json.dump(dict(kept=kept, rejected_channel_check=rej, annotation_onset_vs_energy=pr),
              open(OUT / "manifest.json", "w"), indent=1)
    print(len(kept), "calls;", "rejected:", rej); print(json.dumps(pr, indent=1))


if __name__ == "__main__":
    main()
