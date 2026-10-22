"""Build 2-party stereo "calls" + truth JSON from AMI headset audio and manual word times.

Rules (all documented in truth JSON under "rules"):
- Turn: a speaker's words (punctuation tokens dropped) merged while the gap between one word's end and
  the next word's start is < TURN_GAP (0.5 s). Vocal sounds (laugh/cough/...) are NOT words and never
  extend a turn.
- Floor-holding fillers (uh, um) are NOT backchannel words.
- Backchannel turn: <= 3 words, all in BACKCHANNEL_WORDS, turn duration < 1.0 s.
- Response latency: for each turn of B (the "agent") that starts at b_s, the previous SUBSTANTIVE (non-backchannel)
  A turn that ended at a_e <= b_s, with no A turn covering b_s, no B turn starting between a_e and b_s, and B not
  already talking at a_e. latency = b_s - a_e.
  Flag "backchannel" if the B turn is a backchannel turn.
- Overlap: B turn starts while an A turn is running (a_s < b_s < a_e). Kind = "backchannel" if backchannel
  turn and overlap < 1 s, else "interruption".
- Cut-off truth (mirrors the product meaning, uses callcheck's 1.0 s agent turn gap on human words):
  a non-backchannel B turn (merged at 1.0 s) whose start b_s is within 2 s after A's last word end (or inside A speech)
  and A has a word starting in (b_s+0.1, b_e). "cutoffs" = that broad set (B may talk over A mid-word);
  "cutoffs_pause" = the subset where b_s falls in a real gap between A words (matches callcheck's "starts in a pause" meaning).
- Vocal sounds: every <vocalsound> on A and B, in clip time.
"""
import json
import re
import sys
import wave
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AMI = ROOT / "data/.corpora/ami"
OUT = ROOT / "data/ami_pairs"
SR = 16000
TURN_GAP = 0.5
AGENT_GAP = 1.0
WIN = 240.0
BACKCHANNEL_WORDS = {"mm", "mhm", "hmm", "mm-hmm", "mmhmm", "uh-huh", "uhhuh", "yeah", "yep", "yes", "right", "okay",
                     "ok", "sure", "oh", "ah", "'kay", "kay", "huh", "mm-hm", "hm", "aha", "yeah.", "alright", "true"}
MEETINGS = ["ES2002a", "ES2008c", "ES2012b", "IS1009b", "IS1003b", "TS3005a", "ES2010c", "ES2014b", "IS1008c"]
PER_MEETING = 4


def norm(w):
    return re.sub(r"[^a-z\-']", "", w.lower())


def load_speakers(meeting):
    tree = ET.parse(AMI / "ann/corpusResources/meetings.xml")
    for m in tree.getroot().iter("meeting"):
        if m.get("observation") == meeting:
            return {s.get("nxt_agent"): int(s.get("channel")) for s in m.iter("speaker")}
    raise KeyError(meeting)


def load_words(meeting, letter):
    root = ET.parse(AMI / f"ann/words/{meeting}.{letter}.words.xml").getroot()
    words, vocal = [], []
    for el in root:
        tag = el.tag.split("}")[-1]
        try:
            s, e = float(el.get("starttime")), float(el.get("endtime"))
        except (TypeError, ValueError):
            continue
        if tag == "w" and el.get("punc") != "true":
            words.append((s, e, norm(el.text or "")))
        elif tag == "vocalsound":
            vocal.append((s, e, el.get("type")))
    words.sort()
    return words, vocal


def turns_of(words, gap):
    out = []
    for s, e, w in words:
        if out and s - out[-1]["end"] < gap:
            out[-1]["end"] = max(out[-1]["end"], e)
            out[-1]["words"].append(w)
        else:
            out.append({"start": s, "end": e, "words": [w]})
    return out


def is_bc(t):
    return len(t["words"]) <= 3 and all(w in BACKCHANNEL_WORDS for w in t["words"]) and t["end"] - t["start"] < 1.0


def pick_windows(speakers_words, dur):
    """Return list of (score, t0, A, B) for 240 s windows, step 30 s."""
    cands = []
    letters = list(speakers_words)
    t0 = 60.0
    while t0 + WIN <= dur - 10:
        t1 = t0 + WIN
        talk = {}
        for l in letters:
            talk[l] = sum(min(e, t1) - max(s, t0) for s, e, _ in speakers_words[l] if e > t0 and s < t1)
        top = sorted(talk, key=talk.get, reverse=True)[:2]
        a, b = top
        tot = sum(talk.values())
        share = (talk[a] + talk[b]) / max(tot, 1e-6)
        bal = min(talk[a], talk[b]) / max(talk[a] + talk[b], 1e-6)
        ta = turns_of([w for w in speakers_words[a] if t0 <= w[0] < t1], TURN_GAP)
        tb = turns_of([w for w in speakers_words[b] if t0 <= w[0] < t1], TURN_GAP)
        alt = sum(1 for x in tb if any(0 < x["start"] - y["end"] < 10 for y in ta)) + \
            sum(1 for x in ta if any(0 < x["start"] - y["end"] < 10 for y in tb))
        score = share * min(1.0, bal / 0.25) * min(1.0, alt / 40)
        if share >= 0.5 and bal >= 0.12 and alt >= 14:
            cands.append((score, t0, a, b))
        t0 += 30
    return cands


def read_wav(path, t0, t1):
    with wave.open(str(path)) as w:
        w.setpos(int(t0 * SR))
        return np.frombuffer(w.readframes(int((t1 - t0) * SR)), dtype=np.int16)


def clip_truth(sw, sv, A, B, t0, t1):
    def sub(l):
        w = [(s - t0, e - t0, x) for s, e, x in sw[l] if s >= t0 and e <= t1]
        v = [(s - t0, e - t0, k) for s, e, k in sv[l] if s >= t0 and e <= t1]
        return w, v
    wa, va = sub(A)
    wb, vb = sub(B)
    ta, tb = turns_of(wa, TURN_GAP), turns_of(wb, TURN_GAP)
    for t in tb:
        t["backchannel"] = is_bc(t)
    for t in ta:
        t["backchannel"] = is_bc(t)
    tb_agent = turns_of(wb, AGENT_GAP)
    for t in tb_agent:
        t["backchannel"] = is_bc(t)

    lat, overlaps = [], []
    for t in tb:
        bs = t["start"]
        if any(a["start"] < bs < a["end"] for a in ta):
            a = next(a for a in ta if a["start"] < bs < a["end"])
            ov = min(a["end"], t["end"]) - bs
            kind = "backchannel" if (t["backchannel"] and ov < 1.0) else "interruption"
            overlaps.append({"b_start": round(bs, 3), "b_end": round(t["end"], 3), "overlap_s": round(ov, 3),
                             "kind": kind, "a_turn": [round(a["start"], 3), round(a["end"], 3)],
                             "a_resumes_after": any(w[0] > bs + 0.1 for w in wa if w[0] > bs and w[0] < t["end"]),
                             "b_words": t["words"]})
            continue
        # reference = last SUBSTANTIVE A turn (caller "mm"/"yeah" acknowledgements do not ask the agent to answer)
        prev = [a for a in ta if a["end"] <= bs and not a["backchannel"]]
        if not prev:
            continue
        ae = prev[-1]["end"]
        if any(ae < x["start"] < bs for x in tb):
            continue
        if any(x["start"] < ae < x["end"] for x in tb):  # agent already talking when the caller stopped
            continue
        lat.append({"b_start": round(bs, 3), "a_end": round(ae, 3), "latency_s": round(bs - ae, 3),
                    "backchannel": t["backchannel"], "b_words": t["words"][:6]})

    cutoff, cutoff_pause = [], []
    for t in tb_agent:
        if t["backchannel"]:
            continue
        bs = t["start"]
        near = any(bs - 2.0 <= e <= bs or s <= bs <= e for s, e, _ in wa)
        resumed = [w[0] for w in wa if bs + 0.1 < w[0] < t["end"]]
        if near and resumed:
            cutoff.append({"b_start": round(bs, 3), "a_resumes_at": round(resumed[0], 3)})
            if not any(w[0] - 0.05 <= bs <= w[1] + 0.05 for w in wa):
                cutoff_pause.append({"b_start": round(bs, 3), "a_resumes_at": round(resumed[0], 3)})

    # false-barge candidates: vocal sound on A while B is talking, with no A word within +-0.3 s
    fb = []
    for s, e, k in va:
        b_talking = any(bt["start"] <= s <= bt["end"] + 0.3 for bt in tb_agent)
        words_near = any(w[0] < e + 0.3 and w[1] > s - 0.3 for w in wa)
        if b_talking and not words_near:
            fb.append({"start": round(s, 3), "end": round(e, 3), "type": k})
    r = lambda ts: [{"start": round(t["start"], 3), "end": round(t["end"], 3), "backchannel": t["backchannel"],
                     "words": t["words"][:8], "n_words": len(t["words"])} for t in ts]
    return {
        "duration_s": round(t1 - t0, 3), "A_turns": r(ta), "B_turns": r(tb), "B_agent_turns": r(tb_agent),
        "latencies": lat, "overlaps": overlaps, "cutoffs": cutoff, "cutoffs_pause": cutoff_pause,
        "vocal_sounds": {"A": [{"start": round(s, 3), "end": round(e, 3), "type": k} for s, e, k in va],
                         "B": [{"start": round(s, 3), "end": round(e, 3), "type": k} for s, e, k in vb]},
        "false_barge_candidates": fb,
        "A_words": [[round(s, 3), round(e, 3), w] for s, e, w in wa],
        "B_words": [[round(s, 3), round(e, 3), w] for s, e, w in wb],
    }


def main():
    (OUT / "wav").mkdir(parents=True, exist_ok=True)
    (OUT / "truth").mkdir(parents=True, exist_ok=True)
    manifest = []
    for m in MEETINGS:
        spk = load_speakers(m)
        sw, sv = {}, {}
        for l in spk:
            sw[l], sv[l] = load_words(m, l)
        dur = max(w[1] for l in sw for w in sw[l])
        cands = sorted(pick_windows(sw, dur), reverse=True)
        chosen = []
        for c in cands:
            if all(abs(c[1] - o[1]) >= WIN for o in chosen):
                chosen.append(c)
            if len(chosen) == PER_MEETING:
                break
        for i, (score, t0, a, b) in enumerate(sorted(chosen, key=lambda c: c[1])):
            # caller = whoever speaks first in window
            fa = min((w[0] for w in sw[a] if w[0] >= t0), default=1e9)
            fb_ = min((w[0] for w in sw[b] if w[0] >= t0), default=1e9)
            if fb_ < fa:
                a, b = b, a
            t1 = t0 + WIN
            ca = read_wav(AMI / f"audio/{m}.Headset-{spk[a]}.wav", t0, t1)
            cb = read_wav(AMI / f"audio/{m}.Headset-{spk[b]}.wav", t0, t1)
            n = min(len(ca), len(cb))
            st = np.stack([ca[:n], cb[:n]], axis=1)
            name = f"{m}_{i}"
            with wave.open(str(OUT / "wav" / f"{name}.wav"), "wb") as w:
                w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
                w.writeframes(st.tobytes())
            tr = clip_truth(sw, sv, a, b, t0, t1)
            tr.update({"clip": name, "meeting": m, "window": [t0, t1], "caller": a, "agent": b,
                       "headsets": {"caller": spk[a], "agent": spk[b]}, "dominance_score": round(score, 3),
                       "license": "CC BY 4.0, AMI Meeting Corpus, https://groups.inf.ed.ac.uk/ami/corpus/license.shtml",
                       "rules": __doc__})
            (OUT / "truth" / f"{name}.json").write_text(json.dumps(tr, indent=1))
            manifest.append({"clip": name, "meeting": m, "t0": t0, "caller": a, "agent": b,
                             "lat": len(tr["latencies"]), "ovl": len(tr["overlaps"]), "cutoff": len(tr["cutoffs"]),
                             "voc_A": len(tr["vocal_sounds"]["A"]), "fb_cand": len(tr["false_barge_candidates"])})
            print(manifest[-1])
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(len(manifest), "clips")


if __name__ == "__main__":
    main()
