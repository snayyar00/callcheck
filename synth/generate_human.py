"""Generate two-channel calls where the CALLER is real human speech.

Caller turns are LibriSpeech dev-clean utterances (real people, read speech,
CC BY 4.0; see synth/HUMAN_SOURCES.md). The agent is still macOS `say -v Samantha`.
Faults are planted with the same rules as synth/generate.py, except
wrong_readback (the caller's words would not match the digits).

Mid-turn pauses are made by joining two utterances of one speaker with a
0.9-1.6 s gap. The caller is scaled to the same active-speech loudness as the
agent. A 0.002 RMS noise floor goes on both lines. Wideband and phone copies
plus truth JSON are written in the generate.py schema.

    python synth/generate_human.py --calls 40 --clean 6 --seed 11 --out data/human
"""

import argparse
import json
import random
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate as G  # noqa: E402  (helpers only; generate.py is not modified)

SR = G.SR
CORPUS = Path("data/.corpora/ls/LibriSpeech/dev-clean")
INDEX = Path("data/.corpora/dev-clean-index.json")
FAULTS = ["cutoff", "slow_response", "dead_air", "false_barge_in"]


def build_index():
    """speaker -> list of {path, text, dur}; cached as JSON."""
    if INDEX.exists():
        return json.loads(INDEX.read_text())
    import soundfile as sf
    idx = {}
    for trans in sorted(CORPUS.glob("*/*/*.trans.txt")):
        for line in trans.read_text().splitlines():
            uid, text = line.split(" ", 1)
            p = trans.parent / f"{uid}.flac"
            info = sf.info(str(p))
            idx.setdefault(uid.split("-")[0], []).append(
                {"path": str(p), "text": text.lower(), "dur": round(info.duration, 3)})
    INDEX.write_text(json.dumps(idx))
    return idx


def load_utt(item):
    import soundfile as sf
    x, sr = sf.read(item["path"], dtype="float32")
    if sr != SR:
        raise ValueError(sr)
    return x


def trim(x):
    """Cut leading/trailing near-silence AFTER gain, so labelled edges match what an energy detector sees."""
    n = SR // 100
    fr = x[: len(x) // n * n].reshape(-1, n)
    rms = np.sqrt((fr ** 2).mean(axis=1))
    loud = np.flatnonzero(rms > 0.007)
    x = x[loud[0] * n: (loud[-1] + 1) * n].copy()
    f = int(0.005 * SR)
    x[:f] *= np.linspace(0, 1, f)
    x[-f:] *= np.linspace(1, 0, f)
    return x


def active_rms(x):
    n = SR // 50
    fr = x[: len(x) // n * n].reshape(-1, n)
    r = np.sqrt((fr ** 2).mean(axis=1))
    a = r[r > 0.01]
    return float(np.sqrt((a ** 2).mean())) if len(a) else 0.0


class Picker:
    """Hands out non-repeating utterances of one speaker by duration range."""

    def __init__(self, items, rng):
        self.items, self.rng, self.used = items, rng, set()

    def get(self, lo, hi):
        ok = []
        for widen in (1.0, 1.5, 2.0, 3.0, 5.0):  # speakers with no short clips get the nearest longer ones
            ok = [i for i, it in enumerate(self.items)
                  if lo / widen <= it["dur"] <= hi * widen and i not in self.used]
            if ok:
                break
        i = self.rng.choice(ok)
        self.used.add(i)
        return self.items[i]


class HumanCall(G.Call):
    def say_caller_h(self, parts, gap=None):
        """parts: list of (text, audio) tuples and float pauses. Same bookkeeping as Call.say_caller."""
        start = self.t + (gap if gap is not None else self.rng.uniform(0.3, 0.8))
        t, spans, texts = start, [], []
        for p in parts:
            if isinstance(p, float):
                t += p
                continue
            text, audio = p
            audio = trim(audio * self.gain)
            end = self._place("caller", audio, t)
            spans.append((t, end))
            texts.append(text)
            t = end
        self.turns.append({"speaker": "caller", "start": round(start, 3), "end": round(t, 3), "text": " ".join(texts)})
        self.t = self.last_caller_end = t
        return spans


def build(call_id, rng, faults, speaker, items):
    c = HumanCall(call_id, rng, faults)
    c.caller_voice = f"librispeech:{speaker}"
    pick = Picker(items, rng)
    biz = rng.choice(G.BUSINESSES)

    def utt(lo, hi):
        it = pick.get(lo, hi)
        return it["text"], load_utt(it)

    # Agent reference loudness: one TTS line fixes the caller gain for the whole call.
    ref = active_rms(G.tts(G.AGENT_VOICE, f"Thanks for calling {biz}. I'm an AI assistant. Can I get your first and last name?"))
    slots = {
        "name": [utt(1.5, 4.0)],
        "phone": [utt(2.0, 5.0), utt(2.0, 5.0)],
        "yes": [utt(1.0, 3.0)],
        "issue": [utt(2.5, 7.0), utt(2.5, 7.0)],
        "cough": [utt(1.5, 4.0)],
        "thanks": [utt(1.0, 3.0)],
    }
    allaudio = np.concatenate([a for v in slots.values() for _, a in v])
    c.gain = min(ref / active_rms(allaudio), 0.9 / np.abs(allaudio).max())

    slow_turn = rng.choice([1, 3, 4]) if "slow_response" in faults else None
    dead_turn = rng.choice([1, 3, 4]) if "dead_air" in faults else None
    if slow_turn is not None and slow_turn == dead_turn:
        dead_turn = 4 if slow_turn != 4 else 3
    cutoff_where = rng.choice(["phone", "issue"]) if "cutoff" in faults else None

    def lat(n):
        if n == dead_turn:
            return rng.uniform(3.2, 5.0)
        if n == slow_turn:
            return rng.uniform(1.7, 2.6)
        return None

    def mark_latency(n):
        if n not in (dead_turn, slow_turn):
            return
        if not c.latencies or c.latencies[-1]["turn"] != len(c.turns) - 1:
            c.faults.discard("dead_air" if n == dead_turn else "slow_response")
            return
        last = c.latencies[-1]["latency_s"]
        c.event("dead_air" if n == dead_turn else "slow_response", c.turns[-1]["start"], latency_s=last)

    def interrupt(spans, text):
        jump = spans[0][1] + rng.uniform(0.35, 0.6)
        audio = G.tts(G.AGENT_VOICE, text)
        c._place("agent", audio, jump)
        c.turns.append({"speaker": "agent", "start": round(jump, 3), "end": round(jump + len(audio) / SR, 3),
                        "text": text, "interrupts_caller": True})
        c.event("cutoff", jump, caller_still_talking_until=round(c.last_caller_end, 3))
        c.t = max(c.t, jump + len(audio) / SR)

    c.t = 0.3
    c.say_agent(f"Thanks for calling {biz}. I'm an AI assistant. Can I get your first and last name?", latency=0.0)
    c.say_caller_h(slots["name"])
    c.say_agent("Thanks. And what's the best phone number to reach you?", latency=lat(1))
    mark_latency(1)

    spans = c.say_caller_h([slots["phone"][0], rng.uniform(0.9, 1.3), slots["phone"][1]])
    if cutoff_where == "phone":
        interrupt(spans, "Sorry, could you repeat the full number?")

    c.say_agent("Let me read that back. Is that right?")
    c.say_caller_h(slots["yes"])

    c.say_agent("Got it. And briefly, what's the reason for your call?", latency=lat(3))
    mark_latency(3)

    spans = c.say_caller_h([slots["issue"][0], rng.uniform(1.0, 1.6), slots["issue"][1]])
    if cutoff_where == "issue":
        interrupt(spans, "Okay. I can help with that.")

    reply = "I'm sorry to hear that. I'm going to connect you with someone on our team who can help right now."
    if "false_barge_in" in faults:
        latency = lat(4) or rng.uniform(0.45, 0.95)
        start = c.t + latency
        noise_at = start + rng.uniform(0.8, 1.4)
        c._place("caller", G.cough(np.random.default_rng(c.rng.randrange(2**32))), noise_at)
        c.say_agent(reply, latency=latency, truncate_at=noise_at + 0.3)
        mark_latency(4)
        c.event("false_barge_in", noise_at, agent_stopped_at=round(noise_at + 0.3, 3))
        c.say_agent("Sorry, go ahead.", latency=0.8)
        c.say_caller_h(slots["cough"], gap=1.2)
        c.say_agent("No problem. I'll connect you with someone on our team now.")
    else:
        c.say_agent(reply, latency=lat(4))
        mark_latency(4)

    c.say_caller_h(slots["thanks"])
    c.say_agent("Thanks for calling. Goodbye.")

    end = int((c.t + 0.5) * SR)
    stereo = np.stack([c.caller[:end], c.agent[:end]], axis=1)
    stereo += 0.002 * np.random.default_rng(call_id).standard_normal(stereo.shape).astype(np.float32)
    G.check_labels(c)
    truth = {
        "call_id": f"call_{call_id:03d}",
        "sample_rate": SR,
        "channels": {"left": "caller", "right": "agent"},
        "caller_voice": c.caller_voice,
        "caller_source": "LibriSpeech dev-clean (real human, read speech)",
        "duration_s": round(end / SR, 3),
        "planted_faults": sorted(c.faults),
        "thresholds": {"slow_response_s": G.SLOW_RESPONSE_S, "dead_air_s": G.DEAD_AIR_S},
        "entities": {},
        "turns": c.turns,
        "agent_latencies": c.latencies,
        "events": sorted(c.events, key=lambda e: e["time"]),
    }
    return stereo, truth


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calls", type=int, default=40)
    ap.add_argument("--clean", type=int, default=6)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="data/human")
    args = ap.parse_args()

    idx = build_index()
    speakers = sorted(idx)
    rng = random.Random(args.seed)
    order = speakers[:]
    rng.shuffle(order)
    out = Path(args.out)
    for sub in ("wideband", "phone", "truth"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    manifest = []
    for i in range(args.calls):
        faults = [] if i < args.clean else ([f for f in FAULTS if rng.random() < 0.4] or [rng.choice(FAULTS)])
        spk = order[i % len(order)]
        stereo, truth = build(i, random.Random(rng.random()), faults, spk, idx[spk])
        name = truth["call_id"]
        G.write_wav(out / "wideband" / f"{name}.wav", stereo)
        G.phone_copy(out / "wideband" / f"{name}.wav", out / "phone" / f"{name}.wav")
        (out / "truth" / f"{name}.json").write_text(json.dumps(truth, indent=2))
        manifest.append({"call_id": name, "faults": truth["planted_faults"], "duration_s": truth["duration_s"]})
        print(f"{name}  {truth['duration_s']:5.1f}s  spk {spk}  {', '.join(truth['planted_faults']) or 'clean'}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    counts = {f: sum(f in m["faults"] for m in manifest) for f in FAULTS}
    print("planted:", counts, "| clean:", sum(not m["faults"] for m in manifest))


if __name__ == "__main__":
    main()
