"""Generate synthetic two-channel voice-agent calls with planted faults.

Each call is a stereo WAV (left = caller, right = agent) plus a ground-truth JSON
that records every turn, every planted fault, and the phone number the caller
said vs what the agent read back. A phone-line copy (300-3400 Hz, 8 kHz mu-law)
is written next to each wideband file.

Speech comes from macOS `say`, so this runs offline with no API keys.

    python synth/generate.py --calls 24 --seed 7 --out data/synthetic
"""

import argparse
import json
import random
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

SR = 16000
AGENT_VOICE = "Samantha"
CALLER_VOICES = [
    "Reed (English (US))", "Daniel", "Karen", "Moira",
    "Rocko (English (US))", "Sandy (English (US))", "Shelley (English (UK))",
]

# Thresholds the ground truth is labelled against. callcheck uses the same ones.
SLOW_RESPONSE_S = 1.5
DEAD_AIR_S = 3.0

FIRST = ["Maria", "James", "Priya", "Daniel", "Aisha", "Tom", "Elena", "Kevin", "Grace", "Omar"]
LAST = ["Lopez", "Carter", "Shah", "Nguyen", "Okafor", "Brooks", "Rossi", "Walsh", "Kim", "Haddad"]
BUSINESSES = ["Northside Clinic", "Harbor Insurance", "Metro Auto Care", "Summit Legal", "Lakeview Dental"]
ISSUES = [
    ("I was in a car accident last week,", "and the other driver's insurance won't call me back."),
    ("I need to move my appointment,", "because I'm out of town on Thursday."),
    ("my car's been making a grinding noise,", "mostly when I brake on the highway."),
    ("I got a bill I don't understand,", "it's for a visit I never had."),
    ("I slipped on the stairs at my building,", "and my wrist still hurts."),
]
DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

FAULTS = ["cutoff", "slow_response", "dead_air", "false_barge_in", "wrong_readback"]

_tts_cache = {}


def tts(voice, text):
    """Synthesize `text` with macOS say and return float32 mono at 16 kHz, trimmed."""
    key = (voice, text)
    if key in _tts_cache:
        return _tts_cache[key]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "t.wav"
        subprocess.run(
            ["say", "-v", voice, "-o", str(path), "--file-format=WAVE", "--data-format=LEI16@16000", text],
            check=True,
        )
        with wave.open(str(path)) as w:
            audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    loud = np.flatnonzero(np.abs(audio) > 0.01)
    audio = audio[loud[0]: loud[-1] + 1] if len(loud) else audio
    _tts_cache[key] = audio
    return audio


def cough(rng):
    """A 250 ms band-limited noise burst with a fast attack, standing in for a cough."""
    n = int(0.25 * SR)
    noise = rng.standard_normal(n).astype(np.float32)
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    spec[(freqs < 300) | (freqs > 3000)] = 0
    burst = np.fft.irfft(spec, n).astype(np.float32)
    env = np.exp(-np.linspace(0, 6, n)).astype(np.float32)
    burst *= env
    return 0.5 * burst / np.max(np.abs(burst))


class Call:
    def __init__(self, call_id, rng, faults):
        self.id = call_id
        self.rng = rng
        self.faults = set(faults)
        self.caller_voice = rng.choice(CALLER_VOICES)
        self.caller = np.zeros(SR * 120, dtype=np.float32)
        self.agent = np.zeros(SR * 120, dtype=np.float32)
        self.t = 0.0
        self.last_caller_end = None
        self.turns = []
        self.events = []
        self.latencies = []

    def _place(self, channel, audio, start):
        i = int(start * SR)
        buf = self.caller if channel == "caller" else self.agent
        buf[i: i + len(audio)] += audio
        return start + len(audio) / SR

    def say_caller(self, parts, gap=None):
        """parts: list of strings and float pauses. Returns list of (start, end) per spoken part."""
        start = self.t + (gap if gap is not None else self.rng.uniform(0.3, 0.8))
        t = start
        spans = []
        for p in parts:
            if isinstance(p, float):
                t += p
                continue
            end = self._place("caller", tts(self.caller_voice, p), t)
            spans.append((t, end))
            t = end
        text = " ".join(p for p in parts if isinstance(p, str))
        self.turns.append({"speaker": "caller", "start": round(start, 3), "end": round(t, 3), "text": text})
        self.t = t
        self.last_caller_end = t
        return spans

    def say_agent(self, text, latency=None, label=None, truncate_at=None):
        if latency is None:
            latency = self.rng.uniform(0.45, 0.95)
        start = self.t + latency
        audio = tts(AGENT_VOICE, text)
        if truncate_at is not None:
            audio = audio[: int((truncate_at - start) * SR)]
        end = self._place("agent", audio, start)
        self.turns.append({"speaker": "agent", "start": round(start, 3), "end": round(end, 3), "text": text,
                           **({"truncated": True} if truncate_at is not None else {})})
        # Response latency only exists when this agent turn answers a caller turn.
        answers_caller = len(self.turns) >= 2 and self.turns[-2]["speaker"] == "caller"
        if self.last_caller_end is not None and answers_caller:
            self.latencies.append({"turn": len(self.turns) - 1, "latency_s": round(start - self.last_caller_end, 3)})
        self.t = end
        return start, end

    def event(self, kind, time, **detail):
        self.events.append({"type": kind, "time": round(time, 3), **detail})


def digits_spoken(d):
    return " ".join(DIGIT_WORDS[int(c)] for c in d)


def build(call_id, rng, faults):
    c = Call(call_id, rng, faults)
    biz = rng.choice(BUSINESSES)
    first, last = rng.choice(FIRST), rng.choice(LAST)
    phone = "".join(str(rng.randrange(10)) for _ in range(10))
    issue = rng.choice(ISSUES)

    # Which agent turn gets the slow / dead-air latency, and where the cut-off lands.
    slow_turn = rng.choice([1, 3, 4]) if "slow_response" in faults else None
    dead_turn = rng.choice([1, 3, 4]) if "dead_air" in faults else None
    if slow_turn is not None and slow_turn == dead_turn:
        dead_turn = 4 if slow_turn != 4 else 3
    cutoff_where = rng.choice(["phone", "issue"]) if "cutoff" in faults else None

    def lat(turn_no):
        if turn_no == dead_turn:
            return rng.uniform(3.2, 5.0)
        if turn_no == slow_turn:
            return rng.uniform(1.7, 2.6)
        return None

    def mark_latency(turn_no):
        last = c.latencies[-1]["latency_s"]
        if turn_no == dead_turn:
            c.event("dead_air", c.turns[-1]["start"], latency_s=last)
        elif turn_no == slow_turn:
            c.event("slow_response", c.turns[-1]["start"], latency_s=last)

    c.t = 0.3
    c.say_agent(f"Thanks for calling {biz}. I'm an AI assistant. Can I get your first and last name?", latency=0.0)
    c.say_caller([f"Yeah, it's {first} {last}."])
    c.say_agent(f"Thanks, {first}. And what's the best phone number to reach you?", latency=lat(1))
    mark_latency(1)

    # Caller reads the number in three groups with thinking pauses, like real people do.
    p1, p2 = rng.uniform(0.9, 1.3), rng.uniform(0.6, 1.0)
    spans = c.say_caller([
        f"Sure, it's {digits_spoken(phone[:3])},", p1,
        f"{digits_spoken(phone[3:6])},", p2,
        f"{digits_spoken(phone[6:])}.",
    ])
    if cutoff_where == "phone":
        jump = spans[0][1] + rng.uniform(0.35, 0.6)
        audio = tts(AGENT_VOICE, "Sorry, could you repeat the full number?")
        c._place("agent", audio, jump)
        c.turns.append({"speaker": "agent", "start": round(jump, 3), "end": round(jump + len(audio) / SR, 3),
                        "text": "Sorry, could you repeat the full number?", "interrupts_caller": True})
        c.event("cutoff", jump, caller_still_talking_until=round(c.last_caller_end, 3))
        c.t = max(c.t, jump + len(audio) / SR)
        c.last_caller_end = None  # next agent turn follows agent speech, so it has no response latency

    readback = phone
    if "wrong_readback" in faults:
        i = rng.randrange(10)
        wrong = str((int(phone[i]) + rng.randrange(1, 10)) % 10)
        readback = phone[:i] + wrong + phone[i + 1:]
    c.say_agent(f"Let me read that back. {digits_spoken(readback[:3])}, {digits_spoken(readback[3:6])}, "
                f"{digits_spoken(readback[6:])}. Is that right?")
    if readback != phone:
        c.event("wrong_readback", c.turns[-1]["start"], said=phone, read_back=readback)
        c.say_caller([f"No, it's {digits_spoken(phone[:3])}, {digits_spoken(phone[3:6])}, {digits_spoken(phone[6:])}."])
        c.say_agent(f"Sorry about that. {digits_spoken(phone[:3])}, {digits_spoken(phone[3:6])}, "
                    f"{digits_spoken(phone[6:])}.")
        c.say_caller(["Yes, that's it."])
    else:
        c.say_caller(["Yes, that's right."])

    c.say_agent("Got it. And briefly, what's the reason for your call?", latency=lat(3))
    mark_latency(3)

    pause = rng.uniform(1.0, 1.6)
    spans = c.say_caller([f"Well, {issue[0]}", pause, issue[1]])
    if cutoff_where == "issue":
        jump = spans[0][1] + rng.uniform(0.35, 0.6)
        text = "Okay. I can help with that."
        audio = tts(AGENT_VOICE, text)
        c._place("agent", audio, jump)
        c.turns.append({"speaker": "agent", "start": round(jump, 3), "end": round(jump + len(audio) / SR, 3),
                        "text": text, "interrupts_caller": True})
        c.event("cutoff", jump, caller_still_talking_until=round(c.last_caller_end, 3))
        c.t = max(c.t, jump + len(audio) / SR)
        c.last_caller_end = None

    reply = "I'm sorry to hear that. I'm going to connect you with someone on our team who can help right now."
    if "false_barge_in" in faults:
        latency = lat(4) or rng.uniform(0.45, 0.95)
        start = c.t + latency
        noise_at = start + rng.uniform(0.8, 1.4)
        c._place("caller", cough(np.random.default_rng(c.rng.randrange(2**32))), noise_at)
        c.say_agent(reply, latency=latency, truncate_at=noise_at + 0.3)
        mark_latency(4)
        c.event("false_barge_in", noise_at, agent_stopped_at=round(noise_at + 0.3, 3))
        c.last_caller_end = None  # the cough is not a caller turn
        c.say_agent("Sorry, go ahead.", latency=0.8)
        c.say_caller(["Oh, no, that was just a cough."], gap=1.2)
        c.say_agent("No problem. I'll connect you with someone on our team now.")
    else:
        c.say_agent(reply, latency=lat(4))
        mark_latency(4)

    c.say_caller(["Okay, thanks."])
    c.say_agent("Thanks for calling. Goodbye.")

    end = int((c.t + 0.5) * SR)
    stereo = np.stack([c.caller[:end], c.agent[:end]], axis=1)
    # A quiet noise floor on both lines, so silence is never digital zero.
    stereo += 0.002 * np.random.default_rng(call_id).standard_normal(stereo.shape).astype(np.float32)
    check_labels(c)
    truth = {
        "call_id": f"call_{call_id:03d}",
        "sample_rate": SR,
        "channels": {"left": "caller", "right": "agent"},
        "caller_voice": c.caller_voice,
        "duration_s": round(end / SR, 3),
        "planted_faults": sorted(faults),
        "thresholds": {"slow_response_s": SLOW_RESPONSE_S, "dead_air_s": DEAD_AIR_S},
        "entities": {"name": f"{first} {last}", "phone_said": phone, "phone_read_back": readback},
        "turns": c.turns,
        "agent_latencies": c.latencies,
        "events": sorted(c.events, key=lambda e: e["time"]),
    }
    return stereo, truth


def check_labels(c):
    """Fail loudly if a latency in the ground truth disagrees with its label."""
    labelled = {e["time"]: e["type"] for e in c.events if e["type"] in ("slow_response", "dead_air")}
    for item in c.latencies:
        start = c.turns[item["turn"]]["start"]
        kind = labelled.get(start)
        lat_s = item["latency_s"]
        if kind == "dead_air":
            assert lat_s > DEAD_AIR_S, (c.id, item)
        elif kind == "slow_response":
            assert SLOW_RESPONSE_S < lat_s <= DEAD_AIR_S, (c.id, item)
        else:
            assert lat_s < SLOW_RESPONSE_S, (c.id, item, "unlabelled slow turn")
    assert len(labelled) == sum(e["type"] in ("slow_response", "dead_air") for e in c.events)


def write_wav(path, stereo):
    pcm = (np.clip(stereo, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def phone_copy(src, dst):
    """Simulate a phone line: 300-3400 Hz band, 8 kHz, mu-law."""
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(src),
         "-af", "highpass=f=300,lowpass=f=3400", "-ar", "8000", "-acodec", "pcm_mulaw", str(dst)],
        check=True,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calls", type=int, default=24)
    ap.add_argument("--clean", type=int, default=4, help="calls with no planted faults (false-positive controls)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/synthetic")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    for sub in ("wideband", "phone", "truth"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    manifest = []
    for i in range(args.calls):
        if i < args.clean:
            faults = []
        else:
            faults = [f for f in FAULTS if rng.random() < 0.4] or [rng.choice(FAULTS)]
        stereo, truth = build(i, random.Random(rng.random()), faults)
        name = truth["call_id"]
        write_wav(out / "wideband" / f"{name}.wav", stereo)
        phone_copy(out / "wideband" / f"{name}.wav", out / "phone" / f"{name}.wav")
        (out / "truth" / f"{name}.json").write_text(json.dumps(truth, indent=2))
        manifest.append({"call_id": name, "faults": truth["planted_faults"], "duration_s": truth["duration_s"]})
        print(f"{name}  {truth['duration_s']:5.1f}s  {', '.join(truth['planted_faults']) or 'clean'}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    counts = {f: sum(f in m["faults"] for m in manifest) for f in FAULTS}
    print("planted:", counts, "| clean:", sum(not m["faults"] for m in manifest))


if __name__ == "__main__":
    main()
