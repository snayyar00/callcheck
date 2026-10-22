"""Find the turn-taking failures callers notice in a two-channel call recording.

Input is a stereo recording with the caller on one channel and the agent on the
other (most telephony platforms can export this). Output is every agent
response latency plus a list of events:

- slow_response   agent answered after more than `slow_s`
- dead_air        agent answered after more than `dead_s`
- cutoff          agent started talking in a caller's pause and the caller kept going
- false_barge_in  a short non-speech sound on the caller line stopped the agent
"""

import subprocess
from dataclasses import dataclass, field

import numpy as np

from .vad import SR, SileroVAD, sound_segments


@dataclass
class Config:
    slow_s: float = 1.5
    dead_s: float = 3.0
    caller_turn_gap_s: float = 2.0   # caller pauses shorter than this stay in one turn
    agent_turn_gap_s: float = 1.0
    min_speech_s: float = 0.35       # shorter caller sounds are not treated as words
    cutoff_window_s: float = 2.0     # agent start this soon after caller speech = started in a pause
    barge_stop_s: float = 1.0        # agent must stop within this long after the sound
    barge_quiet_s: float = 0.4       # and stay quiet at least this long


@dataclass
class Result:
    duration_s: float
    segments: dict
    latencies: list = field(default_factory=list)
    events: list = field(default_factory=list)


def load(path, caller_channel=0):
    """Decode any audio file to float32 16 kHz. Returns (caller, agent)."""
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"],
        capture_output=True, check=True,
    ).stdout
    stereo = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2)
    caller, agent = stereo[:, caller_channel], stereo[:, 1 - caller_channel]
    return np.ascontiguousarray(caller), np.ascontiguousarray(agent)


def union(segs):
    """Sort and join overlapping (start, end) segments."""
    out = []
    for s, e in sorted(segs):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def merge(segs, gap_s, other_starts):
    """Join segments on one channel when the pause is short and the other side did not start talking in it."""
    turns = []
    for s, e in segs:
        if turns and s - turns[-1][1] < gap_s and not any(turns[-1][1] <= o < s for o in other_starts):
            turns[-1][1] = e
        else:
            turns.append([s, e])
    return [tuple(t) for t in turns]


def analyze(caller, agent, cfg=Config(), vad=None):
    vad = vad or SileroVAD()
    # VAD first; a sustained loud sound VAD missed (>= min_speech_s, longer than a cough) also counts
    # as the caller holding the floor. Silero under-detects some voices, especially on phone audio.
    # Pieces of any length are joined into turns first; only whole turns shorter than min_speech_s
    # (a lone cough or bump) are dropped below. A short last word still extends its turn.
    c_segs = union(vad.segments(caller) + sound_segments(caller, min_gap_s=0.25))
    a_segs = vad.segments(agent)
    # Caller turns: join pieces first, then drop turns too short to be words (a cough, a bump).
    # Judging length per piece would drop a quick "Yes, that's right" that VAD hears as two blips.
    c_turns = [t for t in merge(c_segs, cfg.caller_turn_gap_s, [s for s, _ in a_segs])
               if t[1] - t[0] >= cfg.min_speech_s]
    c_words = [seg for seg in c_segs if any(ts <= seg[0] and seg[1] <= te for ts, te in c_turns)]
    # Agent turns join across short gaps even if the caller talks in them: an agent does not
    # yield the floor in a 200 ms breath, and splitting there hides cut-offs.
    a_turns = merge(a_segs, cfg.agent_turn_gap_s, [])
    res = Result(duration_s=len(caller) / SR,
                 segments={"caller": c_segs, "agent": a_segs, "caller_turns": c_turns, "agent_turns": a_turns})

    # Response latency: agent turns whose last preceding sound was the caller finishing.
    for a_start, _ in a_turns:
        prev_c = [t for t in c_turns if t[1] <= a_start]
        if not prev_c:
            continue
        c_end = prev_c[-1][1]
        if any(c_end < s < a_start for s, _ in a_turns) or any(s < a_start < e for s, e in c_turns):
            continue
        lat = a_start - c_end
        res.latencies.append({"agent_start": round(a_start, 3), "latency_s": round(lat, 3)})
        if lat > cfg.dead_s:
            res.events.append({"type": "dead_air", "time": round(a_start, 3), "latency_s": round(lat, 3)})
        elif lat > cfg.slow_s:
            res.events.append({"type": "slow_response", "time": round(a_start, 3), "latency_s": round(lat, 3)})

    # Cut-off: the agent started inside a caller pause, and the caller then kept talking over it.
    for a_start, a_end in a_turns:
        spoke_recently = any(a_start - cfg.cutoff_window_s <= e <= a_start for _, e in c_words)
        resumed = [s for s, _ in c_words if a_start + 0.1 < s < a_end]
        if spoke_recently and resumed:
            res.events.append({"type": "cutoff", "time": round(a_start, 3), "caller_resumed_at": round(resumed[0], 3)})

    # False barge-in: a short sound that is not speech, then the agent stops and stays quiet.
    for s, e in sound_segments(caller):
        if e - s >= cfg.min_speech_s or any(ws < e and s < we for ws, we in c_words):
            continue
        # The agent was talking: inside a segment, or in a breath gap within the last 300 ms.
        if not any(a_s < s and a_e > s - 0.3 for a_s, a_e in a_segs):
            continue
        # Where did the agent next go quiet for at least barge_quiet_s?
        stop = None
        for i, (a_s, a_e) in enumerate(a_segs):
            if a_e < s:
                continue
            nxt = a_segs[i + 1][0] if i + 1 < len(a_segs) else None
            if nxt is None or nxt - a_e >= cfg.barge_quiet_s:
                stop = a_e
                break
        if stop is None:
            prior = [a_e for a_s, a_e in a_segs if a_s < s]
            stop = prior[-1] if prior else None
        if stop is not None and 0 <= stop - s <= cfg.barge_stop_s or (stop is not None and s - 0.3 < stop < s):
            res.events.append({"type": "false_barge_in", "time": round(s, 3), "agent_stopped_at": round(stop, 3)})

    res.events.sort(key=lambda ev: ev["time"])
    return res
