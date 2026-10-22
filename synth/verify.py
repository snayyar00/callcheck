"""Check that the synthetic audio really contains what the ground truth says.

This reads the WAV files back from disk, finds speech on each channel with a
plain energy detector (no VAD model, so it is independent of what callcheck
uses), and checks every labelled turn against it:

- each turn's start and end lines up with audio on the right channel
- the other channel is silent where the truth says only one side talks,
  except for planted cut-offs and coughs
- the phone-line copies are 8 kHz mu-law stereo and the same length

    python synth/verify.py data/synthetic
"""

import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

FRAME_S = 0.02
TOL_S = 0.06  # one frame of slop on each side, plus rounding


def speech_frames(channel, sr):
    n = int(FRAME_S * sr)
    frames = channel[: len(channel) // n * n].reshape(-1, n)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    return rms > 0.01  # noise floor is about 0.002 RMS


def frame(t):
    return int(round(t / FRAME_S))


def check_call(wav_path, truth):
    with wave.open(str(wav_path)) as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, 2) / 32768
    active = {"caller": speech_frames(pcm[:, 0], sr), "agent": speech_frames(pcm[:, 1], sr)}
    errors = []

    for k, turn in enumerate(truth["turns"]):
        ch = active[turn["speaker"]]
        s, e = frame(turn["start"]), frame(turn["end"])
        tol = frame(TOL_S)
        # Audio must start within tolerance of the labelled start ...
        head = ch[max(0, s - tol): s + tol + 1]
        if not head.any():
            errors.append(f"turn {k} ({turn['speaker']}): no audio near start {turn['start']}")
        # ... and must not run on past the labelled end.
        tail = ch[e + tol: e + tol + frame(0.2)]
        if tail.any() and k + 1 < len(truth["turns"]) and truth["turns"][k + 1]["speaker"] != turn["speaker"]:
            errors.append(f"turn {k} ({turn['speaker']}): audio continues after end {turn['end']}")
        # Nothing on this channel in the gap before the turn starts (since its previous turn).
        prev_end = max((t["end"] for t in truth["turns"][:k] if t["speaker"] == turn["speaker"]), default=0)
        gap = ch[frame(prev_end) + tol: s - tol]
        if gap.any():
            # The only allowed sound in a caller gap is a planted cough.
            coughs = [frame(e_["time"]) for e_ in truth["events"] if e_["type"] == "false_barge_in"]
            idx = np.flatnonzero(gap) + frame(prev_end) + tol
            if not (turn["speaker"] == "caller" and coughs and all(any(abs(i - c) <= frame(0.3) for c in coughs) for i in idx)):
                errors.append(f"turn {k} ({turn['speaker']}): unexpected audio before start {turn['start']}")

    # Overlap only where a cut-off or cough was planted.
    both = active["caller"] & active["agent"]
    allowed = np.zeros_like(both)
    for t in truth["turns"]:
        if t.get("interrupts_caller"):
            allowed[frame(t["start"]) - 2: frame(t["end"]) + 2] = True
    for e_ in truth["events"]:
        if e_["type"] == "false_barge_in":
            allowed[frame(e_["time"]) - 2: frame(e_["agent_stopped_at"]) + 2] = True
    stray = np.flatnonzero(both & ~allowed)
    if len(stray):
        errors.append(f"unplanned overlap at {stray[0] * FRAME_S:.2f}s ({len(stray)} frames)")
    if truth["events"] and any(e_["type"] == "cutoff" for e_ in truth["events"]) and not (both & allowed).any():
        errors.append("cut-off planted but no overlap found")
    return errors


def check_phone(wb_path, ph_path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,sample_rate,channels,duration",
         "-of", "json", str(ph_path)],
        capture_output=True, text=True, check=True,
    )
    st = json.loads(out.stdout)["streams"][0]
    with wave.open(str(wb_path)) as w:
        wb_dur = w.getnframes() / w.getframerate()
    errs = []
    if (st["codec_name"], st["sample_rate"], st["channels"]) != ("pcm_mulaw", "8000", 2):
        errs.append(f"phone copy format {st['codec_name']} {st['sample_rate']} {st['channels']}ch")
    if abs(float(st["duration"]) - wb_dur) > 0.05:
        errs.append(f"phone copy length {st['duration']} vs {wb_dur:.3f}")
    return errs


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic")
    total, bad = 0, 0
    for truth_path in sorted((root / "truth").glob("*.json")):
        truth = json.loads(truth_path.read_text())
        name = truth["call_id"]
        errs = check_call(root / "wideband" / f"{name}.wav", truth)
        errs += check_phone(root / "wideband" / f"{name}.wav", root / "phone" / f"{name}.wav")
        total += 1
        if errs:
            bad += 1
            print(f"FAIL {name}: " + "; ".join(errs))
        else:
            print(f"ok   {name}  {len(truth['turns'])} turns, events: "
                  + (", ".join(e['type'] for e in truth['events']) or "none"))
    print(f"\n{total - bad}/{total} calls match their ground truth")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
