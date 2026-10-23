"""Measure how far SpokenWOZ utterance boundaries sit from the voice on the speaker's own channel.

build_real_tel.py removes a constant pad (PAD_START, PAD_END) from every annotated utterance. This
script is where those constants come from: for each annotated turn in the tuning calls, find the
first and last 20 ms frame above the channel's speech threshold within 1 s of the boundary.

    uv run --python 3.12 --with numpy python realdata/telephone/measure_pad.py
"""
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/real_tel"
FRAME = 0.02


def frames_rms(path, ch):
    x = np.frombuffer(subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "s16le", "-ac", "2",
                                      "-ar", "8000", "-"], capture_output=True, check=True).stdout, np.int16)
    x = x.reshape(-1, 2)[:, ch].astype(float)
    n = int(FRAME * 8000); k = len(x) // n
    return np.sqrt((x[: k * n].reshape(k, n) ** 2).mean(1))


def main():
    starts, ends = [], []
    for tp in sorted((DATA / "truth").glob("*.json")):
        t = json.load(open(tp))
        rms = {who: frames_rms(DATA / "wav" / f"{t['id']}.wav", ch) for who, ch in (("caller", 0), ("agent", 1))}
        for turn, raw in zip(t["turns"], t["turns_raw"]):
            r = rms[turn["who"]]
            thr = max(60.0, 0.1 * np.percentile(r, 95))  # same threshold as build_real_tel.onset_error
            a, b = int(raw["start"] / FRAME), int(raw["end"] / FRAME)
            on = [i for i in range(max(0, a - 50), min(len(r), a + 50)) if r[i] > thr]
            off = [i for i in range(max(0, b - 50), min(len(r), b + 50)) if r[i] > thr]
            if on:
                starts.append(on[0] * FRAME - raw["start"])
            if off:
                ends.append(raw["end"] - (off[-1] + 1) * FRAME)
    for name, v in (("annotated start is before voice onset by", starts), ("annotated end is after voice offset by", ends)):
        v = np.array(v)
        print(f"{name}: median {np.median(v):.2f} s, IQR {np.percentile(v, 25):.2f}-{np.percentile(v, 75):.2f} s, n={len(v)} turns")


if __name__ == "__main__":
    main()
