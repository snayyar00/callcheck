"""Degrade clean synthetic calls with realistic phone problems, keeping timing intact.

Reads <src>/wideband/*.wav + <src>/truth/*.json, writes <dst>/wideband, <dst>/phone, <dst>/truth.
Truth events and turns are copied unchanged; a "hardening" field records what was applied.

    python synth/harden.py data/final data/hard --seed 1

Per call, independently at random: caller-channel noise (traffic, tv, fan, car) at 5-20 dB SNR,
speakerphone echo of the agent into the caller channel, a quiet caller, clipping, a low-bitrate
Opus round trip, and packet loss (zeroed 20 ms frames). Nothing shifts the file in time.
"""

import argparse
import json
import random
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from generate import phone_copy  # noqa: E402

SR = 16000
NOISE_TYPES = ["traffic", "tv", "fan", "car"]
P = dict(noise=0.85, echo=0.35, quiet=0.35, clip=0.2, codec=0.4, loss=0.3)


def read(path):
    with wave.open(str(path)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    return x.reshape(-1, w.getnchannels())


def write(path, x):
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(x.shape[1]); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def speech_power(x):
    """Mean power over active 20 ms frames (frames within 10 dB of the loud ones)."""
    n = len(x) // 320 * 320
    p = (x[:n].reshape(-1, 320) ** 2).mean(1)
    thr = np.percentile(p, 95) / 10
    act = p[p > thr]
    return float(act.mean()) if len(act) else float(p.mean() + 1e-12)


def shaped(rng, n, fn):
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR); f[0] = f[1]
    return np.fft.irfft(spec * fn(f), n).astype(np.float32)


def pink(rng, n): return shaped(rng, n, lambda f: 1 / np.sqrt(f))
def brown(rng, n): return shaped(rng, n, lambda f: 1 / f)


def lowpass(x, fc):
    return np.fft.irfft(np.fft.rfft(x) * (np.fft.rfftfreq(len(x), 1 / SR) < fc), len(x)).astype(np.float32)


def babble(rng, n, pool, voices):
    """Sum of speech snippets taken from other calls (half time-reversed), as TV or a crowd."""
    out = np.zeros(n, dtype=np.float32)
    for _ in range(voices):
        src = pool[rng.integers(len(pool))]
        ch = src[:, rng.integers(2)]
        i = rng.integers(0, max(1, len(ch) - n)) if len(ch) > n else 0
        seg = ch[i:i + n]
        if rng.random() < 0.5:
            seg = seg[::-1]
        out[:len(seg)] += np.roll(seg, rng.integers(n))[:len(seg)]
    return out


def make_noise(kind, rng, n, pool):
    t = np.arange(n) / SR
    if kind == "traffic":  # rumble + passing-car swells + tyre hiss
        x = lowpass(brown(rng, n), 400) * 3 + 0.3 * lowpass(pink(rng, n), 3000)
        for _ in range(max(1, int(n / SR / 6))):
            c, w = rng.uniform(0, n / SR), rng.uniform(1.0, 3.0)
            x += 2.5 * np.exp(-((t - c) / w) ** 2) * lowpass(pink(rng, n), 2000)
        return x
    if kind == "tv":  # babble + a little music-like tone bed
        x = babble(rng, n, pool, 6) + 0.05 * sum(np.sin(2 * np.pi * f * t) for f in rng.uniform(200, 900, 4))
        return x * (0.7 + 0.3 * np.sin(2 * np.pi * rng.uniform(0.1, 0.4) * t))
    if kind == "fan":  # broadband hiss + harmonic hum
        f0 = rng.uniform(100, 130)
        return pink(rng, n) + 0.4 * sum(np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6)) / k for k in range(1, 6))
    # car interior: strong low rumble, engine harmonics, a bit of wind
    f0 = rng.uniform(40, 70)
    return (lowpass(brown(rng, n), 300) * 4 + 0.3 * sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8))
            + 0.4 * lowpass(pink(rng, n), 2500))


def opus_roundtrip(x, kbps):
    """Each channel is its own mono Opus stream: a joint-stereo encode at <=12 kbps collapses both channels
    into one, which no telephony recorder does (each leg is recorded separately)."""
    cols = []
    with tempfile.TemporaryDirectory() as d:
        for ch in range(x.shape[1]):
            a, b, c = Path(d, f"a{ch}.wav"), Path(d, f"b{ch}.ogg"), Path(d, f"c{ch}.wav")
            write(a, x[:, ch:ch + 1])
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(a), "-c:a", "libopus", "-b:a", f"{kbps}k",
                            "-application", "voip", "-vbr", "on", str(b)], check=True)
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(b), "-ar", str(SR), "-ac", "1",
                            "-c:a", "pcm_s16le", str(c)], check=True)
            y = read(c)[:, 0]
            col = np.zeros(len(x), dtype=np.float32)
            m = min(len(x), len(y))
            col[:m] = y[:m]
            cols.append(col)
    return np.stack(cols, 1)


def harden(x, rng, pool):
    """x: (n, 2) float32, col 0 caller, col 1 agent. Returns (degraded, description)."""
    n = len(x)
    caller, agent = x[:, 0].copy(), x[:, 1].copy()
    info = {}
    if rng.random() < P["quiet"]:
        db = float(rng.uniform(6, 15))
        caller *= 10 ** (-db / 20); info["quiet_caller_db"] = round(-db, 1)
    if rng.random() < P["noise"]:
        kind = NOISE_TYPES[int(rng.integers(len(NOISE_TYPES)))]
        snr = float(rng.uniform(5, 20))
        nz = make_noise(kind, rng, n, pool)
        nz *= np.sqrt(speech_power(caller) / 10 ** (snr / 10) / (float((nz ** 2).mean()) + 1e-12))
        caller += nz; info["noise"] = {"type": kind, "snr_db": round(snr, 1)}
    if rng.random() < P["echo"]:
        db, ms = float(rng.uniform(15, 25)), float(rng.uniform(20, 80))
        d = int(ms / 1000 * SR)
        caller[d:] += agent[:-d] * 10 ** (-db / 20)
        info["echo"] = {"level_db": round(-db, 1), "delay_ms": round(ms, 1)}
    if rng.random() < P["clip"]:
        pct = float(rng.uniform(1, 4))
        lim = np.percentile(np.abs(caller), 100 - pct)
        caller = np.clip(caller, -lim, lim); info["clipping"] = {"pct_samples": round(pct, 1)}
    y = np.stack([caller, agent], 1)
    if rng.random() < P["codec"]:
        kbps = int(rng.choice([8, 10, 12]))
        y = opus_roundtrip(y, kbps); info["codec"] = {"name": "opus", "kbps": kbps}
    if rng.random() < P["loss"]:
        pct = float(rng.uniform(1, 3))
        fr = n // 320
        for ch in (0, 1):
            for i in np.flatnonzero(rng.random(fr) < pct / 100):
                y[i * 320:(i + 1) * 320, ch] = 0
        info["packet_loss_pct"] = round(pct, 1)
    return np.clip(y, -1, 1).astype(np.float32), info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    for s in ("wideband", "phone", "truth"):
        (dst / s).mkdir(parents=True, exist_ok=True)
    files = sorted((src / "wideband").glob("*.wav"))
    pool = [read(f) for f in files]
    for k, f in enumerate(files):
        rng = np.random.default_rng([a.seed, k])
        x = pool[k]
        y, info = harden(x, rng, pool)
        assert y.shape == x.shape
        write(dst / "wideband" / f.name, y)
        phone_copy(dst / "wideband" / f.name, dst / "phone" / f.name)
        truth = json.loads((src / "truth" / f"{f.stem}.json").read_text())
        truth["hardening"] = {"seed": a.seed, **info}
        (dst / "truth" / f"{f.stem}.json").write_text(json.dumps(truth, indent=2))
        print(f.stem, json.dumps(info))


if __name__ == "__main__":
    main()
