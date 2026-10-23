"""Silero VAD (ONNX) over a whole channel, returning speech segments in seconds."""

import hashlib
import sys
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort

SR = 16000
CHUNK = 512  # 32 ms at 16 kHz, what Silero expects
CONTEXT = 64
MODEL = Path(__file__).resolve().parent.parent / "models" / "silero_vad.onnx"
# Pinned: every number in the README was produced with this exact file (Silero VAD v6.2).
MODEL_URL = "https://github.com/snakers4/silero-vad/raw/v6.2.3/src/silero_vad/data/silero_vad.onnx"
MODEL_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
CACHE = Path.home() / ".cache" / "callcheck" / "silero_vad.onnx"


def model_path():
    """Repo-local models/ first, then the user cache. Downloads once and checks the hash."""
    if MODEL.exists():
        return MODEL
    if not CACHE.exists():
        print(f"Downloading Silero VAD v6.2 to {CACHE} (one time, about 2 MB)", file=sys.stderr)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        if hashlib.sha256(tmp.read_bytes()).hexdigest() != MODEL_SHA256:
            tmp.unlink()
            raise RuntimeError(f"{MODEL_URL} does not match the pinned hash")
        tmp.rename(CACHE)
    return CACHE


class SileroVAD:
    def __init__(self, path=None):
        path = path or model_path()
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"], sess_options=opts)

    def probs(self, audio):
        """Speech probability for every 32 ms chunk of a mono float32 16 kHz signal."""
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, CONTEXT), dtype=np.float32)
        n = len(audio) // CHUNK
        out = np.empty(n, dtype=np.float32)
        sr = np.array(SR, dtype=np.int64)
        for i in range(n):
            x = np.concatenate([context, audio[i * CHUNK:(i + 1) * CHUNK].reshape(1, -1)], axis=1).astype(np.float32)
            p, state = self.session.run(None, {"input": x, "state": state, "sr": sr})
            context = x[:, -CONTEXT:]
            out[i] = p[0][0]
        return out

    def segments(self, audio, on=0.5, off=0.35, min_speech_s=0.1, min_silence_s=0.1):
        """Hysteresis over the probabilities: open at `on`, close after `min_silence_s` below `off`."""
        probs = self.probs(audio)
        step = CHUNK / SR
        segs, start, quiet = [], None, 0
        for i, p in enumerate(probs):
            if start is None:
                if p >= on:
                    start, quiet = i, 0
            elif p < off:
                quiet += 1
                if quiet * step >= min_silence_s:
                    end = i - quiet + 1
                    if (end - start) * step >= min_speech_s:
                        segs.append((start * step, end * step))
                    start = None
            else:
                quiet = 0
        if start is not None and (len(probs) - start) * step >= min_speech_s:
            segs.append((start * step, len(probs) * step))
        return segs


FLOOR_K = 4.0  # a sound must be this many times louder than the channel's noise floor


def sound_segments(audio, rms_on=0.02, frame_s=0.02, min_gap_s=0.1):
    """Any loud sound, speech or not (coughs, bumps). Used next to VAD, not instead of it.

    The threshold is rms_on, raised to FLOOR_K times the channel's noise floor when the line
    is noisy, so steady background noise does not read as one long sound.
    """
    n = int(frame_s * SR)
    frames = audio[: len(audio) // n * n].reshape(-1, n)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    floor = np.percentile(rms, 10) if len(rms) else 0.0  # quietest frames: the line with nobody talking
    loud = rms > max(rms_on, FLOOR_K * floor)
    segs, start, gap = [], None, 0
    for i, v in enumerate(loud):
        if v:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap * frame_s >= min_gap_s:
                segs.append((start * frame_s, (i - gap + 1) * frame_s))
                start = None
    if start is not None:
        segs.append((start * frame_s, len(loud) * frame_s))
    return segs
