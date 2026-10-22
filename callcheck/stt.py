"""Speech to text for one channel, with an on-disk cache.

    transcribe(audio) -> [{"start", "end", "text", "words": [{"start", "end", "word"}]}]

Uses faster-whisper `base.en` on CPU (int8) with word timestamps. The cache key
is a hash of the audio samples, so a rerun on the same file and channel is fast.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

CACHE_DIR = Path("data/.stt_cache")
MODEL_NAME = "base.en"

_model = None


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
    return _model


def _key(audio):
    pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    return hashlib.sha1(pcm.tobytes() + MODEL_NAME.encode()).hexdigest()[:20]


def transcribe(audio, cache_dir=CACHE_DIR):
    """Transcribe float32 16 kHz mono audio. Returns a list of segment dicts."""
    path = Path(cache_dir) / f"{_key(audio)}.json"
    if path.exists():
        return json.loads(path.read_text())
    segments, _ = _get_model().transcribe(
        audio.astype(np.float32), language="en", word_timestamps=True,
        beam_size=5, condition_on_previous_text=False,
    )
    out = []
    for s in segments:
        words = [{"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word.strip()}
                 for w in (s.words or [])]
        out.append({"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip(), "words": words})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out
