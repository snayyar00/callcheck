import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from callcheck import cli  # noqa: E402

SR = 16000


def tone(n, f=180):
    t = np.arange(n) / SR
    # harmonic-rich and amplitude-modulated so it looks voiced. This file tests the CLI plumbing only;
    # the detection rules are tested on exact segments in test_detectors.py.
    sig = sum(np.sin(2 * np.pi * f * k * t) / k for k in range(1, 8)) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))
    return (0.3 * sig / 2).astype(np.float32)


def make_call(path):
    """Stereo call: caller speaks, agent answers 2.2 s later (a slow response), twice."""
    n = 14 * SR
    caller, agent = np.zeros(n, np.float32), np.zeros(n, np.float32)
    for c0, a0 in [(1.0, 5.2), (8.0, 11.0)]:
        caller[int(c0 * SR):int((c0 + 3) * SR)] = tone(3 * SR)
        agent[int(a0 * SR):int((a0 + 2) * SR)] = tone(2 * SR, 130)
    pcm = (np.stack([caller, agent], axis=1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2), w.setsampwidth(2), w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def test_cli_json(tmp_path):
    wav = tmp_path / "call.wav"
    make_call(wav)
    out = tmp_path / "out.json"
    html = tmp_path / "out.html"
    code = cli.main([str(wav), "--json", str(out), "--html", str(html)])
    assert code == 0
    data = json.loads(out.read_text())
    assert "latencies" in data and "events" in data
    assert len(data["latencies"]) >= 1, data  # the synthetic tone must be heard by the VAD
    assert data["event_counts"].keys() >= {"slow_response", "dead_air", "cutoff", "false_barge_in"}
    assert html.read_text().startswith("<!doctype html>")


def test_cli_missing_file(tmp_path):
    assert cli.main([str(tmp_path / "nope.wav")]) == 2


def test_event_detail_and_mmss():
    from callcheck.report import event_detail, mmss
    assert mmss(75.4) == "01:15"
    assert "2.50" in event_detail({"type": "slow_response", "time": 1, "latency_s": 2.5})
