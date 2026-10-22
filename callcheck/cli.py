"""Command line: `callcheck call.wav` or `callcheck batch <dir>`."""

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

from . import vad as vadmod
from .analyze import analyze, load

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".mp4"}
CACHE = Path.home() / ".cache" / "callcheck"
TYPES = ["slow_response", "dead_air", "cutoff", "false_barge_in", "wrong_readback"]


def model_path():
    """Repo-local models/ first, then the user cache. Download on first use."""
    if vadmod.MODEL.exists():
        return vadmod.MODEL
    cached = CACHE / "silero_vad.onnx"
    if not cached.exists():
        print(f"Silero VAD model not found. Downloading to {cached} (one time, about 2 MB)...", file=sys.stderr)
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_suffix(".part")
        try:
            urllib.request.urlretrieve(vadmod.MODEL_URL, tmp)
        except Exception as e:  # network or HTTP failure
            tmp.unlink(missing_ok=True)
            sys.exit(f"callcheck: could not download the VAD model from {vadmod.MODEL_URL}: {e}\n"
                     f"Download it by hand and save it as {cached}")
        tmp.rename(cached)
    return cached


def channels(path):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=channels",
                              "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout.strip()
        return int(out)
    except Exception:
        return None


def stt_available():
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def run_call(path, vad, caller_channel=0, use_stt=False):
    if channels(path) == 1:
        raise ValueError("mono file: callcheck needs the caller and the agent on separate channels (stereo)")
    caller, agent = load(path, caller_channel)
    res = analyze(caller, agent, vad=vad)
    events = list(res.events)
    tti = None
    if use_stt:
        from . import entities, stt
        ent = entities.extract(stt.transcribe(caller), stt.transcribe(agent),
                               caller_speech=res.segments["caller_turns"], agent_speech=res.segments["agent_turns"])
        events = sorted(events + ent["events"], key=lambda e: e["time"])
        tti = ent["time_to_info_s"]
    return {"name": Path(path).name, "path": str(path), "duration_s": res.duration_s, "caller_channel": caller_channel,
            "segments": res.segments, "latencies": res.latencies, "events": events,
            "time_to_info_s": tti, "entities": use_stt}


def lat_stats(lats):
    x = [l["latency_s"] for l in lats]
    if not x:
        return {"median": None, "p90": None, "max": None}
    return {"median": float(np.median(x)), "p90": float(np.percentile(x, 90)), "max": float(max(x))}


def counts(call):
    return {t: sum(1 for e in call["events"] if e["type"] == t) for t in TYPES}


def mmss(t):
    t = int(round(t))
    return "%02d:%02d" % (t // 60, t % 60)


def fs(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f} s"


def detail(ev):
    from .report import event_detail
    return event_detail(ev)


def print_call(call):
    st, cn = lat_stats(call["latencies"]), counts(call)
    print(f"File:             {call['name']}")
    print(f"Duration:         {mmss(call['duration_s'])} ({call['duration_s']:.1f} s)")
    print(f"Agent responses:  {len(call['latencies'])}")
    print(f"Latency median:   {fs(st['median'])}")
    print(f"Latency p90:      {fs(st['p90'])}")
    print(f"Latency max:      {fs(st['max'])}")
    if call["entities"]:
        print(f"Time to info:     {fs(call['time_to_info_s'], 1)}")
    print("Events:")
    for t in TYPES:
        if t == "wrong_readback" and not call["entities"]:
            print(f"  {t:<15} not checked (use --stt)")
        else:
            print(f"  {t:<15} {cn[t]}")
    if call["events"]:
        print("\nTimeline of events:")
        print(f"  {'time':<6} {'type':<15} detail")
        for e in call["events"]:
            print(f"  {mmss(e['time']):<6} {e['type']:<15} {detail(e)}")
    else:
        print("\nNo events found.")


def call_json(call):
    st = lat_stats(call["latencies"])
    seg = call["segments"]
    return {"name": call["name"], "duration_s": round(call["duration_s"], 3), "caller_channel": call["caller_channel"],
            "agent_responses": len(call["latencies"]), "latency_stats": st, "latencies": call["latencies"],
            "event_counts": counts(call), "events": call["events"], "stt": call["entities"],
            "time_to_info_s": call["time_to_info_s"],
            "caller_turns": [[round(a, 3), round(b, 3)] for a, b in seg["caller_turns"]],
            "agent_turns": [[round(a, 3), round(b, 3)] for a, b in seg["agent_turns"]]}


def write(path, text):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    print(f"wrote {p}", file=sys.stderr)


def stt_check(args):
    if args.stt and not stt_available():
        print("callcheck: --stt skipped, faster-whisper is not installed. Run: pip install 'callcheck[stt]'", file=sys.stderr)
        return False
    return args.stt


def cmd_call(args):
    from .report import call_page
    path = Path(args.audio)
    if not path.is_file():
        print(f"callcheck: file not found: {path}", file=sys.stderr)
        return 2
    use_stt = stt_check(args)
    vad = vadmod.SileroVAD(model_path())
    try:
        call = run_call(path, vad, args.caller_channel, use_stt)
    except (ValueError, subprocess.CalledProcessError) as e:
        print(f"callcheck: {path.name}: {e}", file=sys.stderr)
        return 2
    print_call(call)
    if args.json:
        write(args.json, json.dumps(call_json(call), indent=2))
    if args.html:
        write(args.html, call_page(call))
    return 0


def cmd_batch(args):
    from .report import batch_page, call_page
    root = Path(args.dir)
    files = sorted(p for p in root.iterdir() if p.suffix.lower() in AUDIO_EXT) if root.is_dir() else []
    if not files:
        print(f"callcheck: no audio files in {root}", file=sys.stderr)
        return 2
    use_stt = stt_check(args)
    vad = vadmod.SileroVAD(model_path())
    calls, failed = [], []
    for i, p in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {p.name}", file=sys.stderr)
        try:
            calls.append(run_call(p, vad, args.caller_channel, use_stt))
        except (ValueError, subprocess.CalledProcessError) as e:
            failed.append(p.name)
            print(f"  skipped: {e}", file=sys.stderr)
    if not calls:
        return 2
    print(f"{'call':<18} {'dur':>6} {'resp':>5} {'median':>8} {'p90':>8} {'max':>8} {'events':>7}  by type")
    for c in calls:
        st, cn = lat_stats(c["latencies"]), counts(c)
        by = " ".join(f"{t}={n}" for t, n in cn.items() if n) or "-"
        print(f"{c['name']:<18} {mmss(c['duration_s']):>6} {len(c['latencies']):>5} {fs(st['median']):>8} {fs(st['p90']):>8} "
              f"{fs(st['max']):>8} {len(c['events']):>7}  {by}")
    allat = [l for c in calls for l in c["latencies"]]
    st = lat_stats(allat)
    p95 = float(np.percentile([l["latency_s"] for l in allat], 95)) if allat else None
    print(f"\nAggregate over {len(calls)} calls, {len(allat)} agent responses")
    print(f"  Latency median:     {fs(st['median'])}")
    print(f"  Latency p95:        {fs(p95)}")
    print(f"  Events per call:    {sum(len(c['events']) for c in calls) / len(calls):.2f}")
    for t in TYPES:
        if t == "wrong_readback" and not use_stt:
            continue
        n = sum(counts(c)[t] for c in calls)
        print(f"  {t:<19} {n} ({n / len(calls):.2f} per call)")
    if failed:
        print(f"  skipped files:      {', '.join(failed)}")
    if args.json:
        write(args.json, json.dumps([call_json(c) for c in calls], indent=2))
    if args.html:
        out = Path(args.html)
        sub = out.with_name(out.stem + "_calls")
        links = []
        for c in calls:
            page = sub / (Path(c["name"]).stem + ".html")
            write(page, call_page(c, back=f"../{out.name}"))
            links.append(f"{sub.name}/{page.name}")
        write(out, batch_page(calls, links))
    return 0


def build_parser():
    ap = argparse.ArgumentParser(prog="callcheck", description="Find turn-taking failures in a two-channel call recording. "
                                 "Use `callcheck batch <dir>` for a folder.")
    ap.add_argument("audio", help="two-channel recording (wav, mp3, m4a, ...)")
    add_common(ap)
    return ap


def add_common(ap):
    ap.add_argument("--caller-channel", type=int, choices=[0, 1], default=0, help="channel that carries the caller (default 0)")
    ap.add_argument("--html", help="write a self-contained HTML report")
    ap.add_argument("--json", help="write results as JSON")
    ap.add_argument("--stt", action="store_true", help="also run speech to text (needs callcheck[stt]) to check phone-number read-backs")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "batch":
        ap = argparse.ArgumentParser(prog="callcheck batch", description="Run every audio file in a folder.")
        ap.add_argument("dir")
        add_common(ap)
        return cmd_batch(ap.parse_args(argv[1:]))
    return cmd_call(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
