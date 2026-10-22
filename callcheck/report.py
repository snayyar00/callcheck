"""Build a self-contained HTML report from analyzed calls, plus a turn-taking projection.

The projection is arithmetic on measured latencies. It is not a measurement of any vendor.
"""

import html
from pathlib import Path

import numpy as np

SLOW_S, DEAD_S = 1.5, 3.0  # same defaults as callcheck.analyze.Config
EVENT_TYPES = ["slow_response", "dead_air", "cutoff", "false_barge_in", "wrong_readback"]
COLORS = {"slow_response": "#d97706", "dead_air": "#dc2626", "cutoff": "#7c3aed",
          "false_barge_in": "#0891b2", "wrong_readback": "#db2777"}


# ---------- analysis ----------

def analyze_file(path, vad):
    """Analyze one WAV. Entity fields are filled only if callcheck.entities is present."""
    from .analyze import analyze, load
    caller, agent = load(path)
    res = analyze(caller, agent, vad=vad)
    events = list(res.events)
    time_to_info = None
    try:  # optional: needs callcheck.entities, callcheck.stt and faster-whisper
        from . import entities, stt
        ent = entities.extract(stt.transcribe(caller), stt.transcribe(agent))
        events += ent["events"]
        events.sort(key=lambda ev: ev["time"])
        time_to_info = ent["time_to_info_s"]
        entities_ran = True
    except ImportError:
        entities_ran = False
    return {"name": Path(path).name, "duration_s": res.duration_s, "segments": res.segments,
            "latencies": res.latencies, "events": events, "time_to_info_s": time_to_info, "entities": entities_ran}


# ---------- projection ----------

def project_call(call, target_s):
    """Cap every response latency at target_s and remove the gap. Returns a dict of before/after values."""
    lats = call["latencies"]
    saved = sum(max(0.0, l["latency_s"] - target_s) for l in lats)
    after_lats = [min(l["latency_s"], target_s) for l in lats]
    tti = call.get("time_to_info_s")
    tti_after = None
    if tti is not None:
        before_info = sum(max(0.0, l["latency_s"] - target_s) for l in lats if l["agent_start"] <= tti)
        tti_after = tti - before_info
    # A slow/dead event stays only if its capped latency still exceeds its threshold.
    gone = {"slow_response": 0, "dead_air": 0}
    for ev in call["events"]:
        thr = {"slow_response": SLOW_S, "dead_air": DEAD_S}.get(ev["type"])
        if thr is not None and min(ev["latency_s"], target_s) <= thr:
            gone[ev["type"]] += 1
    return {"saved_s": saved, "duration_after_s": call["duration_s"] - saved, "after_lats": after_lats,
            "time_to_info_after_s": tti_after, "gone": gone}


def summarize(calls, target_s):
    """Headline numbers for a call set, before and after the projection."""
    projs = [project_call(c, target_s) for c in calls]
    lats = [l["latency_s"] for c in calls for l in c["latencies"]]
    after = [x for p in projs for x in p["after_lats"]]
    count = lambda t: sum(1 for c in calls for e in c["events"] if e["type"] == t)
    n = max(len(calls), 1)
    turns = sum(len(c["segments"]["agent_turns"]) for c in calls)
    ttis = [c["time_to_info_s"] for c in calls if c.get("time_to_info_s") is not None]
    ttis_after = [p["time_to_info_after_s"] for p in projs if p["time_to_info_after_s"] is not None]
    mean = lambda x: float(np.mean(x)) if len(x) else None
    p95 = lambda x: float(np.percentile(x, 95)) if len(x) else None
    return {
        "calls": len(calls), "agent_turns": turns,
        "mean_latency": mean(lats), "p95_latency": p95(lats),
        "mean_latency_after": mean(after), "p95_latency_after": p95(after),
        "dead_air_s": sum(e["latency_s"] for c in calls for e in c["events"] if e["type"] == "dead_air"),
        "cutoffs_per_100_turns": 100 * count("cutoff") / turns if turns else None,
        "false_barge_per_call": count("false_barge_in") / n,
        "wrong_readbacks": count("wrong_readback") if any(c.get("entities") for c in calls) else None,
        "mean_time_to_info": mean(ttis), "mean_time_to_info_after": mean(ttis_after),
        "wait_per_call": sum(lats) / n, "wait_per_call_after": sum(after) / n,
        "duration_mean": mean([c["duration_s"] for c in calls]),
        "duration_mean_after": mean([p["duration_after_s"] for p in projs]),
        "saved_per_call": sum(p["saved_s"] for p in projs) / n,
        "slow": count("slow_response"), "dead": count("dead_air"),
        "slow_gone": sum(p["gone"]["slow_response"] for p in projs),
        "dead_gone": sum(p["gone"]["dead_air"] for p in projs),
        "info_calls": len(ttis),
    }


# ---------- html ----------

def f(x, nd=2, unit=""):
    return "n/a" if x is None else f"{x:.{nd}f}{unit}"


def timeline_svg(call, width=720):
    dur = max(call["duration_s"], 0.1)
    x = lambda t: 8 + (width - 16) * t / dur
    seg = call["segments"]
    parts = [f'<svg viewBox="0 0 {width} 64" role="img" aria-label="Timeline of {html.escape(call["name"])}" class="tl">']
    for y, key, cls in [(8, "caller", "lane-c"), (34, "agent", "lane-a")]:
        parts.append(f'<rect x="8" y="{y}" width="{width - 16}" height="18" class="lane"/>')
        for s, e in seg[key]:
            parts.append(f'<rect x="{x(s):.1f}" y="{y}" width="{max(x(e) - x(s), 0.8):.1f}" height="18" class="{cls}"/>')
    for ev in call["events"]:
        c = COLORS.get(ev["type"], "#888")
        parts.append(f'<path d="M{x(ev["time"]):.1f} 2 l-4 -0 l4 6 l4 -6 z" fill="{c}"><title>{ev["type"]} at {ev["time"]:.1f}s</title></path>')
        parts.append(f'<line x1="{x(ev["time"]):.1f}" x2="{x(ev["time"]):.1f}" y1="8" y2="52" stroke="{c}" stroke-width="1" opacity=".6"/>')
    parts.append(f'<text x="8" y="62" class="ax">0 s</text><text x="{width - 8}" y="62" text-anchor="end" class="ax">{dur:.0f} s</text>')
    parts.append('</svg>')
    return "".join(parts)


CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#e3e3e3;--card:#f7f7f8;--c:#2563eb;--a:#16a34a;--lane:#ececee}
@media (prefers-color-scheme:dark){:root{--bg:#121214;--fg:#eaeaea;--mut:#9a9a9a;--line:#2c2c30;--card:#1b1b1f;--c:#60a5fa;--a:#4ade80;--lane:#26262b}}
*{box-sizing:border-box}body{margin:0;padding:24px 16px;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:980px;margin:0 auto}h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 8px}
.mut{color:var(--mut)}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.card b{display:block;font-size:22px}.card span{color:var(--mut);font-size:13px}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}
.wrap{overflow-x:auto}.proj{border:2px dashed var(--line);border-radius:10px;padding:4px 16px 14px;margin-top:28px}
.tag{display:inline-block;font-size:11px;font-weight:600;letter-spacing:.05em;border:1px solid var(--mut);border-radius:4px;padding:1px 6px;color:var(--mut)}
.tl{width:100%;height:auto}.lane{fill:var(--lane)}.lane-c{fill:var(--c)}.lane-a{fill:var(--a)}.ax{fill:var(--mut);font-size:9px}
details{border-top:1px solid var(--line);padding:6px 0}summary{cursor:pointer}.key span{margin-right:14px;font-size:13px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px}
"""


def card(label, val):
    return f'<div class="card"><b>{val}</b><span>{label}</span></div>'


def render(calls, title, target_s):
    s = summarize(calls, target_s)
    have_ent = s["wrong_readbacks"] is not None
    cards = [
        card("calls", s["calls"]), card("mean agent response latency", f(s["mean_latency"], 2, " s")),
        card("p95 agent response latency", f(s["p95_latency"], 2, " s")),
        card("total dead air (latency of dead-air events)", f(s["dead_air_s"], 1, " s")),
        card("cut-offs per 100 agent turns", f(s["cutoffs_per_100_turns"], 1)),
        card("false barge-ins per call", f(s["false_barge_per_call"], 2)),
        card("mean caller wait per call (sum of latencies)", f(s["wait_per_call"], 1, " s")),
    ]
    if have_ent:
        cards.append(card("wrong readbacks", s["wrong_readbacks"]))
        cards.append(card(f"mean time to collect phone number ({s['info_calls']} calls)", f(s["mean_time_to_info"], 1, " s")))
    rows = []
    for c in calls:
        lat = [l["latency_s"] for l in c["latencies"]]
        cnt = {t: sum(1 for e in c["events"] if e["type"] == t) for t in EVENT_TYPES}
        rows.append("<tr><td>%s</td><td>%.0f s</td><td>%s</td><td>%s</td>%s<td>%s</td></tr>" % (
            html.escape(c["name"]), c["duration_s"], f(np.mean(lat) if lat else None), f(max(lat) if lat else None),
            "".join(f"<td>{cnt[t]}</td>" for t in EVENT_TYPES), f(c.get("time_to_info_s"), 1, " s")))
    head = "".join(f"<th>{t.replace('_', ' ')}</th>" for t in EVENT_TYPES)
    key = "".join(f'<span><i class="dot" style="background:{COLORS[t]}"></i>{t.replace("_", " ")}</span>' for t in EVENT_TYPES)
    lines = [
        ("Mean response latency", f(s["mean_latency"], 2, " s"), f(s["mean_latency_after"], 2, " s")),
        ("p95 response latency", f(s["p95_latency"], 2, " s"), f(s["p95_latency_after"], 2, " s")),
        ("Caller wait per call", f(s["wait_per_call"], 1, " s"), f(s["wait_per_call_after"], 1, " s")),
        ("Mean call duration", f(s["duration_mean"], 1, " s"), f(s["duration_mean_after"], 1, " s")),
    ]
    if have_ent:
        lines.append((f"Mean time to phone number ({s['info_calls']} calls)", f(s["mean_time_to_info"], 1, " s"),
                      f(s["mean_time_to_info_after"], 1, " s")))
    lines.append(("Slow-response events (over 1.5 s)", s["slow"], s["slow"] - s["slow_gone"]))
    lines.append(("Dead-air events (over 3 s)", s["dead"], s["dead"] - s["dead_gone"]))
    proj_rows = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in lines)
    tl = "".join(f"<details><summary>{html.escape(c['name'])} <span class='mut'>{c['duration_s']:.0f} s, {len(c['events'])} events</span></summary>{timeline_svg(c)}</details>" for c in calls)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style></head><body><main>
<h1>{html.escape(title)}</h1><p class="mut">Measured from two-channel recordings. {s['calls']} calls, {s['agent_turns']} agent turns.</p>
<h2>Headline numbers</h2><div class="grid">{''.join(cards)}</div>
<h2>Per call</h2><div class="wrap"><table><tr><th>call</th><th>duration</th><th>mean latency (s)</th><th>max latency (s)</th>{head}<th>time to info</th></tr>{''.join(rows)}</table></div>
<h2>Timelines</h2><p class="key mut">Top lane caller, bottom lane agent. {key}</p>{tl}
<section class="proj"><h2>What faster turn-taking would change <span class="tag">PROJECTION</span></h2>
<p>This is a projection from arithmetic on the measured calls above. It is not a measurement of any vendor or product. Each agent response latency above the target is replaced by the target, and the removed gap is cut out of the call.</p>
<p><b>Target response latency: {target_s * 1000:.0f} ms</b> (vendors such as Bland publish sub-400 ms).</p>
<div class="wrap"><table><tr><th>metric</th><th>measured</th><th>projected</th></tr>{proj_rows}</table></div>
<p>Mean time removed per call: {f(s['saved_per_call'], 1, ' s')}.</p>
<p class="mut">Cut-offs, false barge-ins and wrong readbacks are not projected. They need the agent itself fixed and re-tested.</p></section>
</main></body></html>"""


def write_report(calls, out, title, target_s=0.4):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(render(calls, title, target_s), encoding="utf-8")


# ---------- single-call and batch pages (used by callcheck.cli) ----------

MEANING = {
    "slow_response": "The agent took over 1.5 s to start speaking after the caller stopped. Callers often start to repeat themselves around this point.",
    "dead_air": "The agent took over 3 s to respond. Callers commonly think the line dropped and hang up or say hello again.",
    "cutoff": "The agent started speaking in a short pause and the caller then kept talking. The caller was talked over and may have lost part of what they meant to say.",
    "false_barge_in": "A short non-speech sound on the caller line (cough, bump, noise) made the agent stop talking. The caller never asked the agent to stop.",
    "wrong_readback": "The agent read a phone number back that differs from what the caller said. Needs a transcript check; speech to text can also be wrong.",
}

CSS2 = """
.sev-ok{fill:#16a34a;opacity:.12}.sev-slow{fill:#d97706;opacity:.35}.sev-dead{fill:#dc2626;opacity:.4}
.mean li{margin-bottom:6px}.mono{font-variant-numeric:tabular-nums}td.t,th.t{white-space:nowrap}
a{color:var(--c)}.note{font-size:13px}
"""


def event_detail(ev):
    t = ev["type"]
    if t in ("slow_response", "dead_air"):
        return "agent answered %.2f s after the caller stopped" % ev["latency_s"]
    if t == "cutoff":
        return "agent started at %s, caller resumed at %s" % (mmss(ev["time"]), mmss(ev["caller_resumed_at"]))
    if t == "false_barge_in":
        return "agent stopped at %s after a short sound on the caller line" % mmss(ev["agent_stopped_at"])
    if t == "wrong_readback":
        return "caller said %s, agent read back %s" % (ev.get("said"), ev.get("read_back"))
    return ""


def mmss(t):
    t = int(round(t))
    return "%02d:%02d" % (t // 60, t % 60)


def latency_svg(call, width=720):
    """Two-lane timeline with response gaps shaded by severity (green ok, amber over 1.5 s, red over 3 s)."""
    dur = max(call["duration_s"], 0.1)
    x = lambda t: 8 + (width - 16) * t / dur
    seg = call["segments"]
    p = [f'<svg viewBox="0 0 {width} 84" role="img" aria-label="Timeline of {html.escape(call["name"])}" class="tl">']
    for y, key, cls in [(18, "caller", "lane-c"), (48, "agent", "lane-a")]:
        p.append(f'<rect x="8" y="{y}" width="{width - 16}" height="20" class="lane"/>')
        for s, e in seg[key]:
            p.append(f'<rect x="{x(s):.1f}" y="{y}" width="{max(x(e) - x(s), 0.8):.1f}" height="20" class="{cls}"/>')
    p.append('<text x="8" y="13" class="ax">caller</text><text x="8" y="79" class="ax">agent</text>')
    for l in call["latencies"]:
        s, e = l["agent_start"] - l["latency_s"], l["agent_start"]
        cls = "sev-dead" if l["latency_s"] > DEAD_S else "sev-slow" if l["latency_s"] > SLOW_S else "sev-ok"
        p.append(f'<rect x="{x(s):.1f}" y="18" width="{max(x(e) - x(s), 1):.1f}" height="50" class="{cls}"><title>{l["latency_s"]:.2f} s response</title></rect>')
    for ev in call["events"]:
        c, xx = COLORS.get(ev["type"], "#888"), x(ev["time"])
        p.append(f'<path d="M{xx - 4:.1f} 2 h8 l-4 7 z" fill="{c}"><title>{ev["type"]} at {mmss(ev["time"])}</title></path>')
        p.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="9" y2="68" stroke="{c}" stroke-width="1" opacity=".7"/>')
    p.append(f'<text x="{width - 8}" y="79" text-anchor="end" class="ax">{mmss(dur)}</text></svg>')
    return "".join(p)


def stats(call):
    lat = sorted(l["latency_s"] for l in call["latencies"])
    return {"n": len(lat), "median": float(np.median(lat)) if lat else None,
            "p90": float(np.percentile(lat, 90)) if lat else None, "max": lat[-1] if lat else None}


def counts(call):
    return {t: sum(1 for e in call["events"] if e["type"] == t) for t in EVENT_TYPES}


def call_page(call, title=None, back=None):
    st, cn = stats(call), counts(call)
    title = title or f"callcheck: {call['name']}"
    cards = [card("duration", mmss(call["duration_s"])), card("agent responses", st["n"]),
             card("latency median", f(st["median"], 2, " s")), card("latency p90", f(st["p90"], 2, " s")),
             card("latency max", f(st["max"], 2, " s")), card("events", len(call["events"]))]
    if call.get("entities"):
        cards.append(card("time to phone number", f(call.get("time_to_info_s"), 1, " s")))
    key = "".join(f'<span><i class="dot" style="background:{COLORS[t]}"></i>{t.replace("_", " ")}</span>' for t in EVENT_TYPES)
    rows = "".join("<tr><td class='t'>%s</td><td class='t'>%s</td><td>%s</td></tr>" % (
        mmss(e["time"]), e["type"], html.escape(event_detail(e))) for e in call["events"])
    table = (f'<div class="wrap"><table><tr><th class="t">time</th><th class="t">type</th><th>detail</th></tr>{rows}</table></div>'
             if rows else "<p>No events found.</p>")
    present = [t for t in EVENT_TYPES if cn[t]]
    mean = "".join(f"<li><b>{t.replace('_', ' ')}</b> ({cn[t]}): {MEANING[t]}</li>" for t in present) or "<li>No events, so nothing to explain.</li>"
    ent = "" if call.get("entities") else '<p class="mut note">Speech to text was not run, so wrong read-backs and time to phone number are not checked.</p>'
    nav = f'<p><a href="{html.escape(back)}">All calls</a></p>' if back else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}{CSS2}</style></head><body><main>{nav}
<h1>{html.escape(call['name'])}</h1><p class="mut">Measured from a two-channel recording. Latency is the time from the caller finishing to the agent starting.</p>
<div class="grid">{''.join(cards)}</div>
<h2>Timeline</h2><p class="key mut">Top lane caller, bottom lane agent. Shaded bands are response gaps: green up to 1.5 s, amber over 1.5 s, red over 3 s. {key}</p>
{latency_svg(call)}
<h2>Events</h2>{table}
<h2>What this means</h2><ul class="mean">{mean}</ul>{ent}
</main></body></html>"""


def batch_page(calls, links, title="callcheck batch"):
    allat = [l["latency_s"] for c in calls for l in c["latencies"]]
    evs = [len(c["events"]) for c in calls]
    n = max(len(calls), 1)
    cards = [card("calls", len(calls)), card("agent responses", len(allat)),
             card("latency median (all calls)", f(float(np.median(allat)) if allat else None, 2, " s")),
             card("latency p95 (all calls)", f(float(np.percentile(allat, 95)) if allat else None, 2, " s")),
             card("events per call (mean)", f(sum(evs) / n, 2))]
    head = "".join(f"<th>{t.replace('_', ' ')}</th>" for t in EVENT_TYPES)
    rows = []
    for c, link in zip(calls, links):
        st, cn = stats(c), counts(c)
        rows.append("<tr><td><a href='%s'>%s</a></td><td>%s</td><td>%d</td><td>%s</td><td>%s</td><td>%s</td>%s</tr>" % (
            html.escape(link), html.escape(c["name"]), mmss(c["duration_s"]), st["n"], f(st["median"]), f(st["p90"]), f(st["max"]),
            "".join(f"<td>{cn[t]}</td>" for t in EVENT_TYPES)))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}{CSS2}</style></head><body><main>
<h1>{html.escape(title)}</h1><p class="mut">Measured from two-channel recordings. Latency is in seconds.</p>
<div class="grid">{''.join(cards)}</div>
<h2>Calls</h2><div class="wrap"><table><tr><th>call</th><th>duration</th><th>responses</th><th>median</th><th>p90</th><th>max</th>{head}</tr>{''.join(rows)}</table></div>
</main></body></html>"""
