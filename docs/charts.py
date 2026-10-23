"""Render the README charts in light and dark. Cut-off numbers come from results/real_phone_holdout_first_version.txt, results/real_phone_holdout.txt, results/ami_first_version.txt and results/ami.txt (non-backchannel precision/recall); endpointing numbers from research/endpointing/RESULTS.md."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent
THEMES = {
    "light": dict(bg="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e1e0d9", s=["#2a78d6", "#eb6834", "#1baf7a"]),
    "dark": dict(bg="#1a1a19", ink="#f0efec", ink2="#c3c2b7", grid="#2c2c2a", s=["#3987e5", "#d95926", "#199e70"]),
}


def style(ax, t):
    ax.set_facecolor(t["bg"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["ink2"], length=0)
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def cutoff(t, name):
    # realdata/telephone/out_score_holdout.txt (now) vs the e8f3587 analyzer on the same 40 calls (before);
    # AMI from data/ami_pairs/score_all.txt (before) and score_all_v2.txt (now).
    groups = ["Phone calls\nprecision", "Phone calls\nrecall", "Meetings\nprecision", "Meetings\nrecall"]
    before, now = [13, 13, 29, 28], [57, 39, 43, 27]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200, facecolor=t["bg"])
    style(ax, t)
    w, x = 0.36, range(len(groups))
    for vals, off, color, label in ((before, -w / 2 - 0.01, t["s"][1], "First version"), (now, w / 2 + 0.01, t["s"][0], "Now")):
        bars = ax.bar([i + off for i in x], vals, w, color=color, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v}%", ha="center", va="bottom", fontsize=8, color=t["ink"])
    ax.set_xticks(list(x), groups, fontsize=8.5, color=t["ink2"])
    ax.set_ylim(0, 70); ax.set_yticks([0, 20, 40, 60], ["0%", "20%", "40%", "60%"], fontsize=8)
    ax.set_title("Cut-off detection on real conversations it was not tuned on", loc="left", fontsize=10.5, color=t["ink"], pad=12)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=t["ink2"], loc="upper right", ncols=2)
    fig.tight_layout(); fig.savefig(OUT / f"cutoff-{name}.png", facecolor=t["bg"]); plt.close(fig)


def endpointing(t, name):
    # research/endpointing/RESULTS.md, "minimum cut-off % at mean added wait" table (fallback 1.2 s).
    waits = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    series = [("Silence timeout", [80.4, 62.3, 49.4, 40.4, 30.8, 26.1]),
              ("Prosody model (trained)", [71.2, 56.6, 47.8, 40.8, 31.5, 26.3]),
              ("smart-turn v3.2 (zero-shot)", [65.7, 55.2, 46.7, 39.9, 34.7, 34.7])]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200, facecolor=t["bg"])
    style(ax, t)
    for (label, ys), color in zip(series, t["s"]):
        ax.plot(waits, ys, color=color, linewidth=2, marker="o", markersize=4, label=label)
    ax.set_xticks(waits, [f"{w:.1f} s" for w in waits], fontsize=8)
    ax.set_yticks([0, 20, 40, 60, 80], ["0%", "20%", "40%", "60%", "80%"], fontsize=8)
    ax.set_ylim(0, 88)
    ax.set_xlabel("How long the agent waits after the caller pauses", fontsize=8.5, color=t["ink2"])
    ax.set_title("How often the agent would cut in on a caller who wasn't done", loc="left", fontsize=10.5, color=t["ink"], pad=12)
    ax.legend(frameon=False, fontsize=8, labelcolor=t["ink2"], loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / f"endpointing-{name}.png", facecolor=t["bg"]); plt.close(fig)


for name, t in THEMES.items():
    cutoff(t, name); endpointing(t, name)
