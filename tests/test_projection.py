import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from callcheck.report import project_call  # noqa: E402


def make(lats, tti=None, events=()):
    return {"duration_s": 60.0, "latencies": [{"agent_start": 10.0 * (i + 1), "latency_s": x} for i, x in enumerate(lats)],
            "events": list(events), "time_to_info_s": tti, "segments": {"agent_turns": []}}


def test_seconds_saved():
    p = project_call(make([0.3, 1.0, 2.4]), 0.4)
    assert abs(p["saved_s"] - (0.6 + 2.0)) < 1e-9  # 0.3 is already under target
    assert abs(p["duration_after_s"] - 57.4) < 1e-9
    assert p["after_lats"] == [0.3, 0.4, 0.4]


def test_time_to_info_shift_only_counts_before_info():
    # latencies at t=10 (1.4 s over), t=20 (2.0 s over), t=30 (0.6 s over); info at t=25
    p = project_call(make([0.8, 2.4, 1.0], tti=25.0), 0.4)
    assert abs(p["time_to_info_after_s"] - (25.0 - 0.4 - 2.0)) < 1e-9


def test_events_removed():
    evs = [{"type": "slow_response", "time": 1, "latency_s": 2.0}, {"type": "dead_air", "time": 2, "latency_s": 4.0},
           {"type": "cutoff", "time": 3}]
    assert project_call(make([2.0, 4.0], events=evs), 0.4)["gone"] == {"slow_response": 1, "dead_air": 1}
    # a 2.0 s target still leaves the slow event (2.0 > 1.5) but clears dead air
    assert project_call(make([2.0, 4.0], events=evs), 2.0)["gone"] == {"slow_response": 0, "dead_air": 1}
