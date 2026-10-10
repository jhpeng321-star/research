"""Unit tests for tools/select_checkpoint.py scoring/gating logic (pure, no SUMO)."""

import math

from tools.select_checkpoint import score_candidate


def _metrics(tt=60.0, ttc_events=0, collisions=0, time_loss=10.0, ml=62.0):
    return {
        "travel_time": {
            "all": {"mean": tt},
            "mainline": {"mean": ml},
            "ramp": {"mean": tt - 5},
        },
        "ttc": {"total_events": ttc_events},
        "collisions": collisions,
        "mean_time_loss": time_loss,
    }


def _hdv_refs(collisions=0, time_loss=10.0, tt=65.0, ml=67.0, ttc=100, n=3):
    # Vary TT slightly so median is well-defined; keep all "easy" relative to a
    # hard outlier when tests need it.
    out = []
    for i in range(n):
        out.append(
            {
                "collisions": collisions,
                "mean_time_loss": time_loss,
                "travel_time": {
                    "all": {"mean": tt + 0.1 * i},
                    "mainline": {"mean": ml + 0.1 * i},
                    "ramp": {"mean": 55.0},
                },
                "ttc": {"total_events": ttc},
            }
        )
    return out


def test_all_gates_pass_score_is_mean_travel_time():
    per_seed = [_metrics(tt=60.0), _metrics(tt=62.0), _metrics(tt=64.0)]
    score, report = score_candidate(per_seed, _hdv_refs())
    assert score == 62.0
    assert report["all_pass"]
    assert all(report[g]["pass"] for g in ("ttc", "collisions", "time_loss", "easy_mainline"))


def test_ttc_above_hdv_fails_gate():
    refs = _hdv_refs(ttc=10)
    per_seed = [_metrics(ttc_events=5), _metrics(ttc_events=11), _metrics(ttc_events=5)]
    score, report = score_candidate(per_seed, refs)
    assert score == math.inf
    assert not report["ttc"]["pass"]
    assert not report["all_pass"]


def test_ttc_equal_hdv_passes():
    refs = _hdv_refs(ttc=10)
    per_seed = [_metrics(ttc_events=10) for _ in range(3)]
    score, report = score_candidate(per_seed, refs)
    assert math.isfinite(score)
    assert report["ttc"]["pass"]


def test_collisions_sum_above_hdv_fails():
    refs = _hdv_refs(collisions=1)  # sum = 3
    per_seed = [
        _metrics(collisions=2),
        _metrics(collisions=1),
        _metrics(collisions=1),
    ]  # sum = 4
    score, report = score_candidate(per_seed, refs)
    assert score == math.inf
    assert not report["collisions"]["pass"]
    assert report["collisions"]["pol_sum"] == 4
    assert report["collisions"]["hdv_sum"] == 3


def test_collisions_sum_equal_hdv_passes():
    refs = _hdv_refs(collisions=1)
    per_seed = [_metrics(collisions=1) for _ in range(3)]
    score, report = score_candidate(per_seed, refs)
    assert math.isfinite(score)
    assert report["collisions"]["pass"]


def test_time_loss_above_105_percent_of_that_seed_fails():
    refs = _hdv_refs(time_loss=10.0)  # threshold 10.5 per seed
    per_seed = [_metrics(), _metrics(time_loss=10.6), _metrics()]
    score, report = score_candidate(per_seed, refs)
    assert score == math.inf
    assert not report["time_loss"]["pass"]


def test_easy_mainline_over_intervention_fails():
    # All HDV TTs similar → all are "easy" relative to median; +3s mainline fails.
    refs = _hdv_refs(tt=65.0, ml=67.0)
    per_seed = [_metrics(tt=66.0, ml=70.0) for _ in range(3)]  # dML = +3
    score, report = score_candidate(per_seed, refs, easy_ml_delta_max=2.0)
    assert score == math.inf
    assert not report["easy_mainline"]["pass"]


def test_easy_mainline_within_delta_passes():
    refs = _hdv_refs(tt=65.0, ml=67.0)
    per_seed = [_metrics(tt=65.5, ml=68.5) for _ in range(3)]  # dML = +1.5
    score, report = score_candidate(per_seed, refs, easy_ml_delta_max=2.0)
    assert math.isfinite(score)
    assert report["easy_mainline"]["pass"]
