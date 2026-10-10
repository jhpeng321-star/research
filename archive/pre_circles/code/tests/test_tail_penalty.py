"""Tests for the one-sided p90 local-delay penalty (_tail_penalty)."""

from unittest.mock import patch

from env.rewards import _tail_penalty


class _TL:
    """TimeLossTracker stand-in: fixed rates per vehicle."""

    def __init__(self, rates):
        self._rates = rates

    def get(self, veh_id):
        return self._rates[veh_id]


def _run(local_ids, rates, tail_r_ref=0.34, tracker=True):
    tl = _TL(rates) if tracker else None
    with patch("env.rewards._local_vehicles", return_value=local_ids):
        return _tail_penalty(
            "cav1", 200.0, cav_ids={"cav1"}, time_loss_tracker=tl, tail_r_ref=tail_r_ref
        )


def test_tail_zero_when_p90_at_baseline():
    # 10 HDVs, p90 rate = 0.3 < ref 0.34 -> no penalty
    ids = [f"h{i}" for i in range(10)]
    rates = {v: (0.3 if i >= 8 else 0.05) for i, v in enumerate(ids)}
    assert _run(ids, rates) == 0.0


def test_tail_proportional_above_baseline():
    # p90 = 0.51 -> -(0.51-0.34)/0.34 = -0.5
    ids = [f"h{i}" for i in range(10)]
    rates = {v: (0.51 if i >= 8 else 0.05) for i, v in enumerate(ids)}
    r = _run(ids, rates)
    assert abs(r - (-0.5)) < 1e-6


def test_tail_clips_at_minus_one():
    # p90 = 0.8 -> -(0.8-0.34)/0.34 = -1.35 -> clip -1
    ids = [f"h{i}" for i in range(10)]
    rates = {v: (0.8 if i >= 8 else 0.05) for i, v in enumerate(ids)}
    assert _run(ids, rates) == -1.0


def test_tail_never_positive():
    # p90 = 0.0 -> (0-0.34)/0.34 = -1 -> one-sided clamp to 0
    ids = [f"h{i}" for i in range(10)]
    rates = {v: 0.0 for v in ids}
    assert _run(ids, rates) == 0.0


def test_tail_zero_with_few_local_hdvs():
    ids = ["h0", "h1"]
    rates = {"h0": 0.9, "h1": 0.9}
    assert _run(ids, rates) == 0.0


def test_tail_zero_without_tracker():
    assert _run(["h0", "h1", "h2"], {}, tracker=False) == 0.0


def test_tail_excludes_cavs():
    ids = ["cav9", "h1", "h2", "h3"]
    rates = {"cav9": 0.9, "h1": 0.1, "h2": 0.1, "h3": 0.1}
    # cav9 is in the active CAV set -> excluded from the HDV delay statistic;
    # p90 over remaining HDVs = 0.1 -> no penalty
    tl = _TL(rates)
    with patch("env.rewards._local_vehicles", return_value=ids):
        r = _tail_penalty(
            "cav1", 200.0, cav_ids={"cav1", "cav9"}, time_loss_tracker=tl, tail_r_ref=0.34
        )
    assert r == 0.0
