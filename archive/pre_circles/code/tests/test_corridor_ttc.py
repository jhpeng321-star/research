"""Tests for the corridor-level TTC penalty (reward term + core counting).

The TTC redistribution is corridor-systemic (wake probe 2026-08-04): the
reward needs a corridor-level count, computed from the traffic snapshot,
feeding ``corridor_ttc_penalty`` in reward_weights.
"""

from unittest.mock import MagicMock, patch

from env.core import TrafficShepherdMultiAgentEnv
from env.rewards import _corridor_ttc_penalty, compute_shepherd_reward


def test_penalty_helper():
    assert _corridor_ttc_penalty(None) == 0.0
    assert _corridor_ttc_penalty(0) == 0.0
    assert _corridor_ttc_penalty(1) == -1.0
    assert _corridor_ttc_penalty(3) == -3.0


def _reward_kwargs(**over):
    w = {
        "efficiency": 0.0,
        "safety": 0.0,
        "stability": 0.0,
        "fairness": 0.0,
        "delay_budget": 0.0,
        "delay_win": 0.0,
        "fair_win": 0.0,
        "easy_intervention_tax": 0.0,
        "control_cost": 0.0,
        "mask_penalty": 0.0,
        "ineligible_penalty": 0.0,
        "mode_use_bonus": 0.0,
        "flow": 0.0,
        "tail": 0.0,
        "corridor_ttc_penalty": 0.0,
    }
    w.update(over)
    return w


def test_reward_applies_weight():
    base = compute_shepherd_reward("cav0", 0, 35.0, _reward_kwargs(), 200.0)
    penalized = compute_shepherd_reward(
        "cav0",
        0,
        35.0,
        _reward_kwargs(corridor_ttc_penalty=0.2),
        200.0,
        corridor_ttc_count=2,
    )
    assert abs((penalized - base) - (-0.4)) < 1e-6
    # weight 0 (default/off) ignores the count entirely
    off = compute_shepherd_reward(
        "cav0",
        0,
        35.0,
        _reward_kwargs(),
        200.0,
        corridor_ttc_count=5,
    )
    assert off == base


class _StubSnap:
    """Minimal TrafficSnapshot stand-in for _corridor_ttc_count."""

    def __init__(self, rows):
        # rows: list of (vid, road, lane, pos, speed)
        self._rows = rows

    def ordered_ids(self):
        return [r[0] for r in self._rows]

    def road(self, vid):
        return next(r[1] for r in self._rows if r[0] == vid)

    def lane_id(self, vid):
        return next(r[2] for r in self._rows if r[0] == vid)

    def lane_pos(self, vid):
        return next(r[3] for r in self._rows if r[0] == vid)

    def speed(self, vid):
        return next(r[4] for r in self._rows if r[0] == vid)


def _env_stub(threshold=1.5):
    env = TrafficShepherdMultiAgentEnv.__new__(TrafficShepherdMultiAgentEnv)
    env._pre_edge = "pre"
    env._merge_edge = "merge"
    env._ramp_edge = "ramp"
    env._shepherd_engine = MagicMock(down_edge="down")
    env.config = {"corridor_ttc_threshold": threshold}
    return env


def test_count_detects_close_fast_pair():
    env = _env_stub()
    # follower 10 m behind leader, 5 m/s closing -> ttc = (10-5)/5 = 1.0 < 1.5
    snap = _StubSnap(
        [
            ("fol", "pre", "pre_0", 90.0, 25.0),
            ("lead", "pre", "pre_0", 100.0, 20.0),
        ]
    )
    with patch("env.core.get_active_snapshot", return_value=snap):
        assert env._corridor_ttc_count() == 1


def test_count_ignores_non_closing_and_far():
    env = _env_stub()
    snap = _StubSnap(
        [
            ("fol", "pre", "pre_0", 90.0, 20.0),  # slower than leader: no closing
            ("lead", "pre", "pre_0", 100.0, 25.0),
            ("far", "pre", "pre_0", 0.0, 30.0),  # far ahead on same lane
        ]
    )
    with patch("env.core.get_active_snapshot", return_value=snap):
        assert env._corridor_ttc_count() == 0


def test_count_scopes_to_corridor_edges():
    env = _env_stub()
    snap = _StubSnap(
        [
            ("fol", "other_edge", "other_0", 90.0, 25.0),
            ("lead", "other_edge", "other_0", 100.0, 20.0),
        ]
    )
    with patch("env.core.get_active_snapshot", return_value=snap):
        assert env._corridor_ttc_count() == 0


def test_count_none_without_snapshot():
    env = _env_stub()
    with patch("env.core.get_active_snapshot", return_value=None):
        assert env._corridor_ttc_count() is None


def test_threshold_configurable():
    env = _env_stub(threshold=0.5)
    snap = _StubSnap(
        [
            ("fol", "pre", "pre_0", 90.0, 25.0),
            ("lead", "pre", "pre_0", 100.0, 20.0),  # ttc = 1.0 > 0.5 threshold
        ]
    )
    with patch("env.core.get_active_snapshot", return_value=snap):
        assert env._corridor_ttc_count() == 0
