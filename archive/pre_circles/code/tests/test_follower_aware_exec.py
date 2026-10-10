"""Tests for follower-aware execution (user steer 2026-08-05).

The mode DECISION stays discrete; a braking mode's magnitude adapts to the
follower behind: full strength when the road behind is open, gentle cap
when a close follower would inherit the brake as a compression wave.
"""

from tests.test_shepherd_actions import _make_engine


def _engine(**overrides):
    mc = {
        "follower_aware_exec": True,
        "follower_gentle_cap": 0.97,
        "follower_gap_near": 15.0,
        "follower_gap_far": 60.0,
    }
    mc.update(overrides)
    return _make_engine(mode_config=mc)


def test_off_passthrough():
    engine = _engine(follower_aware_exec=False)
    engine._follower_gap = lambda cav: 5.0
    assert engine._follower_aware_factor("cav0", 0.90) == 0.90


def test_no_follower_full_strength():
    engine = _engine()
    engine._follower_gap = lambda cav: None
    assert engine._follower_aware_factor("cav0", 0.90) == 0.90


def test_close_follower_gentle_cap():
    engine = _engine()
    engine._follower_gap = lambda cav: 10.0  # <= near 15
    assert engine._follower_aware_factor("cav0", 0.90) == 0.97
    engine._follower_gap = lambda cav: 15.0  # at boundary -> cap
    assert engine._follower_aware_factor("cav0", 0.90) == 0.97


def test_open_road_full_strength():
    engine = _engine()
    engine._follower_gap = lambda cav: 100.0  # >= far 60
    assert engine._follower_aware_factor("cav0", 0.90) == 0.90
    engine._follower_gap = lambda cav: 60.0  # at boundary -> base
    assert engine._follower_aware_factor("cav0", 0.90) == 0.90


def test_interpolation_midpoint():
    engine = _engine()
    engine._follower_gap = lambda cav: 37.5  # midpoint of [15, 60]
    # cap 0.97 + 0.5 * (0.90 - 0.97) = 0.935
    assert abs(engine._follower_aware_factor("cav0", 0.90) - 0.935) < 1e-9


def test_composes_with_adaptive_gentle():
    # Follower-aware scales whatever base factor adaptive gentleness returns.
    engine = _engine()
    engine._follower_gap = lambda cav: 10.0
    assert engine._follower_aware_factor("cav0", 0.85) == 0.97
