"""Pure decision helpers of the PILOT v2 protocol (no torch, no GPU)."""
import pytest

from tools.wmfixmerge import pilot_v2 as p


def v(a, b):
    return {120: a, 300: b}


def test_mini_update_streak_and_best():
    best, streak = p.mini_update([(5, v(10, 20)), (10, v(8, 18)), (15, v(9, 30)), (20, v(9, 40))], None)
    assert best == (10, 13.0) and streak == 2
    best, streak = p.mini_update([(15, v(9, 30)), (5, v(10, 20)), (10, v(8, 18))], None)
    assert best == (10, 13.0) and streak == 1  # chronological order enforced


def test_choose_step_earliest_tie():
    assert p.choose_step([(10, v(1, 3)), (5, v(2, 2)), (15, v(0, 4))]) == 5
    with pytest.raises(ValueError):
        p.choose_step([])


def test_assemble_fail_closed_and_mae():
    rows = [dict(group=g, horizon=h, arm=a, contrast=(-1) ** a * (g + 1))
            for g in range(2) for h in (120, 300) for a in (0, 1)]
    panel = p.assemble([rows[:4], rows[4:]], [0, 1])
    assert panel["120"]["paired_J_MAE"] == pytest.approx(1.5) and panel["300"]["rows"] == 4
    with pytest.raises(ValueError):
        p.assemble([rows[:4]], [0, 1])


def test_decide_v2_strict_both_horizons():
    assert p.decide_v2(v(10, 10), v(9, 9))["passed"]
    assert not p.decide_v2(v(10, 10), v(9, 10))["passed"]
    assert not p.decide_v2(v(10, 10), v(9, 9), frozen_bitwise=False)["passed"]
