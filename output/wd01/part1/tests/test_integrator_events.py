import numpy as np
import pytest

import wd1lib as L


def test_raw_propagate_clamps_negative_speed():
    d = np.array([100.0, 50.0])
    v = np.array([10.0, 20.0])
    dv = np.array([-15.0, 1.0])          # v+dv = -5 and 21
    d2, v2, v_raw = L.raw_propagate(d, v, dv)
    assert v_raw[0] == -5.0
    assert v2[0] == 0.0                  # clamped
    assert v2[1] == 21.0
    np.testing.assert_allclose(d2, d - (v + v2) / 2.0)


def test_raw_propagate_unclamped_variant():
    d = np.array([100.0])
    v = np.array([10.0])
    d2, v2, v_raw = L.raw_propagate(d, v, np.array([-15.0]), clamp=False)
    assert v2[0] == -5.0                 # negative passes through
    np.testing.assert_allclose(d2, [100.0 - (10.0 + -5.0) / 2.0])


def test_constrained_accel_clip():
    d = np.array([100.0])
    v = np.array([20.0])
    for dv, expect in ((-50.0, -9.0), (50.0, 3.0), (-2.0, -2.0)):
        d2, v2, ov = L.constrained_propagate(
            d, v, np.array([0]), np.array([dv]), [40.0, 40.0, 40.0, 40.0])
        assert v2[0] == pytest.approx(20.0 + expect)
        assert not ov[0]


def test_constrained_speed_clip_zero_and_cap():
    d = np.array([100.0, 90.0])
    v = np.array([1.0, 39.0])
    g = np.array([0, 1])
    v_cap = [35.0, 40.0, 40.0, 40.0]
    d2, v2, ov = L.constrained_propagate(d, v, g, np.array([-5.0, +5.0]),
                                         v_cap)
    assert v2[0] == 0.0                  # clipped at 0
    assert v2[1] == 40.0                 # clipped at group cap (39+5=44 > 40)
    assert not ov.any()


def test_constrained_overlap_flagged_and_placed_behind():
    # follower (d=100) brakes hard; leader (d=95) at higher speed pulls away,
    # but make the follower FAST so it passes the leader in one step.
    d = np.array([100.0, 95.0])
    v = np.array([30.0, 10.0])           # follower faster than leader
    g = np.array([0, 0])
    d2, v2, ov = L.constrained_propagate(
        d, v, g, np.array([-9.0, 0.0]), [40.0] * 4)
    assert ov[0] and not ov[1]           # follower flagged, leader untouched
    assert d2[0] == pytest.approx(d2[1] + 5.0)   # placed 5 m behind


def test_constrained_cascade_front_to_back():
    # three vehicles same lane; front is slowest -> rear ones catch up,
    # each flagged and corrected sequentially, order preserved
    d = np.array([100.0, 99.0, 98.0])
    v = np.array([40.0, 35.0, 30.0])     # rear fastest, front slowest
    g = np.array([0, 0, 0])
    d2, v2, ov = L.constrained_propagate(d, v, g, np.array([0.0, 0.0, 0.0]),
                                         [40.0] * 4)
    assert ov[0] and ov[1] and not ov[2]
    assert d2[2] < d2[1] < d2[0]         # front first, order preserved
    assert d2[1] == pytest.approx(d2[2] + 5.0)
    assert d2[0] == pytest.approx(d2[1] + 5.0)


def test_constrained_no_false_overlap_with_healthy_gaps():
    d = np.array([200.0, 120.0, 60.0])
    v = np.array([20.0, 20.0, 20.0])
    g = np.array([2, 1, 0])
    d2, v2, ov = L.constrained_propagate(d, v, g, np.zeros(3), [40.0] * 4)
    assert not ov.any()
    np.testing.assert_allclose(d2, d - 20.0)


def test_ttc_and_brake_events_synthetic():
    w = L.wm()
    # follower at 10 m behind, closing at 4 m/s -> TTC 2.5 s < 3 s
    snap = {"a": (100.0, 24.0, 0), "b": (90.0, 20.0, 0)}
    assert L.ttc_events_at(snap) == {"a"}
    # opening: follower slower -> no event
    snap2 = {"a": (100.0, 18.0, 0), "b": (90.0, 20.0, 0)}
    assert L.ttc_events_at(snap2) == set()
    # free driver: no in-group leader -> no event
    snap3 = {"a": (100.0, 30.0, 0), "b": (90.0, 20.0, 1)}
    assert L.ttc_events_at(snap3) == set()
    # hard braking over 1 s (transition t=10 -> 11, half-open range)
    states = {10: {"a": (0.0, 20.0, 0)}, 11: {"a": (-17.5, 16.5, 0)}}
    assert L.brake_events(states, 10, 11) == {"a"}
    states_soft = {10: {"a": (0.0, 20.0, 0)}, 11: {"a": (-17.5, 17.5, 0)}}
    assert L.brake_events(states_soft, 10, 11) == set()


def test_event_matching_tolerance_and_one_to_one():
    truth = [("a", 10), ("a", 11)]
    pred = [("a", 12)]
    tm, nt, pm, np_ = L.match_events(truth, pred, tol=2)
    assert (tm, nt, pm, np_) == (1, 2, 1, 1)
    # outside tolerance: dt=3 not matched
    tm, nt, pm, np_ = L.match_events(truth, [("a", 14)], tol=2)
    assert tm == 0
    # different vehicle never matches
    tm, nt, pm, np_ = L.match_events(truth, [("b", 10)], tol=2)
    assert tm == 0 and pm == 0
    # greedy nearest takes the closest pair
    truth = [("a", 10)]
    pred = [("a", 8), ("a", 12)]
    tm, nt, pm, np_ = L.match_events(truth, pred, tol=2)
    assert (tm, nt, pm, np_) == (1, 1, 1, 2)
