"""R6 gap-opener headway controller sanity (A2.3, zero SUMO)."""
from __future__ import annotations

import math

from inc1_gen.generators import (R6_HEADWAY_GAIN_1PS, R6_HEADWAY_S,
                                 R6GapOpening, headway_accel)

from inc1_gen.tests._helpers import OBSTACLE, T0_S, as_states

GEN_FREE_ACCEL = 5.0      # generous native cap so the controller binds


def opener(gap, v, native=GEN_FREE_ACCEL, lane=0, d=100.0, vid="op"):
    return dict(vehicle_id=vid, is_cav=True, lane_phys=lane,
                s_down=OBSTACLE.s_down - d, v=v, a=0.0, leader_gap=gap,
                leader_v=None, native_accel=native, present=True)


def companion(vid="op2", d=550.0):
    """A second, passive opener so the single-opener probes stay feasible."""
    return opener(gap=None, v=20.0, native=0.0, d=d, vid=vid)


def opener_world(gap, v, native=GEN_FREE_ACCEL):
    return [opener(gap=gap, v=v, native=native), companion()]


def test_headway_accel_formula():
    # a_hc = k * (max(0, gap)/h - v); exactly on the tracked headway -> 0
    assert headway_accel(R6_HEADWAY_S * 10.0, 10.0) == 0.0
    assert headway_accel(90.0, 30.0) == 0.0
    # too much gap: accelerate toward the headway-implied speed
    assert headway_accel(90.0, 10.0) == R6_HEADWAY_GAIN_1PS * (30.0 - 10.0)
    # too little gap: brake
    assert headway_accel(30.0, 20.0) == R6_HEADWAY_GAIN_1PS * (10.0 - 20.0)
    # a negative gap is never trusted
    assert headway_accel(-5.0, 0.0) == 0.0


def test_opener_command_is_min_native_headway_within_limits():
    # headway term small: command = a_hc (below the native cap)
    rows = opener_world(gap=30.0, v=20.0)   # a_hc = 0.5*(10-20) = -5
    gen = R6GapOpening(as_states(rows), OBSTACLE, t0_s=T0_S)
    step = gen.step(T0_S + 1.0, as_states(rows))
    assert step.commands["op"] == -1.5      # ...then clipped to the floor
    assert step.commands["op2"] == 0.0      # passive companion: native-only
    # headway term mild: command = a_hc unchanged (inside the limits)
    rows = opener_world(gap=54.0, v=20.0)   # a_hc = 0.5*(18 - 20) = -1.0
    gen = R6GapOpening(as_states(rows), OBSTACLE, t0_s=T0_S)
    step = gen.step(T0_S + 1.0, as_states(rows))
    assert step.commands["op"] == -1.0
    # headway wants MORE than native: capped at the native IDM bound
    rows = opener_world(gap=120.0, v=5.0, native=0.7)  # a_hc = 0.5*(40-5)
    gen = R6GapOpening(as_states(rows), OBSTACLE, t0_s=T0_S)
    step = gen.step(T0_S + 1.0, as_states(rows))
    assert step.commands["op"] == 0.7


def test_opener_without_a_leader_runs_native():
    rows = opener_world(gap=None, v=10.0, native=1.3)
    gen = R6GapOpening(as_states(rows), OBSTACLE, t0_s=T0_S)
    step = gen.step(T0_S + 1.0, as_states(rows))
    assert step.commands["op"] == 1.3


def test_headway_tracking_closes_the_gap_error_closed_loop():
    # a leader receding at constant speed; the opener relaxes its speed
    # toward gap/h and the time headway converges toward h
    leader_v = 12.0
    gap = 30.0
    v = 18.0
    gen = R6GapOpening(as_states(opener_world(gap=gap, v=v)), OBSTACLE,
                       t0_s=T0_S)
    for k in range(1, 121):
        rows = opener_world(gap=gap, v=v)
        step = gen.step(T0_S + float(k), as_states(rows))
        cmd = step.commands["op"]
        assert -1.5 - 1e-12 <= cmd <= GEN_FREE_ACCEL
        v = max(0.0, v + cmd)
        gap = max(0.0, gap + (leader_v - v))   # leader recedes at 12 m/s
    headway = gap / max(v, 1e-9)
    assert abs(headway - R6_HEADWAY_S) < 0.5, (gap, v, headway)
    assert abs(v - leader_v) < 0.5


def test_approacher_tracks_min_12_native():
    # one opener + one approacher: feasible (2 members, >= 1 opener)
    def world_for(row):
        return [opener(gap=None, v=20.0, native=0.0, d=550.0), row]

    row = dict(vehicle_id="ap", is_cav=True, lane_phys=1,
               s_down=OBSTACLE.s_down - 50.0, v=8.0, a=0.0, leader_gap=None,
               leader_v=None, native_accel=2.0, present=True)
    gen = R6GapOpening(as_states(world_for(row)), OBSTACLE, t0_s=T0_S)
    step = gen.step(T0_S + 1.0, as_states(world_for(row)))
    # target = min(12, 8 + 2*1) = 10 -> command = (10 - 8)/1
    assert step.commands["ap"] == 2.0
    fast = dict(row, v=15.0)
    gen = R6GapOpening(as_states(world_for(fast)), OBSTACLE, t0_s=T0_S)
    step = gen.step(T0_S + 1.0, as_states(world_for(fast)))
    # target = min(12, ...) = 12 -> decel toward 12, (12-15)/1 = -3 -> floor
    assert step.commands["ap"] == -1.5
    weak = dict(row, v=11.5, native_accel=0.2)
    gen = R6GapOpening(as_states(world_for(weak)), OBSTACLE, t0_s=T0_S)
    step = gen.step(T0_S + 1.0, as_states(world_for(weak)))
    # target = min(12, 11.5+0.2) = 11.7 -> (11.7-11.5)/1 = 0.2 (= native)
    assert math.isclose(step.commands["ap"], 0.2)
    unknown = dict(row, v=11.5, native_accel=None)
    gen = R6GapOpening(as_states(world_for(unknown)), OBSTACLE, t0_s=T0_S)
    step = gen.step(T0_S + 1.0, as_states(world_for(unknown)))
    assert step.commands["ap"] == 0.0
