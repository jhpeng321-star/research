"""End-to-end two-call-site identity for R5/R6 (ADDENDUM_2 A2.6).

The SAME synthetic state sequence is driven through the SUMO-side schedule
(``inc1.scripted.ScriptedSchedule`` -> ``states_from_traci_like``) and the
WM-side mover (``inc1eval.scripted_mover.ScriptedCavMover`` ->
``states_from_wm_arrays``).  Both call the ONE shared ``inc1_gen`` generator,
so every issued R5/R6 acceleration must match bit-for-bit.
"""
from __future__ import annotations

import pytest

from inc1 import scripted as SC
from inc1.schedules import CavState
from inc1eval.scripted_mover import ScriptedCavMover, wm_frame_from_states

OBSTACLE_S = 40.0
T0_S = 300.0


def _cav(vid, s_down, lane, *, speed, native, gap=None, leader_v=None):
    return CavState(vehicle_id=vid, speed_mps=speed, s_down_m=s_down,
                    lane_phys=lane, allowed_speed_mps=30.0,
                    native_accel_mps2=native, leader_gap_m=gap,
                    leader_v_mps=leader_v)


def _sequence(arm):
    """A 5-tick closed-loop-ish sequence exercising both generators."""
    if arm == "R5":
        # members on mainline lanes 0/1 with d in [100, 1500)
        frames = [
            [_cav("cav.a", OBSTACLE_S - 150.0, 0, speed=20.0, native=1.0),
             _cav("cav.b", OBSTACLE_S - 300.0, 1, speed=18.0, native=0.8)],
            [_cav("cav.a", OBSTACLE_S - 137.0, 0, speed=19.0, native=1.0),
             _cav("cav.b", OBSTACLE_S - 282.0, 1, speed=17.5, native=0.8)],
            [_cav("cav.a", OBSTACLE_S - 125.0, 0, speed=18.0, native=1.0),
             _cav("cav.b", OBSTACLE_S - 265.0, 1, speed=17.0, native=0.8)],
            [_cav("cav.a", OBSTACLE_S - 114.0, 0, speed=17.0, native=1.0)],
            [_cav("cav.a", OBSTACLE_S - 104.0, 0, speed=16.5, native=1.0)],
        ]
        return frames
    # R6: lane-0 gap openers + lane-1 blocked-lane approachers
    return [
        [_cav("cav.o", OBSTACLE_S - 200.0, 0, speed=20.0, native=1.0,
              gap=40.0, leader_v=15.0),
         _cav("cav.p", OBSTACLE_S - 80.0, 1, speed=22.0, native=0.5)],
        [_cav("cav.o", OBSTACLE_S - 188.0, 0, speed=19.0, native=1.0,
              gap=30.0, leader_v=15.0),
         _cav("cav.p", OBSTACLE_S - 72.0, 1, speed=21.5, native=0.5)],
        [_cav("cav.o", OBSTACLE_S - 176.0, 0, speed=18.0, native=1.0,
              gap=25.0, leader_v=15.0),
         _cav("cav.p", OBSTACLE_S - 64.0, 1, speed=21.0, native=0.5)],
        [_cav("cav.o", OBSTACLE_S - 164.0, 0, speed=17.0, native=1.0,
              gap=20.0, leader_v=15.0),
         _cav("cav.p", OBSTACLE_S - 56.0, 1, speed=20.5, native=0.5)],
        [_cav("cav.o", OBSTACLE_S - 153.0, 0, speed=16.0, native=1.0,
              gap=15.0, leader_v=15.0)],
    ]


@pytest.mark.parametrize("arm", ["R5", "R6"])
def test_sumo_schedule_and_wm_mover_commands_are_identical(arm):
    frames = _sequence(arm)
    schedule = SC.ScriptedSchedule(arm, frames[0], OBSTACLE_S, t0_s=T0_S)
    mover = ScriptedCavMover(arm, OBSTACLE_S, t0_s=T0_S)
    for k, states in enumerate(frames):
        now = T0_S + float(k)
        plan = schedule.step(now, states)
        commands, native = mover.commands_from_states(now, states)
        assert plan.wall == commands, (arm, k)
        assert set(plan.native) == set(native), (arm, k)
    assert schedule.feasible == mover.generator.feasible


@pytest.mark.parametrize("arm", ["R5", "R6"])
def test_identity_breaks_on_a_perturbed_wm_rendering(arm):
    frames = _sequence(arm)
    schedule = SC.ScriptedSchedule(arm, frames[0], OBSTACLE_S, t0_s=T0_S)
    mover = ScriptedCavMover(arm, OBSTACLE_S, t0_s=T0_S)
    plan = schedule.step(T0_S, frames[0])
    tampered = [dict(vars(s)) for s in frames[0]]
    # a genuine state change: the first vehicle has left -> membership differs
    tampered[0]["present"] = False
    tampered = [CavState(**row) for row in tampered]
    commands, _ = mover.commands_from_states(T0_S, tampered)
    assert plan.wall != commands


def test_wm_frame_and_sumo_rows_render_equal_records():
    from inc1_gen.adapters import (states_equal, states_from_traci_like,
                                   states_from_wm_arrays)
    for arm in ("R5", "R6"):
        states = _sequence(arm)[0]
        sumo = states_from_traci_like(SC.to_traci_rows(states))
        ids, kwargs = wm_frame_from_states(states)
        assert states_equal(sumo, states_from_wm_arrays(ids, **kwargs))
