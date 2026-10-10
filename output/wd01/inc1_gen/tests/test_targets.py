"""Staggered-target and window-behaviour tests for R5 (A2.2, zero SUMO)."""
from __future__ import annotations

import math

from inc1_gen.generators import (R5_DURATION_S, R5Harmonisation,
                                 staggered_target)

from inc1_gen.tests._helpers import OBSTACLE, T0_S, as_states


def make_member(d, v, native=10.0, vid=None):
    return dict(vehicle_id=(vid or f"cav@{d}"), is_cav=True, lane_phys=0,
                s_down=OBSTACLE.s_down - d, v=v, a=0.0, leader_gap=None,
                leader_v=None, native_accel=native, present=True)


def member_world(d, v, native=10.0):
    """One probed member plus a far passive companion (feasibility >= 2)."""
    return [make_member(d, v, native=native),
            make_member(1450.0, v=staggered_target(1450.0), native=5.0,
                        vid="cav.far")]


def test_staggered_target_formula_and_clips():
    assert staggered_target(100.0) == 10.0
    assert staggered_target(1500.0) == 22.0
    assert staggered_target(800.0) == 16.0          # 10 + 12*700/1400
    assert math.isclose(staggered_target(450.0), 13.0)
    # clip: below d=100 the raw value dips under 10 and stays 10; above 1500
    # it saturates at 22
    assert staggered_target(50.0) == 10.0
    assert staggered_target(0.0) == 10.0
    assert staggered_target(2000.0) == 22.0
    assert staggered_target(-500.0) == 10.0


def test_members_nearer_the_obstacle_are_slowed_more():
    near = make_member(100.0, v=10.5)     # v_tgt 10 -> mild decel
    far = make_member(1400.0, v=20.0)     # v_tgt ~21.14 -> gentle accel
    gen = R5Harmonisation(as_states([near, far]), OBSTACLE, t0_s=T0_S)
    step = gen.step(T0_S + 1.0, as_states([near, far]))
    cmds = step.commands
    assert set(cmds) == {near["vehicle_id"], far["vehicle_id"]}
    assert cmds[near["vehicle_id"]] < cmds[far["vehicle_id"]]
    # free flow: with a generous native cap the command is exactly
    # (v_tgt - v) / dt
    assert cmds[near["vehicle_id"]] == (10.0 - 10.5) / 1.0
    assert cmds[far["vehicle_id"]] == (staggered_target(1400.0) - 20.0) / 1.0
    # both braking hard saturate at the -1.5 floor (still members: 2 total)
    near_fast = make_member(100.0, v=40.0)
    far_fast = make_member(1400.0, v=40.0)
    gen = R5Harmonisation(as_states([near_fast, far_fast]), OBSTACLE,
                          t0_s=T0_S)
    cmds = gen.step(T0_S + 1.0, as_states([near_fast, far_fast])).commands
    assert cmds[near_fast["vehicle_id"]] == -1.5
    assert cmds[far_fast["vehicle_id"]] == -1.5


def test_target_is_recomputed_each_tick_from_updated_d():
    gen = R5Harmonisation(as_states(member_world(300.0, 12.0)), OBSTACLE,
                          t0_s=T0_S)
    world = member_world(300.0, 12.0)
    for k in range(1, 6):
        step = gen.step(T0_S + float(k), as_states(world))
        state = world[0]
        vid = state["vehicle_id"]
        cmd = step.commands[vid]
        d = OBSTACLE.s_down - state["s_down"]
        v_tgt = staggered_target(d)
        assert cmd == (v_tgt - state["v"]) / 1.0
        v_new = max(0.0, state["v"] + cmd)
        world = [dict(state, v=v_new, s_down=state["s_down"] + v_new)] + \
            world[1:]


def test_window_closes_after_120_s_then_native():
    world = member_world(300.0, 12.0)
    gen = R5Harmonisation(as_states(world), OBSTACLE, t0_s=T0_S)
    last_cmd = None
    for k in range(1, int(R5_DURATION_S) + 3):
        step = gen.step(T0_S + float(k), as_states(world))
        if T0_S + float(k) - T0_S < R5_DURATION_S:
            assert step.commands, f"tick {k} inside the window must command"
            last_cmd = step.commands
        else:
            assert step.commands == {}
            assert "cav@300.0" in step.native
    assert last_cmd is not None


def test_member_above_target_is_only_never_accelerated_past_native():
    # v above v_tgt: negative command; decel bounded at -1.5 even if the gap
    # is huge
    world = member_world(1400.0, 40.0, native=2.5)
    gen = R5Harmonisation(as_states(world), OBSTACLE, t0_s=T0_S)
    step = gen.step(T0_S + 1.0, as_states(world))
    assert step.commands["cav@1400.0"] == -1.5
    # v below v_tgt: positive request clipped to the native IDM bound
    world = member_world(1400.0, 5.0, native=0.8)
    gen = R5Harmonisation(as_states(world), OBSTACLE, t0_s=T0_S)
    step = gen.step(T0_S + 1.0, as_states(world))
    assert step.commands["cav@1400.0"] == 0.8


def test_unknown_native_accel_never_accelerates():
    world = [make_member(800.0, 5.0, native=None),
             make_member(1450.0, 21.857142857142858, native=5.0,
                         vid="cav.far")]
    assert world[0]["native_accel"] is None
    gen = R5Harmonisation(as_states(world), OBSTACLE, t0_s=T0_S)
    step = gen.step(T0_S + 1.0, as_states(world))
    assert step.commands["cav@800.0"] == 0.0


def test_target_update_uses_current_d_not_t0_d():
    row = make_member(1400.0, 22.0)
    companion = make_member(1450.0, 21.857142857142858, native=5.0,
                            vid="cav.far")
    gen = R5Harmonisation(as_states([row, companion]), OBSTACLE, t0_s=T0_S)
    # verify the command follows d from the CURRENT s_down, not t0's
    frozen = [dict(row), dict(companion)]
    step = gen.step(T0_S + 1.0, as_states(frozen))
    d_now = OBSTACLE.s_down - row["s_down"]
    assert step.commands[row["vehicle_id"]] == (
        staggered_target(d_now) - row["v"])
    moved = [dict(row, s_down=row["s_down"] + 50.0), dict(companion)]
    step = gen.step(T0_S + 2.0, as_states(moved))
    d_new = OBSTACLE.s_down - moved[0]["s_down"]
    assert step.commands[row["vehicle_id"]] == (
        staggered_target(d_new) - row["v"])
