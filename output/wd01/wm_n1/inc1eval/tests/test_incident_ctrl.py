"""R1-R4 incident mirrors: selection, scalar parity, R3/R4 wiring."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from inc1eval import incident_ctrl as IC
from inc1eval.tests._helpers import make_world


def _inp(world, t):
    return SimpleNamespace(
        s_m=world.s_m, v_mps=world.v_mps, lane=world.lane,
        length_m=world.length_m, present=world.present, is_cav=world.is_cav,
        wall_group=world.wall_group, segment_lo_m=world.s_m,
        segment_hi_m=world.s_m, segment_limit_mps=world.s_m, t_s=float(t),
        control_t_s=float(t))


def test_r1_selects_nearest_upstream_mainline_within_range():
    world = make_world([(1, 150.0, 10.0, True), (0, 60.0, 12.0, True),
                        (1, 10.0, 12.0, True)])
    sel = IC.r1_select(world, 200.0)
    assert bool(sel[0, 0]) and not bool(sel[0, 1]) and not bool(sel[0, 2])
    # beyond 500 m: no selection
    world2 = make_world([(1, 150.0, 10.0, True)])
    assert not bool(IC.r1_select(world2, 900.0).any())


def test_r2_selects_window():
    world = make_world([(1, 900.0, 20.0, True), (1, 100.0, 20.0, True),
                        (1, 850.0, 20.0, True)])
    # obstacle at 1200: window [200, 1000)
    sel = IC.r2_select(world, 1200.0)
    assert bool(sel[0, 0]) and bool(sel[0, 2]) and not bool(sel[0, 1])


def test_r1_mirror_parity_against_scalar_reference():
    world = make_world([(1, 150.0, 10.0, True)])
    obs_s = 200.0
    mover = IC.IncidentCavMover("R1", obs_s)
    state = mover.init_state(1, world.slots)
    for t in range(0, 40):
        out = mover.step(state, _inp(world, t))
        got = float(out.target_speed_mps[0, 0])
        want = IC.r1_scalar_target(10.0, float(t))
        if want != want:                                  # NaN native handoff
            assert got != got
        else:
            assert got == pytest.approx(want, abs=1e-9)
    # after t_stop (5 s) + hold (30 s) the arm is native
    assert float(out.target_speed_mps[0, 0]) != float(out.target_speed_mps[0, 0])


def test_r2_mirror_parity_against_scalar_reference():
    world = make_world([(1, 900.0, 20.0, True)])
    mover = IC.IncidentCavMover("R2", 1200.0)
    state = mover.init_state(1, world.slots)
    for t in range(0, 130):
        out = mover.step(state, _inp(world, t))
        got = float(out.target_speed_mps[0, 0])
        want = IC.r2_scalar_target(20.0, float(t))
        if want != want:
            assert got != got
        else:
            assert got == pytest.approx(want, abs=1e-9)
    assert float(out.target_speed_mps[0, 0]) != float(out.target_speed_mps[0, 0])


def test_r1_mover_in_rollout_brakes_at_two():
    from inc1eval.rollout_baseline import idea_a_rollout
    from rollout2.engine import IdeaABundle

    world = make_world([(1, 150.0, 10.0, True)])
    mover = IC.IncidentCavMover("R1", 200.0)
    bundle = IdeaABundle(ctrl=mover, hdv=None, arrival=None, corr=None)
    out = idea_a_rollout(bundle, world, n_steps=12, seed=0)
    v = out.speeds[0, :, 0, 0].numpy()
    # speed falls ~2 m/s per second (two 0.5 s ticks) while above 0
    assert v[0] == pytest.approx(10.0)
    assert v[-1] < v[0]
    assert v[-1] >= 0.0


def test_r3_action_codes_damping_upstream():
    world = make_world([(1, 150.0, 10.0, True), (1, 250.0, 10.0, True)])
    codes = IC.r3_action_codes(world, 200.0)
    assert int(codes[0, 0]) == IC.ACTION_CODES["damping"]
    assert int(codes[0, 1]) == IC.ACTION_CODES["native"]   # downstream


def test_r4_members_one_per_lane_and_feasibility():
    world = make_world([(0, -90.0, 25.0, True), (1, -100.0, 25.0, True),
                        (1, -300.0, 25.0, True),
                        (1, 40.0, 0.0, False)])
    members, refusal = IC.r4_members(world, 40.0)
    assert refusal is None
    assert [ln for ln, _ in members] == [0, 1]
    # near the merge start with a slow wall speed the D_min gate passes
    _m, feasible, _r = IC.r4_assembly(world, 40.0, 1.0)
    assert feasible
    # a fast wall speed makes the same pair infeasible (D_min = 70 * v)
    _m, feasible_fast, refusal_fast = IC.r4_assembly(world, 40.0, 30.0)
    assert not feasible_fast and refusal_fast["reason"] == "dmin_unreachable"


def test_r4_action_codes_and_speeds():
    world = make_world([(0, -90.0, 25.0, True), (1, -100.0, 25.0, True)])
    members = [(0, 0), (1, 1)]
    codes = IC.r4_action_codes(world, members)
    assert int(codes[0, 0]) == IC.ACTION_CODES["wall"]
    assert int(codes[0, 1]) == IC.ACTION_CODES["wall"]
    speeds = IC.r4_wall_speeds(world, members, 20.0)
    assert float(speeds[0, 0]) == 20.0 and float(speeds[0, 1]) == 20.0
    assert torch.isnan(speeds[0, 2])
