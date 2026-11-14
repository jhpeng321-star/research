"""Synthetic tests for the flag-gated calibrated merge (``merge_cal``).

The calibrated physics (``rollout2.merge_cal``) replaces the frozen v2g
merge rule with a strategic (LC2013-style) mandatory merge plus bounded
cooperative mainline yielding, fitted on TRAIN chunk-4 recorded merge
statistics only (``merge_cal_params.json``).  The flag is default OFF:
``step_once`` keeps the nominal ``V2GCore`` unless
``Rollout2Config.merge_cal`` is set.
"""
from __future__ import annotations

import inspect
from dataclasses import fields

import torch

from rollout2.engine import IdeaABundle
from rollout2.merge_cal import MergeCalParams
from rollout2.state import (Rollout2Config, Rollout2State, StrategyActions,
                            WorldState)
from rollout2.step import step_once


def _world(rows):
    k = len(rows)
    s = torch.zeros(1, k, dtype=torch.float64)
    v = torch.zeros(1, k, dtype=torch.float64)
    lane = torch.full((1, k), -1, dtype=torch.int64)
    present = torch.zeros((1, k), dtype=torch.bool)
    for i, (ln, ss, vv) in enumerate(rows):
        s[0, i] = ss
        v[0, i] = vv
        lane[0, i] = ln
        present[0, i] = True
    return WorldState(
        s_m=s, v_mps=v, a_mps2=torch.zeros(1, k, dtype=torch.float64),
        lane=lane, length_m=torch.full((1, k), 5.0, dtype=torch.float64),
        present=present, is_cav=torch.zeros(1, k, dtype=torch.bool),
        wall_group=torch.full((1, k), -1, dtype=torch.int64),
        slot_id=torch.arange(k).expand(1, k).contiguous())


def _run(world, *, merge_cal, n_steps=1, t0=0.0):
    bundle = IdeaABundle(ctrl=None, hdv=None, arrival=None, corr=None)
    cfg = Rollout2Config(t0_s=t0, sumo_fidelity=True, merge_cal=merge_cal)
    cfg.state = Rollout2State(world.batch, world.slots)
    acts = StrategyActions(
        action_code=torch.full((world.batch, world.slots), -1,
                               dtype=torch.int64))
    w = world
    for i in range(n_steps):
        w, _ = step_once(w, acts, bundle=bundle, config=cfg,
                         t_s=t0 + 0.5 * i)
    return w


def test_merge_cal_default_off():
    assert Rollout2Config().merge_cal is False
    import closedloop.a11_predictor as a11

    assert inspect.signature(a11.batched_rollout).parameters[
        "merge_cal"].default is False
    assert inspect.signature(a11.A11Predictor.__init__).parameters[
        "merge_cal"].default is False


def test_fitted_params_load():
    mp = MergeCalParams.load()
    assert mp.b_yield_max_mps2 == 1.75
    assert mp.yield_min_speed_mps == 2.5
    assert mp.yield_min_merger_speed_mps == 3.5
    assert set(mp.as_dict()) == {f.name for f in fields(MergeCalParams)}


def test_merge_cal_accepts_safe_strategic_gap():
    # Urgent lane-2 vehicle, comfortable gap to a same-speed lane-1
    # follower: the strategic merge takes the gap in one step.
    world = _world([(2, 45.0, 25.0), (1, 5.0, 25.0)])
    fixed = _run(world, merge_cal=True, n_steps=1)
    assert int(fixed.lane[0, 0]) == 1


def test_merge_cal_refuses_overlapping_gap():
    # Bumper gap below the floor is never accepted, urgency or not.
    world = _world([(2, 45.0, 25.0), (1, 42.0, 25.0)])
    fixed = _run(world, merge_cal=True, n_steps=1)
    assert int(fixed.lane[0, 0]) == 2


def test_yield_only_to_moving_merger():
    # A mainline follower brakes (bounded) for a MOVING urgent merger but
    # never for a (nearly) stopped one (yield_min_merger_speed_mps).
    move = _run(_world([(1, 30.0, 20.0), (2, 45.0, 10.0)]),
                merge_cal=True, n_steps=1)
    stop = _run(_world([(1, 30.0, 20.0), (2, 45.0, 0.0)]),
                merge_cal=True, n_steps=1)
    # follower (row 0): free-road accel vs the bounded yield brake
    assert float(stop.v_mps[0, 0]) > float(move.v_mps[0, 0]) + 0.5
    # and in the moving case the follower actually decelerated
    assert float(move.v_mps[0, 0]) < 20.0


def test_default_off_is_byte_identical():
    world = _world([(0, 300.0, 25.0), (1, -200.0, 20.0), (2, 60.0, 15.0)])
    a = _run(world, merge_cal=False, n_steps=8)
    b = _run(world, merge_cal=False, n_steps=8)
    assert torch.equal(a.s_m, b.s_m)
    assert torch.equal(a.v_mps, b.v_mps)
    assert torch.equal(a.present, b.present)
