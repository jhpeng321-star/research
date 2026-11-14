"""Generated-fixture tests for the packed per-CAV schedule rollout (E1).

Every fixture is synthetic (tiny K, a few steps, stub bundle); no real pack,
CAL shard, checkpoint or outcome file is touched.
"""
from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from evalh.scorer.actions import (MissingActionSchedule, actions_for_step,
                                  rollout_with_schedule, schedule_from_packed)
from features.constants import ACTIONS, PLAN_ACTION_SLOTS, PLAN_WALL_SPEED
from rollout2.engine import IdeaABundle
from rollout2.state import WorldState
from wm_api.engine import _actions_for_tick
from wm_api.plans import ACTION_CODES as WAPI_CODES
from wm_api.plans import ResolvedPlan

_K = 4
_CAV = (0, 1)


def _world(*, cav_slots=_CAV, k_n=_K) -> WorldState:
    s = torch.tensor([[100.0, 130.0, 160.0, 190.0][:k_n]])
    v = torch.full((1, k_n), 20.0)
    a = torch.zeros((1, k_n))
    lane = torch.zeros((1, k_n), dtype=torch.int64)
    length = torch.full((1, k_n), 5.0)
    present = torch.ones((1, k_n), dtype=torch.bool)
    is_cav = torch.zeros((1, k_n), dtype=torch.bool)
    for j in cav_slots:
        is_cav[0, j] = True
    return WorldState(
        s_m=s, v_mps=v, a_mps2=a, lane=lane, length_m=length, present=present,
        is_cav=is_cav,
        wall_group=torch.full((1, k_n), -1, dtype=torch.int64),
        slot_id=torch.arange(k_n, dtype=torch.int64).reshape(1, k_n))


def _mask(cav_slots=_CAV, k_n=_K):
    return [j in cav_slots for j in range(k_n)]


def _encode(tokens, *, k_n=_K, cav_slots=_CAV, wall_speed_mps=None):
    """``tokens`` is ``[n_steps, len(cav_slots)]`` action names."""
    tokens = np.asarray(tokens, dtype=object)
    n = tokens.shape[0]
    plan = np.zeros((n, k_n, 8), dtype=np.float64)
    for t in range(n):
        for c, slot in enumerate(cav_slots):
            plan[t, slot, PLAN_ACTION_SLOTS + ACTIONS.index(str(tokens[t, c]))] = 1.0
            if str(tokens[t, c]) == "wall" and wall_speed_mps is not None:
                plan[t, slot, PLAN_WALL_SPEED] = wall_speed_mps / 40.0
    return plan


def _schedule(tokens, **kw):
    return schedule_from_packed(
        _encode(tokens, **kw), run_key="synth/run", local_t0_tick=0,
        n_steps=np.asarray(tokens, dtype=object).shape[0],
        is_cav_t0=_mask(kw.get("cav_slots", _CAV), kw.get("k_n", _K)))


class _StubMover:
    """Minimal P1 mover: damping/pi/wall map to deterministic target speeds."""

    def init_state(self, batch, slots, **kwargs):
        return types.SimpleNamespace()

    def step(self, state, inp, actions, *, wall_speed_mps=None):
        shape = inp.s_m.shape
        target = torch.full(shape, float("nan"), dtype=torch.float64)
        codes = actions.to(torch.int64)
        target = torch.where(codes == 1, torch.full_like(target, 10.0), target)
        target = torch.where(codes == 2, torch.full_like(target, 12.0), target)
        if wall_speed_mps is not None:
            target = torch.where(codes == 3, wall_speed_mps.to(torch.float64),
                                 target)
        reason = torch.zeros(shape, dtype=torch.int64)
        return types.SimpleNamespace(target_speed_mps=target,
                                     reason_code=reason)


class _StubResidual:
    def __call__(self, kin, mask):
        b_n, k_n = mask.shape
        return {"accel_delta_mps2": torch.zeros((b_n, k_n), dtype=torch.float64),
                "lane_logit_delta": torch.zeros((b_n, k_n, 3),
                                                dtype=torch.float64)}


def _bundle(hdv=None) -> IdeaABundle:
    return IdeaABundle(ctrl=_StubMover(), hdv=hdv, arrival=None, corr=None)


def test_two_schedules_differ_in_cav_trajectories():
    world = _world()
    native = _schedule([["native", "native"]] * 6)
    damping = _schedule([["damping", "native"]] * 6)
    bundle = _bundle()
    r_n = rollout_with_schedule(bundle, world, native, n_steps=6,
                               integrator="d1", residual_enabled=False)
    r_d = rollout_with_schedule(bundle, world, damping, n_steps=6,
                               integrator="d1", residual_enabled=False)
    cav = r_n.positions[0, :, 0, 0]
    assert not np.allclose(cav.numpy(), r_d.positions[0, :, 0, 0].numpy())
    assert float(r_d.speeds[0, -1, 0, 0]) < float(r_n.speeds[0, -1, 0, 0])


def test_actions_for_step_matches_real_action_helper():
    world = _world()
    schedule = _schedule([["damping", "pi"]] * 3)
    mine = actions_for_step(schedule, 1, world)
    resolved = ResolvedPlan(
        index=0, n_segments=1,
        segment_tokens=[{"0": "damping", "1": "pi"}],
        segment_codes=[{"0": WAPI_CODES["damping"], "1": WAPI_CODES["pi"]}],
        wall=None)
    reference = _actions_for_tick(resolved, world, {0: "0", 1: "1"}, _K,
                                  torch.full((1, _K), float("nan")), 0)
    assert torch.equal(mine.action_code, reference.action_code)


def test_schedule_slice_and_native_default_match():
    world = _world()
    plan = _encode([["native", "native"]] * 5)
    schedule = schedule_from_packed(plan, run_key="synth/run", local_t0_tick=2,
                                    n_steps=3, is_cav_t0=_mask())
    assert schedule.plan.shape == (3, _K, 8)
    acts = actions_for_step(schedule, 0, world)
    assert int(acts.action_code[0, 0]) == 0
    assert int(acts.action_code[0, 2]) == -1
    assert int(acts.action_code[0, 3]) == -1


def test_hdv_slots_never_receive_actions():
    world = _world()
    plan = _encode([["native", "native"]] * 3)
    plan[:, 2, :] = np.nan          # junk in HDV rows is never read
    plan[:, 3, 0] = 5.0
    schedule = schedule_from_packed(plan, run_key="synth/run", local_t0_tick=0,
                                    n_steps=3, is_cav_t0=_mask())
    acts = actions_for_step(schedule, 0, world)
    assert int(acts.action_code[0, 2]) == -1
    assert int(acts.action_code[0, 3]) == -1
    assert int(acts.action_code[0, 0]) == 0


def test_wall_action_decodes_wall_speed():
    world = _world()
    schedule = _schedule([["wall", "native"]] * 2, wall_speed_mps=6.5)
    acts = actions_for_step(schedule, 0, world)
    assert int(acts.action_code[0, 0]) == WAPI_CODES["wall"]
    assert float(acts.wall_speed_mps[0, 0]) == pytest.approx(6.5)
    r = rollout_with_schedule(_bundle(), world, schedule, n_steps=2,
                              integrator="v2g", residual_enabled=False)
    assert torch.isfinite(r.speeds).all()


def test_failure_modes_fail_closed():
    mask = _mask()
    good = _encode([["native", "native"]] * 3)

    with pytest.raises(MissingActionSchedule):
        actions_for_step(None, 0, _world())
    with pytest.raises(MissingActionSchedule):
        schedule_from_packed(None, run_key="r", local_t0_tick=0, n_steps=3,
                             is_cav_t0=mask)
    with pytest.raises(MissingActionSchedule):
        schedule_from_packed(np.zeros((3, _K)), run_key="r", local_t0_tick=0,
                             n_steps=3, is_cav_t0=mask)

    nan = good.copy()
    nan[1, 0, 0] = np.nan
    with pytest.raises(MissingActionSchedule):
        schedule_from_packed(nan, run_key="r", local_t0_tick=0, n_steps=3,
                             is_cav_t0=mask)

    zero = good.copy()
    zero[:, 0, :5] = 0.0
    with pytest.raises(MissingActionSchedule):
        schedule_from_packed(zero, run_key="r", local_t0_tick=0, n_steps=3,
                             is_cav_t0=mask)

    with pytest.raises(MissingActionSchedule):
        schedule_from_packed(good, run_key="r", local_t0_tick=1, n_steps=3,
                             is_cav_t0=mask)

    schedule = schedule_from_packed(good, run_key="r", local_t0_tick=0,
                                    n_steps=3, is_cav_t0=mask)
    with pytest.raises(MissingActionSchedule):
        actions_for_step(schedule, 0, _world(cav_slots=(0, 2)))
    with pytest.raises(MissingActionSchedule):
        actions_for_step(schedule, 5, _world())
    with pytest.raises(MissingActionSchedule):
        rollout_with_schedule(_bundle(), _world(), schedule, n_steps=9,
                              integrator="d1", residual_enabled=False)
    with pytest.raises(MissingActionSchedule):
        rollout_with_schedule(_bundle(), _world(), schedule, n_steps=1,
                              integrator="bogus", residual_enabled=False)


def test_v2g_path_runs_with_residual_off_and_carries_channels():
    world = _world()
    schedule = _schedule([["native", "native"]] * 4)
    r = rollout_with_schedule(_bundle(), world, schedule, n_steps=4,
                              integrator="v2g", residual_enabled=False)
    assert r.positions.shape == (1, 5, 1, _K)
    assert r.speeds.shape == (1, 5, 1, _K)
    assert r.accel.shape == (1, 4, 1, _K)
    assert r.masks.shape == (1, 5, 1, _K)
    assert r.slot_ids.shape == (1, 5, 1, _K)
    assert r.t0_kin.shape == (1, _K, 7)
    assert r.t0_mask.shape == (1, _K)
    assert r.t0_is_cav.shape == (1, _K)
    assert r.t0_slot_id.shape == (1, _K)
    assert torch.isfinite(r.positions).all()
    assert torch.isfinite(r.speeds).all()
    assert torch.equal(r.masks[0, 0], r.t0_mask)
    assert torch.equal(r.slot_ids[0, 0], r.t0_slot_id)


def test_d1_path_runs_with_stub_residual():
    world = _world()
    schedule = _schedule([["damping", "pi"]] * 4)
    r = rollout_with_schedule(_bundle(hdv=_StubResidual()), world, schedule,
                              n_steps=4, integrator="d1",
                              residual_enabled=True, n_samples=2, seed=3)
    assert r.positions.shape == (2, 5, 1, _K)
    assert torch.isfinite(r.positions).all()
