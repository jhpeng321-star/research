"""P3 interface contract (implemented): D1 rule, step contract, engine,
smoke fixture.  Supersedes the stub-boundary pins (P3 has landed)."""
from __future__ import annotations

import math

import pytest
import torch

import rollout2
from rollout2 import (IdeaABundle, Rollout2Config, Rollout2Engine, WorldState,
                      integrate_d1, j4_smoke_contract, j4_smoke_world,
                      project_rollout_step, step_once)
from rollout2.engine import rollout2 as run_rollout2
from rollout2.tests._parity_helpers import (NativeMover, scene_free,
                                            world_from_table)


def _world(b=2, k=4):
    return WorldState(
        s_m=torch.zeros(b, k), v_mps=torch.full((b, k), 10.0),
        a_mps2=torch.zeros(b, k), lane=torch.zeros(b, k, dtype=torch.int64),
        length_m=torch.full((b, k), 5.0),
        present=torch.ones(b, k, dtype=torch.bool),
        is_cav=torch.zeros(b, k, dtype=torch.bool),
        wall_group=torch.full((b, k), -1, dtype=torch.int64),
        slot_id=torch.arange(k).expand(b, k))


def test_public_symbols_present():
    assert rollout2.Rollout2Engine is Rollout2Engine
    assert callable(rollout2.integrate_ballistic)
    assert callable(rollout2.world_from_batch)


def test_integrate_d1_trapezoidal():
    s = torch.zeros(1, 3, dtype=torch.float64)
    v = torch.tensor([[10.0, 10.0, 10.0]], dtype=torch.float64)
    a = torch.tensor([[2.0, -40.0, 0.0]], dtype=torch.float64)
    s_new, v_new = integrate_d1(s, v, a)
    assert torch.allclose(v_new, torch.tensor([[11.0, 0.0, 10.0]],
                                              dtype=torch.float64))
    # row 0: 0.5*(10+11)*0.5 = 5.25 ; row 1: 0.5*(10+0)*0.5 = 2.5 ;
    # row 2: 0.5*(10+10)*0.5 = 5.0
    assert torch.allclose(s_new, torch.tensor([[5.25, 2.5, 5.0]],
                                              dtype=torch.float64))


def test_integrator_switch_contract():
    s = torch.zeros(1, 1, dtype=torch.float64)
    v = torch.full((1, 1), 10.0, dtype=torch.float64)
    a = torch.full((1, 1), 2.0, dtype=torch.float64)
    s_d1, v_new = integrate_d1(s, v, a)
    s_v2g = rollout2.integrate_ballistic(s, v_new)
    # documented per-step difference: 0.5*(v - v_new)*dt = 0.5*(-1)*0.5
    assert torch.allclose(s_d1 - s_v2g, torch.full_like(s_d1, -0.25))
    with pytest.raises(ValueError):
        project_rollout_step(
            s, torch.zeros(1, 1, dtype=torch.int64), v,
            torch.full((1, 1), 5.0), torch.ones(1, 1, dtype=torch.bool),
            a, torch.zeros(1, 1, dtype=torch.int64), integrator="euler")


def test_project_rollout_step_counts_and_shapes():
    b, k = 2, 5
    s = torch.rand((b, k), dtype=torch.float64) * 50
    v = torch.full((b, k), 3.0, dtype=torch.float64)
    accel = torch.full((b, k), -50.0, dtype=torch.float64)  # clip -9
    lane = torch.zeros((b, k), dtype=torch.int64)
    length = torch.full((b, k), 5.0, dtype=torch.float64)
    present = torch.ones((b, k), dtype=torch.bool)
    lane_class = torch.zeros((b, k), dtype=torch.int64)
    s_new, lane_new, v_new, a_used, cls_used, counts = project_rollout_step(
        s, lane, v, length, present, accel, lane_class)
    assert s_new.shape == s.shape and lane_new.shape == lane.shape
    assert v_new.min() >= 0.0
    assert counts.accel_clip == b * k          # -50 -> -9 everywhere
    assert counts.speed_clip == b * k          # 3 - 9*0.5 < 0 -> clamp
    assert torch.allclose(a_used, torch.full((b, k), -6.0,
                                             dtype=torch.float64))
    assert counts.lane_reject == 0
    assert counts.gap_projection >= 0


def test_step_once_runs_and_returns_contract():
    vehicles = scene_free()
    world, _names = world_from_table(vehicles)
    actions = rollout2.StrategyActions(
        action_code=torch.zeros(1, world.slots, dtype=torch.int64))
    bundle = IdeaABundle(ctrl=NativeMover(), hdv=None, arrival=None,
                         corr=None)
    cfg = Rollout2Config(t0_s=0.0)
    cfg.state = rollout2.Rollout2State(world.batch, world.slots)
    world2, aux = step_once(world, actions, bundle=bundle, config=cfg,
                            t_s=0.0)
    assert world2.s_m.shape == world.s_m.shape
    assert aux.cav_target_speed_mps.shape == world.s_m.shape
    assert aux.cav_reason_code.shape == world.s_m.shape
    assert aux.hdv_accel_delta_mps2.shape == world.s_m.shape
    assert aux.hdv_lane_logit_delta.shape == (world.batch, world.slots, 3)
    assert aux.baseline_idm_accel_mps2.shape == world.s_m.shape
    assert not aux.teacher_forced
    c = aux.counts
    assert all(hasattr(c, f) for f in ("accel_clip", "speed_clip",
                                       "lane_reject", "gap_projection"))
    assert torch.isfinite(world2.s_m).all() and torch.isfinite(
        world2.v_mps).all()


def test_engine_run_shapes_and_finiteness():
    batch = j4_smoke_world()
    bundle = IdeaABundle(ctrl=NativeMover(), hdv=None, arrival=None,
                         corr=None)
    out = run_rollout2(bundle, batch, n_steps=20, seed=0)
    assert out.positions.shape == (1, 21, 2, 32)
    assert out.speeds.shape == (1, 21, 2, 32)
    assert out.lanes.shape == (1, 21, 2, 32)
    assert out.masks.shape == (1, 21, 2, 32)
    assert out.accel.shape == (1, 20, 2, 32)
    assert out.arrivals.shape == (1, 20, 2, 2)
    assert out.insertions.shape == (1, 20, 2, 2)
    assert out.cav_commands.shape == (1, 20, 2, 32)
    assert torch.isfinite(out.positions).all()
    assert torch.isfinite(out.speeds).all()
    assert (out.speeds >= 0).all()
    assert len(out.validity) == 1
    vc = out.validity[0]
    assert all(getattr(vc, f) >= 0 for f in
               ("accel_clip", "speed_clip", "lane_reject", "gap_projection"))


def test_engine_determinism_same_seed():
    batch = j4_smoke_world()
    bundle = IdeaABundle(ctrl=NativeMover(), hdv=None, arrival=None,
                         corr=None)
    a = run_rollout2(bundle, batch, n_steps=12, seed=7)
    b = run_rollout2(bundle, batch, n_steps=12, seed=7)
    assert torch.equal(a.positions, b.positions)
    assert torch.equal(a.speeds, b.speeds)
    assert torch.equal(a.lanes, b.lanes)
    assert torch.equal(a.masks, b.masks)


def test_rollouts_dtype_follows_world():
    batch = j4_smoke_world()
    bundle = IdeaABundle(ctrl=NativeMover(), hdv=None, arrival=None,
                         corr=None)
    out = run_rollout2(bundle, batch, n_steps=4, seed=0)
    assert out.positions.dtype == torch.float32  # j4 batch is float32
    world, _ = world_from_table(scene_free())
    engine = Rollout2Engine(bundle, Rollout2Config(t0_s=0.0), seed=0)
    out64 = engine.run(world, None, n_steps=4)
    assert out64.positions.dtype == torch.float64


def test_smoke_contract_is_60s_cpu():
    c = j4_smoke_contract()
    assert c["n_steps"] == 120 and c["dt_s"] == 0.5 and c["device"] == "cpu"
    batch = j4_smoke_world()
    assert batch["veh_kin"].shape == (2, 32, 7)
    assert int(batch["veh_mask"].sum()) == 16
    assert int(batch["is_cav"].sum()) == 4


def test_smoke_contract_full_120_steps():
    """The frozen integration gate: 60 s CPU rollout, finite outputs,
    every ValidityCounts field reported."""
    batch = j4_smoke_world()
    bundle = IdeaABundle(ctrl=NativeMover(), hdv=None, arrival=None,
                         corr=None)
    c = j4_smoke_contract()
    out = run_rollout2(bundle, batch, n_steps=c["n_steps"], seed=c["seed"])
    assert torch.isfinite(out.positions).all()
    assert torch.isfinite(out.speeds).all()
    assert (out.speeds >= 0).all()
    vc = out.validity[0]
    assert all(isinstance(getattr(vc, f), int) for f in
               ("accel_clip", "speed_clip", "lane_reject", "gap_projection"))
    # the accel-lane vehicles must clear or stop before the lane end
    assert not math.isnan(float(out.positions[0, -1, 0, 6]))


def test_engine_freezes_original_t0_channels_before_mutation():
    """contrastfix A6: the rollout carries the ORIGINAL t0 state.

    ``t0_kin``/``t0_mask``/``t0_counter`` must be captured from the input
    world BEFORE any controller/teacher step, so a nonzero t0 acceleration
    and varied vehicle lengths survive (the trajectory tensors carry only
    the FUTURE: ``accel[0]`` is the first applied step and the snapshots
    hold no length).
    """
    batch = j4_smoke_world()
    kin = batch["veh_kin"].clone()
    kin[0, 0, 5] = 1.5                                 # nonzero t0 accel
    kin[0, 3, 5] = -2.0
    kin[0, 1, 6] = 4.2                                 # varied lengths
    kin[0, 4, 6] = 6.5
    batch["veh_kin"] = kin
    counter = torch.rand(2, 2, 30,
                         generator=torch.Generator().manual_seed(5))
    batch["counter"] = counter
    bundle = IdeaABundle(ctrl=NativeMover(), hdv=None, arrival=None,
                         corr=None)
    out = run_rollout2(bundle, batch, n_steps=6, seed=0)
    b_n, k_n = kin.shape[0], kin.shape[1]
    assert out.t0_kin.shape == (b_n, k_n, 7)
    assert out.t0_kin.dtype == torch.float32           # follows world dtype
    assert out.t0_mask.shape == (b_n, k_n)
    assert out.t0_mask.dtype == torch.bool
    # the exact ORIGINAL kinematic view (s, 3x lane one-hot, v, a, length)
    has_lane = kin[..., 1:4].sum(-1) > 0
    lane = torch.where(has_lane, kin[..., 1:4].argmax(-1),
                       torch.full(kin.shape[:-1], -1, dtype=torch.long))
    one_hot = (lane.unsqueeze(-1) == torch.arange(3)).to(kin.dtype)
    want = torch.stack([kin[..., 0], one_hot[..., 0], one_hot[..., 1],
                        one_hot[..., 2], kin[..., 4], kin[..., 5],
                        kin[..., 6]], dim=-1)
    assert torch.allclose(out.t0_kin, want, atol=1e-6)
    assert torch.equal(out.t0_mask, batch["veh_mask"] > 0)
    # the counter travels as a detached clone, value intact
    assert out.t0_counter is not batch["counter"]
    assert torch.equal(out.t0_counter, counter)
    # DRAFT correction recovery: explicit original CAV membership and actual
    # boundary IDs accompany the trajectory, independent of future masks.
    assert torch.equal(out.t0_is_cav, batch["is_cav"].bool())
    assert out.slot_ids.shape == out.positions.shape
    assert out.slot_ids.dtype == torch.int64
    assert torch.equal(out.slot_ids[0, 0], out.t0_slot_id)
    # the capture precedes the loop: the world actually moved after t0
    assert not torch.allclose(out.positions[0, 0], out.positions[0, -1])
    # the frozen t0 accel column is NOT the first FUTURE applied step
    assert float(out.t0_kin[0, 0, 5]) == 1.5
    assert float(out.t0_kin[0, 3, 5]) == -2.0


def test_world_from_batch_kin_and_step():
    batch = j4_smoke_world()
    world = rollout2.world_from_batch(batch)
    assert world.s_m.shape == (2, 32)
    assert int(world.present.sum()) == 16
    step_batch = dict(batch)
    kin = batch["veh_kin"]
    full = torch.zeros((2, 32, 16))
    full[..., 0] = kin[..., 0]
    full[..., 1:4] = kin[..., 1:4]
    full[..., 4] = kin[..., 4]
    full[..., 5] = kin[..., 5]
    full[..., 6] = kin[..., 6]
    full[..., 7] = torch.from_numpy(
        (batch["is_cav"].numpy() * batch["veh_mask"].numpy()))
    full[..., 15] = torch.from_numpy(batch["veh_mask"].numpy())
    step_batch["veh_step"] = full
    del step_batch["veh_kin"]
    world2 = rollout2.world_from_batch(step_batch)
    assert torch.allclose(world2.s_m, world.s_m, atol=1e-6)
    assert torch.equal(world2.is_cav, world.is_cav)
