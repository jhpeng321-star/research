"""P3 parity: the batched baseline == the scalar partA_v2g engine.

Covers the G1-shaped acceptance criteria on synthetic scenes (no run
data, no SUMO):

* baseline batched == scalar v2g ONE-STEP on synthetic scenes (exact);
* multi-step zero-residual / arrivals-off equivalence in the 'v2g'
  integrator mode (native CAVs and replayed commands);
* the 'd1' vs 'v2g' integrator difference matches the documented form;
* teacher-forced mode uses the recorded CAV states;
* K-sample determinism under a seed.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from rollout2 import (IdeaABundle, Rollout2Config, Rollout2State,
                      StrategyActions)
from rollout2.engine import Rollout2Engine, rollout2 as run_rollout2
from rollout2.smoke import j4_smoke_world
from rollout2.step import step_once
from rollout2.tests._parity_helpers import (
    ConstantDeltaNet, NativeMover, ScriptedMover, compare_engine,
    default_params, run_batched, run_scalar, scene_dense, scene_end_clamp,
    scene_free, scene_lc, scene_merge, world_from_table)

P = default_params()
DT = 0.5


# ----------------------------------------------------------- one-step exact
@pytest.mark.parametrize("vehicles", [scene_free(), scene_merge(),
                                      scene_lc(), scene_dense(),
                                      scene_end_clamp()])
def test_one_step_batched_equals_scalar_v2g(vehicles):
    t0 = 240.0
    res = run_batched(P.raw, vehicles, t0=t0, n_steps=1)
    w, names = res.world, res.names
    eng = run_scalar(P.raw, vehicles, t0=t0, n_steps=1)
    n, max_err = compare_engine(eng, w, names, step_j=1)
    assert n == len(names)
    assert max_err == 0.0, f"one-step mismatch {max_err}"


def test_one_step_multiple_batches_independent():
    """Batch row 0 and row 1 carry different scenes; each must match its
    own scalar reference (per-world isolation of the sequential rules)."""
    scenes = [scene_merge(), scene_dense()]
    world0, _ = world_from_table(scenes[0])
    world1, _ = world_from_table(scenes[1])
    b_n, k_n = 2, max(world0.slots, 16)
    from rollout2.state import WorldState

    def _pad(w):
        z = torch.zeros((1, k_n - w.slots), dtype=w.s_m.dtype)
        zi = torch.full((1, k_n - w.slots), -1, dtype=torch.int64)
        zb = torch.zeros((1, k_n - w.slots), dtype=torch.bool)
        return WorldState(
            s_m=torch.cat([w.s_m, z], 1), v_mps=torch.cat([w.v_mps, z], 1),
            a_mps2=torch.cat([w.a_mps2, z], 1),
            lane=torch.cat([w.lane, zi], 1),
            length_m=torch.cat([w.length_m, torch.full_like(z, 5.0)], 1),
            present=torch.cat([w.present, zb], 1),
            is_cav=torch.cat([w.is_cav, zb], 1),
            wall_group=torch.cat([w.wall_group,
                                  torch.full_like(zi, -1)], 1),
            slot_id=torch.cat([w.slot_id,
                               torch.arange(w.slots, k_n).unsqueeze(0)], 1))
    world = WorldState(
        s_m=torch.cat([_pad(world0).s_m, _pad(world1).s_m], 0),
        v_mps=torch.cat([_pad(world0).v_mps, _pad(world1).v_mps], 0),
        a_mps2=torch.cat([_pad(world0).a_mps2, _pad(world1).a_mps2], 0),
        lane=torch.cat([_pad(world0).lane, _pad(world1).lane], 0),
        length_m=torch.cat([_pad(world0).length_m, _pad(world1).length_m], 0),
        present=torch.cat([_pad(world0).present, _pad(world1).present], 0),
        is_cav=torch.cat([_pad(world0).is_cav, _pad(world1).is_cav], 0),
        wall_group=torch.full((b_n, k_n), -1, dtype=torch.int64),
        slot_id=torch.arange(k_n).expand(b_n, k_n).contiguous())
    actions = StrategyActions(
        action_code=torch.full((b_n, k_n), -1, dtype=torch.int64))
    bundle = IdeaABundle(ctrl=NativeMover(), hdv=None, arrival=None,
                         corr=None)
    cfg = Rollout2Config(params=P.raw, integrator="v2g", t0_s=10.0)
    cfg.state = Rollout2State(b_n, k_n)
    t0 = 10.0
    world2, _aux = step_once(world, actions, bundle=bundle, config=cfg,
                             t_s=t0)
    for b, scene in enumerate(scenes):
        eng = run_scalar(P.raw, scene, t0=t0, n_steps=1)
        snap = eng.snapshots[1]
        names = sorted(scene)
        for i, vid in enumerate(names):
            ref = snap[vid]
            got = (int(world2.lane[b, i]), float(world2.s_m[b, i]),
                   float(world2.v_mps[b, i]))
            assert got[0] == ref[0] and got[1] == ref[1] and got[2] == ref[2]


# ------------------------------------------------------- multi-step (G1)
@pytest.mark.parametrize("vehicles,steps", [
    (scene_free(), 120), (scene_merge(), 120), (scene_lc(), 60),
    (scene_dense(), 120), (scene_end_clamp(), 40),
])
def test_multistep_zero_residual_arrivals_off_v2g_mode(vehicles, steps):
    """The G1 shape: residual 0, arrivals OFF, 'v2g' integrator ->
    per vehicle per tick identical to the scalar engine."""
    t0 = 61.0
    res = run_batched(P.raw, vehicles, t0=t0, n_steps=steps)
    eng = run_scalar(P.raw, vehicles, t0=t0, n_steps=steps)
    pos = res.rollouts.positions[0, :, 0]
    spd = res.rollouts.speeds[0, :, 0]
    lns = res.rollouts.lanes[0, :, 0]
    max_err = 0.0
    for j in range(1, steps + 1):
        snap = eng.snapshots[j]
        for i, vid in enumerate(res.names):
            ref = snap.get(vid)
            if ref is None:
                continue      # exited the modelled road in the scalar run
            assert int(lns[j, i]) == ref[0], (j, vid, "lane")
            max_err = max(max_err, abs(float(pos[j, i]) - ref[1]),
                          abs(float(spd[j, i]) - ref[2]))
    assert max_err == 0.0, f"multi-step mismatch {max_err}"
    assert sum(res.state.collisions_by_step.values()) == eng.collisions
    assert res.state.anomalies.get("accel_end_clamped", 0) == \
        eng.anomalies.get("accel_end_clamped", 0)


def test_commands_replay_equivalence_v2g_mode():
    """Commanded CAVs: the same replayed commands into the scripted
    mover (P3) and the v2g control hook must give identical motion
    (execution-rule port check, engine.py:649-658)."""
    vehicles = scene_merge()
    slot_cav = 0                       # the front-most mainline slot
    vehicles[sorted(vehicles)[slot_cav]]["is_cav"] = True
    t0 = 100.0
    schedule = {}
    for k in range(24):                # 12 s of 1 s control ticks
        t = round(t0 + k, 6)
        schedule[t] = {slot_cav: 5.0 if k % 2 == 0 else 21.0}
    res = run_batched(P.raw, vehicles, t0=t0, n_steps=24, schedule=schedule)
    eng = run_scalar(P.raw, vehicles, t0=t0, n_steps=24, schedule=schedule)
    pos = res.rollouts.positions[0, :, 0]
    spd = res.rollouts.speeds[0, :, 0]
    lns = res.rollouts.lanes[0, :, 0]
    max_err = 0.0
    for j in range(1, 25):
        snap = eng.snapshots[j]
        for i, vid in enumerate(res.names):
            ref = snap.get(vid)
            if ref is None:
                continue
            assert int(lns[j, i]) == ref[0]
            max_err = max(max_err, abs(float(pos[j, i]) - ref[1]),
                          abs(float(spd[j, i]) - ref[2]))
    assert max_err == 0.0, f"commanded mismatch {max_err}"


# ------------------------------------------------- integrator difference
def test_d1_mode_differs_by_the_documented_rule():
    """With everything else equal, s_d1 - s_v2g = 0.5*(v - v_new)*dt per
    free step (INTERFACES_V2 Mismatch 5 documentation); on a single
    free-road vehicle the speeds stay identical."""
    vehicles = {"veh000": dict(lane=1, s=2350.0, v=29.0)}
    t0 = 5.0
    res_d1 = run_batched(P.raw, vehicles, t0=t0, n_steps=1, integrator="d1")
    res_v2g = run_batched(P.raw, vehicles, t0=t0, n_steps=1,
                          integrator="v2g")
    w_d1, w_v2g = res_d1.world, res_v2g.world
    assert not torch.equal(w_d1.s_m, w_v2g.s_m)
    assert torch.equal(w_d1.v_mps, w_v2g.v_mps)   # same v_new ...
    v_new = float(w_d1.v_mps[0, 0])
    v_old = 29.0
    # ... and the documented per-step position difference
    expected = 0.5 * (v_old - v_new) * DT
    got = float(w_d1.s_m[0, 0] - w_v2g.s_m[0, 0])
    assert got == pytest.approx(expected, abs=1e-12)
    assert got < 0.0     # accelerating: the trapezoid travels shorter


def test_d1_is_the_default_integrator():
    cfg = Rollout2Config()
    assert cfg.integrator == "d1" and cfg.gap_projection
    cfg2 = Rollout2Config(integrator="v2g")
    assert not cfg2.gap_projection


# --------------------------------------------------------- residual path
def test_residual_accel_changes_hdvs_not_cavs():
    vehicles = scene_free()
    names = sorted(vehicles)
    cav_vid = names[-1]              # the front-most (leader-free) slot
    vehicles[cav_vid]["is_cav"] = True
    cav_slot = names.index(cav_vid)
    t0 = 20.0
    net = ConstantDeltaNet(accel_delta=0.3)
    w_res, names, _st, aux = None, None, None, None
    res = run_batched(P.raw, vehicles, t0=t0, n_steps=2, residual_net=net)
    w_res, names, aux = res.world, res.names, res.aux_last
    res_base = run_batched(P.raw, vehicles, t0=t0, n_steps=2)
    w_base = res_base.world
    assert net.seen_shapes and net.seen_shapes[0][-1] == 7  # kinematics only
    assert torch.any(w_res.s_m != w_base.s_m)      # HDV motion differs
    # CAV slot: residual must not leak (same baseline motion)
    assert float(w_res.s_m[0, cav_slot]) == float(w_base.s_m[0, cav_slot])
    assert float(w_res.v_mps[0, cav_slot]) == \
        float(w_base.v_mps[0, cav_slot])
    for i, vid in enumerate(names):
        expected = 0.0 if vid == cav_vid else 0.3
        assert float(aux.hdv_accel_delta_mps2[0, i]) == \
            pytest.approx(expected, abs=1e-6)


def test_lane_delta_only_shifts_incentive_margin():
    """A large positive left-margin delta may pull a merge earlier but
    can NEVER override the safety gates (a hard-overlap merge stays
    rejected)."""
    from rollout2 import baseline as B

    vehicles = scene_merge()
    t0 = 0.0
    core = B.V2GCore(P)
    world, names = world_from_table(vehicles)
    rows = np.arange(len(names))
    wv = B.WorldRows(P.off, world.lane[0].numpy().copy(),
                     world.s_m[0].numpy().copy(),
                     world.v_mps[0].numpy().copy(),
                     world.length_m[0].numpy().copy(), rows)
    big = np.full(len(names), 1e3)
    out = core.lane_change_stage(wv, t_change=t0, delta_left=big,
                                 enforce_legality=True)
    # margin shift may fire merges only when the safe-gap tests pass
    ref = core.lane_change_stage(
        B.WorldRows(P.off, world.lane[0].numpy().copy(),
                    world.s_m[0].numpy().copy(),
                    world.v_mps[0].numpy().copy(),
                    world.length_m[0].numpy().copy(), rows), t_change=t0)
    assert out["lane_reject"] == 0
    assert out["merge"] >= ref["merge"]


# ------------------------------------------------------- teacher forcing
def test_teacher_forced_cavs_use_recorded_states():
    vehicles = scene_free()
    for i, vid in enumerate(sorted(vehicles)):
        if i < 2:
            vehicles[vid]["is_cav"] = True
    world, names = world_from_table(vehicles)
    b_n, k_n = world.batch, world.slots
    n_steps = 4
    teacher = torch.zeros((n_steps + 1, b_n, k_n, 4), dtype=torch.float64)
    teacher[..., 0] = 3000.0        # recorded CAV position (far ahead)
    teacher[..., 1] = 1             # recorded lane ...
    teacher[:, :, 0, 1] = 0         # ... slot 0 lane 0, slot 1 lane 1
    teacher[..., 2] = 30.0          # recorded speed
    teacher[..., 3] = 0.0
    actions = StrategyActions(
        action_code=torch.where(world.is_cav,
                                torch.zeros_like(world.wall_group),
                                torch.full_like(world.wall_group, -1)))
    bundle = IdeaABundle(ctrl=NativeMover(), hdv=None, arrival=None,
                         corr=None)
    cfg = Rollout2Config(params=P.raw, t0_s=0.0)
    cfg.state = Rollout2State(b_n, k_n)
    w, aux = step_once(world, actions, bundle=bundle, config=cfg, t_s=0.0,
                       teacher_state=None)
    s_unforced = w.s_m.clone()
    cfg2 = Rollout2Config(params=P.raw, t0_s=0.0)
    cfg2.state = Rollout2State(b_n, k_n)
    w2, aux2 = step_once(world, actions, bundle=bundle, config=cfg2,
                         t_s=0.0, teacher_state=_teacher_ws(teacher[0], world))
    assert aux2.teacher_forced and not aux.teacher_forced
    # the forced CAVs were integrated FROM the recorded state (s=3000,
    # v=30), not from the scene state (s~2350); the D1 identity
    # s_new = s + 0.5*(v + v_new)*dt holds on the recorded start
    for slot in (0, 1):
        v_new = float(w2.v_mps[0, slot])
        assert float(w2.s_m[0, slot]) == pytest.approx(
            3000.0 + 0.5 * (30.0 + v_new) * DT, abs=1e-9)
    assert float(w2.s_m[0, 0]) == pytest.approx(float(w2.s_m[0, 1]))
    # the unforced run kept the original CAV state (scene s ~ 2350)
    assert float(s_unforced[0, 0]) < 2400.0
    assert not torch.equal(w2.s_m, s_unforced)


def _teacher_ws(row, world):
    from rollout2.state import WorldState

    return WorldState(
        s_m=row[..., 0].to(torch.float64), v_mps=row[..., 2].to(torch.float64),
        a_mps2=row[..., 3].to(torch.float64), lane=row[..., 1].long(),
        length_m=world.length_m.clone(), present=world.present.clone(),
        is_cav=world.is_cav.clone(), wall_group=world.wall_group.clone(),
        slot_id=world.slot_id.clone())


def test_engine_teacher_forcing_history():
    vehicles = scene_free()
    for i, vid in enumerate(sorted(vehicles)):
        if i == 0:
            vehicles[vid]["is_cav"] = True
    world, _names = world_from_table(vehicles)
    n_steps = 3
    teacher = torch.zeros((n_steps + 1, 1, world.slots, 4),
                          dtype=torch.float64)
    teacher[..., 0] = 3000.0
    teacher[..., 1] = 1
    teacher[..., 2] = 30.0
    # command the CAV to exactly its recorded speed: the execution rule
    # then keeps v = 30 and the forced position advances exactly v*dt
    mover = ScriptedMover({round(float(k), 6): {0: 30.0}
                           for k in range(n_steps)})
    bundle = IdeaABundle(ctrl=mover, hdv=None, arrival=None, corr=None)
    engine = Rollout2Engine(bundle, Rollout2Config(params=P.raw, t0_s=0.0),
                            seed=0)
    out = engine.run(world, None, n_steps=n_steps, teacher=teacher)
    # the CAV is RE-FORCED to the recorded state (s=3000, v=30) before
    # every step, so each executed step advances exactly v*dt from the
    # recorded position - the recorded motion, not the model's, is seen
    # by the HDVs
    for i in range(n_steps):
        assert float(out.positions[0, i + 1, 0, 0]) == pytest.approx(
            3015.0, abs=1e-9)


# ------------------------------------------------------- determinism
def test_k_sample_determinism_under_seed():
    batch = j4_smoke_world()
    bundle = IdeaABundle(ctrl=NativeMover(), hdv=None, arrival=None,
                         corr=None)
    a = run_rollout2(bundle, batch, n_steps=30, n_samples=3, seed=42)
    b = run_rollout2(bundle, batch, n_steps=30, n_samples=3, seed=42)
    for key in ("positions", "speeds", "lanes", "masks", "accel"):
        assert torch.equal(getattr(a, key), getattr(b, key))
    assert a.positions.shape[0] == 3


def test_exits_and_freeze():
    """A vehicle beyond the lane length leaves the modelled road: its
    mask drops to False and its state freezes."""
    vehicles = _table_exit()
    t0 = 0.0
    res = run_batched(P.raw, vehicles, t0=t0, n_steps=6)
    w, names = res.world, res.names
    eng = run_scalar(P.raw, vehicles, t0=t0, n_steps=6)
    assert names[0] in eng.exited      # scalar engine saw the exit
    assert not bool(w.present[0, 0])   # batched path froze the slot
    assert sum(res.state.collisions_by_step.values()) == eng.collisions


def _table_exit():
    # lane 1 ends at 3197.55 m: a 30 m/s vehicle crosses it within steps
    return {"veh000": dict(lane=1, s=3195.0, v=30.0),
            "veh001": dict(lane=1, s=3100.0, v=30.0)}


# ----------------------------------------------------- engine.rollout2()
def test_rollout2_wrapper_batch_path():
    batch = j4_smoke_world()
    bundle = IdeaABundle(ctrl=NativeMover(), hdv=None, arrival=None,
                         corr=None)
    batch["integrator"] = "v2g"
    out = run_rollout2(bundle, batch, n_steps=10, seed=0)
    assert out.positions.shape == (1, 11, 2, 32)
    assert torch.isfinite(out.positions).all()


def test_v2g_mode_long_run_smoke_world():
    """120-step 'v2g'-mode run on the smoke world: finite and
    deterministic (the P3 side of the frozen integration fixture)."""
    batch = j4_smoke_world()
    batch["integrator"] = "v2g"
    bundle = IdeaABundle(ctrl=NativeMover(), hdv=None, arrival=None,
                         corr=None)
    a = run_rollout2(bundle, batch, n_steps=120, seed=0)
    b = run_rollout2(bundle, batch, n_steps=120, seed=0)
    assert torch.equal(a.positions, b.positions)
    assert torch.isfinite(a.positions).all()
    assert torch.isfinite(a.speeds).all()
