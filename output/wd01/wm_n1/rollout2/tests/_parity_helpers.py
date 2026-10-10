"""Shared helpers for the rollout2 parity/contract tests.

The v2g engine (partA_v2g) is imported READ-ONLY and used as the scalar
reference; the batched P3 path is compared against it exactly.
"""
from __future__ import annotations

import torch

from rollout2 import baseline
from rollout2.engine import IdeaABundle
from rollout2.state import Rollout2Config, Rollout2State, StrategyActions
from rollout2.step import step_once

DT = 0.5
# slot width is a format parameter (amendment 5): 128 = v2, 192 = v3; the
# synthetic defaults here keep the v2 width while every real rollout takes K
# from its tensors (WorldState.slots).
K_SLOTS = 128


# ------------------------------------------------------------- fake movers
class NativeMover:
    """Fake mover mirroring the native contract: NaN targets, reason 1."""

    def __init__(self) -> None:
        self.calls = 0

    def init_state(self, batch, slots=K_SLOTS, *, device=None):
        return {"batch": batch, "slots": slots}

    def step(self, state, inp, actions, *, wall_speed_mps=None):
        from ctrl.mover import CavStepOutput

        self.calls += 1
        target = torch.full_like(inp.s_m, float("nan"), dtype=torch.float64)
        reason = torch.ones_like(inp.lane)
        return CavStepOutput(
            target_speed_mps=target,
            accel_cmd_mps2=torch.full_like(inp.s_m, float("nan"),
                                           dtype=torch.float64),
            reason_code=reason, state=state)


class ScriptedMover:
    """Replays per-control-tick, per-slot target speeds (G1-style).

    ``schedule``: {control_t_s: {slot: target_mps}}.  Slots not listed in
    a tick keep their previous target (the hold rule); a slot never
    listed stays NaN (no command).
    """

    def __init__(self, schedule) -> None:
        self.schedule = dict(schedule)
        self.calls = 0
        self._last = {}

    def init_state(self, batch, slots=K_SLOTS, *, device=None):
        return {"batch": batch, "slots": slots}

    def step(self, state, inp, actions, *, wall_speed_mps=None):
        from ctrl.mover import CavStepOutput

        self.calls += 1
        t = round(float(inp.control_t_s if inp.control_t_s is not None
                        else inp.t_s), 6)
        for slot, val in self.schedule.get(t, {}).items():
            self._last[slot] = float(val)
        target = torch.full_like(inp.s_m, float("nan"), dtype=torch.float64)
        for slot, val in self._last.items():
            target[:, slot] = val
        reason = torch.where(torch.isfinite(target),
                             torch.zeros_like(inp.lane),
                             torch.ones_like(inp.lane))
        return CavStepOutput(
            target_speed_mps=target,
            accel_cmd_mps2=torch.full_like(inp.s_m, float("nan"),
                                           dtype=torch.float64),
            reason_code=reason, state=state)


class ConstantDeltaNet:
    """Fake P2 residual: constant accel delta on HDV slots."""

    def __init__(self, accel_delta: float = 0.3,
                 lane_delta: float = 0.0) -> None:
        self.accel_delta = float(accel_delta)
        self.lane_delta = float(lane_delta)
        self.seen_shapes = []

    def __call__(self, kin, kin_mask, *, cav_flag=None):
        import torch as _torch

        self.seen_shapes.append(tuple(kin.shape))
        b_n, k_n = kin.shape[:2]
        return dict(
            accel_delta_mps2=_torch.full((b_n, k_n), self.accel_delta),
            lane_logit_delta=_torch.full((b_n, k_n, 3), self.lane_delta))


# ----------------------------------------------------------- scene builders
def _table(rows):
    """rows: list of (lane, s, v[, a, length]) -> {vid: rec} sorted ids."""
    out = {}
    for i, row in enumerate(rows):
        lane, s, v = row[0], row[1], row[2]
        a = row[3] if len(row) > 3 else 0.0
        length = row[4] if len(row) > 4 else 5.0
        out[f"veh{i:03d}"] = dict(lane=lane, s=s, v=v, a=a, length_m=length)
    return out


def scene_free():
    """Free mainline platoon (no leader for the front vehicle)."""
    return _table([(1, 2350.0, 29.0), (1, 2382.0, 29.0), (1, 2416.0, 28.6)])


def scene_merge():
    """Accel-lane vehicle in the merge zone behind mainline traffic."""
    return _table([(1, 2445.0, 29.0), (1, 2480.0, 28.8),
                   (2, 400.0, 22.0), (2, 436.0, 20.5)])


def scene_lc():
    """Discretionary pressure: slow leader on lane 0, free lane 1."""
    return _table([(0, 2380.0, 20.0), (0, 2400.0, 28.0),
                   (1, 2450.0, 29.0)])


def scene_dense():
    """Mixed lanes, tight gaps (IDM interaction + collision regime)."""
    return _table([(1, 2350.0, 27.0), (1, 2359.0, 27.0), (1, 2370.0, 26.5),
                   (0, 2360.0, 27.5), (0, 2371.5, 27.5),
                   (2, 380.0, 21.0), (2, 425.0, 19.0), (2, 450.5, 9.0)])


def scene_end_clamp():
    """Accel-lane vehicle at the zone end (lane-end clamp regime)."""
    return _table([(2, 453.5, 8.0), (1, 2500.0, 30.0)])


def world_from_table(vehicles, *, slots=16, dtype=torch.float64):
    """WorldState with slot order == sorted-vid order (the v2g name
    order), float64 by default for exact parity runs."""
    names = sorted(vehicles)
    b_n, k_n = 1, int(slots)
    s = torch.zeros((b_n, k_n), dtype=dtype)
    v = torch.zeros((b_n, k_n), dtype=dtype)
    a = torch.zeros((b_n, k_n), dtype=dtype)
    lane = torch.full((b_n, k_n), -1, dtype=torch.int64)
    length = torch.full((b_n, k_n), 5.0, dtype=dtype)
    present = torch.zeros((b_n, k_n), dtype=torch.bool)
    is_cav = torch.zeros((b_n, k_n), dtype=torch.bool)
    slot_id = torch.arange(k_n).expand(b_n, k_n).contiguous()
    for i, vid in enumerate(names):
        rec = vehicles[vid]
        s[0, i] = rec["s"]
        v[0, i] = rec["v"]
        a[0, i] = rec.get("a", 0.0)
        lane[0, i] = rec["lane"]
        length[0, i] = rec.get("length_m", 5.0)
        present[0, i] = True
        is_cav[0, i] = bool(rec.get("is_cav", False))
    from rollout2.state import WorldState

    return WorldState(s_m=s, v_mps=v, a_mps2=a, lane=lane, length_m=length,
                      present=present, is_cav=is_cav,
                      wall_group=torch.full((b_n, k_n), -1,
                                            dtype=torch.int64),
                      slot_id=slot_id), names


def run_batched(params, vehicles, *, t0, n_steps, schedule=None,
                integrator="v2g", lane_changes_enabled=True,
                residual_net=None, mover=None, slots=16):
    """Run the P3 batched path with full history recording.

    Returns a namespace with ``rollouts`` (Rollouts, [1, n_steps+1, B, K]
    histories), ``world`` (the final WorldState), ``names``, ``state``
    (the Rollout2State) and ``aux_last`` (the last StepAux)."""
    from types import SimpleNamespace

    import numpy as np

    from model.rollout import ValidityCounts


    world, names = world_from_table(vehicles, slots=slots)
    actions = StrategyActions(
        action_code=torch.where(
            world.is_cav, torch.zeros_like(world.wall_group),
            torch.full_like(world.wall_group, -1)))
    if mover is None and schedule:
        mover = ScriptedMover(schedule)
    bundle = IdeaABundle(
        ctrl=mover if mover is not None else NativeMover(),
        hdv=residual_net, arrival=None, corr=None)
    cfg = Rollout2Config(
        params=params, integrator=integrator, residual_enabled=residual_net
        is not None, lane_changes_enabled=lane_changes_enabled, t0_s=t0)
    st = Rollout2State(world.batch, world.slots)
    cfg.state = st
    n_steps = int(n_steps)
    b_n, k_n = world.batch, world.slots
    pos = np.zeros((n_steps + 1, b_n, k_n))
    spd = np.zeros((n_steps + 1, b_n, k_n))
    lns = np.zeros((n_steps + 1, b_n, k_n), dtype=np.int64)
    msk = np.zeros((n_steps + 1, b_n, k_n), dtype=bool)
    acc = np.zeros((n_steps, b_n, k_n))
    pos[0] = world.s_m.numpy()
    spd[0] = world.v_mps.numpy()
    lns[0] = world.lane.numpy()
    msk[0] = world.present.numpy()
    totals = ValidityCounts()
    aux_last = None
    w = world
    for i in range(n_steps):
        w, aux_last = step_once(world=w, actions=actions, bundle=bundle,
                                config=cfg, t_s=t0 + i * DT)
        totals.add(aux_last.counts)
        pos[i + 1] = w.s_m.numpy()
        spd[i + 1] = w.v_mps.numpy()
        lns[i + 1] = w.lane.numpy()
        msk[i + 1] = w.present.numpy()
        acc[i] = w.a_mps2.numpy()
    from rollout2.state import Rollouts

    rollouts = Rollouts(
        positions=torch.from_numpy(pos).unsqueeze(0),
        speeds=torch.from_numpy(spd).unsqueeze(0),
        lanes=torch.from_numpy(lns).unsqueeze(0),
        masks=torch.from_numpy(msk).unsqueeze(0),
        accel=torch.from_numpy(acc).unsqueeze(0),
        arrivals=torch.zeros((1, n_steps, b_n, 2), dtype=torch.int64),
        insertions=torch.zeros((1, n_steps, b_n, 2), dtype=torch.int64),
        validity=[totals],
        cav_commands=torch.zeros((1, n_steps, b_n, k_n)),
        collisions_by_step=[dict(st.collisions_by_step)],
        anomalies=[dict(st.anomalies)])
    return SimpleNamespace(rollouts=rollouts, world=w, names=names,
                           state=st, aux_last=aux_last)


def run_scalar(params, vehicles, *, t0, n_steps, schedule=None,
               lane_changes_enabled=True):
    """Run the READ-ONLY scalar partA_v2g engine; returns the engine."""
    from partA_v2g.engine import PartAV2GEngine

    hook = None
    if schedule:
        names = sorted(vehicles)

        def hook(t_now, _eng, _names=names, _sched=schedule):
            cmds = _sched.get(round(float(t_now), 6), {})
            return {_names[slot]: val for slot, val in cmds.items()}

    eng = PartAV2GEngine(params, t0=t0, vehicles=vehicles, control_hook=hook,
                         lane_changes_enabled=lane_changes_enabled)
    eng.run(int(n_steps))
    return eng


def compare_engine(eng, world, names, *, step_j=None, tol=1e-9):
    """Compare the scalar engine snapshot at physics step ``step_j``
    (default: the last) with the batched world; returns (n, max_err)."""
    if step_j is None:
        step_j = eng.step_i
    snap = eng.snapshots[step_j]
    n = 0
    max_err = 0.0
    for i, vid in enumerate(names):
        ref = snap.get(vid)
        got = (int(world.lane[0, i]), float(world.s_m[0, i]),
               float(world.v_mps[0, i]))
        if ref is None:
            assert got is not None
            continue
        assert got[0] == ref[0], f"lane mismatch {vid}: {got} vs {ref}"
        max_err = max(max_err, abs(got[1] - ref[1]), abs(got[2] - ref[2]))
        n += 1
    return n, max_err


def default_params():
    return baseline.Params(baseline.load_params())
