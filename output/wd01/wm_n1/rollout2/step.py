"""P3 one Idea A rollout step (0.5 s): (a) CAV ctrl, (b) baseline HDV +
residual, (c) arrivals, (d) validity/D1.

Order is fixed.  CAVs are moved by the REAL controllers (P1); HDVs by the
Part A v2g IDM/MOBIL/merge baseline plus the P2 residual; arrivals by the
P4 head; everything then passes the validity layer with the D1 integration.

Teacher-forced CAV mode (training only) replaces the controller-produced CAV
trajectory by the recorded one, so the HDV residual learns the response to the
executed motion.  HDV targets always come from data.

Baseline faithfulness (INTERFACES_V2 Mismatch 5): with
``residual_enabled=False`` and ``arrivals_enabled=False`` the stage order,
decision rules and execution rule are exactly ``partA_v2g.engine.step``;
with ``Rollout2Config.integrator="v2g"`` the position rule is the v2g
ballistic one and the gap repair is skipped, so the G1 comparison is like
for like.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from ctrl.config import CtrlConfig
from ctrl.mover import CavStepInput, execution_speed

from . import baseline
from .state import Rollout2Config, Rollout2State, StrategyActions, WorldState
from .validity import (integrate_ballistic, integrate_d1_vnew,
                       project_same_lane_gaps)
from model.rollout import ValidityCounts

_NOMINAL_LENGTH_M = 5.0
_CTRL_TICK_EPS = 1e-9


@dataclass
class StepAux:
    """Per-step telemetry used by losses and the correction head."""

    cav_target_speed_mps: Tensor      # [B, K], nan = no command
    cav_reason_code: Tensor           # [B, K] int64
    hdv_accel_delta_mps2: Tensor      # [B, K]
    hdv_lane_logit_delta: Tensor      # [B, K, 3]
    baseline_idm_accel_mps2: Tensor   # [B, K]
    counts: object                    # model.rollout.ValidityCounts
    teacher_forced: bool = False


def _f64(t: Tensor) -> Tensor:
    return t.detach().to(torch.float64)


def _kin_features(s: Tensor, lane: Tensor, v: Tensor, a: Tensor,
                  length: Tensor) -> Tensor:
    """``[B, K, 7]`` kinematics-only HDV features (KIN_FIELDS order; the
    CAV flag is never an input, INTERFACES_V2 Mismatch 3)."""
    b_n, k_n = s.shape
    kin = torch.zeros((b_n, k_n, 7), dtype=torch.float64, device=s.device)
    kin[..., 0] = s
    kin[..., 1] = (lane == 0).to(torch.float64)
    kin[..., 2] = (lane == 1).to(torch.float64)
    kin[..., 3] = (lane == 2).to(torch.float64)
    kin[..., 4] = v
    kin[..., 5] = a
    kin[..., 6] = length
    return kin


def _segment_view(params: baseline.Params, lane_np: np.ndarray,
                  s_np: np.ndarray) -> tuple:
    """Per-slot containing-segment ``(lo, hi, limit)`` from the frozen
    geometry (the segment-local sensing convention of partA_ctrl)."""
    lane_np = np.asarray(lane_np)
    s_np = np.asarray(s_np, dtype=np.float64)
    lo = np.zeros(lane_np.shape, dtype=np.float64)
    llen_row = params.llen[np.clip(lane_np, 0, len(params.llen) - 1)]
    hi = np.where(lane_np >= 0, llen_row, 0.0).astype(np.float64)
    limit = np.full(lane_np.shape, np.nan, dtype=np.float64)
    for li in baseline.LANES:
        m = lane_np == li
        if not m.any():
            continue
        starts = params._seg_starts[li]
        lims = params._seg_lims[li]
        idx = np.searchsorted(starts, s_np[m], side="right") - 1
        idx = np.clip(idx, 0, len(lims) - 1)
        lo[m] = starts[idx]
        hi[m] = np.where(idx + 1 < len(starts),
                         starts[np.minimum(idx + 1, len(starts) - 1)],
                         params.llen[li])
        limit[m] = lims[idx]
    return lo, hi, limit


def _insert_arrivals(s_np, v_np, a_np, length_np, lane_np, present_np,
                     is_cav_np, slot_np, b: int, counts_row, speeds_row, *,
                     max_insert: int, next_slot_id) -> tuple:
    """Mirror of ``model.rollout._insert_arrivals`` on the world arrays.

    Entry lanes: entrance 0 -> main_right (1), entrance 1 -> accel (2).
    New vehicles enter behind the current lane occupancy so the rear
    bumper gap stays >= 0; they are generic (is_cav=0) tokens.  Returns
    ``(inserted [2], slots)`` so the caller can set the inserted slots'
    ``v_new`` (their acceleration this tick is 0).
    """
    inserted = np.zeros(2, dtype=np.int64)
    slots = []
    k_n = present_np.shape[1]
    for e in range(2):
        entry_lane = 1 if e == 0 else 2
        n = int(counts_row[e])
        for _ in range(min(n, max_insert)):
            free = np.where(~present_np[b])[0]
            if free.size == 0:
                break
            slot = int(free[0])
            occ = [float(s_np[b, j]) for j in range(k_n)
                   if bool(present_np[b, j]) and int(lane_np[b, j])
                   == entry_lane]
            pos = (min(occ) - _NOMINAL_LENGTH_M if occ
                   else 0.0 - _NOMINAL_LENGTH_M)
            spd = max(0.0, float(speeds_row[e]))
            s_np[b, slot] = pos
            v_np[b, slot] = spd
            a_np[b, slot] = 0.0
            length_np[b, slot] = _NOMINAL_LENGTH_M
            lane_np[b, slot] = entry_lane
            present_np[b, slot] = True
            is_cav_np[b, slot] = False
            slot_np[b, slot] = int(next_slot_id[b])
            next_slot_id[b] += 1
            inserted[e] += 1
            slots.append(slot)
    return inserted, slots


def step_once(world: WorldState, actions: StrategyActions, *, bundle,
              config: Optional[Rollout2Config] = None, t_s: float,
              control_t_s: Optional[float] = None, rng=None,
              teacher_state: Optional[WorldState] = None,
              ) -> tuple[WorldState, StepAux]:
    """One batched 0.5 s Idea A step.

    Parameters
    ----------
    world:
        The step-boundary world (any float dtype; internal math is
        float64 and the returned world is cast back to ``world``'s
        dtype).
    actions:
        :class:`~rollout2.state.StrategyActions` (per-slot CAV action
        codes + wall targets; consumed by the P1 mover only).
    bundle:
        :class:`~rollout2.engine.IdeaABundle`; ``ctrl``/``hdv``/
        ``arrival`` may each be ``None`` (no control stage / zero
        residual / arrivals off).
    config:
        :class:`~rollout2.state.Rollout2Config`; ``config.state`` MUST
        be provided for persistent rollouts (control-command hold,
        lane-change cooldowns, counters); a fresh throwaway state is
        created when omitted.
    t_s:
        The physics-step start time (``t0 + j*dt``); a control tick
        fires when ``t_s`` is an integer second
        (partA_v2g/engine.py:568-572).
    teacher_state:
        Training-only recorded world; CAV slots are replaced by it
        before the step so HDVs react to the executed CAV motion.
    """
    cfg = config if config is not None else Rollout2Config()
    if bundle is None:
        raise ValueError("bundle is required (IdeaABundle)")
    merge_cal_on = bool(getattr(cfg, "merge_cal", False))
    merge_params = None
    if merge_cal_on:
        from .merge_cal import MergeCalCore, MergeCalParams

        raw = getattr(cfg, "merge_cal_params", None)
        merge_params = (raw if isinstance(raw, MergeCalParams)
                        else MergeCalParams(**(raw or {})) if isinstance(raw, dict)
                        else MergeCalParams.load())
    use_frame = bool(getattr(cfg, "sumo_fidelity", False)) or merge_cal_on
    speed_factor = getattr(cfg, "sumo_speed_factor", None)
    if (use_frame and speed_factor is None and merge_params is not None
            and merge_params.speed_factor is not None):
        speed_factor = merge_params.speed_factor
    if use_frame:
        params = baseline.Params(baseline.model_frame_params(
            cfg.params, sumo_idm=bool(getattr(cfg, "sumo_idm", True)),
            speed_factor=speed_factor))
    else:
        params = baseline.Params(cfg.params)
    if merge_cal_on:
        core = MergeCalCore(params, merge_params,
                            lane_changes_enabled=cfg.lane_changes_enabled)
    else:
        core = baseline.V2GCore(
            params, lane_changes_enabled=cfg.lane_changes_enabled)
    dt = params.dt
    b_n, k_n = world.batch, world.slots
    st = cfg.state
    if st is None:
        st = Rollout2State(b_n, k_n)
    counts = ValidityCounts()
    out_dtype = world.s_m.dtype

    # ---- float64 working copies (numpy mirrors share memory) ----------
    s_np = _f64(world.s_m).numpy().copy()
    v_np = _f64(world.v_mps).numpy().copy()
    a_np = _f64(world.a_mps2).numpy().copy()
    length_np = _f64(world.length_m).numpy().copy()
    present_np = world.present.detach().to(torch.bool).numpy().copy()
    is_cav_np = world.is_cav.detach().to(torch.bool).numpy().copy()
    lane_np = world.lane.detach().to(torch.int64).numpy().copy()
    slot_np = world.slot_id.detach().to(torch.int64).numpy().copy()
    wall_group = world.wall_group.detach().to(torch.int64).clone()
    # INC1 fixed obstacle: pin one or more slots at their t0 position with
    # v = 0, a = 0.  They are NOT moved by the HDV residual or the baseline
    # and are never dropped, but the baseline still sees them as stopped
    # same-lane leaders (they stay in the compact WorldRows).
    obst_np = np.zeros_like(present_np)
    if getattr(world, "obstacle", None) is not None:
        obst_np = world.obstacle.detach().to(torch.bool).numpy().copy()
        obst_np &= present_np
        # an obstacle is always a stopped vehicle
        v_np[obst_np] = 0.0
        a_np[obst_np] = 0.0

    # teacher forcing (training only): replace the CAV slots by the
    # recorded state BEFORE any decision, so HDVs see executed motion.
    teacher_forced = teacher_state is not None
    if teacher_forced:
        ts = _f64(teacher_state.s_m).numpy()
        tv = _f64(teacher_state.v_mps).numpy()
        ta = _f64(teacher_state.a_mps2).numpy()
        tlen = _f64(teacher_state.length_m).numpy()
        tlane = teacher_state.lane.detach().to(torch.int64).numpy()
        tpres = teacher_state.present.detach().to(torch.bool).numpy()
        sel = is_cav_np & np.isfinite(ts) & tpres
        s_np[sel] = ts[sel]
        v_np[sel] = tv[sel]
        a_np[sel] = ta[sel]
        length_np[sel] = tlen[sel]
        lane_np[sel] = tlane[sel]
        present_np[sel] = True

    t = float(t_s)

    # ---- (a) CAV control stage (1 s cadence; command holds 2 ticks) ---
    ctrl = getattr(bundle, "ctrl", None)
    if ctrl is not None and abs(t - round(t)) < _CTRL_TICK_EPS:
        ctrl_state = st.ctrl_state
        if ctrl_state is None:
            ctrl_state = ctrl.init_state(b_n, k_n)
            st.ctrl_state = ctrl_state
        lo, hi, lim = _segment_view(params, lane_np, s_np)
        inp = CavStepInput(
            s_m=torch.from_numpy(s_np), v_mps=torch.from_numpy(v_np),
            lane=torch.from_numpy(lane_np),
            length_m=torch.from_numpy(length_np),
            present=torch.from_numpy(present_np),
            is_cav=torch.from_numpy(is_cav_np),
            wall_group=wall_group,
            segment_lo_m=torch.from_numpy(lo),
            segment_hi_m=torch.from_numpy(hi),
            segment_limit_mps=torch.from_numpy(lim),
            t_s=t,
            control_t_s=(round(t, 6) if control_t_s is None
                         else float(control_t_s)))
        out = ctrl.step(ctrl_state, inp, actions.action_code,
                        wall_speed_mps=actions.wall_speed_mps)
        st.held_target_mps = _f64(out.target_speed_mps)
        st.held_reason = out.reason_code.detach().to(torch.int64)
        st.cmd_t_s = inp.control_t_s

    # ---- (b) HDV residual + baseline ----------------------------------
    a_delta = np.zeros((b_n, k_n), dtype=np.float64)
    lane_delta = np.zeros((b_n, k_n, 3), dtype=np.float64)
    hdv_net = getattr(bundle, "hdv", None)
    if cfg.residual_enabled and hdv_net is not None:
        kin = _kin_features(torch.from_numpy(s_np),
                            torch.from_numpy(lane_np),
                            torch.from_numpy(v_np),
                            torch.from_numpy(a_np),
                            torch.from_numpy(length_np))
        res = hdv_net(kin, torch.from_numpy(present_np))
        a_delta = np.asarray(
            _f64(res["accel_delta_mps2"]).reshape(b_n, k_n).numpy()).copy()
        lane_delta = np.asarray(
            _f64(res["lane_logit_delta"]).reshape(
                b_n, k_n, 3).numpy()).copy()
        a_delta[~(present_np & ~is_cav_np)] = 0.0
        a_delta[obst_np] = 0.0
        lane_delta[obst_np] = 0.0
    lc_last_np = st.lc_last.numpy()
    baseline_idm = np.zeros((b_n, k_n), dtype=np.float64)
    idm_raw_np = np.zeros((b_n, k_n), dtype=np.float64)
    enforce = bool(cfg.integrator == "d1" and cfg.enforce_lane_legality)
    last_margins = {}
    for b in range(b_n):
        rows = np.where(present_np[b])[0]
        lane_rows = lane_np[b][rows]
        s_rows = s_np[b][rows]
        lc_rows = lc_last_np[b][rows].copy()
        wv = baseline.WorldRows(
            params.off, lane_rows, s_rows, v_np[b][rows],
            length_np[b][rows], rows, lc_last=lc_rows)
        last_margins[b] = core.lane_margins(wv, t_change=t)
        d_left = lane_delta[b, :, 1] if cfg.residual_enabled else None
        d_right = lane_delta[b, :, 2] if cfg.residual_enabled else None
        core.lane_change_stage(wv, t_change=t, delta_left=d_left,
                               delta_right=d_right,
                               enforce_legality=enforce, counts=counts)
        v0_ov = None
        if getattr(cfg, "desired_speed_mps", None) is not None:
            v0_ov = np.asarray(cfg.desired_speed_mps, dtype=np.float64)[rows]
        acc_clip, idm_raw, acc_pre, _gap, _dv, _v0 = \
            core.accelerations(wv, v0_override=v0_ov)
        counts.accel_clip += int((acc_pre != acc_clip).sum())
        idm_raw_np[b, rows] = idm_raw
        baseline_idm[b, rows] = acc_clip
        orow = rows[obst_np[b, rows]]
        if orow.size:                      # obstacle never accelerates
            baseline_idm[b, orow] = 0.0
            idm_raw_np[b, orow] = 0.0
        lane_np[b][rows] = lane_rows      # write the executed changes back
        s_np[b][rows] = s_rows
        lc_last_np[b][rows] = lc_rows
    st.last_lane_margins = last_margins

    # ---- speeds: baseline + residual, then the commanded execution ----
    v_t = torch.from_numpy(v_np)
    a_eff = torch.from_numpy(baseline_idm + a_delta)
    v_raw = v_t + a_eff * dt
    v_new = v_raw.clamp(min=0.0)
    counts.speed_clip += int(((v_raw < 0.0)
                              & torch.from_numpy(present_np)).sum().item())
    cmd = torch.isfinite(st.held_target_mps) \
        & torch.from_numpy(present_np) & torch.from_numpy(is_cav_np)
    if cmd.any():
        ctrl_cfg = CtrlConfig(
            dt_s=dt,
            accel_min_mps2=-float(params.exec_p["b_kin_max_mps2"]),
            accel_max_mps2=float(params.exec_p["a_kin_max_mps2"]))
        v_exec = execution_speed(st.held_target_mps, v_t,
                                 torch.from_numpy(idm_raw_np),
                                 config=ctrl_cfg)
        v_new = torch.where(cmd, v_exec, v_new)
    v_new_np = v_new.numpy()
    v_new_np[obst_np] = 0.0

    # ---- (c) arrivals (P4 head + sampler; engine-side insertion) ------
    arr_counts = np.zeros((b_n, 2), dtype=np.int64)
    st.last_arrival_counts = arr_counts.copy()
    st.last_inserted = arr_counts.copy()
    if cfg.arrivals_enabled:
        arrival = getattr(bundle, "arrival", None)
        if arrival is None:
            raise ValueError(
                "arrivals_enabled=True requires bundle.arrival (P4)")
        if rng is None:
            raise ValueError("arrivals_enabled=True requires rng")
        counter = getattr(cfg, "counter", None)
        if counter is None:
            raise ValueError("arrivals_enabled=True requires config.counter")
        head_out = arrival(torch.as_tensor(counter, dtype=torch.float64))
        sampler = getattr(arrival, "sampler", None)
        if sampler is None:
            from arrival.sampler import ArrivalSampler

            sampler = ArrivalSampler()
        counts_c = np.asarray(sampler.sample_counts(head_out, rng=rng),
                              dtype=np.int64).reshape(b_n, 2)
        speeds_c = np.asarray(sampler.sample_speeds(head_out, rng=rng),
                              dtype=np.float64).reshape(b_n, 2)
        inserted_np = np.zeros((b_n, 2), dtype=np.int64)
        for b in range(b_n):
            ins_row, ins_slots = _insert_arrivals(
                s_np, v_np, a_np, length_np, lane_np, present_np,
                is_cav_np, slot_np, b, counts_c[b], speeds_c[b],
                max_insert=cfg.max_insert_per_entrance,
                next_slot_id=st.next_slot_id.numpy())
            inserted_np[b] = ins_row
            for slot in ins_slots:
                v_new_np[b, slot] = max(0.0, v_np[b, slot])
        st.last_arrival_counts = counts_c.copy()
        st.last_inserted = inserted_np.copy()

    # ---- (d) validity + integration -----------------------------------
    s_t = torch.from_numpy(s_np)
    if cfg.integrator == "d1":
        s_new = integrate_d1_vnew(s_t, v_t, torch.from_numpy(v_new_np),
                                  dt_s=dt)
    else:
        s_new = integrate_ballistic(s_t, torch.from_numpy(v_new_np),
                                    dt_s=dt)
    # accel-lane end guard (engine.py:662-669)
    s_new_np = s_new.numpy()
    past = present_np & (lane_np == 2) & (s_new_np > params.s1z)
    if past.any():
        st.anomalies["accel_end_clamped"] = \
            st.anomalies.get("accel_end_clamped", 0) + int(past.sum())
        s_new_np[past] = params.s1z
        v_new_np[past] = 0.0
    if cfg.gap_projection:
        s_fix, v_fix, counts = project_same_lane_gaps(
            torch.from_numpy(s_new_np), torch.from_numpy(lane_np),
            torch.from_numpy(v_new_np), torch.from_numpy(length_np),
            torch.from_numpy(present_np), dt_s=dt, counts=counts)
        s_new_np = s_fix.numpy()
        v_new_np = v_fix.numpy()
    if obst_np.any():                      # INC1: hold the obstacle fixed
        s_new_np[obst_np] = s_np[obst_np]
        v_new_np[obst_np] = 0.0
    a_used_np = (v_new_np - v_np) / dt

    # collisions (counted, never repaired) + exits (engine.py:672-697)
    collisions = 0
    for b in range(b_n):
        rows = np.where(present_np[b])[0]
        if rows.size == 0:
            continue
        wv2 = baseline.WorldRows(
            params.off, lane_np[b][rows], s_new_np[b][rows],
            v_new_np[b][rows], length_np[b][rows], rows)
        step_coll = core.collisions(wv2)
        if step_coll:
            collisions += step_coll
            key = st.step_i + 1
            st.collisions_by_step[key] = \
                st.collisions_by_step.get(key, 0) + step_coll
        gone = rows[(s_new_np[b][rows] > params.llen[lane_np[b][rows]])
                    & ~obst_np[b, rows]]
        if getattr(cfg, "physical_exits", False):
            # ``s_new`` is the model ``s_down`` coordinate; the lane end
            # (``params.llen``) is physical, so convert before comparing.
            lim = params.llen[lane_np[b][rows]] - params.off[lane_np[b][rows]]
            gone = rows[(s_new_np[b][rows] > lim) & ~obst_np[b, rows]]
        present_np[b, gone] = False
    st.step_i += 1
    st.last_step_collisions = collisions

    out_world = WorldState(
        s_m=torch.from_numpy(s_new_np.copy()).to(out_dtype),
        v_mps=torch.from_numpy(v_new_np.copy()).to(out_dtype),
        a_mps2=torch.from_numpy(a_used_np.copy()).to(out_dtype),
        lane=torch.from_numpy(lane_np.copy()),
        length_m=torch.from_numpy(length_np.copy()).to(out_dtype),
        present=torch.from_numpy(present_np.copy()),
        is_cav=torch.from_numpy(is_cav_np.copy()),
        wall_group=wall_group.clone(),
        slot_id=torch.from_numpy(slot_np.copy()),
        obstacle=(None if getattr(world, "obstacle", None) is None
                  else torch.from_numpy(obst_np.copy())),
    )
    aux = StepAux(
        cav_target_speed_mps=st.held_target_mps.to(out_dtype).clone(),
        cav_reason_code=st.held_reason.clone(),
        hdv_accel_delta_mps2=torch.from_numpy(a_delta.copy()).to(out_dtype),
        hdv_lane_logit_delta=torch.from_numpy(
            lane_delta.copy()).to(out_dtype),
        baseline_idm_accel_mps2=torch.from_numpy(
            baseline_idm.copy()).to(out_dtype),
        counts=counts,
        teacher_forced=teacher_forced,
    )
    return out_world, aux
