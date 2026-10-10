"""The scoring engine: segmented controller-in-the-loop rollouts.

Wraps the existing ``rollout2`` engine (``step_once`` with a persistent
:class:`rollout2.state.Rollout2State`) and the P5 correction head; nothing
in ``rollout2`` / ``train.idea_a`` is modified.  One scored plan = one
segmented rollout per (member, sample):

* the plan's 30 s segments drive the per-tick ``StrategyActions`` (action
  codes + wall target speed), so scripted schedules switch controllers
  exactly at segment boundaries while control commands hold for their two
  0.5 s physics steps (the frozen 1 s control / 0.5 s physics cadence);
* obstacles are appended as non-CAV slots and PINNED each step (s fixed,
  v = 0, a = 0) — the INC1 held-obstacle convention;
* arrivals run only when the state carries entrance counters (P4 head +
  seeded sampler); otherwise the rollout is closed;
* per member, the rollout CAV trajectory feeds the P5 CAV-motion summary
  and the member's correction head, giving the per-tau outcome quantiles
  ``[delay_s, ttc_lt3_s, hard_brake_s] x 19 levels``.

Determinism: rollout math is float64; correction inference follows the
loaded encoder and correction modules' dtype/device. Each (member, sample) rollout
uses ``numpy.random.default_rng`` seeded from the request seed (documented
derivation in ``scorer.py``), so a fixed seed reproduces the result
bit-for-bit on CPU.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np
import torch

from .errors import KOverflowError
from .plans import ACTION_CODES, ResolvedPlan
from .state import K_MAX, StateSnapshot

_DT_S = 0.5
_HEADROOM_SLOTS = 16

#: metric name -> correction-head outcome target (model.constants).
METRIC_TARGETS = {"delay_s_per_veh": "delay_s",
                  "ttc3_exposure_s": "ttc_lt3_s",
                  "hard_brake_s": "hard_brake_s"}


def k_slots_for(state: StateSnapshot, *, arrivals_enabled: bool) -> int:
    """The padded slot budget (vehicle pool + arrival headroom, <= K_MAX)."""
    k_used = state.n_model_vehicles
    if k_used > K_MAX:
        raise KOverflowError(
            f"K overflow: {k_used} concurrent > K_MAX = {K_MAX} slots; "
            "refusing (CAVs are never silently dropped)")
    if not arrivals_enabled:
        return max(1, k_used)
    return int(min(K_MAX, max(1, k_used) + _HEADROOM_SLOTS))


def build_world(state: StateSnapshot, plan: ResolvedPlan, k_slots: int):
    """``(WorldState, code_slot_ids, obstacle_slots)`` for one plan.

    ``code_slot_ids`` maps model-world slot -> CAV id (slot order =
    ``state.model_rows()`` order, obstacles appended).  Wall membership
    (``wall_group``) is per plan, so the world is rebuilt per plan.
    """
    from rollout2.state import WorldState

    rows = state.model_rows()
    k_used = len(rows) + len(state.obstacles)
    if k_used > K_MAX:
        raise KOverflowError(
            f"K overflow: {len(rows)} vehicles + {len(state.obstacles)} "
            f"obstacles = {k_used} concurrent > K_MAX = {K_MAX} slots; "
            "refusing (CAVs are never silently dropped)")
    s = torch.zeros((1, k_slots), dtype=torch.float64)
    v = torch.zeros((1, k_slots), dtype=torch.float64)
    a = torch.zeros((1, k_slots), dtype=torch.float64)
    lane = torch.full((1, k_slots), -1, dtype=torch.int64)
    length = torch.zeros((1, k_slots), dtype=torch.float64)
    present = torch.zeros((1, k_slots), dtype=torch.bool)
    is_cav = torch.zeros((1, k_slots), dtype=torch.bool)
    wall_group = torch.full((1, k_slots), -1, dtype=torch.int64)
    slot_id = torch.arange(k_slots, dtype=torch.int64).reshape(1, k_slots)
    wall_ids = {cid: int(plan.wall["wall_id"])
                for cid in plan.wall_members} if plan.wall else {}
    code_slot_ids = {}
    row_i = 0
    for (idx, ln, sd, vv, aa, ll, cav) in rows:
        s[0, row_i] = sd
        v[0, row_i] = vv
        a[0, row_i] = aa
        lane[0, row_i] = ln
        length[0, row_i] = ll
        present[0, row_i] = True
        is_cav[0, row_i] = cav
        cid = str(state.ids[idx])
        if cav:
            code_slot_ids[row_i] = cid
            if cid in wall_ids:
                wall_group[0, row_i] = wall_ids[cid]
        row_i += 1
    obstacle_slots = []
    for ob in state.obstacles:
        s[0, row_i] = float(ob.s_down_m)
        lane[0, row_i] = int(ob.lane_phys)
        length[0, row_i] = float(ob.length_m)
        present[0, row_i] = True
        obstacle_slots.append(row_i)
        row_i += 1
    world = WorldState(s_m=s, v_mps=v, a_mps2=a, lane=lane, length_m=length,
                       present=present, is_cav=is_cav,
                       wall_group=wall_group, slot_id=slot_id)
    return world, code_slot_ids, obstacle_slots


def wall_speed_tensor(plan: ResolvedPlan, k_slots: int,
                      code_slot_ids: dict) -> torch.Tensor:
    """``[1, K]`` float64 wall target speed (NaN off the wall slots).

    Constant over the window: wall membership ends per segment through the
    plan's action codes, and the mover reads the speed only on WALL slots.
    """
    t = torch.full((1, k_slots), float("nan"), dtype=torch.float64)
    if plan.wall is not None:
        for slot, cid in code_slot_ids.items():
            if cid in plan.wall_members:
                t[0, slot] = float(plan.wall["speed_mps"])
    return t


def _actions_for_tick(plan: ResolvedPlan, world, code_slot_ids: dict,
                      k_slots: int, wall_speed, step_i: int):
    from rollout2.state import StrategyActions

    seg = min(int(step_i * _DT_S / 30.0), plan.n_segments - 1)
    codes = plan.segment_codes[seg]
    action_code = torch.full((1, k_slots), ACTION_CODES["native"],
                             dtype=torch.int64)
    for slot, cid in code_slot_ids.items():
        action_code[0, slot] = codes.get(cid, ACTION_CODES["native"])
    action_code[0, ~world.is_cav.to(torch.bool)[0]] = ACTION_CODES["none"]
    return StrategyActions(action_code=action_code,
                           wall_speed_mps=wall_speed)


def _pin_obstacles(world, obstacle_slots, obstacles) -> None:
    for j, ob in zip(obstacle_slots, obstacles):
        world.s_m[0, j] = float(ob.s_down_m)
        world.v_mps[0, j] = 0.0
        world.a_mps2[0, j] = 0.0


def _add_counts(a, b):
    from model.rollout import ValidityCounts

    out = ValidityCounts(**a.as_dict())
    out.add(b)
    return out


def _state_channels(state, k_slots: int):
    """Original t0 channels in build_world's exact filtered row order."""
    rows = state.model_rows()
    if len(rows) + len(state.obstacles) > k_slots:
        raise ValueError("StateSnapshot exceeds trajectory slot width")
    kin = np.zeros((1, k_slots, 7), dtype=np.float64)
    mask = np.zeros((1, k_slots), dtype=bool)
    cav = np.zeros((1, k_slots), dtype=bool)
    for j, (_, lane, s, v, a, length, is_cav) in enumerate(rows):
        kin[0, j, [0, 4, 5, 6]] = [s, v, a, length]
        kin[0, j, 1 + lane] = 1.0
        mask[0, j], cav[0, j] = True, is_cav
    for j, ob in enumerate(state.obstacles, start=len(rows)):
        kin[0, j, 0] = ob.s_down_m
        kin[0, j, 1 + ob.lane_phys] = 1.0
        kin[0, j, 6] = ob.length_m
        mask[0, j] = True
    counter = state.entrance_counters
    return dict(t0_kin=kin, t0_mask=mask, t0_is_cav=cav,
                t0_slot_id=np.arange(k_slots, dtype=np.int64).reshape(1, -1),
                t0_counter=None if counter is None else np.asarray(counter)[None])


def _state_kin(state, k_slots: int):
    """Compatibility view of the state channels in model-world slot order."""
    data = _state_channels(state, k_slots)
    return tuple(None if data[k] is None else torch.as_tensor(data[k])
                 for k in ("t0_kin", "t0_mask", "t0_counter"))


@torch.no_grad()
def outcome_head(bundle, pos: np.ndarray, spd: np.ndarray, acc: np.ndarray,
                 msk: np.ndarray, *, state=None, slot_ids=None,
                 t0_kin=None, t0_mask=None, t0_is_cav=None, t0_slot_id=None,
                 t0_counter=None):
    """Score raw16 through the member's trained projection and head.

    Positions/speeds/masks/slot_ids include t0: [T+1,1,K]; acceleration
    describes executed transitions: [T,1,K]. Original CAV identity and
    initial acceleration come from explicit channels or StateSnapshot.
    Temporal slot IDs are mandatory, including for injected trajectories.
    """
    from corr.evaluation import (module_data, state_token, trained_correction,
                                 trajectory_summary)
    from model import constants as C

    corr = trained_correction(bundle)
    # The scoring rollout runs the full 300 s window; the correction head was
    # trained on the summary built over ``rollout_horizon_steps``.  Slice the
    # trajectory back to that registered window (single source
    # ``model.constants.CORRECTION_WINDOW_STEPS``) so eval matches training.
    window = C.correction_window_steps(pos.shape[0])
    pos, spd, msk = pos[:window + 1], spd[:window + 1], msk[:window + 1]
    if acc is not None:
        acc = acc[:window]
    if slot_ids is not None:
        slot_ids = slot_ids[:window + 1]
    explicit = dict(t0_kin=t0_kin, t0_mask=t0_mask, t0_is_cav=t0_is_cav,
                    t0_slot_id=t0_slot_id, t0_counter=t0_counter)
    # Partial explicit identity is ambiguous and must not be patched with a
    # second roster. Counter-only injection likewise needs explicit state.
    if any(value is not None for value in explicit.values()):
        data = explicit
    elif state is not None:
        data = _state_channels(state, int(pos.shape[-1]))
    else:
        data = explicit
    raw = trajectory_summary(
        pos, spd, acc, msk, slot_ids=slot_ids,
        **{k: data[k] for k in ("t0_kin", "t0_mask", "t0_is_cav", "t0_slot_id")})
    raw_input = module_data(corr, raw)
    token = state_token(bundle, raw_input, kin=data["t0_kin"],
                        mask=data["t0_mask"], counter=data["t0_counter"],
                        d_model=int(corr.cfg.d_model))
    out = corr(raw_input, token)
    q = out["outcome_quantiles"]
    norm = getattr(bundle, "outcome_norm", None)
    if norm is not None:
        from model.outcome_norm import denormalize_quantiles

        q = denormalize_quantiles(q, norm)
    return (q.detach().cpu().to(torch.float64).numpy()[0],
            raw.detach().cpu().to(torch.float64).numpy()[0])


def rollout_member(member, state: StateSnapshot, plan: ResolvedPlan, *,
                   n_steps: int, t0_s: float, seed: int,
                   arrivals_enabled: bool, counters: Optional[np.ndarray],
                   k_slots: int) -> dict:
    """One (member, sample) segmented rollout of ``plan`` from ``state``."""
    from rollout2.baseline import Params
    from rollout2.state import Rollout2Config, Rollout2State
    from rollout2.step import step_once

    bundle = member.bundle
    cfg0 = Rollout2Config(
        params=None,
        integrator="d1",
        residual_enabled=getattr(bundle, "hdv", None) is not None,
        arrivals_enabled=bool(arrivals_enabled),
        t0_s=float(t0_s))
    world, code_slot_ids, obstacle_slots = build_world(state, plan, k_slots)
    wall_speed = wall_speed_tensor(plan, k_slots, code_slot_ids)
    rng = np.random.default_rng(int(seed))
    params = Params(cfg0.params)
    st = Rollout2State(world.batch, world.slots)
    st.next_slot_id = world.slot_id.detach().cpu().amax(dim=1) + 1
    # Freeze before any controller step; future slot occupants cannot change
    # the original self-identified CAV selection or the A6 state token.
    from rollout2.engine import _t0_kin

    t0_channels = dict(
        t0_kin=_t0_kin(world).detach().clone(),
        t0_mask=world.present.detach().clone(),
        t0_is_cav=world.is_cav.detach().clone(),
        t0_slot_id=world.slot_id.detach().clone(),
        t0_counter=(None if state.entrance_counters is None else
                    np.asarray(state.entrance_counters).copy()[None]))
    cfg = replace(cfg0)
    cfg.state = st
    cfg.counter = (torch.as_tensor(
        np.asarray(counters, dtype=np.float64)).reshape(1, 2, 30)
        if arrivals_enabled else None)
    pos = np.zeros((n_steps + 1, 1, k_slots), dtype=np.float64)
    spd = np.zeros((n_steps + 1, 1, k_slots), dtype=np.float64)
    lns = np.zeros((n_steps + 1, 1, k_slots), dtype=np.int64)
    msk = np.zeros((n_steps + 1, 1, k_slots), dtype=bool)
    ids = np.zeros((n_steps + 1, 1, k_slots), dtype=np.int64)
    acc = np.zeros((n_steps, 1, k_slots), dtype=np.float64)
    arr = np.zeros((n_steps, 1, 2), dtype=np.int64)
    ins = np.zeros((n_steps, 1, 2), dtype=np.int64)
    totals = None

    def _snap(j: int) -> None:
        pos[j] = world.s_m.numpy()
        spd[j] = world.v_mps.numpy()
        lns[j] = world.lane.numpy()
        msk[j] = world.present.numpy()
        ids[j] = world.slot_id.numpy()

    _snap(0)
    with torch.no_grad():
        for i in range(int(n_steps)):
            acts = _actions_for_tick(plan, world, code_slot_ids, k_slots,
                                     wall_speed, i)
            world, aux = step_once(
                world, acts, bundle=bundle, config=cfg,
                t_s=t0_s + i * params.dt, rng=rng)
            if obstacle_slots:
                _pin_obstacles(world, obstacle_slots, state.obstacles)
            acc[i] = world.a_mps2.numpy()
            arr[i] = getattr(st, "last_arrival_counts",
                             np.zeros((1, 2), dtype=np.int64))
            ins[i] = getattr(st, "last_inserted",
                             np.zeros((1, 2), dtype=np.int64))
            totals = aux.counts if totals is None else _add_counts(totals,
                                                                   aux.counts)
            _snap(i + 1)
    outcome_q, summary_raw = outcome_head(bundle, pos, spd, acc, msk,
                                          slot_ids=ids, **t0_channels)
    return dict(positions=pos, speeds=spd, accel=acc, masks=msk,
                outcome_quantiles=outcome_q, summary_raw=summary_raw,
                validity=(totals.as_dict() if totals else {}),
                collisions=dict(st.collisions_by_step),
                anomalies=dict(st.anomalies),
                arrivals=int(arr.sum()), inserted=int(ins.sum()),
                steps=int(n_steps))
