"""INC1 predictor: per-arm delay / TTC<3 / hard-brake over the 300 s window.

The predictor takes one saved t0 state plus the resolved obstacle record and
rolls every arm (R0-R6) for both models -- WM-N1 Idea A (D1, the per-member
ensemble) and the Part A v2g rollout baseline -- with no SUMO information
after t0.  Event tokens are OFF by default (``event_tokens=False``); the
``True`` path is an explicitly labelled ablation.

Truth-derivation is deliberately separated from rollout execution: the pure
:func:`trajectory_metrics` is what the synthetic tests pin, while
:class:`Inc1Predictor` only orchestrates arm selection and the two rollouts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch

from rollout2 import baseline
from rollout2.state import StrategyActions

from . import incident_ctrl as IC
from .obstacle import Obstacle

MODELS = ("idea_a", "v2g")
HARD_BRAKE_MPS2 = -3.0
TTC_TARGET_S = 3.0
DT_S = 0.5


# ------------------------------------------------------------------ metrics
def _to_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu").numpy()
    return np.asarray(x)


@dataclass
class WindowMetrics:
    """Window aggregates for one arm/model on one world."""

    delay_s_per_veh: np.ndarray          # [B]
    ttc_lt3_s: np.ndarray                # [B]
    hard_brake_s: np.ndarray             # [B]
    collisions: np.ndarray               # [B]
    teleports: np.ndarray                # [B]
    dropped: np.ndarray                  # [B]
    per_vehicle_delay_s: list = field(default_factory=list)
    per_vehicle_ttc_s: list = field(default_factory=list)
    per_vehicle_hard_brake_s: list = field(default_factory=list)


def _trajectory_metrics_sample(pos, spd, msk, acc, lane, length, params, *,
                               dt_s: float = DT_S) -> dict:
    """One world's metrics from ``[T, K]`` trajectories."""
    t_n, k_n = pos.shape
    present0 = msk[0]
    delay = np.zeros(k_n, dtype=np.float64)
    ttc = np.zeros(k_n, dtype=np.float64)
    hard = np.zeros(k_n, dtype=np.float64)
    collisions = 0
    if acc is None:
        acc = np.zeros((t_n - 1, k_n), dtype=np.float64)
    for t in range(t_n):
        li = np.where(msk[t])[0]
        if li.size == 0:
            continue
        limit = params.seg_limit_vec(lane[t, li], pos[t, li])
        delay[li] += dt_s * np.clip(
            1.0 - spd[t, li] / np.maximum(limit, 1e-9), 0.0, 1.0)
    # same-lane TTC / collisions over the frozen positions
    for t in range(t_n):
        li = np.where(msk[t])[0]
        if li.size < 1:
            continue
        for ln in (0, 1, 2):
            same = li[lane[t, li] == ln]
            if same.size < 2:
                continue
            order = same[np.argsort(pos[t, same], kind="stable")]
            for back, front in zip(order[:-1], order[1:]):
                gap = pos[t, front] - length[t, front] - pos[t, back]
                dv = spd[t, back] - spd[t, front]
                if gap < 0.0:
                    collisions += 1
                elif dv > 1e-9:
                    ttc_gap = gap / dv
                    if ttc_gap < TTC_TARGET_S:
                        ttc[back] += dt_s
    if acc.size:
        hard += dt_s * (acc < HARD_BRAKE_MPS2).sum(axis=0).astype(np.float64)
    n_veh = int(present0.sum())
    mean_delay = float(delay[present0].sum() / n_veh) if n_veh else 0.0
    return dict(delay=delay, ttc=ttc, hard=hard,
                delay_s_per_veh=mean_delay,
                ttc_lt3_s=float(ttc.sum()), hard_brake_s=float(hard.sum()),
                collisions=float(collisions), teleports=0.0, dropped=0.0,
                present0=present0)


def trajectory_metrics(rollouts, params, *, dt_s: float = DT_S) -> WindowMetrics:
    """Per-arm window metrics from a :class:`rollout2.state.Rollouts`.

    ``delay_s_per_veh`` is the mean per-vehicle lost time against the current
    segment limit; ``ttc_lt3_s``/``hard_brake_s`` are same-lane exposure and
    time with ``a < -3.0 m/s^2``.  All quantities are per world.
    """
    pos = _to_np(rollouts.positions)
    spd = _to_np(rollouts.speeds)
    msk = _to_np(rollouts.masks).astype(bool)
    acc = _to_np(rollouts.accel)
    lanes = _to_np(rollouts.lanes)
    length = None
    if pos.ndim == 4:                       # [S, T, B, K]
        pos = pos.mean(axis=0)
        spd = spd.mean(axis=0)
        msk = msk.any(axis=0)
        acc = acc.mean(axis=0)
        lanes = lanes[0]
    k_n = pos.shape[-1]
    length = np.full((pos.shape[0], k_n), 5.0, dtype=np.float64)
    b_n = pos.shape[1]
    out = {k: [] for k in ("delay_s_per_veh", "ttc_lt3_s", "hard_brake_s",
                           "collisions", "teleports", "dropped")}
    per_veh = {k: [] for k in ("delay", "ttc", "hard")}
    for b in range(b_n):
        m = _trajectory_metrics_sample(pos[:, b, :], spd[:, b, :],
                                       msk[:, b, :], acc[:, b, :],
                                       lanes[:, b, :], length, params,
                                       dt_s=dt_s)
        for k in out:
            out[k].append(m[k])
        per_veh["delay"].append(m["delay"])
        per_veh["ttc"].append(m["ttc"])
        per_veh["hard"].append(m["hard"])
    return WindowMetrics(
        delay_s_per_veh=np.asarray(out["delay_s_per_veh"]),
        ttc_lt3_s=np.asarray(out["ttc_lt3_s"]),
        hard_brake_s=np.asarray(out["hard_brake_s"]),
        collisions=np.asarray(out["collisions"]),
        teleports=np.asarray(out["teleports"]),
        dropped=np.asarray(out["dropped"]),
        per_vehicle_delay_s=per_veh["delay"],
        per_vehicle_ttc_s=per_veh["ttc"],
        per_vehicle_hard_brake_s=per_veh["hard"])


# ---------------------------------------------------------------- arm wiring
def _base_actions(world, *, damping=False, wall_members=(), wall_speed=None):
    codes = torch.full_like(world.lane, -1)
    codes = torch.where(world.is_cav, torch.zeros_like(codes), codes)
    wall_speeds = None
    if damping:
        codes = torch.where(world.is_cav,
                            torch.full_like(codes, IC.ACTION_CODES["damping"]),
                            codes)
    if wall_members:
        codes = torch.where(world.is_cav, torch.zeros_like(codes), codes)
        for _, idx in wall_members:
            codes[0, idx] = IC.ACTION_CODES["wall"]
        wall_speeds = IC.r4_wall_speeds(world, wall_members, wall_speed)
    return StrategyActions(action_code=codes, wall_speed_mps=wall_speeds)


@dataclass
class ArmPlan:
    """The resolved bundle/actions for one arm on one world."""

    arm: str
    bundle: object
    actions: Optional[StrategyActions]
    bundle_is_incident_mover: bool = False
    log: dict = field(default_factory=dict)


def build_arm_plan(arm: str, model: str, base_bundle, world, obstacle: Obstacle,
                   params) -> ArmPlan:
    """Resolve the controller bundle/actions for ``arm`` (registration s4)."""
    from dataclasses import replace

    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}")
    if arm not in IC.ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    bundle = base_bundle
    log: dict = {}
    if arm == "R0":
        return ArmPlan(arm, bundle, _base_actions(world), log=log)
    if arm == "R1":
        sel = IC.r1_select(world, obstacle.s_m)
        if not bool(sel.any()):
            log["r1_no_qualifying_cav"] = True
            return ArmPlan(arm, bundle, _base_actions(world), log=log)
        mover = IC.IncidentCavMover("R1", obstacle.s_m)
        bundle = replace(base_bundle, ctrl=mover)
        return ArmPlan(arm, bundle, None, True, log=log)
    if arm == "R2":
        mover = IC.IncidentCavMover("R2", obstacle.s_m)
        bundle = replace(base_bundle, ctrl=mover)
        return ArmPlan(arm, bundle, None, True, log=log)
    if arm == "R3":
        return ArmPlan(arm, bundle,
                       _base_actions(world, damping=True), log=log)
    if arm in ("R5", "R6"):
        # ADDENDUM_2 A2.2/A2.3: the shared generator is the in-the-loop CAV
        # mover for both models; infeasible membership -> native (R0-like).
        from .scripted_mover import ScriptedCavMover, feasibility_from_world
        feasible, reason, members = feasibility_from_world(
            arm, world, obstacle.s_m)
        log[f"{arm.lower()}_feasible"] = bool(feasible)
        log[f"{arm.lower()}_reason"] = reason
        log[f"{arm.lower()}_members"] = list(members)
        if not feasible:
            return ArmPlan(arm, bundle, _base_actions(world), log=log)
        mover = ScriptedCavMover(arm, obstacle.s_m, t0_s=0.0)
        return ArmPlan(arm, replace(base_bundle, ctrl=mover), None, True,
                       log=log)
    # R4
    lead = _lead_speed(world, params)
    members, refusal = IC.r4_members(world, obstacle.s_m)
    if not members:
        log["r4_infeasible"] = refusal
        return ArmPlan(arm, bundle, _base_actions(world), log=log)
    wall_speed = IC.r4_wall_speed(world, members, free_flow_mps=lead)
    members, feasible, refusal = IC.r4_assembly(
        world, obstacle.s_m, wall_speed)
    log["r4_feasible"] = bool(feasible)
    log["r4_refusal"] = refusal
    if not feasible:
        return ArmPlan(arm, bundle, _base_actions(world), log=log)
    return ArmPlan(arm, bundle,
                   _base_actions(world, wall_members=members,
                                 wall_speed=wall_speed), log=log)


def _lead_speed(world, params) -> float:
    """Segment limit at the world's most upstream CAV (a free-flow reference)."""
    is_cav = world.is_cav.to(torch.bool) & world.present.to(torch.bool)
    if not bool(is_cav.any()):
        return 20.0
    s = world.s_m.to(torch.float64)
    masked = torch.where(is_cav, s, torch.full_like(s, 1e18))
    idx = int(torch.argmin(masked).item())
    lane = int(world.lane.reshape(-1)[idx].item())
    return float(params.seg_limit(max(lane, 0), float(s.reshape(-1)[idx])))


# ---------------------------------------------------------------- predictor
@dataclass
class WorldPrediction:
    world: str
    obstacle: dict
    idea_a: dict = field(default_factory=dict)
    v2g: dict = field(default_factory=dict)
    arms_meta: dict = field(default_factory=dict)


class Inc1Predictor:
    """Roll every arm for both models from one t0 world.

    ``rollout_fn(model, arm_plan, world, n_steps, seed) -> Rollouts`` is
    injectable so CPU tests can supply a synthetic rollout; the default runs
    the Idea A / v2g rollouts with the arm bundle resolved by
    :func:`build_arm_plan`.
    """

    def __init__(self, base_bundle, params=None, *, n_steps: int = 600,
                 seed: int = 0, n_members: int = 1,
                 rollout_fn: Optional[Callable] = None,
                 event_tokens: bool = False) -> None:
        self.base_bundle = base_bundle
        self.params = baseline.Params(params) if not isinstance(
            params, baseline.Params) else params
        self.raw_params = params
        self.n_steps = int(n_steps)
        self.seed = int(seed)
        self.n_members = int(n_members)
        self.rollout_fn = rollout_fn
        self.event_tokens = bool(event_tokens)

    def _rollout(self, model, plan, world, seed):
        if self.rollout_fn is not None:
            return self.rollout_fn(model, plan, world, self.n_steps, seed)
        from .rollout_baseline import idea_a_rollout, rollout_v2g_baseline

        if model == "v2g":
            return rollout_v2g_baseline(
                plan.bundle, world, n_steps=self.n_steps,
                actions=plan.actions, seed=seed, params=self.raw_params)
        return idea_a_rollout(
            plan.bundle, world, n_steps=self.n_steps, actions=plan.actions,
            seed=seed, params=self.raw_params, event_tokens=self.event_tokens)

    def predict(self, world, obstacle: Obstacle, *,
                world_name: str = "world") -> WorldPrediction:
        pred = WorldPrediction(world=world_name, obstacle=obstacle.to_record())
        for model in MODELS:
            per_arm = {}
            for arm in IC.ARMS:
                plan = build_arm_plan(arm, model, self.base_bundle, world,
                                      obstacle, self.params)
                rollouts = self._rollout(model, plan, world,
                                         self.seed + hash(arm) % 1000
                                         + (0 if model == "idea_a" else 5000))
                per_arm[arm] = trajectory_metrics(rollouts, self.params)
                pred.arms_meta[f"{model}:{arm}"] = plan.log
            setattr(pred, model, per_arm)
        return pred


def mean_metric(pred: WorldPrediction, model: str, arm: str,
                metric: str) -> float:
    """Mean of a per-world metric array (small helper for the metrics script)."""
    wm = getattr(pred, model)[arm]
    return float(np.mean(getattr(wm, metric)))
