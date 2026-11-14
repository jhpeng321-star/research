"""A11 world model behind the planner ``WorldModel`` contract (WM-N1-REPL-CL Part C).

``A11Predictor.predict_many(state_samples, plans)`` returns one
``traffic_shepherd.planner.interface.Prediction`` per plan (contract
``wd01_predict/1.1``, M = 1 member, K = 1 cell -- A11 member_1 is a single
deterministic member), built from:

1. **Dynamics rollout** -- ONE batched A11 P3 rollout
   (``rollout2.step.step_once``, integrator ``d1`` + HDV residual, arrivals
   off) over all distinct plans plus the all-native reference, CAVs driven by
   the REAL controller mirrors in ``bundle.ctrl`` (``ctrl.mover.CavMover``:
   native = no command / IDM, damping = CIRCLES ONNX policy, pi = PI
   external-U). Per-CAV action codes are constant over the rollout (v1 plans
   are constant from t0). This is the batched form of
   ``evalh.scorer.actions.rollout_with_schedule`` (same config, same step
   function; the scorer's version is single-world only).
2. **delay_vs_native_s** -- the A11 plan-aware paired-effect head
   (``bundle.paired_effect``, ``plan_raw_f = 60``) on the decision state's A6
   token and the plan's 120 s long-plan vector, built EXACTLY like training
   (``model.paired_effect.long_plan_summary`` over the CAV speed (field 0) and
   acceleration (field 1) planned channels, float16-rounded at the pack
   scales, alive-since-t0 mask), here from the plan's own rollout CAV motion.
   Decision metric = the MEAN of the monotone dV quantiles at tau = 300 s
   (registered tau; the scorer's decision-quality construction); the q0.9
   quantile is recorded in ``meta`` (risk variant). The all-native plan is
   exactly 0 (scorer convention ``native == 0``); the raw head value on the
   native rollout is logged as ``meta['head_native_dv_mean']`` for audit.
   ``native_reference="head"`` (opt-in, NOT the registered default) subtracts
   it instead.
3. **Safety signals** (documented approximations, computed on the rollout
   over the planner's trajectory window, default 60 s = one decision period):
   ``min_ttc_s`` = min over same-lane follower/leader pairs with closing
   speed of bumper gap / closing speed (NaN with ``has_closing_pair=False``
   when no closing pair); ``ttc_lt3_exposure_s`` / ``ttc_lt1p5_exposure_s`` =
   follower-seconds with TTC below 3 / 1.5 s; ``hard_brake_s`` =
   vehicle-seconds with acceleration < -3 m/s^2 (``v0lib.HARD_BRAKE_ACCL``);
   ``collision`` = any same-lane bumper overlap (gap < 0) in the window.
   ``native_*`` twins come from the shared all-native rollout.
4. ``trajectories_tracked`` -- ``{vehicle_id: [1, 1, T, 4]}`` columns
   ``(t_s, s_m, lane_id, speed_mps)`` for every vehicle tracked at the
   decision state, NaN after it leaves (or its slot is reused).

Absolute delays are not predicted by this head: ``native_delay_s = 0`` and
``plan_delay_s = delay_part_a_s = delay_vs_native_s`` (relative frame),
``delay_part_b_s = 0``. Walls are not supported here (fail closed): a plan
carrying walls raises unless a ``wall_codes`` hook is injected (the walls
variant supplies it).
"""
from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import torch

#: Defaults (overridable by env / runner args: WM_N1_A11_MEMBER,
#: WM_N1_A11_MEMBER_SHA256, WM_N1_PAIRED_EFFECT_CLIMATOLOGY, WM_N1_DEVICE).
A11_MEMBER_PATH = Path(
    "E:/research-dense/runs/active/WM-N1-A11-20261005/members/member_1/member_1.pt")
A11_MEMBER_SHA256 = "41221852a72d68e190100ef6a96b7583b7571f1ce1ccd185a68d2b1df63f21e8"
A11_CLIMATOLOGY = Path(
    "E:/research-dense/runs/active/WM-N1-SCORE-A9-20261005/TRAIN_CLIMATOLOGY.json")


def member_path() -> Path:
    return Path(os.environ.get("WM_N1_A11_MEMBER") or A11_MEMBER_PATH)


def member_sha256() -> str:
    return str(os.environ.get("WM_N1_A11_MEMBER_SHA256") or A11_MEMBER_SHA256)


def default_device() -> str:
    return str(os.environ.get("WM_N1_DEVICE") or "cpu")

DT_S = 0.5
DECISION_TAU_S = 300.0
PLAN_HORIZON_S = 120.0
HARD_BRAKE_ACCL = -3.0
CONTRACT = "wd01_predict/1.1"

#: planner action -> ctrl.config.ACTION_CODES
PLANNER_ACTION_CODES = {"native": 0, "damping": 1, "pi-external-u": 2}
NONE_CODE = -1


class A11PredictorError(RuntimeError):
    """Fail-closed refusal of the A11 adapter."""


# --------------------------------------------------------------- loading
def load_a11_bundle(path=None, sha256=None, climatology=None):
    """Load the A11 member through the scorer's pinned checkpoint binder.

    ``path`` / ``sha256`` default to env WM_N1_A11_MEMBER /
    WM_N1_A11_MEMBER_SHA256 (else the registered E: member + pin); the
    climatology override goes through WM_N1_PAIRED_EFFECT_CLIMATOLOGY (an
    explicit ``climatology`` argument wins; else the env; else the E: file).
    """
    path = Path(path) if path is not None else member_path()
    sha256 = str(sha256) if sha256 is not None else member_sha256()
    if climatology is not None:
        os.environ["WM_N1_PAIRED_EFFECT_CLIMATOLOGY"] = str(climatology)
    elif not os.environ.get("WM_N1_PAIRED_EFFECT_CLIMATOLOGY") and A11_CLIMATOLOGY.is_file():
        os.environ["WM_N1_PAIRED_EFFECT_CLIMATOLOGY"] = str(A11_CLIMATOLOGY)
    from evalh.scorer.checkpoint import load_scoring_checkpoint

    ck = load_scoring_checkpoint(str(path), expected_sha256=str(sha256))
    bundle = ck.bundle
    head = getattr(bundle, "paired_effect", None)
    if head is None or not bool(getattr(head, "plan_aware", False)):
        raise A11PredictorError("A11 bundle has no plan-aware paired-effect head")
    h = getattr(getattr(bundle, "config", None), "paired_effect_plan_horizon_s", None)
    if h is None or abs(float(h) - PLAN_HORIZON_S) > 1e-9:
        raise A11PredictorError(f"A11 plan horizon {h!r} != registered {PLAN_HORIZON_S}")
    if getattr(bundle, "ctrl", None) is None:
        raise A11PredictorError("A11 bundle has no controller mover (ctrl)")
    return ck


# ------------------------------------------------------------ rollout core
@dataclass
class BatchRollout:
    pos: np.ndarray      # [T+1, B, K]
    spd: np.ndarray      # [T+1, B, K]
    acc: np.ndarray      # [T, B, K] (accel over step i)
    lns: np.ndarray      # [T+1, B, K]
    msk: np.ndarray      # [T+1, B, K] bool
    ids: np.ndarray      # [T+1, B, K]
    length: np.ndarray   # [B, K]
    a0: np.ndarray       # [B, K] initial accel
    seconds: float = 0.0
    idm_raw: Optional[np.ndarray] = None       # [T, B, K] (traced; skip.py)
    baseline_idm: Optional[np.ndarray] = None  # [T, B, K] (traced; skip.py)

    def at(self, b: int) -> "BatchRollout":
        """Single-plan view (B = 1) of plan ``b``."""
        sl = slice(b, b + 1)
        return BatchRollout(
            pos=self.pos[:, sl], spd=self.spd[:, sl], acc=self.acc[:, sl],
            lns=self.lns[:, sl], msk=self.msk[:, sl], ids=self.ids[:, sl],
            length=self.length[sl], a0=self.a0[sl], seconds=0.0,
            idm_raw=None if self.idm_raw is None else self.idm_raw[:, sl],
            baseline_idm=None if self.baseline_idm is None else self.baseline_idm[:, sl])


def world_from_live(state, batch: int):
    """rollout2 WorldState for a LiveState, repeated over ``batch`` plans."""
    from rollout2.engine import world_from_batch

    kin = torch.as_tensor(np.asarray(state.veh_kin, np.float32)).unsqueeze(0)
    world = world_from_batch({
        "veh_kin": kin.expand(batch, -1, -1).contiguous(),
        "veh_mask": torch.as_tensor(_rollout_mask(state)).unsqueeze(0)
        .expand(batch, -1).contiguous(),
        "is_cav": torch.as_tensor(np.asarray(state.is_cav, bool)).unsqueeze(0)
        .expand(batch, -1).contiguous(),
        "slot_id": torch.as_tensor(np.asarray(state.slot_id, np.int64)).unsqueeze(0)
        .expand(batch, -1).contiguous(),
    })
    return world


def _rollout_mask(state) -> np.ndarray:
    m = getattr(state, "rollout_mask", None)
    if m is None:
        return np.asarray(state.veh_mask, bool)
    return np.asarray(m, bool)


def batched_rollout(bundle, state, codes: np.ndarray, *, n_steps: int,
                    t0_s: float = 0.0, seed: int = 0,
                    wall_speed: Optional[np.ndarray] = None,
                    trace: bool = False, physical_exits: bool = False,
                    open_entries: bool = False, entry_cfg=None,
                    sumo_fidelity: bool = False,
                    sumo_speed_factor: Optional[float] = None,
                    sumo_idm: bool = True,
                    desired_speed_mps=None,
                    merge_cal: bool = False,
                    merge_cal_params=None,
                    residual_enabled: bool = True,
                    residual: Optional[bool] = None) -> BatchRollout:
    """Batched controller-mode rollout: ``codes`` int64 [B, K] held constant.

    Same configuration as ``evalh.scorer.actions.rollout_with_schedule``
    (integrator d1, residual on, arrivals off, fresh Rollout2State, the
    world's slot ids), batched over plans.

    ``open_entries=True`` adds vehicles at the network entrances from the
    observed entrance counters (``closedloop.rollout_entries``); default
    ``False`` is byte-identical to the historical rollout.

    ``physical_exits=True`` turns on the rollout's physical-coordinate road
    departures (``Rollout2Config.physical_exits``): a vehicle whose position
    passes the lane end is masked out from that step on, mirroring the
    recorded ``cav_plan_ch`` "on the road" definition.  Default ``False``
    preserves the historical rollout byte-for-byte.

    ``sumo_fidelity=True`` (default OFF = byte-identical) enables the
    ``s_down``-frame geometry + SUMO-vType-matched IDM
    (``Rollout2Config.sumo_fidelity``).

    ``desired_speed_mps`` (default ``None`` = OFF, byte-identical) is the
    optional per-slot desired speed (WM-S T3) estimated from each vehicle's
    RSU-observable pre-``t0`` motion; it overrides the segment limit in the
    own-lane IDM term (``driver.desired_speed``).
    """
    from rollout2 import baseline
    from rollout2.engine import _world_f64
    from rollout2.state import Rollout2Config, Rollout2State, StrategyActions
    from rollout2.step import step_once

    codes = np.asarray(codes, np.int64)
    b_n = int(codes.shape[0])
    world = world_from_live(state, b_n)
    k_n = world.slots
    if codes.shape != (b_n, k_n):
        raise A11PredictorError(f"codes {codes.shape} != (B, K)=({b_n}, {k_n})")
    cav = np.asarray(state.is_cav, bool) & _rollout_mask(state)
    if (codes[:, ~cav] != NONE_CODE).any():
        raise A11PredictorError("non-CAV slots must carry the 'none' code")
    if (codes[:, cav] < 0).any():
        raise A11PredictorError("every present CAV slot needs an action code (no default)")
    cfg = Rollout2Config(params=None, integrator="d1",
                         residual_enabled=(bool(residual_enabled) if residual is None else bool(residual)),
                         arrivals_enabled=False, t0_s=float(t0_s),
                         physical_exits=bool(physical_exits),
                         sumo_fidelity=bool(sumo_fidelity),
                         sumo_speed_factor=sumo_speed_factor,
                         sumo_idm=bool(sumo_idm),
                         desired_speed_mps=desired_speed_mps,
                         merge_cal=bool(merge_cal),
                         merge_cal_params=merge_cal_params)
    params = baseline.Params(cfg.params)
    dt = float(params.dt)
    st = Rollout2State(b_n, k_n)
    st.next_slot_id = world.slot_id.detach().cpu().amax(dim=1) + 1
    cfg.state = st
    w = _world_f64(world)
    ws = torch.full((b_n, k_n), float("nan"), dtype=torch.float64)
    if wall_speed is not None:
        ws = torch.as_tensor(np.asarray(wall_speed, np.float64))
    entries = None
    if open_entries:
        from .rollout_entries import EntryProcess

        codes = codes.copy()             # the process writes spawned-slot codes
        entries = EntryProcess(state, params, seed=int(seed), cfg=entry_cfg, batch=b_n)
    acts = StrategyActions(action_code=torch.as_tensor(codes), wall_speed_mps=ws)
    T = int(n_steps)
    pos = np.zeros((T + 1, b_n, k_n))
    spd = np.zeros((T + 1, b_n, k_n))
    lns = np.zeros((T + 1, b_n, k_n), np.int64)
    msk = np.zeros((T + 1, b_n, k_n), bool)
    ids = np.zeros((T + 1, b_n, k_n), np.int64)
    acc = np.zeros((T, b_n, k_n))
    rng = np.random.default_rng(int(seed))

    def snap(j):
        pos[j] = w.s_m.numpy()
        spd[j] = w.v_mps.numpy()
        lns[j] = w.lane.numpy()
        msk[j] = w.present.numpy()
        ids[j] = w.slot_id.numpy()

    a0 = w.a_mps2.numpy().copy()
    length = w.length_m.numpy().copy()
    t_start = time.perf_counter()
    snap(0)
    base_idm = np.zeros((T, b_n, k_n)) if trace else None
    import contextlib

    from .skip import Trace

    tracer = Trace() if trace else contextlib.nullcontext()
    with torch.no_grad(), tracer:
        for i in range(T):
            if entries is not None:
                entries.step(w, acts, st)
            w, aux = step_once(w, acts, bundle=bundle, config=cfg,
                               t_s=float(t0_s) + i * dt, rng=rng)
            acc[i] = w.a_mps2.numpy()
            if trace:
                base_idm[i] = aux.baseline_idm_accel_mps2.detach().to(torch.float64).numpy()
            snap(i + 1)
    out = BatchRollout(pos=pos, spd=spd, acc=acc, lns=lns, msk=msk, ids=ids,
                       length=(w.length_m.numpy().copy() if entries is not None else length),
                       a0=a0,
                       seconds=time.perf_counter() - t_start,
                       idm_raw=(tracer.idm_raw(T, b_n, k_n) if trace else None),
                       baseline_idm=base_idm)
    out.entry_summary = None if entries is None else [entries.summary(b) for b in range(b_n)]
    return out


# ------------------------------------------------------- plan + head
def alive_mask(roll: BatchRollout, n: int) -> np.ndarray:
    """[n, B, K] alive-since-t0 (training: present0 * cumprod(target_mask))."""
    same = roll.msk[1:n] & roll.msk[:-1][: n - 1] & (roll.ids[1:n] == roll.ids[:-1][: n - 1])
    alive = np.concatenate([roll.msk[:1], same], axis=0).astype(np.int64)
    return np.cumprod(alive, axis=0).astype(bool)


def long_plan_from_rollout(roll: BatchRollout, is_cav: np.ndarray,
                           horizon_s: float = PLAN_HORIZON_S) -> torch.Tensor:
    """A11 long-plan vector [B, W] from the rollout CAV motion (training form).

    Training reads ``cav_plan_ch`` field 0 = speed/40 and field 1 = accel/3
    at ticks t0..t0+n-1 (stored float16; accel at a tick = the acceleration
    that produced that tick's speed), alive-since-t0 mask from the t0 mask
    and ``target_mask``; binned by ``model.paired_effect.long_plan_summary``.
    """
    from features.idea_a import CAV_ACCEL_SCALE, CAV_TGT_SPEED_SCALE
    from model.paired_effect import long_plan_summary

    n = max(1, int(round(float(horizon_s) / DT_S)))
    if roll.pos.shape[0] < n:
        raise A11PredictorError(f"rollout has {roll.pos.shape[0]} ticks < plan {n}")
    sp = roll.spd[:n]
    ac = np.concatenate([roll.a0[None], roll.acc[: n - 1]], axis=0)
    sp = (sp / CAV_TGT_SPEED_SCALE).astype(np.float16).astype(np.float32) * float(CAV_TGT_SPEED_SCALE)
    ac = (ac / CAV_ACCEL_SCALE).astype(np.float16).astype(np.float32) * float(CAV_ACCEL_SCALE)
    alive = alive_mask(roll, n)
    cav = (np.asarray(is_cav, bool)[None, None, :] & alive).astype(np.float32)
    return long_plan_summary(torch.as_tensor(sp), torch.as_tensor(ac),
                             torch.as_tensor(cav), horizon_s=float(horizon_s))


def head_dv_quantiles(bundle, state, plan_long: torch.Tensor) -> np.ndarray:
    """[B, N_TAU, NQ] monotone dV quantiles: one shared A6 token, B plans.

    Batched form of ``wm_api.eval.paired_effect_quantiles_for_plan`` (same
    ``corr.evaluation.state_token`` + ``head.dv_quantiles``).
    """
    from corr.evaluation import state_token

    head = bundle.paired_effect
    d = int(head.d_model)
    ref_param = next(p for p in head.parameters() if p.is_floating_point())
    ref = torch.zeros(1, d, dtype=ref_param.dtype)
    kin = torch.as_tensor(np.asarray(state.veh_kin, np.float32)).unsqueeze(0)
    mask = torch.as_tensor(np.asarray(state.veh_mask, bool)).unsqueeze(0)
    counter = torch.as_tensor(np.asarray(state.counter, np.float32)).unsqueeze(0)
    plan = plan_long.to(ref_param.dtype)
    if int(plan.shape[-1]) != int(head.plan_raw_f):
        raise A11PredictorError(f"plan width {plan.shape[-1]} != head {head.plan_raw_f}")
    with torch.no_grad():
        token = state_token(bundle, ref, kin=kin, mask=mask, counter=counter, d_model=d)
        token = token.expand(int(plan.shape[0]), -1)
        q = head.dv_quantiles(token, plan=plan, monotone=True)
    return q.detach().cpu().to(torch.float64).numpy()


def tau_index(tau_s: float) -> int:
    from model import constants as MC

    taus = [round(float(t), 6) for t in MC.TAUS]
    return taus.index(round(float(tau_s), 6))


# ------------------------------------------------------- safety signals
@dataclass
class SafetyStats:
    min_ttc_s: float
    min_ttc_pair: Optional[tuple]
    has_closing_pair: bool
    ttc_lt3_s: float
    ttc_lt1p5_s: float
    hard_brake_s: float
    collision: bool
    min_gap_m: float


def safety_stats(roll: BatchRollout, b: int, n_window: int, slot_vid) -> SafetyStats:
    """Same-lane follower/leader TTC, exposures, hard brake, overlap (window)."""
    min_ttc = math.inf
    pair = None
    lt3 = lt15 = hb = 0.0
    coll = False
    min_gap = math.inf
    T = min(int(n_window), roll.acc.shape[0])
    length = roll.length[b]
    for j in range(1, T + 1):
        m = roll.msk[j, b] & (roll.lns[j, b] >= 0)
        rows = np.flatnonzero(m)
        if rows.size >= 2:
            lane = roll.lns[j, b, rows]
            s = roll.pos[j, b, rows]
            v = roll.spd[j, b, rows]
            order = np.lexsort((s, lane))
            lane, s, v, r = lane[order], s[order], v[order], rows[order]
            same = lane[1:] == lane[:-1]
            if same.any():
                f = np.flatnonzero(same)          # follower f, leader f+1
                gap = s[f + 1] - s[f] - length[r[f + 1]]
                if gap.size:
                    min_gap = min(min_gap, float(gap.min()))
                if (gap < 0).any():
                    coll = True
                dv = v[f] - v[f + 1]
                closing = (dv > 1e-9) & (gap >= 0)
                if closing.any():
                    ttc = gap[closing] / dv[closing]
                    lt3 += float((ttc < 3.0).sum()) * DT_S
                    lt15 += float((ttc < 1.5).sum()) * DT_S
                    k = int(np.argmin(ttc))
                    if float(ttc[k]) < min_ttc:
                        min_ttc = float(ttc[k])
                        ff = f[closing][k]
                        pair = (slot_vid[int(r[ff])], slot_vid[int(r[ff + 1])])
        a = roll.acc[j - 1, b]
        hb += float(((a < HARD_BRAKE_ACCL) & roll.msk[j, b]).sum()) * DT_S
    has = math.isfinite(min_ttc)
    return SafetyStats(min_ttc_s=(min_ttc if has else float("nan")), min_ttc_pair=pair,
                       has_closing_pair=has, ttc_lt3_s=lt3, ttc_lt1p5_s=lt15,
                       hard_brake_s=hb, collision=coll,
                       min_gap_m=(min_gap if math.isfinite(min_gap) else float("nan")))


def tracked_trajectories(roll: BatchRollout, b: int, n_window: int, slot_vid,
                         t0_s: float, vehicles: Optional[set] = None,
                         slots=None) -> dict:
    """{vid: [1, 1, T, 4]} (t_s, s_m, lane_id, speed) with NaN after exit.

    ``slots`` (default: the rollout's t0 slots) may add tracked slots that are
    outside the rollout world (junction rows): all-NaN motion columns."""
    T = min(int(n_window), roll.pos.shape[0] - 1) + 1
    alive = alive_mask(roll, T)[:, b]                    # [T, K]
    t = t0_s + DT_S * np.arange(T)
    out = {}
    if slots is None:
        slots = np.flatnonzero(roll.msk[0, b])
    for s in slots:
        vid = slot_vid[int(s)]
        if vid is None or (vehicles is not None and vid not in vehicles):
            continue
        a = alive[:, s]
        arr = np.full((T, 4), np.nan)
        arr[:, 0] = t
        arr[a, 1] = roll.pos[:T, b, s][a]
        arr[a, 2] = roll.lns[:T, b, s][a]
        arr[a, 3] = roll.spd[:T, b, s][a]
        out[vid] = arr[None, None]
    return out


# ------------------------------------------------------------ predictor
@dataclass
class LiveSample:
    """Planner state sample wrapping a LiveState (contract section 3.1)."""

    hybrid: Any                      # closedloop.live_state.LiveState
    arrivals: Any = None
    meta: Mapping[str, Any] = field(default_factory=dict)
    frames: Optional[Mapping[str, float]] = None
    frames_distance_to_merge_m: Optional[Mapping[str, float]] = None


def _level(q, level: float) -> float:
    """The quantile at ``level`` on the model's 0.05..0.95 grid."""
    q = np.asarray(q, float)
    levels = np.round(np.arange(1, q.size + 1) / (q.size + 1), 4)
    idx = np.flatnonzero(np.isclose(levels, level))
    if idx.size == 1:
        return float(q[idx[0]])
    return float(np.interp(level, levels, q))


def _pair_record(pair, cav_ids):
    """T-B (#7) min-TTC pair record: follower/leader ids + pair class."""
    if not pair:
        return None
    follower, leader = pair
    cls = "cav_involved" if (follower in cav_ids or leader in cav_ids) else "hdv_hdv"
    return {"follower_id": follower, "leader_id": leader, "pair_class": cls}


def _cell(x, dtype=float):
    return np.full((1, 1), x, dtype=dtype)


class A11Predictor:
    """The planner WorldModel backed by A11 member_1 (CPU)."""

    contract_version = CONTRACT

    def __init__(self, bundle, *, tau_s: float = DECISION_TAU_S,
                 plan_horizon_s: float = PLAN_HORIZON_S,
                 native_reference: str = "zero",
                 wall_codes: Optional[Callable] = None,
                 rollout_fn: Callable = batched_rollout,
                 head_fn: Callable = head_dv_quantiles,
                 max_batch: int = 64,
                 track_vehicles: str = "all",
                 fast: bool = True,
                 skip_identical: bool = True,
                 device: Optional[str] = None,
                 model_sha256: Optional[str] = None,
                 physical_exits: bool = False,
                 open_entries: bool = False,
                 open_road: bool = False,
                 sumo_fidelity: bool = False,
                 sumo_speed_factor: Optional[float] = None,
                 sumo_idm: bool = True,
                 merge_cal: bool = False,
                 merge_cal_params=None):
        if native_reference not in ("zero", "head"):
            raise A11PredictorError("native_reference must be 'zero' or 'head'")
        self.device = str(device or default_device())
        if self.device != "cpu" and not fast:
            raise A11PredictorError("a non-CPU device requires the fast HDV path")
        if fast:
            # exact inference fast paths (closedloop.fast_hdv; parity-tested);
            # device != cpu runs the HDV residual net there (the numpy
            # rollout engine itself stays on the CPU)
            from .fast_hdv import fast_bundle, install_fast_gap_projection

            bundle = fast_bundle(bundle, device=self.device)
            install_fast_gap_projection()
        self.model_sha256 = str(model_sha256 or member_sha256())
        self.bundle = bundle
        self.fast = bool(fast)
        self.tau_s = float(tau_s)
        self.plan_horizon_s = float(plan_horizon_s)
        self.native_reference = native_reference
        self.wall_codes = wall_codes
        self.rollout_fn = rollout_fn
        self.head_fn = head_fn
        self.max_batch = int(max_batch)
        self.open_entries = bool(open_entries or open_road)
        self.track_vehicles = track_vehicles
        self.physical_exits = bool(physical_exits or open_road)
        self.sumo_fidelity = bool(sumo_fidelity)
        self.sumo_speed_factor = sumo_speed_factor
        self.sumo_idm = bool(sumo_idm)
        self.merge_cal = bool(merge_cal)
        self.merge_cal_params = merge_cal_params
        self._tau_i = tau_index(self.tau_s) if head_fn is head_dv_quantiles else None
        self.calls: list = []        # per predict_many timing log
        self.skip_identical = bool(skip_identical)
        self._cache_key = None
        self._roll_cache: dict = {}
        self._ident_cache: dict = {}

    def _identical_reference(self, state, codes, ws, n_steps):
        """A cached traced rollout this single-CAV candidate provably equals.

        Reference = a cached row differing in exactly one CAV slot that is
        native in the reference and damping/PI here (no walls); the
        candidate reuses it iff :func:`closedloop.skip.identical_slots`
        proves the commanded execution never changes that CAV's speed.
        """
        from .skip import identical_slots

        if np.isfinite(ws).any():
            return None
        for ref_key, (ref_src, ref_codes, ref_ws) in list(self._roll_cache.items()):
            roll, b = ref_src
            if getattr(roll, "idm_raw", None) is None or np.isfinite(ref_ws).any():
                continue
            diff = np.flatnonzero(ref_codes != codes)
            if diff.size != 1:
                continue
            i = int(diff[0])
            if ref_codes[i] != 0 or codes[i] not in (1, 2):
                continue
            ck = (ref_key, int(codes[i]))
            if ck not in self._ident_cache:
                self._ident_cache[ck] = identical_slots(
                    self.bundle, roll, b, ref_codes, int(codes[i]),
                    np.asarray(state.is_cav, bool) & _rollout_mask(state),
                    float(state.time_s), int(n_steps))
            if bool(self._ident_cache[ck][i]):
                return ref_src
        return None

    # -- plan -> per-slot codes
    def plan_codes(self, state, plan) -> tuple:
        cav_slots = state.cav_slots
        K = len(state.slot_vid)
        codes = np.full(K, NONE_CODE, np.int64)
        for s in cav_slots.values():
            codes[s] = PLANNER_ACTION_CODES["native"]
        wall_speed = np.full(K, np.nan)
        unknown = [c for c in plan.per_cav if c not in cav_slots]
        if unknown:
            raise A11PredictorError(
                f"plan assigns CAVs absent from the decision state: {sorted(unknown)[:5]}")
        for cav, action in plan.per_cav.items():
            if action not in PLANNER_ACTION_CODES:
                raise A11PredictorError(f"unsupported action {action!r}")
            codes[cav_slots[cav]] = PLANNER_ACTION_CODES[action]
        if plan.walls:
            if self.wall_codes is None:
                raise A11PredictorError("walls are disabled in the pl01_a11 policy (fail closed)")
            codes, wall_speed = self.wall_codes(state, plan, codes, wall_speed)
        # junction-lane CAVs are outside the rollout world (LiveState.rollout_mask)
        off = ~(np.asarray(state.is_cav, bool) & _rollout_mask(state))
        codes[off] = NONE_CODE
        wall_speed[off] = np.nan
        return codes, wall_speed

    def _native_plan_codes(self, state):
        K = len(state.slot_vid)
        codes = np.full(K, NONE_CODE, np.int64)
        for s in state.cav_slots.values():
            codes[s] = PLANNER_ACTION_CODES["native"]
        codes[~(np.asarray(state.is_cav, bool) & _rollout_mask(state))] = NONE_CODE
        return codes

    def predict(self, state_samples, plan, controller_state=None, **kw):
        return self.predict_many(state_samples, [plan], controller_state=controller_state, **kw)[0]

    def predict_many(self, state_samples: Sequence, plans: Sequence, *,
                     controller_state=None, horizon_s: float = 600.0,
                     trajectory_s: Optional[float] = 60.0, commit_s: float = 30.0,
                     continuation: str = "planner") -> list:
        samples = list(state_samples)
        if len(samples) != 1:
            raise A11PredictorError("A11 adapter takes exactly one state sample (K = 1)")
        state = samples[0].hybrid
        plans = list(plans)
        if not plans:
            return []
        t_call = time.perf_counter()
        window_s = 60.0 if trajectory_s is None else float(trajectory_s)
        n_window = max(1, int(round(window_s / DT_S)))
        n_steps = max(int(round(self.plan_horizon_s / DT_S)), n_window)
        # distinct code rows: native reference first
        native_codes = self._native_plan_codes(state)
        native_ws = np.full(len(state.slot_vid), np.nan)

        def row_key(codes, ws):
            return codes.tobytes() + np.nan_to_num(ws, nan=-1.0).tobytes()

        rows = [native_codes]
        wall_rows = [native_ws]
        keys = {row_key(native_codes, native_ws): 0}
        plan_row = []
        for plan in plans:
            codes, ws = self.plan_codes(state, plan)
            key = row_key(codes, ws)
            if key not in keys:
                keys[key] = len(rows)
                rows.append(codes)
                wall_rows.append(ws)
            plan_row.append(keys[key])
        codes_all = np.stack(rows)
        ws_all = np.stack(wall_rows)
        # per-decision-state rollout cache (+ exact identical-candidate reuse)
        st_key = (id(state), float(state.time_s), int(n_steps), int(n_window))
        if self._cache_key != st_key:
            self._cache_key = st_key
            self._roll_cache = {}
            self._ident_cache = {}
        src = [None] * len(rows)
        need = []
        n_cached = n_skipped = 0
        t_skip0 = time.perf_counter()
        for i in range(len(rows)):
            key = row_key(rows[i], wall_rows[i])
            if key in self._roll_cache:
                src[i] = self._roll_cache[key][0]
                n_cached += 1
                continue
            alias = (self._identical_reference(state, rows[i], wall_rows[i], n_steps)
                     if self.skip_identical else None)
            if alias is not None:
                src[i] = alias
                self._roll_cache[key] = (alias, rows[i], wall_rows[i])
                n_skipped += 1
                continue
            need.append(i)
        t_skip = time.perf_counter() - t_skip0
        t_roll = 0.0
        traced = self.skip_identical and self.rollout_fn is batched_rollout
        for lo in range(0, len(need), self.max_batch):
            idxs = need[lo:lo + self.max_batch]
            kw = {"trace": True} if traced else {}
            if np.isfinite(ws_all[idxs]).any():
                kw["wall_speed"] = ws_all[idxs]
            if self.open_entries:
                kw["open_entries"] = True
            if self.physical_exits:
                kw["physical_exits"] = True
            if self.sumo_fidelity:
                kw["sumo_fidelity"] = True
                kw["sumo_idm"] = self.sumo_idm
                if self.sumo_speed_factor is not None:
                    kw["sumo_speed_factor"] = self.sumo_speed_factor
            if self.merge_cal:
                kw["merge_cal"] = True
                kw["merge_cal_params"] = self.merge_cal_params
            r = self.rollout_fn(self.bundle, state, codes_all[idxs], n_steps=n_steps,
                                t0_s=float(state.time_s), **kw)
            t_roll += float(getattr(r, "seconds", 0.0))
            for j, i in enumerate(idxs):
                src[i] = (r, j)
                self._roll_cache[row_key(rows[i], wall_rows[i])] = ((r, j), rows[i], wall_rows[i])
        t_head0 = time.perf_counter()
        is_cav = np.asarray(state.is_cav, bool)
        dv_rows = np.zeros((len(rows), ))
        q90_rows = np.zeros((len(rows), ))
        q_rows = [None] * len(rows)
        pls = [long_plan_from_rollout(r.at(b), is_cav, self.plan_horizon_s) for r, b in src]
        q = np.asarray(self.head_fn(self.bundle, state, torch.cat(pls, dim=0)), float)
        ti = self._tau_i if self._tau_i is not None else tau_index(self.tau_s)
        qt = q[:, ti, :]
        for j in range(qt.shape[0]):
            dv_rows[j] = float(np.mean(qt[j]))
            q90_rows[j] = _level(qt[j], 0.9)
            q_rows[j] = qt[j].tolist()
        t_head = time.perf_counter() - t_head0
        head_native = float(dv_rows[0])
        q90_native = float(q90_rows[0])

        def rel(i):
            if i == 0:
                return 0.0, 0.0
            if self.native_reference == "head":
                return dv_rows[i] - head_native, q90_rows[i] - q90_native
            return dv_rows[i], q90_rows[i]

        def roll_of(i):
            return src[i]

        stats_cache = {}
        traj_cache = {}
        vehicles = None if self.track_vehicles == "all" else set(state.cav_slots)

        def stats(i):
            if i not in stats_cache:
                r, b = roll_of(i)
                stats_cache[i] = safety_stats(r, b, n_window, state.slot_vid)
                traj_cache[i] = tracked_trajectories(
                    r, b, n_window, state.slot_vid, float(state.time_s), vehicles,
                    slots=np.flatnonzero(np.asarray(state.veh_mask, bool)))
            return stats_cache[i]

        nat = stats(0)
        cav_ids = set(state.cav_slots)
        out = []
        for plan, i in zip(plans, plan_row):
            s = stats(i)
            dv, q90 = rel(i)
            meta = {
                "contract_version": CONTRACT,
                "predictor": "A11_member_1",
                "model_sha256": self.model_sha256,
                "device": self.device,
                "n_junction_excluded": int((np.asarray(state.veh_mask, bool)
                                            & ~_rollout_mask(state)).sum()),
                "decision_tau_s": self.tau_s,
                "dv_mean_tau": dv,
                "dv_q90_tau": q90,
                "dv_quantiles_tau": q_rows[i],
                "head_raw_dv_mean": float(dv_rows[i]),
                "head_native_dv_mean": head_native,
                "native_reference": self.native_reference,
                "min_ttc_pairs": [_pair_record(s.min_ttc_pair, cav_ids)],
                "min_gap_m": s.min_gap_m,
                "window_s": window_s,
                "plan_horizon_s": self.plan_horizon_s,
                "part_a_scope": "tracked",
                "part_a_model": "A11_rollout",
            }
            out.append(_prediction(dv, s, nat, traj_cache[i], traj_cache[0], meta))
        self.calls.append({
            "n_plans": len(plans), "n_rows": len(rows), "n_rollouts": len(need),
            "n_cached": n_cached, "n_identical_reused": n_skipped, "n_steps": n_steps,
            "skip_check_s": round(t_skip, 3),
            "rollout_s": round(t_roll, 3), "head_s": round(t_head, 3),
            "total_s": round(time.perf_counter() - t_call, 3)})
        return out


def _prediction(dv, s: SafetyStats, nat: SafetyStats, traj, native_traj, meta):
    from traffic_shepherd.planner.interface import Prediction

    return Prediction(
        delay_vs_native_s=_cell(dv),
        plan_delay_s=_cell(dv),
        native_delay_s=_cell(0.0),
        delay_part_a_s=_cell(dv),
        delay_part_b_s=_cell(0.0),
        collision=_cell(s.collision, bool),
        min_ttc_s=_cell(s.min_ttc_s),
        ttc_lt3_exposure_s=_cell(s.ttc_lt3_s),
        native_ttc_lt3_exposure_s=_cell(nat.ttc_lt3_s),
        hard_brake_s=_cell(s.hard_brake_s),
        native_hard_brake_s=_cell(nat.hard_brake_s),
        native_collision=_cell(nat.collision, bool),
        native_min_ttc_s=_cell(nat.min_ttc_s),
        trajectories_tracked=traj,
        native_trajectories_tracked=native_traj,
        meta=meta,
        cell_valid=_cell(True, bool),
        native_cell_valid=_cell(True, bool),
        has_closing_pair=_cell(s.has_closing_pair, bool),
        native_has_closing_pair=_cell(nat.has_closing_pair, bool),
        ttc_lt1p5_exposure_s=_cell(s.ttc_lt1p5_s),
        native_ttc_lt1p5_exposure_s=_cell(nat.ttc_lt1p5_s),
    )


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
