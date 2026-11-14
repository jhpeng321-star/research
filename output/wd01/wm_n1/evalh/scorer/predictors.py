"""E7 integration glue: the REAL per-item predictors of the CAL-tune scorer.

This module turns a released CAL item into the exact t0 :class:`~rollout2.
state.WorldState` training/evaluation build from packed arrays, applies one of
two explicit CAV-motion modes, and runs the trained model A outcome head (or
the Part A v2g baseline) once per item, reusing the rollout across that item's
taus.

Everything is fail-closed (R8): a missing schedule, a missing A6 entrance
counter, a missing shard array or a teacher window that escapes the run raises
a named exception -- never a native/zero fallback.

World builder (REUSED, never duplicated)
----------------------------------------
The t0 world is built through :func:`rollout2.engine.world_from_batch`
(``rollout2/engine.py:214-262``), the same builder the evaluation batch path
uses, from the pack arrays ``veh_kin``/``veh_mask``/``veh_step`` at the item's
t0 tick (``run`` INDEX ``tick_start`` offset); slot identity is
``slot_index[t0]`` and the CAV roster is ``veh_step[t0, :, IS_CAV_IDX]``
(``features.constants.IS_CAV_IDX = 7``), exactly as ``train/idea_a.py:977-988``
reads the canonical window.  The A6 t0 entrance counter is
``counter.npy[t0]`` (``[2,30]``) exactly as ``train/idea_a.py`` feeds
``StateGlobalToken(kin, kin_mask, counter)`` (``train/idea_a.py:1772-1778``);
its absence refuses -- there is no counter-zero fallback.

CAV-motion modes (an explicit, REQUIRED owner decision)
-------------------------------------------------------
``controller`` -- per-CAV action codes decoded from the packed plan tensor
``plan.npy[plan_row]`` (``[K, N_SEGMENTS, PLAN_F]``), expanded to the per-tick
``[n_steps, K, 8]`` schedule exactly on the featurizer's segment timing
(``features/featurizer.py:336-362``: ``seg_t = t_dec + SEGMENT_S*seg``) and
applied through :func:`evalh.scorer.actions.rollout_with_schedule`.  A missing
``plan.npy`` or a non-one-hot CAV action refuses.

``teacher`` -- the arm's recorded CAV kinematics ``veh_kin`` over the window
are teacher-forced exactly as training (``train/idea_a.py:1086-1096`` build the
``[T+1,B,K,4]`` tensor and ``train/idea_a.py:1935-1975`` replaces the CAV state
after every step); the recorded CAV columns are restored on the returned
rollout so the head sees the same CAV motion the trainer summarized.  In this
mode the head INPUT (contract 3) is NOT the executed-rollout trajectory: the
correction ``summary_raw`` is rebuilt EXACTLY as training
(``train/idea_a.py:1055-1085``) from the pinned ``cav_plan_ch`` speed/accel
channels and the recorded positions, alive-since-t0 forward-filled over
``model.constants.CORRECTION_WINDOW_STEPS`` steps, then run through the same
trained projection/head with the A6 state token
(:func:`training_summary_raw`, :func:`head_quantiles_from_summary`).  The
predictor records ``summary_source`` accordingly
(:data:`SUMMARY_SOURCE_TRAINING_PACKED` vs
:data:`SUMMARY_SOURCE_ROLLOUT_EXECUTED`).

Model A runs ``model.constants.CORRECTION_WINDOW_STEPS`` steps with integrator
``d1`` and the HDV residual, then the trained correction/outcome head through
:func:`wm_api.engine.outcome_head`; each row's own-tau quantiles are selected
with :func:`evalh.idea_a.delay_quantiles_from_head`.  The optional
``--dynamics-extrapolation`` flag runs a longer rollout reported separately
(labelled ``extrapolation``), never feeding the head.

The v2g predictor is :class:`evalh.scorer.comparator_v2g.V2GBaselinePredictor`
with an explicit owner ``option``; ``trajectory_window_delay`` rolls the v2g
baseline for ``tau`` steps per row under the same CAV-motion mode, and the
unimplemented ``hybrid_correction_head`` keeps its ``ComparatorUnavailable``.
"""
from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, replace
from typing import Optional, Sequence

import numpy as np
import torch

from evalh.protocol import N_QUANTILES, QUANTILE_LEVELS
from features.constants import (N_SEGMENTS, PLAN_F, SEGMENT_S, IS_CAV_IDX,
                                VEH_F)
from model.constants import CORRECTION_WINDOW_STEPS
from evalh.real_source import _item_key

DT_S = 0.5

#: The two explicit CAV-motion modes (no default; the owner selects one).
CAV_MOTION_CONTROLLER = "controller"
CAV_MOTION_TEACHER = "teacher"
CAV_MOTIONS = (CAV_MOTION_CONTROLLER, CAV_MOTION_TEACHER)

#: Arrays the world/teacher builder reads that E3's ``planned_reads`` did not
#: yet list (declared in ``labels._EXTRA_INPUT_ARRAYS``).
WORLD_ARRAYS = ("veh_kin", "veh_mask", "veh_step", "slot_index", "counter")

#: The two recorded correction-summary provenances (contract 3 / R8).  Teacher
#: mode rebuilds ``summary_raw`` from the pinned packed planned CAV channels;
#: controller mode reads the executed-rollout trajectory summary.
SUMMARY_SOURCE_TRAINING_PACKED = "training_packed_plan"
SUMMARY_SOURCE_ROLLOUT_EXECUTED = "rollout_executed"


class PredictorRefusal(RuntimeError):
    """A real per-item predictor input is absent or inconsistent (fail closed)."""


def _plan_aware_head_helper():
    """The optional plan-aware paired-effect entry point, or ``None``.

    A parallel WM-N1 task adds
    ``wm_api.eval.paired_effect_quantiles_for_plan(bundle, *, t0_kin,
    t0_mask, t0_counter, summary_raw)`` with the fixed interface returning a
    float64 ``[N_TAU, N_QUANTILES]`` dV of the arm's plan vs native.  The
    import is guarded, but its presence is NOT a capability probe: whether a
    bundle is plan-aware is decided from its head in :func:`_bundle_plan_aware`.
    """
    try:
        from wm_api.eval import paired_effect_quantiles_for_plan
    except (ImportError, AttributeError):
        return None
    return paired_effect_quantiles_for_plan


def _bundle_plan_aware(bundle) -> bool:
    """Whether the bundle's paired-effect head consumes the arm's plan.

    Decided from the HEAD itself (``head.plan_aware`` / presence of
    ``plan_proj``), falling back to the recorded training config
    ``paired_effect_plan_aware`` -- NEVER from the availability of the
    ``wm_api.eval.paired_effect_quantiles_for_plan`` helper.  The helper ships
    in this checkout regardless of which checkpoint is loaded, so probing it
    misclassified the A8 **state-only** head as plan-aware (Fix 2a, 2026-10-05).
    A state-only head uses the state-only path; sibling arms of one decision
    then share the identical t0-state dV, i.e. A8's decision quality is
    degenerate by construction.
    """
    head = getattr(bundle, "paired_effect", None)
    if head is not None:
        flag = getattr(head, "plan_aware", None)
        if flag is not None:
            return bool(flag)
        if getattr(head, "plan_proj", None) is not None:
            return True
    cfg = getattr(bundle, "config", None)
    return bool(getattr(cfg, "paired_effect_plan_aware", False))


def _tau_index(tau_s) -> int:
    from model import constants as MC

    taus = [round(float(t), 6) for t in MC.TAUS]
    return taus.index(round(float(tau_s), 6))


def _row_arm_token(row) -> str:
    """Canonical controller arm token of one scorer row (policy/arm_kind)."""
    from evalh.idea_a import canon_action

    kind = getattr(row, "policy", None)
    if kind is None or not str(kind).strip():
        kind = getattr(row, "arm_kind", "")
    token = str(kind or "").strip().lower().split("_")[0]
    return canon_action(token)


def _contrast_treatment(contrast) -> str:
    """Treatment arm of a ``"<arm>-native"`` contrast (canonical)."""
    from evalh.idea_a import canon_action

    token = str(contrast or "").split("-")[0].strip().lower()
    return canon_action(token)


@dataclass(frozen=True)
class ItemWorld:
    """The t0 world of one selected CAL item plus its A6 channels."""

    item: dict
    run: dict
    local_t0_tick: int
    world: object                     # rollout2.state.WorldState, batch 1
    counter_t0: np.ndarray            # [2,30] float64 entrance counters at t0
    is_cav_t0: np.ndarray             # [K] bool CAV roster at t0


def _entry(source, position) -> tuple:
    """Resolve a row's ``pack_item_position`` (the pack MANIFEST position that
    the label reader records, i.e. entry[0]) to its selected entry."""
    try:
        position = int(position)
    except (TypeError, ValueError) as exc:
        raise PredictorRefusal(
            f"row does not resolve to a selected CAL item: {exc}") from exc
    matches = [entry for entry in source.selected if int(entry[0]) == position]
    if len(matches) != 1:
        raise PredictorRefusal(
            f"row does not resolve to exactly one selected CAL item "
            f"(manifest position {position}, {len(matches)} matches)")
    return matches[0]


def _array(source, item, name):
    try:
        return np.load(source._checked_file(item, name), mmap_mode="r")
    except Exception as exc:                             # noqa: BLE001
        raise PredictorRefusal(
            f"selected shard {_item_key(item)} lacks pinned {name}: "
            f"{type(exc).__name__}: {exc}") from exc


def build_item_world(source, item, run) -> ItemWorld:
    """Build the exact t0 :class:`WorldState` training/evaluation build.

    Reuses :func:`rollout2.engine.world_from_batch` (the evaluation batch
    builder) and requires the A6 t0 entrance counter (no zero fallback).
    """
    kin = _array(source, item, "veh_kin.npy")
    mask = _array(source, item, "veh_mask.npy")
    step = _array(source, item, "veh_step.npy")
    slot = _array(source, item, "slot_index.npy")
    counter = _array(source, item, "counter.npy")
    tick = int(item["tick"]) - int(run["tick_start"])
    n_ticks = int(run["n_ticks"])
    if not 0 <= tick < n_ticks:
        raise PredictorRefusal(
            f"item t0 tick {item['tick']} escapes run "
            f"[{run['tick_start']}, {run['tick_start'] + n_ticks})")
    if (kin.ndim != 3 or kin.shape[-1] != 7 or mask.shape != kin.shape[:2]
            or step.shape[:2] != kin.shape[:2] or step.shape[-1] != VEH_F
            or slot.shape != kin.shape[:2] or counter.shape != (n_ticks, 2, 30)
            or any(len(a) != n_ticks for a in (kin, mask, step, slot, counter))):
        raise PredictorRefusal(
            "selected shard arrays disagree with the run length/shape "
            "(veh_kin[7]/veh_mask/veh_step[16]/slot_index/counter[2,30])")
    counter_t0 = np.asarray(counter[tick], dtype=np.float64)
    if counter_t0.shape != (2, 30) or not np.isfinite(counter_t0).all():
        raise PredictorRefusal(
            "A6 t0 entrance counter is absent/nonfinite; refusing the "
            "counter-zero fallback")
    from rollout2.engine import world_from_batch

    batch = {
        "veh_kin": torch.as_tensor(
            np.array(kin[tick], dtype=np.float32)).unsqueeze(0),
        "veh_mask": torch.as_tensor(
            np.array(mask[tick] > 0)).unsqueeze(0),
        "is_cav": torch.as_tensor(
            np.array(step[tick, :, IS_CAV_IDX] > 0.5)).unsqueeze(0),
        "slot_id": torch.as_tensor(
            np.array(slot[tick], dtype=np.int64)).unsqueeze(0),
    }
    world = world_from_batch(batch)
    is_cav_t0 = world.is_cav.detach().to(torch.bool).cpu().numpy().reshape(-1)
    return ItemWorld(item=item, run=run, local_t0_tick=tick, world=world,
                     counter_t0=counter_t0, is_cav_t0=is_cav_t0)


def expand_plan_to_ticks(plan, *, n_steps, n_segments=N_SEGMENTS):
    """Expand a ``[K, N_SEGMENTS, PLAN_F]`` plan slice to ``[n_steps, K, F]``.

    Tick ``u`` (from t0) belongs to segment ``floor(u*DT/SEGMENT_S)`` on the
    featurizer's timing (``features/featurizer.py:336-362``), capped at the
    last segment.  The columns are the frozen ``PLAN_FEATURE_NAMES`` layout.
    """
    plan = np.asarray(plan)
    if plan.ndim != 3 or plan.shape[-1] != PLAN_F:
        raise PredictorRefusal(
            f"plan slice must be [K, n_segments, {PLAN_F}]; got "
            f"{tuple(plan.shape)}")
    n_seg = min(int(plan.shape[1]), int(n_segments))
    if n_seg < 1:
        raise PredictorRefusal("plan slice has no segments")
    u = np.arange(int(n_steps), dtype=np.int64)
    seg_idx = np.minimum((u * DT_S // SEGMENT_S).astype(np.int64), n_seg - 1)
    return np.ascontiguousarray(np.transpose(plan[:, seg_idx, :], (1, 0, 2)))


def build_schedule(source, item, run, *, n_steps, is_cav_t0):
    """The validated packed per-CAV schedule for one item (controller mode)."""
    from evalh.scorer.actions import MissingActionSchedule, schedule_from_packed

    try:
        plan = np.load(source._checked_file(item, "plan.npy"), mmap_mode="r")
    except Exception as exc:                             # noqa: BLE001
        raise MissingActionSchedule(
            f"selected shard {_item_key(item)} lacks the pinned plan.npy "
            f"packed per-CAV action schedule: {type(exc).__name__}: {exc}"
        ) from exc
    if plan.ndim != 4 or plan.shape[-1] != PLAN_F:
        raise MissingActionSchedule(
            f"plan.npy must be [N_DECISION, K, N_SEGMENTS, {PLAN_F}]; got "
            f"{tuple(plan.shape)}")
    row = int(item["plan_row"])
    if not 0 <= row < plan.shape[0]:
        raise MissingActionSchedule(
            f"plan_row {row} escapes the plan array ({plan.shape[0]})")
    try:
        expanded = expand_plan_to_ticks(plan[row], n_steps=n_steps)
    except PredictorRefusal as exc:
        raise MissingActionSchedule(str(exc)) from exc
    return schedule_from_packed(
        expanded, run_key=_item_key(item), local_t0_tick=0,
        n_steps=int(n_steps), is_cav_t0=is_cav_t0)


def build_teacher(source, item, run, *, n_steps):
    """Recorded CAV kinematics ``(seq [n_steps+1,K,7], present [n_steps+1,K])``.

    Mirrors ``train/idea_a.py:1035-1096``: the same ``veh_kin`` window, with
    the ``(s, lane, v, a)`` teacher columns passed to the P3 engine's
    training-only teacher path.
    """
    kin = _array(source, item, "veh_kin.npy")
    mask = _array(source, item, "veh_mask.npy")
    tick = int(item["tick"]) - int(run["tick_start"])
    end = tick + int(n_steps) + 1
    if end > kin.shape[0]:
        raise PredictorRefusal(
            f"recorded teacher window needs ticks [{tick}, {end}) but the "
            f"run has {kin.shape[0]}; refusing to extrapolate the recording")
    seq = np.asarray(kin[tick:end], dtype=np.float64)
    seq_mask = np.asarray(mask[tick:end]).astype(bool)
    lane_oh = seq[..., 1:4]
    has_lane = lane_oh.sum(axis=-1) > 0
    lane = np.where(has_lane, lane_oh.argmax(axis=-1), -1)
    return seq, seq_mask, lane


def _teacher_tensor(seq, seq_mask, lane, *, n_steps):
    """``[n_steps+1, 1, K, 4]`` teacher tensor (train/idea_a.py:1086-1096)."""
    k_n = int(seq.shape[1])
    t = np.full((int(n_steps) + 1, 1, k_n, 4), np.nan, dtype=np.float64)
    finite = seq_mask & np.isfinite(seq[..., 0]) & np.isfinite(seq[..., 4])
    t[:, 0, :, 0] = np.where(finite, seq[..., 0], np.nan)     # s
    t[:, 0, :, 1] = np.where(finite, lane, -1)                # lane
    t[:, 0, :, 2] = np.where(finite, seq[..., 4], np.nan)     # v
    t[:, 0, :, 3] = np.where(finite, seq[..., 5], np.nan)     # a (kin col 5)
    return torch.as_tensor(t)


def _training_window_summary(kin_seq, plan_seq, present0, tm, is_cav):
    """The exact ``train/idea_a.py:1055-1085`` alive forward-fill summary.

    ``present0``/``tm`` build the alive-since-t0 mask (a reused slot never
    becomes alive again), the recorded position and packed plan channels are
    forward-filled on it, and the FROZEN training helper
    :func:`train.idea_a._cav_motion_summary` builds the 16 raw features.
    """
    alive_state = torch.cumprod(
        torch.cat([present0.unsqueeze(0).to(torch.int64),
                   tm.to(torch.int64)], dim=0),
        dim=0).to(torch.bool)
    t_steps = int(alive_state.shape[0])
    ar = torch.arange(t_steps).view(t_steps, 1)
    fill = torch.cummax(
        torch.where(alive_state, ar, torch.zeros_like(ar)),
        dim=0).values
    pos_ff = kin_seq[..., 0].gather(0, fill)
    plan_ff = plan_seq.gather(
        0, fill.unsqueeze(-1).expand(-1, -1, plan_seq.shape[-1]))
    cav_alive = (is_cav.unsqueeze(0) & alive_state).unsqueeze(1)
    from train.idea_a import _cav_motion_summary

    return _cav_motion_summary(pos_ff.unsqueeze(1), plan_ff.unsqueeze(1),
                               cav_alive)


def training_summary_raw(source, item, run, *, n_steps, is_cav):
    """Rebuild ``summary_raw`` EXACTLY as training (contract 3, teacher mode).

    Training builds the correction head input in ``train/idea_a.py:1031-1085``
    over the 20-step window from the pinned ``cav_plan_ch`` speed/accel
    channels (de-scaled inside :func:`train.idea_a._cav_motion_summary` by
    ``features.idea_a.CAV_TGT_SPEED_SCALE`` / ``CAV_ACCEL_SCALE``) and the
    recorded positions.  This reads the same pinned shard arrays through
    :meth:`ReleasedCalSource._checked_file` and reproduces that window via
    :func:`_training_window_summary`; a missing ``cav_plan_ch.npy`` refuses.
    """
    tick = int(item["tick"]) - int(run["tick_start"])
    n_steps = int(n_steps)
    if tick < 0:
        raise PredictorRefusal(
            f"item t0 tick {item['tick']} precedes run {run['tick_start']}")
    kin = _array(source, item, "veh_kin.npy")
    mask = _array(source, item, "veh_mask.npy")
    target = _array(source, item, "target_mask.npy")
    plan = _array(source, item, "cav_plan_ch.npy")
    stop = tick + n_steps + 1
    if (plan.ndim != 3 or plan.shape[:2] != kin.shape[:2]
            or target.shape != kin.shape[:2]):
        raise PredictorRefusal(
            f"pinned cav_plan_ch {tuple(plan.shape)} / target_mask "
            f"{tuple(target.shape)} disagree with veh_kin {tuple(kin.shape)}")
    if (stop > kin.shape[0] or stop > plan.shape[0]
            or tick + n_steps > target.shape[0]):
        raise PredictorRefusal(
            f"training summary window needs ticks [{tick}, {stop}) plus "
            f"{n_steps} target_mask steps but the shard has veh_kin="
            f"{kin.shape[0]}/cav_plan_ch={plan.shape[0]}/target_mask="
            f"{target.shape[0]}")
    kin_seq = torch.as_tensor(np.array(kin[tick:stop], np.float32))
    plan_seq = torch.as_tensor(np.array(plan[tick:stop], np.float32))
    if plan_seq.shape[-1] < 2:
        raise PredictorRefusal(
            "pinned cav_plan_ch has no speed/accel channels (fields 0/1)")
    present0 = torch.as_tensor(np.asarray(mask[tick]).astype(bool))
    tm = torch.as_tensor(
        np.asarray(target[tick:tick + n_steps], bool)).to(torch.bool)
    is_cav_t = torch.as_tensor(np.asarray(is_cav).astype(bool))
    return _training_window_summary(kin_seq, plan_seq, present0, tm, is_cav_t)


def _bundle_plan_horizon_s(bundle):
    """The checkpoint's A11 full-horizon plan length ``H`` (seconds), or None.

    ``None`` means the bundle uses the historical 10 s / 20-step
    ``summary_raw`` plan.  A recorded non-finite/≤0 value refuses (fail closed).
    """
    cfg = getattr(bundle, "config", None)
    h = getattr(cfg, "paired_effect_plan_horizon_s", None)
    if h is None:
        return None
    try:
        h = float(h)
    except (TypeError, ValueError) as exc:
        raise PredictorRefusal(
            f"paired_effect_plan_horizon_s is not a number: {h!r}") from exc
    if not math.isfinite(h) or h <= 0.0:
        raise PredictorRefusal(
            f"paired_effect_plan_horizon_s must be finite and > 0; got {h!r}")
    return h


def training_plan_long(source, item, run, *, horizon_s, is_cav):
    """Rebuild the A11 full-horizon plan vector EXACTLY as training (contract 3).

    Training's long-plan block (``train/idea_a.py``, A11) reads the pinned
    ``cav_plan_ch`` speed/accel channels (fields 0/1) over ``[t0, t0 + H]``,
    de-scales them by ``features.idea_a.CAV_TGT_SPEED_SCALE`` /
    ``CAV_ACCEL_SCALE``, builds the alive-since-t0 mask from ``veh_mask[t0]`` and
    ``target_mask`` and bins the result through
    :func:`model.paired_effect.long_plan_summary` (the SAME function training
    uses).  Missing/mismatched pinned channels, a t0 before the run, or a plan
    escaping the shard refuse (fail closed).  Returns a float32 ``[1, W]``
    torch tensor with ``W = long_plan_raw_f(H)``.
    """
    from features.idea_a import CAV_ACCEL_SCALE, CAV_TGT_SPEED_SCALE
    from model.paired_effect import long_plan_summary

    h = float(horizon_s)
    if not math.isfinite(h) or h <= 0.0:
        raise PredictorRefusal(
            f"plan horizon must be finite and > 0; got {horizon_s!r}")
    tick = int(item["tick"]) - int(run["tick_start"])
    if tick < 0:
        raise PredictorRefusal(
            f"item t0 tick {item['tick']} precedes run {run['tick_start']}")
    kin = _array(source, item, "veh_kin.npy")
    mask = _array(source, item, "veh_mask.npy")
    target = _array(source, item, "target_mask.npy")
    plan = _array(source, item, "cav_plan_ch.npy")
    if (plan.ndim != 3 or plan.shape[:2] != kin.shape[:2]
            or target.shape != kin.shape[:2]):
        raise PredictorRefusal(
            f"pinned cav_plan_ch {tuple(plan.shape)} / target_mask "
            f"{tuple(target.shape)} disagree with veh_kin {tuple(kin.shape)}")
    if plan.shape[-1] < 2:
        raise PredictorRefusal(
            "pinned cav_plan_ch has no speed/accel channels (fields 0/1)")
    end = int(kin.shape[0])
    n_req = max(1, int(round(h / DT_S)))
    n_long = max(0, min(n_req, end - tick))
    if n_long >= 1 and tick + n_long > plan.shape[0]:
        raise PredictorRefusal(
            f"full-horizon plan window needs ticks [{tick}, {tick + n_long}) "
            f"but the pinned cav_plan_ch has {plan.shape[0]}")
    present0 = torch.as_tensor(np.asarray(mask[tick]).astype(bool))
    is_cav_t = torch.as_tensor(np.asarray(is_cav).astype(bool))
    if n_long >= 1:
        pl = np.array(plan[tick:tick + n_long], np.float32)
        sp = torch.as_tensor(pl[..., 0]).to(torch.float32) * \
            float(CAV_TGT_SPEED_SCALE)
        ac = torch.as_tensor(pl[..., 1]).to(torch.float32) * \
            float(CAV_ACCEL_SCALE)
        tm = torch.as_tensor(
            np.array(target[tick:tick + n_long - 1], bool)).to(torch.bool)
        alive = torch.cumprod(
            torch.cat([present0.unsqueeze(0).to(torch.int64),
                       tm.to(torch.int64)], dim=0), dim=0).to(torch.bool)
    else:
        sp = torch.zeros(1, int(kin.shape[1]))
        ac = torch.zeros(1, int(kin.shape[1]))
        alive = present0.unsqueeze(0)
    cav = (is_cav_t.unsqueeze(0) & alive).to(torch.float32).unsqueeze(1)
    return long_plan_summary(sp.unsqueeze(1), ac.unsqueeze(1), cav, horizon_s=h)


def head_quantiles_from_summary(bundle, summary_raw, *, t0_kin, t0_mask,
                                t0_counter):
    """Trained projection/head + A6 state token on a raw16 summary.

    Mirrors ``wm_api.engine.outcome_head`` after its ``trajectory_summary``
    step -- i.e. EXACTLY the trained path ``train/idea_a.py``
    ``correction_crps`` uses (``summary_raw`` -> ``CavMotionSummary`` ->
    ``CorrectionHead`` with the ``StateGlobalToken`` on the ORIGINAL t0
    state).  Returns the ``[n_tau, n_targets, n_quantiles]`` head output.
    """
    from corr.evaluation import (module_data, state_token, trained_correction)

    corr = trained_correction(bundle)
    raw_input = module_data(corr, summary_raw)
    token = state_token(bundle, raw_input, kin=t0_kin, mask=t0_mask,
                        counter=t0_counter, d_model=int(corr.cfg.d_model))
    out = corr(raw_input, token)
    q = out["outcome_quantiles"]
    norm = getattr(bundle, "outcome_norm", None)
    if norm is not None:
        from model.outcome_norm import denormalize_quantiles

        q = denormalize_quantiles(q, norm)
    return q.detach().cpu().to(torch.float64).numpy()[0]


def _t0_state_channels(source, window):
    """ORIGINAL t0 ``(kin[1,K,7], mask[1,K], counter[1,2,30])`` for the token."""
    kin = _array(source, window.item, "veh_kin.npy")
    mask = _array(source, window.item, "veh_mask.npy")
    t = int(window.local_t0_tick)
    if not 0 <= t < kin.shape[0]:
        raise PredictorRefusal("t0 tick escapes the pinned veh_kin array")
    kin0 = torch.as_tensor(np.array(kin[t], np.float32)).unsqueeze(0)
    mask0 = torch.as_tensor(np.asarray(mask[t]).astype(bool)).unsqueeze(0)
    counter0 = torch.as_tensor(window.counter_t0).unsqueeze(0)
    return kin0, mask0, counter0


def _no_ctrl(bundle):
    """The teacher path must not let the controller override the recording."""
    if bundle is None:
        return None
    try:
        return replace(bundle, ctrl=None)
    except (TypeError, ValueError):
        return bundle


def _restore_teacher_cavs(rolls, *, is_cav_t0, seq, seq_mask, lane, n_steps):
    """Restore the recorded CAV columns on the engine's teacher rollout.

    The engine applies ``teacher[i]`` before step ``i`` (so HDVs react to the
    executed CAV motion) but snapshots the controller/baseline-driven CAV
    state; overwriting the CAV columns with the recording reproduces the same
    CAV trajectory the trainer's ``torch.where(cav, teacher[t+1], ...)``
    wrote (``train/idea_a.py:1966-1970``).
    """
    cav = np.asarray(is_cav_t0, dtype=bool).reshape(-1)
    pos = rolls.positions
    spd = rolls.speeds
    lns = rolls.lanes
    acc = rolls.accel
    for s in range(pos.shape[0]):
        for i in range(int(n_steps) + 1):
            sel = cav & seq_mask[i]
            if not sel.any():
                continue
            idx = torch.as_tensor(np.flatnonzero(sel), dtype=torch.long)
            pos[s, i, 0, idx] = torch.as_tensor(seq[i, sel, 0],
                                                dtype=pos.dtype)
            spd[s, i, 0, idx] = torch.as_tensor(seq[i, sel, 4],
                                                dtype=spd.dtype)
            lns[s, i, 0, idx] = torch.as_tensor(lane[i, sel].astype(np.int64))
        for i in range(int(n_steps)):
            sel = cav & seq_mask[i + 1]
            if not sel.any():
                continue
            idx = torch.as_tensor(np.flatnonzero(sel), dtype=torch.long)
            acc[s, i, 0, idx] = torch.as_tensor(seq[i + 1, sel, 5],
                                                dtype=acc.dtype)
    return rolls


def rollout_teacher(bundle, world, *, seq, seq_mask, lane, n_steps,
                    integrator, residual_enabled, counter=None,
                    n_samples=1, seed=0, t0_s=0.0, is_cav_t0=None):
    """Run the P3 engine with training-parity CAV teacher forcing.

    Reuses :class:`rollout2.engine.Rollout2Engine` (the same engine the training
    rollout wraps) with the controller disabled, then restores the recorded CAV
    columns so the outcome head sees the trainer's CAV motion.
    """
    from rollout2.engine import Rollout2Engine
    from rollout2.state import Rollout2Config

    if is_cav_t0 is None:
        is_cav_t0 = world.is_cav.detach().to(torch.bool).cpu().numpy().reshape(-1)
    teacher = _teacher_tensor(seq, seq_mask, lane, n_steps=n_steps)
    if counter is not None and not isinstance(counter, torch.Tensor):
        counter = torch.as_tensor(counter)
    cfg = Rollout2Config(params=None, integrator=str(integrator),
                         residual_enabled=bool(residual_enabled),
                         arrivals_enabled=False, t0_s=float(t0_s))
    engine = Rollout2Engine(_no_ctrl(bundle), cfg, seed=int(seed))
    rolls = engine.run(world, None, n_steps=int(n_steps),
                       n_samples=int(n_samples), teacher=teacher,
                       counter=counter)
    return _restore_teacher_cavs(rolls, is_cav_t0=is_cav_t0, seq=seq,
                                 seq_mask=seq_mask, lane=lane, n_steps=n_steps)


def _row_position(row) -> int:
    feats = getattr(row, "features", None) or {}
    pos = feats.get("pack_item_position")
    if pos is None:
        raise PredictorRefusal(
            "an OutcomeRow carries no features['pack_item_position']; the "
            "real scorer cannot resolve its released CAL item")
    return int(pos)


class _WindowCache:
    """Per-item world/rollout resolution shared by the two predictors."""

    def __init__(self, source):
        self.source = source
        self._worlds: dict = {}

    def world(self, position) -> ItemWorld:
        if position not in self._worlds:
            _, item, run = _entry(self.source, position)
            self._worlds[position] = build_item_world(self.source, item, run)
        return self._worlds[position]


class ModelAPredictor:
    """Variant A real predictor: per-item P3 rollout -> trained outcome head."""

    quantile_levels = QUANTILE_LEVELS

    def __init__(self, bundle, *, source, cav_motion, dynamics_extrapolation=False,
                 seed: int = 0, n_samples: int = 1, paired_head_dv=None):
        if cav_motion not in CAV_MOTIONS:
            raise PredictorRefusal(
                f"cav_motion must be one of {list(CAV_MOTIONS)}; got "
                f"{cav_motion!r} (no default)")
        self.bundle = bundle
        self.source = source
        self.cav_motion = str(cav_motion)
        self.dynamics_extrapolation = bool(dynamics_extrapolation)
        self.seed = int(seed)
        self.n_samples = int(n_samples)
        #: Optional paired-effect dV source for the ``paired_head`` dv-method.
        #: ``None`` unless the owner wires the optional climatology-anchored
        #: paired-effect head (owner-approved 2026-10-04).  The scorer CLI
        #: refuses ``--dv-method paired_head`` when this is None, so a release
        #: with no head fails closed instead of substituting a marginal.
        self._pos_by_world_t0: dict = {}
        #: (world, t0, canonical arm) -> item position, so the paired_head
        #: dv-method can read the TREATMENT arm's own plan summary.
        self._pos_by_world_arm: dict = {}
        if paired_head_dv is None and isinstance(
                getattr(bundle, "paired_effect", None), torch.nn.Module):
            paired_head_dv = _PairedHeadDV(self)
        self.paired_head_dv = paired_head_dv
        self.name = f"A:{self.cav_motion}"
        #: Contract-3 provenance of the correction head input: teacher mode
        #: rebuilds the training ``summary_raw`` from the packed plan; the
        #: controller mode keeps the executed-rollout summary.
        self.summary_source = (
            SUMMARY_SOURCE_TRAINING_PACKED
            if self.cav_motion == CAV_MOTION_TEACHER
            else SUMMARY_SOURCE_ROLLOUT_EXECUTED)
        self._cache = _WindowCache(source)
        self.extrapolation_report: Optional[dict] = None

    def _rollout(self, window: ItemWorld, *, n_steps: int):
        item, run = window.item, window.run
        if self.cav_motion == CAV_MOTION_CONTROLLER:
            from evalh.scorer.actions import rollout_with_schedule

            schedule = build_schedule(self.source, item, run, n_steps=n_steps,
                                      is_cav_t0=window.is_cav_t0)
            return rollout_with_schedule(
                self.bundle, window.world, schedule, n_steps=int(n_steps),
                integrator="d1", residual_enabled=True,
                n_samples=self.n_samples, seed=self.seed)
        seq, seq_mask, lane = build_teacher(self.source, item, run,
                                            n_steps=n_steps)
        counter = torch.as_tensor(window.counter_t0[None])
        return rollout_teacher(
            self.bundle, window.world, seq=seq, seq_mask=seq_mask, lane=lane,
            n_steps=int(n_steps), integrator="d1", residual_enabled=True,
            counter=counter, n_samples=self.n_samples, seed=self.seed,
            is_cav_t0=window.is_cav_t0)

    def _head_quantiles(self, window: ItemWorld, rows):
        from evalh.idea_a import delay_quantiles_from_head

        n_steps = int(CORRECTION_WINDOW_STEPS)
        if self.cav_motion == CAV_MOTION_TEACHER:
            # Contract 3 teacher parity: the head input is the training-built
            # ``summary_raw`` from the pinned packed plan, NOT the executed
            # rollout.  The A6 token sees the same ORIGINAL t0 state.
            summary_raw = training_summary_raw(
                self.source, window.item, window.run, n_steps=n_steps,
                is_cav=window.is_cav_t0)
            t0_kin, t0_mask, t0_counter = _t0_state_channels(self.source,
                                                             window)
            head_q = head_quantiles_from_summary(
                self.bundle, summary_raw, t0_kin=t0_kin, t0_mask=t0_mask,
                t0_counter=t0_counter)
            batched = np.repeat(head_q[None], len(rows), axis=0)
            return delay_quantiles_from_head(batched, list(rows))

        from wm_api.engine import outcome_head

        rolls = self._rollout(window, n_steps=n_steps)
        head_q, _raw = outcome_head(
            self.bundle,
            rolls.positions[0].detach().cpu().numpy(),
            rolls.speeds[0].detach().cpu().numpy(),
            rolls.accel[0].detach().cpu().numpy(),
            rolls.masks[0].detach().cpu().numpy(),
            slot_ids=rolls.slot_ids[0].detach().cpu().numpy(),
            t0_kin=rolls.t0_kin.detach().cpu().numpy(),
            t0_mask=rolls.t0_mask.detach().cpu().numpy(),
            t0_is_cav=rolls.t0_is_cav.detach().cpu().numpy(),
            t0_slot_id=rolls.t0_slot_id.detach().cpu().numpy(),
            t0_counter=window.counter_t0[None])
        batched = np.repeat(head_q[None], len(rows), axis=0)
        return delay_quantiles_from_head(batched, list(rows))

    def _extrapolate(self, window: ItemWorld, rows) -> None:
        steps = [int(round(float(getattr(r, "tau_s", 0.0)) / DT_S))
                 for r in rows]
        n_steps = max([s for s in steps if s > 0] or [CORRECTION_WINDOW_STEPS])
        rolls = self._rollout(window, n_steps=n_steps)
        pos = rolls.positions.detach().cpu().numpy()
        self.extrapolation_report = dict(
            label="extrapolation", n_steps=int(n_steps),
            mode=self.cav_motion,
            finale_positions=pos[0, -1, 0].astype(float).tolist())

    def predict_quantiles(self, rows: Sequence) -> np.ndarray:
        rows = list(rows)
        if not rows:
            return np.zeros((0, N_QUANTILES), dtype=float)
        groups: dict = {}
        for i, row in enumerate(rows):
            groups.setdefault(_row_position(row), []).append(i)
            # (world, t0) -> an item position, for the paired-effect head:
            # sibling arms of one decision share the identical t0 state.
            self._pos_by_world_t0.setdefault(
                (str(row.world), round(float(row.t0_s), 6)),
                _row_position(row))
            self._pos_by_world_arm.setdefault(
                (str(row.world), round(float(row.t0_s), 6),
                 _row_arm_token(row)), _row_position(row))
        out = np.empty((len(rows), N_QUANTILES), dtype=float)
        for position in sorted(groups):
            idx = groups[position]
            window = self._cache.world(position)
            item_rows = [rows[i] for i in idx]
            out[idx] = self._head_quantiles(window, item_rows)
            if self.dynamics_extrapolation:
                self._extrapolate(window, item_rows)
        return np.maximum.accumulate(out, axis=1)

    @property
    def plan_aware(self) -> bool:
        """Whether the bundle's paired-effect head consumes the arm's plan."""
        return _bundle_plan_aware(self.bundle)

    def _state_only_dv(self, window: ItemWorld, tau_s) -> np.ndarray:
        """State-only head dV quantiles for a resolved item window."""
        head = getattr(self.bundle, "paired_effect", None)
        enc = getattr(self.bundle, "state_enc", None)
        if not isinstance(head, torch.nn.Module) or not isinstance(
                enc, torch.nn.Module):
            raise PredictorRefusal(
                "paired_head needs a bundle with paired_effect and state_enc")
        kin0, mask0, counter0 = _t0_state_channels(self.source, window)
        dtype = next(enc.parameters()).dtype
        with torch.no_grad():
            token = enc(kin0.to(dtype), mask0, counter0.to(dtype))
            q = head.dv_quantiles(token.to(next(head.parameters()).dtype))
        return np.maximum.accumulate(
            q[0, _tau_index(tau_s)].detach().cpu().to(torch.float64).numpy())

    def _plan_aware_dv(self, summary_raw, t0_kin, t0_mask, t0_counter,
                       tau_s, plan_long=None) -> np.ndarray:
        """Per-arm plan-aware head dV for one arm's plan (short or long).

        A11 full-horizon heads consume ``plan_long`` (built by
        :func:`training_plan_long`); the historical 16-D head consumes
        ``summary_raw``.  The helper fails closed on a missing/mismatched plan.
        """
        from model import constants as MC

        helper = _plan_aware_head_helper()
        if helper is None:
            raise PredictorRefusal(
                "the plan-aware paired-effect helper "
                "wm_api.eval.paired_effect_quantiles_for_plan is unavailable")
        kwargs = dict(t0_kin=t0_kin, t0_mask=t0_mask,
                      t0_counter=t0_counter, summary_raw=summary_raw)
        if plan_long is not None:
            kwargs["plan_long"] = plan_long
        try:
            q = np.asarray(helper(self.bundle, **kwargs), dtype=float)
        except PredictorRefusal:
            raise
        except (TypeError, ValueError) as exc:
            raise PredictorRefusal(
                f"the plan-aware paired-effect head refused the arm's plan: "
                f"{exc}") from exc
        if q.shape != (len(MC.TAUS), N_QUANTILES):
            raise PredictorRefusal(
                f"the plan-aware head returned {q.shape}; expected "
                f"({len(MC.TAUS)}, {N_QUANTILES})")
        if not np.isfinite(q).all():
            raise PredictorRefusal("the plan-aware head returned nonfinite dV")
        return np.maximum.accumulate(q[_tau_index(tau_s)])

    def _arm_long_plan(self, head, window):
        """The A11 full-horizon plan vector for one arm, or ``None`` for a
        historical 16-D plan head.

        The head's ``plan_raw_f`` selects the input width: a head built with
        ``--paired-effect-plan-horizon-s H`` requires the long vector, whose
        length must match the head exactly (fail closed).
        """
        from model.paired_effect import PLAN_RAW_F

        expected = int(getattr(head, "plan_raw_f", PLAN_RAW_F))
        if expected == int(PLAN_RAW_F):
            return None
        horizon = _bundle_plan_horizon_s(self.bundle)
        if horizon is None:
            raise PredictorRefusal(
                "the paired-effect head consumes a full-horizon plan but the "
                "checkpoint records no paired_effect_plan_horizon_s")
        plan_long = training_plan_long(
            self.source, window.item, window.run, horizon_s=horizon,
            is_cav=window.is_cav_t0)
        if int(plan_long.shape[-1]) != expected:
            raise PredictorRefusal(
                f"full-horizon plan has {int(plan_long.shape[-1])} features but "
                f"the head expects {expected} (plan_raw_f)")
        return plan_long

    def paired_head_arm_dv(self, window: ItemWorld, tau_s) -> np.ndarray:
        """Per-arm dV at ``tau_s`` for one resolved window.

        Uses the arm's teacher ``training_summary_raw`` -- and, for an A11
        full-horizon head, the arm's ``training_plan_long`` vector -- on the
        decision's shared t0 state when the head is plan-aware; otherwise the
        state-only head.
        """
        head = getattr(self.bundle, "paired_effect", None)
        if not isinstance(head, torch.nn.Module):
            raise PredictorRefusal(
                "paired_head needs a bundle with a paired_effect module")
        if self.plan_aware:
            summary_raw = training_summary_raw(
                self.source, window.item, window.run,
                n_steps=int(CORRECTION_WINDOW_STEPS), is_cav=window.is_cav_t0)
            t0_kin, t0_mask, t0_counter = _t0_state_channels(self.source,
                                                             window)
            plan_long = self._arm_long_plan(head, window)
            return self._plan_aware_dv(summary_raw, t0_kin, t0_mask,
                                       t0_counter, tau_s, plan_long=plan_long)
        return self._state_only_dv(window, tau_s)

    def paired_head_arm_quantiles(self, world, t0_s, tau_s, arm) -> np.ndarray:
        """Per-arm dV for a ``(world, t0)`` decision's treatment ``arm``."""
        key = (str(world), round(float(t0_s), 6), _contrast_treatment(arm))
        position = self._pos_by_world_arm.get(key)
        if position is None:
            # A plan-aware head MUST read the named arm's own plan; only the
            # shared-t0 state-only head may reuse a sibling window.
            if self.plan_aware:
                raise PredictorRefusal(
                    f"no scored item for the plan-aware arm {key}")
            position = self._pos_by_world_t0.get(
                (str(world), round(float(t0_s), 6)))
        if position is None:
            raise PredictorRefusal(f"no scored item for decision {key}")
        window = self._cache.world(position)
        return self.paired_head_arm_dv(window, tau_s)

    def paired_head_quantiles(self, world, t0_s, tau_s) -> np.ndarray:
        """dV quantiles of the state-only climatology-anchored paired-effect head.

        Uses the training-time token exactly: ``state_enc(kin, kin_mask,
        counter)`` on the ORIGINAL t0 state of the decision (shared by the
        sibling arms), then the head's monotone dV quantiles at ``tau_s``.
        """
        key = (str(world), round(float(t0_s), 6))
        position = self._pos_by_world_t0.get(key)
        if position is None:
            raise PredictorRefusal(f"no scored item for decision {key}")
        window = self._cache.world(position)
        return self._state_only_dv(window, tau_s)

    def predict_paired_head_row_quantiles(self, rows) -> np.ndarray:
        """``[N, N_QUANTILES]`` predicted dV vs native per row (native == 0).

        The decision-quality ``paired_head`` source scores each arm by the mean
        of this vector; the native arm is exactly zero by definition.
        """
        rows = list(rows)
        out = np.zeros((len(rows), N_QUANTILES), dtype=float)
        for i, row in enumerate(rows):
            if _row_arm_token(row) in ("", "native"):
                continue
            window = self._cache.world(_row_position(row))
            out[i] = self.paired_head_arm_dv(window, float(row.tau_s))
        return out


class _PairedHeadDV:
    """``dv_quantiles(pair_infos)`` adapter over :class:`ModelAPredictor`.

    The ``paired_head`` dv-method scores a contrast's TREATMENT arm, so it reads
    that arm's own plan summary (plan-aware head) rather than the shared
    state-only climatology whenever available.
    """

    def __init__(self, predictor):
        self.predictor = predictor

    def dv_quantiles(self, pair_infos) -> np.ndarray:
        rows = [self.predictor.paired_head_arm_quantiles(
            p["world"], p["t0_s"], p["tau_s"], p["contrast"])
            for p in pair_infos]
        return np.maximum.accumulate(np.asarray(rows, dtype=float), axis=1)


@dataclass(frozen=True)
class TeacherSpec:
    """Resolution token: v2g teacher mode forces this item's recording."""

    source: object
    item: dict
    run: dict


class V2GWindowResolver:
    """``resolve_window(row) -> (bundle, world, schedule|TeacherSpec)``."""

    #: The registered plan covers 19 x 30 s; the longest tau needs 1140 steps.
    MAX_STEPS = int(N_SEGMENTS * SEGMENT_S / DT_S)

    def __init__(self, source, bundle, *, cav_motion, max_steps=None):
        if cav_motion not in CAV_MOTIONS:
            raise PredictorRefusal(
                f"cav_motion must be one of {list(CAV_MOTIONS)}; got "
                f"{cav_motion!r} (no default)")
        self.source = source
        self.bundle = bundle
        self.cav_motion = str(cav_motion)
        self.max_steps = int(max_steps or self.MAX_STEPS)
        self._cache = _WindowCache(source)

    def __call__(self, row):
        position = _row_position(row)
        window = self._cache.world(position)
        if self.cav_motion == CAV_MOTION_CONTROLLER:
            schedule = build_schedule(self.source, window.item, window.run,
                                      n_steps=self.max_steps,
                                      is_cav_t0=window.is_cav_t0)
            return self.bundle, window.world, schedule
        return self.bundle, window.world, TeacherSpec(
            source=self.source, item=window.item, run=window.run)


def v2g_teacher_rollout(bundle, world, spec, *, n_steps, integrator,
                        residual_enabled, n_samples, seed, t0_s):
    """The v2g baseline under recorded CAV motion (teacher mode)."""
    from rollout2.engine import IdeaABundle

    seq, seq_mask, lane = build_teacher(spec.source, spec.item, spec.run,
                                        n_steps=int(n_steps))
    empty = IdeaABundle(ctrl=None, hdv=None, arrival=None, corr=None)
    return rollout_teacher(empty, world, seq=seq, seq_mask=seq_mask, lane=lane,
                           n_steps=int(n_steps), integrator=str(integrator),
                           residual_enabled=bool(residual_enabled),
                           counter=None, n_samples=int(n_samples),
                           seed=int(seed), t0_s=float(t0_s))


def build_v2g_predictor(source, bundle, *, option, cav_motion, artifacts,
                        max_steps=None):
    """Construct the fail-closed Part A v2g predictor for one owner option."""
    from evalh.scorer.comparator_v2g import V2GBaselinePredictor

    resolver = V2GWindowResolver(source, bundle, cav_motion=cav_motion,
                                 max_steps=max_steps)
    rollout_fn = (v2g_teacher_rollout
                  if cav_motion == CAV_MOTION_TEACHER else None)
    predictor = V2GBaselinePredictor(
        artifacts, option=option, resolve_window=resolver,
        rollout_fn=rollout_fn)
    # expose the bound params LF-sha to the DONE receipt builder
    predictor.params_lf_sha256 = artifacts["params_lf_sha256"]
    return predictor


__all__ = [
    "DT_S", "CAV_MOTION_CONTROLLER", "CAV_MOTION_TEACHER", "CAV_MOTIONS",
    "WORLD_ARRAYS", "SUMMARY_SOURCE_TRAINING_PACKED",
    "SUMMARY_SOURCE_ROLLOUT_EXECUTED", "PredictorRefusal", "ItemWorld",
    "TeacherSpec", "build_item_world", "expand_plan_to_ticks",
    "build_schedule", "build_teacher", "rollout_teacher",
    "training_summary_raw", "training_plan_long", "head_quantiles_from_summary",
    "ModelAPredictor",
    "V2GWindowResolver", "v2g_teacher_rollout", "build_v2g_predictor",
    "CORRECTION_WINDOW_STEPS",
]
