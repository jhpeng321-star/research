"""WM-S next-state predictor for the PL01 planner contract.

The predictor deliberately has no value head.  ``predict_states`` is the
model boundary: a trained WM-S model can replace the A11 stand-in without
changing planner code.  The stand-in uses the existing A11 physics rollout
with open-road entries/exits and SUMO-fidelity dynamics enabled.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from .a11_predictor import (
    BatchRollout,
    NONE_CODE,
    PLANNER_ACTION_CODES,
    _rollout_mask,
    _prediction,
    safety_stats,
    tracked_trajectories,
)

DT_S = 0.5
TAU_S = 300.0
CONTRACT = "wd01_predict/1.1"


class WMSPredictorError(RuntimeError):
    """Fail-closed refusal of the WM-S adapter."""


@dataclass(frozen=True)
class PredictedStates:
    """Normalized next-state trajectory returned by a WM-S plug-in.

    Arrays are ``[T+1, K]`` and include the live state at index zero.  A
    plug-in may instead return a mapping or an A11 :class:`BatchRollout`; the
    adapter normalizes both forms before calculating metrics.
    """

    positions: np.ndarray
    speeds: np.ndarray
    lanes: np.ndarray
    masks: np.ndarray
    accelerations: Optional[np.ndarray] = None
    slot_ids: Optional[np.ndarray] = None
    lengths: Optional[np.ndarray] = None
    speed_limits_mps: Optional[np.ndarray] = None
    times: Optional[np.ndarray] = None


# "t0 equality" tolerances: the plug-in's first sample must reproduce the live
# state.  Positions are metres and speeds m/s, both stored float32 upstream.
ATOL_POS_M = 1e-2
ATOL_SPEED_MPS = 1e-2
ATOL_TIME_S = 1e-6
# The time axis must be the uniform control/physics grid; 1 ms tolerates a
# float32 round-trip of the sample times without admitting a nonuniform grid.
ATOL_DT_S = 1e-3


def _value(result: Any, names: Sequence[str], default=None):
    if isinstance(result, Mapping):
        for name in names:
            if name in result:
                return result[name]
    for name in names:
        if hasattr(result, name):
            return getattr(result, name)
    return default


def _one_batch(value, *, name: str, dtype=None) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise WMSPredictorError(f"{name} must have shape [T+1, K], got {arr.shape}")
    return arr


def _validate_live_anchor(positions, speeds, lanes, masks, state) -> np.ndarray:
    """Fail closed unless every live, on-lane slot is present at sample 0 and
    that sample reproduces the live state.  Returns the boolean slot mask used
    for the equality check."""
    state_kin = np.asarray(state.veh_kin, dtype=float)
    k_n = positions.shape[1]
    if state_kin.ndim != 2 or state_kin.shape[0] != k_n:
        raise WMSPredictorError(
            f"predicted slots K={k_n} do not match the live state "
            f"(K={None if state_kin.ndim != 2 else state_kin.shape[0]})")
    live = np.asarray(_rollout_mask(state), bool)
    if live.shape != (k_n,):
        raise WMSPredictorError("live on-lane mask does not match predicted slots")
    masks0 = np.asarray(masks[0], bool)
    missing0 = live & ~masks0
    if missing0.any():
        raise WMSPredictorError(
            f"prediction omits {int(missing0.sum())} live on-lane slot(s) at t0; "
            "every live slot must be present in the plug-in output (fail closed)")
    present0 = masks0 & live
    if not present0.any():
        raise WMSPredictorError("prediction has no present on-lane slot at t0")
    if not np.allclose(positions[0, present0], state_kin[present0, 0],
                       rtol=0.0, atol=ATOL_POS_M):
        raise WMSPredictorError(
            "prediction t0 positions do not match the live state (t0 equality)")
    if not np.allclose(speeds[0, present0], state_kin[present0, 4],
                       rtol=0.0, atol=ATOL_SPEED_MPS):
        raise WMSPredictorError(
            "prediction t0 speeds do not match the live state (t0 equality)")
    live_lanes = np.argmax(state_kin[:, 1:4], axis=1).astype(np.int64)
    if not np.array_equal(np.asarray(lanes[0], np.int64)[present0],
                          live_lanes[present0]):
        raise WMSPredictorError(
            "prediction t0 lanes do not match the live state (t0 equality)")
    return present0


def _resolve_times(result: Any, state, t_n: int) -> np.ndarray:
    """Finite, strictly increasing, uniform time axis starting at ``state.time_s``.

    A plug-in may supply ``times``/``time_s``/``t_s``; otherwise the implicit
    ``t0 + i*DT_S`` grid is synthesized.  A non-monotonic, mis-anchored or
    nonuniform axis (spacing != ``DT_S``) is refused (fail closed), so the
    ``window_label_delay`` scoring step always integrates over real ``DT_S``
    intervals instead of assuming them.
    """
    raw = _value(result, ("times", "time_s", "t_s"))
    if raw is None:
        return float(state.time_s) + DT_S * np.arange(t_n, dtype=float)
    times = np.asarray(raw, dtype=float).reshape(-1)
    if times.shape != (t_n,):
        raise WMSPredictorError(f"times must have shape [T+1], got {times.shape}")
    if not np.isfinite(times).all():
        raise WMSPredictorError("prediction times contain non-finite entries")
    deltas = np.diff(times)
    if not np.all(deltas > 0.0):
        raise WMSPredictorError("prediction times are not strictly increasing")
    if not np.allclose(deltas, DT_S, rtol=0.0, atol=ATOL_DT_S):
        raise WMSPredictorError(
            "prediction times must be the uniform DT_S grid "
            f"({DT_S}s); got spacing {float(deltas.min()):.6g}.."
            f"{float(deltas.max()):.6g}s")
    if abs(float(times[0]) - float(state.time_s)) > ATOL_TIME_S:
        raise WMSPredictorError("prediction times do not start at the live state time")
    return times


def normalize_states(result: Any, state) -> PredictedStates:
    """Convert a plug-in result to the small state trajectory contract.

    Fails closed on a one-state/too-short trajectory, non-finite arrays, a
    first sample that does not reproduce the live state (t0 equality) with
    every live slot present, or a missing/nonuniform/mis-anchored time axis.
    """
    if isinstance(result, BatchRollout):
        result = result.at(0)
        positions = result.pos[:, 0]
        speeds = result.spd[:, 0]
        lanes = result.lns[:, 0]
        masks = result.msk[:, 0]
        accelerations = result.acc[:, 0]
        slot_ids = result.ids[:, 0]
        lengths = result.length[0]
        limits = _value(result, ("speed_limits_mps", "speed_limits"))
    else:
        positions = _value(result, ("positions", "pos"))
        speeds = _value(result, ("speeds", "spd", "velocities"))
        lanes = _value(result, ("lanes", "lns"))
        masks = _value(result, ("masks", "msk", "present"))
        accelerations = _value(result, ("accelerations", "acc"))
        slot_ids = _value(result, ("slot_ids", "ids"))
        lengths = _value(result, ("lengths", "length"))
        limits = _value(result, ("speed_limits_mps", "speed_limits"))
    if positions is None or speeds is None or lanes is None or masks is None:
        raise WMSPredictorError(
            "predict_states must return positions, speeds, lanes, and masks"
        )
    positions = _one_batch(positions, name="positions", dtype=float)
    speeds = _one_batch(speeds, name="speeds", dtype=float)
    lanes = _one_batch(lanes, name="lanes", dtype=np.int64)
    masks = _one_batch(masks, name="masks", dtype=bool)
    if not (positions.shape == speeds.shape == lanes.shape == masks.shape):
        raise WMSPredictorError("predicted state arrays do not have equal shapes")
    t_n, k_n = positions.shape
    if t_n < 2:
        raise WMSPredictorError(
            "predict_states returned a single state; at least two samples "
            "(the live state plus one step) are required")
    if not np.isfinite(positions[masks]).all():
        raise WMSPredictorError("predicted positions contain non-finite entries")
    if not np.isfinite(speeds[masks]).all():
        raise WMSPredictorError("predicted speeds contain non-finite entries")
    if accelerations is None:
        accelerations = np.zeros((max(0, t_n - 1), k_n), dtype=float)
    else:
        accelerations = _one_batch(accelerations, name="accelerations", dtype=float)
        if accelerations.shape == (t_n, k_n):
            accelerations = accelerations[:-1]
        if accelerations.shape != (max(0, t_n - 1), k_n):
            raise WMSPredictorError("accelerations must have shape [T, K]")
    if slot_ids is None:
        slot_ids = np.repeat(np.asarray(state.slot_id, dtype=np.int64)[None, :], t_n, axis=0)
    else:
        slot_ids = _one_batch(slot_ids, name="slot_ids", dtype=np.int64)
    if lengths is None:
        lengths = np.full(k_n, 5.0, dtype=float)
    lengths = np.asarray(lengths, dtype=float)
    if lengths.ndim == 2 and lengths.shape[0] == 1:
        lengths = lengths[0]
    if lengths.shape != (k_n,):
        raise WMSPredictorError("lengths must have shape [K]")
    if limits is not None:
        limits = np.asarray(limits, dtype=float)
        if limits.ndim == 3 and limits.shape[0] == 1:
            limits = limits[0]
        if limits.ndim == 1:
            limits = np.repeat(limits[None, :], t_n, axis=0)
        if limits.shape != (t_n, k_n):
            raise WMSPredictorError("speed_limits_mps must have shape [T+1, K]")
    _validate_live_anchor(positions, speeds, lanes, masks, state)
    times = _resolve_times(result, state, t_n)
    return PredictedStates(
        positions=positions,
        speeds=speeds,
        lanes=lanes,
        masks=masks,
        accelerations=accelerations,
        slot_ids=slot_ids,
        lengths=lengths,
        speed_limits_mps=limits,
        times=times,
    )


def _a11_speed_limits(states: PredictedStates, *, sumo_idm: bool = True,
                      sumo_speed_factor: Optional[float] = None) -> np.ndarray:
    """Get the frozen model-frame segment limits for the A11 stand-in."""
    try:
        from rollout2.baseline import Params, model_frame_params

        params = Params(model_frame_params(
            sumo_idm=sumo_idm, speed_factor=sumo_speed_factor))
        out = np.full(states.positions.shape, np.nan, dtype=float)
        for t in range(states.positions.shape[0]):
            present = states.masks[t] & (states.lanes[t] >= 0)
            if present.any():
                out[t, present] = params.seg_limit_vec(
                    states.lanes[t, present], states.positions[t, present])
        return out
    except (ImportError, OSError, KeyError, ValueError):
        # A trained plug-in normally supplies limits.  This fallback keeps
        # the adapter usable with a light-weight fake and the highway limit.
        return np.full(states.positions.shape, 31.29, dtype=float)


def window_label_delay(states: PredictedStates, horizon_s: float, *, dt_s: float = DT_S,
                       sumo_idm: bool = True,
                       sumo_speed_factor: Optional[float] = None) -> float:
    """v0.4a ``window_label`` delay in seconds for all modeled vehicles.

    Refuses (raises) instead of returning a fail-open zero when the prediction
    is too short to span the requested horizon.
    """
    if not np.isfinite(float(horizon_s)) or float(horizon_s) < 0.0:
        raise WMSPredictorError("horizon_s must be finite and non-negative")
    if float(dt_s) <= 0.0:
        raise WMSPredictorError("dt_s must be positive")
    need = int(math.ceil(float(horizon_s) / float(dt_s)))
    t_n = int(states.positions.shape[0])
    if t_n < 2:
        raise WMSPredictorError(
            "window_label_delay needs at least two state samples; the "
            "prediction is too short to price a candidate")
    if t_n - 1 < need:
        raise WMSPredictorError(
            f"prediction spans {t_n - 1} steps < requested horizon {need} "
            f"steps ({float(horizon_s)}s / {float(dt_s)}s)")
    times = states.times
    if times is not None:
        times = np.asarray(times, dtype=float).reshape(-1)
        if times.shape != (t_n,) or not np.isfinite(times).all():
            raise WMSPredictorError("prediction time axis is invalid")
        deltas = np.diff(times)
        if not np.all(deltas > 0.0):
            raise WMSPredictorError("prediction time axis is not monotonic")
        if not np.allclose(deltas, float(dt_s), rtol=0.0, atol=ATOL_DT_S):
            raise WMSPredictorError(
                f"prediction time axis must be uniform with spacing dt_s={dt_s}s")
        if float(times[need] - times[0]) < float(horizon_s) - ATOL_TIME_S:
            raise WMSPredictorError(
                "prediction time axis does not span the requested horizon")
    n = need
    limits = states.speed_limits_mps
    if limits is None:
        limits = _a11_speed_limits(
            states, sumo_idm=sumo_idm, sumo_speed_factor=sumo_speed_factor)
    speed = states.speeds[1:n + 1]
    mask = states.masks[1:n + 1]
    limit = limits[1:n + 1]
    valid = mask & np.isfinite(speed) & np.isfinite(limit) & (limit > 0.0)
    lost = np.zeros_like(speed, dtype=float)
    lost[valid] = np.clip(1.0 - speed[valid] / limit[valid], 0.0, 1.0)
    # Integrate with the actual time width of each scored sample; under the
    # enforced uniform DT_S grid this equals ``lost.sum() * dt_s``.
    if times is not None:
        weights = np.diff(times[:n + 1])[:, None]
    else:
        weights = np.full((n, 1), float(dt_s), dtype=float)
    return float(np.sum(lost * weights))


def _as_rollout(states: PredictedStates) -> BatchRollout:
    t_n, k_n = states.positions.shape
    acc = np.asarray(states.accelerations, dtype=float)
    a0 = acc[:1] if acc.shape[0] else np.zeros((1, k_n), dtype=float)
    ids = np.asarray(states.slot_ids, dtype=np.int64)
    if ids.shape != (t_n, k_n):
        ids = np.repeat(ids[:1], t_n, axis=0)
    return BatchRollout(
        pos=states.positions[:, None, :], spd=states.speeds[:, None, :],
        acc=acc[:, None, :], lns=states.lanes[:, None, :],
        msk=states.masks[:, None, :], ids=ids[:, None, :],
        length=np.asarray(states.lengths, dtype=float)[None, :], a0=a0,
    )


def _controllers_to_codes(state, controllers: Mapping[str, str]) -> np.ndarray:
    codes = np.full(len(state.slot_vid), NONE_CODE, dtype=np.int64)
    cav_slots = state.cav_slots
    for cav in cav_slots:
        action = controllers.get(cav, "native")
        try:
            codes[cav_slots[cav]] = PLANNER_ACTION_CODES[action]
        except KeyError as exc:
            raise WMSPredictorError(f"unsupported CAV action {action!r}") from exc
    unknown = sorted(set(controllers) - set(cav_slots))
    if unknown:
        raise WMSPredictorError(f"plan assigns CAVs absent from state: {unknown[:5]}")
    codes[~(np.asarray(state.is_cav, bool) & _rollout_mask(state))] = NONE_CODE
    return codes


def a11_predict_states(bundle, state, cav_controllers, horizon_s, seed, *,
                       open_road: bool = True, sumo_fidelity: bool = True,
                       sumo_idm: bool = True,
                       sumo_speed_factor: Optional[float] = None):
    """Stand-in ``predict_states`` implementation backed by A11 rollout2."""
    from .a11_predictor import batched_rollout

    n_steps = max(1, int(math.ceil(float(horizon_s) / DT_S)))
    codes = _controllers_to_codes(state, cav_controllers)
    return batched_rollout(
        bundle, state, codes[None, :], n_steps=n_steps,
        t0_s=float(state.time_s), seed=int(seed),
        open_entries=bool(open_road), physical_exits=bool(open_road),
        sumo_fidelity=bool(sumo_fidelity), sumo_idm=bool(sumo_idm),
        sumo_speed_factor=sumo_speed_factor,
    )


class WMSPredictor:
    """Planner predictor that prices plans from paired next-state rollouts."""

    contract_version = CONTRACT

    def __init__(self, bundle=None, *, predict_states: Optional[Callable] = None,
                 tau_s: float = TAU_S, max_horizon_s: Optional[float] = None,
                 open_road: bool = True, sumo_fidelity: bool = True,
                 sumo_idm: bool = True, sumo_speed_factor: Optional[float] = None,
                 track_vehicles: str = "all", model_sha256: Optional[str] = None):
        if float(tau_s) <= 0.0:
            raise WMSPredictorError("tau_s must be positive")
        if predict_states is None and bundle is None:
            raise WMSPredictorError("bundle is required for the A11 stand-in")
        self.bundle = bundle
        self.predict_states_fn = predict_states
        self.tau_s = float(tau_s)
        self.max_horizon_s = (None if max_horizon_s is None else float(max_horizon_s))
        self.open_road = bool(open_road)
        self.sumo_fidelity = bool(sumo_fidelity)
        self.sumo_idm = bool(sumo_idm)
        self.sumo_speed_factor = sumo_speed_factor
        self.track_vehicles = track_vehicles
        self.model_sha256 = model_sha256
        self.calls: list = []

    def _predict_states(self, state, controllers, horizon_s, seed):
        fn = self.predict_states_fn
        if fn is not None:
            return fn(state, controllers, horizon_s, seed)
        return a11_predict_states(
            self.bundle, state, controllers, horizon_s, seed,
            open_road=self.open_road, sumo_fidelity=self.sumo_fidelity,
            sumo_idm=self.sumo_idm, sumo_speed_factor=self.sumo_speed_factor)

    def predict(self, state_samples, plan, controller_state=None, **kw):
        return self.predict_many(state_samples, [plan], controller_state=controller_state, **kw)[0]

    def predict_many(self, state_samples: Sequence, plans: Sequence, *,
                     controller_state=None, horizon_s: float = 600.0,
                     trajectory_s: Optional[float] = 60.0, commit_s: float = 30.0,
                     continuation: str = "planner") -> list:
        del controller_state, commit_s, continuation
        samples = list(state_samples)
        if len(samples) != 1:
            raise WMSPredictorError("WM-S adapter takes exactly one state sample (K = 1)")
        plans = list(plans)
        if not plans:
            return []
        state = samples[0].hybrid
        requested = min(float(horizon_s), self.tau_s)
        if self.max_horizon_s is not None:
            requested = min(requested, self.max_horizon_s)
        seed = int((samples[0].meta or {}).get("seed", 0))
        started = time.perf_counter()
        native = {c: "native" for c in state.cav_slots}
        controller_rows = [native]
        row_keys = {tuple(sorted(native.items())): 0}
        plan_rows = []
        for plan in plans:
            if plan.walls:
                raise WMSPredictorError("WM-S stand-in does not support walls")
            row = {c: plan.per_cav.get(c, "native") for c in state.cav_slots}
            key = tuple(sorted(row.items()))
            if key not in row_keys:
                row_keys[key] = len(controller_rows)
                controller_rows.append(row)
            plan_rows.append(row_keys[key])

        trajectories = []
        for row in controller_rows:
            # Deliberately reuse the exact decision seed for every candidate.
            trajectories.append(normalize_states(
                self._predict_states(state, row, requested, seed), state))
        native_states = trajectories[0]
        native_roll = _as_rollout(native_states)
        window_s = requested if trajectory_s is None else min(float(trajectory_s), requested)
        n_window = max(1, int(math.ceil(window_s / DT_S)))
        vehicles = None if self.track_vehicles == "all" else set(state.cav_slots)
        native_stats = safety_stats(native_roll, 0, n_window, state.slot_vid)
        native_delay = window_label_delay(
            native_states, requested, sumo_idm=self.sumo_idm,
            sumo_speed_factor=self.sumo_speed_factor)
        native_traj = tracked_trajectories(
            native_roll, 0, n_window, state.slot_vid, float(state.time_s), vehicles,
            slots=np.flatnonzero(np.asarray(state.veh_mask, bool)))
        out = []
        for plan, row_i in zip(plans, plan_rows):
            states = trajectories[row_i]
            roll = _as_rollout(states)
            stats = safety_stats(roll, 0, n_window, state.slot_vid)
            delay = window_label_delay(
                states, requested, sumo_idm=self.sumo_idm,
                sumo_speed_factor=self.sumo_speed_factor)
            traj = tracked_trajectories(
                roll, 0, n_window, state.slot_vid, float(state.time_s), vehicles,
                slots=np.flatnonzero(np.asarray(state.veh_mask, bool)))
            dv = delay - native_delay
            meta = {
                "contract_version": CONTRACT,
                "predictor": "WM-S-next-state/A11-standin" if self.predict_states_fn is None else "WM-S-next-state/plugin",
                "model_sha256": self.model_sha256,
                "delay_metric": "v0.4a_window_label",
                "delay_scope": "all_vehicles",
                "decision_tau_s": requested,
                "dv_mean_tau": float(dv),
                "dv_q90_tau": float(dv),
                "native_delay_s": float(native_delay),
                "plan_delay_s": float(delay),
                "seed": seed,
                "window_s": window_s,
                "open_road": self.open_road,
                "sumo_fidelity": self.sumo_fidelity,
                "part_a_scope": "all_vehicles",
                "part_a_model": "WM-S-next-state",
                "min_gap_m": stats.min_gap_m,
                "native_min_gap_m": native_stats.min_gap_m,
            }
            prediction = _prediction(dv, stats, native_stats, traj, native_traj, meta)
            # The old helper's optional head audit is not meaningful here.
            prediction.meta["head_native_dv_mean"] = 0.0
            out.append(prediction)
        self.calls.append({
            "n_plans": len(plans), "n_unique_rollouts": len(trajectories),
            "n_rollouts": len(trajectories), "seed": seed,
            "horizon_s": requested, "total_s": round(time.perf_counter() - started, 3),
        })
        return out


# Simple name for plug-in users and hidden integration callers.
Predictor = WMSPredictor


__all__ = [
    "DT_S", "TAU_S", "PredictedStates", "Predictor", "WMSPredictor",
    "WMSPredictorError", "a11_predict_states", "normalize_states",
    "window_label_delay",
]
