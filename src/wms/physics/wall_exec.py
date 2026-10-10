"""Batched, rollout-local executor for the simulator wall-group laws.

Inputs are predicted state tensors/arrays shaped ``[B, K]``. Each batch row
has at most one wall; ``member_mask`` fixes its exclusive roster for the
episode. Results use the action-timeline vocabulary (native=0, damping=1,
PI=2, direct=3); a live wall emits the direct owner and its absolute target
speed, while nonmembers and handed-back members retain their prior timeline
controller. No simulator imports or GPU are required.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

NATIVE, DAMPING, PI_EXTERNAL_U, DIRECT = range(4)
PHASE_ASSEMBLY, PHASE_HOLD, PHASE_HANDBACK = range(3)
PHASE_NAMES = ("assembly", "hold", "handback")
V21 = "wall-v2.1-closure"
V22 = "wall-v2.2-merge-release"

SAFE_SPEED_MIN_MPS = 2.0
MAX_WALL_SPAN_M = 75.0
ABORT_SPAN_M = 150.0
ASSEMBLY_SPEED_TOL_MPS = 1.0
ASSEMBLY_SPAN_TOL_M = 6.0
BUFFER_DECEL_MPS2 = 3.0
COMFORT_ACCEL_MPS2 = 1.5
REAR_GAP_M = 10.0
REAR_SPEED_EXCESS_MPS = 2.0
ASSEMBLY_CLOSE_MAX_DELTA_MPS = 2.0
ASSEMBLY_CLOSE_GAIN_PER_S = 0.1
ASSEMBLY_TIMEOUT_S = 60.0
MERGE_RELEASE_POINT_S = 2644.2


def _validate(mask: np.ndarray, speed: np.ndarray, law: str) -> None:
    if mask.ndim != 2 or speed.shape != (mask.shape[0],):
        raise ValueError("member_mask must be [B,K] and wall_speed_mps [B]")
    if not mask.any(axis=1).all() or (mask.sum(axis=1) < 2).any():
        raise ValueError("each wall needs at least two members")
    if not np.isfinite(speed).all() or (speed <= SAFE_SPEED_MIN_MPS).any():
        raise ValueError("wall speeds must be finite and strictly above 2 m/s")
    if law not in (V21, V22):
        raise ValueError(f"unsupported wall law {law!r}")


def _numpy_pace(target: float, speed: float, dt: float) -> float:
    return float(np.clip((target - speed) / dt, -BUFFER_DECEL_MPS2,
                         COMFORT_ACCEL_MPS2))


@dataclass
class _State:
    phase: np.ndarray
    assembly_since: np.ndarray
    assembled_at: np.ndarray
    reason: list[str | None]


class NumpyWallExecutor:
    """Readable NumPy reference implementation; one independent wall per B."""

    def __init__(self, member_mask: Any, wall_speed_mps: Any, *,
                 prior_controller: Any, law: str = V21, control_dt_s: float = 1.0):
        self.members = np.asarray(member_mask, dtype=bool).copy()
        self.speed = np.asarray(wall_speed_mps, dtype=np.float64).copy()
        _validate(self.members, self.speed, law)
        self.law, self.dt = law, float(control_dt_s)
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("control_dt_s must be positive and finite")
        self.prior = np.asarray(prior_controller, dtype=np.int64).copy()
        if self.prior.shape != self.members.shape or ((self.prior < 0) | (self.prior > 3)).any():
            raise ValueError("prior_controller must be [B,K] timeline controller ids")
        b = len(self.speed)
        self.state = _State(np.full(b, PHASE_ASSEMBLY, np.int8),
                            np.full(b, np.nan), np.full(b, np.nan), [None] * b)

    def step(self, position_m: Any, speed_mps: Any, *, now_s: float,
             lane_index: Any = None, present: Any = None, abort: Any = None,
             release: Any = None, base_acceleration_mps2: Any = None,
             base_target_speed_mps: Any = None) -> dict[str, Any]:
        p, v = np.asarray(position_m, dtype=np.float64), np.asarray(speed_mps, dtype=np.float64)
        if p.shape != self.members.shape or v.shape != self.members.shape:
            raise ValueError("predicted positions and speeds must be [B,K]")
        b, k = p.shape
        live = np.ones((b, k), bool) if present is None else np.asarray(present, bool)
        lanes = None if lane_index is None else np.asarray(lane_index)
        abort = np.zeros(b, bool) if abort is None else np.asarray(abort, bool)
        release = np.zeros(b, bool) if release is None else np.asarray(release, bool)
        base_a = np.zeros((b, k)) if base_acceleration_mps2 is None else np.asarray(base_acceleration_mps2)
        base_t = np.zeros((b, k)) if base_target_speed_mps is None else np.asarray(base_target_speed_mps)
        for x in (live, abort, release, base_a, base_t):
            if x.shape != (b, k) and x.shape != (b,):
                raise ValueError("step input has incompatible shape")
        owner = self.prior.copy()
        accel, target = base_a.astype(np.float64, copy=True), base_t.astype(np.float64, copy=True)
        events: list[dict] = []
        for i in range(b):
            if self.state.phase[i] == PHASE_HANDBACK:
                continue
            mi = self.members[i]
            reason = "planner_abort" if abort[i] else None
            if reason is None and not live[i, mi].all():
                reason = "member_frame_missing"
            if reason is None and not np.isfinite(p[i, mi]).all():
                reason = "member_position_unavailable"
            pos = p[i, mi]
            span = float(np.ptp(pos)) if pos.size else 0.0
            if reason is None and span > ABORT_SPAN_M:
                reason = "member_span_exceeded"
            if reason is None:
                rear = float(pos.min())
                other = ~mi & live[i] & np.isfinite(p[i]) & np.isfinite(v[i])
                risk = other & (p[i] >= rear - REAR_GAP_M) & (p[i] <= rear) & (v[i] > self.speed[i] + REAR_SPEED_EXCESS_MPS)
                if risk.any():
                    reason = f"non_member_rear_end_risk:{int(np.flatnonzero(risk)[0])}"
            phase = int(self.state.phase[i])
            if reason is None and phase == PHASE_ASSEMBLY:
                if np.isnan(self.state.assembly_since[i]):
                    self.state.assembly_since[i] = now_s
                    events.append({"batch": i, "event": "assembly_start", "timestamp_s": now_s})
                if now_s - self.state.assembly_since[i] >= ASSEMBLY_TIMEOUT_S:
                    reason = "assembly_timeout"
                elif lanes is not None and np.isfinite(lanes[i, mi].astype(float)).all() and len(set(lanes[i, mi].tolist())) != int(mi.sum()):
                    reason = "lane_coverage_violation"
                elif self.law == V22 and float(pos.max()) >= MERGE_RELEASE_POINT_S - 1e-9:
                    reason = "assembly_past_release"
            if reason is None and phase == PHASE_HOLD and self.law == V22 and float(pos.max()) >= MERGE_RELEASE_POINT_S - 1e-9:
                self.state.phase[i] = PHASE_HANDBACK
                events.append({"batch": i, "event": "merge_release", "timestamp_s": now_s})
                continue
            if reason is None and phase == PHASE_HOLD and release[i]:
                self.state.phase[i] = PHASE_HANDBACK
                events.append({"batch": i, "event": "released", "timestamp_s": now_s})
                continue
            if reason is not None:
                self.state.reason[i] = reason
                self.state.phase[i] = PHASE_HANDBACK
                events.append({"batch": i, "event": "safety_abort", "reason": reason, "timestamp_s": now_s})
                continue
            positions = p[i, mi]
            speeds = v[i, mi]
            if phase == PHASE_ASSEMBLY:
                front = float(positions.max())
                for j in np.flatnonzero(mi):
                    gap = max(0.0, front - p[i, j])
                    tgt = self.speed[i] + min(ASSEMBLY_CLOSE_MAX_DELTA_MPS, ASSEMBLY_CLOSE_GAIN_PER_S * gap)
                    accel[i, j] = _numpy_pace(tgt, v[i, j], self.dt)
                    target[i, j] = v[i, j] + accel[i, j] * self.dt
                if np.all(np.abs(speeds - self.speed[i]) <= ASSEMBLY_SPEED_TOL_MPS) and span <= ASSEMBLY_SPAN_TOL_M:
                    self.state.phase[i] = PHASE_HOLD
                    self.state.assembled_at[i] = now_s
                    events.append({"batch": i, "event": "assembled", "timestamp_s": now_s})
            else:
                for j in np.flatnonzero(mi):
                    accel[i, j] = _numpy_pace(self.speed[i], v[i, j], self.dt)
                    target[i, j] = v[i, j] + accel[i, j] * self.dt
            owner[i, mi] = DIRECT
        return {"owner": owner, "controller": owner.copy(), "acceleration_mps2": accel,
                "target_speed_mps": target, "phase": self.state.phase.copy(),
                "phase_names": tuple(PHASE_NAMES[x] for x in self.state.phase),
                "abort_reason": tuple(self.state.reason), "events": events}


class TorchWallExecutor:
    """Torch batched wall executor. CPU tensors are recommended for WMS rollouts."""

    def __init__(self, member_mask: Any, wall_speed_mps: Any, *, prior_controller: Any,
                 law: str = V21, control_dt_s: float = 1.0, device: str = "cpu",
                 dtype: torch.dtype = torch.float64):
        # Validate using the reference's shape/range contract, then own compact tensors.
        mask_np = np.asarray(member_mask, dtype=bool)
        speed_np = np.asarray(wall_speed_mps, dtype=np.float64)
        _validate(mask_np, speed_np, law)
        self.device, self.dtype, self.law, self.dt = torch.device(device), dtype, law, float(control_dt_s)
        if self.device.type != "cpu":
            raise ValueError("wall executor is CPU-only")
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("control_dt_s must be positive and finite")
        self.members = torch.as_tensor(mask_np, dtype=torch.bool, device=self.device)
        self.speed = torch.as_tensor(speed_np, dtype=dtype, device=self.device)
        self.prior = torch.as_tensor(prior_controller, dtype=torch.long, device=self.device)
        if tuple(self.prior.shape) != tuple(self.members.shape) or bool(((self.prior < 0) | (self.prior > 3)).any()):
            raise ValueError("prior_controller must be [B,K] timeline controller ids")
        b = self.speed.shape[0]
        self.phase = torch.full((b,), PHASE_ASSEMBLY, dtype=torch.int64, device=self.device)
        self.assembly_since = torch.full((b,), float("nan"), dtype=dtype, device=self.device)
        self.assembled_at = torch.full((b,), float("nan"), dtype=dtype, device=self.device)
        self.abort_reason: list[str | None] = [None] * b

    def step(self, position_m: Any, speed_mps: Any, *, now_s: float, **kwargs) -> dict[str, Any]:
        # State transitions are discrete; arithmetic and per-member command
        # generation remain tensor-batched. This avoids device-dependent branch
        # behavior while matching the simulator's Python lifecycle exactly.
        p = torch.as_tensor(position_m, dtype=self.dtype, device=self.device)
        v = torch.as_tensor(speed_mps, dtype=self.dtype, device=self.device)
        if p.shape != self.members.shape or v.shape != self.members.shape:
            raise ValueError("predicted positions and speeds must be [B,K]")
        b, k = p.shape
        present = torch.as_tensor(kwargs.get("present", torch.ones_like(self.members)), dtype=torch.bool, device=self.device)
        lanes_arg = kwargs.get("lane_index")
        lanes = None if lanes_arg is None else torch.as_tensor(lanes_arg, device=self.device)
        abort = torch.as_tensor(kwargs.get("abort", torch.zeros(b)), dtype=torch.bool, device=self.device).reshape(b)
        release = torch.as_tensor(kwargs.get("release", torch.zeros(b)), dtype=torch.bool, device=self.device).reshape(b)
        accel = torch.as_tensor(kwargs.get("base_acceleration_mps2", torch.zeros_like(p)), dtype=self.dtype, device=self.device).clone()
        target = torch.as_tensor(kwargs.get("base_target_speed_mps", torch.zeros_like(p)), dtype=self.dtype, device=self.device).clone()
        if present.shape != p.shape or accel.shape != p.shape or target.shape != p.shape or (lanes is not None and lanes.shape != p.shape):
            raise ValueError("step input has incompatible shape")
        owner = self.prior.clone()
        events: list[dict] = []
        for i in range(b):
            mi = self.members[i]
            if int(self.phase[i]) == PHASE_HANDBACK:
                continue
            reason = "planner_abort" if bool(abort[i]) else None
            pos_t, speeds_t = p[i, mi], v[i, mi]
            if reason is None and not bool(present[i, mi].all()):
                reason = "member_frame_missing"
            if reason is None and not bool(torch.isfinite(pos_t).all()):
                reason = "member_position_unavailable"
            span = float((pos_t.max() - pos_t.min()).item())
            if reason is None and span > ABORT_SPAN_M:
                reason = "member_span_exceeded"
            if reason is None:
                rear = pos_t.min()
                other = ~mi & present[i] & torch.isfinite(p[i]) & torch.isfinite(v[i])
                risk = other & (p[i] >= rear - REAR_GAP_M) & (p[i] <= rear) & (v[i] > self.speed[i] + REAR_SPEED_EXCESS_MPS)
                ids = torch.nonzero(risk).flatten()
                if ids.numel():
                    reason = f"non_member_rear_end_risk:{int(ids[0])}"
            phase = int(self.phase[i])
            if reason is None and phase == PHASE_ASSEMBLY:
                if torch.isnan(self.assembly_since[i]):
                    self.assembly_since[i] = now_s
                    events.append({"batch": i, "event": "assembly_start", "timestamp_s": now_s})
                if now_s - float(self.assembly_since[i]) >= ASSEMBLY_TIMEOUT_S:
                    reason = "assembly_timeout"
                elif (lanes is not None
                      and bool(torch.isfinite(lanes[i, mi].to(self.dtype)).all())
                      and torch.unique(lanes[i, mi]).numel() != int(mi.sum())):
                    reason = "lane_coverage_violation"
                elif self.law == V22 and float(pos_t.max()) >= MERGE_RELEASE_POINT_S - 1e-9:
                    reason = "assembly_past_release"
            if reason is None and phase == PHASE_HOLD and self.law == V22 and float(pos_t.max()) >= MERGE_RELEASE_POINT_S - 1e-9:
                self.phase[i] = PHASE_HANDBACK
                events.append({"batch": i, "event": "merge_release", "timestamp_s": now_s})
                continue
            if reason is None and phase == PHASE_HOLD and bool(release[i]):
                self.phase[i] = PHASE_HANDBACK
                events.append({"batch": i, "event": "released", "timestamp_s": now_s})
                continue
            if reason is not None:
                self.abort_reason[i] = reason
                self.phase[i] = PHASE_HANDBACK
                events.append({"batch": i, "event": "safety_abort", "reason": reason, "timestamp_s": now_s})
                continue
            front = pos_t.max()
            gap = (front - p[i]).clamp_min(0)
            closure = self.speed[i] + torch.minimum(torch.full_like(gap, ASSEMBLY_CLOSE_MAX_DELTA_MPS), gap * ASSEMBLY_CLOSE_GAIN_PER_S)
            pace_to = closure if phase == PHASE_ASSEMBLY else torch.full_like(v[i], self.speed[i])
            raw = (pace_to - v[i]) / self.dt
            paced = raw.clamp(-BUFFER_DECEL_MPS2, COMFORT_ACCEL_MPS2)
            accel[i] = torch.where(mi, paced, accel[i])
            target[i] = torch.where(mi, v[i] + paced * self.dt, target[i])
            if phase == PHASE_ASSEMBLY and bool((torch.abs(speeds_t - self.speed[i]) <= ASSEMBLY_SPEED_TOL_MPS).all()) and span <= ASSEMBLY_SPAN_TOL_M:
                self.phase[i] = PHASE_HOLD
                self.assembled_at[i] = now_s
                events.append({"batch": i, "event": "assembled", "timestamp_s": now_s})
            owner[i, mi] = DIRECT
        return {"owner": owner, "controller": owner.clone(), "acceleration_mps2": accel,
                "target_speed_mps": target, "phase": self.phase.clone(),
                "phase_names": tuple(PHASE_NAMES[int(x)] for x in self.phase),
                "abort_reason": tuple(self.abort_reason), "events": events}
