"""Versioned, sensor-local foundation for continuous learned IDM damping.

No SUMO calls, global traffic snapshot, P30 target, selector, or reward lives
here. The nominal IDM equation is a model, not SUMO's exact next acceleration.
An untouched zero-action rollout must therefore use native passthrough.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from math import isfinite, sqrt

import numpy as np


SCHEMA = "sensor_local_native_idm_residual/1"
FRAME_FIELDS = (
    "speed_over_30",
    "acceleration_over_4_5",
    "allowed_speed_over_30",
    "lane_index_over_2",
    "distance_to_merge_over_1000",
    "distance_valid",
    "previous_command_delta",
    "front_gap_over_range",
    "front_closing_over_30",
    "front_detected",
    "front_coverage_over_range",
    "rear_gap_over_range",
    "rear_closing_over_30",
    "rear_detected",
    "rear_coverage_over_range",
    "age_over_history",
    "frame_valid",
)


def _finite(**values: float) -> None:
    if not all(isfinite(float(value)) for value in values.values()):
        raise ValueError(f"nonfinite values in {tuple(values)}")


@dataclass(frozen=True)
class DampingContract:
    control_dt_s: float = 1.0
    history_frames: int = 5
    sensor_range_m: float = 100.0
    max_age_s: float = 1.0
    residual_min: float = -0.5
    residual_max: float = 0.3
    comfortable_decel: float = 1.0
    comfortable_accel: float = 0.5
    jerk_limit: float = 0.5
    front_ttc_s: float = 3.0
    rear_ttc_s: float = 4.0
    zero_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        _finite(**asdict(self))
        if self.history_frames < 1 or int(self.history_frames) != self.history_frames:
            raise ValueError("history_frames must be a positive integer")
        positive = (
            self.control_dt_s,
            self.sensor_range_m,
            self.comfortable_decel,
            self.comfortable_accel,
            self.jerk_limit,
            self.front_ttc_s,
            self.rear_ttc_s,
            self.zero_tolerance,
        )
        if min(positive) <= 0 or self.max_age_s < 0:
            raise ValueError("invalid physical bounds")
        if not self.residual_min < 0 < self.residual_max:
            raise ValueError("residual bounds must straddle zero")

    @property
    def observation_dim(self) -> int:
        return len(FRAME_FIELDS) * self.history_frames

    def manifest(self) -> dict:
        payload = {
            "schema": SCHEMA,
            "parameters": asdict(self),
            "frame_fields": FRAME_FIELDS,
            "frame_order": "oldest_to_newest",
            "zero_action": "native_passthrough_not_formula_command",
            "nominal": "local_idm_model_not_exact_sumo_acceleration",
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return {**payload, "sha256": hashlib.sha256(encoded.encode()).hexdigest()}


@dataclass(frozen=True)
class NeighborMeasurement:
    """Measured bumper gap and other-minus-ego speed; no hidden type fields."""

    gap_m: float
    relative_speed_mps: float

    def __post_init__(self) -> None:
        _finite(gap=self.gap_m, relative_speed=self.relative_speed_mps)
        if self.gap_m < 0:
            raise ValueError("overlap is invalid for this measurement contract")


@dataclass(frozen=True)
class LocalFrame:
    vehicle_id: str
    timestamp_s: float
    speed_mps: float
    acceleration_mps2: float
    allowed_speed_mps: float
    lane_index: int
    distance_to_merge_m: float | None = None
    front: NeighborMeasurement | None = None
    rear: NeighborMeasurement | None = None
    front_coverage_m: float = 0.0
    rear_coverage_m: float = 0.0
    previous_command_delta: float = 0.0

    def __post_init__(self) -> None:
        _finite(
            timestamp=self.timestamp_s,
            speed=self.speed_mps,
            acceleration=self.acceleration_mps2,
            allowed=self.allowed_speed_mps,
            front_coverage=self.front_coverage_m,
            rear_coverage=self.rear_coverage_m,
            command=self.previous_command_delta,
        )
        if (
            min(self.speed_mps, self.allowed_speed_mps, self.front_coverage_m, self.rear_coverage_m)
            < 0
        ):
            raise ValueError("speed and coverage must be nonnegative")
        if self.distance_to_merge_m is not None:
            _finite(distance=self.distance_to_merge_m)
        if self.lane_index < 0 or int(self.lane_index) != self.lane_index:
            raise ValueError("invalid measured lane index")
        for neighbor, coverage in (
            (self.front, self.front_coverage_m),
            (self.rear, self.rear_coverage_m),
        ):
            if neighbor is not None and neighbor.gap_m > coverage:
                raise ValueError("detection beyond observed coverage")


class LocalHistory:
    """Time-aligned histories with explicit missing frames; reads are pure."""

    def __init__(self, contract: DampingContract = DampingContract()):
        self.contract = contract
        self.frames: dict[str, list[LocalFrame]] = {}

    def add(self, frame: LocalFrame, *, now_s: float) -> None:
        _finite(now=now_s)
        if frame.timestamp_s > now_s + 1e-8:
            raise ValueError("future observation")
        if max(frame.front_coverage_m, frame.rear_coverage_m) > self.contract.sensor_range_m:
            raise ValueError("coverage exceeds frozen sensor range")
        rows = self.frames.setdefault(frame.vehicle_id, [])
        if rows and frame.timestamp_s <= rows[-1].timestamp_s:
            if frame == rows[-1]:
                return
            raise ValueError("nonmonotone or conflicting observation")
        rows.append(frame)
        cutoff = now_s - self.contract.control_dt_s * self.contract.history_frames
        self.frames[frame.vehicle_id] = [row for row in rows if row.timestamp_s > cutoff]

    def drop(self, vehicle_id: str) -> None:
        self.frames.pop(vehicle_id, None)

    def vector(self, vehicle_id: str, *, now_s: float) -> np.ndarray:
        _finite(now=now_s)
        cfg = self.contract
        rows = self.frames.get(vehicle_id, [])
        if rows and rows[-1].timestamp_s > now_s + 1e-8:
            raise ValueError("history queried before its latest observation")
        output = np.zeros((cfg.history_frames, len(FRAME_FIELDS)), dtype=np.float32)
        for index in range(cfg.history_frames):
            expected = now_s - (cfg.history_frames - 1 - index) * cfg.control_dt_s
            row = next((r for r in reversed(rows) if abs(r.timestamp_s - expected) < 1e-7), None)
            if row is None:
                continue
            front, rear = row.front, row.rear
            output[index] = [
                row.speed_mps / 30,
                row.acceleration_mps2 / 4.5,
                row.allowed_speed_mps / 30,
                row.lane_index / 2,
                0 if row.distance_to_merge_m is None else row.distance_to_merge_m / 1000,
                float(row.distance_to_merge_m is not None),
                row.previous_command_delta,
                0 if front is None else front.gap_m / cfg.sensor_range_m,
                0 if front is None else -front.relative_speed_mps / 30,
                float(front is not None),
                row.front_coverage_m / cfg.sensor_range_m,
                0 if rear is None else rear.gap_m / cfg.sensor_range_m,
                0 if rear is None else rear.relative_speed_mps / 30,
                float(rear is not None),
                row.rear_coverage_m / cfg.sensor_range_m,
                (now_s - row.timestamp_s) / (cfg.control_dt_s * cfg.history_frames),
                1,
            ]
        return output.ravel()


@dataclass(frozen=True)
class OwnIDMParameters:
    """Known CAV controller parameters, never parameters of a detected HDV."""

    max_accel: float
    comfortable_decel: float
    tau_s: float
    min_gap_m: float
    emergency_decel: float
    exponent: float = 4.0

    def __post_init__(self) -> None:
        _finite(**asdict(self))
        if (
            min(
                self.max_accel,
                self.comfortable_decel,
                self.tau_s,
                self.emergency_decel,
                self.exponent,
            )
            <= 0
            or self.min_gap_m < 0
        ):
            raise ValueError("invalid own IDM parameters")


def nominal_idm_acceleration(frame: LocalFrame, own: OwnIDMParameters) -> float:
    """Finite local IDM approximation; missing distant traffic stays unobserved."""
    v = frame.speed_mps
    if frame.allowed_speed_mps <= 1e-8:
        return -own.emergency_decel if v > 0 else 0.0
    interaction = 0.0
    if frame.front is not None:
        gap = max(frame.front.gap_m, 1e-6)
        closing = -frame.front.relative_speed_mps
        desired = own.min_gap_m + max(
            0.0, v * own.tau_s + v * closing / (2 * sqrt(own.max_accel * own.comfortable_decel))
        )
        interaction = (desired / gap) ** 2
    result = own.max_accel * (1 - (v / frame.allowed_speed_mps) ** own.exponent - interaction)
    return float(np.clip(result, -own.emergency_decel, own.max_accel))


@dataclass(frozen=True)
class CommandDecision:
    requested_delta: float
    bounded_delta: float
    nominal_acceleration: float
    target_acceleration: float | None
    target_speed: float | None
    projected_delta: float
    reasons: tuple[str, ...]


def project_residual(
    frame: LocalFrame,
    own: OwnIDMParameters,
    requested_delta: float,
    *,
    now_s: float,
    contract: DampingContract = DampingContract(),
) -> CommandDecision:
    """Intersect command bounds once; conflicts fall back to native safety.

    Bounds describe issued targets, not guaranteed realized SUMO acceleration.
    The actuator's native safety can override them. No inherited P30 selector,
    follow-speed equality cap, or five-second action filter is imposed here.
    """
    _finite(action=requested_delta, now=now_s)
    cfg = contract
    bounded = float(np.clip(requested_delta, cfg.residual_min, cfg.residual_max))
    nominal = nominal_idm_acceleration(frame, own)
    reasons: list[str] = []

    def native(reason: str) -> CommandDecision:
        return CommandDecision(
            requested_delta, bounded, nominal, None, None, 0.0, tuple([*reasons, reason])
        )

    age = now_s - frame.timestamp_s
    if age < -1e-8:
        raise ValueError("future observation")
    if age > cfg.max_age_s:
        return native("stale_observation")
    if abs(requested_delta) <= cfg.zero_tolerance:
        return native("zero_native_passthrough")
    if min(frame.front_coverage_m, frame.rear_coverage_m) <= 0:
        return native("unavailable_sensor_coverage")
    if bounded != requested_delta:
        reasons.append("residual_clip")
    dt, v = cfg.control_dt_s, frame.speed_mps
    lower = max(
        nominal + cfg.residual_min,
        -cfg.comfortable_decel,
        frame.acceleration_mps2 - cfg.jerk_limit * dt,
        -v / dt,
    )
    upper = min(
        nominal + cfg.residual_max,
        cfg.comfortable_accel,
        frame.acceleration_mps2 + cfg.jerk_limit * dt,
        (frame.allowed_speed_mps - v) / dt,
    )
    if frame.front is not None:
        upper = min(
            upper,
            (
                frame.front.relative_speed_mps
                + max(0.0, frame.front.gap_m - own.min_gap_m) / cfg.front_ttc_s
            )
            / dt,
        )
    if frame.rear is not None:
        lower = max(lower, (frame.rear.relative_speed_mps - frame.rear.gap_m / cfg.rear_ttc_s) / dt)
    if lower > upper:
        return native("infeasible_command_interval")
    target = float(np.clip(nominal + bounded, lower, upper))
    effective = target - nominal
    if abs(effective) <= cfg.zero_tolerance:
        return native("projection_to_native")
    if effective * bounded < 0:
        return native("projection_would_reverse_action")
    if abs(effective - bounded) > cfg.zero_tolerance:
        reasons.append("safety_projection")
    return CommandDecision(
        requested_delta,
        bounded,
        nominal,
        target,
        max(0.0, v + target * dt),
        effective,
        tuple(reasons),
    )
