"""Frozen CIRCLES singleton inference and the adopted direct command target.

Only three raw SI sensor values enter ONNX. Targets are not measurements.
"""
from dataclasses import asdict, dataclass
import hashlib
from math import isfinite
from typing import TYPE_CHECKING
import numpy as np

from .config import AdapterConfig, load_config
if TYPE_CHECKING:
    from .sensing import SensorFrame


@dataclass(frozen=True)
class DampingDecision:
    network_output_mps2: float | None = None
    command_acceleration_mps2: float | None = None
    target_speed_mps: float | None = None
    fallback_reason: str | None = None
    actor_input_si: tuple[float, float, float] | None = None

    def as_dict(self) -> dict:
        return record_dict(self)


def record_dict(record) -> dict:
    """Strict-JSON telemetry, retaining why an unavailable output is null."""
    result = asdict(record)
    for key, value in list(result.items()):
        if isinstance(value, float) and not isfinite(value):
            result[key] = None
            result[key + "_nonfinite"] = "nan" if np.isnan(value) else "positive_infinity" if value > 0 else "negative_infinity"
    return result


class CirclesController:
    def __init__(self, config: AdapterConfig | None = None):
        self.config = config or load_config()
        if hashlib.sha256(self.config.model_path.read_bytes()).hexdigest() != self.config.model_sha256:
            raise ValueError("CIRCLES model checksum mismatch")
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(str(self.config.model_path), sess_options=options,
                                             providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name

    def infer(self, row) -> float:
        """Exactly one [ego speed, leader speed, bumper gap] row; no normalization."""
        value = np.asarray(row, dtype=np.float32)
        if value.shape != (3,) or not np.isfinite(value).all() or (value < 0).any():
            raise ValueError("expected one finite nonnegative raw SI triple")
        output = np.asarray(self._session.run(None, {self._input_name: value[None, :]})[0])
        if output.size != 1:
            raise ValueError("singleton ONNX output must contain one acceleration")
        return float(output.item())

    def decide(self, frame: "SensorFrame | None", *, now_s: float) -> DampingDecision:
        reason = frame_problem(frame, now_s, self.config)
        if reason:
            return DampingDecision(fallback_reason=reason)
        if frame.front is None or min(frame.front_coverage_m, frame.rear_coverage_m) <= 0:
            return DampingDecision(fallback_reason="missing_or_uncovered_leader")
        row = (frame.speed_mps, frame.speed_mps + frame.front.relative_speed_mps, frame.front.gap_m)
        if not all(isfinite(float(x)) and x >= 0 for x in row):
            return DampingDecision(fallback_reason="invalid_actor_input")
        try:
            raw = self.infer(row)
        except Exception:
            return DampingDecision(fallback_reason="inference_error", actor_input_si=row)
        if not isfinite(raw):
            return DampingDecision(network_output_mps2=raw, fallback_reason="invalid_model_output", actor_input_si=row)
        command, target = direct_target(frame, raw, self.config)
        return DampingDecision(raw, command, target,
                               "infeasible_speed_bounds" if command is None else None, row)


def frame_problem(frame, now_s: float, config: AdapterConfig) -> str | None:
    if frame is None:
        return "missing_observation"
    if not isfinite(now_s) or not isfinite(frame.timestamp_s) or abs(frame.timestamp_s - now_s) > 1e-6:
        return "noncurrent_observation"
    values = (frame.speed_mps, frame.allowed_speed_mps, frame.front_coverage_m, frame.rear_coverage_m)
    if not all(isfinite(float(x)) and x >= 0 for x in values):
        return "invalid_observation"
    if max(frame.front_coverage_m, frame.rear_coverage_m) > config.sensor_range_m:
        return "coverage_exceeds_contract"
    for neighbor, coverage in ((frame.front, frame.front_coverage_m), (frame.rear, frame.rear_coverage_m)):
        if neighbor is not None and (not isfinite(neighbor.gap_m) or not isfinite(neighbor.relative_speed_mps)
                                     or neighbor.gap_m < 0 or neighbor.gap_m > coverage):
            return "invalid_neighbor_measurement"
    return None


def direct_target(frame, acceleration: float, config: AdapterConfig, accel_max: float | None = None):
    """Preserve the tested speed-bound clip; no IDM addition or jerk/gap projection.

    ``accel_max`` (optional) replaces the adopted 1.5 upper bound for C4 DIRECT-output commands
    only (``CavSumoEnv`` direct mode, ACCELCAP_NOTES.md); damping / wall / PI keep the config."""
    dt = config.control_dt_s
    upper = config.acceleration_max_mps2 if accel_max is None else float(accel_max)
    lo = max(config.acceleration_min_mps2, -frame.speed_mps / dt)
    hi = min(upper, (frame.allowed_speed_mps - frame.speed_mps) / dt)
    if not isfinite(acceleration) or lo > hi:
        return None, None
    command = float(np.clip(acceleration, lo, hi))
    return command, max(0., frame.speed_mps + dt * command)
