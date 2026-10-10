"""Published-2018 PI-with-saturation speed-target transcription."""
from collections import deque
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class PISaturationParameters:
    lower_gap_m: float = 7.0
    upper_gap_m: float = 30.0
    catch_speed_mps: float = 1.0
    gamma_m: float = 2.0
    history_seconds: float = 38.0


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _nonnegative(name: str, value: float) -> float:
    value = _finite(name, value)
    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


class PISaturationController:
    """Equations (3)-(5), with published 2018 zero-filled 38 s startup.

    Callers provide exactly one valid sample per ``sample_period_s``.
    """
    def __init__(self, *, sample_period_s: float,
                 parameters: PISaturationParameters | None = None):
        self.sample_period_s = _finite("sample_period_s", sample_period_s)
        if self.sample_period_s <= 0.0:
            raise ValueError("sample_period_s must be positive")
        self.parameters = parameters or PISaturationParameters()
        p = self.parameters
        numeric = (p.lower_gap_m, p.upper_gap_m, p.catch_speed_mps,
                   p.gamma_m, p.history_seconds)
        if not all(isfinite(float(value)) for value in numeric):
            raise ValueError("PI-saturation parameters must be finite")
        if not (0.0 < p.lower_gap_m < p.upper_gap_m and p.gamma_m > 0.0
                and p.catch_speed_mps >= 0.0 and p.history_seconds > 0.0):
            raise ValueError("invalid PI-saturation parameters")
        self._max_samples = max(1, round(p.history_seconds / self.sample_period_s))
        self._state: dict[str, tuple[deque[float], float, float | None]] = {}

    def reset(self, vehicle_id: str, *, initial_command_mps: float = 0.0,
              timestamp_s: float | None = None) -> None:
        if not vehicle_id:
            raise ValueError("vehicle_id must be nonempty")
        initial = _nonnegative("initial_command_mps", initial_command_mps)
        last = None if timestamp_s is None else _finite("timestamp_s", timestamp_s)
        self._state[vehicle_id] = (deque([0.0] * self._max_samples,
                                          maxlen=self._max_samples), initial, last)

    def step(self, vehicle_id: str, *, timestamp_s: float,
             ego_speed_mps: float, leader_speed_mps: float, gap_m: float) -> float:
        """Return the command, committing state only after all arithmetic succeeds."""
        timestamp = _finite("timestamp_s", timestamp_s)
        ego = _nonnegative("ego_speed_mps", ego_speed_mps)
        leader = _nonnegative("leader_speed_mps", leader_speed_mps)
        gap = _nonnegative("gap_m", gap_m)
        if vehicle_id not in self._state:
            raise ValueError("vehicle must be reset before first step")
        history, previous, last = self._state[vehicle_id]
        if last is not None and abs(timestamp - (last + self.sample_period_s)) > 1e-9:
            raise ValueError("timestamp must advance by exactly one sample period")
        candidate_history = deque(history, maxlen=self._max_samples)
        candidate_history.append(ego)
        p = self.parameters
        desired = sum(candidate_history) / len(candidate_history)
        relative = leader - ego
        target = desired + p.catch_speed_mps * min(
            max((gap - p.lower_gap_m) / (p.upper_gap_m - p.lower_gap_m), 0.0), 1.0)
        safety_gap = max(2.0 * relative, 4.0)
        alpha = min(max((gap - safety_gap) / p.gamma_m, 0.0), 1.0)
        beta = 1.0 - 0.5 * alpha
        command = beta * (alpha * target + (1.0 - alpha) * leader) + (1.0 - beta) * previous
        if not isfinite(command):
            raise ValueError("computed command is nonfinite")
        self._state[vehicle_id] = (candidate_history, command, timestamp)
        return command
