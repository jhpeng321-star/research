"""Pure PI-external-U/1 proposal adapter.

This is an explicit operational adaptation of the published PI equations.  It
does not estimate U, write actuation commands, or import the active pipeline.
"""
from dataclasses import dataclass
from math import isfinite

from output.controller_verification_v2.pi_saturation_v1 import PISaturationParameters


VERSION = "pi_external_u/1"


def _finite(name, value):
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class UReport:
    value_mps: float
    provenance: str
    evidence_id: str
    report_timestamp_s: float | None = None
    receipt_timestamp_s: float | None = None
    max_age_s: float | None = None

    def validate(self, *, now_s=None):
        value = _finite("U", self.value_mps)
        if value < 0.0 or not self.evidence_id:
            raise ValueError("U must be nonnegative and evidence-identified")
        if self.provenance == "fixed_registered_setpoint":
            if (self.report_timestamp_s is not None or self.receipt_timestamp_s is not None
                    or self.max_age_s is not None):
                raise ValueError("fixed U must use configuration provenance, not fabricated timestamps/age")
        elif self.provenance == "timestamped_measurement":
            if self.report_timestamp_s is None or self.receipt_timestamp_s is None or self.max_age_s is None:
                raise ValueError("timestamped U requires report/receipt timestamps and max age")
            report = _finite("U report timestamp", self.report_timestamp_s)
            receipt = _finite("U receipt timestamp", self.receipt_timestamp_s)
            age = _finite("U max age", self.max_age_s)
            now = _finite("current timestamp", now_s)
            if age < 0.0 or report > receipt or receipt > now or now - report > age + 1e-12:
                raise ValueError("U report is stale or temporally invalid")
        else:
            raise ValueError("unsupported U provenance")
        return value


@dataclass(frozen=True)
class PIProposal:
    vehicle_id: str
    timestamp_s: float
    command_mps: float
    u_report: UReport
    generation: int
    token: object


class PIExternalU:
    """Published recurrence with an explicitly supplied desired-speed report."""
    def __init__(self, *, sample_period_s: float,
                 parameters: PISaturationParameters | None = None):
        self.sample_period_s = _finite("sample_period_s", sample_period_s)
        if self.sample_period_s <= 0.0:
            raise ValueError("sample_period_s must be positive")
        self.parameters = parameters or PISaturationParameters()
        p = self.parameters
        numeric = (p.lower_gap_m, p.upper_gap_m, p.catch_speed_mps, p.gamma_m, p.history_seconds)
        if not all(isfinite(float(value)) for value in numeric):
            raise ValueError("PI parameters must be finite")
        if not (0.0 < p.lower_gap_m < p.upper_gap_m and p.gamma_m > 0.0
                and p.catch_speed_mps >= 0.0 and p.history_seconds > 0.0):
            raise ValueError("invalid PI parameters")
        self._state: dict[str, tuple[float, float]] = {}
        self._generation: dict[str, int] = {}
        self._pending: dict[str, PIProposal] = {}

    def reset(self, vehicle_id: str, *, measured_ego_speed_mps: float) -> None:
        if not vehicle_id:
            raise ValueError("vehicle_id must be nonempty")
        speed = _finite("measured ego speed", measured_ego_speed_mps)
        if speed < 0.0:
            raise ValueError("measured ego speed must be nonnegative")
        self._state[vehicle_id] = (speed, float("nan"))
        self._generation[vehicle_id] = self._generation.get(vehicle_id, 0) + 1
        self._pending.pop(vehicle_id, None)

    def invalidate(self, vehicle_ids=(), *, departed_ids=(), native_ids=(), wall_ids=(),
                  missing_ids=(), stale_ids=(), id_reuse_ids=()):
        remove = (set(departed_ids) | set(native_ids) | set(wall_ids) |
                  set(missing_ids) | set(stale_ids) | set(id_reuse_ids))
        live = set(vehicle_ids)
        self._state = {key: value for key, value in self._state.items()
                       if key in live and key not in remove}
        for key in remove | (set(self._generation) - live):
            self._generation[key] = self._generation.get(key, 0) + 1
            self._pending.pop(key, None)

    def propose(self, vehicle_id: str, *, timestamp_s: float, ego_speed_mps: float,
                leader_speed_mps: float, gap_m: float, u_report: UReport) -> PIProposal:
        timestamp = _finite("timestamp_s", timestamp_s)
        ego = _finite("ego speed", ego_speed_mps)
        leader = _finite("leader speed", leader_speed_mps)
        gap = _finite("gap", gap_m)
        if min(ego, leader, gap) < 0.0 or vehicle_id not in self._state:
            raise ValueError("invalid state or vehicle must be reset")
        u = u_report.validate(now_s=timestamp)
        previous, last = self._state[vehicle_id]
        if isfinite(last) and abs(timestamp - last - self.sample_period_s) > 1e-9:
            raise ValueError("timestamp must advance by exactly one sample period")
        if vehicle_id in self._pending:
            raise RuntimeError("an uncommitted PI proposal already exists")
        p = self.parameters
        relative = leader - ego
        target = u + p.catch_speed_mps * min(max((gap - p.lower_gap_m) /
            (p.upper_gap_m - p.lower_gap_m), 0.0), 1.0)
        safety_gap = max(2.0 * relative, 4.0)
        alpha = min(max((gap - safety_gap) / p.gamma_m, 0.0), 1.0)
        beta = 1.0 - alpha / 2.0
        command = beta * (alpha * target + (1.0 - alpha) * leader) + (1.0 - beta) * previous
        if not isfinite(command):
            raise ValueError("computed PI command is nonfinite")
        proposal = PIProposal(vehicle_id, timestamp, command, u_report,
                              self._generation[vehicle_id], object())
        self._pending[vehicle_id] = proposal
        return proposal

    def commit(self, proposal: PIProposal, *, writer_ack: bool) -> float:
        current = self._pending.get(proposal.vehicle_id)
        if (proposal.vehicle_id not in self._state or current is not proposal
                or proposal.generation != self._generation.get(proposal.vehicle_id)):
            raise RuntimeError("writer did not acknowledge PI proposal")
        if writer_ack is not True:
            self._pending.pop(proposal.vehicle_id, None)
            raise RuntimeError("writer did not acknowledge PI proposal")
        self._state[proposal.vehicle_id] = (proposal.command_mps, proposal.timestamp_s)
        self._pending.pop(proposal.vehicle_id, None)
        return proposal.command_mps

    def discard(self, proposal: PIProposal) -> None:
        if self._pending.get(proposal.vehicle_id) is proposal:
            self._pending.pop(proposal.vehicle_id, None)
