"""Causal, phase-resolved diagnostics for IDM damping treatment replays.

The helpers in this module are deliberately outside the controller.  They do
not alter observations, action selection, rewards, or safety projection.  Their
only purpose is to make an already-frozen paired treatment auditable at each
post-decision step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import math


PHASE_DIAGNOSTIC_SCHEMA = "idm_phase_response/1"
PHASES = ("inactive", "probe", "hold", "release", "post_release")


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


@dataclass
class TreatmentPhaseTracker:
    """Assign diagnostic phases from realized commands and member changes.

    ``hold`` labels the current fixed latched option. ``probe`` is available
    only when a future treatment explicitly opts into a positive
    ``probe_steps`` contract. ``release`` begins on the first step with no
    command after treatment, including a step carrying an explicit
    completion/abort member change. These labels are an analysis contract only;
    they never cause early release.
    """

    probe_steps: int = 0
    release_steps: int = 5
    _first_active_step: int | None = field(default=None, init=False)
    _last_active_step: int | None = field(default=None, init=False)
    _release_start_step: int | None = field(default=None, init=False)
    _release_reason: str | None = field(default=None, init=False)
    _previously_active: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.probe_steps < 0:
            raise ValueError("probe_steps must be nonnegative")
        if self.release_steps < 0:
            raise ValueError("release_steps must be nonnegative")

    def observe(
        self,
        *,
        relative_step: int,
        active_vehicle_ids: Iterable[str],
        member_changes: Iterable[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        step = int(relative_step)
        if step < 1:
            raise ValueError("relative_step must be at least one")
        active_ids = tuple(sorted({str(value) for value in active_vehicle_ids}))
        changes = [dict(value) for value in member_changes]
        active = bool(active_ids)

        if active:
            if self._first_active_step is None:
                self._first_active_step = step
            self._last_active_step = step
            self._previously_active = True
            self._release_start_step = None
            self._release_reason = None
            active_age = step - self._first_active_step + 1
            phase = (
                "probe"
                if self.probe_steps > 0 and active_age <= self.probe_steps
                else "hold"
            )
        else:
            reason = next(
                (
                    str(change.get("reason"))
                    for change in changes
                    if change.get("reason") is not None
                ),
                None,
            )
            if self._previously_active and self._release_start_step is None:
                self._release_start_step = step
                self._release_reason = reason or "command_ended"
            in_release = bool(
                self._release_start_step is not None
                and step - self._release_start_step < self.release_steps
            )
            if in_release:
                phase = "release"
            elif self._first_active_step is None:
                phase = "inactive"
            else:
                phase = "post_release"
            active_age = None

        return {
            "schema": PHASE_DIAGNOSTIC_SCHEMA,
            "relative_step": step,
            "phase": phase,
            "active_vehicle_ids": list(active_ids),
            "active_vehicle_count": len(active_ids),
            "active_age_steps": active_age,
            "first_active_step": self._first_active_step,
            "last_active_step": self._last_active_step,
            "release_start_step": self._release_start_step,
            "release_reason": self._release_reason,
        }


def upstream_follower_states(
    snapshot: Any,
    target_vehicle_id: str,
    *,
    count: int = 5,
) -> list[dict[str, Any]]:
    """Return the nearest same-lane upstream followers in causal order."""
    if count < 0:
        raise ValueError("count must be nonnegative")
    target_id = str(target_vehicle_id)
    if count == 0 or not snapshot.has(target_id):
        return []
    target_lane = snapshot.lane_id(target_id)
    target_position = snapshot.lane_pos(target_id)
    if target_lane is None or target_position is None:
        return []

    candidates: list[tuple[float, str]] = []
    for vehicle_id in snapshot.ordered_ids():
        vehicle_id = str(vehicle_id)
        if vehicle_id == target_id or snapshot.lane_id(vehicle_id) != target_lane:
            continue
        position = snapshot.lane_pos(vehicle_id)
        if position is None or float(position) >= float(target_position):
            continue
        candidates.append((float(position), vehicle_id))
    candidates.sort(reverse=True)

    states: list[dict[str, Any]] = []
    leader_id = target_id
    leader_position = float(target_position)
    leader_speed = _finite_float(snapshot.speed(target_id))
    for rank, (position, vehicle_id) in enumerate(candidates[:count], start=1):
        speed = _finite_float(snapshot.speed(vehicle_id))
        acceleration = _finite_float(snapshot.accel(vehicle_id))
        gap = max(leader_position - position - 5.0, 0.0)
        closing_speed = None if speed is None or leader_speed is None else speed - leader_speed
        ttc = (
            gap / closing_speed
            if closing_speed is not None and closing_speed > 1e-6
            else None
        )
        states.append(
            {
                "vehicle_id": vehicle_id,
                "relation_to_target": f"upstream_follower_{rank}",
                "leader_vehicle_id": leader_id,
                "road_id": snapshot.road(vehicle_id),
                "lane_id": target_lane,
                "lane_position_m": position,
                "speed_mps": speed,
                "acceleration_mps2": acceleration,
                "gap_to_leader_m": gap,
                "closing_speed_mps": closing_speed,
                "ttc_s": ttc,
            }
        )
        leader_id = vehicle_id
        leader_position = position
        leader_speed = speed
    return states


def vehicle_safety_events(
    snapshot: Any,
    vehicle_ids: Iterable[str],
    *,
    relative_step: int,
    simulation_time_s: float,
    relation_by_vehicle: Mapping[str, str] | None = None,
    severe_threshold_mps2: float = -3.0,
    hard_threshold_mps2: float = -4.5,
    low_ttc_threshold_s: float = 1.5,
) -> list[dict[str, Any]]:
    """Record every vehicle-level braking or low-TTC event for one step."""
    relations = relation_by_vehicle or {}
    events: list[dict[str, Any]] = []
    for raw_id in sorted({str(value) for value in vehicle_ids}):
        vehicle_id = str(raw_id)
        if not snapshot.has(vehicle_id):
            continue
        acceleration = _finite_float(snapshot.accel(vehicle_id))
        speed = _finite_float(snapshot.speed(vehicle_id))
        lane_position = _finite_float(snapshot.lane_pos(vehicle_id))
        leader = snapshot.leader(vehicle_id)
        leader_id = str(leader[0]) if leader and leader[0] else None
        leader_gap = _finite_float(leader[1]) if leader else None
        leader_speed = (
            _finite_float(snapshot.speed(leader_id)) if leader_id is not None else None
        )
        closing_speed = None if speed is None or leader_speed is None else speed - leader_speed
        ttc = (
            max(leader_gap, 0.0) / closing_speed
            if leader_gap is not None and closing_speed is not None and closing_speed > 1e-6
            else None
        )
        severe = acceleration is not None and acceleration < severe_threshold_mps2
        hard = acceleration is not None and acceleration < hard_threshold_mps2
        low_ttc = ttc is not None and ttc < low_ttc_threshold_s
        if not (severe or hard or low_ttc):
            continue
        events.append(
            {
                "relative_step": int(relative_step),
                "simulation_time_s": float(simulation_time_s),
                "vehicle_id": vehicle_id,
                "relation_to_target": relations.get(vehicle_id, "corridor_other"),
                "road_id": snapshot.road(vehicle_id),
                "lane_id": snapshot.lane_id(vehicle_id),
                "lane_position_m": lane_position,
                "speed_mps": speed,
                "acceleration_mps2": acceleration,
                "leader_vehicle_id": leader_id,
                "leader_gap_m": leader_gap,
                "ttc_s": ttc,
                "severe_braking": bool(severe),
                "hard_braking": bool(hard),
                "low_ttc": bool(low_ttc),
            }
        )
    return events


_CUMULATIVE_METRICS = (
    "severe_braking_count",
    "hard_braking_count",
    "low_ttc_count",
    "acceleration_energy_m2ps4",
    "integrated_negative_acceleration_mps",
    "arrived_this_step",
    "merge_discharge_this_step",
)


def phase_step_diagnostic(
    phase_row: Mapping[str, Any],
    spatial_trace: Mapping[str, Any],
    *,
    merge_discharge_this_step: int = 0,
) -> dict[str, Any]:
    """Flatten one spatial trace into the versioned phase-audit schema."""
    corridor = dict(spatial_trace.get("corridor") or {})
    merge_service = dict(spatial_trace.get("merge_service") or {})
    return {
        **dict(phase_row),
        "schema": PHASE_DIAGNOSTIC_SCHEMA,
        "simulation_time_s": spatial_trace.get("simulation_time_s"),
        "snapshot_available": bool(spatial_trace.get("snapshot_available", False)),
        "severe_braking_count": int(corridor.get("severe_braking_count") or 0),
        "hard_braking_count": int(corridor.get("hard_braking_count") or 0),
        "low_ttc_count": int(corridor.get("low_ttc_count") or 0),
        "acceleration_energy_m2ps4": float(
            corridor.get("acceleration_energy_m2ps4") or 0.0
        ),
        "integrated_negative_acceleration_mps": float(
            corridor.get("integrated_negative_acceleration_mps") or 0.0
        ),
        "minimum_ttc_s": corridor.get("minimum_ttc_s"),
        "corridor_delay_rate_s_per_vehicle": corridor.get(
            "delay_rate_s_per_vehicle"
        ),
        "arrived_this_step": int(merge_service.get("arrived_this_step") or 0),
        "merge_discharge_this_step": int(merge_discharge_this_step),
        "collapse_condition": bool(merge_service.get("collapse_condition", False)),
        "target_states": list(spatial_trace.get("absorbers") or []),
        "upstream_followers": list(spatial_trace.get("upstream_followers") or []),
        "safety_events": list(spatial_trace.get("safety_events") or []),
    }


def cumulative_paired_deltas(
    control_rows: Sequence[Mapping[str, Any]],
    treatment_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Align per-step rows and return cumulative treatment-minus-control deltas."""
    control_by_step = {int(row["relative_step"]): row for row in control_rows}
    treatment_by_step = {int(row["relative_step"]): row for row in treatment_rows}
    if set(control_by_step) != set(treatment_by_step):
        raise ValueError("control and treatment diagnostic steps must match exactly")

    cumulative = {metric: 0.0 for metric in _CUMULATIVE_METRICS}
    output: list[dict[str, Any]] = []
    for step in sorted(control_by_step):
        control = control_by_step[step]
        treatment = treatment_by_step[step]
        step_delta: dict[str, float] = {}
        for metric in _CUMULATIVE_METRICS:
            control_value = _finite_float(control.get(metric)) or 0.0
            treatment_value = _finite_float(treatment.get(metric)) or 0.0
            delta = treatment_value - control_value
            step_delta[metric] = delta
            cumulative[metric] += delta
        output.append(
            {
                "relative_step": step,
                "phase": treatment.get("phase", "inactive"),
                "step_treatment_minus_control": step_delta,
                "cumulative_treatment_minus_control": dict(cumulative),
            }
        )
    return output
