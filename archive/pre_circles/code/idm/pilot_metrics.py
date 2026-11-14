"""Trajectory and controller-dose metrics for the frozen IDM pilot."""

from __future__ import annotations

from collections import defaultdict
from math import inf
from pathlib import Path
from typing import Iterable

import numpy as np

from config.constants import DEFAULT_VEH_LENGTH, MERGE_EDGE, PRE_MERGE_EDGE
from evaluation import parse_fcd

DOWNSTREAM_EDGE = "1487606178"
CORRIDOR_EDGES = {PRE_MERGE_EDGE, MERGE_EDGE, DOWNSTREAM_EDGE}


def _edge(lane_id: str) -> str:
    return lane_id.rsplit("_", 1)[0]


def trajectory_metrics(
    fcd_path: Path,
    control_events: Iterable[dict] = (),
    *,
    low_speed_mps: float = 15.0,
    severe_brake_mps2: float = -3.0,
) -> dict:
    """Audit all-vehicle trajectories; no policy-side counters are used."""
    frames = parse_fcd(Path(fcd_path))
    prior: dict[str, tuple[float, float, float | None]] = {}
    acceleration: dict[str, dict[float, float]] = defaultdict(dict)
    oscillation_energy = 0.0
    integrated_negative_accel = 0.0
    integrated_abs_jerk = 0.0
    severe_events = 0
    hard_events = 0
    minimum_ttc = inf
    low_times: list[float] = []
    exposed: set[str] = set()
    observed_vehicles: set[str] = set()
    upstream_vars: list[float] = []
    downstream_vars: list[float] = []
    upstream_means: list[float] = []
    spillback_times: list[float] = []
    minimum_speed = inf
    collapse_run_steps = 0
    collapse_max_steps = 0
    collapse_onset_s: float | None = None

    times = sorted(frames)
    nominal_dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    for time in times:
        vehicles = frames[time]
        observed_vehicles.update(vehicle["id"] for vehicle in vehicles)
        upstream = [v for v in vehicles if _edge(v["lane"]) == PRE_MERGE_EDGE]
        if upstream:
            speeds = np.asarray([v["speed"] for v in upstream], dtype=float)
            upstream_vars.append(float(np.var(speeds)))
            upstream_means.append(float(np.mean(speeds)))
            minimum_speed = min(minimum_speed, float(np.min(speeds)))
            slow_count = int(np.sum(speeds < low_speed_mps))
            collapse_step = float(np.mean(speeds)) <= low_speed_mps and slow_count >= 5
            if float(np.mean(speeds)) <= low_speed_mps:
                low_times.append(time)
            if collapse_step:
                collapse_run_steps += 1
                collapse_max_steps = max(collapse_max_steps, collapse_run_steps)
                if collapse_run_steps == 60:
                    collapse_onset_s = time - 59.0 * nominal_dt
            else:
                collapse_run_steps = 0
            exposed.update(v["id"] for v in upstream if v["speed"] <= low_speed_mps)
            if any(v["pos"] <= 100.0 for v in upstream):
                spillback_times.append(time)
        if not upstream:
            collapse_run_steps = 0

        lane_groups: dict[str, list[dict]] = defaultdict(list)
        for vehicle in vehicles:
            if _edge(vehicle["lane"]) in CORRIDOR_EDGES:
                lane_groups[vehicle["lane"]].append(vehicle)
            previous = prior.get(vehicle["id"])
            if previous is None:
                prior[vehicle["id"]] = (time, vehicle["speed"], None)
                continue
            old_time, old_speed, old_accel = previous
            dt = time - old_time
            if dt <= 0:
                continue
            accel = (vehicle["speed"] - old_speed) / dt
            acceleration[vehicle["id"]][time] = accel
            if _edge(vehicle["lane"]) in CORRIDOR_EDGES:
                oscillation_energy += accel * accel * dt
                integrated_negative_accel += max(-accel, 0.0) * dt
                severe_events += int(accel < severe_brake_mps2)
                hard_events += int(accel < -4.5)
                if old_accel is not None:
                    integrated_abs_jerk += abs((accel - old_accel) / dt) * dt
            prior[vehicle["id"]] = (time, vehicle["speed"], accel)

        for lane_rows in lane_groups.values():
            lane_rows.sort(key=lambda row: row["pos"])
            for follower, leader in zip(lane_rows, lane_rows[1:]):
                closing = follower["speed"] - leader["speed"]
                if closing <= 0:
                    continue
                gap = leader["pos"] - follower["pos"] - DEFAULT_VEH_LENGTH
                ttc = 0.0 if gap <= 0 else gap / closing
                minimum_ttc = min(minimum_ttc, ttc)

    follower_min_accels: list[float] = []
    follower_severe_events = 0
    for event in control_events:
        follower = event.get("follower_id")
        if not follower:
            continue
        start = float(event["time_s"])
        values = [
            accel
            for time, accel in acceleration.get(follower, {}).items()
            if start < time <= start + 5.0
        ]
        if values:
            follower_min_accels.append(min(values))
            follower_severe_events += int(min(values) < severe_brake_mps2)

    upstream_variance = float(np.mean(upstream_vars)) if upstream_vars else float("nan")
    downstream_variance = float(np.mean(downstream_vars)) if downstream_vars else float("nan")
    low_duration = len(low_times) * nominal_dt
    propagation_duration = low_times[-1] - low_times[0] + nominal_dt if low_times else 0.0
    return {
        "definitions": {
            "low_speed_threshold_mps": low_speed_mps,
            "severe_brake_threshold_mps2": severe_brake_mps2,
            "hard_brake_threshold_mps2": -4.5,
            "spillback_position_threshold_m": 100.0,
            "follower_response_window_s": 5.0,
            "collapse_slow_vehicle_count": 5,
            "collapse_sustained_steps": 60,
        },
        "wave": {
            "upstream_speed_variance_m2ps2": upstream_variance,
            "downstream_speed_variance_m2ps2": downstream_variance,
            "variance_amplification": (
                downstream_variance / upstream_variance
                if upstream_variance > 1e-12
                else float("nan")
            ),
            "oscillation_energy_m2ps3": oscillation_energy,
            "low_speed_wave_duration_s": low_duration,
            "minimum_premerge_speed_mps": (minimum_speed if minimum_speed < inf else float("nan")),
            "propagation_duration_s": propagation_duration,
            "vehicles_exposed": len(exposed),
            "sustained_collapse": collapse_max_steps >= 60,
            "collapse_max_consecutive_steps": collapse_max_steps,
            "collapse_onset_s": collapse_onset_s,
            "mean_premerge_speed_mps": (
                float(np.mean(upstream_means)) if upstream_means else float("nan")
            ),
        },
        "safety": {
            "minimum_longitudinal_ttc_s": (minimum_ttc if minimum_ttc < inf else float("inf")),
            "severe_braking_events": severe_events,
            "hard_braking_events_recomputed": hard_events,
            "integrated_negative_acceleration_mps": integrated_negative_accel,
            "integrated_absolute_jerk_mps2": integrated_abs_jerk,
            "follower_response_samples": len(follower_min_accels),
            "follower_min_acceleration_mps2": (
                min(follower_min_accels) if follower_min_accels else float("nan")
            ),
            "follower_severe_response_events": follower_severe_events,
        },
        "service": {
            "spillback_duration_s": len(spillback_times) * nominal_dt,
            "vehicles_observed": len(observed_vehicles),
        },
    }


def summarize_control(history: Iterable[dict]) -> tuple[dict, list[dict]]:
    """Aggregate engine diagnostics and return follower audit events."""
    controlled_seconds = 0.0
    unique_absorbers: set[str] = set()
    max_simultaneous = 0
    magnitudes: list[float] = []
    projected = 0
    clipped = 0
    conflicts = 0
    handoffs = 0
    member_changes = 0
    follower_events: list[dict] = []
    reason_counts: dict[str, int] = defaultdict(int)
    nominal_accelerations: list[float] = []
    filtered_references: list[float] = []
    reference_errors: list[float] = []
    for row in history:
        controlled_seconds += float(row.get("controlled_cav_seconds", 0.0))
        max_simultaneous = max(max_simultaneous, int(row.get("simultaneous_absorber_count", 0)))
        time_s = float(row.get("time_s", row.get("step", 0.0)))
        for command in row.get("commands", []):
            unique_absorbers.add(command["vehicle_id"])
            projection = command.get("projected", {})
            accel = float(projection.get("acceleration_mps2", 0.0))
            magnitudes.append(abs(accel))
            projected += 1
            reasons = projection.get("reasons", ())
            clipped += int(bool(reasons))
            for reason in reasons:
                reason_counts[str(reason)] += 1
            conflicts += int(bool(projection.get("constraint_conflict", False)))
            nominal_accelerations.append(float(command.get("nominal_acceleration_mps2", 0.0)))
            reference = command.get("reference", {})
            if "filtered_reference_mps" in reference:
                filtered_references.append(float(reference["filtered_reference_mps"]))
                reference_errors.append(
                    float(reference.get("raw_reference_mps", 0.0))
                    - float(reference["filtered_reference_mps"])
                )
            leader = command.get("leader", {})
            follower_events.append(
                {
                    "time_s": time_s,
                    "absorber_id": command["vehicle_id"],
                    "follower_id": leader.get("follower_id"),
                }
            )
        for change in row.get("member_changes", []):
            member_changes += 1
            handoffs += int(change.get("reason") == "handoff")
    return {
        "controlled_cav_seconds": controlled_seconds,
        "unique_absorber_count": len(unique_absorbers),
        "max_simultaneous_absorbers": max_simultaneous,
        "mean_absolute_command_mps2": (float(np.mean(magnitudes)) if magnitudes else 0.0),
        "projection_count": projected,
        "projection_constraint_frequency": clipped / projected if projected else 0.0,
        "front_follower_conflicts": conflicts,
        "member_changes": member_changes,
        "handoffs": handoffs,
        "projection_reason_counts": dict(sorted(reason_counts.items())),
        "mean_nominal_acceleration_mps2": (
            float(np.mean(nominal_accelerations)) if nominal_accelerations else 0.0
        ),
        "mean_filtered_reference_mps": (
            float(np.mean(filtered_references)) if filtered_references else float("nan")
        ),
        "mean_raw_minus_filtered_reference_mps": (
            float(np.mean(reference_errors)) if reference_errors else float("nan")
        ),
    }, follower_events
