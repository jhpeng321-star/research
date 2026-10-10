"""Bounded telemetry and endpoint classification for the IDM density mapper.

This module is deliberately independent of any wall-treatment or predictor
code.  It consumes an exact damping replay, keeps only anonymous 1 Hz
aggregates, and separates congestion already prevalent at the absolute SUMO
anchor from congestion that becomes sustained afterwards.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from statistics import mean, pstdev
from typing import Any, Callable, Mapping, Sequence

from config.constants import (
    DEFAULT_VEH_LENGTH,
    DOWNSTREAM_EDGE,
    DOWNSTREAM_EDGE_NUM_LANES,
    DOWNSTREAM_LANE_LENGTH,
    MERGE_DROPPED_LANE_LENGTH,
    MERGE_EDGE,
    MERGE_EDGE_NUM_LANES,
    PRE_MERGE_EDGE,
    PRE_MERGE_LANE_LENGTH,
    RAMP_EDGE,
    RAMP_LANE_LENGTH,
)
from counterfactual.outcomes import OutcomeDefinition, step_metrics
from counterfactual.twin_replay import ReplayTrace, StepSnapshot


PRE_MERGE_LANES = 3
RAMP_LANES = 1
_REGIONS = {
    "premerge": (PRE_MERGE_EDGE, PRE_MERGE_LANE_LENGTH, PRE_MERGE_LANES),
    "ramp": (RAMP_EDGE, RAMP_LANE_LENGTH, RAMP_LANES),
    "merge": (MERGE_EDGE, MERGE_DROPPED_LANE_LENGTH, MERGE_EDGE_NUM_LANES),
    "downstream": (
        DOWNSTREAM_EDGE,
        DOWNSTREAM_LANE_LENGTH,
        DOWNSTREAM_EDGE_NUM_LANES,
    ),
}


def _number(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _mean(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return mean(finite) if finite else None


def _std(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return pstdev(finite) if len(finite) >= 2 else (0.0 if finite else None)


def _slope(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    pairs = [
        (_number(row.get("sim_time_s")), _number(row.get(key)))
        for row in rows
        if math.isfinite(_number(row.get("sim_time_s")))
        and math.isfinite(_number(row.get(key)))
    ]
    if len(pairs) < 2:
        return None
    xbar = mean(value[0] for value in pairs)
    ybar = mean(value[1] for value in pairs)
    denominator = sum((x - xbar) ** 2 for x, _ in pairs)
    if denominator <= 0.0:
        return 0.0
    return sum((x - xbar) * (y - ybar) for x, y in pairs) / denominator


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _vehicle_map(snapshot: StepSnapshot) -> dict[str, tuple[Any, ...]]:
    return {str(row[0]): row for row in snapshot.vehicles}


def _audit_map(trace: ReplayTrace) -> dict[int, Mapping[str, Any]]:
    return {
        int(row["relative_step"]): row
        for row in trace.controller_audits
        if row.get("relative_step") is not None
    }


def _region_values(snapshot: StepSnapshot, edge: str) -> list[tuple[Any, ...]]:
    return [row for row in snapshot.vehicles if str(row[1]) == edge]


def _region_metrics(rows: Sequence[tuple[Any, ...]], length_m: float, lanes: int) -> dict:
    speeds = [_number(row[4]) for row in rows]
    accelerations = [_number(row[5]) for row in rows]
    road_km = max(float(length_m) * int(lanes) / 1000.0, 1e-9)
    road_space_m = max(float(length_m) * int(lanes), 1e-9)
    return {
        "count": len(rows),
        "density_veh_per_lane_km": len(rows) / road_km,
        "occupancy_proxy": min(1.0, len(rows) * DEFAULT_VEH_LENGTH / road_space_m),
        "mean_speed_mps": _mean(speeds),
        "min_speed_mps": min(speeds) if speeds else None,
        "speed_std_mps": _std(speeds),
        "mean_acceleration_mps2": _mean(accelerations),
    }


def _command_metrics(
    snapshot: StepSnapshot, audit: Mapping[str, Any] | None
) -> dict[str, Any]:
    audit = audit or {}
    diagnostics = audit.get("base_diagnostics") or {}
    commands = [row for row in diagnostics.get("commands", ()) if isinstance(row, Mapping)]
    vehicles = _vehicle_map(snapshot)
    projected: list[float] = []
    realized: list[float] = []
    projected_minus_observed: list[float] = []
    requested_deltas: list[float] = []
    for command in commands:
        projection = command.get("projected") or {}
        target_acceleration = _number(projection.get("acceleration_mps2"))
        vehicle = vehicles.get(str(command.get("vehicle_id")))
        actual_acceleration = _number(vehicle[5]) if vehicle is not None else math.nan
        observed_before = _number(command.get("projection_input", {}).get("prior_acceleration_mps2"))
        if math.isfinite(target_acceleration):
            projected.append(target_acceleration)
        if math.isfinite(actual_acceleration):
            realized.append(actual_acceleration)
        if math.isfinite(target_acceleration) and math.isfinite(observed_before):
            requested_deltas.append(target_acceleration - observed_before)
        if math.isfinite(target_acceleration) and math.isfinite(actual_acceleration):
            projected_minus_observed.append(actual_acceleration - target_acceleration)
    return {
        "damping_command_count": len(commands),
        "damping_controlled_cav_seconds": float(len(commands)),
        "damping_projected_acceleration_mean_mps2": _mean(projected),
        "damping_realized_acceleration_mean_mps2": _mean(realized),
        "damping_requested_delta_mean_mps2": _mean(requested_deltas),
        "damping_tracking_error_mean_mps2": _mean(projected_minus_observed),
    }


def build_telemetry(
    trace: ReplayTrace,
    *,
    definition: OutcomeDefinition,
    primary_queue_count: int,
    primary_speed_threshold_mps: float,
    wall_band_lo_m: float = 200.0,
    wall_band_hi_m: float = 1000.0,
) -> list[dict[str, Any]]:
    """Convert a full trace to bounded anonymous 1 Hz aggregate telemetry."""

    audits = _audit_map(trace)
    result: list[dict[str, Any]] = []
    previous: dict[str, tuple[Any, ...]] = {}
    seen: set[str] = set()
    seen_cavs: set[str] = set()
    completed_ids: set[str] = set()

    for snapshot in trace.snapshots:
        current = _vehicle_map(snapshot)
        active_cavs = set(str(value) for value in snapshot.active_cavs)
        seen_cavs.update(active_cavs)
        new_ids = set(current) - seen
        disappeared = set(previous) - set(current)
        completed_ids.update(
            vehicle_id
            for vehicle_id in disappeared
            if str(previous[vehicle_id][1]) == DOWNSTREAM_EDGE
        )
        lane_changes = sum(
            1
            for vehicle_id, row in current.items()
            if vehicle_id in previous and str(previous[vehicle_id][2]) != str(row[2])
        )
        mainline_merge_entries = sum(
            1
            for vehicle_id, row in current.items()
            if vehicle_id in previous
            and str(previous[vehicle_id][1]) == PRE_MERGE_EDGE
            and str(row[1]) == MERGE_EDGE
        )
        ramp_merge_entries = sum(
            1
            for vehicle_id, row in current.items()
            if vehicle_id in previous
            and str(previous[vehicle_id][1]) == RAMP_EDGE
            and str(row[1]) == MERGE_EDGE
        )
        downstream_discharges = sum(
            1
            for vehicle_id, row in current.items()
            if vehicle_id in previous
            and str(previous[vehicle_id][1]) == MERGE_EDGE
            and str(row[1]) == DOWNSTREAM_EDGE
        )

        primary = step_metrics(snapshot, definition)
        regions = {
            name: _region_values(snapshot, edge)
            for name, (edge, _length, _lanes) in _REGIONS.items()
        }
        metrics = {
            name: _region_metrics(regions[name], length, lanes)
            for name, (_edge, length, lanes) in _REGIONS.items()
        }
        slow_premerge = [
            row
            for row in regions["premerge"]
            if _number(row[4]) < float(primary_speed_threshold_mps)
        ]
        slow_ramp = [
            row
            for row in regions["ramp"]
            if _number(row[4]) < float(primary_speed_threshold_mps)
        ]
        occupied_lanes = {str(row[2]) for row in regions["premerge"]}
        band_cavs = []
        for vehicle_id in active_cavs:
            row = current.get(vehicle_id)
            if row is None or str(row[1]) != PRE_MERGE_EDGE:
                continue
            distance = PRE_MERGE_LANE_LENGTH - _number(row[3])
            if wall_band_lo_m <= distance <= wall_band_hi_m:
                band_cavs.append(row)
        band_lanes = {str(row[2]) for row in band_cavs}

        health = _number(snapshot.supervisor_observation[0], default=math.nan)
        pre_speed = primary["pre_merge_mean_speed_mps"]
        primary_predicate = bool(
            int(primary["queue_count"]) >= int(primary_queue_count)
            and pre_speed is not None
            and float(pre_speed) <= float(primary_speed_threshold_mps)
        )
        row = {
            "relative_step": int(snapshot.relative_step),
            "sim_time_s": _number(snapshot.sim_time),
            "health": health if math.isfinite(health) else None,
            "health_delta": None,
            "primary_congestion_predicate": primary_predicate,
            "queue_count": int(primary["queue_count"]),
            "queue_tail_distance_to_merge_m": (
                max(PRE_MERGE_LANE_LENGTH - _number(value[3]) for value in slow_premerge)
                if slow_premerge
                else 0.0
            ),
            "ramp_slow_vehicle_count": len(slow_ramp),
            "ramp_queue_tail_distance_to_merge_m": (
                max(RAMP_LANE_LENGTH - _number(value[3]) for value in slow_ramp)
                if slow_ramp
                else 0.0
            ),
            "hard_brake_vehicle_count": len(primary["hard_brake_vehicle_ids"]),
            "low_ttc_pair_count": len(primary["low_ttc_pairs"]),
            "collision_vehicle_count": len(snapshot.collision_vehicle_ids),
            "starting_teleport_vehicle_count": len(snapshot.starting_teleport_vehicle_ids),
            "mainline_arrivals": sum(
                str(current[vehicle_id][1]) == PRE_MERGE_EDGE for vehicle_id in new_ids
            ),
            "ramp_arrivals": sum(
                str(current[vehicle_id][1]) == RAMP_EDGE for vehicle_id in new_ids
            ),
            "mainline_merge_entries": mainline_merge_entries,
            "ramp_merge_entries": ramp_merge_entries,
            "downstream_discharges": downstream_discharges,
            "disappeared_vehicle_count": len(disappeared),
            "inferred_completed_vehicle_count_cumulative": len(completed_ids),
            "lane_change_count": lane_changes,
            "active_cav_count": len(active_cavs),
            "network_vehicle_count": len(current),
            "observed_unique_vehicle_count_cumulative": len(seen | set(current)),
            "observed_unique_cav_count_cumulative": len(seen_cavs),
            "occupied_premerge_lane_count": len(occupied_lanes),
            "wall_band_cav_count": len(band_cavs),
            "wall_band_lane_count": len(band_lanes),
            "wall_band_occupied_lane_coverage": (
                len(band_lanes & occupied_lanes) / len(occupied_lanes)
                if occupied_lanes
                else 1.0
            ),
            **_command_metrics(snapshot, audits.get(int(snapshot.relative_step))),
        }
        for name, values in metrics.items():
            for key, value in values.items():
                row[f"{name}_{key}"] = value
        if result and row["health"] is not None and result[-1]["health"] is not None:
            row["health_delta"] = float(row["health"]) - float(result[-1]["health"])
        result.append(row)
        previous = current
        seen.update(current)

    for index, row in enumerate(result):
        now = float(row["sim_time_s"])
        for window_s in (20, 60):
            recent = [
                prior
                for prior in result[: index + 1]
                if now - window_s < float(prior["sim_time_s"]) <= now
            ]
            scale = 3600.0 / float(window_s)
            for key in (
                "mainline_arrivals",
                "ramp_arrivals",
                "mainline_merge_entries",
                "ramp_merge_entries",
                "downstream_discharges",
            ):
                row[f"{key}_{window_s}s_vph"] = sum(int(item[key]) for item in recent) * scale
    return result


def _runs(
    rows: Sequence[Mapping[str, Any]], predicate: Callable[[Mapping[str, Any]], bool]
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    start: float | None = None
    end: float | None = None
    count = 0
    previous_time: float | None = None
    for row in rows:
        current_time = float(row["sim_time_s"])
        contiguous = previous_time is None or abs(current_time - previous_time - 1.0) <= 1e-6
        if predicate(row):
            if start is None or not contiguous:
                if start is not None:
                    runs.append({"onset_s": start, "end_s": end, "sample_count": count})
                start = current_time
                count = 1
            else:
                count += 1
            end = current_time
        elif start is not None:
            runs.append({"onset_s": start, "end_s": end, "sample_count": count})
            start = None
            end = None
            count = 0
        previous_time = current_time
    if start is not None:
        runs.append({"onset_s": start, "end_s": end, "sample_count": count})
    return runs


def classify_endpoint(
    rows: Sequence[Mapping[str, Any]],
    *,
    predicate: Callable[[Mapping[str, Any]], bool],
    sustain_s: int,
    analysis_start_s: int,
    analysis_end_s: int,
    terminal_reason: str,
    prevalence_lookback_s: int | None = None,
) -> dict[str, Any]:
    """Classify prevalence, incidence, recovery, and right censoring."""

    observed = [row for row in rows if float(row["sim_time_s"]) <= analysis_end_s]
    if not observed:
        return {"classification": "invalid", "reason": "no_observations"}
    runs = _runs(observed, predicate)
    sustained = [row for row in runs if int(row["sample_count"]) >= int(sustain_s)]
    lookback = int(prevalence_lookback_s or sustain_s)
    prevalent = next(
        (
            row
            for row in sustained
            if float(row["onset_s"]) <= analysis_start_s - lookback + 1
            and float(row["end_s"]) >= analysis_start_s
        ),
        None,
    )
    incident = next(
        (
            row
            for row in sustained
            if float(row["onset_s"]) + sustain_s - 1 > analysis_start_s
            and float(row["onset_s"]) <= analysis_end_s - sustain_s + 1
        ),
        None,
    )
    last_run = runs[-1] if runs else None
    unfinished = bool(
        last_run
        and float(last_run["end_s"]) == float(observed[-1]["sim_time_s"])
        and int(last_run["sample_count"]) < int(sustain_s)
    )
    natural_clearance = terminal_reason == "natural_clearance"
    ended_before_window = float(observed[-1]["sim_time_s"]) < analysis_end_s
    invalid_termination = terminal_reason in {"gridlock", "truncated", "unknown"}
    if prevalent is not None:
        classification = "prevalent_at_anchor"
        selected = prevalent
    elif incident is not None:
        classification = "incident_after_anchor"
        selected = incident
    elif invalid_termination:
        classification = "censored_or_invalid"
        selected = None
    elif unfinished and not natural_clearance:
        classification = "censored_or_invalid"
        selected = None
    elif ended_before_window and not natural_clearance:
        classification = "censored_or_invalid"
        selected = None
    else:
        classification = "no_incident_or_recovered"
        selected = None
    sustained_event = classification in {"prevalent_at_anchor", "incident_after_anchor"}
    actual_observed_end = min(float(observed[-1]["sim_time_s"]), float(analysis_end_s))
    return {
        "classification": classification,
        "sustained_event": sustained_event,
        "prevalent_at_anchor": classification == "prevalent_at_anchor",
        "incident_after_anchor": classification == "incident_after_anchor",
        "selected_onset_s": None if selected is None else float(selected["onset_s"]),
        "selected_confirmation_s": (
            None if selected is None else float(selected["onset_s"]) + sustain_s - 1
        ),
        "first_sustained_onset_s": (
            None if not sustained else float(sustained[0]["onset_s"])
        ),
        "max_run_s": max((int(row["sample_count"]) for row in runs), default=0),
        "terminal_unfinished_run_s": (
            int(last_run["sample_count"])
            if last_run and float(last_run["end_s"]) == float(observed[-1]["sim_time_s"])
            else 0
        ),
        "latest_fully_observable_onset_s": actual_observed_end - sustain_s + 1,
        "terminal_recovered": natural_clearance,
        "sustained_then_recovered": bool(sustained_event and natural_clearance),
        "natural_clearance_treated_as_recovery": bool(
            natural_clearance and not sustained_event
        ),
        "terminal_reason": terminal_reason,
    }


def _terminal_trends(rows: Sequence[Mapping[str, Any]], end_s: int) -> dict[str, Any]:
    terminal = [row for row in rows if end_s - 60 < float(row["sim_time_s"]) <= end_s]
    prior = [row for row in rows if end_s - 120 < float(row["sim_time_s"]) <= end_s - 60]
    last30 = [row for row in terminal if float(row["sim_time_s"]) > end_s - 30]
    preceding30 = [
        row for row in terminal if end_s - 60 < float(row["sim_time_s"]) <= end_s - 30
    ]
    queue_last = _mean([float(row["queue_count"]) for row in last30])
    queue_prior = _mean([float(row["queue_count"]) for row in preceding30])
    speed_last = _mean(
        [
            float(row["premerge_mean_speed_mps"])
            for row in last30
            if row["premerge_mean_speed_mps"] is not None
        ]
    )
    speed_prior = _mean(
        [
            float(row["premerge_mean_speed_mps"])
            for row in preceding30
            if row["premerge_mean_speed_mps"] is not None
        ]
    )
    adverse = bool(
        queue_last is not None
        and queue_prior is not None
        and speed_last is not None
        and speed_prior is not None
        and queue_last >= queue_prior + 1.0
        and speed_last <= speed_prior - 1.0
    )
    return {
        "terminal_60s_sample_count": len(terminal),
        "preceding_60s_sample_count": len(prior),
        "terminal_queue_slope_veh_per_s": _slope(terminal, "queue_count"),
        "terminal_premerge_speed_slope_mps2": _slope(
            terminal, "premerge_mean_speed_mps"
        ),
        "terminal_health_slope_per_s": _slope(terminal, "health"),
        "terminal_adverse_trend_diagnostic": adverse,
        "adverse_trend_rule": (
            "last30 queue mean >= previous30 +1 and last30 premerge speed mean "
            "<= previous30 -1 m/s; diagnostic only"
        ),
    }


def summarize_density_trace(
    trace: ReplayTrace,
    *,
    analysis_start_s: int,
    analysis_end_s: int,
    prevalence_lookback_s: int = 60,
    primary_queue_count: int,
    primary_speed_threshold_mps: float,
    primary_sustain_s: int,
    secondary_health_threshold: float,
    secondary_health_sustain_s: int,
) -> dict[str, Any]:
    definition = OutcomeDefinition(
        queue_speed_threshold_mps=primary_speed_threshold_mps,
        congestion_queue_count=primary_queue_count,
        congestion_speed_threshold_mps=primary_speed_threshold_mps,
        sustain_steps=primary_sustain_s,
    )
    full_telemetry = build_telemetry(
        trace,
        definition=definition,
        primary_queue_count=primary_queue_count,
        primary_speed_threshold_mps=primary_speed_threshold_mps,
    )
    telemetry = [
        row for row in full_telemetry if float(row["sim_time_s"]) <= analysis_end_s
    ]
    window_snapshots = [
        snapshot
        for snapshot in trace.snapshots
        if float(snapshot.sim_time) <= analysis_end_s
    ]
    if not telemetry or not window_snapshots:
        raise ValueError("trace contains no samples at or before the absolute endpoint")
    if len(telemetry) >= 2:
        spacings = [
            float(right["sim_time_s"]) - float(left["sim_time_s"])
            for left, right in zip(telemetry, telemetry[1:])
        ]
    else:
        spacings = []
    metadata_reason = str(trace.controller_metadata.get("termination_reason", "unknown"))
    exact_endpoint_observed = abs(float(telemetry[-1]["sim_time_s"]) - analysis_end_s) <= 1e-6
    if exact_endpoint_observed:
        terminal_reason = "horizon_reached"
    elif metadata_reason == "terminated" and window_snapshots[-1].vehicle_count == 0:
        terminal_reason = "natural_clearance"
    else:
        terminal_reason = metadata_reason

    primary = classify_endpoint(
        telemetry,
        predicate=lambda row: bool(row["primary_congestion_predicate"]),
        sustain_s=primary_sustain_s,
        analysis_start_s=analysis_start_s,
        analysis_end_s=analysis_end_s,
        terminal_reason=terminal_reason,
        prevalence_lookback_s=prevalence_lookback_s,
    )
    secondary = classify_endpoint(
        telemetry,
        predicate=lambda row: (
            row["health"] is not None
            and float(row["health"]) < float(secondary_health_threshold)
        ),
        sustain_s=secondary_health_sustain_s,
        analysis_start_s=analysis_start_s,
        analysis_end_s=analysis_end_s,
        terminal_reason=terminal_reason,
        prevalence_lookback_s=prevalence_lookback_s,
    )
    analysis_rows = [
        row
        for row in telemetry
        if analysis_start_s <= float(row["sim_time_s"]) <= analysis_end_s
    ]
    all_vehicle_ids = {
        str(vehicle[0]) for snapshot in window_snapshots for vehicle in snapshot.vehicles
    }
    all_cav_ids = {
        str(vehicle_id) for snapshot in window_snapshots for vehicle_id in snapshot.active_cavs
    }
    wall_commands = [
        audit
        for audit in trace.controller_audits
        if int(audit.get("command", 0)) != 0
        or bool(audit.get("wall_active"))
        or bool(audit.get("wall_override_ids"))
    ]
    collision_steps = sum(int(row["collision_vehicle_count"]) for row in analysis_rows)
    teleport_steps = sum(
        int(row["starting_teleport_vehicle_count"]) for row in analysis_rows
    )
    final_row = telemetry[-1]
    final_snapshot = window_snapshots[-1]
    trends = _terminal_trends(telemetry, analysis_end_s)
    primary["diagnostic_right_censored"] = bool(
        primary["classification"] == "censored_or_invalid"
        or (
            primary["classification"] == "no_incident_or_recovered"
            and trends["terminal_adverse_trend_diagnostic"]
            and terminal_reason != "natural_clearance"
        )
    )
    return {
        "schema": "idm_v5_5_density_mapper_branch/1",
        "seed": trace.seed,
        "trace_digest": trace.trace_digest,
        "controller_audit_digest": _sha256_json(trace.controller_audits),
        "telemetry_digest": _sha256_json(telemetry),
        "reset_info": dict(trace.reset_info),
        "terminal": {
            "reason": terminal_reason,
            "relative_step": int(final_snapshot.relative_step),
            "sim_time_s": float(final_snapshot.sim_time),
            "vehicle_count": int(final_snapshot.vehicle_count),
            "expected_vehicle_count": (
                None
                if len(full_telemetry) != len(telemetry)
                else trace.controller_metadata.get("terminal_expected_vehicle_count")
            ),
            "terminated_early": bool(not exact_endpoint_observed and trace.terminated_early),
            "post_endpoint_samples_discarded": len(full_telemetry) - len(telemetry),
        },
        "clock_integrity": {
            "sample_count": len(telemetry),
            "absolute_analysis_sample_count": len(analysis_rows),
            "one_hz": bool(spacings and all(abs(value - 1.0) <= 1e-6 for value in spacings)),
            "reset_offset_s": float(trace.snapshots[0].sim_time),
            "first_sim_time_s": float(trace.snapshots[0].sim_time),
            "last_sim_time_s": float(final_snapshot.sim_time),
            "exact_anchor_observed": any(
                abs(float(row["sim_time_s"]) - analysis_start_s) <= 1e-6
                for row in telemetry
            ),
            "exact_endpoint_observed": exact_endpoint_observed,
        },
        "primary_premerge_endpoint": primary,
        "secondary_legacy_health_diagnostic": secondary,
        "label_agreement": (
            bool(primary.get("sustained_event")) == bool(secondary.get("sustained_event"))
        ),
        "terminal_trends": trends,
        "flow_and_population": {
            "observed_unique_vehicle_count_after_reset": len(all_vehicle_ids),
            "observed_unique_cav_count_after_reset": len(all_cav_ids),
            "realized_observed_cav_fraction": (
                len(all_cav_ids) / len(all_vehicle_ids) if all_vehicle_ids else 0.0
            ),
            "mainline_arrivals_after_reset": sum(
                int(row["mainline_arrivals"]) for row in telemetry
            ),
            "ramp_arrivals_after_reset": sum(int(row["ramp_arrivals"]) for row in telemetry),
            "mainline_merge_entries": sum(
                int(row["mainline_merge_entries"]) for row in telemetry
            ),
            "ramp_merge_entries": sum(int(row["ramp_merge_entries"]) for row in telemetry),
            "downstream_discharges": sum(
                int(row["downstream_discharges"]) for row in telemetry
            ),
            "inferred_completed_vehicle_count": int(
                final_row["inferred_completed_vehicle_count_cumulative"]
            ),
            "remaining_vehicle_count": int(final_row["network_vehicle_count"]),
        },
        "control": {
            "wall_command_or_override_count": len(wall_commands),
            "damping_command_steps": sum(
                int(row["damping_command_count"] > 0) for row in analysis_rows
            ),
            "damping_controlled_cav_seconds": sum(
                float(row["damping_controlled_cav_seconds"]) for row in analysis_rows
            ),
            "max_wall_band_occupied_lane_coverage": max(
                (float(row["wall_band_occupied_lane_coverage"]) for row in analysis_rows),
                default=0.0,
            ),
        },
        "safety": {
            "hard_brake_vehicle_steps": sum(
                int(row["hard_brake_vehicle_count"]) for row in analysis_rows
            ),
            "low_ttc_pair_steps": sum(
                int(row["low_ttc_pair_count"]) for row in analysis_rows
            ),
            "collision_vehicle_steps": collision_steps,
            "starting_teleport_vehicle_steps": teleport_steps,
        },
        "telemetry": telemetry,
    }


def branch_integrity(
    record: Mapping[str, Any],
    *,
    analysis_start_s: int,
    analysis_end_s: int,
    prevalence_lookback_s: int,
) -> dict[str, bool]:
    terminal_reason = str(record["terminal"]["reason"])
    valid_terminal = terminal_reason in {"horizon_reached", "natural_clearance"}
    start_required = analysis_start_s - int(prevalence_lookback_s) + 1
    observed_times = {
        round(float(row["sim_time_s"]), 6) for row in record.get("telemetry", ())
    }
    terminal_time = float(record["terminal"]["sim_time_s"])
    coverage_end = analysis_end_s if terminal_reason == "horizon_reached" else terminal_time
    required_times = {
        round(float(second), 6)
        for second in range(int(start_required), int(math.floor(coverage_end)) + 1)
    }
    return {
        "one_hz_observations": bool(record["clock_integrity"]["one_hz"]),
        "prevalence_lookback_observed": (
            float(record["clock_integrity"]["first_sim_time_s"]) <= start_required
        ),
        "exact_anchor_observed": bool(record["clock_integrity"]["exact_anchor_observed"]),
        "absolute_window_coverage": bool(required_times)
        and required_times.issubset(observed_times),
        "exact_endpoint_or_proven_natural_clearance": (
            bool(record["clock_integrity"]["exact_endpoint_observed"])
            or terminal_reason == "natural_clearance"
        ),
        "valid_terminal_classification": valid_terminal,
        "no_wall_command_or_override": int(record["control"]["wall_command_or_override_count"])
        == 0,
        "no_collision": int(record["safety"]["collision_vehicle_steps"]) == 0,
        "no_teleport": int(record["safety"]["starting_teleport_vehicle_steps"]) == 0,
    }


__all__ = [
    "branch_integrity",
    "build_telemetry",
    "classify_endpoint",
    "summarize_density_trace",
]
