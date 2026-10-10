"""Pure Gate-F assessment for the timing-aligned v5.5 wall pilot."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def gate_f_assessment(
    *,
    formation: Mapping[str, Any],
    selected_lane_ids: Sequence[str],
    activation_safety: Mapping[str, Any] | None,
    terminal_runway_margin_m: float | None,
    requested_duration_s: float,
    maximum_span_m: float = 50.0,
) -> dict[str, Any]:
    occupied = set(str(value) for value in formation.get("occupied_lane_ids", ()))
    required = set(str(value) for value in formation.get("required_lane_ids", ()))
    selected = set(str(value) for value in selected_lane_ids)
    full_coverage = bool(occupied) and occupied == required == selected
    initial_span_raw = formation.get("longitudinal_span_m")
    predicted_max = formation.get("predicted_max_span_m")
    predicted_terminal = formation.get("predicted_terminal_span_m")
    compact = bool(
        initial_span_raw is not None
        and math.isfinite(float(initial_span_raw))
        and float(initial_span_raw) <= maximum_span_m
        and predicted_max is not None
        and math.isfinite(float(predicted_max))
        and float(predicted_max) <= maximum_span_m
        and predicted_terminal is not None
        and math.isfinite(float(predicted_terminal))
        and float(predicted_terminal) <= maximum_span_m
    )
    runway = bool(
        terminal_runway_margin_m is not None
        and math.isfinite(float(terminal_runway_margin_m))
        and float(terminal_runway_margin_m) >= 0.0
    )
    safe = bool(activation_safety and activation_safety.get("safe"))
    formed = bool(formation.get("feasible"))
    admitted = bool(formed and full_coverage and compact and runway and safe)
    projected = float(requested_duration_s) if admitted else 0.0
    return {
        "member_availability_pass": formed,
        "full_occupied_lane_coverage_pass": full_coverage,
        "compactness_pass": compact,
        "terminal_runway_pass": runway,
        "front_rear_activation_safety_pass": safe,
        "admitted": admitted,
        "projected_duration_s": projected,
        "projected_duration_fraction": (
            projected / float(requested_duration_s) if requested_duration_s > 0.0 else 0.0
        ),
    }


def congestion_run_summary(
    telemetry: Sequence[Mapping[str, Any]],
    *,
    start_s: float,
    end_s: float,
    sustain_s: int,
) -> dict[str, Any]:
    """Summarize contiguous 1 Hz physical-congestion runs in a fixed clock window."""

    rows = [
        row
        for row in telemetry
        if float(start_s) <= float(row["sim_time_s"]) <= float(end_s)
    ]
    runs: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    previous_time: float | None = None
    for row in rows:
        sim_time = float(row["sim_time_s"])
        contiguous = previous_time is None or abs(sim_time - previous_time - 1.0) <= 1e-6
        if bool(row.get("primary_congestion_predicate")):
            if current is None or not contiguous:
                if current is not None:
                    runs.append(current)
                current = {"onset_s": sim_time, "end_s": sim_time, "duration_s": 1}
            else:
                current["end_s"] = sim_time
                current["duration_s"] = int(current["duration_s"]) + 1
        elif current is not None:
            runs.append(current)
            current = None
        previous_time = sim_time
    if current is not None:
        runs.append(current)
    sustained = [row for row in runs if int(row["duration_s"]) >= int(sustain_s)]
    for row in sustained:
        row["confirmation_s"] = float(row["onset_s"]) + int(sustain_s) - 1
        row["recovery_s"] = (
            float(row["end_s"]) + 1.0 if float(row["end_s"]) < float(end_s) else None
        )
    return {
        "start_s": float(start_s),
        "end_s": float(end_s),
        "observed_sample_count": len(rows),
        "predicate_duration_s": sum(int(row["duration_s"]) for row in runs),
        "longest_run_s": max((int(row["duration_s"]) for row in runs), default=0),
        "sustained_event": bool(sustained),
        "sustained_run_count": len(sustained),
        "sustained_duration_s": sum(int(row["duration_s"]) for row in sustained),
        "first_sustained_onset_s": (
            None if not sustained else float(sustained[0]["onset_s"])
        ),
        "first_sustained_confirmation_s": (
            None if not sustained else float(sustained[0]["confirmation_s"])
        ),
        "first_recovery_s": None if not sustained else sustained[0]["recovery_s"],
        "final_recovery_s": None if not sustained else sustained[-1]["recovery_s"],
        "terminal_recovered": bool(sustained and sustained[-1]["recovery_s"] is not None),
        "runs": runs,
        "sustained_runs": sustained,
    }


def upstream_storage_summary(
    telemetry: Sequence[Mapping[str, Any]], *, start_s: float, end_s: float
) -> dict[str, Any]:
    rows = [
        row
        for row in telemetry
        if float(start_s) < float(row["sim_time_s"]) <= float(end_s)
    ]
    premerge = [int(row.get("premerge_count", 0)) for row in rows]
    ramp = [int(row.get("ramp_count", 0)) for row in rows]
    return {
        "start_exclusive_s": float(start_s),
        "end_inclusive_s": float(end_s),
        "observed_interval_count": len(rows),
        "premerge_vehicle_seconds": float(sum(premerge)),
        "ramp_vehicle_seconds": float(sum(ramp)),
        "combined_upstream_vehicle_seconds": float(sum(premerge) + sum(ramp)),
        "max_premerge_vehicle_count": max(premerge, default=0),
        "max_ramp_vehicle_count": max(ramp, default=0),
        "max_combined_upstream_vehicle_count": max(
            (left + right for left, right in zip(premerge, ramp)), default=0
        ),
    }


__all__ = ["congestion_run_summary", "gate_f_assessment", "upstream_storage_summary"]
