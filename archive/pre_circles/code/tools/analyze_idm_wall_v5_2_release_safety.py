"""Diagnose when and where v5.2 wall hard-braking divergence appears.

This is a read-only analysis of existing paired artifacts.  It does not alter
the treatment, labels, thresholds, or seed partitions.  In particular, it
separates evidence that harm occurs *after* release from evidence that an
abrupt release *causes* the harm; the latter requires concentration near the
release boundary and cannot be inferred merely from a long-horizon excess.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA = "idm_wall_v5_2_release_safety_diagnosis/1"
DEFAULT_INPUT = Path(
    "runs/eval/idm_wall_v5_2_prospective_p30_s4301_4312"
)


def _counter(value: Mapping[str, Any] | None) -> Counter[str]:
    return Counter({str(key): int(count) for key, count in (value or {}).items()})


def _dose_qualified(row: Mapping[str, Any]) -> bool:
    dose = row.get("treatment_dose") or {}
    geometry = dose.get("wall_geometry") or {}
    compact = geometry.get("geometry_mode") == "compact_moving"
    return bool(
        bool(dose.get("activation_feasible"))
        and float(dose.get("realized_duration_fraction", 0.0)) >= 0.8
        and float(dose.get("lane_coverage_fraction", 0.0)) >= 0.8
        and (
            not compact
            or (
                bool(dose.get("assembly_completed"))
                and float(dose.get("holding_fraction_of_override", 0.0)) >= 0.5
                and float(geometry.get("fraction_within_configured_span", 0.0))
                >= 0.8
            )
        )
    )


def _time_bin(step: int, *, decision_step: int, release_step: int) -> str:
    if step <= decision_step:
        return "through_decision"
    if step < release_step:
        return "active_wall"
    offset = step - release_step
    if offset < 20:
        return "post_release_0_20s"
    if offset < 60:
        return "post_release_20_60s"
    return "post_release_after_60s"


def _onset_counts(
    outcomes: Mapping[str, Any], *, decision_step: int, release_step: int
) -> tuple[Counter[str], Counter[str], Counter[str]]:
    attribution = outcomes.get("hard_brake_attribution") or {}
    by_time: Counter[str] = Counter()
    by_road: Counter[str] = Counter()
    by_class: Counter[str] = Counter()
    for event in attribution.get("event_starts") or []:
        by_time[
            _time_bin(
                int(event["relative_step"]),
                decision_step=decision_step,
                release_step=release_step,
            )
        ] += 1
        by_road[str(event.get("road_id") or "unknown")] += 1
        by_class[str(event.get("vehicle_class") or "unknown")] += 1
    return by_time, by_road, by_class


def _delta(left: Counter[str], right: Counter[str]) -> dict[str, int]:
    return {
        key: int(left.get(key, 0) - right.get(key, 0))
        for key in sorted(set(left) | set(right))
    }


def _positive_share(delta: Mapping[str, int], keys: Iterable[str]) -> int:
    return sum(max(0, int(delta.get(key, 0))) for key in keys)


def _terminal_mechanics(dose: Mapping[str, Any]) -> dict[str, float | int | None]:
    geometry = (dose.get("wall_geometry") or {}).get("samples") or []
    tail = geometry[-5:]
    terminal = geometry[-1] if geometry else {}
    delivery = dose.get("terminal_tail_speed_delivery") or {}
    return {
        "terminal_span_m": (
            float(terminal["longitudinal_span_m"])
            if terminal.get("longitudinal_span_m") is not None
            else None
        ),
        "terminal_member_speed_std_mps": (
            float(terminal["member_speed_std_mps"])
            if terminal.get("member_speed_std_mps") is not None
            else None
        ),
        "terminal_five_sample_span_growth_m": (
            float(tail[-1]["longitudinal_span_m"])
            - float(tail[0]["longitudinal_span_m"])
            if len(tail) >= 2
            else None
        ),
        "projection_clip_count": int(dose.get("safety_projection_clip_count", 0)),
        "terminal_mean_absolute_target_error_mps": (
            float(delivery["mean_absolute_target_error_mps"])
            if delivery.get("mean_absolute_target_error_mps") is not None
            else None
        ),
        "terminal_fraction_within_1_mps": (
            float(delivery["fraction_within_1_mps"])
            if delivery.get("fraction_within_1_mps") is not None
            else None
        ),
    }


def _metric_ranges(
    safe: list[Mapping[str, Any]], unsafe: list[Mapping[str, Any]]
) -> dict[str, Any]:
    keys = (
        "terminal_span_m",
        "terminal_member_speed_std_mps",
        "terminal_five_sample_span_growth_m",
        "projection_clip_count",
        "terminal_mean_absolute_target_error_mps",
        "terminal_fraction_within_1_mps",
    )
    result: dict[str, Any] = {}
    for key in keys:
        safe_values = [
            float(row["terminal_mechanics"][key])
            for row in safe
            if row["terminal_mechanics"].get(key) is not None
        ]
        unsafe_values = [
            float(row["terminal_mechanics"][key])
            for row in unsafe
            if row["terminal_mechanics"].get(key) is not None
        ]
        safe_range = [min(safe_values), max(safe_values)] if safe_values else None
        unsafe_range = [min(unsafe_values), max(unsafe_values)] if unsafe_values else None
        ranges_overlap = bool(
            safe_range
            and unsafe_range
            and max(safe_range[0], unsafe_range[0])
            <= min(safe_range[1], unsafe_range[1])
        )
        result[key] = {
            "safe_range": safe_range,
            "unsafe_range": unsafe_range,
            "ranges_overlap": ranges_overlap,
        }
    return result


def analyze_case(row: Mapping[str, Any]) -> dict[str, Any] | None:
    arms = row.get("arms") or {}
    treatment_arm = arms.get("1") or {}
    if not bool(treatment_arm.get("wall_engaged")) or not _dose_qualified(row):
        return None

    decision_step = int(row["decision_step"])
    dose = row.get("treatment_dose") or {}
    requested_steps = int(dose.get("requested_duration_steps", 0))
    step_length_s = float(
        ((treatment_arm.get("outcomes") or {}).get("step_length_s")) or 1.0
    )
    # Current artifacts predate the explicit planned-release diagnostic.  The
    # contract applies on decision+1 ... decision+requested_steps inclusive.
    release_step = decision_step + requested_steps + 1

    control_outcomes = (arms.get("0") or {}).get("outcomes") or {}
    treatment_outcomes = treatment_arm.get("outcomes") or {}
    control_time, control_road, control_class = _onset_counts(
        control_outcomes,
        decision_step=decision_step,
        release_step=release_step,
    )
    treatment_time, treatment_road, treatment_class = _onset_counts(
        treatment_outcomes,
        decision_step=decision_step,
        release_step=release_step,
    )
    onset_delta = _delta(treatment_time, control_time)
    immediate_positive = _positive_share(onset_delta, ("post_release_0_20s",))
    delayed_positive = _positive_share(
        onset_delta,
        ("post_release_20_60s", "post_release_after_60s"),
    )
    active_positive = _positive_share(onset_delta, ("active_wall",))
    positive_total = immediate_positive + delayed_positive + active_positive

    treatment_attr = treatment_outcomes.get("hard_brake_attribution") or {}
    control_attr = control_outcomes.get("hard_brake_attribution") or {}
    treatment_phase_exposure = _counter(
        (treatment_attr.get("exposure_counts") or {}).get("phase")
    )
    control_phase_exposure = _counter(
        (control_attr.get("exposure_counts") or {}).get("phase")
    )
    post_active_exposure = sum(
        count
        for phase, count in treatment_phase_exposure.items()
        if phase not in {"assembling", "holding", "treatment_window"}
    )
    treatment_exposure = int(treatment_outcomes.get("hard_brake_vehicle_steps", 0))
    effects = ((row.get("safety_service_oracle") or {}).get("effects") or {})
    safety_admissible = bool(
        (row.get("safety_service_oracle") or {}).get("safety_admissible")
    )

    if positive_total == 0:
        temporal_pattern = "no_positive_onset_excess"
    elif immediate_positive > delayed_positive + active_positive:
        temporal_pattern = "release_boundary_localized"
    elif delayed_positive > immediate_positive + active_positive:
        temporal_pattern = "delayed_downstream_divergence"
    else:
        temporal_pattern = "mixed"

    return {
        "case_id": str(row.get("case_id")),
        "seed": int(row["seed"]),
        "decision_step": decision_step,
        "step_length_s": step_length_s,
        "derived_release_step": release_step,
        "dose_qualified": True,
        "safety_admissible": safety_admissible,
        "hard_brake_excess_vehicle_steps": int(
            effects.get(
                "hard_brake_excess_vehicle_steps",
                int(treatment_outcomes.get("hard_brake_vehicle_steps", 0))
                - int(control_outcomes.get("hard_brake_vehicle_steps", 0)),
            )
        ),
        "hard_brake_excess_budget": int(
            ((row.get("safety_service_oracle") or {}).get("budgets") or {}).get(
                "hard_brake_excess_budget", 0
            )
        ),
        "control_hard_brake_vehicle_steps": int(
            control_outcomes.get("hard_brake_vehicle_steps", 0)
        ),
        "treatment_hard_brake_vehicle_steps": treatment_exposure,
        "treatment_exposure_by_recorded_phase": dict(treatment_phase_exposure),
        "control_exposure_by_recorded_phase": dict(control_phase_exposure),
        "treatment_post_active_exposure_fraction": (
            post_active_exposure / treatment_exposure if treatment_exposure else 0.0
        ),
        "control_onsets_by_release_aligned_time": dict(control_time),
        "treatment_onsets_by_release_aligned_time": dict(treatment_time),
        "paired_onset_delta_by_release_aligned_time": onset_delta,
        "paired_onset_delta_by_road": _delta(treatment_road, control_road),
        "paired_onset_delta_by_vehicle_class": _delta(
            treatment_class, control_class
        ),
        "positive_onset_excess": {
            "active_wall": active_positive,
            "first_20s_after_release": immediate_positive,
            "more_than_20s_after_release": delayed_positive,
            "total": positive_total,
        },
        "temporal_pattern": temporal_pattern,
        "terminal_mechanics": _terminal_mechanics(dose),
    }


def analyze_payloads(payloads: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [row for payload in payloads for row in payload.get("cases") or []]
    cases = [result for row in rows if (result := analyze_case(row)) is not None]
    unsafe = [row for row in cases if not row["safety_admissible"]]
    safe = [row for row in cases if row["safety_admissible"]]
    pattern_counts = Counter(row["temporal_pattern"] for row in unsafe)
    phase_exposure: Counter[str] = Counter()
    onset_delta: Counter[str] = Counter()
    road_delta: Counter[str] = Counter()
    class_delta: Counter[str] = Counter()
    for row in unsafe:
        phase_exposure.update(row["treatment_exposure_by_recorded_phase"])
        onset_delta.update(row["paired_onset_delta_by_release_aligned_time"])
        road_delta.update(row["paired_onset_delta_by_road"])
        class_delta.update(row["paired_onset_delta_by_vehicle_class"])

    unsafe_excess = sum(
        int(row["hard_brake_excess_vehicle_steps"]) for row in unsafe
    )
    unsafe_treatment_exposure = sum(
        int(row["treatment_hard_brake_vehicle_steps"]) for row in unsafe
    )
    active_exposure = sum(
        count
        for phase, count in phase_exposure.items()
        if phase in {"assembling", "holding", "treatment_window"}
    )
    post_active_exposure = sum(phase_exposure.values()) - active_exposure
    immediate_positive = _positive_share(onset_delta, ("post_release_0_20s",))
    delayed_positive = _positive_share(
        onset_delta,
        ("post_release_20_60s", "post_release_after_60s"),
    )

    # A boundary-localized excess would support testing a taper.  A delayed
    # excess supports only the weaker conclusion that the wall changes later
    # traffic evolution; it does not isolate abrupt release as the cause.
    release_boundary_localized = bool(
        immediate_positive > 0 and immediate_positive > delayed_positive
    )
    mechanical_ranges = _metric_ranges(safe, unsafe)
    every_mechanical_range_overlaps = bool(mechanical_ranges) and all(
        row["ranges_overlap"] for row in mechanical_ranges.values()
    )
    return {
        "schema": SCHEMA,
        "source_case_count": len(rows),
        "dose_qualified_case_count": len(cases),
        "unsafe_dose_qualified_case_count": len(unsafe),
        "unsafe_case_ids": [row["case_id"] for row in unsafe],
        "unsafe_hard_brake_excess_vehicle_steps": unsafe_excess,
        "unsafe_treatment_hard_brake_vehicle_steps": unsafe_treatment_exposure,
        "unsafe_treatment_exposure_by_recorded_phase": dict(phase_exposure),
        "unsafe_treatment_active_exposure_fraction": (
            active_exposure / unsafe_treatment_exposure
            if unsafe_treatment_exposure
            else 0.0
        ),
        "unsafe_treatment_post_active_exposure_fraction": (
            post_active_exposure / unsafe_treatment_exposure
            if unsafe_treatment_exposure
            else 0.0
        ),
        "unsafe_paired_onset_delta_by_release_aligned_time": dict(onset_delta),
        "unsafe_paired_onset_delta_by_road": dict(road_delta),
        "unsafe_paired_onset_delta_by_vehicle_class": dict(class_delta),
        "unsafe_temporal_pattern_counts": dict(pattern_counts),
        "release_boundary_localized": release_boundary_localized,
        "terminal_mechanical_metric_ranges": mechanical_ranges,
        "every_terminal_mechanical_safe_unsafe_range_overlaps": (
            every_mechanical_range_overlaps
        ),
        "interpretation": (
            "Existing paired onsets are concentrated near the release boundary; "
            "a frozen taper pilot is mechanistically motivated but not proven."
            if release_boundary_localized
            else "Existing evidence shows post-treatment trajectory divergence but "
            "does not localize excess hard braking to the release boundary; an "
            "in-horizon taper alone is not yet a causally supported repair."
        ),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((args.input / "seeds").glob("seed*.json"))
    ]
    if not payloads:
        raise SystemExit(f"no seed artifacts found under {args.input / 'seeds'}")
    result = analyze_payloads(payloads)
    output = args.output or args.input / "release_safety_diagnosis.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
