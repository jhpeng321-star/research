"""Compare the frozen P30 arm with the modified-FollowerStopper P30 arm.

The treatment is development-only.  A previously viewed seed-4129 reversal
smoke is reported separately and is never counted in the validation gate.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from experiments.io import atomic_write_json
from idm.decentralized_damping import (
    ACTIVATION_MODIFIED_FOLLOWER_STOPPER_P30_ACTION_SET,
    ACTIVATION_SPEED_CEILING_ACTION_SET,
)
from idm.follower_stopper import MODIFIED_FOLLOWER_STOPPER_P30_V1


SCHEMA = "idm_modified_followerstopper_p30_development_screen/1"
DESIGN_CASE_ID = "m2000.seed4129.t354"
EXPECTED_VALIDATION_CASE_IDS = frozenset(
    {
        "m2000.seed4123.t61",
        "m2000.seed4127.t76",
        "m2000.seed4130.t55",
        "m2000.seed4131.t382",
        "m2000.seed4132.t49",
        "m2000.seed4123.t146",
        "m2000.seed4127.t200",
        "m2000.seed4129.t436",
        "m2000.seed4130.t192",
        "m2000.seed4132.t291",
        "m2000.seed4127.t289",
        "m2000.seed4129.t528",
        "m2000.seed4130.t331",
        "m2000.seed4132.t574",
    }
)
COMMAND_DELTA_MPS2 = 0.05
MINIMUM_DISTINCT_COMMAND_STEPS = 2
HARD_TTC_FLOOR_S = 4.0
SEVERE_BRAKING_THRESHOLD_MPS2 = -3.0
HARD_BRAKING_THRESHOLD_MPS2 = -4.5


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _case_id(seed: int, decision_step: int, demand: int = 2000) -> str:
    return f"m{int(demand)}.seed{int(seed)}.t{int(decision_step)}"


def _effect_map(root: Path) -> dict[str, dict[str, Any]]:
    summary = _read(root / "summary.json")
    result: dict[str, dict[str, Any]] = {}
    for case in summary.get("cases", []):
        effects = [
            row
            for row in case.get("effects", [])
            if int(row.get("candidate_slot") or 0) == 1
        ]
        if len(effects) != 1:
            raise ValueError(
                f"expected one candidate-1 effect in {root} case {case.get('case_id')}"
            )
        result[str(case["case_id"])] = dict(effects[0])
    return result


def _branch_map(root: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted((root / "branches").glob("*.json")):
        row = _read(path)
        if row.get("is_control") or int(row.get("candidate_slot") or 0) != 1:
            continue
        result[str(row["case_id"])] = row
    return result


def _discovery_map(root: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted((root / "discoveries").glob("*.json")):
        row = _read(path)
        decision = row.get("decision")
        if decision is None:
            continue
        result[
            _case_id(row["seed"], decision["decision_step"], row.get("demand", 2000))
        ] = row
    return result


def _protocol(root: Path) -> dict[str, Any]:
    summary = _read(root / "summary.json")
    return dict(summary["protocol"])


def _target_commands(branch: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    target = str(branch.get("target_vehicle_id"))
    return sorted(
        (
            row
            for row in branch.get("command_rows", [])
            if str(row.get("vehicle_id")) == target
        ),
        key=lambda row: int(row["relative_step"]),
    )


def _command_contrast(
    p30: Mapping[str, Any], treatment: Mapping[str, Any]
) -> dict[str, Any]:
    lhs = {
        int(row["relative_step"]): float(row["projected_acceleration_mps2"])
        for row in _target_commands(p30)
        if row.get("projected_acceleration_mps2") is not None
    }
    rhs_rows = {
        int(row["relative_step"]): row for row in _target_commands(treatment)
    }
    shared = sorted(set(lhs) & set(rhs_rows))
    deltas = [
        abs(float(rhs_rows[step]["projected_acceleration_mps2"]) - lhs[step])
        for step in shared
    ]
    distinct = sum(value + 1e-12 >= COMMAND_DELTA_MPS2 for value in deltas)
    zones = Counter(
        str((rhs_rows[step].get("modified_follower_stopper") or {}).get("zone"))
        for step in shared
    )
    accelerations = [
        float(rhs_rows[step]["projected_acceleration_mps2"]) for step in shared
    ]
    p30_accelerations = [lhs[step] for step in shared]

    def maximum_jerk(values: Sequence[float]) -> float | None:
        if len(values) < 2:
            return None
        return max(abs(values[index] - values[index - 1]) for index in range(1, len(values)))

    return {
        "shared_command_steps": len(shared),
        "p30_command_steps": len(lhs),
        "treatment_command_steps": len(rhs_rows),
        "all_treatment_commands_compared": bool(rhs_rows)
        and set(rhs_rows).issubset(lhs),
        "distinct_command_steps": distinct,
        "maximum_absolute_acceleration_delta_mps2": max(deltas, default=0.0),
        "meaningfully_distinct": distinct >= MINIMUM_DISTINCT_COMMAND_STEPS,
        "treatment_zone_counts": dict(sorted(zones.items())),
        "all_feedback_diagnostics_present": bool(rhs_rows)
        and all(row.get("modified_follower_stopper") for row in rhs_rows.values()),
        "any_treatment_infeasible_command": any(
            bool(row.get("treatment_infeasible", False))
            for row in rhs_rows.values()
        ),
        "p30_maximum_command_jerk_mps3": maximum_jerk(p30_accelerations),
        "treatment_maximum_command_jerk_mps3": maximum_jerk(accelerations),
    }


def _minimum(values: Iterable[Any]) -> float | None:
    finite = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            finite.append(number)
    return min(finite) if finite else None


def _cohort_ids(branch: Mapping[str, Any]) -> tuple[str, ...]:
    for step in (branch.get("phase_response") or {}).get("steps", []):
        rows = step.get("upstream_followers") or []
        if rows:
            return tuple(str(row["vehicle_id"]) for row in rows[:5])
    return ()


def _local_safety(
    branch: Mapping[str, Any], cohort_ids: Sequence[str]
) -> dict[str, Any]:
    cohort = set(cohort_ids)
    follower_rows = []
    immediate_rows = []
    for step in (branch.get("phase_response") or {}).get("steps", []):
        follower_rows.extend(
            row
            for row in (step.get("upstream_followers") or [])
            if str(row.get("vehicle_id")) in cohort
        )
        for target in step.get("target_states") or []:
            immediate = target.get("immediate_follower")
            if immediate:
                immediate_rows.append(immediate)

    def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        accelerations = [row.get("acceleration_mps2") for row in rows]
        minimum_acceleration = _minimum(accelerations)
        return {
            "vehicle_step_count": len(rows),
            "minimum_ttc_s": _minimum(row.get("ttc_s") for row in rows),
            "minimum_acceleration_mps2": minimum_acceleration,
            "severe_braking_exposure_steps": sum(
                value is not None
                and math.isfinite(float(value))
                and float(value) < SEVERE_BRAKING_THRESHOLD_MPS2
                for value in accelerations
            ),
            "hard_braking_exposure_steps": sum(
                value is not None
                and math.isfinite(float(value))
                and float(value) < HARD_BRAKING_THRESHOLD_MPS2
                for value in accelerations
            ),
        }

    return {
        "decision_cohort_vehicle_ids": list(cohort_ids),
        "stable_upstream_cohort": summarize(follower_rows),
        "dynamic_immediate_follower": summarize(immediate_rows),
    }


def _at_least(value: float | None, lower: float) -> bool:
    return value is None or value + 1e-12 >= lower


def _not_lower(value: float | None, reference: float | None) -> bool:
    if reference is None:
        return True
    return value is not None and value + 1e-12 >= reference


def _case_row(
    case_id: str,
    *,
    p30_effect: Mapping[str, Any],
    treatment_effect: Mapping[str, Any],
    p30_branch: Mapping[str, Any],
    treatment_branch: Mapping[str, Any],
    treatment_discovery: Mapping[str, Any],
    deadband: float,
) -> dict[str, Any]:
    cohort = _cohort_ids(treatment_branch)
    p30_local = _local_safety(p30_branch, cohort)
    treatment_local = _local_safety(treatment_branch, cohort)
    contrast = _command_contrast(p30_branch, treatment_branch)
    matched = (treatment_discovery.get("decision") or {}).get(
        "matched_decision_source"
    ) or {}
    score_p30 = float(p30_effect["safety_composite_reference_units"])
    score_treatment = float(
        treatment_effect["safety_composite_reference_units"]
    )
    treatment_stable = treatment_local["stable_upstream_cohort"]
    treatment_immediate = treatment_local["dynamic_immediate_follower"]
    local_safety_diagnostics_present = (
        bool(cohort)
        and int(treatment_stable["vehicle_step_count"]) > 0
        and treatment_stable["minimum_ttc_s"] is not None
        and treatment_stable["minimum_acceleration_mps2"] is not None
        and int(treatment_immediate["vehicle_step_count"]) > 0
        and treatment_immediate["minimum_ttc_s"] is not None
        and treatment_immediate["minimum_acceleration_mps2"] is not None
    )
    return {
        "case_id": case_id,
        "seed": int(treatment_branch["seed"]),
        "decision_step": int(treatment_branch["decision_step"]),
        "target_vehicle_id": treatment_branch.get("target_vehicle_id"),
        "same_target_vehicle": treatment_branch.get("target_vehicle_id")
        == p30_branch.get("target_vehicle_id"),
        "matched_source_identity": {
            key: bool(matched.get(key, False))
            for key in (
                "raw_decision_context_identity",
                "nonclassic_observation_identity",
                "treatment_difference_expected",
            )
        },
        "p30_effect": dict(p30_effect),
        "treatment_effect": dict(treatment_effect),
        "effect_improvement_over_p30": score_treatment - score_p30,
        "effect_above_replay_deadband": abs(score_treatment) > deadband,
        "command_contrast": contrast,
        "p30_local_safety": p30_local,
        "treatment_local_safety": treatment_local,
        "local_safety_diagnostics_present": local_safety_diagnostics_present,
        "hard_local_safety": {
            "stable_cohort_ttc_at_least_4s": _at_least(
                treatment_stable["minimum_ttc_s"], HARD_TTC_FLOOR_S
            ),
            "immediate_follower_ttc_at_least_4s": _at_least(
                treatment_immediate["minimum_ttc_s"], HARD_TTC_FLOOR_S
            ),
            "stable_cohort_acceleration_not_below_minus_4p5": _at_least(
                treatment_stable["minimum_acceleration_mps2"],
                HARD_BRAKING_THRESHOLD_MPS2,
            ),
            "immediate_follower_acceleration_not_below_minus_4p5": _at_least(
                treatment_immediate["minimum_acceleration_mps2"],
                HARD_BRAKING_THRESHOLD_MPS2,
            ),
        },
    }


def _collect_pairs(
    p30_roots: Sequence[Path], candidate_roots: Sequence[Path]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(p30_roots) != len(candidate_roots):
        raise ValueError("--p30-runs and --candidate-runs must have equal lengths")
    rows = []
    source_rows = []
    for p30_root, candidate_root in zip(p30_roots, candidate_roots):
        p30_summary = _read(p30_root / "summary.json")
        candidate_summary = _read(candidate_root / "summary.json")
        p30_protocol = dict(p30_summary["protocol"])
        candidate_protocol = dict(candidate_summary["protocol"])
        if p30_protocol.get("action_set") != ACTIVATION_SPEED_CEILING_ACTION_SET:
            raise ValueError(f"{p30_root} is not frozen P30")
        if (
            candidate_protocol.get("action_set")
            != ACTIVATION_MODIFIED_FOLLOWER_STOPPER_P30_ACTION_SET
        ):
            raise ValueError(f"{candidate_root} is not the frozen feedback treatment")
        expected_feedback_contract = json.loads(
            json.dumps(MODIFIED_FOLLOWER_STOPPER_P30_V1.to_dict())
        )
        if candidate_protocol.get("feedback_contract") != expected_feedback_contract:
            raise ValueError(f"{candidate_root} feedback contract drifted")
        frozen_keys = (
            "demand",
            "candidate_slots",
            "option_steps",
            "minimum_dose_steps",
            "cav_penetration",
            "decision_index",
            "candidate_ranking",
            "candidate_count",
            "safety_horizon_steps",
            "service_horizon_steps",
            "landmark",
            "reward_refs_sha256",
        )
        field_identity = {
            key: p30_protocol.get(key) == candidate_protocol.get(key)
            for key in frozen_keys
        }
        source_rows.append(
            {
                "p30_root": str(p30_root),
                "candidate_root": str(candidate_root),
                "protocol_field_identity": field_identity,
                "candidate_abstain_replay_exact": bool(
                    (candidate_summary.get("smoke_gates") or {}).get(
                        "abstain_replay_exact", False
                    )
                ),
            }
        )
        p30_effects, candidate_effects = _effect_map(p30_root), _effect_map(candidate_root)
        p30_branches, candidate_branches = _branch_map(p30_root), _branch_map(candidate_root)
        candidate_discoveries = _discovery_map(candidate_root)
        requested = set(candidate_effects)
        common = requested & set(p30_effects) & set(p30_branches) & set(candidate_branches) & set(candidate_discoveries)
        if common != requested:
            raise ValueError(
                f"paired artifacts incomplete for {candidate_root}: "
                f"missing {sorted(requested - common)}"
            )
        deadband = float(_read(candidate_root / "summary.json")["frozen_deadband"])
        for case_id in sorted(common):
            rows.append(
                _case_row(
                    case_id,
                    p30_effect=p30_effects[case_id],
                    treatment_effect=candidate_effects[case_id],
                    p30_branch=p30_branches[case_id],
                    treatment_branch=candidate_branches[case_id],
                    treatment_discovery=candidate_discoveries[case_id],
                    deadband=deadband,
                )
            )
    return rows, source_rows


def _arm_metrics(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    effects = [row[f"{arm}_effect"] for row in rows]
    scores = [float(effect["safety_composite_reference_units"]) for effect in effects]
    helpful_seeds = sorted(
        {
            int(row["seed"])
            for row in rows
            if row[f"{arm}_effect"]["classification"] == "helpful"
        }
    )
    nonbeneficial_seeds = sorted(
        {
            int(row["seed"])
            for row in rows
            if row[f"{arm}_effect"]["classification"] != "helpful"
        }
    )
    return {
        "case_count": len(rows),
        "classification_counts": dict(
            Counter(effect["classification"] for effect in effects)
        ),
        "harmful_case_count": sum(
            effect["classification"] == "harmful" for effect in effects
        ),
        "mean_effect": float(np.mean(scores)) if scores else None,
        "p10_effect": float(np.percentile(scores, 10.0)) if scores else None,
        "helpful_seeds": helpful_seeds,
        "nonbeneficial_seeds": nonbeneficial_seeds,
    }


def analyze(
    p30_roots: Sequence[Path],
    candidate_roots: Sequence[Path],
    *,
    design_p30_root: Path | None = None,
    design_candidate_root: Path | None = None,
) -> dict[str, Any]:
    rows, sources = _collect_pairs(p30_roots, candidate_roots)
    if any(row["case_id"] == DESIGN_CASE_ID for row in rows):
        raise ValueError("the viewed design case must be excluded from validation")

    design = None
    if design_p30_root is not None or design_candidate_root is not None:
        if design_p30_root is None or design_candidate_root is None:
            raise ValueError("both design roots are required")
        design_rows, design_sources = _collect_pairs(
            [design_p30_root], [design_candidate_root]
        )
        if [row["case_id"] for row in design_rows] != [DESIGN_CASE_ID]:
            raise ValueError("design artifact is not the frozen reversal case")
        row = design_rows[0]
        p30_stable = row["p30_local_safety"]["stable_upstream_cohort"]
        treatment_stable = row["treatment_local_safety"]["stable_upstream_cohort"]
        p30_immediate = row["p30_local_safety"]["dynamic_immediate_follower"]
        treatment_immediate = row["treatment_local_safety"]["dynamic_immediate_follower"]
        contrast = row["command_contrast"]
        strict = {
            "matched_identity": row["same_target_vehicle"]
            and all(row["matched_source_identity"].values()),
            "full_dose_and_constraints": int(
                row["treatment_effect"]["target_command_step_count"]
            )
            == 20
            and bool(row["treatment_effect"]["constraints_pass"]),
            "meaningful_command_difference": contrast["meaningfully_distinct"],
            "helpful_vs_native": row["treatment_effect"]["classification"]
            == "helpful",
            "better_than_p30_by_deadband": row["effect_improvement_over_p30"]
            > 0.02,
            "stable_cohort_minimum_ttc_not_lower": _not_lower(
                treatment_stable["minimum_ttc_s"], p30_stable["minimum_ttc_s"]
            ),
            "stable_cohort_peak_braking_not_worse": _not_lower(
                treatment_stable["minimum_acceleration_mps2"],
                p30_stable["minimum_acceleration_mps2"],
            ),
            "immediate_follower_minimum_ttc_not_lower": _not_lower(
                treatment_immediate["minimum_ttc_s"],
                p30_immediate["minimum_ttc_s"],
            ),
            "immediate_follower_peak_braking_not_worse": _not_lower(
                treatment_immediate["minimum_acceleration_mps2"],
                p30_immediate["minimum_acceleration_mps2"],
            ),
            "command_jerk_not_worse": _not_lower(
                contrast["p30_maximum_command_jerk_mps3"],
                contrast["treatment_maximum_command_jerk_mps3"],
            ),
            "merge_discharge_not_lower": int(
                row["treatment_effect"]["merge_discharge_count_delta"]
            )
            >= int(row["p30_effect"]["merge_discharge_count_delta"]),
        }
        bounded = {
            "all_hard_local_safety_bounds": all(row["hard_local_safety"].values()),
            "stable_severe_exposure_not_worse": treatment_stable[
                "severe_braking_exposure_steps"
            ]
            <= p30_stable["severe_braking_exposure_steps"],
            "immediate_severe_exposure_not_worse": treatment_immediate[
                "severe_braking_exposure_steps"
            ]
            <= p30_immediate["severe_braking_exposure_steps"],
            "helpful_vs_native": strict["helpful_vs_native"],
            "better_than_p30_by_deadband": strict["better_than_p30_by_deadband"],
        }
        design = {
            "source": design_sources,
            "row": row,
            "original_strict_gates": strict,
            "original_strict_pass": all(strict.values()),
            "bounded_safety_diagnostic": bounded,
            "bounded_safety_diagnostic_pass": all(bounded.values()),
            "note": "viewed design smoke; excluded from validation pass decision",
        }

    p30_metrics = _arm_metrics(rows, "p30")
    treatment_metrics = _arm_metrics(rows, "treatment")
    identity_gates = {
        "exactly_fourteen_treatment_naive_cases": len(rows) == 14,
        "exact_frozen_validation_case_set": {
            row["case_id"] for row in rows
        }
        == EXPECTED_VALIDATION_CASE_IDS,
        "design_case_excluded": all(row["case_id"] != DESIGN_CASE_ID for row in rows),
        "all_protocol_fields_match": all(
            all(source["protocol_field_identity"].values()) for source in sources
        ),
        "all_matched_source_identities": bool(rows)
        and all(all(row["matched_source_identity"].values()) for row in rows),
        "all_targets_match": bool(rows) and all(row["same_target_vehicle"] for row in rows),
        "all_feedback_diagnostics_present": bool(rows)
        and all(row["command_contrast"]["all_feedback_diagnostics_present"] for row in rows),
        "all_treatment_commands_compared": bool(rows)
        and all(
            row["command_contrast"]["all_treatment_commands_compared"]
            for row in rows
        ),
        "all_candidate_abstain_replays_exact": bool(sources)
        and all(source["candidate_abstain_replay_exact"] for source in sources),
    }
    hard_safety_service_gates = {
        "all_branch_constraints_pass": bool(rows)
        and all(row["treatment_effect"]["constraints_pass"] for row in rows),
        "all_hard_local_safety_bounds_pass": bool(rows)
        and all(all(row["hard_local_safety"].values()) for row in rows),
        "all_local_safety_diagnostics_present": bool(rows)
        and all(row["local_safety_diagnostics_present"] for row in rows),
        "zero_projection_interval_conflicts": bool(rows)
        and not any(
            row["command_contrast"]["any_treatment_infeasible_command"] for row in rows
        ),
        "aggregate_stable_severe_exposure_not_worse": sum(
            row["treatment_local_safety"]["stable_upstream_cohort"][
                "severe_braking_exposure_steps"
            ]
            for row in rows
        )
        <= sum(
            row["p30_local_safety"]["stable_upstream_cohort"][
                "severe_braking_exposure_steps"
            ]
            for row in rows
        ),
        "aggregate_immediate_severe_exposure_not_worse": sum(
            row["treatment_local_safety"]["dynamic_immediate_follower"][
                "severe_braking_exposure_steps"
            ]
            for row in rows
        )
        <= sum(
            row["p30_local_safety"]["dynamic_immediate_follower"][
                "severe_braking_exposure_steps"
            ]
            for row in rows
        ),
    }
    distinct_fraction = (
        sum(row["command_contrast"]["meaningfully_distinct"] for row in rows)
        / len(rows)
        if rows
        else 0.0
    )
    response_fraction = (
        sum(row["effect_above_replay_deadband"] for row in rows) / len(rows)
        if rows
        else 0.0
    )
    efficacy_gates = {
        "harmful_case_count_lower_than_p30": bool(rows)
        and treatment_metrics["harmful_case_count"]
        < p30_metrics["harmful_case_count"],
        "p10_effect_not_lower_than_p30": bool(rows)
        and treatment_metrics["p10_effect"] >= p30_metrics["p10_effect"],
        "mean_effect_no_more_than_0p02_below_p30": bool(rows)
        and treatment_metrics["mean_effect"] >= p30_metrics["mean_effect"] - 0.02,
        "helpful_in_at_least_three_seeds": bool(rows)
        and len(treatment_metrics["helpful_seeds"]) >= 3,
        "nonbeneficial_in_at_least_three_seeds": len(
            treatment_metrics["nonbeneficial_seeds"]
        )
        >= 3,
        "meaningful_command_difference_in_at_least_30pct": distinct_fraction >= 0.30,
        "effect_beyond_replay_in_at_least_30pct": response_fraction >= 0.30,
    }
    validation_gates = {
        **identity_gates,
        **hard_safety_service_gates,
        **efficacy_gates,
    }
    return {
        "schema": SCHEMA,
        "development_only": True,
        "treatment": ACTIVATION_MODIFIED_FOLLOWER_STOPPER_P30_ACTION_SET,
        "design_case": design,
        "validation_sources": sources,
        "validation_case_count": len(rows),
        "validation_seeds": sorted({row["seed"] for row in rows}),
        "thresholds": {
            "command_delta_mps2": COMMAND_DELTA_MPS2,
            "minimum_distinct_command_steps": MINIMUM_DISTINCT_COMMAND_STEPS,
            "hard_ttc_floor_s": HARD_TTC_FLOOR_S,
            "severe_braking_threshold_mps2": SEVERE_BRAKING_THRESHOLD_MPS2,
            "hard_braking_threshold_mps2": HARD_BRAKING_THRESHOLD_MPS2,
        },
        "p30_metrics": p30_metrics,
        "treatment_metrics": treatment_metrics,
        "command_distinct_case_fraction": distinct_fraction,
        "effect_beyond_replay_fraction": response_fraction,
        "identity_gates": identity_gates,
        "hard_safety_service_gates": hard_safety_service_gates,
        "efficacy_gates": efficacy_gates,
        "validation_gates": validation_gates,
        "validation_pass": bool(rows) and all(validation_gates.values()),
        "rl_training_authorized": False,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p30-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--design-p30", type=Path)
    parser.add_argument("--design-candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(
        [path.resolve() for path in args.p30_runs],
        [path.resolve() for path in args.candidate_runs],
        design_p30_root=(args.design_p30.resolve() if args.design_p30 else None),
        design_candidate_root=(
            args.design_candidate.resolve() if args.design_candidate else None
        ),
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "summary.json", payload)
    print(
        json.dumps(
            {
                "design_strict_pass": (
                    None
                    if payload["design_case"] is None
                    else payload["design_case"]["original_strict_pass"]
                ),
                "design_bounded_safety_pass": (
                    None
                    if payload["design_case"] is None
                    else payload["design_case"][
                        "bounded_safety_diagnostic_pass"
                    ]
                ),
                "validation_case_count": payload["validation_case_count"],
                "validation_pass": payload["validation_pass"],
                "p30_metrics": payload["p30_metrics"],
                "treatment_metrics": payload["treatment_metrics"],
                "failed_validation_gates": [
                    key
                    for key, value in payload["validation_gates"].items()
                    if not value
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
