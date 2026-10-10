"""Matched development-only comparison of frozen p30 and candidate p50 ceilings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiments.io import atomic_write_json
from idm.decentralized_damping import (
    ACTIVATION_SPEED_CEILING_ACTION_SET,
    ACTIVATION_SPEED_CEILING_P50_ACTION_SET,
)


SCHEMA = "idm_p30_p50_development_screen/1"
COMMAND_DELTA_MPS2 = 0.05
CASE_DISTINCT_STEP_FRACTION = 0.30


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _effect_by_seed(summary: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for case in summary.get("cases", ()):
        effects = [
            row
            for row in case.get("effects", ())
            if int(row.get("candidate_slot") or 0) == 1
        ]
        if len(effects) != 1:
            raise ValueError(
                f"expected one candidate-1 treatment for seed {case.get('seed')}; "
                f"found {len(effects)}"
            )
        result[int(case["seed"])] = {**effects[0], "decision_step": case["decision_step"]}
    return result


def _treatment_branch_by_seed(root: Path) -> dict[int, dict[str, Any]]:
    result = {}
    for path in sorted((root / "branches").glob("*.json")):
        row = _read_json(path)
        if row.get("is_control") or int(row.get("candidate_slot") or 0) != 1:
            continue
        result[int(row["seed"])] = row
    return result


def _discovery_by_seed(root: Path) -> dict[int, dict[str, Any]]:
    return {
        int(row["seed"]): row
        for path in sorted((root / "discoveries").glob("*.json"))
        for row in [_read_json(path)]
    }


def _physical_predecision_identity(
    p30: Mapping[str, Any], p50: Mapping[str, Any]
) -> dict[str, bool]:
    lhs, rhs = p30.get("decision"), p50.get("decision")
    if lhs is None or rhs is None:
        return {
            "both_decisions_present": False,
            "same_decision_step": False,
            "same_lane_and_candidate": False,
            "same_physical_observation": False,
            "same_pattern_context": False,
        }
    lhs_candidates = lhs.get("candidates") or []
    rhs_candidates = rhs.get("candidates") or []
    lhs_ids = [(row.get("vehicle_id"), row.get("lane_id")) for row in lhs_candidates]
    rhs_ids = [(row.get("vehicle_id"), row.get("lane_id")) for row in rhs_candidates]
    lhs_obs = lhs.get("candidate_observations") or {}
    rhs_obs = rhs.get("candidate_observations") or {}
    common_ids = sorted(set(lhs_obs) & set(rhs_obs))
    physical_equal = bool(common_ids) and lhs_ids == rhs_ids and all(
        np.array_equal(
            np.asarray(lhs_obs[vehicle_id], dtype=np.float32)[:134],
            np.asarray(rhs_obs[vehicle_id], dtype=np.float32)[:134],
        )
        for vehicle_id in common_ids
    )
    return {
        "both_decisions_present": True,
        "same_decision_step": int(lhs["decision_step"]) == int(rhs["decision_step"]),
        "same_lane_and_candidate": lhs.get("lane_id") == rhs.get("lane_id")
        and lhs_ids == rhs_ids,
        "same_physical_observation": physical_equal,
        "same_pattern_context": lhs.get("pattern_context") == rhs.get("pattern_context"),
    }


def _command_trace_contrast(
    p30: Mapping[str, Any], p50: Mapping[str, Any]
) -> dict[str, Any]:
    def commands(branch: Mapping[str, Any]) -> dict[int, float]:
        target = branch.get("target_vehicle_id")
        return {
            int(row["relative_step"]): float(row["projected_acceleration_mps2"])
            for row in branch.get("command_rows", ())
            if row.get("vehicle_id") == target
            and row.get("projected_acceleration_mps2") is not None
        }

    lhs, rhs = commands(p30), commands(p50)
    shared = sorted(set(lhs) & set(rhs))
    deltas = [abs(rhs[step] - lhs[step]) for step in shared]
    distinct = sum(delta >= COMMAND_DELTA_MPS2 for delta in deltas)
    fraction = distinct / len(shared) if shared else 0.0
    return {
        "shared_command_steps": len(shared),
        "distinct_command_steps": distinct,
        "distinct_step_fraction": fraction,
        "maximum_absolute_acceleration_delta_mps2": max(deltas, default=0.0),
        "case_trace_distinct": bool(shared)
        and fraction + 1e-12 >= CASE_DISTINCT_STEP_FRACTION,
    }


def _qualified(effect: Mapping[str, Any], minimum_dose: int) -> bool:
    return int(effect["target_command_step_count"]) >= minimum_dose and all(
        bool(value) for value in effect["constraints"].values()
    )


def analyze(p30_root: Path, p50_root: Path) -> dict[str, Any]:
    p30_summary = _read_json(p30_root / "summary.json")
    p50_summary = _read_json(p50_root / "summary.json")
    p30_protocol = p30_summary["protocol"]
    p50_protocol = p50_summary["protocol"]
    if p30_protocol["action_set"] != ACTIVATION_SPEED_CEILING_ACTION_SET:
        raise ValueError("p30 input is not the frozen activation_speed_ceiling_v1 action set")
    if p50_protocol["action_set"] != ACTIVATION_SPEED_CEILING_P50_ACTION_SET:
        raise ValueError("p50 input is not activation_speed_ceiling_p50_v1")
    frozen_keys = (
        "demand",
        "seeds",
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
        "landmark_contract",
        "maximum_landmarks_per_cell",
        "reward_refs_sha256",
    )
    protocol_matches = {
        key: p30_protocol.get(key) == p50_protocol.get(key) for key in frozen_keys
    }
    p30_effects, p50_effects = _effect_by_seed(p30_summary), _effect_by_seed(p50_summary)
    p30_branches = _treatment_branch_by_seed(p30_root)
    p50_branches = _treatment_branch_by_seed(p50_root)
    p30_discoveries = _discovery_by_seed(p30_root)
    p50_discoveries = _discovery_by_seed(p50_root)
    requested = set(int(seed) for seed in p50_protocol["seeds"])
    common = sorted(
        requested
        & set(p30_effects)
        & set(p50_effects)
        & set(p30_branches)
        & set(p50_branches)
        & set(p30_discoveries)
        & set(p50_discoveries)
    )
    minimum_dose = int(p50_protocol["minimum_dose_steps"])
    rows = []
    for seed in common:
        identity = _physical_predecision_identity(
            p30_discoveries[seed], p50_discoveries[seed]
        )
        trace = _command_trace_contrast(p30_branches[seed], p50_branches[seed])
        lhs, rhs = p30_effects[seed], p50_effects[seed]
        rows.append(
            {
                "seed": seed,
                "identity": identity,
                "same_target_vehicle": p30_branches[seed].get("target_vehicle_id")
                == p50_branches[seed].get("target_vehicle_id"),
                "p30_effect": lhs,
                "p50_effect": rhs,
                "p30_qualified": _qualified(lhs, minimum_dose),
                "p50_qualified": _qualified(rhs, minimum_dose),
                "common_qualified": _qualified(lhs, minimum_dose)
                and _qualified(rhs, minimum_dose),
                "trace": trace,
            }
        )
    adequate_trace_rows = [
        row
        for row in rows
        if int(row["p50_effect"]["target_command_step_count"]) >= minimum_dose
        and int(row["p30_effect"]["target_command_step_count"]) >= minimum_dose
    ]
    trace_distinct_fraction = (
        sum(row["trace"]["case_trace_distinct"] for row in adequate_trace_rows)
        / len(adequate_trace_rows)
        if adequate_trace_rows
        else 0.0
    )
    identity_gates = {
        "all_p30_source_cases_are_common": set(common) == set(p30_effects),
        "all_requested_cases_are_common_when_first_onset": (
            int(p50_protocol.get("decision_index", 1)) != 1 or set(common) == requested
        ),
        "all_frozen_protocol_fields_match": all(protocol_matches.values()),
        "all_cross_treatment_physical_predecisions_match": bool(rows)
        and all(all(row["identity"].values()) for row in rows),
        "all_cross_treatment_targets_match": bool(rows)
        and all(row["same_target_vehicle"] for row in rows),
    }
    p50_integrity_gates = {
        **dict(p50_summary.get("smoke_gates") or {}),
        **identity_gates,
        "p50_trace_distinct_case_fraction_at_least_30pct": (
            trace_distinct_fraction >= 0.30
        ),
    }

    common_qualified = [
        row
        for row in rows
        if row["common_qualified"]
        and all(row["identity"].values())
        and row["same_target_vehicle"]
    ]

    def arm_metrics(name: str) -> dict[str, Any]:
        effects = [row[f"{name}_effect"] for row in common_qualified]
        values = [float(row["safety_composite_reference_units"]) for row in effects]
        return {
            "qualified_case_count": len(effects),
            "qualified_harmful_fraction": (
                sum(row["classification"] == "harmful" for row in effects) / len(effects)
                if effects
                else None
            ),
            "qualified_effect_mean": float(np.mean(values)) if values else None,
            "qualified_effect_p10": float(np.percentile(values, 10.0)) if values else None,
            "helpful_seeds": sorted(
                row["seed"]
                for row in common_qualified
                if row[f"{name}_effect"]["classification"] == "helpful"
            ),
            "nonbeneficial_seeds": sorted(
                row["seed"]
                for row in common_qualified
                if row[f"{name}_effect"]["classification"] != "helpful"
            ),
            "effect_beyond_replay_fraction": (
                sum(abs(value) > float(p50_summary["frozen_deadband"]) for value in values)
                / len(values)
                if values
                else 0.0
            ),
        }

    p30_metrics, p50_metrics = arm_metrics("p30"), arm_metrics("p50")
    efficacy_gates = {
        "has_common_qualified_cases": bool(common_qualified),
        "p50_harmful_fraction_lower_than_p30": bool(common_qualified)
        and p50_metrics["qualified_harmful_fraction"]
        < p30_metrics["qualified_harmful_fraction"],
        "p50_p10_higher_than_p30": bool(common_qualified)
        and p50_metrics["qualified_effect_p10"] > p30_metrics["qualified_effect_p10"],
        "p50_mean_no_more_than_0p02_below_p30": bool(common_qualified)
        and p50_metrics["qualified_effect_mean"]
        >= p30_metrics["qualified_effect_mean"] - 0.02,
        "p50_helpful_in_at_least_three_seeds": len(p50_metrics["helpful_seeds"]) >= 3,
        "p50_nonbeneficial_in_at_least_three_seeds": (
            len(p50_metrics["nonbeneficial_seeds"]) >= 3
        ),
        "p50_effect_beyond_replay_fraction_at_least_30pct": (
            p50_metrics["effect_beyond_replay_fraction"] >= 0.30
        ),
    }
    return {
        "schema": SCHEMA,
        "development_only": True,
        "p30_root": str(p30_root),
        "p50_root": str(p50_root),
        "protocol_field_matches": protocol_matches,
        "common_seed_count": len(common),
        "common_seeds": common,
        "command_trace_contract": {
            "absolute_projected_acceleration_delta_mps2": COMMAND_DELTA_MPS2,
            "minimum_distinct_step_fraction_per_case": CASE_DISTINCT_STEP_FRACTION,
        },
        "frozen_deadband": float(p50_summary["frozen_deadband"]),
        "adequate_trace_case_count": len(adequate_trace_rows),
        "trace_distinct_case_fraction": trace_distinct_fraction,
        "p50_integrity_gates": p50_integrity_gates,
        "p50_integrity_pass": all(p50_integrity_gates.values()),
        "common_qualified_case_count": len(common_qualified),
        "p30_metrics": p30_metrics,
        "p50_metrics": p50_metrics,
        "efficacy_gates": efficacy_gates,
        "efficacy_pass": all(efficacy_gates.values()),
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p30", type=Path, required=True)
    parser.add_argument("--p50", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(args.p30.resolve(), args.p50.resolve())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "summary.json", payload)
    print(
        json.dumps(
            {
                "p50_integrity_pass": payload["p50_integrity_pass"],
                "efficacy_pass": payload["efficacy_pass"],
                "common_seed_count": payload["common_seed_count"],
                "common_qualified_case_count": payload["common_qualified_case_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
