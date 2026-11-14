"""Frozen partitions and gates for the p30 IDM escalation predictor."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

from counterfactual.idm_gate_labels import contract_manifest as gate_b_contract_manifest
from counterfactual.idm_predictor_contract import feature_contract_manifest
from experiments.idm.wall_v5_4_p30_support_contract import (
    CAV_PENETRATION,
    DECISION_STEPS,
    DEMAND_MAINLINE_VPH,
    DEMAND_RAMP_VPH,
    OUTCOME_STEPS,
    TOTAL_CONTROLLER_STEPS,
    TREATMENT_STEPS,
    overlay_overrides,
    profile_manifest,
    require_accepted_damping_factory,
)
from experiments.idm.wall_v5_2_contract import assert_no_prior_seed_artifacts
from policies.idm_analytical import frozen_idm_analytical_manifest


CONTRACT_VERSION = "idm_v5_4_p30_two_stage_predictor/1"
DEVELOPMENT_SEEDS = tuple(range(4401, 4521))
VALIDATION_SEEDS = tuple(range(4601, 4661))
FINAL_TEST_SEEDS = tuple(range(5001, 5061))
SUPPORT_SUMMARY = Path(
    "runs/eval/idm_wall_v5_4_p30_support_pilot_s4333_4352/summary.json"
)


def validate_role_seeds(role: str, seeds: Iterable[int]) -> tuple[int, ...]:
    values = tuple(int(seed) for seed in seeds)
    allowed = {
        "development": DEVELOPMENT_SEEDS,
        "validation": VALIDATION_SEEDS,
        "final_test": FINAL_TEST_SEEDS,
    }
    if role not in allowed:
        raise ValueError(f"unknown predictor seed role {role!r}")
    if values != allowed[role]:
        raise ValueError(f"{role} seeds are frozen at {allowed[role]}")
    return values


def development_availability_rule() -> dict[str, Any]:
    return {
        "minimum_admitted_seeds": 60,
        "minimum_admitted_seed_windows": 80,
        "minimum_distinct_decision_steps": 3,
        "minimum_projected_duration_fraction": 0.80,
        "cav_penetration": CAV_PENETRATION,
    }


def validation_availability_rule() -> dict[str, Any]:
    """Treatment-blind support required before opening validation outcomes."""
    return {
        "minimum_admitted_seeds": 30,
        "minimum_admitted_seed_windows": 40,
        "minimum_distinct_decision_steps": 3,
        "minimum_projected_duration_fraction": 0.80,
        "cav_penetration": CAV_PENETRATION,
    }


def fitting_gates() -> dict[str, int]:
    return {
        "minimum_admitted_independent_seeds": 60,
        "minimum_dose_qualified_independent_seeds": 50,
        "minimum_safe_independent_seeds": 20,
        "minimum_unsafe_independent_seeds": 20,
        "minimum_joint_beneficial_independent_seeds": 15,
        "minimum_joint_nonbeneficial_independent_seeds": 15,
    }


def predictor_manifest(role: str, seeds: Iterable[int]) -> dict[str, Any]:
    values = validate_role_seeds(role, seeds)
    availability = (
        development_availability_rule()
        if role == "development"
        else validation_availability_rule()
    )
    payload = {
        "contract_version": CONTRACT_VERSION,
        "role": role,
        "seeds": list(values),
        "decision_steps": list(DECISION_STEPS),
        "demand_mainline_vph": DEMAND_MAINLINE_VPH,
        "demand_ramp_vph": DEMAND_RAMP_VPH,
        "cav_penetration": CAV_PENETRATION,
        "outcome_steps": OUTCOME_STEPS,
        "treatment_steps": TREATMENT_STEPS,
        "total_controller_steps": TOTAL_CONTROLLER_STEPS,
        "gate_f": {
            "type": "deterministic_predecision_physical_supervisor",
            "learned": False,
            "checks": [
                "full_occupied_lane_coverage",
                "compact_geometry_and_predicted_span",
                "kinematic_authority",
                "activation_gap_ttc_and_acceleration",
                "projected_treatment_dose",
            ],
        },
        "gate_b": {
            "type": "conditional_learned_safety_and_benefit_gate",
            "training_population": "gate_f_admitted_and_dose_qualified",
            "label_contract": gate_b_contract_manifest(),
        },
        "feature_contract": feature_contract_manifest(),
        "wall_profile": profile_manifest(),
        "base_control_arm": frozen_idm_analytical_manifest(),
        "overlay_overrides": overlay_overrides(),
        **{
            (
                "development_availability_rule"
                if role == "development"
                else "validation_availability_rule"
            ): availability
        },
        "fitting_gates": fitting_gates(),
        "sealed_partitions": {
            "validation": list(VALIDATION_SEEDS),
            "final_test": list(FINAL_TEST_SEEDS),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**payload, "predictor_contract_sha256": sha256(encoded).hexdigest()}


def require_support_authorization(repo_root: Path) -> dict[str, Any]:
    path = Path(repo_root) / SUPPORT_SUMMARY
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "idm_wall_v5_4_support_summary/1":
        raise ValueError("wrong v5.4 support summary schema")
    if payload.get("passed") is not True:
        raise RuntimeError("v5.4 support pilot did not pass")
    if payload.get("predictor_development_generation_authorized") is not True:
        raise RuntimeError("predictor development generation is not authorized")
    if payload.get("manifest", {}).get("wall_profile") != profile_manifest():
        raise ValueError("support treatment profile differs from predictor contract")
    if payload.get("gate_b_label_contract") != gate_b_contract_manifest():
        raise ValueError("support Gate-B label differs from predictor contract")
    return payload


def select_repeat_cases(case_root: Path, *, count: int = 4) -> dict[int, int]:
    return select_partition_repeat_cases(
        case_root, seeds=DEVELOPMENT_SEEDS, count=count
    )


def select_partition_repeat_cases(
    case_root: Path, *, seeds: Iterable[int], count: int = 4
) -> dict[int, int]:
    selected: dict[int, int] = {}
    for seed in tuple(int(value) for value in seeds):
        payload = json.loads(
            (Path(case_root) / "seeds" / f"seed{seed}.json").read_text(encoding="utf-8")
        )
        steps = sorted(
            int(row["decision_step"])
            for row in payload["results"]
            if bool(row.get("admitted"))
        )
        if steps:
            selected[seed] = steps[0]
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"{count} independent admitted starts are required for repeats")
    return selected


def load_development_feasibility_scan(scan_root: Path) -> dict[str, Any]:
    return _load_feasibility_scan(
        scan_root,
        expected_schema="idm_wall_v5_4_predictor_development_feasibility/1",
        expected_seeds=DEVELOPMENT_SEEDS,
        expected_rule=development_availability_rule(),
    )


def load_validation_feasibility_scan(scan_root: Path) -> dict[str, Any]:
    return _load_feasibility_scan(
        scan_root,
        expected_schema="idm_wall_v5_4_predictor_validation_feasibility/1",
        expected_seeds=VALIDATION_SEEDS,
        expected_rule=validation_availability_rule(),
    )


def _load_feasibility_scan(
    scan_root: Path,
    *,
    expected_schema: str,
    expected_seeds: tuple[int, ...],
    expected_rule: dict[str, Any],
) -> dict[str, Any]:
    root = Path(scan_root).resolve()
    analysis_path = root / "analysis.json"
    analysis_bytes = analysis_path.read_bytes()
    analysis = json.loads(analysis_bytes.decode("utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if analysis.get("schema") != expected_schema:
        raise ValueError("wrong predictor feasibility schema")
    if analysis.get("treatment_outcomes_read") is not False:
        raise ValueError("predictor feasibility scan must remain treatment blind")
    if analysis.get("selection_passed") is not True:
        raise RuntimeError("predictor availability gate did not pass")
    if manifest.get("wall_profile") != profile_manifest():
        raise ValueError("predictor scan wall profile differs from frozen v5.4")
    grid = analysis["frozen_grid"]
    if tuple(int(x) for x in grid["seeds"]) != expected_seeds:
        raise ValueError("predictor scan uses the wrong seed partition")
    if float(grid["cav_penetration"]) != CAV_PENETRATION:
        raise ValueError("predictor scan uses the wrong penetration")
    if analysis.get("availability_rule") != expected_rule:
        raise ValueError("predictor scan uses the wrong availability rule")
    return {
        "scan_root": str(root),
        "case_root": str(root),
        "analysis_path": str(analysis_path),
        "analysis_sha256": sha256(analysis_bytes).hexdigest(),
        "selected_cav_penetration": CAV_PENETRATION,
        "selection_rule": analysis["availability_rule"],
        "selected_summary": analysis["summary"],
    }


__all__ = [
    "CAV_PENETRATION", "CONTRACT_VERSION", "DECISION_STEPS", "DEMAND_MAINLINE_VPH",
    "DEMAND_RAMP_VPH", "DEVELOPMENT_SEEDS", "FINAL_TEST_SEEDS", "OUTCOME_STEPS",
    "TOTAL_CONTROLLER_STEPS", "TREATMENT_STEPS", "VALIDATION_SEEDS",
    "assert_no_prior_seed_artifacts", "development_availability_rule", "fitting_gates",
    "load_development_feasibility_scan", "load_validation_feasibility_scan",
    "overlay_overrides", "predictor_manifest",
    "profile_manifest", "require_accepted_damping_factory", "require_support_authorization",
    "select_partition_repeat_cases", "select_repeat_cases", "validate_role_seeds",
    "validation_availability_rule",
]
