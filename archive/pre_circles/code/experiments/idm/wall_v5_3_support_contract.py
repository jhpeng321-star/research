"""Frozen non-outcome contract for the v5.3 selective-gate support pilot."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

from experiments.idm.wall_v5_2_contract import (
    DECISION_STEPS,
    OUTCOME_STEPS,
    TREATMENT_STEPS,
    TOTAL_CONTROLLER_STEPS,
    assert_no_prior_seed_artifacts,
    require_accepted_damping_factory,
)
from experiments.idm.wall_v5_3_contract import overlay_overrides, profile_manifest
from policies.idm_analytical import frozen_idm_analytical_manifest


SUPPORT_SEEDS = tuple(range(4313, 4333))
CAV_PENETRATIONS = (0.10, 0.15, 0.20, 0.25, 0.30)
DEMAND_MAINLINE_VPH = 2700
DEMAND_RAMP_VPH = 400
REPEAT_CASE_COUNT = 2
FEASIBILITY_ANALYSIS_SCHEMA = "idm_wall_v5_3_feasibility_analysis/1"
FEASIBILITY_MANIFEST_SCHEMA = "idm_wall_v5_3_feasibility_manifest/1"
FEASIBILITY_CELL_SCHEMA = "idm_wall_v5_3_feasibility_seed/1"


def validate_support_seeds(seeds: Iterable[int]) -> tuple[int, ...]:
    values = tuple(int(seed) for seed in seeds)
    if values != SUPPORT_SEEDS:
        raise ValueError(f"v5.3 support seeds are frozen at {SUPPORT_SEEDS}")
    return values


def availability_rule() -> dict[str, Any]:
    return {
        "minimum_admitted_seeds": 14,
        "minimum_admitted_seed_windows": 20,
        "minimum_distinct_decision_steps": 2,
        "minimum_projected_duration_fraction": 0.80,
        "select_lowest_passing_penetration": True,
    }


def support_manifest() -> dict[str, Any]:
    payload = {
        "scientific_role": "prospective_v5_3_selective_gate_support",
        "seeds": list(SUPPORT_SEEDS),
        "decision_steps": list(DECISION_STEPS),
        "outcome_steps": OUTCOME_STEPS,
        "treatment_steps": TREATMENT_STEPS,
        "total_controller_steps": TOTAL_CONTROLLER_STEPS,
        "penetration_grid": list(CAV_PENETRATIONS),
        "demand_mainline_vph": DEMAND_MAINLINE_VPH,
        "demand_ramp_vph": DEMAND_RAMP_VPH,
        "availability_rule": availability_rule(),
        "wall_profile": profile_manifest(),
        "base_control_arm": frozen_idm_analytical_manifest(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "support_contract_sha256": sha256(encoded).hexdigest()}


def select_repeat_cases(case_root: Path) -> dict[int, int]:
    selected: dict[int, int] = {}
    for seed in SUPPORT_SEEDS:
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
        if len(selected) == REPEAT_CASE_COUNT:
            break
    if len(selected) != REPEAT_CASE_COUNT:
        raise RuntimeError("two independent admitted starts are required for exact repeats")
    return selected


def load_selected_feasibility_scan(scan_root: Path) -> dict[str, Any]:
    root = Path(scan_root).resolve()
    analysis_path = root / "analysis.json"
    manifest_path = root / "manifest.json"
    analysis_bytes = analysis_path.read_bytes()
    analysis = json.loads(analysis_bytes.decode("utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if analysis.get("schema") != FEASIBILITY_ANALYSIS_SCHEMA:
        raise ValueError("wrong v5.3 feasibility analysis schema")
    if analysis.get("treatment_outcomes_read") is not False:
        raise ValueError("feasibility selection must remain treatment blind")
    if analysis.get("selection_passed") is not True:
        raise RuntimeError("no penetration passed the frozen v5.3 availability rule")
    if manifest.get("wall_profile") != profile_manifest():
        raise ValueError("scan wall profile differs from frozen v5.3")
    if manifest.get("base_control_arm") != frozen_idm_analytical_manifest():
        raise ValueError("scan control arm differs from frozen analytical damping")
    grid = analysis["frozen_grid"]
    if tuple(int(x) for x in grid["seeds"]) != SUPPORT_SEEDS:
        raise ValueError("scan uses the wrong prospective seed cohort")
    if tuple(int(x) for x in grid["decision_steps"]) != DECISION_STEPS:
        raise ValueError("scan uses the wrong decision steps")
    penetration = float(analysis["selected_lowest_feasible_penetration"])
    if penetration not in CAV_PENETRATIONS:
        raise ValueError("selected penetration is outside the frozen grid")
    case_root = root / f"p{round(penetration * 100):02d}"
    if not case_root.is_dir():
        raise FileNotFoundError(case_root)
    return {
        "scan_root": str(root),
        "case_root": str(case_root),
        "analysis_path": str(analysis_path),
        "analysis_sha256": sha256(analysis_bytes).hexdigest(),
        "selected_cav_penetration": penetration,
        "selection_rule": analysis["availability_rule"],
        "selected_summary": next(
            row
            for row in analysis["penetration_summaries"]
            if float(row["cav_penetration"]) == penetration
        ),
    }


__all__ = [
    "CAV_PENETRATIONS",
    "DECISION_STEPS",
    "DEMAND_MAINLINE_VPH",
    "DEMAND_RAMP_VPH",
    "OUTCOME_STEPS",
    "SUPPORT_SEEDS",
    "TOTAL_CONTROLLER_STEPS",
    "TREATMENT_STEPS",
    "assert_no_prior_seed_artifacts",
    "availability_rule",
    "load_selected_feasibility_scan",
    "overlay_overrides",
    "profile_manifest",
    "require_accepted_damping_factory",
    "select_repeat_cases",
    "support_manifest",
    "validate_support_seeds",
]

