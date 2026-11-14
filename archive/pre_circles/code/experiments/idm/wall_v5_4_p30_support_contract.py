"""Treatment-blind p30 amendment to the v5.4 support contract."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from experiments.idm.wall_v5_4_support_contract import (
    DECISION_STEPS,
    DEMAND_MAINLINE_VPH,
    DEMAND_RAMP_VPH,
    OUTCOME_STEPS,
    REPEAT_CASE_COUNT,
    SUPPORT_SEEDS,
    TOTAL_CONTROLLER_STEPS,
    TREATMENT_STEPS,
    assert_no_prior_seed_artifacts,
    overlay_overrides,
    profile_manifest,
    require_accepted_damping_factory,
    select_repeat_cases,
    validate_support_seeds,
)
from policies.idm_analytical import frozen_idm_analytical_manifest


CAV_PENETRATION = 0.30
FEASIBILITY_ANALYSIS_SCHEMA = "idm_wall_v5_4_p30_feasibility_analysis/1"


def availability_rule() -> dict[str, Any]:
    return {
        "minimum_admitted_seeds": 14,
        "minimum_admitted_seed_windows": 20,
        "minimum_distinct_decision_steps": 2,
        "minimum_projected_duration_fraction": 0.80,
        "penetration_frozen_after_p25_availability_nogo_before_any_v5_4_outcome": 0.30,
    }


def support_manifest() -> dict[str, Any]:
    payload = {
        "scientific_role": "prospective_v5_4_p30_emergency_handoff_support",
        "amends": "v5.4_p25_availability_nogo_without_treatment_outcomes",
        "seeds": list(SUPPORT_SEEDS),
        "decision_steps": list(DECISION_STEPS),
        "outcome_steps": OUTCOME_STEPS,
        "treatment_steps": TREATMENT_STEPS,
        "total_controller_steps": TOTAL_CONTROLLER_STEPS,
        "cav_penetration": CAV_PENETRATION,
        "demand_mainline_vph": DEMAND_MAINLINE_VPH,
        "demand_ramp_vph": DEMAND_RAMP_VPH,
        "availability_rule": availability_rule(),
        "wall_profile": profile_manifest(),
        "base_control_arm": frozen_idm_analytical_manifest(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "support_contract_sha256": sha256(encoded).hexdigest()}


def load_feasibility_scan(scan_root: Path) -> dict[str, Any]:
    root = Path(scan_root).resolve()
    analysis_path = root / "analysis.json"
    analysis_bytes = analysis_path.read_bytes()
    analysis = json.loads(analysis_bytes.decode("utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if analysis.get("schema") != FEASIBILITY_ANALYSIS_SCHEMA:
        raise ValueError("wrong v5.4 p30 feasibility analysis schema")
    if analysis.get("treatment_outcomes_read") is not False:
        raise ValueError("feasibility scan must remain treatment blind")
    if analysis.get("selection_passed") is not True:
        raise RuntimeError("frozen p30 availability rule did not pass")
    if manifest.get("wall_profile") != profile_manifest():
        raise ValueError("scan wall profile differs from frozen v5.4")
    if manifest.get("base_control_arm") != frozen_idm_analytical_manifest():
        raise ValueError("scan control arm differs from analytical damping")
    if tuple(int(x) for x in analysis["frozen_grid"]["seeds"]) != SUPPORT_SEEDS:
        raise ValueError("scan uses the wrong prospective cohort")
    if float(analysis["frozen_grid"]["cav_penetration"]) != CAV_PENETRATION:
        raise ValueError("scan uses the wrong frozen penetration")
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
    "CAV_PENETRATION", "DECISION_STEPS", "DEMAND_MAINLINE_VPH", "DEMAND_RAMP_VPH",
    "FEASIBILITY_ANALYSIS_SCHEMA", "OUTCOME_STEPS", "REPEAT_CASE_COUNT", "SUPPORT_SEEDS",
    "TOTAL_CONTROLLER_STEPS", "TREATMENT_STEPS", "assert_no_prior_seed_artifacts",
    "availability_rule", "load_feasibility_scan", "overlay_overrides", "profile_manifest",
    "require_accepted_damping_factory", "select_repeat_cases", "support_manifest",
    "validate_support_seeds",
]
