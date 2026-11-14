"""Validate the frozen treatment-blind fixed-p25 v5.4 availability scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from experiments.idm.wall_v5_4_support_contract import (
    CAV_PENETRATION,
    DECISION_STEPS,
    DEMAND_MAINLINE_VPH,
    DEMAND_RAMP_VPH,
    FEASIBILITY_ANALYSIS_SCHEMA,
    SUPPORT_SEEDS,
    availability_rule,
    profile_manifest,
)
from experiments.io import atomic_write_json
from policies.idm_analytical import frozen_idm_analytical_manifest


FORBIDDEN = {
    "arms", "benefit", "beneficial", "outcome", "outcomes", "reward", "rewards",
    "safety_service_oracle", "treatment_effect", "wall_selected",
}


def _reject_outcomes(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).lower() in FORBIDDEN:
                raise ValueError(f"forbidden outcome field {path}.{key}")
            _reject_outcomes(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_outcomes(nested, f"{path}[{index}]")


def _computed_admission(row: Mapping[str, Any]) -> bool:
    formation = row["formation"]
    coverage = row["lane_coverage"]
    projected = row["projected_dose"]
    safety = row["activation_safety"]
    occupied = {str(x) for x in formation["occupied_lane_ids"]}
    required = {str(x) for x in formation["required_lane_ids"]}
    selected = {str(x) for x in coverage["selected_lane_ids"]}
    compact = bool(
        formation.get("feasible")
        and formation.get("reason") == "feasible"
        and float(formation.get("longitudinal_span_m", float("inf"))) <= 50.0
        and float(formation.get("predicted_max_span_m", float("inf"))) <= 50.0
    )
    full = bool(occupied) and occupied == required == selected
    dose = bool(
        float(projected["requested_duration_s"]) == 20.0
        and float(projected["projected_duration_fraction"])
        >= float(availability_rule()["minimum_projected_duration_fraction"])
    )
    return bool(compact and full and dose and safety is not None and safety.get("safe"))


def analyze(scan_root: Path) -> dict[str, Any]:
    root = Path(scan_root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    _reject_outcomes(manifest, "manifest")
    expected = {
        "schema": "idm_wall_v5_4_feasibility_manifest/1",
        "scientific_role": "treatment_blind_v5_4_feasibility",
        "wall_was_applied": False,
        "treatment_outcomes_read": False,
        "demand_mainline_vph": DEMAND_MAINLINE_VPH,
        "demand_ramp_vph": DEMAND_RAMP_VPH,
        "cav_penetration": CAV_PENETRATION,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"manifest {key} must equal {value!r}")
    if tuple(int(x) for x in manifest["seeds"]) != SUPPORT_SEEDS:
        raise ValueError("wrong prospective seed cohort")
    if tuple(int(x) for x in manifest["decision_steps"]) != DECISION_STEPS:
        raise ValueError("wrong decision grid")
    if manifest.get("wall_profile") != profile_manifest():
        raise ValueError("wrong v5.4 wall profile")
    if manifest.get("base_control_arm") != frozen_idm_analytical_manifest():
        raise ValueError("wrong analytical control arm")
    audit = manifest.get("prospective_seed_artifact_audit") or {}
    if audit.get("passed") is not True or audit.get("conflicts") not in ([], ()):
        raise ValueError("prospective seed artifact audit did not pass")

    rows: list[dict[str, int]] = []
    errors: list[str] = []
    for seed in SUPPORT_SEEDS:
        path = root / "seeds" / f"seed{seed}.json"
        if not path.exists():
            errors.append(f"missing:{path}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        _reject_outcomes(payload, str(path))
        if payload.get("schema") != "idm_wall_v5_4_feasibility_seed/1":
            errors.append(f"schema:{path}")
            continue
        if (
            int(payload.get("seed", -1)) != seed
            or float(payload.get("cav_penetration", -1.0)) != CAV_PENETRATION
            or payload.get("wall_was_applied") is not False
            or payload.get("treatment_outcomes_read") is not False
        ):
            errors.append(f"contract:{path}")
            continue
        results = payload.get("results", [])
        if tuple(int(row["decision_step"]) for row in results) != DECISION_STEPS:
            errors.append(f"decision_grid:{path}")
            continue
        for row in results:
            admitted = _computed_admission(row)
            if bool(row.get("admitted")) != admitted:
                errors.append(f"admission_mismatch:{path}:step{row['decision_step']}")
            if admitted:
                rows.append({"seed": seed, "decision_step": int(row["decision_step"])})
    if errors:
        raise ValueError("scan integrity failed: " + "; ".join(errors))
    admitted_seeds = sorted({row["seed"] for row in rows})
    admitted_steps = sorted({row["decision_step"] for row in rows})
    rule = availability_rule()
    passed = bool(
        len(admitted_seeds) >= int(rule["minimum_admitted_seeds"])
        and len(rows) >= int(rule["minimum_admitted_seed_windows"])
        and len(admitted_steps) >= int(rule["minimum_distinct_decision_steps"])
    )
    return {
        "schema": FEASIBILITY_ANALYSIS_SCHEMA,
        "scientific_role": (
            f"treatment_blind_v5_4_fixed_p{round(CAV_PENETRATION * 100):02d}_feasibility"
        ),
        "treatment_outcomes_read": False,
        "frozen_grid": {
            "demand_mainline_vph": DEMAND_MAINLINE_VPH,
            "demand_ramp_vph": DEMAND_RAMP_VPH,
            "cav_penetration": CAV_PENETRATION,
            "seeds": list(SUPPORT_SEEDS),
            "decision_steps": list(DECISION_STEPS),
        },
        "availability_rule": rule,
        "summary": {
            "case_count": len(SUPPORT_SEEDS) * len(DECISION_STEPS),
            "admitted_seed_window_count": len(rows),
            "admitted_seed_count": len(admitted_seeds),
            "admitted_seeds": admitted_seeds,
            "admitted_decision_steps": admitted_steps,
        },
        "selection_passed": passed,
        "interpretation": (
            f"Fixed-p{round(CAV_PENETRATION * 100):02d} placement feasibility only; "
            "no outcomes opened."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.scan_root)
    atomic_write_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
