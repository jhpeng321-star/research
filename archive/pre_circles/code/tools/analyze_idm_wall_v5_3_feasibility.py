"""Validate and select the lowest passing treatment-blind v5.3 penetration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from experiments.idm.wall_v5_3_support_contract import (
    CAV_PENETRATIONS,
    DECISION_STEPS,
    DEMAND_MAINLINE_VPH,
    DEMAND_RAMP_VPH,
    SUPPORT_SEEDS,
    availability_rule,
    profile_manifest,
)
from experiments.io import atomic_write_json
from policies.idm_analytical import frozen_idm_analytical_manifest


SCHEMA = "idm_wall_v5_3_feasibility_analysis/1"
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


def _computed_admission(row: Mapping[str, Any]) -> dict[str, bool]:
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
    safe = bool(safety is not None and safety.get("safe"))
    return {
        "compact_geometry": compact,
        "full_occupied_lane_coverage": full,
        "projected_dose_ok": dose,
        "activation_safety_ok": safe,
        "admitted": compact and full and dose and safe,
    }


def analyze(scan_root: Path) -> dict[str, Any]:
    root = Path(scan_root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    _reject_outcomes(manifest, "manifest")
    required_manifest = {
        "schema": "idm_wall_v5_3_feasibility_manifest/1",
        "scientific_role": "treatment_blind_v5_3_feasibility",
        "wall_was_applied": False,
        "treatment_outcomes_read": False,
        "demand_mainline_vph": DEMAND_MAINLINE_VPH,
        "demand_ramp_vph": DEMAND_RAMP_VPH,
    }
    for key, expected in required_manifest.items():
        if manifest.get(key) != expected:
            raise ValueError(f"manifest {key} must equal {expected!r}")
    if tuple(float(x) for x in manifest["cav_penetrations"]) != CAV_PENETRATIONS:
        raise ValueError("wrong penetration grid")
    if tuple(int(x) for x in manifest["seeds"]) != SUPPORT_SEEDS:
        raise ValueError("wrong prospective seed cohort")
    if tuple(int(x) for x in manifest["decision_steps"]) != DECISION_STEPS:
        raise ValueError("wrong decision grid")
    if manifest.get("wall_profile") != profile_manifest():
        raise ValueError("wrong v5.3 wall profile")
    if manifest.get("base_control_arm") != frozen_idm_analytical_manifest():
        raise ValueError("wrong analytical control arm")
    audit = manifest.get("prospective_seed_artifact_audit") or {}
    if audit.get("passed") is not True or audit.get("conflicts") not in ([], ()):
        raise ValueError("prospective seed artifact audit did not pass")

    summaries: list[dict[str, Any]] = []
    errors: list[str] = []
    rule = availability_rule()
    for penetration in CAV_PENETRATIONS:
        rows: list[dict[str, Any]] = []
        for seed in SUPPORT_SEEDS:
            path = root / f"p{round(penetration * 100):02d}" / "seeds" / f"seed{seed}.json"
            if not path.exists():
                errors.append(f"missing:{path}")
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            _reject_outcomes(payload, str(path))
            if payload.get("schema") != "idm_wall_v5_3_feasibility_seed/1":
                errors.append(f"schema:{path}")
                continue
            if (
                int(payload.get("seed", -1)) != seed
                or float(payload.get("cav_penetration", -1.0)) != penetration
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
                computed = _computed_admission(row)
                if bool(row.get("admitted")) != computed["admitted"]:
                    errors.append(f"admission_mismatch:{path}:step{row['decision_step']}")
                rows.append({"seed": seed, "decision_step": int(row["decision_step"]), **computed})
        admitted = [row for row in rows if row["admitted"]]
        admitted_seeds = sorted({row["seed"] for row in admitted})
        admitted_steps = sorted({row["decision_step"] for row in admitted})
        passed = bool(
            len(admitted_seeds) >= int(rule["minimum_admitted_seeds"])
            and len(admitted) >= int(rule["minimum_admitted_seed_windows"])
            and len(admitted_steps) >= int(rule["minimum_distinct_decision_steps"])
        )
        summaries.append(
            {
                "cav_penetration": penetration,
                "case_count": len(rows),
                "admitted_seed_window_count": len(admitted),
                "admitted_seed_count": len(admitted_seeds),
                "admitted_seeds": admitted_seeds,
                "admitted_decision_steps": admitted_steps,
                "admission_fraction": len(admitted) / len(rows) if rows else 0.0,
                "passes_availability_rule": passed,
            }
        )
    if errors:
        raise ValueError("scan integrity failed: " + "; ".join(errors))
    selected = next(
        (row["cav_penetration"] for row in summaries if row["passes_availability_rule"]),
        None,
    )
    return {
        "schema": SCHEMA,
        "scientific_role": "treatment_blind_v5_3_placement_feasibility",
        "treatment_outcomes_read": False,
        "frozen_grid": {
            "demand_mainline_vph": DEMAND_MAINLINE_VPH,
            "demand_ramp_vph": DEMAND_RAMP_VPH,
            "cav_penetrations": list(CAV_PENETRATIONS),
            "seeds": list(SUPPORT_SEEDS),
            "decision_steps": list(DECISION_STEPS),
        },
        "availability_rule": rule,
        "penetration_summaries": summaries,
        "selected_lowest_feasible_penetration": selected,
        "selection_passed": selected is not None,
        "interpretation": "Placement feasibility only; no treatment outcome or predictor claim.",
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

