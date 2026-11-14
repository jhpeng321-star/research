"""Select a compact-wall penetration from a treatment-blind feasibility scan.

The analyzer deliberately has no treatment-outcome vocabulary.  It consumes
only the frozen scan manifest and predecision formation, lane-coverage,
activation-safety, and projected-dose diagnostics.  The selected penetration
is the lowest declared value that passes the predeclared availability rule.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments.io import atomic_write_json
from experiments.idm.wall_v5_2_contract import profile_manifest
from policies.idm_analytical import frozen_idm_analytical_manifest


SCHEMA = "idm_compact_wall_feasibility_analysis/1"
SCAN_MANIFEST_SCHEMA = "idm_compact_wall_feasibility_manifest/1"
SCAN_CELL_SCHEMA = "idm_compact_wall_feasibility_seed/1"
FROZEN_DEMAND_MAINLINE_VPH = 2700
FROZEN_DEMAND_RAMP_VPH = 400
FROZEN_PENETRATIONS = (0.10, 0.15, 0.20, 0.25, 0.30)
FROZEN_SEEDS = tuple(range(4301, 4313))
FROZEN_DECISION_STEPS = (620, 660, 680, 700)
FROZEN_SPAN_CAP_M = 50.0
FROZEN_PLANNED_DURATION_S = 20.0

_FORBIDDEN_OUTCOME_KEYS = {
    "arms",
    "benefit",
    "beneficial",
    "collapse_prevented",
    "outcome",
    "outcomes",
    "reward",
    "rewards",
    "safety_service_oracle",
    "treatment_effect",
    "wall_selected",
}


@dataclass(frozen=True)
class AvailabilityRule:
    minimum_admitted_seeds: int = 8
    minimum_admitted_seed_windows: int = 12
    minimum_distinct_decision_steps: int = 2
    minimum_projected_duration_fraction: float = 0.80


def _penetration_token(value: float) -> str:
    return f"p{round(float(value) * 100):02d}"


def _reject_outcome_fields(value: Any, *, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).lower()
            if normalized in _FORBIDDEN_OUTCOME_KEYS:
                raise ValueError(f"treatment-outcome field {path}.{key} is forbidden")
            _reject_outcome_fields(nested, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_outcome_fields(nested, path=f"{path}[{index}]")


def _as_string_set(value: Iterable[Any]) -> set[str]:
    return {str(item) for item in value}


def _row_admission(row: Mapping[str, Any], rule: AvailabilityRule) -> dict[str, Any]:
    """Reconstruct admission from treatment-blind predecision diagnostics."""
    formation = row["formation"]
    safety = row["activation_safety"]
    projected_dose = row["projected_dose"]
    lane_coverage = row["lane_coverage"]

    occupied = _as_string_set(formation["occupied_lane_ids"])
    required = _as_string_set(formation["required_lane_ids"])
    selected = _as_string_set(lane_coverage["selected_lane_ids"])
    full_occupied_lane_coverage = bool(occupied) and required == occupied and selected == occupied

    compact_geometry = (
        str(formation.get("reason")) == "feasible"
        and bool(formation.get("feasible"))
        and float(formation.get("longitudinal_span_m", float("inf")))
        <= FROZEN_SPAN_CAP_M
        and float(formation.get("predicted_max_span_m", float("inf")))
        <= FROZEN_SPAN_CAP_M
    )
    projected_fraction = float(projected_dose["projected_duration_fraction"])
    projected_dose_ok = (
        float(projected_dose["requested_duration_s"]) == FROZEN_PLANNED_DURATION_S
        and projected_fraction >= rule.minimum_projected_duration_fraction
    )
    safety_ok = bool(safety is not None and safety.get("safe"))
    admitted = compact_geometry and full_occupied_lane_coverage and projected_dose_ok and safety_ok
    return {
        "admitted": admitted,
        "compact_geometry": compact_geometry,
        "full_occupied_lane_coverage": full_occupied_lane_coverage,
        "projected_dose_ok": projected_dose_ok,
        "activation_safety_ok": safety_ok,
    }


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    _reject_outcome_fields(manifest, path="manifest")
    expected = {
        "schema": SCAN_MANIFEST_SCHEMA,
        "scientific_role": "treatment_blind_compact_wall_feasibility",
        "demand_mainline_vph": FROZEN_DEMAND_MAINLINE_VPH,
        "demand_ramp_vph": FROZEN_DEMAND_RAMP_VPH,
        "wall_was_applied": False,
        "treatment_outcomes_read": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"manifest {key} must equal {value!r}")
    if tuple(float(x) for x in manifest["cav_penetrations"]) != FROZEN_PENETRATIONS:
        raise ValueError("manifest penetration grid differs from the frozen p10--p30 grid")
    if tuple(int(x) for x in manifest["seeds"]) != FROZEN_SEEDS:
        raise ValueError("manifest seed grid differs from frozen v5.2 wall seeds 4301--4312")
    if tuple(int(x) for x in manifest["decision_steps"]) != FROZEN_DECISION_STEPS:
        raise ValueError("manifest decision grid differs from frozen steps 620/660/680/700")
    overlay = manifest["overlay_overrides"]
    if (
        overlay.get("geometry_mode") != "compact_moving"
        or overlay.get("compact_selection_mode") != "trajectory_coherent"
        or overlay.get("require_all_occupied_lanes") is not True
        or float(overlay.get("max_longitudinal_span_m", -1.0)) != FROZEN_SPAN_CAP_M
        or float(overlay.get("planned_treatment_duration_s", -1.0))
        != FROZEN_PLANNED_DURATION_S
    ):
        raise ValueError("manifest does not preserve compact full occupied-lane v5.2 geometry")
    if overlay != profile_manifest()["overlay_overrides"]:
        raise ValueError("manifest overlay differs from the exact frozen v5.2 profile")
    if manifest.get("wall_profile") != profile_manifest():
        raise ValueError("manifest wall profile/fingerprint differs from frozen v5.2")
    if manifest.get("base_control_arm") != frozen_idm_analytical_manifest():
        raise ValueError("manifest does not use the canonical frozen analytical damping arm")
    audit = manifest.get("prospective_seed_artifact_audit")
    if not isinstance(audit, Mapping) or audit.get("passed") is not True:
        raise ValueError("manifest lacks a passing prospective seed-artifact audit")
    if tuple(int(x) for x in audit.get("assigned_seeds", ())) != FROZEN_SEEDS:
        raise ValueError("prospective seed-artifact audit covers the wrong cohort")
    if audit.get("conflicts") not in ([], ()):
        raise ValueError("prospective seed-artifact audit reports conflicts")


def analyze(scan_root: Path, *, rule: AvailabilityRule = AvailabilityRule()) -> dict[str, Any]:
    root = Path(scan_root).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest(manifest)

    penetration_summaries: list[dict[str, Any]] = []
    integrity_errors: list[str] = []
    for penetration in FROZEN_PENETRATIONS:
        rows: list[dict[str, Any]] = []
        for seed in FROZEN_SEEDS:
            path = root / _penetration_token(penetration) / "seeds" / f"seed{seed}.json"
            if not path.exists():
                integrity_errors.append(f"missing:{path}")
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            _reject_outcome_fields(payload, path=str(path))
            if payload.get("schema") != SCAN_CELL_SCHEMA:
                integrity_errors.append(f"schema:{path}")
                continue
            if (
                int(payload.get("seed", -1)) != seed
                or int(payload.get("demand_mainline_vph", -1)) != FROZEN_DEMAND_MAINLINE_VPH
                or int(payload.get("demand_ramp_vph", -1)) != FROZEN_DEMAND_RAMP_VPH
                or float(payload.get("cav_penetration", -1.0)) != penetration
                or payload.get("wall_was_applied") is not False
                or payload.get("treatment_outcomes_read") is not False
            ):
                integrity_errors.append(f"contract:{path}")
                continue
            results = payload.get("results", [])
            result_steps = tuple(int(row["decision_step"]) for row in results)
            if result_steps != FROZEN_DECISION_STEPS:
                integrity_errors.append(f"decision_grid:{path}")
                continue
            for row in results:
                computed = _row_admission(row, rule)
                if bool(row.get("admitted")) != computed["admitted"]:
                    integrity_errors.append(
                        f"admission_mismatch:{path}:step{int(row['decision_step'])}"
                    )
                rows.append(
                    {
                        "seed": seed,
                        "decision_step": int(row["decision_step"]),
                        **computed,
                        "formation_reason": str(row["formation"]["reason"]),
                        "safety_reason": (
                            str(row["activation_safety"]["reason"])
                            if row["activation_safety"] is not None
                            else "not_checked"
                        ),
                        "projected_duration_fraction": float(
                            row["projected_dose"]["projected_duration_fraction"]
                        ),
                    }
                )

        admitted = [row for row in rows if row["admitted"]]
        admitted_seeds = sorted({row["seed"] for row in admitted})
        admitted_steps = sorted({row["decision_step"] for row in admitted})
        passes = (
            len(admitted_seeds) >= rule.minimum_admitted_seeds
            and len(admitted) >= rule.minimum_admitted_seed_windows
            and len(admitted_steps) >= rule.minimum_distinct_decision_steps
        )
        reason_counts: dict[str, int] = {}
        for row in rows:
            reason = "admitted" if row["admitted"] else row["formation_reason"]
            if row["compact_geometry"] and not row["full_occupied_lane_coverage"]:
                reason = "incomplete_occupied_lane_coverage"
            elif row["compact_geometry"] and not row["projected_dose_ok"]:
                reason = "insufficient_projected_dose"
            elif (
                row["compact_geometry"]
                and row["full_occupied_lane_coverage"]
                and row["projected_dose_ok"]
                and not row["activation_safety_ok"]
            ):
                reason = f"activation_safety:{row['safety_reason']}"
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        penetration_summaries.append(
            {
                "cav_penetration": penetration,
                "case_count": len(rows),
                "admitted_seed_window_count": len(admitted),
                "admitted_seed_count": len(admitted_seeds),
                "admitted_seeds": admitted_seeds,
                "admitted_decision_steps": admitted_steps,
                "admission_fraction": len(admitted) / len(rows) if rows else 0.0,
                "reason_counts": dict(sorted(reason_counts.items())),
                "passes_availability_rule": passes,
            }
        )

    if integrity_errors:
        raise ValueError("scan integrity failed: " + "; ".join(integrity_errors))
    selected = next(
        (
            row["cav_penetration"]
            for row in penetration_summaries
            if row["passes_availability_rule"]
        ),
        None,
    )
    return {
        "schema": SCHEMA,
        "scientific_role": "treatment_blind_placement_feasibility_not_benefit",
        "treatment_outcomes_read": False,
        "frozen_grid": {
            "demand_mainline_vph": FROZEN_DEMAND_MAINLINE_VPH,
            "demand_ramp_vph": FROZEN_DEMAND_RAMP_VPH,
            "cav_penetrations": list(FROZEN_PENETRATIONS),
            "seeds": list(FROZEN_SEEDS),
            "decision_steps": list(FROZEN_DECISION_STEPS),
        },
        "availability_rule": asdict(rule),
        "penetration_summaries": penetration_summaries,
        "selected_lowest_feasible_penetration": selected,
        "selection_passed": selected is not None,
        "interpretation": (
            "Placement feasibility only; no claim about treatment benefit, safety after "
            "activation, realized dose, or predictor performance."
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
