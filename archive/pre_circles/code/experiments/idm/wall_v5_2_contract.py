"""Frozen, non-outcome contract for the prospective IDM wall v5.2 pilot."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Iterable

from counterfactual.headroom import TreatmentRepairCriteria
from idm.contract import WALL_V5_2_PILOT_SEEDS, validate_seed_role
from policies.idm_analytical import (
    FROZEN_IDM_ANALYTICAL_CONTROL_ARM_ID,
    build_frozen_idm_analytical_policy,
    frozen_idm_analytical_manifest,
)


PROFILE_ID = "adaptive_compact_wall_v5_2_prospective"
DECISION_STEPS = (620, 660, 680, 700)
OUTCOME_STEPS = 260
TREATMENT_STEPS = 20
TOTAL_CONTROLLER_STEPS = 960
REPEAT_SEED_COUNT = 2
FEASIBILITY_ANALYSIS_SCHEMA = "idm_compact_wall_feasibility_analysis/1"


def overlay_overrides() -> dict[str, Any]:
    """Return a fresh copy of the exact v5.2 treatment definition.

    Relative to v5.1, the only treatment change is the declared 20 m front
    and rear activation-gap floor.  Every other value is copied from the
    frozen v5.1 compact-wall profile.
    """

    return {
        "wall_params": {
            "d_lo": 200.0,
            "d_hi": 1000.0,
            "wall_speed": 10.0,
            "wall_tau": 3.0,
            "a_max_dec": 0.3,
            "tau_ramp_step": 0.05,
        },
        "require_all_occupied_lanes": True,
        "geometry_mode": "compact_moving",
        "max_longitudinal_span_m": 50.0,
        "planned_treatment_duration_s": 20.0,
        "compact_release_boundary_m": 50.0,
        "compact_selection_mode": "trajectory_coherent",
        "compact_assembly_duration_s": 10.0,
        "compact_assembly_speed_tolerance_mps": 2.0,
        "compact_assembly_a_max_dec_mps2": 1.0,
        "compact_control_mode": "adaptive_servo",
        "compact_servo_a_max_dec_mps2": 1.0,
        "compact_max_activation_speed_spread_mps": 4.0,
        "compact_max_speed_reduction_fraction": 0.20,
        "compact_shock_gap_lo_mps": 5.0,
        "compact_shock_gap_hi_mps": 15.0,
        "compact_position_kp_s_inv": 0.05,
        "compact_position_kd": 0.5,
        "compact_position_speed_offset_cap_mps": 1.0,
        "compact_safety_min_front_ttc_s": 3.0,
        "compact_safety_min_rear_ttc_s": 4.0,
        "compact_influence_upstream_m": 100.0,
        "compact_influence_max_followers": 3,
        "compact_influence_min_ttc_s": 1.5,
        "activation_min_front_gap_m": 20.0,
        "activation_min_rear_gap_m": 20.0,
    }


def treatment_repair_criteria() -> TreatmentRepairCriteria:
    """Primary v5.2 gate; collapse prevention remains secondary."""

    return TreatmentRepairCriteria(require_collapse_prevention_diversity=False)


def require_accepted_damping_factory(
    factory: Callable[[], Any] | None,
) -> Callable[[], Any]:
    """Fail closed unless the canonical analytical control factory is injected."""

    if factory is None:
        raise RuntimeError("v5.2 requires an explicitly injected analytical damping factory")
    if factory is not build_frozen_idm_analytical_policy:
        raise RuntimeError(
            "v5.2 accepts only build_frozen_idm_analytical_policy; "
            "implicit, legacy, residual, and wrapper factories are forbidden"
        )
    return factory


def validate_pilot_seeds(seeds: Iterable[int]) -> tuple[int, ...]:
    requested = validate_seed_role("wall_v5_2_pilot", seeds)
    if tuple(requested) != tuple(WALL_V5_2_PILOT_SEEDS):
        raise ValueError(
            "the prospective v5.2 pilot freezes the complete ordered cohort "
            f"{WALL_V5_2_PILOT_SEEDS}"
        )
    return requested


def profile_manifest() -> dict[str, Any]:
    payload = {
        "profile_id": PROFILE_ID,
        "base_controller_id": FROZEN_IDM_ANALYTICAL_CONTROL_ARM_ID,
        "base_controller": frozen_idm_analytical_manifest(),
        "overlay_overrides": overlay_overrides(),
        "seeds": list(WALL_V5_2_PILOT_SEEDS),
        "decision_steps": list(DECISION_STEPS),
        "outcome_steps": OUTCOME_STEPS,
        "treatment_steps": TREATMENT_STEPS,
        "total_controller_steps": TOTAL_CONTROLLER_STEPS,
        "repeat_seed_count": REPEAT_SEED_COUNT,
        "treatment_repair_criteria": treatment_repair_criteria().to_dict(),
        "collapse_prevention_role": "prespecified_secondary_not_gate",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**payload, "profile_sha256": sha256(encoded).hexdigest()}


def select_repeat_cases(
    timing_scan: Path,
    seeds: Iterable[int],
    decision_steps: Iterable[int] = DECISION_STEPS,
) -> dict[int, int]:
    """Select the first two admitted opportunities without opening outcomes."""

    allowed_decisions = tuple(sorted({int(step) for step in decision_steps}))
    selected: dict[int, int] = {}
    for seed in sorted(int(value) for value in seeds):
        path = Path(timing_scan) / "seeds" / f"seed{seed}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        admitted = sorted(
            int(row["decision_step"])
            for row in payload.get("results", ())
            if int(row["decision_step"]) in allowed_decisions and bool(row.get("admitted"))
        )
        if admitted:
            selected[seed] = admitted[0]
        if len(selected) == REPEAT_SEED_COUNT:
            break
    if len(selected) < REPEAT_SEED_COUNT:
        raise RuntimeError(
            "v5.2 requires admitted opportunities in at least two independent seeds "
            "for exact damping and engaged-treatment repeats"
        )
    return selected


def load_selected_feasibility_scan(scan_root: Path) -> dict[str, Any]:
    """Load the treatment-blind scan and fail closed on its selected cohort.

    This consumes only the machine-readable placement-feasibility analysis.
    It does not inspect treatment outcomes and does not alter the wall profile.
    """

    root = Path(scan_root).resolve()
    analysis_path = root / "analysis.json"
    manifest_path = root / "manifest.json"
    analysis_bytes = analysis_path.read_bytes()
    analysis = json.loads(analysis_bytes.decode("utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if analysis.get("schema") != FEASIBILITY_ANALYSIS_SCHEMA:
        raise ValueError("wrong compact-wall feasibility analysis schema")
    if analysis.get("treatment_outcomes_read") is not False:
        raise ValueError("penetration selection must remain treatment blind")
    if analysis.get("selection_passed") is not True:
        raise RuntimeError("no CAV penetration passed the frozen availability rule")
    penetration = float(analysis["selected_lowest_feasible_penetration"])
    declared = tuple(float(x) for x in analysis["frozen_grid"]["cav_penetrations"])
    if penetration not in declared:
        raise ValueError("selected penetration is outside the frozen feasibility grid")
    if tuple(int(x) for x in analysis["frozen_grid"]["seeds"]) != tuple(
        WALL_V5_2_PILOT_SEEDS
    ):
        raise ValueError("feasibility analysis uses the wrong seed cohort")
    if tuple(int(x) for x in analysis["frozen_grid"]["decision_steps"]) != DECISION_STEPS:
        raise ValueError("feasibility analysis uses the wrong decision windows")
    if manifest.get("wall_profile") != profile_manifest():
        raise ValueError("feasibility scan wall profile differs from frozen v5.2")
    if manifest.get("base_control_arm") != frozen_idm_analytical_manifest():
        raise ValueError("feasibility scan does not use the frozen analytical control arm")
    token = f"p{round(penetration * 100):02d}"
    case_root = root / token
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


def assert_no_prior_seed_artifacts(
    runs_root: Path,
    seeds: Iterable[int],
    *,
    allowed_roots: Iterable[Path] = (),
) -> dict[str, Any]:
    """Repeat the prospective-cohort artifact audit without reading outcomes.

    Only top-level seed assignments in JSON artifacts are inspected.  The
    designated resumable output and an explicitly supplied timing-scan input
    may be allowed; any other assignment of a v5.2 seed fails closed.
    """

    assigned = {int(seed) for seed in seeds}
    allowed = tuple(Path(path).resolve() for path in allowed_roots)
    conflicts: list[str] = []
    inspected = 0
    root = Path(runs_root).resolve()
    if root.exists():
        for path in root.rglob("*.json"):
            resolved = path.resolve()
            if any(resolved == item or item in resolved.parents for item in allowed):
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            inspected += 1
            if not isinstance(payload, dict):
                continue
            values: set[int] = set()
            if "seed" in payload:
                try:
                    values.add(int(payload["seed"]))
                except (TypeError, ValueError):
                    pass
            if isinstance(payload.get("seeds"), (list, tuple)):
                for value in payload["seeds"]:
                    try:
                        values.add(int(value))
                    except (TypeError, ValueError):
                        pass
            if values & assigned:
                conflicts.append(str(resolved))
    if conflicts:
        raise RuntimeError(
            "prospective v5.2 seed artifacts already exist outside allowed roots: "
            + ", ".join(sorted(conflicts))
        )
    return {
        "passed": True,
        "runs_root": str(root),
        "inspected_json_artifact_count": inspected,
        "assigned_seeds": sorted(assigned),
        "allowed_roots": [str(path) for path in allowed],
        "conflicts": [],
    }
