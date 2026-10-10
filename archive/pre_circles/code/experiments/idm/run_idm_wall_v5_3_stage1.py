"""Run the owner-approved seven-case IDM wall-v5.3 integrity replay."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

from counterfactual.actionability import dose_qualified
from counterfactual.headroom import HeadroomMargins, classify_case
from counterfactual.idm_predictor_contract import feature_contract_manifest, preflight_features
from counterfactual.outcomes import OutcomeDefinition, summarize_branch
from counterfactual.pilot import features_from_trace
from counterfactual.twin_replay import FrozenPulseSchedule, compare_traces, run_replay
from experiments import REPO_ROOT
from experiments.counterfactual.verify_counterfactual_escalation import (
    _audit_contract,
    _prefix_identity,
)
from experiments.idm.run_idm_wall_pilot import DEFAULT_SUMO_EXE, supervisor_overrides
from experiments.idm.wall_v5_2_contract import OUTCOME_STEPS, TOTAL_CONTROLLER_STEPS, TREATMENT_STEPS
from experiments.idm.wall_v5_3_contract import STAGE1_CASES, overlay_overrides, profile_manifest
from experiments.io import atomic_write_json
from idm.contract import SCENARIO_ROOT, transition_scenario_config
from policies.idm_analytical import build_frozen_idm_analytical_policy


DEFAULT_OUTPUT = REPO_ROOT / "runs" / "probe" / "idm_wall_v5_3_stage1_replay"
V5_2_ROOT = REPO_ROOT / "runs" / "eval" / "idm_wall_v5_2_prospective_p30_s4301_4312"
SCAN_ROOT = REPO_ROOT / "runs" / "probe" / "idm_compact_wall_feasibility_p10_p30_s4301_4312" / "p30"
REPEAT_CASES = {(4301, 660), (4302, 620)}


def _replay(seed: int, decision: int, *, treatment: bool, sumo_binary: str):
    schedule = FrozenPulseSchedule(decision, TREATMENT_STEPS) if treatment else (lambda _step: 0)
    return run_replay(
        project_root=SCENARIO_ROOT,
        sumo_binary=sumo_binary,
        sumo_config=transition_scenario_config(2700).name,
        seed=seed,
        decision_step=decision,
        horizon_steps=TOTAL_CONTROLLER_STEPS - decision,
        treatment_steps=TREATMENT_STEPS,
        cav_penetration=0.30,
        sumo_step_length=0.5,
        supervisor_overrides=supervisor_overrides(),
        digits=6,
        base_policy_factory=build_frozen_idm_analytical_policy,
        intervention_schedule=schedule,
        overlay_overrides=overlay_overrides(),
        controller_metadata={
            "scientific_role": "bounded_v5_3_stage1_integrity_replay",
            "wall_profile": profile_manifest(),
        },
    )


def _stored_case(root: Path, seed: int, decision: int) -> dict[str, Any]:
    payload = json.loads((root / "seeds" / f"seed{seed}.json").read_text(encoding="utf-8"))
    return next(row for row in payload["cases"] if int(row["decision_step"]) == decision)


def _scan(seed: int, decision: int) -> dict[str, Any]:
    payload = json.loads((SCAN_ROOT / "seeds" / f"seed{seed}.json").read_text(encoding="utf-8"))
    return next(row for row in payload["results"] if int(row["decision_step"]) == decision)


def _mechanics(audit: dict[str, Any]) -> dict[str, Any]:
    rows = sorted(
        (dict(row) for row in audit.get("treatment_audit_rows", ())),
        key=lambda row: int(row.get("relative_step", -1)),
    )
    # Older audit payloads expose the complete controller rows under this key.
    if not rows:
        rows = sorted(
            (dict(row) for row in audit.get("controller_audits", ())),
            key=lambda row: int(row.get("relative_step", -1)),
        )
    holding_rows = [row for row in rows if row.get("treatment_state") == "HOLDING"]
    taper_rows = [row for row in rows if row.get("treatment_state") == "TAPERING"]
    handoffs = [
        event
        for row in rows
        for event in row.get("member_changes", ())
        if event.get("event") == "handoff"
    ]
    return {
        "holding_partial_lane_count": sum(
            float((row.get("lane_coverage") or {}).get("coverage_fraction", 0.0)) < 1.0
            for row in holding_rows
        ),
        "holding_span_violation_count": sum(
            (row.get("current_member_span_m") is not None)
            and float(row["current_member_span_m"]) > 50.0
            for row in holding_rows
        ),
        "handoff_count": len(handoffs),
        "handoffs": handoffs,
        "taper_step_count": len(taper_rows),
        "taper_blend_fractions": [row.get("taper_blend_fraction") for row in taper_rows],
        "taper_reasons": sorted(
            {str(row.get("taper_reason")) for row in taper_rows if row.get("taper_reason")}
        ),
    }


def run_case(case: tuple[int, int], output: Path, sumo_binary: str, force: bool) -> dict[str, Any]:
    seed, decision = case
    path = output / "cases" / f"seed{seed}_step{decision}.json"
    if path.exists() and not force:
        return {"status": "skipped", "path": str(path)}
    stored = _stored_case(V5_2_ROOT, seed, decision)
    scan = _scan(seed, decision)
    if not scan.get("admitted"):
        raise RuntimeError(f"frozen Stage-1 case {seed}/{decision} is not admitted")
    control = _replay(seed, decision, treatment=False, sumo_binary=sumo_binary)
    treatment = _replay(seed, decision, treatment=True, sumo_binary=sumo_binary)
    definition = OutcomeDefinition()
    control_outcome = summarize_branch(
        control, decision_step=decision, outcome_steps=OUTCOME_STEPS, definition=definition
    )
    treatment_outcome = summarize_branch(
        treatment, decision_step=decision, outcome_steps=OUTCOME_STEPS, definition=definition
    )
    audit = _audit_contract(
        control,
        treatment,
        decision_step=decision,
        treatment_steps=TREATMENT_STEPS,
        step_length_s=1.0,
    )
    # Preserve controller rows explicitly because they carry v5.3 phase diagnostics.
    audit["controller_audits"] = [dict(row) for row in treatment.controller_audits]
    features = features_from_trace(
        control,
        decision_step=decision,
        definition=definition,
        dose_context={
            "proposed_wall_speed_mps": overlay_overrides()["wall_params"]["wall_speed"],
            "release_boundary_m": overlay_overrides()["compact_release_boundary_m"],
            "selected_member_ids": (scan.get("formation") or {}).get("controlled_cav_ids", ()),
        },
    )
    features.update(preflight_features(scan))
    row = {
        "schema": "idm_wall_v5_3_stage1_case/1",
        "seed": seed,
        "decision_step": decision,
        "demand_mainline_vph": 2700,
        "cav_penetration": 0.30,
        "hdv_model": "IDM",
        "features": features,
        "timing_scan": scan,
        "prefix_identity": _prefix_identity(control, treatment, decision),
        "treatment_dose": audit["treatment_dose"],
        "treatment_contract": audit,
        "arms": {
            "0": {"outcomes": control_outcome, "wall_engaged": False},
            "1": {"outcomes": treatment_outcome, "wall_engaged": True},
        },
    }
    row["safety_service_oracle"] = classify_case(row, HeadroomMargins())
    old_wall = stored["arms"]["1"]["outcomes"]
    old_effects = stored["safety_service_oracle"]["effects"]
    new_effects = row["safety_service_oracle"]["effects"]
    row["comparison_to_v5_2"] = {
        "v5_2_compact_dose_qualified": bool(
            dose_qualified(stored)
        ),
        "v5_2_safety_admissible": bool(stored["safety_service_oracle"]["safety_admissible"]),
        "v5_2_meaningful_benefit": bool(stored["safety_service_oracle"]["meaningful_benefit"]),
        "v5_2_hard_brake_excess_vehicle_steps": old_effects["hard_brake_excess_vehicle_steps"],
        "v5_3_hard_brake_excess_vehicle_steps": new_effects["hard_brake_excess_vehicle_steps"],
        "absolute_treatment_hard_brake_vehicle_steps_not_worse": (
            int(treatment_outcome["hard_brake_vehicle_steps"])
            <= int(old_wall["hard_brake_vehicle_steps"])
        ),
        "absolute_treatment_low_ttc_steps_not_worse": (
            int(treatment_outcome["low_ttc_exposure_steps"])
            <= int(old_wall["low_ttc_exposure_steps"])
        ),
        "no_new_collision_or_teleport": (
            int(treatment_outcome["collision_unique_vehicle_count"])
            <= int(old_wall["collision_unique_vehicle_count"])
            and int(treatment_outcome["starting_teleport_unique_vehicle_count"])
            <= int(old_wall["starting_teleport_unique_vehicle_count"])
        ),
    }
    row["mechanics"] = _mechanics(audit)
    if case in REPEAT_CASES:
        repeat = _replay(seed, decision, treatment=True, sumo_binary=sumo_binary)
        row["treatment_repeat"] = compare_traces(treatment, repeat)
    atomic_write_json(path, row)
    return {"status": "completed", "path": str(path)}


def summarize(output: Path) -> dict[str, Any]:
    rows = [
        json.loads((output / "cases" / f"seed{seed}_step{step}.json").read_text(encoding="utf-8"))
        for seed, step in STAGE1_CASES
    ]
    # v5.2 predates the explicit compact-composite Boolean.  Derive it from
    # the frozen dose components so the comparison report does not turn a
    # missing field into a scientific dose failure.
    for row in rows:
        stored = _stored_case(V5_2_ROOT, int(row["seed"]), int(row["decision_step"]))
        row["comparison_to_v5_2"]["v5_2_compact_dose_qualified"] = dose_qualified(stored)
    prior_beneficial = [
        row for row in rows if row["comparison_to_v5_2"]["v5_2_meaningful_benefit"]
    ]
    gates = {
        "all_prefix_identity": all(row["prefix_identity"]["passed"] for row in rows),
        "repeat_identity": all(
            row.get("treatment_repeat", {}).get("passed", False)
            for row in rows
            if (int(row["seed"]), int(row["decision_step"])) in REPEAT_CASES
        ),
        "no_partial_lane_hold": all(
            row["mechanics"]["holding_partial_lane_count"] == 0 for row in rows
        ),
        "release_returns_to_damping": all(
            row["treatment_contract"]["checks"]["release_returns_to_damping"] for row in rows
        ),
        "safety_not_worse_than_v5_2": all(
            row["comparison_to_v5_2"]["absolute_treatment_hard_brake_vehicle_steps_not_worse"]
            and row["comparison_to_v5_2"]["absolute_treatment_low_ttc_steps_not_worse"]
            and row["comparison_to_v5_2"]["no_new_collision_or_teleport"]
            for row in rows
        ),
        "two_of_three_prior_beneficial_remain_safe_beneficial": sum(
            bool(row["safety_service_oracle"]["safety_admissible"])
            and bool(row["safety_service_oracle"]["meaningful_benefit"])
            for row in prior_beneficial
        )
        >= 2,
    }
    summary = {
        "schema": "idm_wall_v5_3_stage1_summary/1",
        "wall_profile": profile_manifest(),
        "feature_contract": feature_contract_manifest(),
        "case_count": len(rows),
        "gates": gates,
        "passed": all(gates.values()),
        "phase_b_authorized": False,
        "cases": [
            {
                "seed": row["seed"],
                "decision_step": row["decision_step"],
                "dose_qualified": row["treatment_dose"].get(
                    "compact_composite_dose_qualified", False
                ),
                "safety_admissible": row["safety_service_oracle"]["safety_admissible"],
                "meaningful_benefit": row["safety_service_oracle"]["meaningful_benefit"],
                "mechanics": row["mechanics"],
                "comparison_to_v5_2": row["comparison_to_v5_2"],
            }
            for row in rows
        ],
    }
    atomic_write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sumo-binary", default=str(DEFAULT_SUMO_EXE))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cases").mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        args.output / "manifest.json",
        {
            "schema": "idm_wall_v5_3_stage1_manifest/1",
            "wall_profile": profile_manifest(),
            "cases": [list(case) for case in STAGE1_CASES],
            "source_v5_2_root": str(V5_2_ROOT),
            "source_treatment_blind_scan": str(SCAN_ROOT),
            "new_seed_generation": False,
            "scientific_role": "bounded_mechanism_replay_not_pilot",
        },
    )
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(run_case, case, args.output, args.sumo_binary, args.force): case
            for case in STAGE1_CASES
        }
        for future in as_completed(futures):
            print(json.dumps(future.result(), sort_keys=True), flush=True)
    print(json.dumps(summarize(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
