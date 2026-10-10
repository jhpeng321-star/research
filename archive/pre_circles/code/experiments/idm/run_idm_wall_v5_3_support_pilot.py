"""Run the frozen prospective v5.3 selective-gate support pilot."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
from typing import Any

from counterfactual.actionability import dose_qualified
from counterfactual.headroom import HeadroomMargins, classify_case
from counterfactual.idm_predictor_contract import (
    CONTRACT_VERSION as FEATURE_CONTRACT_VERSION,
    feature_contract_manifest,
    preflight_features,
    validate_decision_features,
)
from counterfactual.outcomes import (
    OutcomeDefinition,
    summarize_branch,
    summarize_predecision_congestion,
)
from counterfactual.pilot import features_from_trace
from counterfactual.twin_replay import FrozenPulseSchedule, compare_traces, run_replay
from experiments import REPO_ROOT
from experiments.counterfactual.verify_counterfactual_escalation import (
    _audit_contract,
    _prefix_identity,
)
from experiments.idm.run_idm_wall_pilot import DEFAULT_SUMO_EXE, STEP_LENGTH_S, supervisor_overrides
from experiments.idm.run_idm_wall_v5_3_stage1 import _mechanics
from experiments.idm.wall_v5_3_support_contract import (
    DECISION_STEPS,
    DEMAND_MAINLINE_VPH,
    DEMAND_RAMP_VPH,
    OUTCOME_STEPS,
    SUPPORT_SEEDS,
    TOTAL_CONTROLLER_STEPS,
    TREATMENT_STEPS,
    assert_no_prior_seed_artifacts,
    load_selected_feasibility_scan,
    overlay_overrides,
    profile_manifest,
    require_accepted_damping_factory,
    select_repeat_cases,
    support_manifest,
    validate_support_seeds,
)
from experiments.io import atomic_write_json
from idm.contract import SCENARIO_ROOT, transition_scenario_config
from policies.idm_analytical import (
    build_frozen_idm_analytical_policy,
    frozen_idm_analytical_manifest,
)


DEFAULT_OUTPUT = REPO_ROOT / "runs" / "eval" / "idm_wall_v5_3_support_pilot_s4313_4332"
DEFAULT_SCAN = REPO_ROOT / "runs" / "probe" / "idm_wall_v5_3_feasibility_p10_p30_s4313_4332"


def _replay(*, seed: int, decision: int, treatment: bool, penetration: float, sumo_binary: str):
    factory = require_accepted_damping_factory(build_frozen_idm_analytical_policy)
    schedule = FrozenPulseSchedule(decision, TREATMENT_STEPS) if treatment else (lambda _step: 0)
    return run_replay(
        project_root=SCENARIO_ROOT,
        sumo_binary=sumo_binary,
        sumo_config=transition_scenario_config(DEMAND_MAINLINE_VPH).name,
        seed=seed,
        decision_step=decision,
        horizon_steps=TOTAL_CONTROLLER_STEPS - decision,
        treatment_steps=TREATMENT_STEPS,
        cav_penetration=penetration,
        sumo_step_length=0.5,
        supervisor_overrides=supervisor_overrides(),
        digits=6,
        base_policy_factory=factory,
        intervention_schedule=schedule,
        overlay_overrides=overlay_overrides(),
        controller_metadata={
            "scientific_role": "prospective_v5_3_selective_gate_support",
            "wall_profile": profile_manifest(),
        },
    )


def _scan_rows(case_root: Path, seed: int) -> dict[int, dict[str, Any]]:
    payload = json.loads(
        (Path(case_root) / "seeds" / f"seed{seed}.json").read_text(encoding="utf-8")
    )
    return {int(row["decision_step"]): row for row in payload["results"]}


def _infeasible_dose(scan: dict[str, Any]) -> dict[str, Any]:
    return {
        "activation_feasible": False,
        "requested_duration_steps": TREATMENT_STEPS,
        "requested_duration_s": TREATMENT_STEPS * STEP_LENGTH_S,
        "realized_override_duration_steps": 0,
        "realized_override_duration_s": 0.0,
        "realized_duration_fraction": 0.0,
        "lane_coverage_fraction": 0.0,
        "holding_fraction_of_override": 0.0,
        "assembly_completed": False,
        "abort_count": 0,
        "wall_geometry": {
            "geometry_mode": "compact_moving",
            "configured_span_cap_m": 50.0,
            "fraction_within_configured_span": None,
            "sample_count": 0,
        },
        "predecision_scan": scan,
    }


def _features(control: Any, decision: int, scan: dict[str, Any]) -> dict[str, Any]:
    formation = scan.get("formation") or {}
    result = features_from_trace(
        control,
        decision_step=decision,
        definition=OutcomeDefinition(),
        dose_context={
            "proposed_wall_speed_mps": overlay_overrides()["wall_params"]["wall_speed"],
            "release_boundary_m": overlay_overrides()["compact_release_boundary_m"],
            "selected_member_ids": formation.get("controlled_cav_ids", ()),
        },
    )
    result.update(preflight_features(scan))
    return result


def run_seed(case: dict[str, Any]) -> dict[str, Any]:
    seed = int(case["seed"])
    output = Path(case["output"])
    result_path = output / "seeds" / f"seed{seed}.json"
    if result_path.exists() and not bool(case.get("force")):
        return {"status": "skipped", "seed": seed, "path": str(result_path)}
    penetration = float(case["penetration"])
    scan = _scan_rows(Path(case["case_root"]), seed)
    if tuple(sorted(scan)) != DECISION_STEPS:
        raise ValueError(f"seed {seed} scan decision grid is not frozen")
    definition = OutcomeDefinition()
    started = time.time()
    control = _replay(
        seed=seed,
        decision=max(DECISION_STEPS),
        treatment=False,
        penetration=penetration,
        sumo_binary=str(case["sumo_binary"]),
    )
    rows: list[dict[str, Any]] = []
    treatment_traces: dict[int, Any] = {}
    for decision in DECISION_STEPS:
        control_outcome = summarize_branch(
            control, decision_step=decision, outcome_steps=OUTCOME_STEPS, definition=definition
        )
        admitted = bool(scan[decision]["admitted"])
        if admitted:
            treatment = _replay(
                seed=seed,
                decision=decision,
                treatment=True,
                penetration=penetration,
                sumo_binary=str(case["sumo_binary"]),
            )
            treatment_traces[decision] = treatment
            treatment_outcome = summarize_branch(
                treatment,
                decision_step=decision,
                outcome_steps=OUTCOME_STEPS,
                definition=definition,
            )
            audit = _audit_contract(
                control,
                treatment,
                decision_step=decision,
                treatment_steps=TREATMENT_STEPS,
                step_length_s=STEP_LENGTH_S,
            )
            prefix = _prefix_identity(control, treatment, decision)
            dose = audit["treatment_dose"]
            contract: dict[str, Any] = audit
            engaged = bool(audit["wall_override_steps"])
            mechanics = _mechanics(audit)
        else:
            treatment_outcome = control_outcome
            prefix = {"passed": True, "not_replayed_reason": "deterministic_feasibility_veto"}
            dose = _infeasible_dose(scan[decision])
            contract = {
                "passed": True,
                "not_replayed_reason": "deterministic_feasibility_veto",
                "predecision_scan": scan[decision],
            }
            engaged = False
            mechanics = {
                "handoff_count": 0,
                "holding_partial_lane_count": 0,
                "holding_span_violation_count": 0,
                "taper_step_count": 0,
                "taper_blend_fractions": [],
                "taper_reasons": [],
            }
        row: dict[str, Any] = {
            "schema": "idm_wall_v5_3_support_case/1",
            "case_id": f"idm-v53-m2700-p{round(penetration * 100):02d}-seed{seed}-step{decision}",
            "seed": seed,
            "decision_step": decision,
            "decision_sim_time_s": float(control.snapshots[decision].sim_time),
            "demand_mainline_vph": DEMAND_MAINLINE_VPH,
            "demand_ramp_vph": DEMAND_RAMP_VPH,
            "cav_penetration": penetration,
            "hdv_model": "IDM",
            "feature_contract_version": FEATURE_CONTRACT_VERSION,
            "features": _features(control, decision, scan[decision]),
            "predecision_supervisor": {
                "admitted": admitted,
                "profile": "v5.3_deterministic_geometry_authority_activation_safety",
            },
            "predecision_congestion": summarize_predecision_congestion(
                control, decision_step=decision, definition=definition
            ),
            "timing_scan": scan[decision],
            "prefix_identity": prefix,
            "treatment_contract": contract,
            "treatment_mechanics": mechanics,
            "treatment_dose": dose,
            "arms": {
                "0": {"outcomes": control_outcome, "wall_engaged": False},
                "1": {"outcomes": treatment_outcome, "wall_engaged": engaged},
            },
        }
        validate_decision_features(row)
        row["safety_service_oracle"] = classify_case(row, HeadroomMargins())
        rows.append(row)

    repeat = None
    repeat_decision = case.get("repeat_decision")
    if repeat_decision is not None:
        repeat_decision = int(repeat_decision)
        treatment = treatment_traces.get(repeat_decision)
        if treatment is None:
            raise RuntimeError("declared repeat decision was not admitted")
        control_repeat = _replay(
            seed=seed,
            decision=max(DECISION_STEPS),
            treatment=False,
            penetration=penetration,
            sumo_binary=str(case["sumo_binary"]),
        )
        treatment_repeat = _replay(
            seed=seed,
            decision=repeat_decision,
            treatment=True,
            penetration=penetration,
            sumo_binary=str(case["sumo_binary"]),
        )
        repeat = {
            "decision_step": repeat_decision,
            "damping_vs_damping": compare_traces(control, control_repeat),
            "treatment_vs_treatment": compare_traces(treatment, treatment_repeat),
        }
    payload = {
        "schema": "idm_wall_v5_3_support_seed/1",
        "seed": seed,
        "decision_steps": list(DECISION_STEPS),
        "cases": rows,
        "repeat_checks": repeat,
        "wall_clock_seconds": time.time() - started,
    }
    atomic_write_json(result_path, payload)
    return {"status": "completed", "seed": seed, "path": str(result_path)}


def _seed_count(rows: list[dict[str, Any]]) -> int:
    return len({int(row["seed"]) for row in rows})


def build_summary(output: Path) -> dict[str, Any]:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    seed_payloads = [
        json.loads((output / "seeds" / f"seed{seed}.json").read_text(encoding="utf-8"))
        for seed in SUPPORT_SEEDS
    ]
    rows = [row for payload in seed_payloads for row in payload["cases"]]
    admitted = [row for row in rows if row["predecision_supervisor"]["admitted"]]
    refused = [row for row in rows if not row["predecision_supervisor"]["admitted"]]
    qualified = [row for row in admitted if dose_qualified(row)]
    failed_dose = [row for row in admitted if not dose_qualified(row)]
    safe = [row for row in qualified if row["safety_service_oracle"]["safety_admissible"]]
    unsafe = [row for row in qualified if not row["safety_service_oracle"]["safety_admissible"]]
    y1 = [
        row
        for row in qualified
        if row["safety_service_oracle"]["safety_admissible"]
        and row["safety_service_oracle"]["traffic_noninferior"]
        and row["safety_service_oracle"]["meaningful_benefit"]
    ]
    y0 = [row for row in qualified if row not in y1]
    repeats = [payload["repeat_checks"] for payload in seed_payloads if payload.get("repeat_checks")]
    exact_replay = len(repeats) == 2 and all(
        repeat[name]["passed"]
        for repeat in repeats
        for name in ("damping_vs_damping", "treatment_vs_treatment")
    )
    causal = all(
        row["prefix_identity"]["passed"]
        and (
            not row["predecision_supervisor"]["admitted"]
            or all(
                row["treatment_contract"]["checks"][name]
                for name in (
                    "prefix_base_action_identity",
                    "first_request_same_base_actions",
                    "override_only_during_treatment",
                    "out_of_band_damping_preserved",
                    "wall_ids_removed_from_env_action_path",
                    "release_returns_to_damping",
                    "taper_is_monotone_and_reaches_zero",
                    "wall_engaged_at_least_once",
                )
            )
        )
        for row in rows
    )
    no_partial = all(
        int(row["treatment_mechanics"]["holding_partial_lane_count"]) == 0
        for row in admitted
    )
    compactness_not_silently_held = all(
        int(row["treatment_mechanics"]["holding_span_violation_count"]) <= 1
        for row in admitted
    )
    no_collision_teleport_excess = all(
        int(row["safety_service_oracle"]["effects"]["collision_excess_vehicle_count"]) <= 0
        and int(row["safety_service_oracle"]["effects"]["teleport_excess_vehicle_count"]) <= 0
        for row in admitted
    )
    effects_exceed_noise = bool(
        exact_replay
        and any(
            any(abs(float(row["safety_service_oracle"]["effects"][key])) > 0.0 for key in (
                "delay_benefit_s", "merge_discharge_benefit",
                "hard_brake_excess_vehicle_steps", "low_ttc_excess_steps",
            ))
            for row in admitted
        )
    )
    support = {
        "raw_case_count": len(rows),
        "admitted_case_count": len(admitted),
        "admitted_seed_count": _seed_count(admitted),
        "refused_case_count": len(refused),
        "refused_seed_count": _seed_count(refused),
        "dose_qualified_case_count": len(qualified),
        "dose_qualified_seed_count": _seed_count(qualified),
        "failed_attempted_dose_case_count": len(failed_dose),
        "failed_attempted_dose_seed_count": _seed_count(failed_dose),
        "safe_case_count": len(safe),
        "safe_seed_count": _seed_count(safe),
        "unsafe_case_count": len(unsafe),
        "unsafe_seed_count": _seed_count(unsafe),
        "joint_y1_case_count": len(y1),
        "joint_y1_seed_count": _seed_count(y1),
        "joint_y0_case_count": len(y0),
        "joint_y0_seed_count": _seed_count(y0),
        "raw_attempt_safety_fraction": len(safe) / len(qualified) if qualified else 0.0,
        "dose_success_fraction": len(qualified) / len(admitted) if admitted else 0.0,
    }
    gates = {
        "complete_20_seed_4_window_grid": len(rows) == len(SUPPORT_SEEDS) * len(DECISION_STEPS),
        "unique_case_ids": len({str(row["case_id"]) for row in rows}) == len(rows),
        "exact_replay": exact_replay,
        "causal_identity_and_damping_preservation": causal,
        "no_partial_lane_hold": no_partial,
        "compactness_not_silently_held": compactness_not_silently_held,
        "zero_excess_collision_or_teleport": no_collision_teleport_excess,
        "effects_exceed_exact_replay_noise": effects_exceed_noise,
        "dose_success_at_least_80_percent": support["dose_success_fraction"] >= 0.80,
        "dose_qualified_spans_at_least_6_seeds": support["dose_qualified_seed_count"] >= 6,
        "gate_f_positive_spans_at_least_3_seeds": support["dose_qualified_seed_count"] >= 3,
        "gate_f_negative_or_refusal_spans_at_least_3_seeds": len(
            {int(row["seed"]) for row in refused + failed_dose}
        ) >= 3,
        "safe_spans_at_least_3_seeds": support["safe_seed_count"] >= 3,
        "unsafe_spans_at_least_3_seeds": support["unsafe_seed_count"] >= 3,
        "joint_y1_spans_at_least_3_seeds": support["joint_y1_seed_count"] >= 3,
        "joint_y0_spans_at_least_3_seeds": support["joint_y0_seed_count"] >= 3,
    }
    summary = {
        "schema": "idm_wall_v5_3_support_summary/1",
        "scientific_role": "prospective_treatment_and_label_support_not_predictor_validation",
        "manifest": manifest,
        "feature_contract": feature_contract_manifest(),
        "support": support,
        "gates": gates,
        "passed": all(gates.values()),
        "predictor_development_generation_authorized": all(gates.values()),
        "predictor_fitting_authorized": False,
        "final_test_authorized": False,
    }
    atomic_write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sumo-binary", default=str(DEFAULT_SUMO_EXE))
    parser.add_argument("--feasibility-scan", type=Path, default=DEFAULT_SCAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    seeds = validate_support_seeds(SUPPORT_SEEDS)
    feasibility = load_selected_feasibility_scan(args.feasibility_scan)
    output = args.output.resolve()
    scan_root = Path(feasibility["scan_root"])
    audit = assert_no_prior_seed_artifacts(
        REPO_ROOT / "runs", seeds, allowed_roots=(scan_root, output)
    )
    repeat_cases = select_repeat_cases(Path(feasibility["case_root"]))
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        output / "manifest.json",
        {
            "schema": "idm_wall_v5_3_support_manifest/1",
            "created_unix_s": time.time(),
            "scientific_role": "prospective_v5_3_selective_gate_support",
            "support_contract": support_manifest(),
            "seeds": list(seeds),
            "decision_steps": list(DECISION_STEPS),
            "outcome_steps": OUTCOME_STEPS,
            "treatment_steps": TREATMENT_STEPS,
            "repeat_cases": repeat_cases,
            "treatment_blind_feasibility_selection": feasibility,
            "cav_penetration": float(feasibility["selected_cav_penetration"]),
            "base_control_arm": frozen_idm_analytical_manifest(),
            "wall_profile": profile_manifest(),
            "feature_contract": feature_contract_manifest(),
            "headroom_margins": HeadroomMargins().to_dict(),
            "outcome_definition": OutcomeDefinition().to_dict(),
            "prior_seed_artifact_audit": audit,
        },
    )
    cases = [
        {
            "seed": seed,
            "repeat_decision": repeat_cases.get(seed),
            "penetration": float(feasibility["selected_cav_penetration"]),
            "case_root": feasibility["case_root"],
            "sumo_binary": args.sumo_binary,
            "output": str(output),
            "force": args.force,
        }
        for seed in seeds
    ]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(run_seed, case) for case in cases]
        for future in as_completed(futures):
            print(json.dumps(future.result(), sort_keys=True), flush=True)
    print(json.dumps(build_summary(output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
