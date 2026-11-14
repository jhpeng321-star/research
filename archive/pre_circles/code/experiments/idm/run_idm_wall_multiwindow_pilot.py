"""Prospective paired IDM damping-plus-wall v5.2 safety/service pilot."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
from typing import Any, Callable

from counterfactual.idm_predictor_contract import (
    CONTRACT_VERSION as PREDICTOR_CONTRACT_VERSION,
    feature_contract_manifest,
    preflight_features,
    validate_decision_features,
)
from counterfactual.headroom import (
    HeadroomMargins,
    classify_case,
    summarize_treatment_repair,
)
from counterfactual.pilot import features_from_trace
from counterfactual.outcomes import (
    OutcomeDefinition,
    summarize_branch,
    summarize_predecision_congestion,
)
from counterfactual.twin_replay import FrozenPulseSchedule, compare_traces, run_replay
from experiments.io import atomic_write_json as _atomic_json
from idm.contract import (
    CONTRACT_VERSION,
    SCENARIO_ROOT,
    WALL_V5_2_PILOT_SEEDS,
    contract_manifest,
    transition_scenario_config,
)
from experiments.idm.run_idm_wall_pilot import (
    DEFAULT_SUMO_EXE,
    STEP_LENGTH_S,
    supervisor_overrides,
)
from experiments.idm.wall_v5_2_contract import (
    DECISION_STEPS,
    OUTCOME_STEPS,
    PROFILE_ID,
    TOTAL_CONTROLLER_STEPS,
    TREATMENT_STEPS,
    assert_no_prior_seed_artifacts,
    load_selected_feasibility_scan,
    overlay_overrides,
    profile_manifest,
    require_accepted_damping_factory,
    select_repeat_cases,
    treatment_repair_criteria,
    validate_pilot_seeds,
)
from experiments.counterfactual.verify_counterfactual_escalation import (
    _audit_contract,
    _prefix_identity,
)
from policies.idm_analytical import frozen_idm_analytical_manifest


from experiments import REPO_ROOT

ROOT = REPO_ROOT
DEFAULT_OUTPUT = ROOT / "runs" / "eval" / "idm_wall_v5_2_prospective_p30_s4301_4312"
DEFAULT_FEASIBILITY_SCAN = (
    ROOT / "runs" / "probe" / "idm_compact_wall_feasibility_p10_p30_s4301_4312"
)


def _replay(
    *,
    seed: int,
    decision: int,
    total_steps: int,
    sumo_binary: str,
    treatment: bool,
    cav_penetration: float,
    base_policy_factory: Callable[[], Any] | None = None,
):
    base_policy_factory = require_accepted_damping_factory(base_policy_factory)
    schedule = FrozenPulseSchedule(decision, TREATMENT_STEPS) if treatment else (lambda _step: 0)
    return run_replay(
        project_root=SCENARIO_ROOT,
        sumo_binary=sumo_binary,
        sumo_config=transition_scenario_config(2700).name,
        seed=seed,
        decision_step=decision,
        horizon_steps=total_steps - decision,
        treatment_steps=TREATMENT_STEPS,
        cav_penetration=float(cav_penetration),
        sumo_step_length=0.5,
        supervisor_overrides=supervisor_overrides(),
        digits=6,
        base_policy_factory=base_policy_factory,
        intervention_schedule=schedule,
        overlay_overrides=overlay_overrides(),
        controller_metadata={
            "base_controller": frozen_idm_analytical_manifest()["controller_id"],
            "base_control_arm": frozen_idm_analytical_manifest(),
            "overlay_profile": PROFILE_ID,
            "prospective_safety_service_pilot": True,
        },
    )


def _scan_by_decision(scan_root: Path, seed: int) -> dict[int, dict[str, Any]]:
    path = scan_root / "seeds" / f"seed{seed}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
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


def _decision_features(
    control: Any,
    *,
    decision: int,
    definition: OutcomeDefinition,
    scan: dict[str, Any],
) -> dict[str, Any]:
    """Build inputs exclusively from the shared predecision control prefix."""
    formation = scan.get("formation") or {}
    features = features_from_trace(
        control,
        decision_step=decision,
        definition=definition,
        dose_context={
            "proposed_wall_speed_mps": overlay_overrides()["wall_params"]["wall_speed"],
            "release_boundary_m": overlay_overrides()["compact_release_boundary_m"],
            "selected_member_ids": formation.get("controlled_cav_ids", ()),
        },
    )
    features.update(preflight_features(scan))
    return features


def run_seed(case: dict[str, Any]) -> dict[str, Any]:
    from policies.idm_analytical import build_frozen_idm_analytical_policy

    seed = int(case["seed"])
    output = Path(case["output"])
    result_path = output / "seeds" / f"seed{seed}.json"
    if result_path.exists() and not bool(case.get("force")):
        return {"status": "skipped", "seed": seed, "path": str(result_path)}
    decisions = [int(x) for x in case["decision_steps"]]
    cav_penetration = float(case["cav_penetration"])
    scan = _scan_by_decision(Path(case["feasibility_case_root"]), seed)
    missing_scan = [step for step in decisions if step not in scan]
    if missing_scan:
        raise ValueError(f"seed {seed} lacks timing scan steps {missing_scan}")

    started = time.time()
    definition = OutcomeDefinition()
    max_decision = max(decisions)
    total_steps = max_decision + int(case["outcome_steps"])
    control = _replay(
        seed=seed,
        decision=max_decision,
        total_steps=total_steps,
        sumo_binary=str(case["sumo_binary"]),
        treatment=False,
        cav_penetration=cav_penetration,
        base_policy_factory=build_frozen_idm_analytical_policy,
    )
    rows = []
    treatment_traces: dict[int, Any] = {}
    for decision in decisions:
        control_outcome = summarize_branch(
            control,
            decision_step=decision,
            outcome_steps=int(case["outcome_steps"]),
            definition=definition,
        )
        predecision = summarize_predecision_congestion(
            control, decision_step=decision, definition=definition
        )
        admitted = bool(scan[decision]["admitted"])
        if admitted:
            treatment = _replay(
                seed=seed,
                decision=decision,
                total_steps=total_steps,
                sumo_binary=str(case["sumo_binary"]),
                treatment=True,
                cav_penetration=cav_penetration,
                base_policy_factory=build_frozen_idm_analytical_policy,
            )
            treatment_traces[decision] = treatment
            treatment_outcome = summarize_branch(
                treatment,
                decision_step=decision,
                outcome_steps=int(case["outcome_steps"]),
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
            treatment_contract: Any = audit
            wall_engaged = bool(audit["wall_override_steps"])
        else:
            treatment_outcome = control_outcome
            prefix = {"passed": True, "not_replayed_reason": "deterministic_feasibility_veto"}
            dose = _infeasible_dose(scan[decision])
            treatment_contract = {
                "passed": True,
                "not_replayed_reason": "deterministic_feasibility_veto",
                "predecision_scan": scan[decision],
            }
            wall_engaged = False
        features = _decision_features(
            control,
            decision=decision,
            definition=definition,
            scan=scan[decision],
        )
        row: dict[str, Any] = {
            "schema": "idm_damping_wall_multiwindow_case/2",
            "case_id": (
                f"idm-m2700-p{round(cav_penetration * 100):02d}-"
                f"seed{seed}-step{decision}"
            ),
            "seed": seed,
            "demand_mainline_vph": 2700,
            "demand_ramp_vph": 400,
            "cav_penetration": cav_penetration,
            "hdv_model": "IDM",
            "decision_step": decision,
            "decision_sim_time_s": float(control.snapshots[decision].sim_time),
            "eligible": True,
            "feature_contract_version": PREDICTOR_CONTRACT_VERSION,
            "features": features,
            "predecision_supervisor": {
                "admitted": admitted,
                "profile": "deterministic_geometry_and_activation_safety",
            },
            "predecision_congestion": predecision,
            "timing_scan": scan[decision],
            "prefix_identity": prefix,
            "treatment_contract": treatment_contract,
            "treatment_dose": dose,
            "arms": {
                "0": {"outcomes": control_outcome, "wall_engaged": False},
                "1": {"outcomes": treatment_outcome, "wall_engaged": wall_engaged},
            },
        }
        validate_decision_features(row)
        row["safety_service_oracle"] = classify_case(row, HeadroomMargins())
        rows.append(row)

    repeat_checks = None
    repeat_decision = case.get("repeat_decision")
    if repeat_decision is not None:
        repeat_decision = int(repeat_decision)
        treatment = treatment_traces.get(repeat_decision)
        if treatment is None:
            raise ValueError(f"repeat case seed {seed} step {repeat_decision} was not admitted")
        control_repeat = _replay(
            seed=seed,
            decision=max_decision,
            total_steps=total_steps,
            sumo_binary=str(case["sumo_binary"]),
            treatment=False,
            cav_penetration=cav_penetration,
            base_policy_factory=build_frozen_idm_analytical_policy,
        )
        treatment_repeat = _replay(
            seed=seed,
            decision=repeat_decision,
            total_steps=total_steps,
            sumo_binary=str(case["sumo_binary"]),
            treatment=True,
            cav_penetration=cav_penetration,
            base_policy_factory=build_frozen_idm_analytical_policy,
        )
        repeat_checks = {
            "decision_step": repeat_decision,
            "damping_vs_damping": compare_traces(control, control_repeat),
            "engaged_treatment_vs_treatment": compare_traces(treatment, treatment_repeat),
        }

    payload = {
        "schema": "idm_damping_wall_multiwindow_seed/2",
        "seed": seed,
        "decision_steps": decisions,
        "cases": rows,
        "repeat_checks": repeat_checks,
        "wall_clock_seconds": time.time() - started,
    }
    _atomic_json(result_path, payload)
    return {"status": "completed", "seed": seed, "path": str(result_path)}


def build_summary(output: Path, seeds: list[int]) -> dict[str, Any]:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    selected_penetration = float(manifest["cav_penetration"])
    experiment_contract = contract_manifest()
    experiment_contract["cav_penetration"] = selected_penetration
    experiment_contract["cav_penetration_source"] = (
        "treatment_blind_compact_wall_feasibility_selection"
    )
    seed_rows = []
    missing = []
    for seed in seeds:
        path = output / "seeds" / f"seed{seed}.json"
        if path.exists():
            seed_rows.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            missing.append(str(path))
    rows = [row for payload in seed_rows for row in payload["cases"]]
    repeat_rows = [payload for payload in seed_rows if payload.get("repeat_checks")]
    exact_replay_pass = len(repeat_rows) >= 2 and all(
        payload["repeat_checks"][name]["passed"]
        for payload in repeat_rows
        for name in ("damping_vs_damping", "engaged_treatment_vs_treatment")
    )
    engaged_effects = [
        row["safety_service_oracle"]["effects"]
        for row in rows
        if bool(row["arms"]["1"].get("wall_engaged", False))
    ]
    max_abs_effects = {
        key: max((abs(float(effect[key])) for effect in engaged_effects), default=0.0)
        for key in (
            "delay_benefit_s",
            "merge_discharge_benefit",
            "hard_brake_excess_vehicle_steps",
            "low_ttc_excess_steps",
        )
    }
    effect_noise_evidence = {
        "exact_repeat_trace_noise_is_zero": exact_replay_pass,
        "repeat_trace_comparison": "exact_digest_identity_at_6_decimal_state_precision",
        "max_absolute_observed_treatment_effects": max_abs_effects,
        "observed_effect_exceeds_zero_replay_noise": bool(
            exact_replay_pass and any(value > 0.0 for value in max_abs_effects.values())
        ),
    }
    causal_pass = all(
        row["prefix_identity"]["passed"]
        and (
            not row["treatment_dose"]["activation_feasible"]
            or all(
                row["treatment_contract"]["checks"][name]
                for name in (
                    "prefix_base_action_identity",
                    "first_request_same_base_actions",
                    "override_only_during_treatment",
                    "out_of_band_damping_preserved",
                    "wall_ids_removed_from_env_action_path",
                    "release_returns_to_damping",
                    "runtime_formation_loss_is_not_an_abort",
                    "wall_engaged_at_least_once",
                )
            )
        )
        for row in rows
    )
    gate = summarize_treatment_repair(
        rows,
        criteria=treatment_repair_criteria(),
        effects_exceed_repeat_noise=effect_noise_evidence[
            "observed_effect_exceeds_zero_replay_noise"
        ],
    )
    summary = {
        "schema": "idm_damping_wall_multiwindow_summary/3",
        "contract_version": CONTRACT_VERSION,
        "contract": experiment_contract,
        "cav_penetration": selected_penetration,
        "base_control_arm": frozen_idm_analytical_manifest(),
        "predictor_feature_contract": feature_contract_manifest(),
        "development_seed_exclusion": {
            "seeds": seeds,
            "exclude_from_model_and_locked_evaluation": True,
            "reason": "prospective wall timing and treatment development",
        },
        "seed_count": len(seed_rows),
        "case_count": len(rows),
        "missing": missing,
        "exact_engaged_repeat_seed_count": len(repeat_rows),
        "exact_replay_pass": exact_replay_pass,
        "effect_vs_repeat_noise_evidence": effect_noise_evidence,
        "causal_contract_pass": causal_pass,
        "treatment_repair_gate": gate,
        "predictor_authorized": bool(
            not missing and exact_replay_pass and causal_pass and gate["passed"]
        ),
    }
    _atomic_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(WALL_V5_2_PILOT_SEEDS))
    parser.add_argument("--decision-steps", type=int, nargs="+", default=list(DECISION_STEPS))
    parser.add_argument("--outcome-steps", type=int, default=OUTCOME_STEPS)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--sumo-binary", default=str(DEFAULT_SUMO_EXE if DEFAULT_SUMO_EXE.exists() else "sumo")
    )
    parser.add_argument(
        "--feasibility-scan", type=Path, default=DEFAULT_FEASIBILITY_SCAN
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    seeds = validate_pilot_seeds(args.seeds)
    decisions = sorted(set(int(step) for step in args.decision_steps))
    if tuple(decisions) != DECISION_STEPS:
        raise ValueError(f"primary development windows are frozen at {DECISION_STEPS}")
    if args.outcome_steps != OUTCOME_STEPS:
        raise ValueError(f"primary outcome horizon is frozen at {OUTCOME_STEPS}")
    if max(decisions) + args.outcome_steps != TOTAL_CONTROLLER_STEPS:
        raise ValueError("v5.2 requires a shared 960-step terminal horizon")
    feasibility = load_selected_feasibility_scan(args.feasibility_scan)
    feasibility_root = Path(feasibility["scan_root"])
    feasibility_case_root = Path(feasibility["case_root"])
    cav_penetration = float(feasibility["selected_cav_penetration"])
    repeat_cases = select_repeat_cases(feasibility_case_root, seeds, decisions)
    output = args.output.resolve()
    artifact_audit = assert_no_prior_seed_artifacts(
        ROOT / "runs",
        seeds,
        allowed_roots=(feasibility_root, output),
    )
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        output / "manifest.json",
        {
            "schema": "idm_damping_wall_multiwindow_manifest/3",
            "created_unix_s": time.time(),
            "scientific_role": "prospective_v5_2_safety_service_treatment_pilot",
            "seeds": seeds,
            "decision_steps": decisions,
            "outcome_steps": args.outcome_steps,
            "treatment_steps": TREATMENT_STEPS,
            "repeat_cases": repeat_cases,
            "treatment_blind_feasibility_selection": feasibility,
            "cav_penetration": cav_penetration,
            "base_control_arm": frozen_idm_analytical_manifest(),
            "predictor_feature_contract": feature_contract_manifest(),
            "overlay_overrides": overlay_overrides(),
            "wall_profile": profile_manifest(),
            "prior_seed_artifact_audit": artifact_audit,
            "environment": supervisor_overrides(),
            "headroom_margins": HeadroomMargins().to_dict(),
            "treatment_repair_criteria": treatment_repair_criteria().to_dict(),
            "outcome_definition": OutcomeDefinition().to_dict(),
        },
    )
    cases = [
        {
            "seed": seed,
            "decision_steps": decisions,
            "outcome_steps": args.outcome_steps,
            "repeat_decision": repeat_cases.get(seed),
            "sumo_binary": args.sumo_binary,
            "feasibility_case_root": str(feasibility_case_root),
            "cav_penetration": cav_penetration,
            "output": str(output),
            "force": args.force,
        }
        for seed in seeds
    ]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(run_seed, case) for case in cases]
        for future in as_completed(futures):
            print(json.dumps(future.result(), sort_keys=True), flush=True)
    print(json.dumps(build_summary(output, seeds), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
