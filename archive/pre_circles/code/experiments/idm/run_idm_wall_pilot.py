"""Bounded IDM damping-plus-wall treatment verification.

This runner transfers the already-audited adaptive compact wall v5.1 onto the
frozen analytical IDM damping controller. It is resumable and intentionally
runs only the six inspected actuator-development seeds until treatment dose,
safety, and outcome diversity pass.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
from typing import Any

from counterfactual.headroom import (
    HeadroomMargins,
    TreatmentRepairCriteria,
    classify_case,
    summarize_treatment_repair,
)
from counterfactual.outcomes import (
    OutcomeDefinition,
    summarize_branch,
    summarize_predecision_congestion,
)
from counterfactual.twin_replay import (
    FrozenPulseSchedule,
    compare_traces,
    run_replay,
)
from experiments.io import atomic_write_json as _atomic_json
from idm.contract import (
    ANALYTICAL_PILOT_SEEDS,
    CAV_PENETRATION,
    CONTRACT_VERSION,
    SCENARIO_ROOT,
    contract_manifest,
    transition_scenario_config,
    validate_seed_role,
)
from policies.idm_analytical import (
    IDMAnalyticalDampingPolicy,
    build_frozen_idm_analytical_policy,
    frozen_idm_analytical_manifest,
)
from experiments.counterfactual.verify_counterfactual_escalation import (
    _audit_contract,
    _prefix_identity,
)


from experiments import REPO_ROOT

ROOT = REPO_ROOT
DEFAULT_OUTPUT = ROOT / "runs" / "eval" / "idm_wall_pilot_v1"
DEFAULT_SUMO_EXE = Path(r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe")
DEMAND_VPH = 2700
DECISION_STEP = 720
OUTCOME_STEPS = 260
TREATMENT_STEPS = 20
STEP_LENGTH_S = 1.0


def overlay_overrides() -> dict[str, Any]:
    """Exact adaptive compact wall v5.1 treatment definition."""
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
    }


def supervisor_overrides() -> dict[str, Any]:
    """Frozen environment contract shared with the analytical IDM pilot."""
    return {
        "simulation_step": STEP_LENGTH_S,
        "cav_placement": "scatter",
        "cav_force_normal_type": True,
        "assign_hdv_types": False,
        "cav_warmup_steps": 300,
        "cav_empty_extend_steps": 600,
        "cav_empty_retries": 2,
        "use_v2_obs": True,
        "centralized_critic": False,
        "action_masking": False,
        "reward_mode": "shepherd",
        "control_mode": "shepherd",
    }


def _base_policy() -> IDMAnalyticalDampingPolicy:
    return build_frozen_idm_analytical_policy()


def _run_branch(case: dict[str, Any], *, treatment: bool):
    decision = int(case["decision_step"])
    treatment_steps = int(case["treatment_steps"])
    schedule = FrozenPulseSchedule(decision, treatment_steps) if treatment else (lambda _step: 0)
    return run_replay(
        project_root=SCENARIO_ROOT,
        sumo_binary=str(case["sumo_binary"]),
        sumo_config=transition_scenario_config(DEMAND_VPH).name,
        seed=int(case["seed"]),
        decision_step=decision,
        horizon_steps=int(case["outcome_steps"]),
        treatment_steps=treatment_steps,
        cav_penetration=CAV_PENETRATION,
        sumo_step_length=0.5,
        supervisor_overrides=supervisor_overrides(),
        digits=6,
        base_policy_factory=_base_policy,
        intervention_schedule=schedule,
        overlay_overrides=overlay_overrides(),
        controller_metadata={
            "base_controller": frozen_idm_analytical_manifest()["controller_id"],
            "base_control_arm": frozen_idm_analytical_manifest(),
            "overlay_profile": "adaptive_compact_wall_v5_1_transfer",
        },
    )


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    output = Path(case["output"]).resolve()
    path = output / "cases" / f"seed{int(case['seed'])}.json"
    if path.exists() and not bool(case.get("force")):
        return {"status": "skipped", "seed": int(case["seed"]), "path": str(path)}

    started = time.time()
    control = _run_branch(case, treatment=False)
    treatment = _run_branch(case, treatment=True)
    definition = OutcomeDefinition()
    control_outcome = summarize_branch(
        control,
        decision_step=int(case["decision_step"]),
        outcome_steps=int(case["outcome_steps"]),
        definition=definition,
    )
    treatment_outcome = summarize_branch(
        treatment,
        decision_step=int(case["decision_step"]),
        outcome_steps=int(case["outcome_steps"]),
        definition=definition,
    )
    audit = _audit_contract(
        control,
        treatment,
        decision_step=int(case["decision_step"]),
        treatment_steps=int(case["treatment_steps"]),
        step_length_s=STEP_LENGTH_S,
    )
    row: dict[str, Any] = {
        "schema": "idm_damping_wall_case/1",
        "seed": int(case["seed"]),
        "demand_mainline_vph": DEMAND_VPH,
        "demand_ramp_vph": 400,
        "decision_step": int(case["decision_step"]),
        "decision_sim_time_s": float(control.snapshots[int(case["decision_step"])].sim_time),
        "eligible": True,
        "predecision_supervisor": {"admitted": True, "profile": "none"},
        "predecision_congestion": summarize_predecision_congestion(
            control,
            decision_step=int(case["decision_step"]),
            definition=definition,
        ),
        "prefix_identity": _prefix_identity(control, treatment, int(case["decision_step"])),
        "treatment_contract": audit,
        "treatment_dose": audit["treatment_dose"],
        "arms": {
            "0": {"outcomes": control_outcome, "wall_engaged": False},
            "1": {
                "outcomes": treatment_outcome,
                "wall_engaged": bool(audit["wall_override_steps"]),
            },
        },
        "repeat_checks": None,
        "wall_clock_seconds": time.time() - started,
    }
    row["safety_service_oracle"] = classify_case(row, HeadroomMargins())

    if bool(case.get("repeat_check")):
        control_repeat = _run_branch(case, treatment=False)
        treatment_repeat = _run_branch(case, treatment=True)
        row["repeat_checks"] = {
            "damping_vs_damping": compare_traces(control, control_repeat),
            "treatment_vs_treatment": compare_traces(treatment, treatment_repeat),
        }
        row["wall_clock_seconds"] = time.time() - started

    _atomic_json(path, row)
    return {"status": "completed", "seed": int(case["seed"]), "path": str(path)}


def build_summary(output: Path, seeds: list[int]) -> dict[str, Any]:
    rows = []
    missing = []
    for seed in seeds:
        path = output / "cases" / f"seed{seed}.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            missing.append(str(path))
    repeat_rows = [row for row in rows if row.get("repeat_checks")]
    exact_replay_pass = bool(repeat_rows) and all(
        row["repeat_checks"][name]["passed"]
        for row in repeat_rows
        for name in ("damping_vs_damping", "treatment_vs_treatment")
    )
    criteria = TreatmentRepairCriteria(require_collapse_prevention_diversity=True)
    gate = summarize_treatment_repair(
        rows,
        criteria=criteria,
        effects_exceed_repeat_noise=exact_replay_pass,
    )
    causal_invariants = (
        "prefix_base_action_identity",
        "first_request_same_base_actions",
        "override_only_during_treatment",
        "out_of_band_damping_preserved",
        "wall_ids_removed_from_env_action_path",
        "release_returns_to_damping",
        "runtime_formation_loss_is_not_an_abort",
    )
    identity_pass = bool(rows) and all(
        row["prefix_identity"]["passed"]
        and all(row["treatment_contract"]["checks"][name] for name in causal_invariants)
        for row in rows
    )
    summary = {
        "schema": "idm_damping_wall_pilot_summary/1",
        "contract_version": CONTRACT_VERSION,
        "contract": contract_manifest(),
        "development_seed_exclusion": {
            "seeds": seeds,
            "reason": "inspected IDM actuator and wall-treatment development",
            "exclude_from_model_and_locked_evaluation": True,
        },
        "record_count": len(rows),
        "missing": missing,
        "exact_repeat_seed_count": len(repeat_rows),
        "exact_replay_pass": exact_replay_pass,
        "identity_and_overlay_contract_pass": identity_pass,
        "treatment_repair_gate": gate,
        "predictor_authorized": bool(
            not missing and exact_replay_pass and identity_pass and gate["passed"]
        ),
    }
    _atomic_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(ANALYTICAL_PILOT_SEEDS))
    parser.add_argument("--decision-step", type=int, default=DECISION_STEP)
    parser.add_argument("--outcome-steps", type=int, default=OUTCOME_STEPS)
    parser.add_argument("--treatment-steps", type=int, default=TREATMENT_STEPS)
    parser.add_argument("--repeat-seeds", type=int, nargs="+", default=[4101, 4102])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--sumo-binary", default=str(DEFAULT_SUMO_EXE if DEFAULT_SUMO_EXE.exists() else "sumo")
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    seeds = validate_seed_role("analytical_pilot", args.seeds)
    if args.decision_step < 0 or args.outcome_steps <= args.treatment_steps:
        raise ValueError("decision must be nonnegative and outcome must exceed treatment")
    if args.treatment_steps != TREATMENT_STEPS:
        raise ValueError("primary v5.1 transfer freezes a 20-second treatment")
    if any(seed not in seeds for seed in args.repeat_seeds):
        raise ValueError("repeat seeds must be part of the requested cohort")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "idm_damping_wall_manifest/1",
        "created_unix_s": time.time(),
        "command": {
            "seeds": seeds,
            "decision_step": args.decision_step,
            "outcome_steps": args.outcome_steps,
            "treatment_steps": args.treatment_steps,
            "repeat_seeds": args.repeat_seeds,
            "workers": args.workers,
            "sumo_binary": args.sumo_binary,
            "output": str(output),
        },
        "scientific_role": "treatment_development_excluded_from_evaluation",
        "base_control_arm": frozen_idm_analytical_manifest(),
        "environment": supervisor_overrides(),
        "overlay_overrides": overlay_overrides(),
        "headroom_margins": HeadroomMargins().to_dict(),
        "treatment_repair_criteria": TreatmentRepairCriteria().to_dict(),
        "outcome_definition": OutcomeDefinition().to_dict(),
    }
    _atomic_json(output / "manifest.json", manifest)
    cases = [
        {
            "seed": seed,
            "decision_step": args.decision_step,
            "outcome_steps": args.outcome_steps,
            "treatment_steps": args.treatment_steps,
            "repeat_check": seed in set(args.repeat_seeds),
            "sumo_binary": args.sumo_binary,
            "output": str(output),
            "force": args.force,
        }
        for seed in seeds
    ]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_case, case): case for case in cases}
        for future in as_completed(futures):
            result = future.result()
            print(json.dumps(result, sort_keys=True), flush=True)
    print(json.dumps(build_summary(output, seeds), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
