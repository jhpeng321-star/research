"""Run the frozen v5.7 geometry baseline and four paired development cases."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
from typing import Any, Mapping

from counterfactual.outcomes import OutcomeDefinition, step_metrics, summarize_branch
from counterfactual.twin_replay import FrozenPulseSchedule, ReplayTrace, run_replay
from experiments import REPO_ROOT
from experiments.counterfactual.verify_counterfactual_escalation import _audit_contract, _prefix_identity
from experiments.idm.run_idm_wall_pilot import DEFAULT_SUMO_EXE, supervisor_overrides
from experiments.idm.wall_v5_7_extended_zone_contract import (
    EVENT_HORIZON_S, HOLD_S, POST_RELEASE_OBSERVATION_S, PROFILE_ID, TAPER_S,
    TOTAL_TREATMENT_S, WALL_BAND_HI_M, WALL_BAND_LO_M, overlay_overrides,
    profile_manifest,
)
from experiments.io import atomic_write_json
from policies.idm_analytical import build_frozen_idm_analytical_policy


SEEDS = (4401, 4402, 4404, 4405)
SCENARIO_ROOT = REPO_ROOT / "scenarios" / "idm_v5_7_extended_zone"
CONFIG = "osm.dense.idm_m2700r400.v5_7.sumocfg"
OUTPUT = REPO_ROOT / "runs" / "probe" / "idm_v5_7_extended_zone_p30_s4401_4402_4404_4405"
SCOUT_HORIZON_S = 1200
DEFINITION = OutcomeDefinition(queue_speed_threshold_mps=15.0, congestion_queue_count=5, congestion_speed_threshold_mps=15.0, sustain_steps=60)


def _damping(_step: int) -> int:
    return 0


def _replay(seed: int, decision: int, *, treatment: bool, sumo_binary: str, horizon: int) -> ReplayTrace:
    schedule = FrozenPulseSchedule(decision, int(TOTAL_TREATMENT_S)) if treatment else _damping
    return run_replay(
        project_root=SCENARIO_ROOT, sumo_binary=sumo_binary, sumo_config=CONFIG,
        seed=seed, decision_step=decision, horizon_steps=horizon,
        treatment_steps=int(TOTAL_TREATMENT_S), cav_penetration=0.30,
        sumo_step_length=0.5,
        supervisor_overrides={**supervisor_overrides(), "use_health_obs": False, "use_dv_obs": False, "feasibility_mask": False, "action_masking": False},
        base_policy_factory=build_frozen_idm_analytical_policy,
        intervention_schedule=schedule, overlay_overrides=overlay_overrides(),
        controller_metadata={"scientific_role": "idm_v5_7_extended_zone_development", "arm": "damping_plus_v5_7_wall" if treatment else "damping_control", "wall_profile": profile_manifest(), "event_aligned_horizon_s": EVENT_HORIZON_S},
    )


def _scout(seed: int, sumo_binary: str) -> tuple[ReplayTrace, dict[str, Any] | None]:
    trace = _replay(seed, 0, treatment=False, sumo_binary=sumo_binary, horizon=SCOUT_HORIZON_S)
    for row in trace.controller_audits:
        formation = row.get("formation_preview") or {}
        safety = row.get("activation_safety_preview") or {}
        if bool(formation.get("feasible")) and bool(safety.get("safe")):
            return trace, {"decision_relative_step": int(row["relative_step"]), "formation": formation, "activation_safety": safety}
    return trace, None


def _runs(trace: ReplayTrace, decision: int) -> dict[str, Any]:
    rows = [step_metrics(snapshot, DEFINITION) for snapshot in trace.snapshots if snapshot.relative_step >= decision]
    flags = [bool(row["queue_count"] >= 5 and row["pre_merge_mean_speed_mps"] is not None and float(row["pre_merge_mean_speed_mps"]) <= 15.0) for row in rows]
    runs: list[int] = []; current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        if not flag and current: runs.append(current)
    if current: runs.append(current)
    predicate_s = sum(flags)
    return {"predicate_duration_s": predicate_s, "longest_run_s": max(runs, default=0), "sustained_duration_s": sum(run for run in runs if run >= 60), "sustained_event": max(runs, default=0) >= 60}


def _dose(trace: ReplayTrace, decision: int) -> dict[str, Any]:
    rows = [dict(row) for row in trace.controller_audits if decision < int(row.get("relative_step", -1)) <= decision + int(TOTAL_TREATMENT_S)]
    active = [row for row in rows if row.get("wall_override_ids")]
    phases = {phase: [row for row in active if row.get("treatment_state") == phase] for phase in ("ASSEMBLING", "HOLDING", "TAPERING")}
    coverage = [float((row.get("lane_coverage") or {}).get("coverage_fraction", 0.0)) for row in active]
    spans = {phase: max((float(row.get("current_member_span_m") or 0.0) for row in items), default=0.0) for phase, items in phases.items()}
    spreads = [float(row["member_speed_spread_mps"]) for row in active if row.get("member_speed_spread_mps") is not None]
    targets = [float(row["adaptive_common_target_mps"]) for row in active if row.get("adaptive_common_target_mps") is not None]
    control_deltas = [item for row in active for item in (row.get("control_deltas") or {}).values()]
    return {"requested_assembly_max_s": 10, "requested_hold_s": HOLD_S, "requested_taper_s": TAPER_S, "requested_window_s": TOTAL_TREATMENT_S, "realized_assembly_s": len(phases["ASSEMBLING"]), "realized_hold_s": len(phases["HOLDING"]), "realized_taper_s": len(phases["TAPERING"]), "realized_active_s": len(active), "realized_duration_fraction": len(active) / TOTAL_TREATMENT_S, "controlled_cav_seconds": sum(len(row.get("wall_override_ids", ())) for row in active), "minimum_lane_coverage_fraction": min(coverage, default=0.0), "maximum_span_by_phase_m": spans, "maximum_speed_spread_mps": max(spreads, default=None), "common_targets_mps": targets, "member_changes": [change for row in rows for change in row.get("member_changes", ())], "abort_reasons": [row.get("abort_reason") for row in rows if row.get("abort_reason")], "command_count": len(control_deltas), "maximum_wall_authority_mps2": max((float(item.get("wall_a_max_dec_mps2") or 0.0) for item in control_deltas), default=0.0), "wall_speed_tracking_error_summary": "per-member projected target retained in controller audit", "exact_damping_after_release": bool(next((not row.get("wall_override_ids") for row in trace.controller_audits if int(row.get("relative_step", -1)) == decision + int(TOTAL_TREATMENT_S) + 1), False))}


def _effect(control: ReplayTrace, treatment: ReplayTrace, decision: int) -> dict[str, Any]:
    left, right = _runs(control, decision), _runs(treatment, decision)
    predicate = left["predicate_duration_s"] - right["predicate_duration_s"]
    longest = left["longest_run_s"] - right["longest_run_s"]
    return {"control": left, "treatment": right, "predicate_reduction_s": predicate, "longest_run_reduction_s": longest, "material_benefit": predicate >= 10 and longest >= 10, "induces_sustained_congestion": not left["sustained_event"] and right["sustained_event"], "replay_noise_deadband_s": 0.0}


def _baseline_case(seed: int, output: Path, sumo_binary: str, force: bool) -> dict[str, Any]:
    path = output / "baseline" / f"seed{seed}.json"
    if path.exists() and not force:
        return {"status": "skipped", "seed": seed}
    scout, decision_data = _scout(seed, sumo_binary)
    if decision_data is None:
        record = {"schema": "idm_v5_7_baseline_case/1", "seed": seed, "headroom": False, "reason": "no_treatment_blind_safe_distance_band_formation", "scout_trace": scout.summary()}
    else:
        decision = int(decision_data["decision_relative_step"])
        control = _replay(seed, decision, treatment=False, sumo_binary=sumo_binary, horizon=EVENT_HORIZON_S)
        runs = _runs(control, decision)
        outcomes = summarize_branch(control, decision_step=decision, outcome_steps=EVENT_HORIZON_S, definition=DEFINITION)
        record = {"schema": "idm_v5_7_baseline_case/1", "seed": seed, "headroom": bool(runs["predicate_duration_s"] > 0), "decision": decision_data, "control_trace": control.summary(), "congestion": runs, "outcomes": outcomes}
    atomic_write_json(path, record)
    return {"status": "baseline_completed", "seed": seed, "headroom": record["headroom"]}


def _baseline_summary(output: Path) -> dict[str, Any]:
    rows = [json.loads((output / "baseline" / f"seed{seed}.json").read_text()) for seed in SEEDS]
    result = {"schema": "idm_v5_7_baseline_geometry_summary/1", "profile": profile_manifest(), "seeds": list(SEEDS), "all_treatment_blind_formations": all(row.get("decision") for row in rows), "all_merge_congestion_headroom": all(bool(row.get("headroom")) for row in rows), "wall_outcomes_run": False, "rows": rows}
    result["baseline_gate_passed"] = bool(result["all_treatment_blind_formations"] and result["all_merge_congestion_headroom"])
    return result


def _case(seed: int, output: Path, sumo_binary: str, force: bool) -> dict[str, Any]:
    path = output / "cases" / f"seed{seed}.json"
    if path.exists() and not force:
        return {"status": "skipped", "seed": seed}
    started = time.time()
    baseline = json.loads((output / "baseline" / f"seed{seed}.json").read_text())
    if not bool(baseline.get("headroom")):
        raise RuntimeError(f"v5.7 baseline headroom gate failed for seed {seed}; wall outcome is forbidden")
    decision_data = dict(baseline["decision"])
    decision = int(decision_data["decision_relative_step"])
    control = _replay(seed, decision, treatment=False, sumo_binary=sumo_binary, horizon=EVENT_HORIZON_S)
    treatment = _replay(seed, decision, treatment=True, sumo_binary=sumo_binary, horizon=EVENT_HORIZON_S)
    audit = _audit_contract(control, treatment, decision_step=decision, treatment_steps=int(TOTAL_TREATMENT_S), step_length_s=1.0)
    prefix = _prefix_identity(control, treatment, decision)
    control_outcome = summarize_branch(control, decision_step=decision, outcome_steps=EVENT_HORIZON_S, definition=DEFINITION)
    treatment_outcome = summarize_branch(treatment, decision_step=decision, outcome_steps=EVENT_HORIZON_S, definition=DEFINITION)
    dose = _dose(treatment, decision); effect = _effect(control, treatment, decision)
    record = {"schema": "idm_v5_7_case/1", "seed": seed, "baseline_headroom": True, "baseline": baseline, "decision": decision_data, "contract": profile_manifest(), "scenario_manifest": json.loads((SCENARIO_ROOT / "manifest.json").read_text()), "prefix_identity": prefix, "treatment_contract": audit, "treatment_dose": dose, "arms": {"0": {"outcomes": control_outcome, "trace": control.summary()}, "1": {"outcomes": treatment_outcome, "trace": treatment.summary()}}, "physical_effect": effect, "wall_outcome_run": True, "wall_clock_seconds": time.time() - started}
    atomic_write_json(path, record)
    return {"status": "completed", "seed": seed, "material_benefit": effect["material_benefit"], "seconds": record["wall_clock_seconds"]}


def _summary(output: Path) -> dict[str, Any]:
    rows = [json.loads((output / "cases" / f"seed{seed}.json").read_text()) for seed in SEEDS]
    completed = [row for row in rows if row.get("wall_outcome_run")]
    gates = {"all_baseline_headroom": len(completed) == 4, "predecision_identity": all(row["prefix_identity"]["passed"] for row in completed), "three_safe_complete": sum(row["treatment_dose"]["realized_duration_fraction"] >= .8 and row["treatment_dose"]["minimum_lane_coverage_fraction"] >= .8 and row["treatment_dose"]["maximum_span_by_phase_m"]["HOLDING"] <= 50 for row in completed) >= 3, "two_material_benefits": sum(row["physical_effect"]["material_benefit"] for row in completed) >= 2, "one_neutral_or_nonbeneficial": any(not row["physical_effect"]["material_benefit"] for row in completed), "all_low_authority": all(row["treatment_dose"]["maximum_wall_authority_mps2"] <= .3 + 1e-9 for row in completed), "all_release_to_damping": all(row["treatment_dose"]["exact_damping_after_release"] for row in completed)}
    return {"schema": "idm_v5_7_summary/1", "profile": profile_manifest(), "seeds": list(SEEDS), "gates": gates, "four_case_passed": all(gates.values()), "predictor_training_authorized": False, "fresh_seeds_opened": False, "branch_count": len(completed) * 2 + 4, "per_seed": rows}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=OUTPUT); parser.add_argument("--sumo-binary", default=str(DEFAULT_SUMO_EXE)); parser.add_argument("--workers", type=int, default=4); parser.add_argument("--force", action="store_true"); parser.add_argument("--baseline-only", action="store_true"); parser.add_argument("--run-wall", action="store_true"); args = parser.parse_args()
    if args.baseline_only == args.run_wall:
        raise SystemExit("choose exactly one of --baseline-only or --run-wall")
    output = args.output.resolve(); (output / "baseline").mkdir(parents=True, exist_ok=True); (output / "cases").mkdir(parents=True, exist_ok=True)
    worker_count = min(max(1, args.workers), len(SEEDS))
    if args.baseline_only:
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            for future in as_completed([pool.submit(_baseline_case, seed, output, args.sumo_binary, args.force) for seed in SEEDS]): print(json.dumps(future.result(), sort_keys=True), flush=True)
        atomic_write_json(output / "baseline_summary.json", _baseline_summary(output)); return 0
    baseline = json.loads((output / "baseline_summary.json").read_text())
    if not bool(baseline.get("baseline_gate_passed")):
        raise RuntimeError("v5.7 baseline geometry/headroom gate failed; wall pilot is forbidden")
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        for future in as_completed([pool.submit(_case, seed, output, args.sumo_binary, args.force) for seed in SEEDS]): print(json.dumps(future.result(), sort_keys=True), flush=True)
    atomic_write_json(output / "summary.json", _summary(output)); return 0


if __name__ == "__main__": raise SystemExit(main())
