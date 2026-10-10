"""Gated v5.7.1 original-corridor expanded-zone development experiment.

Gate A writes damping-only repeats. Gate B consumes only those treatment-blind
rows at t=880. Gate C is unavailable unless both machine-readable gates pass.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

from counterfactual.outcomes import OutcomeDefinition, step_metrics, summarize_branch
from counterfactual.twin_replay import FrozenPulseSchedule, ReplayTrace, compare_traces, run_replay
from experiments import REPO_ROOT
from experiments.counterfactual.verify_counterfactual_escalation import _audit_contract, _prefix_identity
from experiments.idm.run_idm_wall_pilot import DEFAULT_SUMO_EXE, supervisor_overrides
from experiments.idm.wall_v5_7_1_original_corridor_contract import *
from experiments.io import atomic_write_json
from idm.contract import SCENARIO_ROOT, transition_scenario_config
from policies.idm_analytical import build_frozen_idm_analytical_policy


OUTPUT = REPO_ROOT / "runs" / "probe" / "idm_v5_7_1_original_corridor_expanded_zone_p30_s4401_4402_4404_4405"
DEFINITION = OutcomeDefinition(queue_speed_threshold_mps=15.0, congestion_queue_count=5, congestion_speed_threshold_mps=15.0, sustain_steps=60)


def _damping(_step: int) -> int: return 0


def _congestion_runs(flags: Iterable[bool]) -> list[int]:
    """Lengths in controller seconds of every contiguous predicate interval."""
    result: list[int] = []; current = 0
    for flag in flags:
        if flag:
            current += 1
        elif current:
            result.append(current); current = 0
    if current: result.append(current)
    return result


def _runs(trace: ReplayTrace, decision: int) -> dict[str, Any]:
    metrics = [step_metrics(snapshot, DEFINITION) for snapshot in trace.snapshots if snapshot.relative_step >= decision]
    flags = [bool(row["queue_count"] >= 5 and row["pre_merge_mean_speed_mps"] is not None and float(row["pre_merge_mean_speed_mps"]) <= 15.0) for row in metrics]
    intervals = _congestion_runs(flags)
    return {"controller_step_s": 1.0, "predicate_duration_s": sum(flags), "interval_durations_s": intervals, "longest_run_s": max(intervals, default=0), "sustained_duration_s": sum(value for value in intervals if value >= 60), "sustained_event": max(intervals, default=0) >= 60}


def _decision_step(trace: ReplayTrace) -> int:
    matches = [snapshot.relative_step for snapshot in trace.snapshots if abs(float(snapshot.sim_time) - DECISION_ABSOLUTE_S) <= 1e-9]
    if len(matches) != 1: raise RuntimeError("t=880 is absent or ambiguous in original-corridor trace")
    return int(matches[0])


def _replay(seed: int, decision: int, treatment: bool, sumo_binary: str, horizon: int) -> ReplayTrace:
    schedule = FrozenPulseSchedule(decision, int(TOTAL_TREATMENT_S)) if treatment else _damping
    return run_replay(project_root=SCENARIO_ROOT, sumo_binary=sumo_binary, sumo_config=transition_scenario_config(DEMAND_MAINLINE_VPH).name, seed=seed, decision_step=decision, horizon_steps=horizon, treatment_steps=int(TOTAL_TREATMENT_S), cav_penetration=CAV_PENETRATION, sumo_step_length=.5, supervisor_overrides={**supervisor_overrides(), "use_health_obs": False, "use_dv_obs": False, "feasibility_mask": False, "action_masking": False}, base_policy_factory=build_frozen_idm_analytical_policy, intervention_schedule=schedule, overlay_overrides=overlay_overrides(), controller_metadata={"scientific_role": PROFILE_ID, "arm": "damping_plus_wall" if treatment else "damping_control", "decision_absolute_s": DECISION_ABSOLUTE_S, "outcome_horizon_s": OUTCOME_HORIZON_S, "derived_network_used": False, "profile": profile_manifest()})


def _locate_decision(seed: int, sumo_binary: str) -> int:
    scout = _replay(seed, 0, False, sumo_binary, 1000)
    return _decision_step(scout)


def _baseline_progress_path(output: Path, seed: int) -> Path:
    return output / "baseline_progress" / f"seed{seed}.json"


def _write_baseline_progress(
    output: Path,
    seed: int,
    phase: str,
    *,
    decision: int | None = None,
    elapsed_s: float | None = None,
) -> None:
    """Atomically expose a treatment-blind checkpoint for an interruptible Gate A.

    The decision is discovered exclusively from a damping-only scout.  Reusing a
    recorded decision after an interrupted process therefore cannot leak wall
    information or change the scientific branch definition; it merely avoids
    repeating the longest replay in this gate.
    """
    record: dict[str, Any] = {
        "schema": "idm_v5_7_1_baseline_progress/1",
        "seed": seed,
        "phase": phase,
        "updated_unix_s": time.time(),
        "treatment_blind": True,
    }
    if decision is not None:
        record["decision_relative_step"] = int(decision)
        record["decision_absolute_s"] = DECISION_ABSOLUTE_S
    if elapsed_s is not None:
        record["phase_elapsed_s"] = float(elapsed_s)
    atomic_write_json(_baseline_progress_path(output, seed), record)


def _complete_horizon(trace: ReplayTrace, decision: int) -> bool:
    return not trace.terminated_early and trace.snapshots[-1].relative_step == decision + OUTCOME_HORIZON_S


def _formation_at(trace: ReplayTrace, decision: int) -> dict[str, Any]:
    row = next((dict(row) for row in trace.controller_audits if int(row.get("relative_step", -1)) == decision), None)
    if row is None: raise RuntimeError("decision audit unavailable")
    formation = dict(row.get("formation_preview") or {})
    safety = dict(row.get("activation_safety_preview") or {})
    return {"decision_relative_step": decision, "decision_absolute_s": DECISION_ABSOLUTE_S, "formation": formation, "activation_safety": safety, "feasible": bool(formation.get("feasible")) and bool(safety.get("safe")), "reason": ("admitted" if bool(formation.get("feasible")) and bool(safety.get("safe")) else f"formation:{formation.get('reason', 'unavailable')};safety:{safety.get('reason', 'unavailable')}")}


def _baseline_case(seed: int, output: Path, sumo_binary: str, force: bool) -> dict[str, Any]:
    path = output / "baseline" / f"seed{seed}.json"
    if path.exists() and not force: return {"status": "skipped", "seed": seed}
    progress_path = _baseline_progress_path(output, seed)
    progress = json.loads(progress_path.read_text()) if progress_path.exists() and not force else {}
    cached_decision = progress.get("decision_relative_step")
    if cached_decision is None:
        scout_started = time.time()
        _write_baseline_progress(output, seed, "scout_running")
        decision = _locate_decision(seed, sumo_binary)
        _write_baseline_progress(output, seed, "scout_complete", decision=decision, elapsed_s=time.time() - scout_started)
    else:
        decision = int(cached_decision)
        _write_baseline_progress(output, seed, "scout_reused_after_restart", decision=decision)
    control_started = time.time()
    _write_baseline_progress(output, seed, "control_replay_running", decision=decision)
    control = _replay(seed, decision, False, sumo_binary, OUTCOME_HORIZON_S)
    _write_baseline_progress(output, seed, "control_replay_complete", decision=decision, elapsed_s=time.time() - control_started)
    repeat_started = time.time()
    _write_baseline_progress(output, seed, "repeat_replay_running", decision=decision)
    repeat = _replay(seed, decision, False, sumo_binary, OUTCOME_HORIZON_S)
    _write_baseline_progress(output, seed, "repeat_replay_complete", decision=decision, elapsed_s=time.time() - repeat_started)
    congestion = _runs(control, decision)
    record = {"schema": "idm_v5_7_1_baseline_case/1", "seed": seed, "decision_relative_step": decision, "decision_absolute_s": DECISION_ABSOLUTE_S, "profile": profile_manifest(), "original_scenario": str(transition_scenario_config(DEMAND_MAINLINE_VPH)), "derived_network_used": False, "damping_replay_identity": compare_traces(control, repeat), "complete_horizon": _complete_horizon(control, decision) and _complete_horizon(repeat, decision), "formation": _formation_at(control, decision), "congestion": congestion, "outcomes": summarize_branch(control, decision_step=decision, outcome_steps=OUTCOME_HORIZON_S, definition=DEFINITION), "repeat_outcomes": summarize_branch(repeat, decision_step=decision, outcome_steps=OUTCOME_HORIZON_S, definition=DEFINITION), "control_trace": control.summary(), "repeat_trace": repeat.summary()}
    atomic_write_json(path, record)
    _write_baseline_progress(output, seed, "completed", decision=decision)
    return {"status": "completed", "seed": seed, "headroom": bool(congestion["predicate_duration_s"]), "replay": record["damping_replay_identity"]["passed"]}


def _baseline_summary(output: Path) -> dict[str, Any]:
    rows = [json.loads((output / "baseline" / f"seed{seed}.json").read_text()) for seed in SEEDS]
    gates = {"original_corridor_only": all(not row["derived_network_used"] for row in rows), "frozen_t880": all(row["decision_absolute_s"] == DECISION_ABSOLUTE_S for row in rows), "damping_replay_identity": all(row["damping_replay_identity"]["passed"] for row in rows), "complete_horizon": all(row["complete_horizon"] for row in rows), "at_least_three_headroom": sum(row["congestion"]["predicate_duration_s"] > 0 for row in rows) >= 3}
    return {"schema": "idm_v5_7_1_baseline_summary/1", "profile": profile_manifest(), "seeds": list(SEEDS), "gates": gates, "gate_a_passed": all(gates.values()), "wall_outcomes_run": False, "rows": rows}


def _dose(trace: ReplayTrace, decision: int) -> dict[str, Any]:
    rows = [dict(row) for row in trace.controller_audits if decision < int(row.get("relative_step", -1)) <= decision + int(TOTAL_TREATMENT_S)]
    active = [row for row in rows if row.get("wall_override_ids")]
    by_phase = {name: [row for row in active if row.get("treatment_state") == name] for name in ("ASSEMBLING", "HOLDING", "TAPERING")}
    coverage = [float((row.get("lane_coverage") or {}).get("coverage_fraction", 0.0)) for row in active]
    deltas = [value for row in active for value in (row.get("control_deltas") or {}).values()]
    handoffs = [event for row in rows for event in row.get("member_changes", ()) if event.get("event") == "handoff"]
    return {"requested": {"assembly_min_steps": ASSEMBLY_MIN_STEPS, "assembly_max_s": ASSEMBLY_MAX_S, "hold_s": HOLD_S, "taper_s": TAPER_S, "window_s": TOTAL_TREATMENT_S}, "realized": {"assembly_s": len(by_phase["ASSEMBLING"]), "hold_s": len(by_phase["HOLDING"]), "taper_s": len(by_phase["TAPERING"]), "active_s": len(active), "controlled_cav_seconds": sum(len(row.get("wall_override_ids", ())) for row in active)}, "duration_fraction": len(active) / TOTAL_TREATMENT_S, "lane_coverage": {"minimum": min(coverage, default=0.0), "mean": sum(coverage) / len(coverage) if coverage else 0.0}, "maximum_span_by_phase_m": {name: max((float(row.get("current_member_span_m") or 0.0) for row in phase), default=0.0) for name, phase in by_phase.items()}, "maximum_speed_spread_mps": max((float(row.get("member_speed_spread_mps")) for row in active if row.get("member_speed_spread_mps") is not None), default=None), "common_targets_mps": [row.get("adaptive_common_target_mps") for row in active], "member_changes": [event for row in rows for event in row.get("member_changes", ())], "handoff_count": len(handoffs), "per_lane_handoff_counts": {lane: max((int((row.get("handoff_counts_by_lane") or {}).get(lane, 0)) for row in rows), default=0) for lane in {event.get("lane_id") for event in handoffs}}, "abort_reasons": [row.get("abort_reason") for row in rows if row.get("abort_reason")], "release_reasons": [row.get("taper_reason") for row in rows if row.get("taper_reason")], "maximum_authority_mps2": max((float(item.get("wall_a_max_dec_mps2") or 0.0) for item in deltas), default=0.0), "command_saturation_count": sum(bool((item.get("safety_projection") or {}).get("clip_reasons")) for item in deltas), "realized_control_deltas": deltas}


def _post_release_safety(control: ReplayTrace, treatment: ReplayTrace, decision: int) -> dict[str, Any]:
    """Compare only the frozen 120-second post-release window."""
    release = decision + int(TOTAL_TREATMENT_S)
    control_outcome = summarize_branch(control, decision_step=release, outcome_steps=int(POST_RELEASE_S), definition=DEFINITION)
    treatment_outcome = summarize_branch(treatment, decision_step=release, outcome_steps=int(POST_RELEASE_S), definition=DEFINITION)
    safe = (
        int(treatment_outcome["collision_unique_vehicle_count"]) == 0
        and int(treatment_outcome["starting_teleport_unique_vehicle_count"]) == 0
        and int(treatment_outcome["hard_brake_vehicle_steps"]) <= int(control_outcome["hard_brake_vehicle_steps"])
        and (treatment_outcome["min_ttc_s"] is None or (float(treatment_outcome["min_ttc_s"]) >= 1.5 and (control_outcome["min_ttc_s"] is None or float(treatment_outcome["min_ttc_s"]) >= float(control_outcome["min_ttc_s"]))))
    )
    return {"passed": safe, "control": control_outcome, "treatment": treatment_outcome}


def _safety_service(control: Mapping[str, Any], treatment: Mapping[str, Any], audit: Mapping[str, Any], dose: Mapping[str, Any], prefix: Mapping[str, Any], horizon_ok: bool, post_release_safe: bool) -> dict[str, bool]:
    c, t = control["outcomes"], treatment["outcomes"]
    checks = audit.get("checks") or {}
    return {"predecision_identity": bool(prefix.get("passed")), "causal_inband_only": bool(checks.get("override_only_during_treatment")), "out_of_band_damping_identity": bool(checks.get("out_of_band_damping_preserved")), "release_to_damping": bool(checks.get("release_returns_to_damping")), "complete_horizon": horizon_ok, "no_collision": int(t["collision_unique_vehicle_count"]) == 0, "no_teleport": int(t["starting_teleport_unique_vehicle_count"]) == 0, "no_excess_hard_braking": int(t["hard_brake_vehicle_steps"]) <= int(c["hard_brake_vehicle_steps"]), "no_ttc_deterioration": t["min_ttc_s"] is None or (float(t["min_ttc_s"]) >= 1.5 and (c["min_ttc_s"] is None or float(t["min_ttc_s"]) >= float(c["min_ttc_s"]))), "no_abort": not dose["abort_reasons"], "safe_handoff_limit": all(value <= 1 for value in dose["per_lane_handoff_counts"].values()), "service_noninferior": float(t["incremental_delay_s"]) <= float(c["incremental_delay_s"]) and int(t["merge_discharge_count"]) >= int(c["merge_discharge_count"]), "dose": dose["duration_fraction"] >= .8 and dose["lane_coverage"]["minimum"] >= .8 and ASSEMBLY_MIN_STEPS <= dose["realized"]["assembly_s"] <= ASSEMBLY_MAX_S and dose["realized"]["hold_s"] >= HOLD_S and dose["realized"]["taper_s"] >= TAPER_S, "compact_hold": dose["maximum_span_by_phase_m"]["ASSEMBLING"] <= 50 and dose["maximum_span_by_phase_m"]["HOLDING"] <= 50, "authority": dose["maximum_authority_mps2"] <= MAX_DECELERATION_MPS2 + 1e-9, "delayed_post_release_safety": bool(post_release_safe)}


def _material_benefit(predicate: int, longest: int, replay_noise: int) -> bool:
    return predicate >= MATERIAL_EFFECT_S and longest >= MATERIAL_EFFECT_S and predicate > replay_noise and longest > replay_noise


def _effect(control: ReplayTrace, treatment: ReplayTrace, decision: int, replay_noise: int) -> dict[str, Any]:
    c, t = _runs(control, decision), _runs(treatment, decision)
    predicate, longest = c["predicate_duration_s"] - t["predicate_duration_s"], c["longest_run_s"] - t["longest_run_s"]
    return {"control": c, "treatment": t, "predicate_reduction_s": predicate, "longest_run_reduction_s": longest, "replay_noise_s": replay_noise, "material_benefit": _material_benefit(predicate, longest, replay_noise)}


def _wall_case(seed: int, output: Path, sumo_binary: str, force: bool) -> dict[str, Any]:
    path = output / "cases" / f"seed{seed}.json"
    if path.exists() and not force: return {"status": "skipped", "seed": seed}
    base = json.loads((output / "baseline" / f"seed{seed}.json").read_text())
    if not bool(base["formation"]["feasible"]):
        record = {"schema": "idm_v5_7_1_case/1", "seed": seed, "decision_absolute_s": DECISION_ABSOLUTE_S, "formation": base["formation"], "derived_network_used": False, "wall_outcome_run": False, "reason": "gate_b_infeasible_at_frozen_t880"}
        atomic_write_json(path, record)
        return {"status": "infeasible_no_wall_branch", "seed": seed}
    decision = int(base["decision_relative_step"]); started = time.time()
    control = _replay(seed, decision, False, sumo_binary, OUTCOME_HORIZON_S); treatment = _replay(seed, decision, True, sumo_binary, OUTCOME_HORIZON_S)
    audit = _audit_contract(control, treatment, decision_step=decision, treatment_steps=int(TOTAL_TREATMENT_S), step_length_s=1.0); prefix = _prefix_identity(control, treatment, decision); dose = _dose(treatment, decision)
    arms = {"control": {"outcomes": summarize_branch(control, decision_step=decision, outcome_steps=OUTCOME_HORIZON_S, definition=DEFINITION), "trace": control.summary()}, "treatment": {"outcomes": summarize_branch(treatment, decision_step=decision, outcome_steps=OUTCOME_HORIZON_S, definition=DEFINITION), "trace": treatment.summary()}}
    replay_noise = abs(_runs(control, decision)["predicate_duration_s"] - int(base["congestion"]["predicate_duration_s"]))
    post_release = _post_release_safety(control, treatment, decision)
    safety = _safety_service(arms["control"], arms["treatment"], audit, dose, prefix, _complete_horizon(control, decision) and _complete_horizon(treatment, decision), bool(post_release["passed"])); effect = _effect(control, treatment, decision, replay_noise)
    record = {"schema": "idm_v5_7_1_case/1", "seed": seed, "decision_absolute_s": DECISION_ABSOLUTE_S, "formation": base["formation"], "profile": profile_manifest(), "derived_network_used": False, "prefix_identity": prefix, "treatment_contract": audit, "treatment_dose": dose, "arms": arms, "safety_service_integrity": safety, "post_release_safety": post_release, "effect": effect, "case_safe_complete": all(safety.values()), "wall_clock_seconds": time.time() - started}
    atomic_write_json(path, record); return {"status": "completed", "seed": seed, "safe_complete": record["case_safe_complete"], "benefit": effect["material_benefit"]}


def _summary(output: Path) -> dict[str, Any]:
    rows = [json.loads((output / "cases" / f"seed{seed}.json").read_text()) for seed in SEEDS]
    engaged = [row for row in rows if row.get("wall_outcome_run")]
    gates = {"all_identity": bool(engaged) and all(row["case_safe_complete"] and not row["derived_network_used"] for row in engaged), "three_safe_complete": sum(row["case_safe_complete"] for row in engaged) >= 3, "two_material_benefits": sum(row["effect"]["material_benefit"] for row in engaged) >= 2, "one_neutral": any(not row["effect"]["material_benefit"] for row in engaged), "no_single_extreme_seed": sum(row["effect"]["material_benefit"] for row in engaged) >= 2}
    return {"schema": "idm_v5_7_1_summary/1", "profile": profile_manifest(), "gates": gates, "four_case_passed": all(gates.values()), "predictor_training_authorized": False, "fresh_seeds_opened": False, "branch_count": 8 + 2 * len(engaged), "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=OUTPUT); parser.add_argument("--sumo-binary", default=str(DEFAULT_SUMO_EXE)); parser.add_argument("--workers", type=int, default=4); parser.add_argument("--gate", choices=("baseline", "wall"), required=True); parser.add_argument("--force", action="store_true"); args = parser.parse_args(); output = args.output.resolve(); (output / "baseline").mkdir(parents=True, exist_ok=True); (output / "cases").mkdir(parents=True, exist_ok=True)
    fn = _baseline_case if args.gate == "baseline" else _wall_case
    if args.gate == "wall":
        baseline = json.loads((output / "baseline_summary.json").read_text())
        if not baseline.get("gate_a_passed") or sum(row["formation"]["feasible"] for row in baseline["rows"]) < 3: raise RuntimeError("Gate A/B failed; wall cases are forbidden")
    with ProcessPoolExecutor(max_workers=min(max(1, args.workers), len(SEEDS))) as pool:
        for future in as_completed([pool.submit(fn, seed, output, args.sumo_binary, args.force) for seed in SEEDS]): print(json.dumps(future.result(), sort_keys=True), flush=True)
    if args.gate == "baseline": atomic_write_json(output / "baseline_summary.json", _baseline_summary(output))
    else: atomic_write_json(output / "summary.json", _summary(output))
    return 0


if __name__ == "__main__": raise SystemExit(main())
