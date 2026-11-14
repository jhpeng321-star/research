"""Stage 1a sustained-wall effect-existence pilot runner (one cell).

For each seed: Arm C (native IDM, no commands) is replayed from time zero to
the common 1080 s horizon (or verified natural clearance per the v5.7.1
terminal rule). The health landmark t_d is computed on the Arm C prefix. When
a landmark exists, Arm W (sustained health-gated wall) and Arm P (one-step
0.5 m/s perturbation nudge) are replayed from time zero with the frozen seed;
when no landmark exists before t = 700 s the seed is recorded and only Arm C
is kept.

Prefix identity, digests and outcomes reuse
``counterfactual/twin_replay`` (``capture_step``, ``compare_traces``,
``ReplayTrace``) and ``counterfactual/outcomes``. The wall itself is a new
direct-control policy (``policies/sustained_wall.py``) wrapping the legacy
``WallHoldPolicy`` actuator; see that module for the implementation choice.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import traci

from config.constants import CAV_VTYPE, MERGE_EDGE
from config.defaults import DEFAULT_CONFIG
from counterfactual.outcomes import OutcomeDefinition, step_metrics, summarize_branch
from counterfactual.twin_replay import ReplayTrace, StepSnapshot, capture_step, compare_traces
from env.wall_supervisor import corridor_state
from experiments import REPO_ROOT
from experiments.idm import sustained_wall_v1_contract as C
from experiments.idm.wall_v5_2_contract import assert_no_prior_seed_artifacts
from experiments.io import atomic_write_json as _atomic_json
from idm.contract import contract_manifest, validate_seed_role
from policies.noop_base import NoOpNativeIDMPolicy
from policies.sustained_wall import PerturbationPolicy, SustainedWallPolicy

ROOT = REPO_ROOT
DEFAULT_OUTPUT = (
    ROOT / "runs" / "probe" / "sustained_wall_v1_stage1a_m2200_p20_s6001_6030"
)
DEFAULT_SUMO_EXE = Path(r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe")
TOTAL_STEPS = int(round(C.HORIZON_S))
REPEAT_SEEDS = (6001, 6002)
SCHEMA_VERSION = "sustained_wall_v1_stage1a_seed/1"


def _json_safe(value: Any) -> Any:
    """Convert policy audit values to deterministic JSON-compatible objects."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_json_safe(item) for item in value]
    try:
        return _json_safe(value.item())
    except (AttributeError, TypeError, ValueError):
        return repr(value)


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _branch_clock_seconds(started_s: float) -> float:
    return float(time.time() - started_s)


def _clean_reset_info(info: Mapping[str, Any]) -> Dict[str, Any]:
    """Deterministic reset metadata subset (twin_replay parity)."""
    cleaned: Dict[str, Any] = {}
    for key in ("episode_time", "warmup_steps", "sumo_config", "paired_seed", "reset_attempt"):
        if key in info:
            value = info[key]
            cleaned[key] = float(value) if key == "episode_time" else value
    if "active_cavs" in info:
        cleaned["active_cavs"] = sorted(str(x) for x in info["active_cavs"])
    return cleaned


def env_overrides() -> Dict[str, Any]:
    """Frozen environment contract: native IDM with the shepherd control mode,
    empty action dicts, and the two-step health-EMA call pattern the pilot
    policies rely on (one pre-step policy call, one post-step corridor call)."""
    return {
        "simulation_step": C.STEP_LENGTH_S,
        "sumo_step_length": 0.5,
        "cav_placement": "scatter",
        "cav_force_normal_type": True,
        "assign_hdv_types": False,
        "cav_warmup_steps": 300,
        "cav_empty_extend_steps": 600,
        "cav_empty_retries": 2,
        "control_mode": "shepherd",
        "reward_mode": "shepherd",
        "use_v2_obs": True,
        "centralized_critic": False,
        "action_masking": False,
    }


def _run_branch(
    *,
    seed: int,
    arm: str,
    decision_step: Optional[int],
    sumo_binary: str,
    total_steps: int,
) -> ReplayTrace:
    from env.core import TrafficShepherdMultiAgentEnv

    env_config: Dict[str, Any] = dict(DEFAULT_CONFIG)
    env_config.update(
        {
            "project_root": C.SCENARIO_ROOT_PATH,
            "sumo_binary": str(sumo_binary),
            "sumo_config": C.SCENARIO_FILE,
            "sumo_seed": int(seed),
            "cav_penetration": float(C.CAV_PENETRATION),
            "max_episode_steps": total_steps + 5,
            "eval_mode": False,
        }
    )
    env_config.update(env_overrides())
    env = TrafficShepherdMultiAgentEnv(env_config)
    if arm == "C":
        policy = NoOpNativeIDMPolicy()
    elif arm == "W":
        if decision_step is None:
            raise ValueError("Arm W requires a landmark decision step")
        policy = SustainedWallPolicy(
            decision_step=int(decision_step),
            wall_params=dict(C.WALL_PARAMS),
            recovery_h=C.RECOVERY_H,
            recovery_persist_s=C.RECOVERY_PERSIST_S,
            max_wall_duration_s=C.MAX_WALL_DURATION_S,
            wall_cap_s=C.WALL_CAP_S,
            taper_duration_s=C.TAPER_DURATION_S,
        )
    elif arm == "P":
        if decision_step is None:
            raise ValueError("Arm P requires a landmark decision step")
        policy = PerturbationPolicy(
            decision_step=int(decision_step),
            slow_down_mps=C.NUDGE_SLOW_DOWN_MPS,
            landmark_distance_m=C.NUDGE_LANDMARK_DISTANCE_M,
        )
    else:
        raise ValueError(f"unknown arm {arm!r}")

    snapshots: list[StepSnapshot] = []
    audits: list[Dict[str, Any]] = []
    health_history: list[float] = []
    inflow_tracker: Dict[str, Any] = {}
    previous_health: Optional[float] = None
    engaged_steps = 0
    terminated_early = False
    termination_reason: Optional[str] = None
    terminal_expected: Optional[int] = None
    try:
        obs, info = env.reset(seed=seed)
        policy.reset(env, seed=seed)
        supervisor_observation = corridor_state(
            env, None, 0, 0, 0, inflow_tracker, health_history
        )
        snapshots.append(
            capture_step(
                traci,
                relative_step=0,
                action=0,
                supervisor_observation=supervisor_observation,
                active_cav_ids=obs.keys(),
                digits=6,
            )
        )
        audits.append(
            {"relative_step": 0, "command": 0, "treatment_state": "DAMPING", "wall_active": False}
        )
        for relative_step in range(1, total_steps + 1):
            action_dict = policy.act(obs)
            audit = _json_safe(policy.last_audit)
            audit.setdefault("relative_step", relative_step)
            wall_active = bool(audit.get("wall_active"))
            if wall_active:
                engaged_steps += 1
            audits.append(audit)
            obs, _rewards, terminated, truncated, _infos = env.step(action_dict)
            supervisor_observation = corridor_state(
                env,
                previous_health,
                1 if wall_active else 0,
                engaged_steps,
                relative_step,
                inflow_tracker,
                health_history,
            )
            previous_health = float(supervisor_observation[0])
            health_history.append(previous_health)
            snapshots.append(
                capture_step(
                    traci,
                    relative_step=relative_step,
                    action=1 if wall_active else 0,
                    supervisor_observation=supervisor_observation,
                    active_cav_ids=obs.keys(),
                    digits=6,
                )
            )
            if terminated.get("__all__", False) or truncated.get("__all__", False):
                terminated_early = relative_step < total_steps
                termination_reason = str(_infos.get("end_reason") or "unknown")
                try:
                    terminal_expected = int(traci.simulation.getMinExpectedNumber())
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    terminal_expected = None
                break

        reset_info = _clean_reset_info(info)
        reset_info.setdefault("episode_time", snapshots[0].sim_time)
        reset_info.setdefault("active_cavs", list(snapshots[0].active_cavs))
        reset_info.setdefault("sumo_config", C.SCENARIO_FILE)
        reset_info.setdefault("paired_seed", int(seed))
        metadata = {
            "runner": "sustained_wall_v1_stage1a",
            "arm": arm,
            "decision_step": int(decision_step) if decision_step is not None else None,
            "termination_reason": termination_reason or "horizon_reached",
            "terminal_expected_vehicle_count": terminal_expected,
            "controller_step_s": C.STEP_LENGTH_S,
        }
        return ReplayTrace(
            seed=int(seed),
            decision_step=int(decision_step) if decision_step is not None else 0,
            horizon_steps=total_steps - (int(decision_step) if decision_step is not None else 0),
            prefix_action=0,
            post_action=1 if arm == "W" else 0,
            treatment_steps=None,
            reset_info=reset_info,
            snapshots=tuple(snapshots),
            terminated_early=terminated_early,
            controller_metadata=_json_safe(metadata),
            controller_audits=tuple(audits),
        )
    finally:
        try:
            policy.close()
        finally:
            env.close()


def _terminal_classification(trace: ReplayTrace) -> Dict[str, Any]:
    metadata = trace.controller_metadata
    reason = str(metadata.get("termination_reason") or "horizon_reached")
    expected = metadata.get("terminal_expected_vehicle_count")
    if reason == "horizon_reached":
        classification = "horizon"
    elif reason == "terminated" and expected == 0:
        classification = "natural_clearance"
    else:
        classification = "early_incomplete"
    return {
        "classification": classification,
        "end_reason": reason,
        "terminal_expected_vehicle_count": expected,
        "final_sim_time_s": float(trace.snapshots[-1].sim_time),
        "final_relative_step": int(trace.snapshots[-1].relative_step),
    }


def _event_counts(trace: ReplayTrace) -> Dict[str, Any]:
    colliding: set[str] = set()
    teleporting: set[str] = set()
    collision_steps = 0
    teleport_steps = 0
    for snapshot in trace.snapshots:
        colliding.update(snapshot.collision_vehicle_ids)
        teleporting.update(snapshot.starting_teleport_vehicle_ids)
        collision_steps += len(snapshot.collision_vehicle_ids)
        teleport_steps += len(snapshot.starting_teleport_vehicle_ids)
    return {
        "collision_unique_vehicle_count": len(colliding),
        "collision_vehicle_steps": collision_steps,
        "teleport_unique_vehicle_count": len(teleporting),
        "teleport_vehicle_steps": teleport_steps,
    }


def _branch_summary(trace: ReplayTrace, *, arm: str, decision_step: Optional[int]) -> Dict[str, Any]:
    definition = OutcomeDefinition()
    decision = int(decision_step) if decision_step is not None else 0
    outcomes = summarize_branch(
        trace,
        decision_step=decision,
        outcome_steps=TOTAL_STEPS - decision,
        definition=definition,
    )
    per_step = [step_metrics(snapshot, definition) for snapshot in trace.snapshots]
    h_series = [float(snapshot.supervisor_observation[0]) for snapshot in trace.snapshots]
    seen_ids: set[str] = set()
    for snapshot in trace.snapshots:
        seen_ids.update(str(row[0]) for row in snapshot.vehicles)
    # Per-step incremental delay (positive time-loss increments over all
    # vehicles present) and merge-edge discharge transitions, mirroring
    # counterfactual.outcomes.summarize_branch but step-indexed.
    incremental_delay_steps: list[float] = [0.0]
    discharge_steps: list[int] = [0]
    for previous, current in zip(trace.snapshots, trace.snapshots[1:]):
        previous_time_loss = {str(row[0]): float(row[6]) for row in previous.vehicles}
        current_time_loss = {str(row[0]): float(row[6]) for row in current.vehicles}
        increment = 0.0
        for vehicle_id, cumulative in current_time_loss.items():
            prior = previous_time_loss.get(vehicle_id, 0.0)
            if math.isfinite(cumulative) and math.isfinite(prior):
                increment += max(0.0, cumulative - prior)
        incremental_delay_steps.append(float(increment))
        previous_road = {str(row[0]): str(row[1]) for row in previous.vehicles}
        current_road = {str(row[0]): str(row[1]) for row in current.vehicles}
        discharge = sum(
            1
            for vehicle_id, road_id in previous_road.items()
            if road_id == MERGE_EDGE and current_road.get(vehicle_id) not in (None, MERGE_EDGE)
        )
        discharge_steps.append(discharge)
    window_s = float(trace.snapshots[-1].sim_time) - float(trace.snapshots[decision].sim_time)
    discharge = int(outcomes["merge_discharge_count"])
    taper_start: Optional[int] = None
    wall_active_steps: list[int] = []
    for audit in trace.controller_audits:
        step = int(audit.get("relative_step") or 0)
        if str(audit.get("treatment_state")) == "TAPERING" and taper_start is None:
            taper_start = step
        if bool(audit.get("wall_active")):
            wall_active_steps.append(step)
    series = {
        "step": [int(each.relative_step) for each in trace.snapshots],
        "h": [round(float(value), 6) for value in h_series],
        "queue_count": [int(each["queue_count"]) for each in per_step],
        "pre_merge_mean_speed_mps": [
            None if each["pre_merge_mean_speed_mps"] is None
            else round(float(each["pre_merge_mean_speed_mps"]), 4)
            for each in per_step
        ],
        "hard_brake_count": [len(list(each["hard_brake_vehicle_ids"])) for each in per_step],
        "low_ttc_count": [len(list(each["low_ttc_pairs"])) for each in per_step],
        "low_ttc_step_flag": [1 if each["low_ttc_pairs"] else 0 for each in per_step],
        "incremental_delay_s": [round(float(value), 4) for value in incremental_delay_steps],
        "discharge_count": discharge_steps,
    }
    if arm == "W":
        # Per-step dose telemetry (member count and per-lane coverage), one
        # entry per controller step including the pre-engagement prefix.
        series["member_count"] = [
            int(audit.get("member_count") or 0) for audit in trace.controller_audits
        ]
        series["lane_coverage_fraction"] = [
            round(float(audit.get("lane_coverage_fraction") or 0.0), 4)
            for audit in trace.controller_audits
        ]
    # Tau-integrity probe: the wall sets member tau to original + 3.0 s; it must
    # be restored once a CAV leaves the band (and at release). We report the
    # maximum tau observed among CAV-typed vehicles in a window after release
    # for Arm W. Note: walls at 10 m/s cannot leave the band and cross the merge
    # in one controller step, so every ex-member is released on the pre-edge
    # before reaching the merge.
    tau_probe: Dict[str, Any] = {}
    release_audit_step: Optional[int] = None
    for audit_row in trace.controller_audits:
        if audit_row.get("release_step") is not None:
            release_audit_step = int(audit_row["release_step"])
            break
    if arm == "W" and release_audit_step is not None:
        probe_steps = [release_audit_step + offset for offset in (3, 6, 10)]
        by_step = {int(s.relative_step): s for s in trace.snapshots}
        maxima: Dict[int, Optional[float]] = {}
        for probe_step in probe_steps:
            snapshot = by_step.get(probe_step)
            values = [
                _number(row[9])
                for row in (snapshot.vehicles if snapshot is not None else [])
                if str(row[8]) == CAV_VTYPE
            ]
            maxima[probe_step] = max(values) if values else None
        tau_probe = {
            "window": probe_steps,
            "max_cav_tau_s_by_step": maxima,
            "wall_tau_s": 3.0,
            "restored": all(
                value is not None and value <= C.WALL_PARAMS["wall_tau"] + 1.6
                for value in maxima.values()
            ),
        }
    return {
        "arm": arm,
        "trace_digest": trace.trace_digest,
        "decision_step": decision,
        "termination": _terminal_classification(trace),
        "events": _event_counts(trace),
        "outcomes": outcomes,
        "health_collapse": C.health_collapse_statistics(h_series).to_dict(),
        "physical_collapse": C.physical_collapse_statistics(per_step).to_dict(),
        "tau_integrity": tau_probe,
        "delay": {
            "incremental_delay_s": float(outcomes["incremental_delay_s"]),
            "distinct_vehicle_count": len(seen_ids),
            "mean_per_vehicle_delay_s": (
                float(outcomes["incremental_delay_s"]) / len(seen_ids) if seen_ids else 0.0
            ),
            "window_s": float(window_s),
            "completed_vehicles": float(outcomes["merge_discharge_count"]),
            "vehicles_completed": max(
                0, len(seen_ids) - int(len(trace.snapshots[-1].vehicles))
            ),
            "vehicles_unfinished_at_terminal": int(len(trace.snapshots[-1].vehicles)),
        },
        "discharge": {
            "merge_discharge_count": discharge,
            "throughput_veh_per_h": (discharge / window_s * 3600.0) if window_s > 0 else 0.0,
        },
        "windows": {
            "taper_start_step": taper_start,
            "wall_active_steps": wall_active_steps,
        },
        "series": series,
    }


def _integrity_checks(
    control: ReplayTrace,
    arm_traces: Mapping[str, ReplayTrace],
    *,
    decision_step: Optional[int],
) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    if decision_step is None:
        return {"landmark": "none", "skipped": ["W", "P"]}
    for arm in ("W", "P"):
        treatment = arm_traces[arm]
        comparison = compare_traces(control, treatment)
        mismatch = comparison.get("first_mismatch")
        first_divergence = None if mismatch is None else int(mismatch.get("relative_step"))
        # Prefix identity: all snapshots through t_d must be identical.
        prefix_identical = bool(
            first_divergence is None or first_divergence > int(decision_step)
        )
        checks[arm] = {
            "prefix_identical_through_decision_step": prefix_identical,
            "prefix_identical_through_step": decision_step if prefix_identical else None,
            "first_divergence_step": first_divergence,
            "first_divergence_sim_time_s": (
                None if mismatch is None else mismatch.get("left_sim_time")
            ),
            "first_divergence_reason": (
                None if mismatch is None else str(mismatch.get("reason") or "state_mismatch")
            ),
            "comparison_passed": bool(comparison["passed"]),
        }
    return checks


def run_seed(case: Mapping[str, Any]) -> Dict[str, Any]:
    seed = int(case["seed"])
    output = Path(case["output"])
    result_path = output / "seeds" / f"seed{seed}.json"
    if result_path.exists() and not bool(case.get("force")):
        return {"status": "skipped", "seed": seed, "path": str(result_path)}

    started = time.time()
    sumo_binary = str(case["sumo_binary"])
    branch_times: Dict[str, float] = {}
    branch_started = time.time()
    control = _run_branch(
        seed=seed, arm="C", decision_step=None, sumo_binary=sumo_binary, total_steps=TOTAL_STEPS
    )
    branch_times["C"] = _branch_clock_seconds(branch_started)
    h_series = [float(snapshot.supervisor_observation[0]) for snapshot in control.snapshots]
    t_d = C.find_landmark(h_series)
    t_d_h = h_series[t_d] if t_d is not None else None
    arm_traces: Dict[str, ReplayTrace] = {"C": control}
    if t_d is not None:
        branch_started = time.time()
        arm_traces["W"] = _run_branch(
            seed=seed, arm="W", decision_step=t_d, sumo_binary=sumo_binary, total_steps=TOTAL_STEPS
        )
        branch_times["W"] = _branch_clock_seconds(branch_started)
        branch_started = time.time()
        arm_traces["P"] = _run_branch(
            seed=seed, arm="P", decision_step=t_d, sumo_binary=sumo_binary, total_steps=TOTAL_STEPS
        )
        branch_times["P"] = _branch_clock_seconds(branch_started)
    repeat_checks = None
    if seed in REPEAT_SEEDS:
        branch_started = time.time()
        control_repeat = _run_branch(
            seed=seed, arm="C", decision_step=None, sumo_binary=sumo_binary,
            total_steps=TOTAL_STEPS,
        )
        branch_times["C_repeat"] = _branch_clock_seconds(branch_started)
        repeat_checks = {
            "arm_c_vs_arm_c": compare_traces(control, control_repeat),
            "control_repeat_digest": control_repeat.trace_digest,
        }

    summaries = {
        arm: _branch_summary(trace, arm=arm, decision_step=t_d)
        for arm, trace in arm_traces.items()
    }
    integrity = _integrity_checks(control, arm_traces, decision_step=t_d)
    payload = {
        "schema": SCHEMA_VERSION,
        "seed": seed,
        "cell": {
            "mainline_demand_vph": C.MAINLINE_DEMAND_VPH,
            "ramp_demand_vph": C.RAMP_DEMAND_VPH,
            "cav_penetration": C.CAV_PENETRATION,
            "scenario": C.SCENARIO_FILE,
        },
        "decision_landmark": {
            "exists": t_d is not None,
            "step": t_d,
            "sim_time_s": None if t_d is None else float(control.snapshots[t_d].sim_time),
            "health_at_landmark": t_d_h,
        },
        "arms": summaries,
        "integrity": integrity,
        "repeat_checks": repeat_checks,
        "branch_wall_clock_s": branch_times,
        "wall_dose": _wall_dose(arm_traces.get("W")) if "W" in arm_traces else None,
        "wall_clock_seconds": time.time() - started,
    }
    _atomic_json(result_path, payload)
    return {"status": "completed", "seed": seed, "path": str(result_path)}


def _wall_dose(trace: Optional[ReplayTrace]) -> Optional[Dict[str, Any]]:
    if trace is None:
        return None
    audits = [a for a in trace.controller_audits if a.get("wall_active")]
    holding = [a for a in audits if str(a.get("treatment_state")) == "HOLDING"]
    members_per_step = [int(a.get("member_count") or 0) for a in audits]
    coverage = [
        float(a.get("lane_coverage_fraction") or 0.0)
        for a in audits
        if a.get("lane_coverage_fraction") is not None
    ]
    release_reason: Optional[str] = None
    release_step: Optional[int] = None
    taper_elapsed: list[int] = []
    for a in trace.controller_audits:
        if a.get("release_reason") is not None and release_reason is None:
            release_reason = str(a.get("release_reason"))
        if a.get("release_step") is not None and release_step is None:
            release_step = int(a["release_step"])
        if str(a.get("treatment_state")) == "TAPERING":
            taper_elapsed.append(int(a.get("taper_elapsed_steps") or 0))
    realized_s = len(holding)
    return {
        "realized_wall_seconds": realized_s,
        "realized_wall_steps": realized_s,
        "release_reason": release_reason,
        "release_step": release_step,
        "release_sim_time_s": None if release_step is None or release_step >= len(trace.snapshots)
        else float(trace.snapshots[release_step].sim_time),
        "taper_elapsed_steps": taper_elapsed,
        "member_seconds": float(sum(members_per_step)),
        "member_step_count": int(len(members_per_step)),
        "mean_member_count": (
            float(sum(members_per_step)) / len(members_per_step) if members_per_step else 0.0
        ),
        "coverage_statistics": {
            "mean": float(np.mean(coverage)) if coverage else 0.0,
            "min": float(np.min(coverage)) if coverage else 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(C.SEEDS))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--sumo-binary",
        default=str(DEFAULT_SUMO_EXE if DEFAULT_SUMO_EXE.exists() else "sumo"),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    seeds = validate_seed_role("sustained_wall_v1_stage1a_development", args.seeds)
    output = args.output.resolve()
    artifact_audit = assert_no_prior_seed_artifacts(
        ROOT / "runs", seeds, allowed_roots=(output,)
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "sustained_wall_v1_stage1a_manifest/1",
        "created_unix_s": time.time(),
        "scientific_role": "sustained_wall_effect_existence_stage1a_development",
        "seeds": seeds,
        "repeat_seeds": list(REPEAT_SEEDS),
        "seed_role": "sustained_wall_v1_stage1a_development",
        "prospective_evaluation_exclusion": True,
        "contract": C.contract_manifest(),
        "idm_contract": contract_manifest(),
        "horizon_steps": TOTAL_STEPS,
        "env_overrides": env_overrides(),
        "outcome_definition": OutcomeDefinition().to_dict(),
        "terminal_rule": "common 1080 s horizon or verified natural clearance (v5.7.1)",
        "prior_seed_artifact_audit": artifact_audit,
        "command": {
            "seeds": seeds,
            "workers": args.workers,
            "sumo_binary": args.sumo_binary,
            "output": str(output),
        },
    }
    _atomic_json(output / "manifest.json", manifest)
    cases = [
        {
            "seed": seed,
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
    print(json.dumps({"status": "all_workers_done", "seeds": seeds}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
