"""Stage 2 final-generation runner: D + onset-triggered holds per seed.

Per seed (p20 8001-8240 first, then p30 8241-8360): write the plateau-
decline scenario; replay D with the repaired per-step capture over steps
30-960 (extended with per-lane CAV distances for the unclipped geometry);
detect the onset (first step with h < 0.15 for 10 consecutive s, capped at
t = 900); extract feature vectors at t = 150, t_on - 60 and t_on; and when
an onset exists run WON_H300 / WON_H450 / WON_H650 (damping + wall engaged
at t_on + 1, held 300/450/650 s, 5 s taper) with exact prefix identity
through t_on. D is repeated on every 20th seed.

Outcomes per arm: whole-branch summaries (delay in total s and s/veh,
completions = merge-edge exits within the branch, health/physical collapse
labels, termination), hard-brake and low-TTC by location x phase, and the
paired targets Delta_delay(WON_Hk - D), best_k per seed, and the value of
choosing duration (best_k minus H450). No demand, no absolute time in any
feature.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import traci

from config.constants import PRE_MERGE_EDGE, PRE_MERGE_LANE_LENGTH
from config.defaults import DEFAULT_CONFIG
from counterfactual.twin_replay import ReplayTrace, StepSnapshot, capture_step, compare_traces
from env.wall_supervisor import corridor_state
from experiments import REPO_ROOT
from experiments.idm import deterministic_sumo  # noqa: F401  (race-free launch, must precede env.core)
from experiments.idm.rsu_feature_capture import (
    active_cav_ids,
    first_onset_step,
    per_lane_cav_distances,
    safety_events_by_location,
    step_features_fixed,
)
from experiments.idm.run_sustained_wall_v1_stage1a import (
    DEFAULT_SUMO_EXE,
    TOTAL_STEPS,
    _branch_summary,
    _clean_reset_info,
    _json_safe,
    env_overrides,
)
from experiments.idm.stage2_generation_contract import (
    ALL_SEEDS,
    CAPTURE_HI_STEP,
    CAPTURE_LO_STEP,
    CAV_PENETRATION_P20,
    CAV_PENETRATION_P30,
    FIXED_REFERENCE_STEP,
    HOLDS_S,
    ONSET_SEARCH_MAX_STEP,
    ONSET_SUSTAIN_S,
    REPEAT_EVERY,
    SCENARIO_ROOT,
    contract_manifest,
    demand_for_seed,
    penetration_for_seed,
    per_lane_unclipped,
    stage2_feature_vector,
    strate_for_seed,
    write_seed_scenario,
)
from experiments.io import atomic_write_json as _atomic_json
from idm.contract import contract_manifest as idm_contract_manifest, validate_seed_role
from policies.idm_analytical import build_frozen_idm_analytical_policy
from policies.stage1c2_arms import OnsetTriggeredDampingWallPolicy


def _replay_d_with_stage2_capture(
    *, seed: int, scenario_name: str, sumo_binary: str, total_steps: int,
) -> ReplayTrace:
    """D replay capturing the fixed record + per-lane distances over 30-960."""
    from env.core import TrafficShepherdMultiAgentEnv

    penetration = (CAV_PENETRATION_P20 if seed in range(8001, 8241)
                   else CAV_PENETRATION_P30)
    env_config: Dict[str, Any] = dict(DEFAULT_CONFIG)
    env_config.update(
        {
            "project_root": str(SCENARIO_ROOT),
            "sumo_binary": str(sumo_binary),
            "sumo_config": scenario_name,
            "sumo_seed": int(seed),
            "cav_penetration": float(penetration),
            "max_episode_steps": total_steps + 5,
            "eval_mode": False,
        }
    )
    env_config.update(env_overrides())
    env = TrafficShepherdMultiAgentEnv(env_config)
    policy = build_frozen_idm_analytical_policy()
    snapshots: list[StepSnapshot] = []
    audits: List[Dict[str, Any]] = []
    health_history: List[float] = []
    inflow_tracker: Dict[str, Any] = {}
    previous_health: Optional[float] = None
    live_captures: Dict[int, Dict[str, Any]] = {}
    try:
        obs, info = env.reset(seed=seed)
        policy.reset(env, seed=seed)
        supervisor_observation = corridor_state(env, None, 0, 0, 0, inflow_tracker, health_history)
        snapshots.append(capture_step(traci, relative_step=0, action=0,
                                      supervisor_observation=supervisor_observation,
                                      active_cav_ids=obs.keys(), digits=6))
        audits.append({"relative_step": 0, "treatment_state": "DAMPING", "wall_active": False})
        for relative_step in range(1, total_steps + 1):
            policy.act(obs)
            audit = _json_safe(policy.last_diagnostics)
            audit["relative_step"] = relative_step
            audit["treatment_state"] = "DAMPING"
            audit["wall_active"] = False
            audits.append(audit)
            obs, _rewards, terminated, truncated, _infos = env.step({})
            supervisor_observation = corridor_state(
                env, previous_health, 0, 0, relative_step, inflow_tracker, health_history
            )
            previous_health = float(supervisor_observation[0])
            health_history.append(previous_health)
            snapshots.append(capture_step(traci, relative_step=relative_step, action=0,
                                          supervisor_observation=supervisor_observation,
                                          active_cav_ids=obs.keys(), digits=6))
            if CAPTURE_LO_STEP <= relative_step <= CAPTURE_HI_STEP:
                record = step_features_fixed(env)
                record["per_lane_distances"] = per_lane_cav_distances(env)
                live_captures[relative_step] = record
            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break
        reset_info = _clean_reset_info(info)
        reset_info.setdefault("episode_time", snapshots[0].sim_time)
        reset_info.setdefault("active_cavs", list(snapshots[0].active_cavs))
        reset_info.setdefault("sumo_config", scenario_name)
        reset_info.setdefault("paired_seed", int(seed))
        return ReplayTrace(
            seed=int(seed), decision_step=0, horizon_steps=total_steps,
            prefix_action=0, post_action=0, treatment_steps=None,
            reset_info=reset_info, snapshots=tuple(snapshots), terminated_early=False,
            controller_metadata={"captures": live_captures},
            controller_audits=tuple(audits),
        )
    finally:
        try:
            policy.close()
        finally:
            env.close()


def _extract_stage2_features(trace: ReplayTrace, onset: Optional[int]) -> Dict[str, Any]:
    captures = trace.controller_metadata.get("captures", {})
    h_series = [float(s.supervisor_observation[0]) for s in trace.snapshots]
    times: Dict[str, Optional[int]] = {
        "t150": FIXED_REFERENCE_STEP if FIXED_REFERENCE_STEP in captures else None,
    }
    if onset is not None:
        times["t_on"] = onset if onset in captures else None
        times["t_on_minus_60"] = (onset - 60) if (onset - 60) in captures else None
    out: Dict[str, Any] = {"times": {k: v for k, v in times.items()}}
    for name, t in times.items():
        if t is None:
            out[name] = None
            continue
        record = captures[t]
        per_lane = per_lane_unclipped(record.get("per_lane_distances", {}))
        out[name] = stage2_feature_vector(captures, per_lane, t, h_series[t])
    return out


def _replay_w_stage2(
    *, seed: int, scenario_name: str, sumo_binary: str, factory,
) -> ReplayTrace:
    """Wall-branch replay mirroring the Stage 1c ``_replay_w`` loop, with the
    stage-2 scenario root and per-stratum penetration (the 1c helper pins
    its own module SCENARIO_ROOT, which would point SUMO at idm_stage1c)."""
    from env.core import TrafficShepherdMultiAgentEnv

    env_config: Dict[str, Any] = dict(DEFAULT_CONFIG)
    env_config.update(
        {
            "project_root": str(SCENARIO_ROOT),
            "sumo_binary": str(sumo_binary),
            "sumo_config": scenario_name,
            "sumo_seed": int(seed),
            "cav_penetration": float(penetration_for_seed(seed)),
            "max_episode_steps": TOTAL_STEPS + 5,
            "eval_mode": False,
        }
    )
    env_config.update(env_overrides())
    env = TrafficShepherdMultiAgentEnv(env_config)
    policy = factory()
    snapshots: list[StepSnapshot] = []
    audits: List[Dict[str, Any]] = []
    health_history: List[float] = []
    inflow_tracker: Dict[str, Any] = {}
    previous_health: Optional[float] = None
    try:
        obs, info = env.reset(seed=seed)
        policy.reset(env, seed=seed)
        supervisor_observation = corridor_state(env, None, 0, 0, 0, inflow_tracker, health_history)
        snapshots.append(capture_step(traci, relative_step=0, action=0,
                                      supervisor_observation=supervisor_observation,
                                      active_cav_ids=obs.keys(), digits=6))
        audits.append({"relative_step": 0, "treatment_state": "DAMPING", "wall_active": False})
        for relative_step in range(1, TOTAL_STEPS + 1):
            policy.act(obs)
            audit = _json_safe(policy.last_audit)
            audit["relative_step"] = relative_step
            audits.append(audit)
            obs, _r, terminated, truncated, _i = env.step({})
            supervisor_observation = corridor_state(
                env, previous_health, 0, 0, relative_step, inflow_tracker, health_history
            )
            previous_health = float(supervisor_observation[0])
            health_history.append(previous_health)
            snapshots.append(capture_step(traci, relative_step=relative_step, action=0,
                                          supervisor_observation=supervisor_observation,
                                          active_cav_ids=obs.keys(), digits=6))
            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break
        reset_info = _clean_reset_info(info)
        reset_info.setdefault("episode_time", snapshots[0].sim_time)
        reset_info.setdefault("active_cavs", list(snapshots[0].active_cavs))
        reset_info.setdefault("sumo_config", scenario_name)
        reset_info.setdefault("paired_seed", int(seed))
        return ReplayTrace(
            seed=int(seed), decision_step=0, horizon_steps=TOTAL_STEPS,
            prefix_action=0, post_action=0, treatment_steps=None,
            reset_info=reset_info, snapshots=tuple(snapshots), terminated_early=False,
            controller_metadata={"_w_rule_inputs": dict(getattr(policy, "rule_inputs", {}))},
            controller_audits=tuple(audits),
        )
    finally:
        try:
            policy.close()
        finally:
            env.close()


def run_seed(case: Mapping[str, Any]) -> Dict[str, Any]:
    seed = int(case["seed"])
    output = Path(case["output"])
    result_path = output / "seeds" / f"seed{seed}.json"
    if result_path.exists() and not bool(case.get("force")):
        return {"status": "skipped", "seed": seed, "path": str(result_path)}
    started = time.time()
    demand = demand_for_seed(seed)
    scenario_name = write_seed_scenario(seed, demand)
    sumo_binary = str(case["sumo_binary"])

    d_trace = _replay_d_with_stage2_capture(
        seed=seed, scenario_name=scenario_name, sumo_binary=sumo_binary,
        total_steps=TOTAL_STEPS,
    )
    h_series = [float(s.supervisor_observation[0]) for s in d_trace.snapshots]
    onset = first_onset_step(h_series, sustain_s=ONSET_SUSTAIN_S,
                             max_step=ONSET_SEARCH_MAX_STEP)
    features = _extract_stage2_features(d_trace, onset)

    arms: Dict[str, Any] = {"D": _branch_summary(d_trace, arm="D", decision_step=0)}
    integrity: Dict[str, Any] = {}
    if onset is not None:
        for hold in HOLDS_S:
            arm = f"WON_H{int(hold)}"
            trace = _replay_w_stage2(
                seed=seed, scenario_name=scenario_name, sumo_binary=sumo_binary,
                factory=lambda h=hold: OnsetTriggeredDampingWallPolicy(
                    onset_step=int(onset), hold_s=h),
            )
            comparison = compare_traces(d_trace, trace)
            mismatch = comparison.get("first_mismatch")
            first_divergence = None if mismatch is None else int(mismatch.get("relative_step"))
            integrity[arm] = {
                "prefix_identical_through_t_on": bool(
                    first_divergence is None or first_divergence > onset),
                "first_divergence_step": first_divergence,
                "expected_first_divergence_step": int(onset) + 1,
                "t_on": int(onset),
            }
            release_step = None
            for audit in trace.controller_audits:
                if str(audit.get("treatment_state")) == "TAPERING":
                    release_step = int(audit.get("relative_step"))
                    break
            summary = _branch_summary(trace, arm=arm, decision_step=0)
            summary["engage_step"] = int(onset) + 1
            summary["release_step"] = release_step
            summary["hold_duration_s"] = (
                (release_step - (int(onset) + 1)) if release_step is not None else None
            )
            summary["safety_by_location"] = safety_events_by_location(
                trace.snapshots, engage_step=int(onset) + 1, release_step=release_step)
            arms[arm] = summary

    # Paired targets (total s and s/veh).
    d_delay = arms["D"]["delay"]
    d_total = float(d_delay["incremental_delay_s"])
    d_sveh = float(d_delay["mean_per_vehicle_delay_s"])
    targets: Dict[str, Any] = {}
    if onset is not None:
        deltas_total: Dict[str, float] = {}
        deltas_sveh: Dict[str, float] = {}
        for hold in HOLDS_S:
            arm = f"WON_H{int(hold)}"
            a = arms[arm]["delay"]
            deltas_total[arm] = float(a["incremental_delay_s"]) - d_total
            deltas_sveh[arm] = float(a["mean_per_vehicle_delay_s"]) - d_sveh
            targets[f"delta_delay_total_s_{arm}"] = deltas_total[arm]
            targets[f"delta_delay_s_per_veh_{arm}"] = deltas_sveh[arm]
        best_arm = min(deltas_total, key=deltas_total.get)
        targets["best_k"] = best_arm
        targets["duration_choice_value_total_s"] = (
            deltas_total[best_arm] - deltas_total["WON_H450"])
        targets["duration_choice_value_s_per_veh"] = (
            deltas_sveh[best_arm] - deltas_sveh["WON_H450"])

    payload: Dict[str, Any] = {
        "schema": "stage2_generation_seed/1",
        "seed": seed,
        "strate": strate_for_seed(seed),
        "hidden_demand_vph": demand,
        "cav_penetration": penetration_for_seed(seed),
        "scenario": scenario_name,
        "onset_step": int(onset) if onset is not None else None,
        "no_onset_by_900": onset is None,
        "features": features,
        "arms": arms,
        "targets": targets,
        "integrity": integrity,
        "repeat_check": None,
        "wall_clock_seconds": time.time() - started,
    }
    if (seed - int(case["repeat_offset"])) % REPEAT_EVERY == 0:
        repeated = _replay_d_with_stage2_capture(
            seed=seed, scenario_name=scenario_name, sumo_binary=sumo_binary,
            total_steps=TOTAL_STEPS,
        )
        payload["repeat_check"] = {
            "second_digest": repeated.trace_digest,
            "exact_identity": repeated.trace_digest == d_trace.trace_digest,
        }
    _atomic_json(result_path, payload)
    return {"status": "completed", "seed": seed, "strate": strate_for_seed(seed),
            "demand": demand, "onset": payload["onset_step"], "path": str(result_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strate", choices=("p20", "p30"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--sumo-binary", default=str(DEFAULT_SUMO_EXE if DEFAULT_SUMO_EXE.exists() else "sumo"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    from experiments.idm.stage2_generation_contract import SEEDS_P20, SEEDS_P30

    seeds = (SEEDS_P20 if args.strate == "p20" else SEEDS_P30) if args.seeds is None \
        else tuple(args.seeds)
    seeds = validate_seed_role("stage2_generation", seeds)
    output = (
        args.output
        or REPO_ROOT / "runs" / "probe" / f"stage2_generation_{args.strate}_s{seeds[0]}_{seeds[-1]}"
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        output / "manifest.json",
        {
            "schema": "stage2_generation_manifest/1",
            "created_unix_s": time.time(),
            "scientific_role": "stage2_generation",
            "strate": args.strate,
            "seeds": seeds,
            "seed_role": "stage2_generation",
            "prospective_evaluation_exclusion": True,
            "contract": contract_manifest(),
            "idm_contract": idm_contract_manifest(),
            "horizon_steps": TOTAL_STEPS,
            "command": {
                "seeds": seeds,
                "workers": args.workers,
                "sumo_binary": args.sumo_binary,
                "output": str(output),
            },
        },
    )
    cases = [
        {
            "seed": seed,
            "sumo_binary": args.sumo_binary,
            "output": str(output),
            "force": args.force,
            "repeat_offset": seeds[0],
        }
        for seed in seeds
    ]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(run_seed, case) for case in cases]
        for future in as_completed(futures):
            print(json.dumps(future.result(), sort_keys=True), flush=True)
    print(json.dumps({"status": "all_workers_done", "strate": args.strate,
                      "seeds": len(seeds)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
