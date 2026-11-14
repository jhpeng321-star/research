"""Stage 1b generation runner: per-seed demand scenario, D + W150/250/350 arms.

Per seed: write the demand-drawn scenario; replay Arm D with full observation
vector capture at t = 150/250/350 (extended: 120 s backward differences of
merge density, approach speed std and inflow-minus-discharge; per-lane CAV
positions relative to the band; ramp queue and 30 s arrivals; h); replay
W150/W250/W350 (damping + wall) with exact prefix identity through t_d.
Repeat D on every 20th seed. Outcomes per arm and paired targets per seed.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import traci

from config.constants import MERGE_EDGE, PRE_MERGE_EDGE, PRE_MERGE_LANE_LENGTH, RAMP_EDGE
from config.defaults import DEFAULT_CONFIG
from counterfactual.twin_replay import ReplayTrace, StepSnapshot, capture_step, compare_traces
from env.wall_supervisor import corridor_state
from experiments import REPO_ROOT
from experiments.idm import deterministic_sumo  # noqa: F401  (race-free SUMO launch, must precede env.core)
from experiments.idm import sustained_wall_v1_contract as C
from experiments.idm.run_sustained_wall_v1_stage1a import (
    DEFAULT_SUMO_EXE,
    TOTAL_STEPS,
    _branch_summary,
    _clean_reset_info,
    _json_safe,
    _wall_dose,
    env_overrides,
)
from experiments.idm.run_sustained_wall_v1_stage1b_prep4_observation_map import (
    _step_features,
)
from experiments.idm.stage1b_generation_contract import (
    FEATURE_TIMES,
    REPEAT_EVERY,
    SCENARIO_ROOT,
    contract_manifest,
    demand_for_seed,
    penetration_for_seed,
    strate_for_seed,
    write_seed_scenario,
)
from experiments.idm.stage1b_runner_core import replay_branch
from experiments.idm.wall_v5_2_contract import assert_no_prior_seed_artifacts
from experiments.io import atomic_write_json as _atomic_json
from idm.contract import contract_manifest as idm_contract_manifest
from policies.damping_plus_wall import DampingPlusWallPolicy
from policies.idm_analytical import build_frozen_idm_analytical_policy

DECISION_TIMES = (150, 250, 350)
HISTORY_STEPS = range(30, 361)
APPROACH_EDGE = "1191554636"
POST_MERGE_EDGE = "1487606178"
BAND_LO = 200.0
BAND_HI = 400.0


def _per_lane_cav_positions(env: Any, active_cavs: set) -> Dict[str, Dict[str, float]]:
    """Per mainline lane: CAV count in band, count in 400-500 m, nearest gap to band."""
    out: Dict[str, Dict[str, float]] = {}
    per_lane: Dict[str, list[float]] = {}
    for vid in sorted(active_cavs):
        try:
            road = str(traci.vehicle.getRoadID(vid))
            if road != PRE_MERGE_EDGE:
                continue
            lane = str(traci.vehicle.getLaneID(vid))
            lane_pos = float(traci.vehicle.getLanePosition(vid))
        except Exception:
            continue
        distance = max(0.0, PRE_MERGE_LANE_LENGTH - lane_pos)
        per_lane.setdefault(lane, []).append(distance)
    for lane, distances in per_lane.items():
        band = [d for d in distances if BAND_LO <= d <= BAND_HI]
        near_top = [d for d in distances if BAND_HI < d <= BAND_HI + 100.0]
        nearest = min(distances)
        out[lane] = {
            "count_in_band": float(len(band)),
            "count_in_400_500m": float(len(near_top)),
            "nearest_distance_to_band_top": float(max(0.0, nearest - BAND_HI)),
            "min_distance": float(nearest),
        }
    return out


def _feature_vector(
    live: Dict[str, Any],
    history: Mapping[int, Dict[str, Any]],
    t: int,
    h: float,
) -> Dict[str, float]:
    now = history[t]
    out: Dict[str, float] = {}
    regions = now.get("regions", {})
    for region in ("upstream", "approach", "near_merge", "merge_edge", "post_merge", "ramp"):
        stats = regions.get(region, {})
        out[f"{region}_count"] = float(stats.get("count", 0.0))
        out[f"{region}_mean_speed"] = float(stats.get("mean_speed", 0.0))
        out[f"{region}_speed_std"] = float(stats.get("speed_std", 0.0))
    out["mainline_lane_imbalance"] = float(now.get("mainline_lane_imbalance", 0.0))
    out["ramp_queue_length"] = float(now.get("ramp_queue_length", 0.0))
    out["cav_count_band"] = float(now.get("cav_count_band", 0.0))
    out["cav_count_upstream"] = float(now.get("cav_count_upstream", 0.0))
    out["ramp_30s_entries"] = sum(
        float(history[s].get("ramp_entries", 0.0)) for s in range(t - 30, t + 1) if s in history
    )
    inflow = sum(
        float(history[s].get("upstream_entries", 0.0)) for s in range(t - 30, t + 1) if s in history
    )
    discharge = sum(
        float(history[s].get("merge_discharge", 0.0)) for s in range(t - 30, t + 1) if s in history
    )
    out["upstream_inflow_30s"] = inflow
    out["merge_discharge_30s"] = discharge
    out["inflow_minus_discharge_30s"] = inflow - discharge
    for region, length_m, lanes in (("merge_edge", 300.0, 3), ("approach", 200.0, 3), ("upstream", 744.95, 3)):
        out[f"{region}_density"] = float(now.get("regions", {}).get(region, {}).get("count", 0.0)) / (
            length_m / 1000.0 * lanes
        )
    prev = history.get(t - 120)
    if prev is not None:
        merge_now = out["merge_edge_density"]
        merge_prev = float(prev.get("regions", {}).get("merge_edge", {}).get("count", 0.0)) / (
            300.0 / 1000.0 * 3
        )
        out["merge_density_diff_120s"] = merge_now - merge_prev
        out["approach_std_diff_120s"] = (
            out["approach_speed_std"]
            - float(prev.get("regions", {}).get("approach", {}).get("speed_std", 0.0))
        )
        inflow_prev = sum(
            float(history[s].get("upstream_entries", 0.0))
            for s in range(t - 120, t - 90 + 1)
            if s in history
        )
        discharge_prev = sum(
            float(history[s].get("merge_discharge", 0.0))
            for s in range(t - 120, t - 90 + 1)
            if s in history
        )
        out["inflow_minus_discharge_diff_120s"] = (
            (inflow - discharge) - (inflow_prev - discharge_prev)
        )
    for lane, values in (live.get("per_lane_cav", {}) or {}).items():
        out[f"lane_{lane}_count_in_band"] = float(values["count_in_band"])
        out[f"lane_{lane}_count_400_500m"] = float(values["count_in_400_500m"])
        out[f"lane_{lane}_nearest_to_band_top"] = float(values["nearest_distance_to_band_top"])
    out["h"] = float(h)
    return out


def _capture_record(features: Dict[str, Any]) -> Dict[str, Any]:
    """Compact per-step record for the rolling history (drops harmonic)."""
    record: Dict[str, Any] = {"regions": {}}
    for region, stats in features.get("regions", {}).items():
        record["regions"][region] = {
            "count": float(stats.get("count", 0.0)),
            "mean_speed": float(stats.get("mean_speed", 0.0)),
            "speed_std": float(stats.get("speed_std", 0.0)),
        }
    record["mainline_lane_imbalance"] = float(features["mainline_lane_imbalance"])
    record["ramp_queue_length"] = float(features["ramp_queue_length"])
    record["cav_count_band"] = float(features["cav_count_band"])
    record["cav_count_upstream"] = float(features["cav_count_upstream"])
    record["ramp_entries"] = float(features.get("ramp_entries", 0.0))
    record["upstream_entries"] = float(features.get("upstream_entries", 0.0))
    record["merge_discharge"] = float(features.get("merge_discharge", 0.0))
    return record


def _replay_d_with_features(
    *, seed: int, scenario_name: str, sumo_binary: str, total_steps: int,
) -> ReplayTrace:
    from env.core import TrafficShepherdMultiAgentEnv

    env_config: Dict[str, Any] = dict(DEFAULT_CONFIG)
    env_config.update(
        {
            "project_root": SCENARIO_ROOT,
            "sumo_binary": str(sumo_binary),
            "sumo_config": scenario_name,
            "sumo_seed": int(seed),
            "cav_penetration": float(penetration_for_seed(seed)),
            "max_episode_steps": total_steps + 5,
            "eval_mode": False,
        }
    )
    env_config.update(env_overrides())
    env = TrafficShepherdMultiAgentEnv(env_config)
    policy = build_frozen_idm_analytical_policy()
    snapshots: list[StepSnapshot] = []
    audits: list[Dict[str, Any]] = []
    health_history: list[float] = []
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
            if relative_step in HISTORY_STEPS:
                live_captures[relative_step] = _capture_record(_step_features(env))
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


def _extract_features(trace: ReplayTrace) -> Dict[str, Dict[str, float]]:
    captures = trace.controller_metadata.get("captures", {})
    h_series = [float(s.supervisor_observation[0]) for s in trace.snapshots]
    out = {}
    for t in FEATURE_TIMES:
        snapshot = {int(s.relative_step): s for s in trace.snapshots}.get(t)
        if snapshot is None:
            continue
        active = set(snapshot.active_cavs)
        # Per-lane CAV positions relative to the band, from captured rows.
        per_lane: Dict[str, list[float]] = {}
        for row in snapshot.vehicles:
            if str(row[0]) not in active or str(row[1]) != PRE_MERGE_EDGE:
                continue
            distance = max(0.0, PRE_MERGE_LANE_LENGTH - float(row[3]))
            per_lane.setdefault(str(row[2]), []).append(distance)
        per_lane_summary: Dict[str, Dict[str, float]] = {}
        for lane, distances in per_lane.items():
            band = [d for d in distances if BAND_LO <= d <= BAND_HI]
            near = [d for d in distances if BAND_HI < d <= BAND_HI + 100.0]
            per_lane_summary[lane] = {
                "count_in_band": float(len(band)),
                "count_in_400_500m": float(len(near)),
                "nearest_distance_to_band_top": max(0.0, min(distances) - BAND_HI),
            }
        out[t] = _feature_vector({"per_lane_cav": per_lane_summary}, captures, t, h_series[t])
    return out


def run_seed(case: Mapping[str, Any]) -> Dict[str, Any]:
    seed = int(case["seed"])
    output = Path(case["output"])
    result_path = output / "seeds" / f"seed{seed}.json"
    if result_path.exists() and not bool(case.get("force")):
        return {"status": "skipped", "seed": seed, "path": str(result_path)}
    started = time.time()
    demand = int(case.get("demand_override") or demand_for_seed(seed))
    scenario_name = write_seed_scenario(seed, demand)
    sumo_binary = str(case["sumo_binary"])
    d_trace = _replay_d_with_features(
        seed=seed, scenario_name=scenario_name, sumo_binary=sumo_binary, total_steps=TOTAL_STEPS
    )
    arms: Dict[str, ReplayTrace] = {"D": d_trace}
    for td in DECISION_TIMES:
        arms[f"W{td}"] = replay_branch(
            seed=seed, arm=f"W{td}", decision_step=None, scenario_file=scenario_name,
            scenario_root=SCENARIO_ROOT, sumo_binary=sumo_binary, total_steps=TOTAL_STEPS,
            policy_factory=(lambda t=td: DampingPlusWallPolicy(decision_step=t)),
            cav_penetration=float(penetration_for_seed(seed)),
        )
    integrity: Dict[str, Any] = {}
    for td in DECISION_TIMES:
        comparison = compare_traces(d_trace, arms[f"W{td}"])
        mismatch = comparison.get("first_mismatch")
        first_divergence = None if mismatch is None else int(mismatch.get("relative_step"))
        integrity[f"W{td}"] = {
            "prefix_identical_through_t_d": bool(
                first_divergence is None or first_divergence > td
            ),
            "first_divergence_step": first_divergence,
            "expected_first_divergence_step": td + 1,
        }
    summaries = {
        arm: _branch_summary(trace, arm=arm, decision_step=0) for arm, trace in arms.items()
    }
    d_sum = summaries["D"]
    payload: Dict[str, Any] = {
        "schema": "stage1b_generation_seed/1",
        "seed": seed,
        "hidden_demand_vph": demand,
        "strate": strate_for_seed(seed),
        "cav_penetration": penetration_for_seed(seed),
        "scenario": scenario_name,
        "features": _extract_features(d_trace),
        "arms": summaries,
        "targets": {
            **{
                f"Delta_delay_W{td}_minus_D": (
                    float(summaries[f"W{td}"]["delay"]["mean_per_vehicle_delay_s"])
                    - float(d_sum["delay"]["mean_per_vehicle_delay_s"])
                )
                for td in DECISION_TIMES
            },
            "Delta_delay_W150_minus_W250": (
                float(summaries["W150"]["delay"]["mean_per_vehicle_delay_s"])
                - float(summaries["W250"]["delay"]["mean_per_vehicle_delay_s"])
            ),
            "Delta_delay_W250_minus_W350": (
                float(summaries["W250"]["delay"]["mean_per_vehicle_delay_s"])
                - float(summaries["W350"]["delay"]["mean_per_vehicle_delay_s"])
            ),
        },
        "integrity": integrity,
        "repeat_check": None,
        "wall_dose": {
            f"W{td}": _wall_dose(arms[f"W{td}"]) for td in DECISION_TIMES
        },
        "wall_clock_seconds": time.time() - started,
    }
    if seed % REPEAT_EVERY == 1:
        repeated = _replay_d_with_features(
            seed=seed, scenario_name=scenario_name, sumo_binary=sumo_binary,
            total_steps=TOTAL_STEPS,
        )
        payload["repeat_check"] = {
            "second_digest": repeated.trace_digest,
            "exact_identity": repeated.trace_digest == d_trace.trace_digest,
        }
    _atomic_json(result_path, payload)
    return {"status": "completed", "seed": seed, "strate": strate_for_seed(seed),
            "demand": demand, "path": str(result_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strate", choices=("p20", "p30", "all"), default="p20")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--sumo-binary",
        default=str(DEFAULT_SUMO_EXE if DEFAULT_SUMO_EXE.exists() else "sumo"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-any-seeds", action="store_true",
                        help="gate-only: permit non-generation seed blocks (determinism gate repro)")
    parser.add_argument("--demand-override", type=int, default=None,
                        help="timing-only: force the drawn demand (never used for the population)")
    args = parser.parse_args()
    from experiments.idm.stage1b_generation_contract import ALL_SEEDS, SEEDS_P20, SEEDS_P30

    if args.seeds is None:
        seeds = ALL_SEEDS if args.strate == "all" else (SEEDS_P20 if args.strate == "p20" else SEEDS_P30)
    else:
        seeds = tuple(args.seeds)
    if not args.allow_any_seeds and not set(seeds) <= set(ALL_SEEDS):
        raise ValueError("seeds outside the generation block (use --allow-any-seeds only for the gate)")
    output = (
        args.output
        or REPO_ROOT / "runs" / "probe" / f"stage1b_generation_{args.strate}_s{seeds[0]}_{seeds[-1]}"
    ).resolve()
    family = tuple(
        (REPO_ROOT / "runs" / "probe" / p.name)
        for p in list((REPO_ROOT / "runs" / "probe").glob("sustained_wall_v1_stage1*_*"))
        + list((REPO_ROOT / "runs" / "probe").glob("stage1b_generation_*"))
        + list((REPO_ROOT / "runs" / "probe").glob("det_a*"))
        if p.is_dir()
    )
    artifact_audit = assert_no_prior_seed_artifacts(
        REPO_ROOT / "runs", seeds, allowed_roots=(output, *family,)
    )
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        output / "manifest.json",
        {
            "schema": "stage1b_generation_manifest/1",
            "created_unix_s": time.time(),
            "scientific_role": "stage1b_generation",
            "strate": args.strate,
            "seeds": seeds,
            "seed_role": "stage1b_generation",
            "prospective_evaluation_exclusion": True,
            "contract": contract_manifest(),
            "idm_contract": idm_contract_manifest(),
            "horizon_steps": TOTAL_STEPS,
            "prior_seed_artifact_audit": artifact_audit,
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
            "demand_override": args.demand_override,
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
