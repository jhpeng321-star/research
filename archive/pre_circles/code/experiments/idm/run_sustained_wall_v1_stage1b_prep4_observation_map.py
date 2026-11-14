"""Stage 1b-prep-4 Part C: raw RSU observation map at t = 100..300 (Arm C).

For seeds 6061-6090 (2000), 6091-6120 (2100), 6001-6030 (2200) the Arm C
branch is replayed with the standard machinery (identical calls, so the
evolved state is bit-identical; each replay's trace digest is compared with
the stored cell-C digest). At every step in [40, 310] the live per-region
RSU aggregate vector is recorded (regions: upstream > 400 m, approach
200-400 m, near-merge < 200 m, merge edge, post-merge edge, ramp): count,
mean speed, speed std, harmonic speed; mainline lane counts and imbalance;
ramp queue length and 30 s ramp arrivals; 30 s upstream inflow minus merge
discharge; 10/30/60 s backward differences of merge density and approach
speed std; CAV count in band and within 400 m upstream; h.
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
from counterfactual.twin_replay import ReplayTrace, StepSnapshot, capture_step
from env.wall_supervisor import corridor_state
from experiments import REPO_ROOT
from experiments.idm import sustained_wall_v1_contract as C
from experiments.idm.run_sustained_wall_v1_stage1a import (
    DEFAULT_SUMO_EXE,
    TOTAL_STEPS,
    _clean_reset_info,
    _json_safe,
    env_overrides,
)
from experiments.idm.wall_v5_2_contract import assert_no_prior_seed_artifacts
from experiments.io import atomic_write_json as _atomic_json
from idm.contract import (
    SUSTAINED_WALL_V1_STAGE1A_DEVELOPMENT_SEEDS,
    SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_2000_SEEDS,
    SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_2100_SEEDS,
    contract_manifest,
    scenario_config,
    validate_seed_role,
)
from policies.noop_base import NoOpNativeIDMPolicy

POST_MERGE_EDGE = "1487606178"
OBSERVE_STEPS = list(range(40, 311))
TARGET_STEPS = (100, 150, 200, 250, 300)


def _region_metrics(vehicle_rows: list[tuple]) -> Dict[str, Dict[str, float]]:
    """Per-region aggregates from (region, speed_mps) rows."""
    buckets: Dict[str, list[float]] = {}
    for region, speed in vehicle_rows:
        if speed is not None:
            buckets.setdefault(region, []).append(float(speed))
    out: Dict[str, Dict[str, float]] = {}
    for region, speeds in buckets.items():
        speeds_arr = speeds
        n = len(speeds_arr)
        mean = sum(speeds_arr) / n
        var = sum((s - mean) ** 2 for s in speeds_arr) / n
        harm = n / sum(1.0 / max(s, 0.1) for s in speeds_arr) if n else 0.0
        out[region] = {
            "count": float(n),
            "mean_speed": mean,
            "speed_std": var ** 0.5,
            "harmonic_speed": harm,
        }
    return out


def _step_features(env: Any) -> Dict[str, Any]:
    live = {str(v) for v in traci.vehicle.getIDList()}
    rows: list[tuple[str, float]] = []
    lane_counts: Dict[str, int] = {}
    seen_ramp: set[str] = set()
    cav_band = 0
    cav_upstream = 0
    active_cavs = set(getattr(env, "active_cavs", set()))
    prev_upstream = getattr(env, "_prep4_prev_upstream", set())
    prev_ramp = getattr(env, "_prep4_prev_ramp", set())
    now_upstream: set[str] = set()
    now_ramp: set[str] = set()
    for vid in sorted(live):
        try:
            road = str(traci.vehicle.getRoadID(vid))
            lane = str(traci.vehicle.getLaneID(vid))
            lane_pos = float(traci.vehicle.getLanePosition(vid))
            speed = float(traci.vehicle.getSpeed(vid))
            is_cav = vid in active_cavs
        except Exception:
            continue
        if road == PRE_MERGE_EDGE:
            distance = max(0.0, PRE_MERGE_LANE_LENGTH - lane_pos)
            lane_counts[lane] = lane_counts.get(lane, 0) + 1
            if distance > 400.0:
                rows.append(("upstream", speed))
                now_upstream.add(vid)
                if is_cav:
                    cav_upstream += 1
            elif distance >= 200.0:
                rows.append(("approach", speed))
                if is_cav:
                    cav_band += 1
            else:
                rows.append(("near_merge", speed))
        elif road == MERGE_EDGE or road.startswith(":"):
            rows.append(("merge_edge", speed))
        elif road == POST_MERGE_EDGE:
            rows.append(("post_merge", speed))
        elif road.startswith("11553048"):
            rows.append(("ramp", speed))
            if speed < 15.0:
                seen_ramp.add(vid)
            now_ramp.add(vid)
    env._prep4_prev_upstream = now_upstream
    env._prep4_prev_ramp = now_ramp
    on_merge = set()
    for vid in sorted(live):
        try:
            if str(traci.vehicle.getRoadID(vid)) == MERGE_EDGE:
                on_merge.add(vid)
        except Exception:
            pass
    prev_on_merge = getattr(env, "_prep4_prev_merge", set())
    discharge = len(prev_on_merge - on_merge)
    env._prep4_prev_merge = on_merge
    regions = _region_metrics(rows)
    mainline_counts = [count for count in lane_counts.values()]
    imbalance = (max(mainline_counts) - min(mainline_counts)) if mainline_counts else 0.0
    return {
        "regions": regions,
        "mainline_lane_counts": dict(lane_counts),
        "mainline_lane_imbalance": float(imbalance),
        "ramp_queue_length": float(len(seen_ramp)),
        "cav_count_band": float(cav_band),
        "cav_count_upstream": float(cav_upstream),
        "upstream_entries": float(len(now_upstream - prev_upstream)),
        "ramp_entries": float(len(now_ramp - prev_ramp)),
        "merge_discharge": float(discharge),
    }


def _replay_seed(
    *, seed: int, scenario_file: str, scenario_root: Path, sumo_binary: str,
    total_steps: int,
) -> Dict[str, Any]:
    from env.core import TrafficShepherdMultiAgentEnv

    env_config: Dict[str, Any] = dict(DEFAULT_CONFIG)
    env_config.update(
        {
            "project_root": scenario_root,
            "sumo_binary": str(sumo_binary),
            "sumo_config": scenario_file,
            "sumo_seed": int(seed),
            "cav_penetration": float(C.CAV_PENETRATION),
            "max_episode_steps": total_steps + 5,
            "eval_mode": False,
        }
    )
    env_config.update(env_overrides())
    env = TrafficShepherdMultiAgentEnv(env_config)
    policy = NoOpNativeIDMPolicy()
    step_features: Dict[int, Dict[str, Any]] = {}
    snapshots: list[StepSnapshot] = []
    audits: list[Dict[str, Any]] = []
    health_history: list[float] = []
    inflow_tracker: Dict[str, Any] = {}
    previous_health: Optional[float] = None
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
        audits.append({"relative_step": 0, "treatment_state": "DAMPING", "wall_active": False})
        for relative_step in range(1, total_steps + 1):
            action_dict = policy.act(obs)
            audit = _json_safe(policy.last_audit)
            audit.setdefault("relative_step", relative_step)
            audits.append(audit)
            obs, _rewards, terminated, truncated, _infos = env.step(action_dict)
            supervisor_observation = corridor_state(
                env,
                previous_health,
                0,
                0,
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
                    action=0,
                    supervisor_observation=supervisor_observation,
                    active_cav_ids=obs.keys(),
                    digits=6,
                )
            )
            if relative_step in OBSERVE_STEPS:
                step_features[relative_step] = _step_features(env)
            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break
        reset_info = _clean_reset_info(info)
        reset_info.setdefault("episode_time", snapshots[0].sim_time)
        reset_info.setdefault("active_cavs", list(snapshots[0].active_cavs))
        reset_info.setdefault("sumo_config", scenario_file)
        reset_info.setdefault("paired_seed", int(seed))
        trace = ReplayTrace(
            seed=int(seed),
            decision_step=0,
            horizon_steps=total_steps,
            prefix_action=0,
            post_action=0,
            treatment_steps=None,
            reset_info=reset_info,
            snapshots=tuple(snapshots),
            terminated_early=False,
            controller_metadata={},
            controller_audits=tuple(audits),
        )
        return {"trace_digest": trace.trace_digest, "step_features": step_features,
                "h_series": [float(s.supervisor_observation[0]) for s in snapshots]}
    finally:
        try:
            policy.close()
        finally:
            env.close()


def run_seed(case: Mapping[str, Any]) -> Dict[str, Any]:
    seed = int(case["seed"])
    mainline_vph = int(case["mainline_vph"])
    output = Path(case["output"])
    result_path = output / "seeds" / f"seed{seed}.json"
    if result_path.exists() and not bool(case.get("force")):
        return {"status": "skipped", "seed": seed, "path": str(result_path)}
    started = time.time()
    scenario_file = scenario_config(mainline_vph).name
    result = _replay_seed(
        seed=seed,
        scenario_file=scenario_file,
        scenario_root=C.SCENARIO_ROOT_PATH,
        sumo_binary=str(case["sumo_binary"]),
        total_steps=TOTAL_STEPS,
    )
    stored_root = Path(case["stored_c_root"])
    stored = json.loads(
        (stored_root / "seeds" / f"seed{seed}.json").read_text(encoding="utf-8")
    )
    if case["stored_c_key"] == "arms":
        stored_digest = str(stored["arms"]["C"]["trace_digest"])
    else:
        stored_digest = str(stored["branch"]["trace_digest"])
    payload = {
        "schema": "sustained_wall_v1_stage1b_prep4_observation_map_seed/1",
        "seed": seed,
        "mainline_demand_vph": mainline_vph,
        "trace_digest": result["trace_digest"],
        "stored_c_digest": stored_digest,
        "digest_matches_stored_c": result["trace_digest"] == stored_digest,
        "h_series": [round(float(v), 6) for v in result["h_series"]],
        "step_features": {
            str(step): features
            for step, features in sorted(result["step_features"].items())
        },
        "wall_clock_seconds": time.time() - started,
    }
    _atomic_json(result_path, payload)
    return {"status": "completed", "seed": seed, "mainline_vph": mainline_vph,
            "path": str(result_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mainline-vph", type=int, default=2200)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--sumo-binary",
        default=str(DEFAULT_SUMO_EXE if DEFAULT_SUMO_EXE.exists() else "sumo"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    mainline_vph = int(args.mainline_vph)
    cell_seeds = {
        2000: SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_2000_SEEDS,
        2100: SUSTAINED_WALL_V1_STAGE1B_BAND_LOCATOR_2100_SEEDS,
        2200: SUSTAINED_WALL_V1_STAGE1A_DEVELOPMENT_SEEDS,
    }[mainline_vph]
    role = (
        "sustained_wall_v1_stage1a_development"
        if mainline_vph == 2200
        else "sustained_wall_v1_stage1b_band_locator"
    )
    seeds = validate_seed_role(role, args.seeds or cell_seeds)
    if not set(seeds) <= set(cell_seeds):
        raise ValueError(f"seeds {seeds} are outside the m{mainline_vph} block")
    output = (
        args.output
        or REPO_ROOT
        / "runs"
        / "probe"
        / f"sustained_wall_v1_stage1b_prep4_obsmap_m{mainline_vph}_p20_s{seeds[0]}_{seeds[-1]}"
    ).resolve()
    stored_root = {
        2200: REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_hold_m2200_p20_s6001_6030",
        2100: REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_band_locator_m2100_s6091_6120",
        2000: REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_band_locator_m2000_s6061_6090",
    }[mainline_vph]
    stored_key = "arms" if mainline_vph == 2200 else "branch"
    family_roots = tuple(
        (REPO_ROOT / "runs" / "probe" / name)
        for name in (
            "sustained_wall_v1_stage1a_m2200_p20_s6001_6030",
            "sustained_wall_v1_stage1b_hold_m2200_p20_s6001_6030",
            "sustained_wall_v1_stage1b_prep2_wallspeed_scan_m2200_p20_s6001_6030",
            "sustained_wall_v1_stage1b_prep3_hrule_m2200_p20_s6001_6030",
            "sustained_wall_v1_stage1b_prep3_hrule_m2100_p20_s6091_6120",
            "sustained_wall_v1_stage1b_prep3_hrule_m2000_p20_s6061_6090",
            "sustained_wall_v1_stage1b_prep3_se_m2100_p20_s6091_6120",
            "sustained_wall_v1_stage1b_prep4_minhold_m2100_p20_s6091_6120",
            "sustained_wall_v1_stage1b_band_locator_m1900_s6031_6060",
            "sustained_wall_v1_stage1b_band_locator_m2000_s6061_6090",
            "sustained_wall_v1_stage1b_band_locator_m2100_s6091_6120",
            "sustained_wall_v1_stage1b_prep4_legacy_probe",
            "sustained_wall_v1_stage1b_prep4_obsmap_m2000_p20_s6061_6090",
            "sustained_wall_v1_stage1b_prep4_obsmap_m2100_p20_s6091_6120",
            "sustained_wall_v1_stage1b_prep4_obsmap_m2200_p20_s6001_6030",
        )
    )
    artifact_audit = assert_no_prior_seed_artifacts(
        REPO_ROOT / "runs", seeds, allowed_roots=(output, stored_root, *family_roots)
    )
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        output / "manifest.json",
        {
            "schema": "sustained_wall_v1_stage1b_prep4_observation_map_manifest/1",
            "created_unix_s": time.time(),
            "scientific_role": "stage1b_prep4_raw_observation_map",
            "cell": {
                "mainline_demand_vph": mainline_vph,
                "ramp_demand_vph": C.RAMP_DEMAND_VPH,
                "cav_penetration": C.CAV_PENETRATION,
                "scenario": scenario_config(mainline_vph).name,
            },
            "seeds": seeds,
            "seed_role": role,
            "prospective_evaluation_exclusion": True,
            "observation_steps": list(TARGET_STEPS),
            "feature_window_steps": [40, 310],
            "vector_definition": "per-region count/mean/std/harmonic; lane imbalance; ramp queue + 30 s arrivals; 30 s upstream inflow minus merge discharge; 10/30/60 s merge-density and approach-speed-std back differences; CAV band/upstream counts; h",
            "contract": C.contract_manifest(),
            "idm_contract": contract_manifest(),
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
            "mainline_vph": mainline_vph,
            "sumo_binary": args.sumo_binary,
            "output": str(output),
            "stored_c_root": str(stored_root),
            "stored_c_key": stored_key,
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
