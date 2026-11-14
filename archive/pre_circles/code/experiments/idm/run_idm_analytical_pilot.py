"""Resumable six-seed pilot for the frozen IDM analytical actuator."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import traci

from env.core import TrafficShepherdMultiAgentEnv
from env.idm_residual import IDMResidualActionEngine, IDMResidualConfig
from evaluation import evaluate_run, parse_sumo_stats
from experiments.io import atomic_write_json as _atomic_json
from idm.contract import (
    ANALYTICAL_PILOT_SEEDS,
    CAV_PENETRATION,
    CONTRACT_VERSION,
    DEMANDS_VPH,
    TRANSITION_SCAN_DEMANDS_VPH,
    LEGACY_PRUNED_CHECKPOINT,
    SCENARIO_ROOT,
    contract_manifest,
    scenario_config,
    transition_scenario_config,
    validate_seed_role,
)
from idm.pilot_metrics import summarize_control, trajectory_metrics

from experiments import REPO_ROOT

ROOT = REPO_ROOT
DEFAULT_OUTPUT = ROOT / "runs" / "eval" / "idm_analytical_pilot_v1"
MANDATORY_ARMS = (
    "noop",
    "pruned",
    "analytical",
    "analytical_zero_residual",
)
REDESIGN_ARMS = (
    "analytical_disturbance_reference",
    "analytical_disturbance_reference_zero_residual",
)
ALL_ARMS = MANDATORY_ARMS + REDESIGN_ARMS


def _base_env_config(
    arm: str,
    demand: int,
    seed: int,
    output_dir: Path,
    sumo_binary: str,
    max_episode_steps: int,
) -> dict:
    tag = f"{arm}.m{demand}r400.seed{seed}"
    cfg = {
        "project_root": SCENARIO_ROOT,
        "sumo_config": (
            scenario_config(demand).name
            if demand in DEMANDS_VPH
            else transition_scenario_config(demand).name
        ),
        "sumo_binary": sumo_binary,
        "sumo_seed": seed,
        "sumo_step_length": 0.5,
        "simulation_step": 1.0,
        "eval_mode": True,
        "eval_fcd_period": 1,
        "eval_tripinfo_output": output_dir / f"tripinfo.{tag}.xml",
        "eval_fcd_output": output_dir / f"fcd.{tag}.xml",
        "eval_stats_output": output_dir / f"stats.{tag}.xml",
        "cav_penetration": CAV_PENETRATION,
        "cav_placement": "scatter",
        "cav_force_normal_type": True,
        "assign_hdv_types": False,
        "max_episode_steps": max_episode_steps,
        "cav_warmup_steps": 300,
        "cav_empty_extend_steps": 600,
        "cav_empty_retries": 2,
        "use_v2_obs": True,
        "centralized_critic": False,
        "action_masking": False,
        "reward_mode": "shepherd",
        "control_mode": "shepherd",
    }
    if arm in (
        "analytical_zero_residual",
        "analytical_disturbance_reference_zero_residual",
    ):
        cfg["control_mode"] = "idm_residual_smoothing"
    if arm in REDESIGN_ARMS:
        cfg["idm_residual_reference_mode"] = "disturbance_p30"
    elif arm == "pruned":
        cfg.update(
            {
                "control_mode": "shepherd_hybrid",
                "centralized_critic": True,
                "action_masking": True,
                "shepherd_modes": ["coast", "recover", "guard"],
                "local_obs_dim": 124,
                "max_cavs_critic": 32,
                "max_neighbors": 20,
            }
        )
    return cfg


def _load_pruned_policy(env: TrafficShepherdMultiAgentEnv, seed: int):
    if not LEGACY_PRUNED_CHECKPOINT.exists():
        raise FileNotFoundError(f"comparison checkpoint is missing: {LEGACY_PRUNED_CHECKPOINT}")
    from policies.rllib_policy import RLlibPolicy

    policy = RLlibPolicy(str(LEGACY_PRUNED_CHECKPOINT), policy_id="cav_policy")
    policy.reset(env, seed=seed)
    return policy


def run_case(case: dict) -> dict:
    arm = str(case["arm"])
    demand = int(case["demand"])
    seed = int(case["seed"])
    output_dir = Path(case["output_dir"]).resolve()
    result_path = output_dir / "cases" / f"{arm}.m{demand}.seed{seed}.json"
    if result_path.exists() and not bool(case.get("force", False)):
        return {
            "status": "skipped",
            "path": str(result_path),
            "arm": arm,
            "demand": demand,
            "seed": seed,
        }

    cfg = _base_env_config(
        arm,
        demand,
        seed,
        output_dir / "artifacts",
        str(case["sumo_binary"]),
        int(case["max_episode_steps"]),
    )
    cfg["eval_tripinfo_output"].parent.mkdir(parents=True, exist_ok=True)
    env = TrafficShepherdMultiAgentEnv(cfg)
    direct_engine = (
        IDMResidualActionEngine(
            IDMResidualConfig(reference_mode="disturbance_p30")
            if arm == "analytical_disturbance_reference"
            else IDMResidualConfig()
        )
        if arm in ("analytical", "analytical_disturbance_reference")
        else None
    )
    policy = None
    history: list[dict] = []
    started = time.time()
    try:
        obs, reset_info = env.reset(seed=seed)
        if direct_engine is not None:
            direct_engine.reset()
        if arm == "pruned":
            policy = _load_pruned_policy(env, seed)

        done = False
        steps = 0
        episode_time = float(reset_info.get("warmup_steps", 0.0))
        while not done:
            action_time = float(traci.simulation.getTime())
            if arm == "noop":
                actions = {}
            elif arm == "pruned":
                actions = policy.act(obs)
            elif arm in ("analytical", "analytical_disturbance_reference"):
                diag = direct_engine.apply_all({}, cav_ids=env._active_cavs)
                diag = deepcopy(diag)
                diag["time_s"] = action_time
                history.append(diag)
                actions = {}
            elif arm in (
                "analytical_zero_residual",
                "analytical_disturbance_reference_zero_residual",
            ):
                actions = {cav_id: np.array([0.0], dtype=np.float32) for cav_id in obs}
            else:
                raise ValueError(f"unsupported arm {arm!r}")

            obs, _rewards, terminated, truncated, info = env.step(actions)
            if arm in (
                "analytical_zero_residual",
                "analytical_disturbance_reference_zero_residual",
            ):
                diag = deepcopy(info.get("idm_residual", {}))
                diag["time_s"] = action_time
                history.append(diag)
            steps += 1
            episode_time = float(info.get("episode_time", episode_time))
            done = bool(terminated.get("__all__", False) or truncated.get("__all__", False))

        tripinfo = Path(env.eval_tripinfo_output)
        fcd = Path(env.eval_fcd_output)
        stats = Path(env.eval_stats_output)
    finally:
        env.close()
        if policy is not None:
            policy.close()

    base = evaluate_run(
        tripinfo,
        fcd,
        stats_path=stats,
        sim_seconds=episode_time,
        ttc_threshold=1.5,
    )
    control, follower_events = summarize_control(history)
    trajectory = trajectory_metrics(fcd, follower_events)
    stats_record = parse_sumo_stats(stats)
    payload = {
        "schema": "idm_analytical_pilot_case/1",
        "contract_version": CONTRACT_VERSION,
        "arm": arm,
        "demand_mainline_vph": demand,
        "demand_ramp_vph": 400,
        "seed": seed,
        "steps": steps,
        "wall_clock_seconds": time.time() - started,
        "reset_info": reset_info,
        "metrics": base,
        "trajectory": trajectory,
        "control": control,
        "service": {
            "unfinished_vehicles": max(
                int(trajectory["service"]["vehicles_observed"]) - int(base["arrived"]),
                0,
            ),
            "stats": stats_record,
        },
        "artifacts": {
            "tripinfo": str(tripinfo),
            "fcd": str(fcd),
            "stats": str(stats),
        },
    }
    _atomic_json(result_path, payload)
    return {
        "status": "completed",
        "path": str(result_path),
        "arm": arm,
        "demand": demand,
        "seed": seed,
    }


def _nested(record: dict, path: tuple[str, ...]) -> float:
    value: Any = record
    for key in path:
        value = value[key]
    return float(value)


def build_summary(output_dir: Path, expected: list[dict]) -> dict:
    records = []
    missing = []
    for case in expected:
        path = output_dir / "cases" / (f"{case['arm']}.m{case['demand']}.seed{case['seed']}.json")
        if path.exists():
            records.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            missing.append(str(path))

    by_key = {(row["arm"], row["demand_mainline_vph"], row["seed"]): row for row in records}
    metric_paths = {
        "oscillation_energy": ("trajectory", "wave", "oscillation_energy_m2ps3"),
        "severe_braking": ("trajectory", "safety", "severe_braking_events"),
        "mean_time_loss": ("metrics", "mean_time_loss"),
        "arrived": ("metrics", "arrived"),
        "ttc_events": ("metrics", "ttc", "total_events"),
        "collisions": ("metrics", "collisions"),
    }
    has_redesign = any(key[0] == "analytical_disturbance_reference" for key in by_key)
    analytical_arm = "analytical_disturbance_reference" if has_redesign else "analytical"
    zero_arm = (
        "analytical_disturbance_reference_zero_residual"
        if has_redesign
        else "analytical_zero_residual"
    )
    paired = []
    identity = []
    demands_present = sorted({int(case["demand"]) for case in expected})
    seeds_present = sorted({int(case["seed"]) for case in expected})
    for demand in demands_present:
        for seed in seeds_present:
            noop = by_key.get(("noop", demand, seed))
            analytical = by_key.get((analytical_arm, demand, seed))
            zero = by_key.get((zero_arm, demand, seed))
            if noop and analytical:
                effects = {}
                for name, path in metric_paths.items():
                    baseline = _nested(noop, path)
                    treated = _nested(analytical, path)
                    effects[name] = {
                        "noop": baseline,
                        "analytical": treated,
                        "delta": treated - baseline,
                        "relative_reduction": (
                            (baseline - treated) / baseline
                            if abs(baseline) > 1e-12
                            else float("nan")
                        ),
                    }
                paired.append({"demand": demand, "seed": seed, "effects": effects})
            if analytical and zero:
                differences = {
                    name: abs(_nested(analytical, path) - _nested(zero, path))
                    for name, path in metric_paths.items()
                }
                identity.append(
                    {
                        "demand": demand,
                        "seed": seed,
                        "max_metric_absolute_difference": max(differences.values()),
                        "differences": differences,
                        "pass": max(differences.values()) <= 1e-9,
                    }
                )

    gates = {}
    for demand in demands_present:
        cell = [row for row in paired if row["demand"] == demand]
        if not cell:
            continue
        wave_wins = sum(
            row["effects"]["oscillation_energy"]["relative_reduction"] >= 0.10 for row in cell
        )
        brake_wins = sum(
            row["effects"]["severe_braking"]["relative_reduction"] >= 0.10 for row in cell
        )
        delay_ok = all(row["effects"]["mean_time_loss"]["delta"] <= 1.0 for row in cell)
        discharge_ok = all(row["effects"]["arrived"]["delta"] >= -1.0 for row in cell)
        gates[str(demand)] = {
            "paired_seed_count": len(cell),
            "wave_wins_at_10pct": wave_wins,
            "severe_braking_wins_at_10pct": brake_wins,
            "benefit_four_of_six": max(wave_wins, brake_wins) >= 4,
            "delay_within_1s_per_vehicle": delay_ok,
            "no_material_discharge_loss": discharge_ok,
            "zero_collisions": all(row["effects"]["collisions"]["analytical"] == 0 for row in cell),
        }

    summary = {
        "schema": "idm_analytical_pilot_summary/1",
        "contract": contract_manifest(),
        "record_count": len(records),
        "missing": missing,
        "evaluated_analytical_arm": analytical_arm,
        "evaluated_zero_residual_arm": zero_arm,
        "paired_effects": paired,
        "analytical_zero_residual_identity": identity,
        "identity_all_pass": bool(identity) and all(row["pass"] for row in identity),
        "cell_gates": gates,
        "phase3_pass": (
            not missing
            and bool(identity)
            and all(row["pass"] for row in identity)
            and any(
                gate["benefit_four_of_six"]
                and gate["delay_within_1s_per_vehicle"]
                and gate["no_material_discharge_loss"]
                and gate["zero_collisions"]
                for gate in gates.values()
            )
        ),
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(ANALYTICAL_PILOT_SEEDS))
    parser.add_argument("--demands", type=int, nargs="+", default=list(DEMANDS_VPH))
    parser.add_argument("--arms", nargs="+", default=list(MANDATORY_ARMS), choices=list(ALL_ARMS))
    parser.add_argument("--penetration", type=float, default=CAV_PENETRATION)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--sumo-binary", default="sumo")
    parser.add_argument("--max-episode-steps", type=int, default=1800)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--transition-scan",
        action="store_true",
        help="Allow the labeled 2300--2600 IDM development transition cells.",
    )
    args = parser.parse_args()

    seeds = validate_seed_role("analytical_pilot", args.seeds)
    demands = tuple(int(value) for value in args.demands)
    allowed_demands = set(DEMANDS_VPH)
    if args.transition_scan:
        allowed_demands.update(TRANSITION_SCAN_DEMANDS_VPH)
    if any(value not in allowed_demands for value in demands):
        raise ValueError(f"demands must be drawn from {sorted(allowed_demands)}")
    if any(value in TRANSITION_SCAN_DEMANDS_VPH for value in demands):
        if not args.transition_scan:
            raise ValueError("higher-demand cells require --transition-scan")
        if any(arm not in ("noop",) + REDESIGN_ARMS for arm in args.arms):
            raise ValueError("transition scan permits only noop and the labeled redesign arms")
    if abs(float(args.penetration) - CAV_PENETRATION) > 1e-12:
        raise ValueError(f"penetration is frozen at {CAV_PENETRATION}")
    if not Path(args.sumo_binary).exists() and args.sumo_binary != "sumo":
        raise FileNotFoundError(args.sumo_binary)

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "idm_analytical_pilot_manifest/1",
        "contract": contract_manifest(),
        "command": vars(args) | {"output": str(output_dir)},
        "created_unix_s": time.time(),
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    cases = [
        {
            "arm": arm,
            "demand": demand,
            "seed": seed,
            "output_dir": str(output_dir),
            "sumo_binary": args.sumo_binary,
            "max_episode_steps": args.max_episode_steps,
            "force": args.force,
        }
        for demand in demands
        for seed in seeds
        for arm in args.arms
    ]

    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {pool.submit(run_case, case): case for case in cases}
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            print(
                f"[{index}/{len(cases)}] {result['status']} "
                f"{result['arm']} m{result['demand']} seed{result['seed']}",
                flush=True,
            )
            build_summary(output_dir, cases)
    summary = build_summary(output_dir, cases)
    print(
        json.dumps(
            {
                "record_count": summary["record_count"],
                "missing_count": len(summary["missing"]),
                "identity_all_pass": summary["identity_all_pass"],
                "phase3_pass": summary["phase3_pass"],
                "summary": str(output_dir / "summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
