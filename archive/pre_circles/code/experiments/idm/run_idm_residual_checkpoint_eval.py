"""Paired development evaluation for trained IDM residual checkpoints.

Reconstructs the Phase-4 IDM environment; the generic evaluator would change
the treatment to ordinary continuous acceleration.
"""

from __future__ import annotations
import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any
import numpy as np
from env.core import TrafficShepherdMultiAgentEnv
from evaluation import evaluate_run, parse_sumo_stats
from experiments.io import atomic_write_json as _atomic_json
from idm.contract import (
    ANALYTICAL_PILOT_SEEDS,
    CAV_PENETRATION,
    CONTRACT_VERSION,
    SCENARIO_ROOT,
    contract_manifest,
    scenario_config,
    transition_scenario_config,
    validate_seed_role,
)
from idm.pilot_metrics import summarize_control, trajectory_metrics
from policies.rllib_policy import RLlibPolicy, checkpoint_custom_model_config

from experiments import REPO_ROOT

ROOT = REPO_ROOT
DEFAULT_OUTPUT = ROOT / "runs" / "eval" / "idm_residual_phase4_stage1_eval_v1"
PHASE5_REFS_PATH = ROOT / "runs" / "eval" / "idm_phase5_reward_refs_v1.json"
REWARD_COMPONENT_KEYS = (
    "local",
    "efficiency",
    "delay_budget",
    "corridor_ttc",
    "corridor_severe_braking",
    "corridor_wave_energy",
    "wave",
    "residual_effort",
)


def _env_config(
    demand: int, seed: int, output_dir: Path, arm: str, max_steps: int, checkpoint: Path | None
) -> dict:
    tag = f"{arm}.m{demand}r400.seed{seed}"
    cfg = {
        "project_root": SCENARIO_ROOT,
        "sumo_config": (
            scenario_config(demand).name
            if demand <= 2200
            else transition_scenario_config(demand).name
        ),
        "sumo_binary": "sumo",
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
        "max_episode_steps": max_steps,
        "cav_warmup_steps": 300,
        "cav_empty_extend_steps": 600,
        "cav_empty_retries": 2,
        "use_v2_obs": True,
        "centralized_critic": True,
        "action_masking": False,
        "reward_mode": "idm_damping",
        "control_mode": "idm_residual_smoothing",
        "idm_residual_reference_mode": "disturbance_p30",
        "use_wave_signal": True,
        "use_idm_absorber_gate": True,
        "use_idm_controller_context": True,
        "idm_residual_filter_tau_s": 5.0,
        "scenario_reward_refs": dict(
            json.loads(PHASE5_REFS_PATH.read_text(encoding="utf-8")).get("refs") or {}
        ),
        # Phase-5 actor: base 121 + wave(2) + gate + controller/temporal context(10).
        "local_obs_dim": 134,
        "max_neighbors": 20,
        "max_cavs_critic": 48,
    }
    if checkpoint is not None:
        custom = checkpoint_custom_model_config(checkpoint) or {}
        for key in ("local_obs_dim", "max_neighbors", "max_cavs_critic"):
            if key in custom:
                cfg[key] = int(custom[key])
    return cfg


def _residual_summary(history: list[dict]) -> dict:
    values = [
        float(command.get("residual_mps2", 0.0))
        for row in history
        for command in row.get("commands", [])
    ]
    if not values:
        return dict(
            count=0,
            mean_mps2=0.0,
            mean_abs_mps2=0.0,
            minimum_mps2=0.0,
            maximum_mps2=0.0,
            near_lower_bound_fraction=0.0,
            near_upper_bound_fraction=0.0,
        )
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean_mps2": float(np.mean(array)),
        "mean_abs_mps2": float(np.mean(np.abs(array))),
        "minimum_mps2": float(np.min(array)),
        "maximum_mps2": float(np.max(array)),
        "near_lower_bound_fraction": float(np.mean(array <= -0.49)),
        "near_upper_bound_fraction": float(np.mean(array >= 0.29)),
    }


def run_case(case: dict) -> dict:
    arm, demand, seed = str(case["arm"]), int(case["demand"]), int(case["seed"])
    checkpoint = Path(case["checkpoint"]).resolve() if case.get("checkpoint") else None
    constant = float(case["constant"]) if case.get("constant") is not None else None
    rule = str(case["rule"]) if case.get("rule") else None
    output_dir = Path(case["output_dir"]).resolve()
    result_path = output_dir / "cases" / f"{arm}.m{demand}.seed{seed}.json"
    if result_path.exists() and not bool(case.get("force", False)):
        return {"status": "skipped", "arm": arm, "demand": demand, "seed": seed}

    cfg = _env_config(
        demand,
        seed,
        output_dir / "artifacts",
        arm,
        int(case["max_episode_steps"]),
        checkpoint,
    )
    cfg["eval_tripinfo_output"].parent.mkdir(parents=True, exist_ok=True)
    env = TrafficShepherdMultiAgentEnv(cfg)
    policy = RLlibPolicy(str(checkpoint), "cav_policy") if checkpoint else None
    history, started = [], time.time()
    reward_scalar_total = 0.0
    reward_selected_agent_steps = 0
    reward_component_totals = {key: 0.0 for key in REWARD_COMPONENT_KEYS}
    reward_component_abs_totals = {key: 0.0 for key in REWARD_COMPONENT_KEYS}
    reward_max_accounting_error = 0.0
    prior_commands: dict[str, dict] = {}
    active_rule_lanes: set[str] = set()
    try:
        obs, reset_info = env.reset(seed=seed)
        if policy:
            policy.reset(env, seed=seed)
        done, steps = False, 0
        episode_time = float(reset_info.get("warmup_steps", 0.0))
        while not done:
            residual = 0.0 if constant is None else constant
            if policy:
                actions = policy.act(obs)
            else:
                actions = {}
                for cav_id in obs:
                    command = prior_commands.get(cav_id, {})
                    reference = command.get("reference", {})
                    lane_id = str(command.get("lane_id", ""))
                    gated = (
                        rule == "reference_le_24"
                        and float(reference.get("filtered_reference_mps", 99.0)) <= 24.0
                    ) or (rule == "reference_hysteresis_24_25" and lane_id in active_rule_lanes)
                    value = 0.1 if gated else residual
                    actions[cav_id] = np.array([value], dtype=np.float32)
            obs, step_rewards, terminated, truncated, info = env.step(actions)
            step_scalar_total = float(sum(float(value) for value in step_rewards.values()))
            reward_scalar_total += step_scalar_total
            step_component_total = 0.0
            for row in info.get("idm_reward_components", {}).values():
                reward_selected_agent_steps += 1
                step_component_total += float(row["total"])
                for key in REWARD_COMPONENT_KEYS:
                    value = float(row[key])
                    reward_component_totals[key] += value
                    reward_component_abs_totals[key] += abs(value)
            reward_max_accounting_error = max(
                reward_max_accounting_error,
                abs(step_scalar_total - step_component_total),
            )
            diag = dict(info.get("idm_residual", {}))
            diag["time_s"] = float(info.get("episode_time", episode_time))
            history.append(diag)
            if rule == "reference_hysteresis_24_25":
                for command in diag.get("commands", []):
                    lane_id = str(command.get("lane_id", ""))
                    reference = command.get("reference", {})
                    filtered = float(reference.get("filtered_reference_mps", 99.0))
                    if filtered <= 24.0:
                        active_rule_lanes.add(lane_id)
                    elif filtered >= 25.0:
                        active_rule_lanes.discard(lane_id)
            prior_commands = {
                str(command["vehicle_id"]): command for command in diag.get("commands", [])
            }
            steps += 1
            episode_time = float(info.get("episode_time", episode_time))
            done = bool(terminated.get("__all__", False) or truncated.get("__all__", False))
        tripinfo = Path(env.eval_tripinfo_output)
        fcd = Path(env.eval_fcd_output)
        stats = Path(env.eval_stats_output)
    finally:
        env.close()
        if policy:
            policy.close()

    metrics = evaluate_run(
        tripinfo, fcd, stats_path=stats, sim_seconds=episode_time, ttc_threshold=1.5
    )
    control, follower_events = summarize_control(history)
    trajectory = trajectory_metrics(fcd, follower_events)
    payload = {
        "schema": "idm_residual_checkpoint_case/1",
        "contract_version": CONTRACT_VERSION,
        "arm": arm,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "constant_residual_mps2": constant,
        "diagnostic_rule": rule,
        "demand_mainline_vph": demand,
        "demand_ramp_vph": 400,
        "seed": seed,
        "steps": steps,
        "wall_clock_seconds": time.time() - started,
        "reset_info": reset_info,
        "metrics": metrics,
        "trajectory": trajectory,
        "control": control,
        "residual": _residual_summary(history),
        "learning_reward": {
            "scalar_total": reward_scalar_total,
            "selected_agent_steps": reward_selected_agent_steps,
            "mean_per_selected_agent_step": (
                reward_scalar_total / reward_selected_agent_steps
                if reward_selected_agent_steps
                else 0.0
            ),
            "component_totals": reward_component_totals,
            "component_means": {
                key: value / reward_selected_agent_steps if reward_selected_agent_steps else 0.0
                for key, value in reward_component_totals.items()
            },
            "component_mean_absolute": {
                key: value / reward_selected_agent_steps if reward_selected_agent_steps else 0.0
                for key, value in reward_component_abs_totals.items()
            },
            "max_step_accounting_error": reward_max_accounting_error,
        },
        "service": {
            "unfinished_vehicles": max(
                int(trajectory["service"]["vehicles_observed"]) - int(metrics["arrived"]), 0
            ),
            "stats": parse_sumo_stats(stats),
        },
        "artifacts": {"tripinfo": str(tripinfo), "fcd": str(fcd), "stats": str(stats)},
    }
    _atomic_json(result_path, payload)
    return {"status": "completed", "arm": arm, "demand": demand, "seed": seed}


def _value(row: dict, *keys: str) -> float:
    value: Any = row
    for key in keys:
        value = value[key]
    return float(value)


def build_summary(output_dir: Path, cases: list[dict]) -> dict:
    rows, missing = {}, []
    for case in cases:
        path = output_dir / "cases" / (f"{case['arm']}.m{case['demand']}.seed{case['seed']}.json")
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            key = (row["arm"], int(row["demand_mainline_vph"]), int(row["seed"]))
            rows[key] = row
        else:
            missing.append(str(path))

    candidates = sorted({case["arm"] for case in cases if case["arm"] != "zero"})
    paired, gates = [], {}
    for arm in candidates:
        effects = []
        for (row_arm, demand, seed), trained in rows.items():
            if row_arm != arm:
                continue
            zero = rows.get(("zero", demand, seed))
            if not zero:
                continue
            wave0 = _value(zero, "trajectory", "wave", "oscillation_energy_m2ps3")
            brake0 = _value(zero, "trajectory", "safety", "severe_braking_events")
            effect = {
                "arm": arm,
                "demand": demand,
                "seed": seed,
                "wave_reduction": (
                    wave0 - _value(trained, "trajectory", "wave", "oscillation_energy_m2ps3")
                )
                / wave0
                if wave0
                else math.nan,
                "severe_braking_reduction": (
                    brake0 - _value(trained, "trajectory", "safety", "severe_braking_events")
                )
                / brake0
                if brake0
                else math.nan,
                "delay_delta_s_per_vehicle": (
                    _value(trained, "metrics", "mean_time_loss")
                    - _value(zero, "metrics", "mean_time_loss")
                ),
                "arrived_delta": int(
                    _value(trained, "metrics", "arrived") - _value(zero, "metrics", "arrived")
                ),
                "collision_delta": int(
                    _value(trained, "metrics", "collisions") - _value(zero, "metrics", "collisions")
                ),
                "residual_mean_mps2": _value(trained, "residual", "mean_mps2"),
                "residual_mean_abs_mps2": _value(trained, "residual", "mean_abs_mps2"),
                "near_bound_fraction": (
                    _value(trained, "residual", "near_lower_bound_fraction")
                    + _value(trained, "residual", "near_upper_bound_fraction")
                ),
                "learning_reward_delta": (
                    _value(trained, "learning_reward", "mean_per_selected_agent_step")
                    - _value(zero, "learning_reward", "mean_per_selected_agent_step")
                ),
                "reward_component_deltas": {
                    key: (
                        _value(trained, "learning_reward", "component_means", key)
                        - _value(zero, "learning_reward", "component_means", key)
                    )
                    for key in REWARD_COMPONENT_KEYS
                },
            }
            effects.append(effect)
            paired.append(effect)
        if not effects:
            continue
        wave_wins = sum(x["wave_reduction"] >= 0.10 for x in effects)
        brake_wins = sum(x["severe_braking_reduction"] >= 0.10 for x in effects)
        gate = {
            "paired_seed_count": len(effects),
            "wave_wins_at_10pct": wave_wins,
            "severe_braking_wins_at_10pct": brake_wins,
            "benefit_four_of_six": max(wave_wins, brake_wins) >= 4,
            "delay_within_1s_per_vehicle": all(
                x["delay_delta_s_per_vehicle"] <= 1.0 for x in effects
            ),
            "no_material_discharge_loss": all(x["arrived_delta"] >= -1 for x in effects),
            "zero_collisions": all(x["collision_delta"] <= 0 for x in effects),
            "mean_wave_reduction": mean(x["wave_reduction"] for x in effects),
            "mean_severe_braking_reduction": mean(x["severe_braking_reduction"] for x in effects),
            "mean_delay_delta_s_per_vehicle": mean(x["delay_delta_s_per_vehicle"] for x in effects),
            "mean_residual_mps2": mean(x["residual_mean_mps2"] for x in effects),
            "mean_abs_residual_mps2": mean(x["residual_mean_abs_mps2"] for x in effects),
            "mean_near_bound_fraction": mean(x["near_bound_fraction"] for x in effects),
        }
        gate["development_pass"] = all(
            (
                gate["benefit_four_of_six"],
                gate["delay_within_1s_per_vehicle"],
                gate["no_material_discharge_loss"],
                gate["zero_collisions"],
            )
        )
        gates[arm] = gate

    passing = [arm for arm, gate in gates.items() if gate["development_pass"]]
    summary = {
        "schema": "idm_residual_checkpoint_summary/1",
        "contract": contract_manifest(),
        "development_only": True,
        "excluded_analytical_pilot_seeds": list(ANALYTICAL_PILOT_SEEDS),
        "record_count": len(rows),
        "missing": missing,
        "paired_effects": paired,
        "candidate_gates": gates,
        "passing_candidates": passing,
        "stage1_development_pass": not missing and bool(passing),
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        default=[],
    )
    parser.add_argument(
        "--constant",
        action="append",
        nargs=2,
        metavar=("NAME", "RESIDUAL_MPS2"),
        default=[],
    )
    parser.add_argument(
        "--rule",
        action="append",
        nargs=2,
        metavar=("NAME", "RULE"),
        default=[],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(4107, 4113)))
    parser.add_argument("--demands", type=int, nargs="+", default=[2200])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-episode-steps", type=int, default=1800)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    seeds = validate_seed_role("development", args.seeds)
    if set(seeds) & set(ANALYTICAL_PILOT_SEEDS):
        raise ValueError("checkpoint evaluation must exclude seeds 4101-4106")
    checkpoints = {name: Path(path).resolve() for name, path in args.checkpoint}
    constants = {name: float(value) for name, value in args.constant}
    rules = {name: value for name, value in args.rule}
    supported_rules = {"reference_le_24", "reference_hysteresis_24_25"}
    if any(value not in supported_rules for value in rules.values()):
        raise ValueError(f"supported diagnostic rules: {sorted(supported_rules)}")
    all_names = (
        [name for name, _path in args.checkpoint]
        + [name for name, _value in args.constant]
        + [name for name, _value in args.rule]
    )
    if "zero" in all_names or len(set(all_names)) != len(all_names):
        raise ValueError("arm names must be unique and cannot be zero")
    if any(value < -0.5 or value > 0.3 for value in constants.values()):
        raise ValueError("constant residuals must remain within [-0.5, 0.3]")
    for path in checkpoints.values():
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        {
            "arm": arm,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "constant": constant,
            "rule": rule,
            "demand": demand,
            "seed": seed,
            "output_dir": str(output_dir),
            "max_episode_steps": args.max_episode_steps,
            "force": args.force,
        }
        for demand in args.demands
        for seed in seeds
        for arm, checkpoint, constant, rule in [
            ("zero", None, None, None),
            *((name, path, None, None) for name, path in checkpoints.items()),
            *((name, None, value, None) for name, value in constants.items()),
            *((name, None, None, value) for name, value in rules.items()),
        ]
    ]
    _atomic_json(
        output_dir / "manifest.json",
        {
            "schema": "idm_residual_checkpoint_manifest/1",
            "contract": contract_manifest(),
            "command": vars(args) | {"output": str(output_dir)},
            "checkpoints": {name: str(path) for name, path in checkpoints.items()},
            "constant_residuals_mps2": constants,
            "diagnostic_rules": rules,
            "created_unix_s": time.time(),
        },
    )
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_case, case): case for case in cases}
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            print(
                f"[{index}/{len(cases)}] {result['status']} {result['arm']} "
                f"m{result['demand']} seed{result['seed']}",
                flush=True,
            )
            build_summary(output_dir, cases)

    summary = build_summary(output_dir, cases)
    print(
        json.dumps(
            {
                "record_count": summary["record_count"],
                "missing_count": len(summary["missing"]),
                "passing_candidates": summary["passing_candidates"],
                "stage1_development_pass": summary["stage1_development_pass"],
                "summary": str(output_dir / "summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
