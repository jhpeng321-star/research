"""Paired development evaluation for the clean decentralized IDM smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from env.core import TrafficShepherdMultiAgentEnv
from evaluation import evaluate_run
from experiments.io import atomic_write_json
from policies import Policy
from policies.rllib_policy import RLlibPolicy
from train import _build_env_config


ROOT = Path(__file__).resolve().parents[1]


class AlwaysAbstain(Policy):
    def act(self, obs_dict):
        return {vehicle_id: 0 for vehicle_id in obs_dict}


class AlwaysActivateEligible(Policy):
    def __init__(self) -> None:
        self.raw_local_dim = 0

    def reset(self, env, seed=None) -> None:
        del seed
        self.raw_local_dim = int(env._raw_local_dim)

    def act(self, obs_dict):
        actions = {}
        for vehicle_id, observation in obs_dict.items():
            mask = np.asarray(
                observation[self.raw_local_dim : self.raw_local_dim + 2],
                dtype=np.float32,
            )
            actions[vehicle_id] = 1 if len(mask) == 2 and mask[1] > 0.5 else 0
        return actions


class AnalyticalZeroResidual(Policy):
    def act(self, obs_dict):
        return {vehicle_id: 0.0 for vehicle_id in obs_dict}


def _args_template() -> argparse.Namespace:
    return argparse.Namespace(
        sumo_binary="sumo",
        sumo_config="osm.dense.idm_m2200r400.sumocfg",
        sumo_root=str(ROOT / "scenarios" / "idm_transition"),
        scenario_refs=str(ROOT / "runs" / "eval" / "idm_phase5_reward_refs_v1.json"),
        density="dense",
        cav_penetration=0.10,
        hdv_mixture=None,
        max_episode_steps=900,
        max_neighbors=20,
        max_cavs_critic=32,
        baseline="shepherd",
        control_mode="idm_residual_smoothing",
        idm_residual_filter_tau=0.0,
        idm_decentralized_modes=True,
        idm_decentralized_action_set="activation_speed_ceiling_v1",
        multi_density=False,
        curriculum_densities=None,
        multi_penetration=False,
        penetration_choices=None,
        use_meter_load=False,
        use_cav_count=False,
        use_wave_signal=False,
        use_health_obs=False,
        use_dv_obs=False,
        use_margin_obs=False,
        pruned_modes=False,
        enable_jad=False,
        enable_merge_match=False,
        enable_wave_damp=False,
        paired_refs=None,
        reward_weights=None,
    )


def _environment_config(
    *, seed: int, output: Path, label: str, decentralized: bool, centralized: bool
) -> dict:
    cfg = _build_env_config(_args_template())
    cfg.update(
        project_root=ROOT / "scenarios" / "idm_transition",
        sumo_seed=seed,
        eval_mode=True,
        centralized_critic=centralized,
        max_cavs_critic=32,
        action_masking=decentralized,
        idm_residual_decentralized_damping_enabled=decentralized,
        idm_residual_decentralized_action_set="activation_speed_ceiling_v1",
        eval_tripinfo_output=output / "xml" / f"{label}.seed{seed}.tripinfo.xml",
        eval_fcd_output=output / "xml" / f"{label}.seed{seed}.fcd.xml",
        eval_stats_output=output / "xml" / f"{label}.seed{seed}.stats.xml",
    )
    if not decentralized:
        # The analytical baseline uses the original selected-absorber p30 law
        # with an exact zero residual and no learned binary activation gate.
        cfg["action_masking"] = False
    return cfg


def _validate_inputs(run_dir: Path, checkpoints: dict[str, Path], seeds: list[int]) -> None:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"training run directory does not exist: {run_dir}")
    missing = [str(path) for path in checkpoints.values() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "required smoke checkpoints are missing: " + ", ".join(missing)
        )
    if not seeds:
        raise ValueError("at least one paired development seed is required")
    invalid = sorted(set(seeds) - set(range(4107, 4121)))
    if invalid:
        raise ValueError(
            "clean-smoke checkpoint selection is development-only; expected seeds "
            f"4107-4120, received {invalid}"
        )


def _run_episode(
    *, policy: Policy, label: str, seed: int, output: Path, decentralized: bool,
    centralized: bool
) -> dict[str, Any]:
    cfg = _environment_config(
        seed=seed,
        output=output,
        label=label,
        decentralized=decentralized,
        centralized=centralized,
    )
    for path_key in ("eval_tripinfo_output", "eval_fcd_output", "eval_stats_output"):
        Path(cfg[path_key]).parent.mkdir(parents=True, exist_ok=True)
    env = TrafficShepherdMultiAgentEnv(cfg)
    policy.reset(env, seed=seed)
    step_count = 0
    treatment = {
        "eligible_decisions": 0,
        "activate_decisions": 0,
        "abstain_decisions": 0,
        "winning_activations": 0,
        "completed_treatments": 0,
        "safety_aborts": 0,
        "controlled_cav_seconds": 0.0,
    }
    try:
        obs, _ = env.reset(seed=seed)
        done = False
        episode_time = 0.0
        while not done:
            actions = policy.act(obs)
            obs, _, terminated, truncated, info = env.step(actions)
            diag = info.get("idm_training_metrics") or {}
            treatment["eligible_decisions"] += int(diag.get("eligible_decision_count", 0))
            treatment["activate_decisions"] += int(diag.get("executed_activate_count", 0))
            treatment["abstain_decisions"] += int(diag.get("executed_abstain_count", 0))
            treatment["winning_activations"] += int(diag.get("winning_activation_count", 0))
            treatment["completed_treatments"] += int(
                diag.get("treatment_completion_count", 0)
            )
            treatment["safety_aborts"] += int(diag.get("safety_abort_count", 0))
            treatment["controlled_cav_seconds"] += float(
                diag.get("controlled_cav_seconds", 0.0)
            )
            step_count += 1
            episode_time = float(info.get("episode_time", 0.0))
            done = bool(terminated.get("__all__") or truncated.get("__all__"))
    finally:
        env.close()
        policy.close()
    metrics = evaluate_run(
        Path(cfg["eval_tripinfo_output"]),
        Path(cfg["eval_fcd_output"]),
        stats_path=Path(cfg["eval_stats_output"]),
        sim_seconds=episode_time,
        ttc_threshold=1.5,
    )
    return {
        "label": label,
        "seed": seed,
        "steps": step_count,
        "metrics": metrics,
        "treatment": treatment,
    }


def _endpoint(row: dict) -> dict:
    metrics = row["metrics"]
    return {
        "hard_braking": int(metrics["hard_braking"]["total_events"]),
        "low_ttc": int(metrics["ttc"]["total_events"]),
        "collisions": int(metrics.get("collisions") or 0),
        "teleports": int(metrics.get("teleports") or 0),
        "mean_time_loss": float(metrics["mean_time_loss"]),
        "arrived": int(metrics["arrived"]),
        "throughput": float(metrics["throughput_veh_per_hour"]),
    }


def _summarize(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["label"], []).append(_endpoint(row))
    return {
        label: {
            key: mean(record[key] for record in records)
            for key in records[0]
        }
        for label, records in grouped.items()
    }


def _checkpoint_gate(summary: dict, checkpoint: str, comparators: list[str]) -> dict:
    candidate = summary[checkpoint]
    by_comparator = {}
    for comparator in comparators:
        baseline = summary[comparator]
        safety_improvement = bool(
            candidate["hard_braking"] <= baseline["hard_braking"] - 1.0
            or candidate["low_ttc"] <= baseline["low_ttc"] - 1.0
        )
        by_comparator[comparator] = {
            "zero_added_collisions": candidate["collisions"] <= baseline["collisions"],
            "zero_added_teleports": candidate["teleports"] <= baseline["teleports"],
            "ttc_noninferior": candidate["low_ttc"] <= baseline["low_ttc"],
            "delay_within_1s_per_vehicle": (
                candidate["mean_time_loss"] <= baseline["mean_time_loss"] + 1.0
            ),
            "arrivals_within_one": candidate["arrived"] >= baseline["arrived"] - 1.0,
            "physical_safety_improvement": safety_improvement,
        }
        by_comparator[comparator]["pass"] = all(
            by_comparator[comparator].values()
        )
    return {
        "comparators": by_comparator,
        "pass": all(row["pass"] for row in by_comparator.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[4107, 4109, 4110])
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output = args.output.resolve()
    checkpoints = {
        "checkpoint_0": run_dir / "checkpoints" / "step_0",
        "checkpoint_2000": run_dir / "checkpoints" / "iter_2",
        "checkpoint_5000": run_dir / "checkpoints" / "iter_5",
        "checkpoint_10000": run_dir / "checkpoints" / "iter_10",
    }
    _validate_inputs(run_dir, checkpoints, args.seeds)
    arms: list[tuple[str, Any, bool, bool]] = [
        ("always_abstain", AlwaysAbstain, True, False),
        ("always_activate", AlwaysActivateEligible, True, False),
        ("analytical_zero_residual", AnalyticalZeroResidual, False, False),
        *[
            (
                label,
                lambda path=path: RLlibPolicy(str(path), explore=False),
                True,
                True,
            )
            for label, path in checkpoints.items()
        ],
    ]
    rows = []
    for label, factory, decentralized, centralized in arms:
        for seed in args.seeds:
            row_path = output / "rows" / f"{label}.seed{seed}.json"
            if row_path.exists():
                rows.append(json.loads(row_path.read_text(encoding="utf-8")))
                continue
            policy = factory()
            row = _run_episode(
                policy=policy,
                label=label,
                seed=seed,
                output=output,
                decentralized=decentralized,
                centralized=centralized,
            )
            row_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(row_path, row)
            rows.append(row)
            print(f"completed {label} seed {seed}", flush=True)
    summary = _summarize(rows)
    comparators = ["always_abstain", "always_activate", "analytical_zero_residual"]
    gates = {
        label: _checkpoint_gate(summary, label, comparators)
        for label in checkpoints
    }
    payload = {
        "schema": "idm_decentralized_clean_smoke_evaluation/1",
        "development_only": True,
        "seeds": args.seeds,
        "run_dir": str(run_dir),
        "summary": summary,
        "checkpoint_gates": gates,
        "smoke_pass": any(row["pass"] for row in gates.values()),
    }
    atomic_write_json(output / "summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
