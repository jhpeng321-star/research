"""Evaluate a TrafficShepherd policy or baseline on a SUMO scenario."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np

from config.constants import RL_SPEED_MODE
from config.defaults import (
    FAIRNESS_R_REF_BY_DENSITY,
    FLOW_Q_REF_BY_DENSITY,
    FLOW_V_REF_BY_DENSITY,
)
from env.core import TrafficShepherdMultiAgentEnv
from evaluation import evaluate_run
from policies import (
    ConstantSpeedPolicy,
    ForcedModePolicy,
    NoOpPolicy,
    Policy,
    RandomPolicy,
    RLlibPolicy,
    RuleBasedMergePolicy,
    RuleBasedShepherdPolicy,
)
from policies.rllib_policy import checkpoint_custom_model_config


DEFAULT_SUMO_ROOT = Path(__file__).parent.parent / "highway_merge_shepherd"


def make_policy(
    policy_name: str,
    checkpoint_path: Optional[str] = None,
    force_mode: Optional[int] = None,
    force_radius: Optional[float] = None,
) -> Policy:
    """Instantiate a policy by name."""
    if policy_name == "no_op":
        return NoOpPolicy()
    if policy_name == "random":
        return RandomPolicy()
    if policy_name == "constant_speed":
        return ConstantSpeedPolicy()
    if policy_name == "rule_based":
        return RuleBasedMergePolicy()
    if policy_name == "rule_shepherd":
        return RuleBasedShepherdPolicy()
    if policy_name == "forced_mode":
        if force_mode is None:
            raise ValueError("--force-mode is required for forced_mode policy")
        return ForcedModePolicy(force_mode, radius=force_radius)
    if policy_name in ("selfish", "rllib", "flat_marl", "shepherd"):
        if checkpoint_path is None:
            raise ValueError(f"--checkpoint is required for {policy_name} policy")
        return RLlibPolicy(checkpoint_path, policy_id="cav_policy")
    raise ValueError(f"Unknown policy: {policy_name}")


def _env_config_for_policy(
    policy: Policy,
    policy_name: str,
    density: str,
    project_root: Path,
    sumo_binary: str,
    cav_penetration: float,
    max_episode_steps: Optional[int],
    hdv_mixture: Optional[list],
    seed: int = 42,
    scenario: str = "merge",
    run_tag_prefix: Optional[str] = None,
) -> dict:
    # Unique per-(policy, scenario, density, seed) output paths: concurrent
    # evaluations sharing the default tripinfos.eval.xml corrupt each other's XML.
    tag_base = policy_name or "policy"
    if run_tag_prefix:
        tag_base = f"{run_tag_prefix}_{tag_base}"
    run_tag = f"{tag_base}.{scenario}.{density}.seed{seed}"
    if policy_name == "forced_mode" and isinstance(policy, ForcedModePolicy):
        run_tag = f"forced{policy.mode}_r{policy.radius}.{scenario}.{density}.seed{seed}"
        if run_tag_prefix:
            run_tag = f"{run_tag_prefix}_{run_tag}"
    if scenario == "bottleneck":
        sumo_config = f"bottleneck/osm.{density}.sumocfg"
    else:
        sumo_config = f"osm.{density}.sumocfg"
    env_config: dict = {
        "sumo_binary": sumo_binary,
        "sumo_config": sumo_config,
        "project_root": project_root,
        "eval_mode": True,
        "eval_tripinfo_output": project_root / f"tripinfos.eval.{run_tag}.xml",
        "eval_fcd_output": project_root / f"fcd.eval.{run_tag}.xml",
        "eval_stats_output": project_root / f"stats.eval.{run_tag}.xml",
        "cav_penetration": cav_penetration,
        "max_episode_steps": max_episode_steps,
        "use_v2_obs": True,
        "centralized_critic": False,
        "reward_mode": "selfish",
        "control_mode": "continuous_acceleration",
        "fairness_r_ref": FAIRNESS_R_REF_BY_DENSITY.get(density, 0.5),
        "flow_q_ref": FLOW_Q_REF_BY_DENSITY.get(density, 0.9),
        "flow_v_ref": FLOW_V_REF_BY_DENSITY.get(density, 27.0),
        # Safe-speed clamp: SUMO clamps RL setSpeed() to safe car-following.
        "rl_speed_mode": RL_SPEED_MODE,
    }
    if hdv_mixture is not None:
        # Explicit mixture: re-enable TraCI-side type assignment, overriding
        # the route file's vTypeDistribution (assign_hdv_types default False).
        env_config["hdv_type_proportions"] = hdv_mixture
        env_config["assign_hdv_types"] = True

    if policy_name in ("no_op", "random", "constant_speed", "forced_mode"):
        # Baseline trap fix: these baselines emit Shepherd mode indices, which
        # are undefined under continuous_acceleration — no_op's 0 means
        # "0 m/s^2" -> setSpeed(current speed), i.e. a speed lock that
        # disables SUMO car-following; random's reset() reads action_space.n
        # off a Box (AttributeError); constant_speed's mode index would be
        # misread as an acceleration. Run all three in shepherd mode
        # (Discrete(5)) where action 0 == non_intervention hands control back
        # to SUMO.
        env_config.update(
            {
                "control_mode": "shepherd",
                "reward_mode": "shepherd",
                "centralized_critic": False,
            }
        )
    elif policy_name == "flat_marl":
        from config.constants import FLAT_MARL_ACCEL_MAX, FLAT_MARL_ACCEL_MIN

        env_config.update(
            {
                "control_mode": "continuous_acceleration",
                "reward_mode": "shepherd",
                "centralized_critic": True,
                "acceleration_min": FLAT_MARL_ACCEL_MIN,
                "acceleration_max": FLAT_MARL_ACCEL_MAX,
            }
        )
    elif policy_name == "shepherd":
        env_config.update(
            {
                "control_mode": "shepherd",
                "reward_mode": "shepherd",
                "centralized_critic": True,
            }
        )
    elif policy_name == "rule_shepherd":
        env_config.update(
            {
                "control_mode": "shepherd",
                "reward_mode": "shepherd",
                "centralized_critic": False,
            }
        )
    elif policy_name == "rllib":
        env_config.update(
            {
                "control_mode": "continuous_acceleration",
                "reward_mode": "shepherd",
                "centralized_critic": True,
            }
        )
    elif policy_name == "selfish":
        env_config.update(
            {
                "control_mode": "continuous_acceleration",
                "reward_mode": "selfish",
                "centralized_critic": False,
            }
        )

    return env_config


def run_policy_evaluation(
    policy: Policy,
    density: str,
    project_root: Path,
    sumo_binary: str = "sumo",
    seed: int = 42,
    max_episode_steps: Optional[int] = None,
    hdv_mixture: Optional[list] = None,
    cav_penetration: float = 0.0,
    policy_name: str = "",
    scenario: str = "merge",
    env_overrides: Optional[dict] = None,
    run_tag_prefix: Optional[str] = None,
) -> dict:
    """Run one evaluation episode and return metrics."""
    env_config = _env_config_for_policy(
        policy,
        policy_name,
        density,
        project_root,
        sumo_binary,
        cav_penetration,
        max_episode_steps,
        hdv_mixture,
        seed=seed,
        scenario=scenario,
        run_tag_prefix=run_tag_prefix,
    )
    # Pass the evaluation seed through to SUMO (--seed) so TraCI runs pair
    # with the seeded pure-SUMO All-HDV baseline runs.
    env_config["sumo_seed"] = seed
    if env_overrides:
        env_config.update(env_overrides)
    env = TrafficShepherdMultiAgentEnv(env_config)
    policy.reset(env, seed=seed)

    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    step_count = 0
    episode_time = 0.0
    done = False

    while not done:
        action_dict = policy.act(obs)
        if action_dict and not getattr(policy, "direct_control", False):
            for cav_id, mode in action_dict.items():
                if isinstance(env.action_space, gym.spaces.Discrete):
                    if not isinstance(mode, int) or not (0 <= mode < env.action_space.n):
                        raise ValueError(
                            f"Invalid action for {cav_id}: {mode}. "
                            f"Expected integer in [0, {env.action_space.n})."
                        )
                elif isinstance(env.action_space, gym.spaces.Tuple):
                    # Hybrid composite: (mode_int, dial) — shape checked in env.
                    if not isinstance(mode, (tuple, list)) or len(mode) < 2:
                        raise ValueError(
                            f"Invalid hybrid action for {cav_id}: {mode}. "
                            f"Expected (mode, dial) tuple."
                        )
                else:
                    # Scalars (Box(1)) and vectors (Box(6) legacy hybrid) both legal.
                    arr = np.asarray(mode, dtype=np.float32).reshape(-1)
                    if not env.action_space.contains(arr):
                        raise ValueError(
                            f"Invalid continuous/hybrid action for {cav_id}: {mode}. "
                            f"Expected shape {env.action_space.shape}."
                        )
        obs, rewards, terminated, truncated, info = env.step(action_dict)
        total_reward += sum(rewards.values())
        step_count += 1
        episode_time = float(info.get("episode_time", 0.0))
        done = terminated.get("__all__", False) or truncated.get("__all__", False)

    env.close()
    policy.close()

    tripinfo_path = env.eval_tripinfo_output
    fcd_path = env.eval_fcd_output
    stats_path = env.eval_stats_output

    if tripinfo_path is None or not tripinfo_path.exists():
        raise RuntimeError(f"Tripinfo output not found: {tripinfo_path}")
    if fcd_path is None or not fcd_path.exists():
        raise RuntimeError(f"FCD output not found: {fcd_path}")
    # Collisions/teleports come from SUMO's statistic-output — the same
    # counters (unique collisions, not per-step recounts) the All-HDV
    # baseline uses — so both sides of the comparison share definitions.
    safety_data_missing = stats_path is None or not stats_path.exists()
    if safety_data_missing:
        print(
            f"WARNING: statistic-output not found ({stats_path}); "
            "collisions/teleports marked as missing for this run",
            file=sys.stderr,
        )

    metrics = evaluate_run(
        tripinfo_path,
        fcd_path,
        stats_path=None if safety_data_missing else stats_path,
        sim_seconds=episode_time or None,
        # TTC threshold 1.5 s (the intended project metric; user steer
        # 2026-08-04, restored from a temporary 1.0 s).
        ttc_threshold=1.5,
    )
    metrics.update(
        {
            "policy": policy.__class__.__name__,
            "density": density,
            "seed": seed,
            "steps": step_count,
            "total_reward": total_reward,
            "safety_data": "missing" if safety_data_missing else "ok",
            "tripinfo_path": str(tripinfo_path),
            "fcd_path": str(fcd_path),
        }
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a TrafficShepherd policy/baseline.")
    parser.add_argument(
        "--policy",
        required=True,
        choices=[
            "no_op",
            "random",
            "constant_speed",
            "rule_based",
            "rule_shepherd",
            "forced_mode",
            "selfish",
            "flat_marl",
            "shepherd",
            "rllib",
        ],
        help="Policy/baseline to evaluate.",
    )
    parser.add_argument(
        "--force-mode",
        type=int,
        default=None,
        choices=[0, 1, 2, 3, 4],
        help="Shepherd mode to force on CAVs (forced_mode policy only).",
    )
    parser.add_argument(
        "--force-radius",
        type=float,
        default=None,
        help="Only force CAVs within this many meters of the merge (default: all CAVs).",
    )
    parser.add_argument(
        "--density",
        required=True,
        choices=["light", "medium", "dense", "hyper"],
        help="Traffic density preset.",
    )
    parser.add_argument(
        "--scenario",
        default="merge",
        choices=["merge", "bottleneck"],
        help="Traffic scenario preset.",
    )
    parser.add_argument(
        "--sumo-root",
        type=Path,
        default=DEFAULT_SUMO_ROOT,
        help="Path to the SUMO project directory.",
    )
    parser.add_argument(
        "--sumo-binary",
        default="sumo",
        help="SUMO binary to use.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for CAV selection and stochastic baselines.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=10000,
        help="Cap episode length (default: 10000, large enough to let all vehicles finish).",
    )
    parser.add_argument(
        "--run-until-empty",
        action="store_true",
        help="Ignore --max-episode-steps and run until the network is empty.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "RLlib checkpoint directory required for selfish, flat_marl, "
            "shepherd, and rllib policies."
        ),
    )
    parser.add_argument(
        "--hdv-mixture",
        type=float,
        nargs=3,
        default=None,
        metavar=("AGGRESSIVE", "NORMAL", "CAUTIOUS"),
        help="Heterogeneous HDV proportions (e.g. 0.1 0.7 0.2).",
    )
    parser.add_argument(
        "--cav-penetration",
        type=float,
        default=0.0,
        help="CAV penetration rate (0.0 for pure HDV baseline).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file to save the metrics.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full metrics as JSON.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional run-tag prefix for unique tripinfo/FCD filenames.",
    )
    args = parser.parse_args()

    if args.run_until_empty:
        args.max_episode_steps = None

    policy = make_policy(
        args.policy,
        checkpoint_path=args.checkpoint,
        force_mode=args.force_mode,
        force_radius=args.force_radius,
    )
    env_overrides = None
    if args.checkpoint:
        custom = checkpoint_custom_model_config(args.checkpoint) or {}
        env_overrides = {
            k: custom[k]
            for k in (
                "max_cavs_critic",
                "max_neighbors",
                "local_obs_dim",
                "action_mask_dim",
            )
            if k in custom
        } or None
        if env_overrides and "action_mask_dim" in env_overrides:
            env_overrides["action_masking"] = int(env_overrides.pop("action_mask_dim")) > 0
        elif env_overrides is not None and "local_obs_dim" in custom:
            # Infer masking from local vs assembler size when only local_obs_dim set.
            pass
    # Encode --tag into output filenames only — do NOT alter policy_name used
    # for env control_mode / critic wiring.
    metrics = run_policy_evaluation(
        policy=policy,
        density=args.density,
        project_root=args.sumo_root.resolve(),
        sumo_binary=args.sumo_binary,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        hdv_mixture=args.hdv_mixture,
        cav_penetration=args.cav_penetration,
        policy_name=args.policy,
        scenario=args.scenario,
        env_overrides=env_overrides,
        run_tag_prefix=args.tag,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

    if args.json:
        print(json.dumps(metrics, indent=2, default=str))
    else:
        mainline_p90 = metrics["travel_time"]["mainline"]["p90"]
        ramp_p90 = metrics["travel_time"]["ramp"]["p90"]
        print(f"mainline P90: {mainline_p90:.2f}s")
        print(f"ramp P90: {ramp_p90:.2f}s")


if __name__ == "__main__":
    main()
