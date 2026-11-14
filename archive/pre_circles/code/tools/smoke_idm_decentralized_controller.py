"""Real-SUMO integration smoke for decentralized structured IDM damping."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from env.core import TrafficShepherdMultiAgentEnv
from train import _build_env_config


ROOT = Path(__file__).resolve().parents[1]


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        sumo_binary="sumo",
        sumo_config="osm.dense.idm_m2200r400.sumocfg",
        sumo_root=str(ROOT / "scenarios" / "idm_transition"),
        density="dense",
        cav_penetration=0.2,
        hdv_mixture=None,
        max_episode_steps=1200,
        max_neighbors=20,
        max_cavs_critic=48,
        baseline="shepherd",
        control_mode="idm_residual_smoothing",
        idm_residual_filter_tau=0.0,
        idm_decentralized_modes=True,
        multi_density=False,
        curriculum_densities=None,
        multi_penetration=False,
        penetration_choices=[0.2],
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


def main() -> None:
    cfg = _build_env_config(_args())
    cfg.update(
        project_root=ROOT / "scenarios" / "idm_transition",
        centralized_critic=False,
        action_masking=True,
    )
    env = TrafficShepherdMultiAgentEnv(cfg)
    result = {
        "schema": "idm_decentralized_controller_smoke/1",
        "seed": 4107,
        "demand": 2200,
        "steps_requested": 300,
        "illegal_action_count": 0,
        "all_zero_mask_count": 0,
        "all_zero_mask_examples": [],
        "command_without_legal_bid_count": 0,
        "candidate_rows": 0,
        "rank_two_or_later_rows": 0,
        "admissible_rows": 0,
        "safety_rejections": Counter(),
        "bid_outcomes": Counter(),
        "command_modes": Counter(),
        "controlled_cav_seconds": 0.0,
        "maximum_continuous_member_steps": 0,
    }
    active_runs: dict[str, int] = {}
    try:
        obs, _ = env.reset(seed=result["seed"])
        result["action_space"] = str(env.action_space)
        result["local_observation_dim"] = env._local_obs_dim
        result["physical_observation_dim"] = env._raw_local_dim
        result["action_mask_dim"] = env._action_mask_dim
        for step in range(result["steps_requested"]):
            masks = {
                vehicle_id: np.asarray(values[-5:], dtype=np.float32)
                for vehicle_id, values in obs.items()
            }
            actions: dict[str, int] = {}
            for vehicle_id, mask in masks.items():
                legal = np.flatnonzero(mask > 0.5)
                if not legal.size:
                    result["all_zero_mask_count"] += 1
                    if len(result["all_zero_mask_examples"]) < 10:
                        result["all_zero_mask_examples"].append(
                            {
                                "step": step,
                                "vehicle_id": vehicle_id,
                                "mask": mask.tolist(),
                                "observation_length": len(obs[vehicle_id]),
                            }
                        )
                if 2 in legal:
                    action = 2
                elif legal.size:
                    action = int(legal[0])
                else:
                    action = 0
                actions[vehicle_id] = action
                result["illegal_action_count"] += int(mask[action] <= 0.5)

            next_obs, _, terminated, truncated, info = env.step(actions)
            diag = info.get("idm_residual", {})
            result["controlled_cav_seconds"] += float(
                diag.get("controlled_cav_seconds", 0.0)
            )
            candidates = diag.get("candidate_admissions", [])
            result["candidate_rows"] += len(candidates)
            result["rank_two_or_later_rows"] += sum(
                int(int(row.get("rank", 0)) >= 2) for row in candidates
            )
            result["admissible_rows"] += sum(
                int(bool(row.get("admissible", False))) for row in candidates
            )
            for row in candidates:
                for reason in row.get("reasons", ()):
                    result["safety_rejections"][str(reason)] += 1
            for row in diag.get("local_bids", []):
                result["bid_outcomes"][str(row.get("outcome"))] += 1

            current = set(diag.get("active_members", {}).values())
            for vehicle_id in list(active_runs):
                if vehicle_id not in current:
                    result["maximum_continuous_member_steps"] = max(
                        result["maximum_continuous_member_steps"],
                        active_runs.pop(vehicle_id),
                    )
            for vehicle_id in current:
                active_runs[vehicle_id] = active_runs.get(vehicle_id, 0) + 1
            for command in diag.get("commands", []):
                vehicle_id = str(command["vehicle_id"])
                mode = int(command.get("mode_index", -1))
                result["command_modes"][str(mode)] += 1
                prior_mask = masks.get(vehicle_id, np.zeros(5, dtype=np.float32))
                result["command_without_legal_bid_count"] += int(
                    mode < 0 or mode >= 5 or prior_mask[mode] <= 0.5
                )
            obs = {
                vehicle_id: values
                for vehicle_id, values in next_obs.items()
                if not terminated.get(vehicle_id, False)
                and not truncated.get(vehicle_id, False)
            }
            if terminated.get("__all__") or truncated.get("__all__"):
                break
        result["steps_completed"] = step + 1
        if active_runs:
            result["maximum_continuous_member_steps"] = max(
                result["maximum_continuous_member_steps"], max(active_runs.values())
            )
        result["pass"] = bool(
            result["action_space"] == "Discrete(5)"
            and result["action_mask_dim"] == 5
            and result["illegal_action_count"] == 0
            and result["command_without_legal_bid_count"] == 0
            and result["candidate_rows"] > 0
            and result["rank_two_or_later_rows"] > 0
            and result["admissible_rows"] > 0
            and result["controlled_cav_seconds"] > 0.0
            and result["command_modes"] == Counter({"2": result["controlled_cav_seconds"]})
            and result["maximum_continuous_member_steps"] >= 16
        )
    finally:
        env.close()

    for key in ("safety_rejections", "bid_outcomes", "command_modes"):
        result[key] = dict(sorted(result[key].items()))
    output = ROOT / "runs" / "eval" / "idm_decentralized_controller_smoke_v1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
