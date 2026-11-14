"""Audit that every open binary mask is an actuator-consumed IDM decision."""

from __future__ import annotations

import argparse
import json
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


def run(seed: int, steps: int) -> dict:
    cfg = _build_env_config(_args())
    cfg.update(
        project_root=ROOT / "scenarios" / "idm_transition",
        centralized_critic=False,
        action_masking=True,
        sumo_seed=seed,
    )
    env = TrafficShepherdMultiAgentEnv(cfg)
    mismatches = []
    open_count = bid_count = selected_count = 0
    try:
        obs, _ = env.reset(seed=seed)
        for step in range(steps):
            masks = {
                vehicle_id: np.asarray(values[-2:], dtype=np.float32)
                for vehicle_id, values in obs.items()
            }
            open_ids = {
                vehicle_id
                for vehicle_id, mask in masks.items()
                if bool(np.all(mask > 0.5))
            }
            actions = {}
            for vehicle_id, mask in masks.items():
                legal = np.flatnonzero(mask > 0.5)
                if len(legal) == 2:
                    # Exercise both arms deterministically without changing the
                    # eligibility contract being audited.
                    actions[vehicle_id] = int((step + sum(map(ord, vehicle_id))) % 2)
                else:
                    actions[vehicle_id] = int(legal[0]) if len(legal) else 0
            next_obs, _, terminated, truncated, info = env.step(actions)
            bids = list((info.get("idm_residual") or {}).get("local_bids") or [])
            admissible_ids = {
                str(row["vehicle_id"])
                for row in bids
                if bool(row.get("admissible", False))
            }
            open_count += len(open_ids)
            bid_count += len(admissible_ids)
            selected_count += sum(row.get("outcome") == "selected" for row in bids)
            wrong_modes = [
                {
                    "vehicle_id": str(row["vehicle_id"]),
                    "mask": masks.get(str(row["vehicle_id"]), np.zeros(2)).tolist(),
                    "action": actions.get(str(row["vehicle_id"])),
                    "bid_mode": int(row.get("mode_index", -1)),
                }
                for row in bids
                if bool(row.get("admissible", False))
                and int(row.get("mode_index", -1))
                != int(actions.get(str(row["vehicle_id"]), -1))
            ]
            if open_ids != admissible_ids or wrong_modes:
                mismatches.append(
                    {
                        "step": step,
                        "open_only": sorted(open_ids - admissible_ids),
                        "bid_only": sorted(admissible_ids - open_ids),
                        "wrong_modes": wrong_modes,
                    }
                )
                if len(mismatches) >= 20:
                    break
            obs = {
                vehicle_id: value
                for vehicle_id, value in next_obs.items()
                if not terminated.get(vehicle_id, False)
                and not truncated.get(vehicle_id, False)
            }
            if terminated.get("__all__") or truncated.get("__all__"):
                break
    finally:
        env.close()
    return {
        "schema": "idm_decentralized_mask_action_alignment/1",
        "seed": seed,
        "steps_requested": steps,
        "open_policy_decisions": open_count,
        "admissible_actuator_decisions": bid_count,
        "selected_activations": selected_count,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "pass": bool(open_count > 0 and open_count == bid_count and not mismatches),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=4107)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.seed, args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
