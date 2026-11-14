"""Real-SUMO contract smoke for the Phase-4 IDM residual interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from env.core import TrafficShepherdMultiAgentEnv
from train import _build_env_config


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        sumo_binary="sumo",
        sumo_config="osm.dense.idm_m2200r400.sumocfg",
        sumo_root=str(Path(__file__).resolve().parents[1] / "scenarios" / "idm_transition"),
        density="dense",
        cav_penetration=0.2,
        hdv_mixture=None,
        max_episode_steps=1200,
        max_neighbors=20,
        max_cavs_critic=48,
        baseline="shepherd",
        control_mode="idm_residual_smoothing",
        multi_density=False,
        curriculum_densities=None,
        multi_penetration=False,
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
    cfg["project_root"] = Path(_args().sumo_root)
    cfg["centralized_critic"] = False
    env = TrafficShepherdMultiAgentEnv(cfg)
    result = {
        "schema": "idm_phase4_env_smoke/1",
        "seed": 4102,
        "steps_requested": 240,
        "preview_identity_failures": [],
        "nonselected_reward_failures": [],
        "selected_steps": 0,
        "controlled_cav_seconds": 0.0,
        "max_active_cavs": 0,
        "residual_abs_max": 0.0,
    }
    try:
        obs, _info = env.reset(seed=4102)
        result["observation_dim"] = env.observation_space.shape[0]
        for step in range(result["steps_requested"]):
            result["max_active_cavs"] = max(result["max_active_cavs"], len(obs))
            gated = {agent_id for agent_id, value in obs.items() if value[-1] > 0.5}
            actions = {agent_id: np.array([0.0], dtype=np.float32) for agent_id in obs}
            obs, rewards, terminated, truncated, info = env.step(actions)
            diag = info.get("idm_residual", {})
            selected = set(diag.get("active_members", {}).values())
            if gated != selected:
                result["preview_identity_failures"].append(
                    {
                        "step": step,
                        "gated": sorted(gated),
                        "selected": sorted(selected),
                    }
                )
            for agent_id, reward in rewards.items():
                if agent_id in actions and agent_id not in selected and reward != 0.0:
                    result["nonselected_reward_failures"].append(
                        {
                            "step": step,
                            "agent_id": agent_id,
                            "reward": reward,
                        }
                    )
            commands = diag.get("commands", [])
            if commands:
                result["selected_steps"] += 1
            result["controlled_cav_seconds"] += float(diag.get("controlled_cav_seconds", 0.0))
            for command in commands:
                result["residual_abs_max"] = max(
                    result["residual_abs_max"], abs(float(command["residual_mps2"]))
                )
            if terminated.get("__all__") or truncated.get("__all__"):
                result["ended_early_at_step"] = step + 1
                break
        result["steps_completed"] = step + 1
        result["pass"] = bool(
            not result["preview_identity_failures"]
            and not result["nonselected_reward_failures"]
            and result["residual_abs_max"] == 0.0
        )
    finally:
        env.close()
    out = Path("runs/eval/idm_phase4_env_smoke_v1.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result.get("pass"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
