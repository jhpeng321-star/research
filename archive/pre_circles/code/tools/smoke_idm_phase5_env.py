"""Real-SUMO contract smoke for Phase-5 IDM context and residual filtering."""

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
        sumo_config="osm.dense.idm_m2700r400.sumocfg",
        sumo_root=str(Path(__file__).resolve().parents[1] / "scenarios" / "idm_transition"),
        density="dense",
        cav_penetration=0.2,
        hdv_mixture=None,
        max_episode_steps=1200,
        max_neighbors=20,
        max_cavs_critic=48,
        baseline="shepherd",
        control_mode="idm_residual_smoothing",
        idm_residual_filter_tau=5.0,
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
    args = _args()
    cfg = _build_env_config(args)
    cfg["project_root"] = Path(args.sumo_root)
    cfg["centralized_critic"] = False
    env = TrafficShepherdMultiAgentEnv(cfg)
    result = {
        "schema": "idm_phase5_env_smoke/1",
        "seed": 4112,
        "steps_requested": 300,
        "preview_identity_failures": [],
        "nonselected_reward_failures": [],
        "context_nonzero_samples": 0,
        "requested_residuals": [],
        "executed_residuals": [],
        "controlled_cav_seconds": 0.0,
    }
    try:
        obs, _ = env.reset(seed=result["seed"])
        names = env._obs_assembler.context_feature_names
        context_start = env._obs_assembler.obs_dim - len(names)
        indexes = {name: context_start + i for i, name in enumerate(names)}
        result["context_names"] = list(names)
        result["observation_dim"] = env.observation_space.shape[0]
        prior_selected = set()
        for step in range(result["steps_requested"]):
            gated = {
                agent_id
                for agent_id, value in obs.items()
                if value[indexes["idm_absorber_gate"]] > 0.5
            }
            for agent_id in prior_selected & set(obs):
                values = obs[agent_id]
                controller = [
                    values[indexes[name]]
                    for name in (
                        "idm_nominal_acceleration",
                        "idm_filtered_reference",
                        "idm_reference_error",
                        "idm_projection_delta",
                    )
                ]
                result["context_nonzero_samples"] += int(
                    any(abs(float(value)) > 1e-7 for value in controller)
                )
            actions = {agent_id: np.array([0.1], dtype=np.float32) for agent_id in obs}
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
            prior_selected = selected
            result["controlled_cav_seconds"] += float(diag.get("controlled_cav_seconds", 0.0))
            for command in diag.get("commands", []):
                result["requested_residuals"].append(float(command["requested_residual_mps2"]))
                result["executed_residuals"].append(float(command["residual_mps2"]))
            if terminated.get("__all__") or truncated.get("__all__"):
                break
        result["steps_completed"] = step + 1
        requested = result["requested_residuals"]
        executed = result["executed_residuals"]
        result["pass"] = bool(
            not result["preview_identity_failures"]
            and not result["nonselected_reward_failures"]
            and result["context_nonzero_samples"] > 0
            and requested
            and all(abs(value - 0.1) < 1e-6 for value in requested)
            and all(0.0 < value <= 0.1 + 1e-9 for value in executed)
            and min(executed) < 0.05
        )
    finally:
        env.close()
    for key in ("requested_residuals", "executed_residuals"):
        values = result.pop(key)
        result[key + "_summary"] = {
            "count": len(values),
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
            "mean": float(np.mean(values)) if values else None,
        }
    out = Path("runs/eval/idm_phase5_env_smoke_v1.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
