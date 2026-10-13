"""Config (TOML dict) -> MappoConfig / MappoTrainer wiring for the MAPPO CLI.

The networks, observation builder, RSU field, lane rules code, safety layer and team reward
are NOT configurable: MAPPO uses the C4 ones (fair comparison).  Only the environment is
named (``traffic_shepherd.c4.CavSumoEnv`` or ``fake``).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Mapping

from wms.control.contract import LEGACY_ACCEL_MAX_MPS2, RewardWeights
from wms.control.lanes import LaneRules

from .agent import build_networks
from .ppo import PPOConfig
from .trainer import MappoConfig, MappoTrainer, RolloutConfig

_TUPLE_FLOAT = ("penetrations", "warmup_choices_s")


def _filtered(cls, section: Mapping) -> dict:
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(section) - names
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return dict(section)


def merge_override(cfg: dict, override: Mapping) -> dict:
    """Apply a flat override table (e.g. ``[smoke]``) onto the parsed config sections."""
    ppo_keys = {f.name for f in dataclasses.fields(PPOConfig)}
    rollout_keys = {f.name for f in dataclasses.fields(RolloutConfig)}
    lane_keys = {f.name for f in dataclasses.fields(LaneRules)}
    for key, value in dict(override).items():
        if key == "env_name":
            cfg.setdefault("env", {})["name"] = value
        elif key == "env_kwargs":
            cfg.setdefault("env", {})["kwargs"] = dict(value)
        elif key in ppo_keys:
            cfg.setdefault("mappo", {})[key] = value
        elif key in rollout_keys:
            cfg.setdefault("rollout", {})[key] = value
        elif key in lane_keys:
            cfg.setdefault("lanes", {})[key] = value
        else:
            raise ValueError(f"unknown override key: {key!r}")
    return cfg


def config_from_dict(cfg: Mapping) -> MappoConfig:
    ppo_raw = dict(cfg.get("mappo", {}))
    if ppo_raw.get("target_kl") == 0:
        ppo_raw["target_kl"] = None
    ppo = PPOConfig(**_filtered(PPOConfig, ppo_raw))
    raw = dict(cfg.get("rollout", {}))
    for key in _TUPLE_FLOAT:
        if key in raw:
            raw[key] = tuple(float(x) for x in raw[key])
    if "demands_vph" in raw:
        raw["demands_vph"] = tuple(int(d) for d in raw["demands_vph"])
    rollout = RolloutConfig(**_filtered(RolloutConfig, raw))
    if not rollout.max_env_steps:
        rollout.max_env_steps = None
    lanes = LaneRules.from_mapping(cfg.get("lanes"))
    return MappoConfig(ppo=ppo, rollout=rollout, lane_rules=lanes)


def build_trainer(cfg: Mapping, *, workers: int, out_dir: Path | str, config_path: str = "",
                  smoke: bool = False) -> MappoTrainer:
    env_section = dict(cfg.get("env", {}))
    weights = RewardWeights(**dict(cfg.get("reward_weights", {})))
    env_name = env_section.get("name", "fake")
    env_kwargs = dict(env_section.get("kwargs", {}))
    out_dir = Path(out_dir)
    env_spec = {"env_name": env_name, "env_kwargs": env_kwargs,
                "cleanup_scenarios": bool(env_section.get("cleanup_scenarios", True))}
    if env_name != "fake":
        env_spec["scenario_root"] = str((out_dir / "scenarios").resolve())
    # Actor action cap (ACCELCAP_NOTES.md): LEGACY 1.5 unless ``[policy] accel_max_mps2`` is set
    # (a resumed / --init-from checkpoint brings its own stored bound).
    policy_section = dict(cfg.get("policy", {}))
    unknown = set(policy_section) - {"accel_max_mps2"}
    if unknown:
        raise ValueError(f"unknown [policy] keys: {sorted(unknown)}")
    accel_max = float(policy_section.get("accel_max_mps2", LEGACY_ACCEL_MAX_MPS2))
    policy, encoder, critic = build_networks(accel_max=accel_max)
    meta = {"env": env_spec, "reward_weights": dataclasses.asdict(weights),
            "config_path": config_path, "smoke": smoke,
            "actor": f"wms.control.policy.CavPolicy(lane_head=True, accel_max={accel_max})",
            "accel_max_mps2": accel_max,
            "group_encoder": "wms.control.policy.RsuGroupEncoder",
            "critic": "wms.control.policy.TeamCritic",
            "reward": "wms.control.reward.team_reward (per physics step, summed per control step)"}
    return MappoTrainer(out_dir=out_dir, config=config_from_dict(cfg), policy=policy,
                        group_encoder=encoder, critic=critic, env_spec=env_spec,
                        weights=weights, meta=meta, workers=workers)
