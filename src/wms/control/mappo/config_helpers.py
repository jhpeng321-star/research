"""Config -> trainer wiring for the MAPPO CLI (lazy dotted-name resolution)."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Mapping

from wms.control.contract import RewardWeights

from .ppo import PPOConfig
from .rollout import resolve_name
from .trainer import MappoConfig, MappoTrainer, RolloutConfig


def _filtered(cls, section: Mapping) -> dict:
    names = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in section.items() if k in names}


def merge_override(cfg: dict, smoke: Mapping) -> dict:
    """Apply flat [smoke] overrides onto the parsed config sections."""
    ppo_keys = {f.name for f in dataclasses.fields(PPOConfig)}
    rollout_keys = {f.name for f in dataclasses.fields(RolloutConfig)}
    policy_keys = {
        "actor",
        "actor_kwargs",
        "group_encoder",
        "group_encoder_kwargs",
        "critic",
        "obs_builder",
        "team_reward",
        "safety",
    }
    for key, value in dict(smoke).items():
        if key == "env_name":
            cfg.setdefault("env", {})["name"] = value
        elif key == "env_kwargs":
            cfg.setdefault("env", {})["kwargs"] = dict(value)
        elif key in policy_keys:
            cfg.setdefault("policy", {})[key] = value
        elif key in ppo_keys:
            cfg.setdefault("mappo", {})[key] = value
        elif key in rollout_keys:
            cfg.setdefault("rollout", {})[key] = value
        else:
            raise ValueError(f"unknown [smoke] key: {key!r}")
    return cfg


def config_from_dict(cfg: Mapping) -> MappoConfig:
    ppo = PPOConfig(**_filtered(PPOConfig, cfg.get("mappo", {})))
    raw_rollout = dict(cfg.get("rollout", {}))
    if "penetrations" in raw_rollout:
        raw_rollout["penetrations"] = tuple(raw_rollout["penetrations"])
    if "demands_vph" in raw_rollout:
        raw_rollout["demands_vph"] = tuple(int(d) for d in raw_rollout["demands_vph"])
    rollout = RolloutConfig(**_filtered(RolloutConfig, raw_rollout))
    if rollout.max_env_steps == 0:
        rollout.max_env_steps = None
    return MappoConfig(ppo=ppo, rollout=rollout)


def build_trainer(
    cfg: Mapping, *, workers: int, out_dir: Path | str, config_path: str = "", smoke: bool = False
) -> MappoTrainer:
    env_section = dict(cfg.get("env", {}))
    policy_section = dict(cfg.get("policy", {}))
    reward_weights = RewardWeights(**dict(cfg.get("reward_weights", {})))
    env_name = env_section.get("name", "fake")
    env_kwargs = dict(env_section.get("kwargs", {}))
    obs_name = policy_section.get("obs_builder", "fake_obs")
    reward_name = policy_section.get("team_reward", "fake_reward")
    safety_name = policy_section.get("safety", "fake_safety")
    actor_name = policy_section.get("actor", "fake_actor")
    encoder_name = policy_section.get("group_encoder", "fake_encoder")
    critic_name = policy_section.get("critic", "fake_critic")
    actor_kwargs = dict(policy_section.get("actor_kwargs", {}))
    encoder_kwargs = dict(policy_section.get("group_encoder_kwargs", {}))
    meta = {
        "names": {
            "env_name": env_name,
            "obs_builder_name": obs_name,
            "team_reward_name": reward_name,
            "safety_name": safety_name,
            "actor_name": actor_name,
            "group_encoder_name": encoder_name,
        },
        "env_kwargs": env_kwargs,
        "actor_kwargs": actor_kwargs,
        "encoder_kwargs": encoder_kwargs,
        "reward_weights": dataclasses.asdict(reward_weights),
        "config_path": config_path,
        "smoke": smoke,
    }
    return MappoTrainer(
        out_dir=Path(out_dir),
        config=config_from_dict(cfg),
        env=resolve_name(env_name)(**env_kwargs),
        obs_builder=resolve_name(obs_name),
        group_encoder=resolve_name(encoder_name)(**encoder_kwargs),
        actor=resolve_name(actor_name)(**actor_kwargs),
        critic=resolve_name(critic_name)(),
        safety_layer=resolve_name(safety_name),
        team_reward=resolve_name(reward_name),
        weights=reward_weights,
        meta=meta,
        workers=workers,
    )
