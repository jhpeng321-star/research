"""Episode collection and vectorised SUMO workers for the C4 MAPPO baseline.

Serial collection is dependency-injected (the trainer's live actor/critic
objects).  The worker path keeps ONE SUMO env alive per process
(``multiprocessing`` Pool with an initializer, spawn context) instead of the
archived pilot's one subprocess per episode
(``archive/circles_era/traffic_shepherd/ppo_train.py``); actor/group-encoder
weights travel per job as pickled state dicts, so no checkpoint-file races.
All partner modules are resolved lazily from dotted names, which is exactly the
path the CLI and the workers use.
"""

from __future__ import annotations

import importlib
from typing import Mapping

import torch
from torch import Tensor

from wms.control.contract import (
    CONTROL_DT_S,
    CavEnv,
    ObservationBuilder,
    Policy,
    RewardWeights,
    SafetyLayer,
    TeamReward,
)

from .buffer import EpisodeData

FAKE_NAMES = {
    "fake": "wms.control.mappo.fakeenv.FakeCavEnv",
    "fake_obs": "wms.control.mappo.fakeenv.fake_build_observations",
    "fake_reward": "wms.control.mappo.fakeenv.fake_team_reward",
    "fake_safety": "wms.control.mappo.fakeenv.fake_safe_accel",
    "fake_actor": "wms.control.mappo.fakeenv.FakeActor",
    "fake_encoder": "wms.control.mappo.fakeenv.FakeGroupEncoder",
    "fake_critic": "wms.control.mappo.fakeenv.FakeTeamCritic",
}


def resolve_name(name: str):
    """Dotted-name lazy import with ``fake*`` shortcuts (never imports partner
    modules at package import time)."""
    name = FAKE_NAMES.get(name, name)
    module_name, _, attr = name.rpartition(".")
    if not module_name:
        raise ValueError(f"not a dotted name: {name!r}")
    return getattr(importlib.import_module(module_name), attr)


def collect_episode(
    env: CavEnv,
    *,
    obs_builder: ObservationBuilder,
    group_encoder,
    actor: Policy,
    safety_layer: SafetyLayer,
    team_reward: TeamReward,
    weights: RewardWeights,
    seed: int,
    demand_vph: int,
    penetration: float,
    warmup_s: float,
    horizon_s: float,
) -> EpisodeData:
    """Run one team episode of ``int(horizon_s / CONTROL_DT_S)`` control steps."""
    state = env.reset(
        seed=seed,
        demand_vph=demand_vph,
        penetration=penetration,
        warmup_s=warmup_s,
        horizon_s=horizon_s,
    )
    steps = int(round(horizon_s / CONTROL_DT_S))
    states: list[Mapping[str, Tensor]] = []
    actions, logps, masks, rewards, costs = [], [], [], [], []
    terminated = False
    for _ in range(steps):
        obs = obs_builder(state)
        group_signal = group_encoder(obs)
        accel, logp = actor(obs, group_signal)
        executed = safety_layer(state, accel)
        next_state, info = env.step(executed)
        row = team_reward(state, next_state, executed, weights)
        states.append(state)
        actions.append(accel[0].detach().clone())
        logps.append(logp[0].detach().clone())
        masks.append(obs.cav_mask[0].detach().clone())
        rewards.append(float(row["reward"].detach()[0]))
        costs.append(float(row["cost_ttc"].detach()[0]))
        terminated = bool(info.get("terminated", False))
        state = next_state
        if terminated:
            break
    states.append(state)
    return EpisodeData(
        states=states,
        actions=torch.stack(actions) if actions else torch.zeros(0, state["v_mps"].shape[1]),
        logp=torch.stack(logps) if logps else torch.zeros(0, state["v_mps"].shape[1]),
        cav_mask=torch.stack(masks).to(torch.bool)
        if masks
        else torch.zeros(0, state["v_mps"].shape[1], dtype=torch.bool),
        rewards=torch.tensor(rewards, dtype=torch.float32),
        costs=torch.tensor(costs, dtype=torch.float32),
        terminated=terminated,
        meta={
            "seed": seed,
            "demand_vph": demand_vph,
            "penetration": penetration,
            "warmup_s": warmup_s,
            "horizon_s": horizon_s,
        },
    )


def build_rollout_deps(spec: Mapping) -> dict:
    """Build {env, obs_builder, group_encoder, actor, safety_layer, team_reward,
    weights} from a picklable spec of dotted names + kwargs (worker side)."""
    weights = RewardWeights(**spec.get("reward_weights", {}))
    return {
        "env": resolve_name(spec["env_name"])(**spec.get("env_kwargs", {})),
        "obs_builder": resolve_name(spec["obs_builder_name"]),
        "group_encoder": resolve_name(spec["group_encoder_name"])(
            **spec.get("group_encoder_kwargs", {})
        ),
        "actor": resolve_name(spec["actor_name"])(**spec.get("actor_kwargs", {})),
        "safety_layer": resolve_name(spec["safety_name"]),
        "team_reward": resolve_name(spec["team_reward_name"]),
        "weights": weights,
    }


_WORKER: dict = {}


def _worker_init(spec: dict) -> None:
    torch.set_num_threads(1)
    _WORKER["deps"] = build_rollout_deps(spec)


def run_job(job: Mapping) -> dict:
    """One episode in a worker process (deps from the Pool initializer) or
    in-process when the job carries its own spec (serial fallback)."""
    deps = _WORKER.get("deps")
    if deps is None:
        deps = build_rollout_deps(job["spec"])
        _WORKER["deps"] = deps
    deps["actor"].load_state_dict(job["actor_state"])
    deps["group_encoder"].load_state_dict(job["encoder_state"])
    episode = collect_episode(
        deps["env"],
        obs_builder=deps["obs_builder"],
        group_encoder=deps["group_encoder"],
        actor=deps["actor"],
        safety_layer=deps["safety_layer"],
        team_reward=deps["team_reward"],
        weights=deps["weights"],
        seed=job["seed"],
        demand_vph=job["demand_vph"],
        penetration=job["penetration"],
        warmup_s=job["warmup_s"],
        horizon_s=job["horizon_s"],
    )
    return {
        "episode": episode,
        "job": {k: job[k] for k in ("seed", "demand_vph", "penetration", "warmup_s", "horizon_s")},
    }


def make_pool(workers: int, spec: dict):
    """N parallel SUMO envs, one per process (spawn: safe on Windows)."""
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    return ctx.Pool(processes=workers, initializer=_worker_init, initargs=(spec,))
