"""MAPPO trainer: iterations of budgeted SUMO episode collection + PPO updates.

Budget accounting is dual-unit (SUMO-seconds and env-steps) so the run can be
matched to the C4 method's SUMO interaction budget: an episode charges
``steps * CONTROL_DT_S`` simulated seconds (warmup excluded by default, config
``charge_warmup``) plus its env-steps, and no episode that would exceed
``max_sumo_seconds`` / ``max_env_steps`` is ever started.  Everything is
resumable: ``checkpoints/latest.pt`` restores models, optimisers, the Lagrange
multiplier, the return scaler, both budgets and all RNG states.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from wms.control.contract import CONTROL_DT_S, RewardWeights

from .buffer import EpisodeData, ReturnScaler, episode_gae, flatten_episodes
from .ppo import PPOConfig, PPOUpdater, update_lagrange
from .rollout import collect_episode, make_pool, run_job

DEFAULT_PENETRATIONS = (0.05, 0.1, 0.2, 0.4)
DEFAULT_DEMANDS_VPH = (2400, 2600, 2700)


@dataclass
class RolloutConfig:
    seed: int = 1
    iterations: int = 1000
    episodes_per_iteration: int = 8
    horizon_s: float = 300.0
    warmup_s: float = 60.0
    penetrations: tuple[float, ...] = DEFAULT_PENETRATIONS
    demands_vph: tuple[int, ...] = DEFAULT_DEMANDS_VPH
    max_sumo_seconds: float = 1.0e6
    max_env_steps: int | None = None
    charge_warmup: bool = False


@dataclass
class MappoConfig:
    ppo: PPOConfig = field(default_factory=PPOConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)

    def to_dict(self) -> dict:
        return {"ppo": asdict(self.ppo), "rollout": asdict(self.rollout)}


@dataclass
class Budget:
    max_sumo_seconds: float
    max_env_steps: int | None
    used_sumo_seconds: float = 0.0
    used_env_steps: int = 0

    def can_start(self, horizon_s: float, warmup_s: float, charge_warmup: bool) -> bool:
        seconds = horizon_s + (warmup_s if charge_warmup else 0.0)
        steps = int(round(horizon_s / CONTROL_DT_S))
        if self.used_sumo_seconds + seconds > self.max_sumo_seconds + 1e-9:
            return False
        if self.max_env_steps is not None and self.used_env_steps + steps > self.max_env_steps:
            return False
        return True

    def charge(self, episode: EpisodeData, warmup_s: float, charge_warmup: bool) -> None:
        seconds = episode.sumo_seconds + (warmup_s if charge_warmup else 0.0)
        self.used_sumo_seconds += seconds
        self.used_env_steps += episode.steps

    def to_dict(self) -> dict:
        return {
            "max_sumo_seconds": self.max_sumo_seconds,
            "max_env_steps": self.max_env_steps,
            "used_sumo_seconds": self.used_sumo_seconds,
            "used_env_steps": self.used_env_steps,
        }


def _torch_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch without the kwarg
        return torch.load(path, map_location="cpu")


class MappoTrainer:
    """Dependency-injected trainer; the CLI wires the real SUMO/C4 modules by
    dotted name, tests and --smoke wire the fakes."""

    def __init__(
        self,
        *,
        out_dir: Path | str,
        config: MappoConfig,
        env,
        obs_builder,
        group_encoder,
        actor,
        critic,
        safety_layer,
        team_reward,
        weights: RewardWeights | None = None,
        meta: Mapping[str, str] | None = None,
        workers: int = 1,
    ):
        self.out_dir = Path(out_dir)
        self.cfg = config
        self.env = env
        self.obs_builder = obs_builder
        self.group_encoder = group_encoder
        self.actor = actor
        self.critic = critic
        self.safety_layer = safety_layer
        self.team_reward = team_reward
        self.weights = weights or RewardWeights()
        self.meta = dict(meta or {})
        self.workers = int(workers)
        self.ckpt_dir = self.out_dir / "checkpoints"
        self.log_path = self.out_dir / "log.jsonl"
        self.summary_path = self.out_dir / "RUN_SUMMARY.json"
        self.updater = PPOUpdater(
            actor=actor,
            critic=critic,
            group_encoder=group_encoder,
            obs_builder=obs_builder,
            config=config.ppo,
        )
        self.scaler = ReturnScaler()
        self.budget = Budget(
            max_sumo_seconds=config.rollout.max_sumo_seconds,
            max_env_steps=config.rollout.max_env_steps,
        )
        self.iteration = 0
        self.episode_index = 0
        self.episodes_done = 0
        self.rng = np.random.default_rng(config.rollout.seed)
        self.pool = None
        self.status = "initialized"
        self.last_diag: dict = {}

    # ---------------------------------------------------------------- persistence
    def _checkpoint(self) -> dict:
        return {
            "iteration": self.iteration,
            "episode_index": self.episode_index,
            "episodes_done": self.episodes_done,
            "actor": self.actor.state_dict(),
            "group_encoder": self.group_encoder.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_opt": self.updater.actor_opt.state_dict(),
            "critic_opt": self.updater.critic_opt.state_dict(),
            "lagrange_alpha": self.updater.alpha,
            "return_scaler": self.scaler.state_dict(),
            "budget": self.budget.to_dict(),
            "rng": {
                "numpy": self.rng.bit_generator.state,
                "torch": torch.get_rng_state(),
                "python": random.getstate(),
            },
            "meta": self.meta,
            "config": self.cfg.to_dict(),
        }

    def _save(self) -> None:
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        payload = self._checkpoint()
        torch.save(payload, self.ckpt_dir / "latest.pt")
        torch.save(payload, self.ckpt_dir / f"ckpt_{self.iteration:05d}.pt")

    def _try_resume(self) -> bool:
        latest = self.ckpt_dir / "latest.pt"
        if not latest.is_file():
            return False
        state = _torch_load(latest)
        self.iteration = state["iteration"]
        self.episode_index = state["episode_index"]
        self.episodes_done = state["episodes_done"]
        self.actor.load_state_dict(state["actor"])
        self.group_encoder.load_state_dict(state["group_encoder"])
        self.critic.load_state_dict(state["critic"])
        self.updater.actor_opt.load_state_dict(state["actor_opt"])
        self.updater.critic_opt.load_state_dict(state["critic_opt"])
        self.updater.alpha = state["lagrange_alpha"]
        self.scaler = ReturnScaler.from_state(state["return_scaler"])
        self.budget = Budget(**state["budget"])
        self.rng.bit_generator.state = state["rng"]["numpy"]
        torch.set_rng_state(state["rng"]["torch"].to(torch.uint8))
        random.setstate(_as_pytuple(state["rng"]["python"]))
        return True

    def _log_line(self, row: dict) -> None:
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    def _write_summary(self) -> None:
        row = {
            "status": self.status,
            "iterations_completed": self.iteration,
            "episodes": self.episodes_done,
            "budget": self.budget.to_dict(),
            "checkpoint": "checkpoints/latest.pt",
            "actor_load": "wms.control.mappo.eval_bridge.load_policy_fn",
            "meta": self.meta,
            "config": self.cfg.to_dict(),
            "last": self.last_diag,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.summary_path.write_text(json.dumps(row, indent=1), encoding="utf-8")

    # ---------------------------------------------------------------- rollout
    def _plan_jobs(self) -> list[dict]:
        jobs = []
        r = self.cfg.rollout
        for _ in range(r.episodes_per_iteration):
            if not self.budget.can_start(r.horizon_s, r.warmup_s, r.charge_warmup):
                break
            pen = float(self.rng.choice(r.penetrations))
            dem = int(self.rng.choice(r.demands_vph))
            jobs.append(
                {
                    "seed": r.seed * 100_003 + self.episode_index,
                    "demand_vph": dem,
                    "penetration": pen,
                    "warmup_s": r.warmup_s,
                    "horizon_s": r.horizon_s,
                }
            )
            self.episode_index += 1
        return jobs

    def _run_jobs(self, jobs: list[dict]) -> list[EpisodeData]:
        if self.workers > 1:
            if self.pool is None:
                self.pool = make_pool(self.workers, self._worker_spec())
            payloads = [
                {
                    "actor_state": self.actor.state_dict(),
                    "encoder_state": self.group_encoder.state_dict(),
                    **job,
                }
                for job in jobs
            ]
            return [row["episode"] for row in self.pool.map(run_job, payloads)]
        return [
            collect_episode(
                self.env,
                obs_builder=self.obs_builder,
                group_encoder=self.group_encoder,
                actor=self.actor,
                safety_layer=self.safety_layer,
                team_reward=self.team_reward,
                weights=self.weights,
                **job,
            )
            for job in jobs
        ]

    def _worker_spec(self) -> dict:
        names = self.meta.get("names", {})
        return {
            "env_name": names.get("env_name", "fake"),
            "env_kwargs": self.meta.get("env_kwargs", {}),
            "obs_builder_name": names.get("obs_builder_name", "fake_obs"),
            "group_encoder_name": names.get("group_encoder_name", "fake_encoder"),
            "actor_kwargs": self.meta.get("actor_kwargs", {}),
            "group_encoder_kwargs": self.meta.get("encoder_kwargs", {}),
            "actor_name": names.get("actor_name", "fake_actor"),
            "safety_name": names.get("safety_name", "fake_safety"),
            "team_reward_name": names.get("team_reward_name", "fake_reward"),
            "reward_weights": self.meta.get("reward_weights", {}),
        }

    # ---------------------------------------------------------------- main loop
    def run(self, max_iterations: int | None = None) -> dict:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        resumed = self._try_resume()
        self.pool = None
        if not resumed:
            random.seed(self.cfg.rollout.seed)
            np.random.seed(self.cfg.rollout.seed % (2**32))
            torch.manual_seed(self.cfg.rollout.seed)
            self._log_line({"type": "config", **self.cfg.to_dict(), "meta": self.meta})
        target = self.iteration + (max_iterations or self.cfg.rollout.iterations)
        self.status = "running"
        while self.iteration < target:
            jobs = self._plan_jobs()
            if not jobs:
                self.status = "budget_exhausted"
                break
            episodes = self._run_jobs(jobs)
            for ep in episodes:
                self.budget.charge(ep, self.cfg.rollout.warmup_s, self.cfg.rollout.charge_warmup)
                self.episodes_done += 1
            batch = flatten_episodes(episodes)
            with torch.no_grad():
                v_norm, vf_norm = self.updater.values(batch)
            values_raw = self.scaler.unscale(v_norm) if self.cfg.ppo.return_norm else v_norm
            finals_raw = self.scaler.unscale(vf_norm) if self.cfg.ppo.return_norm else vf_norm
            adv, ret, cost_ret = episode_gae(
                batch, values_raw, finals_raw, gamma=self.cfg.ppo.gamma, lam=self.cfg.ppo.gae_lambda
            )
            diag = self.updater.update(
                batch, advantages=adv, returns=ret, cost_returns=cost_ret, scaler=self.scaler
            )
            if self.cfg.ppo.return_norm:
                self.scaler.update(ret)
            steps = sum(ep.steps for ep in episodes)
            mean_cost = float(sum(float(ep.costs.sum()) for ep in episodes) / max(1, steps))
            self.updater.alpha = update_lagrange(
                self.updater.alpha,
                mean_cost,
                self.cfg.ppo.cost_limit_per_step,
                self.cfg.ppo.lagrange_lr,
                self.cfg.ppo.lagrange_max,
            )
            self.iteration += 1
            self.last_diag = {
                "type": "iter",
                "iteration": self.iteration,
                "episodes": len(episodes),
                "env_steps": steps,
                "sumo_seconds_used": self.budget.used_sumo_seconds,
                "mean_episode_return": float(np.mean([float(ep.rewards.sum()) for ep in episodes])),
                "mean_cost_per_step": mean_cost,
                **diag,
            }
            self._log_line(self.last_diag)
            self._save()
            self._write_summary()
        if self.status == "running":
            self.status = "completed"
        self._write_summary()
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
        return {
            "status": self.status,
            "iterations_completed": self.iteration,
            "budget": self.budget.to_dict(),
        }


def _as_pytuple(state):
    return tuple((tuple(inner) if isinstance(inner, list) else inner) for inner in state)
