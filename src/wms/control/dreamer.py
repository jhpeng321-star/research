"""Analytic, continuous-action Dreamer training over frozen WM-S members.

Control steps follow ``contract.STEPS_PER_CONTROL``; ``horizon=60`` therefore
means 60 seconds under the current shared contract (120 physics steps).
All control/reward dependencies are injected; importing this module needs no
parallel control modules, simulator, dataset, or checkpoint.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import Tensor, nn

from .contract import PHYSICS_DT_S, STEPS_PER_CONTROL, RewardWeights
from .start_states import StartBatch, checked_path


@dataclass
class DreamerConfig:
    batch_size: int = 16
    horizon: int = 60
    gamma: float = 0.99
    lambda_: float = 0.95
    actor_lr: float = 5e-5
    critic_lr: float = 1e-4
    entropy_weight: float = 3e-4
    uncertainty_weight: float = 0.1
    ttc_budget: float = 0.01
    dual_lr: float = 0.01
    lagrange_init: float = 0.0
    grad_clip: float = 100.0
    target_tau: float = 0.02
    return_ema: float = 0.99
    checkpoint_every: int = 100
    validation_every: int = 100

    def __post_init__(self):
        for name in ("batch_size", "horizon"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("gamma", "lambda_", "target_tau", "return_ema"):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if min(self.actor_lr, self.critic_lr, self.grad_clip) <= 0:
            raise ValueError("learning rates and grad_clip must be positive")
        for name in ("checkpoint_every", "validation_every"):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an integer")


def lambda_returns(rewards: Tensor, values: Tensor, gamma=0.99, lambda_=0.95) -> Tensor:
    """Time-first TD(lambda), including a value bootstrap at the final horizon."""
    if rewards.ndim != 2 or values.shape != (rewards.shape[0] + 1, rewards.shape[1]):
        raise ValueError("expected rewards [H,B] and values [H+1,B]")
    if not rewards.shape[0] or not 0 <= gamma <= 1 or not 0 <= lambda_ <= 1:
        raise ValueError("nonempty horizon and gamma/lambda in [0,1] required")
    result, carry = [], values[-1]
    for t in reversed(range(rewards.shape[0])):
        carry = rewards[t] + gamma * ((1 - lambda_) * values[t + 1] + lambda_ * carry)
        result.append(carry)
    return torch.stack(result[::-1])


def ensemble_penalty(rewards: Tensor, weight: float) -> Tensor:
    """Population disagreement in per-control-step reward, zero for one member."""
    if rewards.ndim != 3 or rewards.shape[0] < 1 or weight < 0:
        raise ValueError("expected nonempty [E,H,B] rewards and nonnegative weight")
    if rewards.shape[0] == 1 or weight == 0:
        return torch.zeros_like(rewards[0])
    return weight * rewards.std(dim=0, unbiased=False)


def _rng_state(device):
    numpy_state = np.random.get_state()
    # Plain types/tensors keep checkpoints loadable with weights_only=True.
    numpy_state = (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:])
    out = {"python": random.getstate(), "numpy": numpy_state,
           "torch": torch.get_rng_state()}
    if device.type == "cuda":
        out["cuda"] = torch.cuda.get_rng_state(device)
    return out


def _restore_rng(state, device):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                         *numpy_state[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def _detach(state):
    return {key: value.detach() if torch.is_tensor(value) else value
            for key, value in state.items()}


def _model_identity(model):
    digest = hashlib.sha256(type(model).__qualname__.encode())
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    digest.update(str(getattr(model, "grad_decay_per_step", 1.0)).encode())
    return digest.hexdigest()


class DreamerTrainer:
    """Owns actor/group encoder, critic, EMA target, dual and sampler state.

    The internal controller samples once per control step, holds acceleration,
    reapplies safety and integrates motion on every physics callback. A single
    WM rollout preserves driver encoding, history, lane ages and gradient damping.
    """

    def __init__(self, world_models, policy, group_encoder, critic, start_states, *,
                 observation_builder, reward_fn, safety, controller_factory=None, config=None,
                 reward_weights=None, device="cpu", output_dir=None):
        self.config = config or DreamerConfig()
        self.device = torch.device(device)
        if not world_models:
            raise ValueError("at least one world model is required")
        self.world_models = [model.to(self.device).eval() for model in world_models]
        for model in self.world_models:
            model.requires_grad_(False)
            model.zero_grad(set_to_none=True)
            dt = getattr(getattr(model, "physics", None), "dt_s", PHYSICS_DT_S)
            if not math.isclose(dt, PHYSICS_DT_S):
                raise ValueError("WM physics dt does not match the shared control contract")
        self.world_model_ids = [_model_identity(model) for model in self.world_models]
        self.policy = policy.to(self.device)
        self.group_encoder = group_encoder.to(self.device)
        self.critic = critic.to(self.device)
        self.target_critic = copy.deepcopy(critic).to(self.device).eval().requires_grad_(False)
        self.start_states = start_states
        self.observation_builder, self.reward_fn, self.safety = observation_builder, reward_fn, safety
        self.controller_factory = controller_factory
        self.reward_weights = reward_weights or RewardWeights()
        self.actor_parameters = list(dict.fromkeys(
            list(self.policy.parameters()) + list(self.group_encoder.parameters())))
        self.actor_optimizer = torch.optim.Adam(self.actor_parameters, lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)
        self.lagrange_multiplier = self.config.lagrange_init
        self.return_low = self.return_high = None
        self.update_count = 0
        self.best_validation = -math.inf
        self.validation_states = None
        self._pending_validation_state = None
        self.output_dir = checked_path(output_dir) if output_dir is not None else None

    def imagine(self, batch: StartBatch, deterministic=False):
        cfg = self.config
        forbidden = {"cav_teacher", "lane_teacher", "action_timeline", "present_seq", "is_cav_seq",
                     "junction_seq", "truncation_steps", "cav_controller"}
        if forbidden.intersection(batch.rollout_kwargs):
            raise ValueError("imagination inputs must not contain teachers or recorded controls")
        member_states, member_rewards, member_costs = [], [], []
        member_entropy, member_actions, member_components = [], [], []
        rng = _rng_state(self.device)
        advanced_rng = None
        # Common random numbers: disagreement must measure WM differences, not action noise.
        for model in self.world_models:
            _restore_rng(rng, self.device)
            records = []
            step = 0
            held_accel = None
            held_log_prob = None
            last_decision_step = -1

            def policy_proxy(obs, signal, **kwargs):
                nonlocal held_accel, held_log_prob, last_decision_step
                if step % STEPS_PER_CONTROL == 0 and last_decision_step != step:
                    held_accel, held_log_prob = self.policy(obs, signal, deterministic=deterministic)
                    mask = obs.cav_mask
                    records[-1]["entropy"] = -(held_log_prob * mask).sum(-1) / mask.sum(-1).clamp_min(1)
                    last_decision_step = step
                if held_accel is None:
                    raise RuntimeError("controller factory did not sample an initial policy command")
                return held_accel, held_log_prob

            external_callback = (self.controller_factory(policy_proxy, self.group_encoder, self.safety, None)
                                 if self.controller_factory is not None else None)

            def controller(state):
                nonlocal step, held_accel
                mask = state["present"].bool() & state["is_cav"].bool()
                if step % STEPS_PER_CONTROL == 0:
                    records.append({"mask": mask})
                if external_callback is not None:
                    command = external_callback(state)
                    if step % STEPS_PER_CONTROL == 0:
                        if "entropy" not in records[-1]:
                            raise RuntimeError("controller factory must call the supplied policy at each decision")
                        records[-1]["action"] = command[..., 3]
                    step += 1
                    return command
                if step % STEPS_PER_CONTROL == 0:
                    obs = self.observation_builder(state)
                    signal = self.group_encoder(obs, state)
                    policy_proxy(obs, signal)
                accel = self.safety(state, held_accel)
                accel = torch.where(mask, accel, torch.zeros_like(accel))
                if step % STEPS_PER_CONTROL == 0:
                    records[-1]["action"] = accel
                # SUMO's ballistic position integration, including a stop inside this step.
                speed = (state["v_mps"] + PHYSICS_DT_S * accel).clamp_min(0)
                moving_dt = torch.where(
                    accel < 0, (state["v_mps"] / (-accel).clamp_min(1e-8)).clamp(0, PHYSICS_DT_S),
                    torch.full_like(accel, PHYSICS_DT_S))
                position = state["s_m"] + state["v_mps"] * moving_dt + 0.5 * accel * moving_dt.square()
                step += 1
                return torch.stack((position, state["lane"].to(speed.dtype), speed, accel), -1)

            initial = {key: value.clone() if torch.is_tensor(value) else value
                       for key, value in batch.state.items()}
            roll = model.rollout(initial, n_steps=cfg.horizon * STEPS_PER_CONTROL,
                                 cav_controller=controller, cav_teacher=None,
                                 truncation_steps=None, **batch.rollout_kwargs)
            if step != cfg.horizon * STEPS_PER_CONTROL:
                raise RuntimeError("WM must invoke the CAV callback once per physics step")
            if advanced_rng is None:
                advanced_rng = _rng_state(self.device)
            physics_states = [initial]
            for t in range(step):
                nxt = {**initial, "s_m": roll["positions"][:, t], "v_mps": roll["speeds"][:, t],
                       "lane": roll["lanes"][:, t], "present": roll["masks"][:, t],
                       "a_mps2": roll["accel"][:, t]}
                if "entrant_seq" in roll:
                    nxt["entrant"] = roll["entrant_seq"][:, t]
                physics_states.append(nxt)
            # Preserve WM-specific state fields at the terminal boundary.
            physics_states[-1] = roll["state"]
            parts = []
            for t in range(step):
                parts.append(self.reward_fn(physics_states[t], physics_states[t + 1],
                                            roll["accel"][:, t], self.reward_weights))
            keys = set(parts[0])
            if not {"reward", "cost_ttc"}.issubset(keys):
                raise ValueError("team_reward must provide reward and cost_ttc")
            components = {key: torch.stack([part[key] for part in parts]).reshape(
                cfg.horizon, STEPS_PER_CONTROL, -1).sum(1) for key in keys}
            member_components.append(components)
            member_rewards.append(components["reward"])
            member_costs.append(components["cost_ttc"])
            member_states.append(physics_states[::STEPS_PER_CONTROL])
            member_entropy.append(torch.stack([record["entropy"] for record in records]))
            member_actions.append(torch.stack([record["action"] for record in records]))
        _restore_rng(advanced_rng, self.device)
        rewards, costs = torch.stack(member_rewards), torch.stack(member_costs)
        disagreement = ensemble_penalty(rewards, 1.0)
        adjusted = rewards - cfg.uncertainty_weight * disagreement.unsqueeze(0)
        returns = []
        for e, states in enumerate(member_states):
            # Freeze critic weights, not state derivatives: the terminal bootstrap
            # must provide analytic actor credit through the imagined WM trajectory.
            values = torch.stack([self.target_critic(state) for state in states])
            returns.append(lambda_returns(adjusted[e], values, cfg.gamma, cfg.lambda_))
        masks = torch.stack([torch.stack([state["present"] & state["is_cav"]
                                         for state in states[:-1]]) for states in member_states])
        return {"rewards": rewards, "costs": costs, "returns": torch.stack(returns),
                "disagreement": disagreement, "states": member_states,
                "entropy": torch.stack(member_entropy), "actions": torch.stack(member_actions),
                "action_mask": masks, "components": {key: torch.stack([c[key] for c in member_components])
                                                       for key in member_components[0]}}

    def update_lagrange(self, mean_cost):
        value = float(mean_cost)
        if not math.isfinite(value):
            raise ValueError("nonfinite safety cost")
        self.lagrange_multiplier = max(0.0, self.lagrange_multiplier + self.config.dual_lr *
                                      (value - self.config.ttc_budget))
        return self.lagrange_multiplier

    def _return_scale(self, returns):
        quantiles = torch.quantile(returns.detach().float().flatten(),
                                   torch.tensor([0.05, 0.95], device=self.device))
        low, high = map(float, quantiles)
        if self.return_low is None:
            self.return_low, self.return_high = low, high
        else:
            decay = self.config.return_ema
            self.return_low = decay * self.return_low + (1 - decay) * low
            self.return_high = decay * self.return_high + (1 - decay) * high
        return max(1.0, self.return_high - self.return_low)

    def update(self):
        started = time.perf_counter()
        self.policy.train()
        self.group_encoder.train()
        self.critic.train()
        batch = self.start_states.sample(self.config.batch_size, device=self.device)
        imagined = self.imagine(batch)
        returns = imagined["returns"]
        scale = self._return_scale(returns)
        entropy = imagined["entropy"].mean()
        mean_cost = imagined["costs"].mean()
        actor_loss = (-returns.mean() / scale - self.config.entropy_weight * entropy +
                      self.lagrange_multiplier * mean_cost)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_norm = nn.utils.clip_grad_norm_(self.actor_parameters, self.config.grad_clip,
                                             error_if_nonfinite=True)
        self.actor_optimizer.step()
        predictions = torch.stack([torch.stack([self.critic(_detach(state)) for state in states[:-1]])
                                   for states in imagined["states"]])
        critic_loss = ((predictions - returns.detach()) / scale).square().mean()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_clip,
                                              error_if_nonfinite=True)
        self.critic_optimizer.step()
        with torch.no_grad():
            for target, source in zip(self.target_critic.parameters(), self.critic.parameters()):
                target.lerp_(source, self.config.target_tau)
            for target, source in zip(self.target_critic.buffers(), self.critic.buffers()):
                if target.is_floating_point():
                    target.lerp_(source, self.config.target_tau)
                else:
                    target.copy_(source)
        self.update_lagrange(mean_cost.detach())
        self.update_count += 1
        selected = imagined["actions"].detach()[imagined["action_mask"]]
        action_stats = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        if selected.numel():
            action_stats = {"mean": float(selected.mean()), "std": float(selected.std(unbiased=False)),
                            "min": float(selected.min()), "max": float(selected.max())}
        metrics = {"update": self.update_count, "returns": float(returns.detach().mean()),
                   "reward": float(imagined["rewards"].detach().mean()),
                   "actor_loss": float(actor_loss.detach()), "critic_loss": float(critic_loss.detach()),
                   "lagrange_multiplier": self.lagrange_multiplier, "entropy": float(entropy.detach()),
                   "actor_grad_norm": float(actor_norm), "critic_grad_norm": float(critic_norm),
                   "ensemble_disagreement": float(imagined["disagreement"].detach().mean()),
                   "return_scale": scale, "action_stats": action_stats,
                   "wall_time_s": time.perf_counter() - started}
        for name in ("delay", "throughput", "smooth", "hard_brake", "fairness_delay", "cost_ttc"):
            value = imagined["components"].get(name)
            metrics["fairness" if name == "fairness_delay" else name] = (
                float(value.detach().mean()) if value is not None else 0.0)
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with (self.output_dir / "updates.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")
            if self.config.checkpoint_every and self.update_count % self.config.checkpoint_every == 0:
                self.save_checkpoint(self.output_dir / f"policy_{self.update_count:06d}.pt")
        return metrics

    def validate(self, sampler, batches=1):
        """Deterministic imagined lambda-return on separate VAL windows; no update."""
        if batches < 1:
            raise ValueError("validation requires at least one batch")
        rng = _rng_state(self.device)
        modes = (self.policy.training, self.group_encoder.training)
        self.policy.eval()
        self.group_encoder.eval()
        try:
            with torch.no_grad():
                results = [float(self.imagine(sampler.sample(self.config.batch_size, device=self.device),
                                             deterministic=True)["returns"].mean())
                           for _ in range(batches)]
            score = sum(results) / len(results)
            if not math.isfinite(score):
                raise ValueError("nonfinite validation return")
            return score
        finally:
            _restore_rng(rng, self.device)
            self.policy.train(modes[0])
            self.group_encoder.train(modes[1])

    def fit(self, max_updates, *, validation_states=None, validation_batches=1):
        self.validation_states = validation_states
        if self._pending_validation_state is not None and validation_states is not None:
            validation_states.load_state_dict(self._pending_validation_state)
            self._pending_validation_state = None
        last = None
        while self.update_count < max_updates:
            last = self.update()
            if (validation_states is not None and self.config.validation_every and
                    self.update_count % self.config.validation_every == 0):
                score = self.validate(validation_states, validation_batches)
                if self.output_dir is not None:
                    with (self.output_dir / "validation.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({"update": self.update_count, "returns": score}) + "\n")
                if score > self.best_validation:
                    self.best_validation = score
                    if self.output_dir is not None:
                        self.save_checkpoint(self.output_dir / "policy_best.pt")
        if self.output_dir is not None:
            self.save_checkpoint(self.output_dir / "policy_latest.pt")
        return last

    def save_checkpoint(self, path):
        path = checked_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": "c4_dreamer/1", "config": asdict(self.config),
                   "reward_weights": asdict(self.reward_weights),
                   "policy": self.policy.state_dict(), "group_encoder": self.group_encoder.state_dict(),
                   "critic": self.critic.state_dict(), "target_critic": self.target_critic.state_dict(),
                   "actor_optimizer": self.actor_optimizer.state_dict(),
                   "critic_optimizer": self.critic_optimizer.state_dict(),
                   "update_count": self.update_count, "lagrange_multiplier": self.lagrange_multiplier,
                   "return_low": self.return_low, "return_high": self.return_high,
                   "best_validation": self.best_validation, "rng": _rng_state(self.device),
                   "ensemble_size": len(self.world_models),
                   "world_model_ids": self.world_model_ids,
                   "validation_sampler": (self.validation_states.state_dict()
                                          if hasattr(self.validation_states, "state_dict") else None),
                   "sampler": self.start_states.state_dict() if hasattr(self.start_states, "state_dict") else None}
        torch.save(payload, path)

    def load_checkpoint(self, path, *, policy_only=False):
        payload = torch.load(checked_path(path), map_location=self.device, weights_only=True)
        if payload.get("format") != "c4_dreamer/1":
            raise ValueError("expected c4_dreamer/1 checkpoint")
        if not policy_only and (payload["ensemble_size"] != len(self.world_models) or
                                payload["world_model_ids"] != self.world_model_ids):
            raise ValueError("resume requires the same frozen world-model ensemble")
        self.policy.load_state_dict(payload["policy"])
        self.group_encoder.load_state_dict(payload["group_encoder"])
        if policy_only:
            return
        self.config = DreamerConfig(**payload["config"])
        self.reward_weights = RewardWeights(**payload["reward_weights"])
        for name in ("critic", "target_critic", "actor_optimizer", "critic_optimizer"):
            getattr(self, name).load_state_dict(payload[name])
        for name in ("update_count", "lagrange_multiplier", "return_low", "return_high", "best_validation"):
            setattr(self, name, payload[name])
        if payload["sampler"] is not None:
            self.start_states.load_state_dict(payload["sampler"])
        self._pending_validation_state = payload.get("validation_sampler")
        _restore_rng(payload["rng"], self.device)
