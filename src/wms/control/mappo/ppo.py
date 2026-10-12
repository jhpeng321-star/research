"""Clipped-surrogate PPO update for the C4 MAPPO baseline (CTDE).

Ported from ``archive/circles_era/traffic_shepherd/ppo_train.py``: clipped
surrogate, minibatch epochs, batch advantage normalisation, gradient clipping
and early stop on approx KL.  Added for MAPPO: value clipping, an entropy
bonus, return normalisation (the critic predicts scaled returns) and a
Lagrangian multiplier on the team ``cost_ttc`` exposure (PPO-Lagrangian: a
second clipped surrogate on the normalised discounted cost-to-go, weighted by
a hand-updated multiplier ``alpha`` — no separate cost critic).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from wms.control.contract import Observation

from .buffer import FlatBatch, ReturnScaler


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    clip_value_eps: float = 0.2
    epochs: int = 4
    minibatch_size: int = 512
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03
    cost_limit_per_step: float = 0.05
    lagrange_lr: float = 0.05
    lagrange_max: float = 50.0
    return_norm: bool = True


def evaluate_actions(
    actor, obs: Observation, group_signal: Tensor, accel: Tensor
) -> tuple[Tensor, Tensor]:
    """(log_prob [B,K], entropy [B,K]) of ``accel`` under the actor.

    The C4 ``CavPolicy`` protocol only pins ``__call__``; MAPPO additionally
    needs log-probabilities of stored actions.  Adaptation point (single place):
    the actor must expose ``evaluate_actions`` or ``evaluate`` with this
    signature — a one-line method on ``CavPolicy`` when it lands.
    """
    for name in ("evaluate_actions", "evaluate"):
        fn = getattr(actor, name, None)
        if callable(fn):
            return fn(obs, group_signal, accel)
    raise TypeError(
        "actor must implement evaluate_actions(obs, group_signal, accel) "
        "or evaluate(obs, group_signal, accel) for PPO updates"
    )


def clipped_surrogate(ratio: Tensor, advantage: Tensor, clip_eps: float) -> Tensor:
    """Element-wise min(r*A, clamp(r)*A); the policy loss is ``-mean(...)``."""
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    return torch.min(ratio * advantage, clipped * advantage)


def clipped_value_loss(value: Tensor, value_old: Tensor, target: Tensor, clip_eps: float) -> Tensor:
    """Element-wise 0.5*max((v-t)^2, (v_clip-t)^2) with v clipped near v_old."""
    v_clip = value_old + torch.clamp(value - value_old, -clip_eps, clip_eps)
    return 0.5 * torch.maximum((value - target) ** 2, (v_clip - target) ** 2)


def update_lagrange(
    alpha: float, mean_cost: float, cost_limit: float, lr: float, alpha_max: float
) -> float:
    """Dual-ascent step on the Lagrange multiplier of the cost_ttc constraint."""
    return float(min(max(alpha + lr * (mean_cost - cost_limit), 0.0), alpha_max))


def _normalize(x: Tensor) -> Tensor:
    if x.numel() < 2:
        return x
    std = x.std()
    if not torch.isfinite(std) or float(std) < 1e-8:
        return x - x.mean()
    return (x - x.mean()) / (std + 1e-8)


class PPOUpdater:
    """Minibatch-epoch PPO update over one iteration's flattened episodes."""

    def __init__(
        self,
        *,
        actor: nn.Module,
        critic: nn.Module,
        group_encoder: nn.Module,
        obs_builder,
        config: PPOConfig,
    ):
        self.actor = actor
        self.critic = critic
        self.group_encoder = group_encoder
        self.obs_builder = obs_builder
        self.cfg = config
        self.actor_opt = torch.optim.Adam(
            list(actor.parameters()) + list(group_encoder.parameters()), lr=config.lr_actor
        )
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=config.lr_critic)
        self.alpha = 0.0

    @torch.no_grad()
    def values(self, batch: FlatBatch) -> tuple[Tensor, Tensor]:
        """Critic values of all frames and terminal states (scaled space)."""
        return (self.critic(batch.states), self.critic(batch.final_states))

    def update(
        self,
        batch: FlatBatch,
        *,
        advantages: Tensor,
        returns: Tensor,
        cost_returns: Tensor,
        scaler: ReturnScaler,
    ) -> dict:
        cfg = self.cfg
        with torch.no_grad():
            obs_all = self.obs_builder(batch.states)
            value_all = self.critic(batch.states)
        idx = batch.cav_mask.nonzero(as_tuple=False)
        n_samples = int(idx.shape[0])
        if n_samples == 0:
            raise ValueError("no acting CAV slots in this iteration's batch")
        frames, slots = idx[:, 0], idx[:, 1]
        adv = _normalize(advantages[frames])
        cost_adv = _normalize(cost_returns[frames])
        ret = scaler.scale(returns[frames]) if cfg.return_norm else returns[frames]
        value_old = value_all[frames]
        logp_old = batch.logp[frames, slots]

        diag = dict(
            pi_loss=0.0,
            value_loss=0.0,
            entropy=0.0,
            approx_kl=0.0,
            clip_fraction=0.0,
            cost_surrogate=0.0,
            updates=0,
        )
        stop = False
        m = max(1, min(cfg.minibatch_size, n_samples))
        for _ in range(cfg.epochs):
            if stop:
                break
            perm = torch.randperm(n_samples)
            for start in range(0, n_samples, m):
                mb = perm[start : start + m]
                f, s = frames[mb], slots[mb]
                # Frame-indexed minibatch: the actor/critic consume the full
                # [B, K, ...] Observation/state layout (contract), then the
                # acting slots are gathered for the losses.
                obs_mb = Observation(
                    own=obs_all.own[f],
                    reports=obs_all.reports[f],
                    report_mask=obs_all.report_mask[f],
                    cav_mask=obs_all.cav_mask[f],
                )
                group_signal = self.group_encoder(obs_mb)
                logp_full, entropy_full = evaluate_actions(
                    self.actor, obs_mb, group_signal, batch.actions[f]
                )
                row = torch.arange(f.shape[0])
                logp, entropy = logp_full[row, s], entropy_full[row, s]
                ratio = torch.exp(torch.clamp(logp - logp_old[mb], -20.0, 20.0))
                pi_loss = -clipped_surrogate(ratio, adv[mb], cfg.clip_eps).mean()
                cost_surr = clipped_surrogate(ratio, cost_adv[mb], cfg.clip_eps).mean()
                value = self.critic({k: v[f] for k, v in batch.states.items()})
                v_loss = clipped_value_loss(
                    value, value_old[mb], ret[mb], cfg.clip_value_eps
                ).mean()
                loss = (
                    pi_loss
                    + self.alpha * cost_surr
                    + cfg.value_coef * v_loss
                    - cfg.entropy_coef * entropy.mean()
                )
                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.group_encoder.parameters()),
                    cfg.max_grad_norm,
                )
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.actor_opt.step()
                self.critic_opt.step()
                with torch.no_grad():
                    diag["pi_loss"] += float(pi_loss)
                    diag["value_loss"] += float(v_loss)
                    diag["entropy"] += float(entropy.mean())
                    diag["approx_kl"] += float((logp_old[mb] - logp).mean())
                    diag["clip_fraction"] += float(
                        ((ratio - 1.0).abs() > cfg.clip_eps).float().mean()
                    )
                    diag["cost_surrogate"] += float(cost_surr)
                    diag["updates"] += 1
                    if cfg.target_kl and diag["approx_kl"] / diag["updates"] > 1.5 * cfg.target_kl:
                        stop = True
            if stop:
                break
        for key in (
            "pi_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clip_fraction",
            "cost_surrogate",
        ):
            diag[key] /= max(1, diag["updates"])
        diag["lagrange_alpha"] = self.alpha
        diag["samples"] = n_samples
        diag["epochs"] = cfg.epochs
        diag["finite"] = bool(
            all(math.isfinite(diag[k]) for k in ("pi_loss", "value_loss", "entropy", "approx_kl"))
        )
        return diag
