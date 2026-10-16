"""Clipped-surrogate PPO update for the C4 MAPPO baseline (CTDE, joint per-CAV action).

Action of one CAV per 1 s decision = (acceleration, lane choice): the joint log-probability
is ``log pi(accel) + log pi(lane | feasible mask)`` (lane term 0 when only "stay" is
feasible), the entropy bonus covers both heads (pre-tanh Gaussian entropy for the
acceleration, exact masked categorical entropy for the lane).  The actor (``CavPolicy`` +
``RsuGroupEncoder``) is decentralised; the critic is the privileged ``TeamCritic`` on the
full SUMO state.  Minibatches are sets of FRAMES (the RSU encoder couples the CAVs of one
frame); the policy loss averages over the acting CAVs of those frames, the value loss over
the frames.

Safety constraint (PPO-Lagrangian, same TTC cost and budget semantics as the C4 trainer):
pessimistic clipped surrogate of the batch-normalised discounted ``cost_ttc``-to-go,
``loss = (L_reward + alpha * L_cost) / (1 + alpha)``; ``alpha`` by projected dual ascent on
the mean per-control-step cost vs ``cost_limit_per_step``.

Gradient-noise diagnostic (paper: "noisy learning signal"): before the update, at the
rollout weights (ratio = 1), the policy-gradient loss ``-mean(A * log pi)`` is
differentiated separately on M disjoint frame minibatches.  With g_m the flattened actor
(+encoder) gradient of minibatch m and g = mean_m g_m we log ``pg_grad_mean_norm`` = |g|,
``pg_grad_norm_mean`` / ``pg_grad_norm_std`` (mean / std of |g_m|), ``pg_grad_var_trace`` =
sum_m |g_m - g|^2 / (M - 1) (trace of the covariance of a one-minibatch gradient estimate),
``pg_grad_var_trace_batch`` = that / M (of the full-batch estimate) and ``pg_grad_snr`` =
|g|^2 / pg_grad_var_trace.  Archive ports: clipped surrogate / minibatch epochs / KL stop
from ``archive/circles_era/traffic_shepherd/ppo_train.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .agent import _gather, compact_present, evaluate_actions
from .buffer import FlatBatch, ReturnScaler
from .rollout import StopRequested


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    clip_value_eps: float = 0.2
    epochs: int = 4
    minibatch_frames: int = 256
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    entropy_coef: float = 0.01        # acceleration head (pre-tanh Gaussian entropy)
    lane_entropy_coef: float = 0.01   # lane head (masked categorical entropy)
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03
    cost_limit_per_step: float = 3.6  # = C4 trainer ttc_budget (sum of 2 physics-step costs)
    lagrange_lr: float = 0.01          # = C4 trainer dual_lr
    lagrange_max: float = 50.0
    return_norm: bool = True
    grad_noise_every: int = 1          # iterations between gradient-noise diagnostics (0 = off)
    eval_chunk_frames: int = 512       # no-grad critic evaluation chunk


def clipped_surrogate(ratio: Tensor, advantage: Tensor, clip_eps: float) -> Tensor:
    """Element-wise min(r*A, clamp(r)*A); the policy loss is ``-mean(...)``."""
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    return torch.min(ratio * advantage, clipped * advantage)


def pessimistic_cost_surrogate(ratio: Tensor, cost_adv: Tensor, clip_eps: float) -> Tensor:
    """Element-wise max(r*C, clamp(r)*C): the clipped upper bound of the cost change."""
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    return torch.max(ratio * cost_adv, clipped * cost_adv)


def clipped_value_loss(value: Tensor, value_old: Tensor, target: Tensor, clip_eps: float) -> Tensor:
    """Element-wise 0.5*max((v-t)^2, (v_clip-t)^2) with v clipped near v_old."""
    v_clip = value_old + torch.clamp(value - value_old, -clip_eps, clip_eps)
    return 0.5 * torch.maximum((value - target) ** 2, (v_clip - target) ** 2)


def update_lagrange(alpha: float, mean_cost: float, cost_limit: float, lr: float,
                    alpha_max: float) -> float:
    """Projected dual ascent on the cost_ttc multiplier."""
    return float(min(max(alpha + lr * (mean_cost - cost_limit), 0.0), alpha_max))


def sample_normalized(per_frame: Tensor, weights: Tensor) -> Tensor:
    """Normalise a per-frame signal with the mean/std over CAV samples (frame f counts
    ``weights[f]`` times: the team advantage is broadcast to every acting CAV)."""
    w = weights.float()
    total = float(w.sum())
    if total < 2:
        return per_frame - (per_frame * w).sum() / max(total, 1.0)
    mean = (per_frame * w).sum() / total
    var = ((per_frame - mean) ** 2 * w).sum() / (total - 1.0)
    std = float(var.sqrt())
    if not math.isfinite(std) or std < 1e-8:
        return per_frame - mean
    return (per_frame - mean) / (std + 1e-8)


def _frames(batch: FlatBatch, f: Tensor) -> tuple[dict[str, Tensor], Tensor]:
    """Decision states of frames ``f`` compacted to their present slots (+ slot index)."""
    return compact_present({k: v[f] for k, v in batch.states.items()})


def _flat_grad(params: list[nn.Parameter]) -> Tensor:
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                      for p in params])


class PPOUpdater:
    """Minibatch-epoch PPO update over one iteration's flattened episodes."""

    def __init__(self, *, policy: nn.Module, group_encoder: nn.Module, critic: nn.Module,
                 config: PPOConfig):
        self.policy, self.group_encoder, self.critic = policy, group_encoder, critic
        self.cfg = config
        self.actor_params = list(policy.parameters()) + list(group_encoder.parameters())
        self.actor_opt = torch.optim.Adam(self.actor_params, lr=config.lr_actor)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=config.lr_critic)
        self.alpha = 0.0

    # ---------------------------------------------------------------- values
    @torch.no_grad()
    def critic_values(self, states: dict[str, Tensor]) -> Tensor:
        n = next(iter(states.values())).shape[0]
        chunk = max(1, int(self.cfg.eval_chunk_frames))
        return torch.cat([self.critic(compact_present({k: v[i:i + chunk]
                                                       for k, v in states.items()})[0])
                          for i in range(0, n, chunk)]) if n else torch.zeros(0)

    def values(self, batch: FlatBatch) -> tuple[Tensor, Tensor]:
        """Critic values (scaled space) of all frames and of the bootstrap states."""
        return self.critic_values(batch.states), self.critic_values(batch.final_states)

    # ---------------------------------------------------------------- policy terms
    def _policy_terms(self, batch: FlatBatch, f: Tensor) -> dict:
        states, slots = _frames(batch, f)
        steps = {k: _gather(v[f], slots) for k, v in batch.steps.items()}
        ev = evaluate_actions(self.policy, self.group_encoder, states, steps["lane_mask"],
                              steps["latent"], steps["lane_choice"])
        index = ev["index"]
        acting = ev["cav"] & steps["cav_mask"].gather(1, index)
        old = steps["logp_accel"].gather(1, index) + steps["logp_lane"].gather(1, index)
        new = ev["logp_accel"] + ev["logp_lane"]
        return {"acting": acting, "new": new[acting], "old": old[acting],
                "new_lane": ev["logp_lane"][acting],
                "old_lane": steps["logp_lane"].gather(1, index)[acting],
                "entropy_accel": ev["entropy_accel"][acting],
                "entropy_lane": ev["entropy_lane"][acting],
                "frame_of_sample": acting.nonzero(as_tuple=False)[:, 0]}

    def _minibatches(self, n: int, generator: torch.Generator | None = None) -> list[Tensor]:
        perm = torch.randperm(n, generator=generator)
        m = max(1, min(int(self.cfg.minibatch_frames), n))
        return [perm[i:i + m] for i in range(0, n, m)]

    # ---------------------------------------------------------------- diagnostic
    def gradient_noise(self, batch: FlatBatch, adv_norm: Tensor, stop_check=None) -> dict:
        """Per-minibatch policy-gradient statistics at the rollout weights (no step taken)."""
        grads, norms = [], []
        for f in self._minibatches(batch.n_frames):
            if stop_check is not None and stop_check():
                raise StopRequested("stop requested during the gradient-noise pass")
            terms = self._policy_terms(batch, f)
            if terms["new"].numel() == 0:
                continue
            a = adv_norm[f][terms["frame_of_sample"]]
            loss = -(torch.exp(terms["new"] - terms["old"].detach()) * a).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            loss.backward()
            g = _flat_grad(self.actor_params).detach().double()
            grads.append(g)
            norms.append(float(g.norm()))
        self.actor_opt.zero_grad(set_to_none=True)
        m = len(grads)
        out = {"pg_grad_minibatches": m}
        if m == 0:
            return out
        mean = torch.stack(grads).mean(0)
        out["pg_grad_mean_norm"] = float(mean.norm())
        out["pg_grad_norm_mean"] = float(sum(norms) / m)
        if m >= 2:
            trace = float(sum(float((g - mean).square().sum()) for g in grads) / (m - 1))
            norm_mean = out["pg_grad_norm_mean"]
            out["pg_grad_norm_std"] = float((sum((x - norm_mean) ** 2 for x in norms) / (m - 1)) ** 0.5)
            out["pg_grad_var_trace"] = trace
            out["pg_grad_var_trace_batch"] = trace / m
            out["pg_grad_snr"] = float(mean.square().sum()) / trace if trace > 0 else None
        return out

    # ---------------------------------------------------------------- update
    def update(self, batch: FlatBatch, *, advantages: Tensor, returns: Tensor,
               cost_returns: Tensor, scaler: ReturnScaler, grad_noise: bool = False,
               stop_check=None) -> dict:
        """One PPO update; ``stop_check()`` is polled before every minibatch (raises
        :class:`StopRequested`, leaving the caller to discard the iteration)."""
        cfg = self.cfg
        n_cav = batch.steps["cav_mask"].sum(-1)
        n_samples = int(n_cav.sum())
        if n_samples == 0:
            raise ValueError("no acting CAV slots in this iteration's batch")
        adv = sample_normalized(advantages, n_cav)
        cost_adv = sample_normalized(cost_returns, n_cav)
        target = scaler.scale(returns) if cfg.return_norm else returns
        value_old = self.critic_values(batch.states)
        diag = {"samples": n_samples, "frames": batch.n_frames}
        if grad_noise:
            diag.update(self.gradient_noise(batch, adv, stop_check))
        sums = dict.fromkeys(("pi_loss", "cost_surrogate", "value_loss", "entropy_accel",
                              "entropy_lane", "approx_kl", "approx_kl_lane", "clip_fraction",
                              "actor_grad_norm", "critic_grad_norm"), 0.0)
        max_actor_norm = 0.0
        updates, stop = 0, False
        for _ in range(cfg.epochs):
            for f in self._minibatches(batch.n_frames):
                if stop_check is not None and stop_check():
                    raise StopRequested("stop requested during the PPO update")
                terms = self._policy_terms(batch, f)
                value = self.critic(_frames(batch, f)[0])
                v_loss = clipped_value_loss(value, value_old[f], target[f], cfg.clip_value_eps).mean()
                loss = cfg.value_coef * v_loss
                has_samples = terms["new"].numel() > 0
                if has_samples:
                    s = terms["frame_of_sample"]
                    ratio = torch.exp(torch.clamp(terms["new"] - terms["old"], -20.0, 20.0))
                    pi_loss = -clipped_surrogate(ratio, adv[f][s], cfg.clip_eps).mean()
                    cost_surr = pessimistic_cost_surrogate(ratio, cost_adv[f][s], cfg.clip_eps).mean()
                    ent_a, ent_l = terms["entropy_accel"].mean(), terms["entropy_lane"].mean()
                    loss = loss + (pi_loss + self.alpha * cost_surr) / (1.0 + self.alpha) \
                        - cfg.entropy_coef * ent_a - cfg.lane_entropy_coef * ent_l
                self.actor_opt.zero_grad(set_to_none=True)
                self.critic_opt.zero_grad(set_to_none=True)
                loss.backward()
                a_norm = float(nn.utils.clip_grad_norm_(self.actor_params, cfg.max_grad_norm))
                c_norm = float(nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm))
                self.actor_opt.step()
                self.critic_opt.step()
                updates += 1
                sums["value_loss"] += float(v_loss.detach())
                sums["actor_grad_norm"] += a_norm
                sums["critic_grad_norm"] += c_norm
                max_actor_norm = max(max_actor_norm, a_norm)
                if has_samples:
                    with torch.no_grad():
                        log_ratio = terms["old"] - terms["new"]
                        sums["pi_loss"] += float(pi_loss)
                        sums["cost_surrogate"] += float(cost_surr)
                        sums["entropy_accel"] += float(ent_a)
                        sums["entropy_lane"] += float(ent_l)
                        sums["approx_kl"] += float(log_ratio.mean())
                        sums["approx_kl_lane"] += float((terms["old_lane"] - terms["new_lane"]).mean())
                        sums["clip_fraction"] += float(((ratio - 1.0).abs() > cfg.clip_eps).float().mean())
                if cfg.target_kl and sums["approx_kl"] / updates > 1.5 * cfg.target_kl:
                    stop = True
                    break
            if stop:
                break
        for key in sums:
            diag[key] = sums[key] / max(1, updates)
        diag.update(actor_grad_norm_max=max_actor_norm, updates=updates, kl_early_stop=stop,
                    lagrange_alpha=self.alpha)
        diag["finite"] = all(math.isfinite(diag[k]) for k in
                             ("pi_loss", "value_loss", "entropy_accel", "approx_kl"))
        return diag
