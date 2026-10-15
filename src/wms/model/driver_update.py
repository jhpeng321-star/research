"""WMLANE C: in-rollout driver-setting update from observed motion.

``model.driver_update = "ema"``: whenever a car's TRUE state is observed during a
rollout (deployment: an RSU/belief update; training: a teacher-observed prefix),
its driver settings are refined from that observation:

* **desired speed v0** - free-flow inversion of the IDM: with the observed speed
  ``v``, acceleration ``a`` and leader interaction term ``(s*/s)^2`` the IDM gives
  ``(v/v0)^4 = 1 - a/a_max - (s*/s)^2 =: r``, so ``v0_hat = v / r^(1/4)``.  The
  estimate is trusted only in free flow (``r`` well above zero, small interaction,
  ``v > 2 m/s``); ``v0 <- v0 + rate_v0 * w_free * (v0_hat - v0)``.
* **time headway T** - steady car following (leader present, ``|dv|`` and ``|a|``
  small): ``T_hat = (gap - s0) / v``; ``T <- T + rate_T * w_follow * (T_hat - T)``.
* **embedding** - a zero-initialised linear map of the one-step innovation
  (observed minus predicted speed/acceleration, the two weights and the two
  parameter innovations) refines the residual-net conditioning embedding.

The three rates are learnable (sigmoid-parameterised, initialised at
``driver_update_rate``) and are trained by the reveal mode of the trainer
(``model.driver_reveal_prob``).  With no observation the update is never called,
so open-loop imagination is unchanged (no-op).
"""
from __future__ import annotations

from typing import Mapping, Optional

import torch
from torch import Tensor, nn

from wms.physics.config import PhysicsConfig
from wms.physics.neighbours import lead_neighbour

from .driver_encoder import PARAMETER_LOWER, PARAMETER_NAMES, PARAMETER_UPPER

#: Innovation features fed to the embedding refinement.
INNOVATION_F = 6
_V0 = PARAMETER_NAMES.index("v0")
_T = PARAMETER_NAMES.index("time_headway")


def _soft_step(x: Tensor, lo: float, hi: float) -> Tensor:
    """0 below ``lo``, 1 above ``hi``, linear in between."""
    return ((x - lo) / (hi - lo)).clamp(0.0, 1.0)


def driver_observation_estimates(kin: Tensor, present: Tensor, physics: PhysicsConfig,
                                 params: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Per-car ``v0_hat``/``w_free`` and ``T_hat``/``w_follow`` from an observed state ``[B,K,7]``."""
    v, a = kin[..., 4], kin[..., 5]
    gap, lead_v = lead_neighbour(kin, present, physics)
    has_leader = gap < float(physics.free_road_gap_m) - 1.0e-3
    a_max = params["accel"].clamp_min(0.1)
    b_comf = params["decel"].clamp_min(0.1)
    t_head = params["time_headway"]
    s0 = params["min_gap"]
    closing = v - lead_v
    desired = s0 + (v * t_head + v * closing / (2.0 * (a_max * b_comf).sqrt())).clamp_min(0.0)
    interaction = (desired / gap.clamp_min(0.3)).pow(2)
    interaction = torch.where(has_leader, interaction, torch.zeros_like(interaction))
    r = 1.0 - a / a_max - interaction
    r_safe = r.clamp(0.05, 1.5)
    v0_hat = (v / r_safe.pow(0.25)).clamp(PARAMETER_LOWER[_V0], PARAMETER_UPPER[_V0])
    w_free = (_soft_step(r, 0.3, 0.6) * (1.0 - _soft_step(interaction, 0.05, 0.25))
              * _soft_step(v, 1.0, 3.0))
    t_hat = ((gap - s0) / v.clamp_min(0.5)).clamp(PARAMETER_LOWER[_T], PARAMETER_UPPER[_T])
    w_follow = (has_leader.to(v.dtype) * (1.0 - _soft_step(closing.abs(), 0.5, 1.5))
                * (1.0 - _soft_step(a.abs(), 0.3, 0.8)) * _soft_step(v, 2.0, 4.0)
                * (1.0 - _soft_step(gap, 60.0, 100.0)))
    return {"v0_hat": v0_hat, "w_free": w_free, "t_hat": t_hat, "w_follow": w_follow}


class EmaDriverUpdater(nn.Module):
    """Learnable-rate EMA refinement of (v0, T) plus an innovation embedding update."""

    def __init__(self, embed_dim: int, rate_init: float = 0.3,
                 driver_ema_update: bool = True) -> None:
        super().__init__()
        if not isinstance(driver_ema_update, bool):
            raise ValueError("driver_ema_update must be a boolean")
        self.driver_ema_update = driver_ema_update
        logit = float(torch.logit(torch.tensor(float(rate_init))))
        self.rate_logits = nn.Parameter(torch.full((3,), logit))  # v0, T, embedding
        self.embed_proj = nn.Linear(INNOVATION_F, int(embed_dim))
        nn.init.zeros_(self.embed_proj.weight)
        nn.init.zeros_(self.embed_proj.bias)

    @property
    def rates(self) -> Tensor:
        return torch.sigmoid(self.rate_logits)

    def forward(self, params: Mapping[str, Tensor], embedding: Optional[Tensor], obs_kin: Tensor,
                present: Tensor, observed: Tensor, physics: PhysicsConfig, *,
                predicted_v: Optional[Tensor] = None,
                predicted_accel: Optional[Tensor] = None
                ) -> tuple[dict[str, Tensor], Optional[Tensor]]:
        """Return updated ``(params, embedding)``; only ``observed`` slots change.

        ``obs_kin`` is the post-observation state ``[B, K, 7]`` (observed slots carry
        their true kinematics; neighbours may be predicted).  ``predicted_v`` /
        ``predicted_accel`` are the model's own one-step prediction for the same tick
        (innovation features of the embedding update; zeros when absent).
        """
        est = driver_observation_estimates(obs_kin, present, physics, params)
        rates = self.rates.to(obs_kin.dtype)
        obs = observed.bool() & present.bool()
        w_v0 = rates[0] * est["w_free"]
        w_t = rates[1] * est["w_follow"]
        new_params = dict(params)
        # V6: disable only the algebraic parameter updates; observation-driven
        # embedding refinement remains available and trained by prediction error.
        if self.driver_ema_update:
            new_params["v0"] = torch.where(
                obs, params["v0"] + w_v0 * (est["v0_hat"] - params["v0"]), params["v0"])
            new_params["time_headway"] = torch.where(
                obs, params["time_headway"] + w_t * (est["t_hat"] - params["time_headway"]),
                params["time_headway"])
        if embedding is None:
            return new_params, None
        v, a = obs_kin[..., 4], obs_kin[..., 5]
        dv = torch.zeros_like(v) if predicted_v is None else v - predicted_v
        da = torch.zeros_like(a) if predicted_accel is None else a - predicted_accel
        innovation = torch.stack((dv / 3.0, da / 3.0, est["w_free"], est["w_follow"],
                                  est["w_free"] * (est["v0_hat"] - params["v0"]) / 10.0,
                                  est["w_follow"] * (est["t_hat"] - params["time_headway"])),
                                 -1)
        step = rates[2] * torch.tanh(self.embed_proj(innovation))
        new_embedding = torch.where(obs.unsqueeze(-1), embedding + step.to(embedding.dtype),
                                    embedding)
        return new_params, new_embedding
