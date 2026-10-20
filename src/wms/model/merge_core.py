"""Learned LC2013-like secure-gap merge core (``model.merge_decision='secure_gap'``).

The C4 diagnosis (``merge_sumo_diag/LC2013_MECHANISM.md`` section 4) shows that the SUMO
merges are a low-speed zipper in a queue: a ramp car accepts a two-sided car-following
secure gap, with a fitted anticipation.  This module implements that structure with every
behavioural parameter learned from recorded TRAIN data:

* positive parameters ``s0``, ``T``, ``b`` and the anticipation ``tau_a`` are
  ``softplus(raw)`` and initialised at the values fitted by
  :mod:`tools.merge_core_fit` (``merge_core_init.json`` next to this file);
* a persistent per-driver assertiveness latent ``alpha = exp(loc + sigma * eps)`` divides the
  required gaps (the ``lcAssertive`` analogue).  ``loc`` is a zero-initialised function of the
  driver encoding and ``sigma`` is a learned positive scale.  ``eps`` is drawn once per car and
  carried in the rollout state (``merge_core_alpha``) exactly like ``merge_threshold_u``; a
  slot that was absent and is present again is a new car and redraws it;
* a zero-initialised bounded residual is added to the margin;
* the decision is the smooth probability ``sigmoid(margin / temperature)`` with a learned
  positive temperature.  The rollout samples that probability per feasible in-zone step, so
  every recorded merge has positive likelihood (no zero-mass events).  The hard physical
  envelope (envelope-feasible, in-zone) is still a veto and is never relaxed;
* a small zero-initialised follower-yield head acts on the lane-1 follower of a ramp car whose
  follower margin is near zero, trained on recorded follower accelerations.

Everything here is created only when ``ModelSpec.merge_decision == 'secure_gap'``: the default
``'hazard'`` path never creates the module and is bit-identical.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .merge_negotiation import scatter_follower_accel
from .merge_threshold import target_gap_configuration

INIT_FILE = Path(__file__).with_name("merge_core_init.json")
#: Fallback initialisation (the fitted JSON is the authoritative init when present).
DEFAULT_INIT = {
    "s0_m": 2.1834043522406286, "T_s": 0.8101083991445911,
    "b_mps2": 1.5961324661965697, "tau_a_s": 0.9874870780612045,
    "temperature_m": 1.0591040636807836, "alpha_log_sigma": 0.4462892582004192,
    "residual_scale_m": 0.5, "yield_scale_mps2": 1.5,
}
#: Metric features of the margin residual head.
N_MARGIN_FEATURES = 8
#: Metric features of the follower-yield head.
N_YIELD_FEATURES = 7
#: Large sentinel margin used when a side of the gap is absent.
BIG_MARGIN = 1.0e3


def load_init(path: Optional[str | Path] = None) -> dict[str, float]:
    """Read the fitted initialisation JSON (fallback: :data:`DEFAULT_INIT`)."""
    target = INIT_FILE if path is None else Path(path)
    init = dict(DEFAULT_INIT)
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return init
    for key in DEFAULT_INIT:
        if key in raw:
            value = float(raw[key])
            if math.isfinite(value) and value > 0.0:
                init[key] = value
    return init


def _inv_softplus(value: float) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError("softplus init must be positive")
    return value if value > 20.0 else math.log(math.expm1(value))


def _softplus(raw: Tensor) -> Tensor:
    return torch.nn.functional.softplus(raw) + 1.0e-6


def secure_gap_margin(gap_l: Tensor, gap_f: Tensor, v: Tensor, v_l: Tensor, v_f: Tensor,
                      a: Tensor, a_l: Tensor, a_f: Tensor, has_l: Tensor, has_f: Tensor,
                      s0: Tensor, T: Tensor, b: Tensor, tau_a: Tensor, alpha: Tensor,
                      residual: Optional[Tensor] = None) -> Tensor:
    """Exact per-slot margin: ``min(gap_l - req_l/alpha, gap_f - req_f/alpha) + residual``."""
    ml, mf = secure_gap_sides(gap_l, gap_f, v, v_l, v_f, a, a_l, a_f, has_l, has_f,
                              s0, T, b, tau_a, alpha)
    margin = torch.minimum(ml, mf)
    return margin if residual is None else margin + residual


class MergeCoreNet(nn.Module):
    """Learned secure-gap core: positive physics parameters, assertiveness and residuals."""

    def __init__(self, input_dim: int = 0, hidden: int = 32,
                 init: Optional[dict[str, float]] = None,
                 residual_hidden: int = 32, yield_hidden: int = 32,
                 alpha_bound: float = 4.0) -> None:
        super().__init__()
        if input_dim < 0:
            raise ValueError("merge core input_dim must be >= 0")
        if hidden < 1 or residual_hidden < 1 or yield_hidden < 1:
            raise ValueError("merge core hidden widths must be >= 1")
        if not (math.isfinite(alpha_bound) and alpha_bound > 0.0):
            raise ValueError("merge core alpha_bound must be finite and positive")
        init = load_init() if init is None else {**DEFAULT_INIT, **init}
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.alpha_bound = float(alpha_bound)
        self.register_buffer("init_s0", torch.tensor(float(init["s0_m"])))
        self.s0_raw = nn.Parameter(torch.tensor(_inv_softplus(init["s0_m"])))
        self.t_raw = nn.Parameter(torch.tensor(_inv_softplus(init["T_s"])))
        self.b_raw = nn.Parameter(torch.tensor(_inv_softplus(init["b_mps2"])))
        self.tau_raw = nn.Parameter(torch.tensor(_inv_softplus(init["tau_a_s"])))
        self.temperature_raw = nn.Parameter(torch.tensor(_inv_softplus(init["temperature_m"])))
        self.alpha_log_sigma_raw = nn.Parameter(
            torch.tensor(math.log(float(init["alpha_log_sigma"]))))
        self.alpha_bias = nn.Parameter(torch.zeros(()))
        self.residual_scale_raw = nn.Parameter(
            torch.tensor(_inv_softplus(init["residual_scale_m"])))
        self.yield_scale_raw = nn.Parameter(torch.tensor(_inv_softplus(init["yield_scale_mps2"])))
        if self.input_dim > 0:
            self.alpha_net = nn.Sequential(
                nn.Linear(self.input_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
            nn.init.zeros_(self.alpha_net[-1].weight)
            nn.init.zeros_(self.alpha_net[-1].bias)
        else:
            self.alpha_net = None
        self.residual_head = nn.Sequential(
            nn.Linear(N_MARGIN_FEATURES, residual_hidden), nn.GELU(),
            nn.Linear(residual_hidden, 1))
        self.yield_head = nn.Sequential(
            nn.Linear(N_YIELD_FEATURES, yield_hidden), nn.GELU(),
            nn.Linear(yield_hidden, 1))
        for head in (self.residual_head, self.yield_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    # ------------------------------------------------------------------ parameters
    def params(self) -> dict[str, Tensor]:
        """Positive physics parameters and the margin temperature."""
        return {"s0": _softplus(self.s0_raw), "T": _softplus(self.t_raw),
                "b": _softplus(self.b_raw), "tau_a": _softplus(self.tau_raw),
                "temperature": _softplus(self.temperature_raw)}

    def alpha_scale(self) -> Tensor:
        """Learned positive assertiveness scale ``exp(log_sigma)`` (bounded)."""
        return torch.exp(self.alpha_log_sigma_raw.clamp(-4.0, 2.0))

    def alpha_location(self, features: Optional[Tensor], shape: tuple[int, ...]) -> Tensor:
        """Bounded per-slot log-assertiveness location (zero at creation)."""
        bias = self.alpha_bias.expand(shape)
        if self.alpha_net is None or features is None:
            raw = bias
        else:
            if features.shape[-1] != self.input_dim:
                raise ValueError(f"merge core needs {self.input_dim} driver features")
            raw = self.alpha_net(features.to(self.alpha_bias.dtype))[..., 0] + bias
        return self.alpha_bound * torch.tanh(raw)

    def forward(self, features: Optional[Tensor], shape: tuple[int, ...]) -> dict[str, Tensor]:
        """Per-slot log-assertiveness location and scale for ``shape`` slots."""
        location = self.alpha_location(features, shape)
        return {"alpha_log_location": location,
                "alpha_scale": self.alpha_scale().to(location.dtype).expand(shape)}

    # ------------------------------------------------------------------ heads
    def margin_residual(self, margin_l: Tensor, margin_f: Tensor, v: Tensor, v_l: Tensor,
                        v_f: Tensor, gap_l: Tensor, gap_f: Tensor,
                        intent: Optional[Tensor] = None) -> Tensor:
        """Small zero-initialised residual on the secure-gap margin (metres)."""
        zero = torch.zeros_like(v)
        intent = zero if intent is None else intent.clamp(0.0, 1.0)
        feats = torch.stack((margin_l / 10.0, margin_f / 10.0, v / 20.0, v_l / 20.0,
                             v_f / 20.0, gap_l / 50.0, gap_f / 50.0, intent), -1)
        feats = torch.nan_to_num(feats, nan=0.0, posinf=5.0, neginf=-5.0).clamp(-5.0, 5.0)
        return torch.tanh(self.residual_head(feats)[..., 0]) * _softplus(
            self.residual_scale_raw)

    def yield_accel(self, margin_f: Tensor, intent: Tensor, v: Tensor, v_l: Tensor,
                    v_f: Tensor, gap_f: Tensor, temperature: Tensor) -> Tensor:
        """Non-positive follower yield acceleration (zero at creation)."""
        near = torch.sigmoid(-margin_f.abs() / temperature.clamp_min(1.0e-2))
        feats = torch.stack((margin_f / 10.0, intent.clamp(0.0, 1.0), v / 20.0,
                             v_l / 20.0, v_f / 20.0, gap_f / 50.0,
                             ((v - v_f) / 10.0).clamp(-3.0, 3.0)), -1)
        feats = torch.nan_to_num(feats, nan=0.0).clamp(-5.0, 5.0)
        raw = self.yield_head(feats)[..., 0]
        return -_softplus(self.yield_scale_raw) * torch.tanh(raw / 2.0) * near


def secure_gap_sides(gap_l: Tensor, gap_f: Tensor, v: Tensor, v_l: Tensor, v_f: Tensor,
                     a: Tensor, a_l: Tensor, a_f: Tensor, has_l: Tensor, has_f: Tensor,
                     s0: Tensor, T: Tensor, b: Tensor, tau_a: Tensor,
                     alpha: Tensor) -> tuple[Tensor, Tensor]:
    """Anticipated leader / follower margins at assertiveness ``alpha``."""
    tau = tau_a.clamp(0.0, 5.0)
    v_ant = (v + a.clamp(-8.0, 4.0) * tau).clamp(0.0, 60.0)
    v_l_ant = (v_l + a_l.clamp(-8.0, 4.0) * tau).clamp(0.0, 60.0)
    v_f_ant = (v_f + a_f.clamp(-8.0, 4.0) * tau).clamp(0.0, 60.0)
    b_safe = b.clamp_min(1.0e-3)
    req_l = s0 + T * v_ant + torch.relu(v_ant.square() - v_l_ant.square()) / (2.0 * b_safe)
    req_f = s0 + T * v_f_ant + torch.relu(v_f_ant.square() - v_ant.square()) / (2.0 * b_safe)
    scale = alpha.clamp_min(1.0e-3)
    ml = torch.where(has_l.bool(), gap_l + (v_l - v) * tau - req_l / scale,
                     torch.full_like(gap_l, -BIG_MARGIN))
    mf = torch.where(has_f.bool(), gap_f + (v - v_f) * tau - req_f / scale,
                     torch.full_like(gap_f, -BIG_MARGIN))
    return ml, mf


def update_core_alpha(location: Tensor, sigma: Tensor, ramp: Tensor, present: Tensor, *,
                      alpha: Optional[Tensor] = None, has_alpha: Optional[Tensor] = None,
                      prev_present: Optional[Tensor] = None,
                      normal: Optional[Tensor] = None) -> dict[str, Tensor]:
    """One step of the persistent per-car assertiveness latent.

    A car draws ``alpha = exp(loc + sigma * eps)`` at its first ramp appearance and keeps it
    while present; an absent slot that becomes present again is a new car.  Returns the carried
    ``alpha`` / ``has_alpha`` / previous presence.
    """
    if alpha is None:
        alpha = torch.ones_like(location)
    if has_alpha is None:
        has_alpha = torch.zeros_like(present, dtype=torch.bool)
    if prev_present is None:
        prev_present = torch.zeros_like(present, dtype=torch.bool)
    present = present.bool()
    prev_present = prev_present.bool()
    new_car = present & ~prev_present
    keep = present & ~new_car & has_alpha.bool()
    alpha = torch.where(keep, alpha, torch.ones_like(alpha))
    has = keep
    need = present & ramp.bool() & ~has
    if bool(need.any()):
        if normal is None:
            eps = torch.randn(location.shape, dtype=location.dtype, device=location.device)
        else:
            eps = normal.to(device=location.device, dtype=location.dtype)
        draw = torch.exp((location.detach() + sigma.detach() * eps).clamp(-8.0, 8.0))
        alpha = torch.where(need, draw, alpha)
        has = has | need
    return {"alpha": alpha, "has_alpha": has, "prev_present": present}


def core_features(kin: Tensor, present: Tensor, physics,
                  *, intent: Optional[Tensor] = None) -> dict[str, Tensor]:
    """Secure-gap geometry of every ramp slot's lane-1 leader / follower."""
    gaps = target_gap_configuration(kin, present, physics)
    kin = kin.float()
    v = kin[..., 4]
    return {**gaps, "a": kin[..., 5],
            "intent": (torch.zeros_like(v) if intent is None else intent.clamp(0.0, 1.0))}


__all__ = ["INIT_FILE", "DEFAULT_INIT", "N_MARGIN_FEATURES", "N_YIELD_FEATURES",
           "MergeCoreNet", "load_init", "secure_gap_sides", "secure_gap_margin",
           "update_core_alpha", "core_features", "scatter_follower_accel"]
