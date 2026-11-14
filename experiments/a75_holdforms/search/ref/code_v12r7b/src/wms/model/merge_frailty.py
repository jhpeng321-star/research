"""Persistent per-driver frailty for the ramp merge (``model.merge_decision='frailty'``).

Round 5b's deterministic first-crossing threshold failed on the real data: most recorded
merges happened while the score was below an earlier feasible maximum, so their likelihood
mass was exactly zero (unsupported events).  The hazard-with-frailty alternative keeps the
per-step opportunity but makes it *consistent within a driver*: every ramp car draws ONE
latent ``w`` at its first appearance,

    w = loc + sigma * eps,          eps ~ Normal(0, 1),

and merges at a feasible in-zone step with probability

    h_t = sigmoid(z_t + w),

where ``z_t`` is the existing merge-head logit (plus the negotiation delta when active).
``loc`` is a small learned function of the driver encoding / own history (zero-initialised
and bounded; ``input_dim == 0`` gives one learned scalar), ``sigma`` is a learned positive
scale (``exp(log_sigma)``, initialised to 1).  Because ``h_t > 0`` for every finite score, a
recorded merge below an earlier feasible maximum still has positive likelihood; because
``w`` persists, a driver is not given a fresh independent threshold every 0.5 s.

``w`` and the previous presence are plain rollout-state entries (``merge_frailty_w`` /
``merge_frailty_present``) carried across truncated-BPTT chunks and slot moves.  A slot that
was absent and is present again is a new car and redraws its latent.  The deployment
decision samples ``w`` once and then a per-step Bernoulli with ``h_t``;
:mod:`wms.train.losses.merge_frailty` integrates ``w`` out (Gauss-Hermite) for training.

Everything here is OFF unless ``ModelSpec.merge_decision == "frailty"``: the default
``"hazard"`` path never creates the module and is bit-identical.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

#: Frailty location bound (keeps the learned location on the score's support).
LOCATION_BOUND = 8.0
#: Upper bound of the positive frailty scale (numerical safety of exp()).
SIGMA_MAX = 10.0


@lru_cache(maxsize=16)
def _hermite_nodes(n_quad: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Gauss-Hermite nodes / weights for a standard normal expectation.

    ``numpy.polynomial.hermite.hermgauss`` integrates against ``exp(-x^2)``; the standard
    normal change of variables gives ``E[f(Z)] = sum_q w_q f(sqrt(2) x_q)`` with
    ``w_q = w_q^GH / sqrt(pi)``.  Cached because the nodes never change.
    """
    if n_quad < 1:
        raise ValueError("n_quad must be >= 1")
    from numpy.polynomial.hermite import hermgauss

    nodes, weights = hermgauss(int(n_quad))
    scale = math.sqrt(math.pi)
    return (tuple(float(math.sqrt(2.0) * x) for x in nodes),
            tuple(float(w) / scale for w in weights))


def gauss_hermite(n_quad: int, *, dtype: torch.dtype = torch.float32,
                  device: Optional[torch.device] = None) -> tuple[Tensor, Tensor]:
    """Standard-normal quadrature ``(nodes [Q], weights [Q])`` summing to one."""
    nodes, weights = _hermite_nodes(int(n_quad))
    return (torch.tensor(nodes, dtype=dtype, device=device),
            torch.tensor(weights, dtype=dtype, device=device))


def frailty_hazard(z: Tensor, w: Tensor) -> Tensor:
    """Per-step merge hazard ``sigmoid(z + w)`` (numerically stable)."""
    return torch.sigmoid(z + w)


class MergeFrailtyNet(nn.Module):
    """Zero-initialised, bounded frailty location plus a learned positive scale.

    ``input_dim == 0`` (no structured driver encoding) gives one learned scalar location;
    otherwise a two-layer MLP maps the per-car driver embedding to a location in
    ``(-LOCATION_BOUND, LOCATION_BOUND)``.  Zero-initialised output weights make the location
    exactly 0 at creation.  ``log_sigma`` is a scalar parameter initialised to
    ``log(sigma_init)``; the scale is ``exp(log_sigma)`` clamped to ``SIGMA_MAX``.
    """

    def __init__(self, input_dim: int = 0, hidden: int = 32,
                 location_bound: float = LOCATION_BOUND,
                 sigma_init: float = 1.0) -> None:
        super().__init__()
        if input_dim < 0:
            raise ValueError("merge frailty input_dim must be >= 0")
        if hidden < 1:
            raise ValueError("merge frailty hidden width must be >= 1")
        if not (math.isfinite(location_bound) and location_bound > 0.0):
            raise ValueError("merge frailty location_bound must be finite and positive")
        if not (math.isfinite(sigma_init) and sigma_init > 0.0):
            raise ValueError("merge frailty sigma_init must be finite and positive")
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.location_bound = float(location_bound)
        self.sigma_init = float(sigma_init)
        self.location_bias = nn.Parameter(torch.zeros(()))
        self.log_sigma = nn.Parameter(torch.tensor(math.log(float(sigma_init))))
        if self.input_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden), nn.GELU(),
                nn.Linear(self.hidden, 1))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        else:
            self.net = None

    def sigma(self) -> Tensor:
        """Learned positive scale ``exp(log_sigma)`` (bounded)."""
        return torch.exp(self.log_sigma.clamp(max=math.log(SIGMA_MAX)))

    def forward(self, features: Optional[Tensor],
                shape: tuple[int, ...]) -> tuple[Tensor, Tensor]:
        """Location ``[B, K]`` and scale ``[B, K]`` (broadcast) for ``shape`` slots."""
        bias = self.location_bias.expand(shape)
        if self.net is None or features is None:
            loc = self.location_bound * torch.tanh(bias)
        else:
            if features.shape[-1] != self.input_dim:
                raise ValueError(f"merge frailty needs {self.input_dim} driver features")
            raw = self.net(features.to(self.location_bias.dtype))[..., 0]
            loc = self.location_bound * torch.tanh(raw + self.location_bias)
        return loc, self.sigma().to(loc.dtype).expand(shape)


def update_frailty_state(z: Tensor, ramp: Tensor, feasible: Tensor, present: Tensor,
                         location: Tensor, sigma: Tensor, *,
                         w: Optional[Tensor] = None,
                         has_w: Optional[Tensor] = None,
                         prev_present: Optional[Tensor] = None,
                         uniform: Optional[Tensor] = None,
                         normal: Optional[Tensor] = None) -> dict[str, Tensor]:
    """One step of the persistent frailty decision.

    ``ramp``: present in the ramp lane; ``feasible``: envelope-feasible AND in the merge
    zone (only feasible steps can merge); ``z``: merge-head logit (detached here; the
    likelihood re-evaluates it with gradient); ``location`` / ``sigma``: the learned latent
    distribution.  A car draws its latent at its first ramp appearance and keeps it while it
    stays present; a slot that was absent and is present again is a new car.  Returns
    ``cross`` (merge now), the sampled hazard ``hazard`` and the carried ``w`` / ``has_w`` /
    ``prev_present``.
    """
    if w is None:
        w = torch.zeros_like(z)
    if has_w is None:
        has_w = torch.zeros_like(present, dtype=torch.bool)
    if prev_present is None:
        prev_present = torch.zeros_like(present, dtype=torch.bool)
    z = z.detach()
    present = present.bool()
    prev_present = prev_present.bool()
    new_car = present & ~prev_present
    keep = present & ~new_car & has_w.bool()
    w = torch.where(keep, w, torch.zeros_like(w))
    has = keep
    need = present & ramp.bool() & ~has
    if bool(need.any()):
        if normal is None:
            eps = torch.randn(z.shape, dtype=z.dtype, device=z.device)
        else:
            eps = normal.to(device=z.device, dtype=z.dtype)
        draw = location.detach() + sigma.detach() * eps
        w = torch.where(need, draw, w)
        has = has | need
    hazard = frailty_hazard(z, w)
    if uniform is None:
        uniform = torch.rand(z.shape, dtype=z.dtype, device=z.device)
    cross = feasible.bool() & has & (uniform.to(z.dtype) < hazard)
    return {"w": w, "has_w": has, "prev_present": present, "hazard": hazard,
            "cross": cross}


__all__ = ["LOCATION_BOUND", "SIGMA_MAX", "MergeFrailtyNet", "frailty_hazard",
           "gauss_hermite", "update_frailty_state"]
