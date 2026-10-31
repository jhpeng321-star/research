"""Antithetic ES on the candidate-B residual output head (S-ESLONG conventions).

The perturbed parameter set is EXACTLY the S-ESLONG head-only scope: the linear
``residual_head`` weight+bias (137 scalars at hidden_dim=64: 136+1).  Zero head
is bitwise CTRL0 under the CAPREL law (S-ZEROSTART), so the search starts at
the CTRL0 execution.  The sigma rule is the registered command-scale
calibration: sigma = target residual RMS / local head sensitivity RMS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

WEIGHT_KEY = "residual_head.weight"
BIAS_KEY = "residual_head.bias"


def head_dimension(state_dict: dict) -> int:
    weight, bias = state_dict[WEIGHT_KEY], state_dict[BIAS_KEY]
    if weight.ndim != 2 or weight.shape[0] != 1 or bias.ndim != 1 or bias.shape[0] != 1:
        raise ValueError("residual head must be Linear(in_features, 1)")
    return int(weight.numel() + bias.numel())


def head_vector(state_dict: dict) -> torch.Tensor:
    head_dimension(state_dict)
    return torch.cat([state_dict[WEIGHT_KEY].double().flatten(),
                      state_dict[BIAS_KEY].double().flatten()])


def state_dict_with_head(state_dict: dict, theta: torch.Tensor) -> dict:
    weight, bias = state_dict[WEIGHT_KEY], state_dict[BIAS_KEY]
    dimension = head_dimension(state_dict)
    if theta.numel() != dimension:
        raise ValueError(f"theta has {theta.numel()} scalars; head needs {dimension}")
    out = dict(state_dict)
    flat = theta.detach().cpu().double().flatten()
    out[WEIGHT_KEY] = flat[:weight.numel()].reshape(weight.shape).to(weight.dtype)
    out[BIAS_KEY] = flat[weight.numel():].reshape(bias.shape).to(bias.dtype)
    return out


def sigma_for(target_residual_rms_mps2: float, head_sensitivity_rms: float) -> float:
    """Registered command-scale calibration (ESLONG choose_sigma, explicit target)."""
    if not (target_residual_rms_mps2 > 0 and head_sensitivity_rms > 0):
        raise ValueError("sigma rule needs a positive target and a positive head sensitivity")
    sigma = target_residual_rms_mps2 / head_sensitivity_rms
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    return sigma


def estimate(directions: torch.Tensor, deltas: torch.Tensor, sigma: float) -> torch.Tensor:
    """g = mean_d[(J_plus - J_minus) * eps_d] / (2 sigma); J is the SUMO loss."""
    return (deltas.to(directions)[:, None] * directions).mean(0) / (2 * sigma)


def panel_improved(before: list[float], after: list[float]) -> bool:
    """S-PILOTDIAG option (b) semantics: mean panel J must not worsen (no thresholds)."""
    if len(before) != len(after) or not before:
        raise ValueError("panel vectors must be non-empty and of equal length")
    return sum(after) <= sum(before)


@dataclass
class HeadAdam:
    lr: float
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    m: torch.Tensor | None = None
    v: torch.Tensor | None = None
    step_count: int = 0

    def snapshot(self) -> dict:
        return dict(lr=self.lr, beta1=self.beta1, beta2=self.beta2, eps=self.eps,
                    step_count=self.step_count,
                    m=None if self.m is None else self.m.tolist(),
                    v=None if self.v is None else self.v.tolist())

    def restore(self, snapshot: dict) -> None:
        self.lr, self.beta1 = snapshot["lr"], snapshot["beta1"]
        self.beta2, self.eps = snapshot["beta2"], snapshot["eps"]
        self.step_count = snapshot["step_count"]
        self.m = None if snapshot["m"] is None else torch.as_tensor(
            snapshot["m"], dtype=torch.float64).clone()
        self.v = None if snapshot["v"] is None else torch.as_tensor(
            snapshot["v"], dtype=torch.float64).clone()

    def step(self, theta: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        if not bool(torch.isfinite(grad).all()):
            raise ValueError("nonfinite ES gradient")
        if self.m is None or self.m.shape != theta.shape:
            self.m, self.v = torch.zeros_like(theta), torch.zeros_like(theta)
        self.step_count += 1
        self.m.lerp_(grad, 1 - self.beta1)
        self.v.lerp_(grad.square(), 1 - self.beta2)
        m_hat = self.m / (1 - self.beta1 ** self.step_count)
        v_hat = self.v / (1 - self.beta2 ** self.step_count)
        return theta - self.lr * m_hat / (v_hat.sqrt() + self.eps)


@dataclass
class EsState:
    theta: torch.Tensor
    adam: HeadAdam
    sigma_target_residual_rms_mps2: float
    sigma_min_mps2: float
    sigma_halve_every_accepted: int
    direction_seed: int
    accepted_count: int = 0
    update_count: int = 0
    panel_j: list[float] = field(default_factory=list)

    def sigma(self, head_sensitivity_rms: float) -> float:
        base = sigma_for(self.sigma_target_residual_rms_mps2, head_sensitivity_rms)
        halvings = self.accepted_count // max(1, self.sigma_halve_every_accepted)
        # The floor is in COMMAND units (m/s^2): sigma_min / sensitivity.
        return max(base * 0.5 ** halvings, self.sigma_min_mps2 / head_sensitivity_rms)

    def directions(self, update: int, count: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(self.direction_seed * (update + 1))
        return torch.randn(count, self.theta.numel(), generator=generator).double()

    def to_dict(self) -> dict:
        return dict(theta=self.theta.tolist(), adam=self.adam.snapshot(),
                    sigma_target_residual_rms_mps2=self.sigma_target_residual_rms_mps2,
                    sigma_min_mps2=self.sigma_min_mps2,
                    sigma_halve_every_accepted=self.sigma_halve_every_accepted,
                    direction_seed=self.direction_seed,
                    accepted_count=self.accepted_count,
                    update_count=self.update_count, panel_j=list(self.panel_j))

    @classmethod
    def from_dict(cls, data: dict) -> "EsState":
        adam = data["adam"]
        restored = HeadAdam(lr=adam["lr"], beta1=adam["beta1"], beta2=adam["beta2"],
                            eps=adam["eps"], step_count=adam["step_count"])
        restored.restore(adam)
        return cls(theta=torch.tensor(data["theta"], dtype=torch.float64),
                   adam=restored,
                   sigma_target_residual_rms_mps2=data["sigma_target_residual_rms_mps2"],
                   sigma_min_mps2=data["sigma_min_mps2"],
                   sigma_halve_every_accepted=data["sigma_halve_every_accepted"],
                   direction_seed=data["direction_seed"],
                   accepted_count=data["accepted_count"],
                   update_count=data["update_count"], panel_j=list(data["panel_j"]))
