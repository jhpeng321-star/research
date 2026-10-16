"""Long-horizon stability aids: contractivity, vehicle-count conservation, diagnostics.

Every item is opt-in.  The differentiable feasibility projection lives in
:mod:`wms.physics.projection` and the bounded corrections in
:mod:`wms.model.residual_bound`.

* :func:`residual_amplification` / :func:`stability_regulariser` /
  :func:`apply_spectral_norm` - contractivity aids: the finite-difference
  amplification of a small state perturbation through the residual and an
  optional spectral-norm parametrisation of the residual net.
* :class:`ConservationTracker` - per-step vehicle-count bookkeeping
  ``on_road == initial + entered - exited`` with a conservation-violation metric.
* :func:`error_growth_diagnostic` - per-horizon error ratio and an exponential
  error-growth rate for the evaluation harness.
* :func:`combine_losses` - loss-combination point accepting the section / traffic
  terms unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

import torch
from torch import Tensor, nn

__all__ = [
    "ConservationConfig",
    "ConservationResult",
    "ConservationTracker",
    "apply_spectral_norm",
    "combine_losses",
    "error_growth_diagnostic",
    "per_horizon_error",
    "residual_amplification",
    "stability_regulariser",
]


# --------------------------------------------------------------------------- #
# (B) Contractive residual: finite-difference amplification + spectral norm
# --------------------------------------------------------------------------- #
def _residual_accel(net: Callable[..., object], kin: Tensor, present: Tensor) -> Tensor:
    out = net(kin, present)
    if isinstance(out, Mapping):
        return out["accel_delta_mps2"]
    return out  # a bare callable returning the delta directly (tests / probes)


def residual_amplification(net: Callable[..., object], kin: Tensor, present: Tensor,
                           *, n_probes: int = 1, eps: float = 1.0e-3,
                           generator: Optional[torch.Generator] = None
                           ) -> Tensor:
    """Mean amplification ``||dr|| / ||dx||`` over random input perturbations.

    ``net`` is the residual module (e.g. ``model.residual_net``) or any callable
    ``(kin, present) -> accel_delta``.  A residual that is identically zero (the
    ``zero_residual`` gate) yields exactly ``0``; an input-amplifying residual
    yields a positive value.
    """
    if n_probes < 1:
        raise ValueError("n_probes must be >= 1")
    base = _residual_accel(net, kin, present)
    total = kin.new_zeros(())
    for _ in range(int(n_probes)):
        noise = torch.randn(kin.shape, dtype=kin.dtype, device=kin.device,
                            generator=generator) * float(eps)
        perturbed = _residual_accel(net, kin + noise, present)
        num = torch.linalg.vector_norm((perturbed - base).reshape(kin.shape[0], -1),
                                       dim=-1)
        den = torch.linalg.vector_norm(noise.reshape(kin.shape[0], -1), dim=-1)
        total = total + (num / den.clamp_min(1.0e-12)).mean()
    return total / float(n_probes)


def stability_regulariser(net: Callable[..., object], kin: Tensor, present: Tensor,
                          *, margin: float = 1.0, n_probes: int = 1,
                          eps: float = 1.0e-3,
                          generator: Optional[torch.Generator] = None) -> Tensor:
    """Penalise residual amplification above ``margin`` (identity-safe -> 0).

    ``margin=1.0`` means a residual that does not amplify a small perturbation
    contributes zero; anything that amplifies it contributes the excess.
    """
    amp = residual_amplification(net, kin, present, n_probes=n_probes, eps=eps,
                                 generator=generator)
    return torch.relu(amp - float(margin))


def apply_spectral_norm(net: nn.Module, *, name: str = "weight") -> nn.Module:
    """Parametrise every ``nn.Linear`` weight of ``net`` with spectral norm.

    Returns ``net`` in place; idempotent (a layer already parametrised by this
    name is skipped).  Used as the optional contractivity switch for the
    residual net.
    """
    for module in net.modules():
        if isinstance(module, nn.Linear):
            parametrizations = getattr(module, "parametrizations", None)
            if parametrizations is not None and name in parametrizations:
                continue
            nn.utils.parametrizations.spectral_norm(module, name=name)
    return net


# --------------------------------------------------------------------------- #
# (D) Vehicle-count bookkeeping and conservation
# --------------------------------------------------------------------------- #
@dataclass
class ConservationConfig:
    enabled: bool = False
    weight: float = 0.0


@dataclass
class ConservationResult:
    violation: Tensor                     # [B] signed, last step
    on_road: Tensor                       # [B] present count, last step
    balance: Tensor                       # [B] initial + entered - exited
    per_step_violation: list = field(default_factory=list)
    entered_total: Tensor = None
    exited_total: Tensor = None
    created_total: Tensor = None          # present but not a declared entrant

    @property
    def violation_abs_max(self) -> float:
        return float(self.violation.abs().max()) if self.violation.numel() else 0.0

    def summary(self) -> dict:
        return {
            "per_step_violation": [float(x) for x in self.per_step_violation],
            "violation_abs_max": self.violation_abs_max,
            "entered_total": [int(x) for x in self.entered_total.tolist()],
            "exited_total": [int(x) for x in self.exited_total.tolist()],
            "created_total": [int(x) for x in self.created_total.tolist()],
        }


class ConservationTracker:
    """Per-step ``cars on road == initial + entered - exited`` bookkeeping.

    ``update(present, entrant_mask=...)`` returns the signed conservation
    violation for that step.  ``entrant_mask`` is the per-step mask produced by
    the ``wms/entry2`` branch: a slot that appears *without* being declared an
    entrant is counted as an unexpected creation and shows up in the violation.
    Without the mask, newly-present slots are assumed to be legitimate entries
    so the identity still holds by construction when there are no exits.
    """

    def __init__(self, initial_present: Tensor,
                 config: Optional[ConservationConfig] = None) -> None:
        self.config = config or ConservationConfig()
        self.initial = initial_present.bool().sum(-1).long()
        self.prev = initial_present.bool().clone()
        self.entered = torch.zeros_like(self.initial)
        self.exited = torch.zeros_like(self.initial)
        self.created = torch.zeros_like(self.initial)
        self.history: list[float] = []

    def update(self, present: Tensor, *,
               entrant_mask: Optional[Tensor] = None) -> Tensor:
        present = present.bool()
        appear = present & ~self.prev
        disappear = self.prev & ~present
        if entrant_mask is None:
            entered = appear
            created = torch.zeros_like(appear)
        else:
            declared = entrant_mask.bool() & appear
            entered = declared
            created = appear & ~declared
        self.entered = self.entered + entered.sum(-1).long()
        self.exited = self.exited + disappear.sum(-1).long()
        self.created = self.created + created.sum(-1).long()
        balance = self.initial + self.entered - self.exited
        on_road = present.sum(-1).long()
        violation = (on_road - balance)
        self.prev = present
        self.history.append(float(violation.abs().max()))
        return violation

    def loss(self) -> Tensor:
        """Optional loss term: mean squared conservation violation over steps."""
        if not self.history:
            return torch.zeros((), dtype=torch.float32)
        return torch.as_tensor(self.history, dtype=torch.float32).pow(2).mean()

    def result(self, violation: Tensor, present: Tensor) -> ConservationResult:
        return ConservationResult(
            violation=violation, on_road=present.bool().sum(-1).long(),
            balance=self.initial + self.entered - self.exited,
            per_step_violation=list(self.history),
            entered_total=self.entered, exited_total=self.exited,
            created_total=self.created)

    def track_rollout(self, masks: Tensor, *,
                      entrant_seq: Optional[Tensor] = None) -> ConservationResult:
        """Fold a rollout's per-step ``masks [B, T, K]`` through ``update``."""
        if entrant_seq is not None and entrant_seq.shape[:2] != masks.shape[:2]:
            raise ValueError("entrant_seq must align with masks [B,T,K]")
        violation = None
        for i in range(masks.shape[1]):
            mask = None if entrant_seq is None else entrant_seq[:, i]
            violation = self.update(masks[:, i], entrant_mask=mask)
        if violation is None:
            violation = torch.zeros(masks.shape[0], dtype=torch.long)
        return self.result(violation, masks[:, -1])


# --------------------------------------------------------------------------- #
# (D) Error-growth diagnostic for the evaluation harness
# --------------------------------------------------------------------------- #
def per_horizon_error(pred: Tensor, target: Tensor, valid: Optional[Tensor] = None,
                      *, reduce: str = "mean") -> Tensor:
    """Per-horizon absolute error ``[H]`` from ``[B, H, K]`` predictions.

    ``valid`` selects the scored entries; the default scores every entry.
    """
    err = (pred - target).abs()
    if valid is not None:
        mask = valid.to(err.dtype)
        denom = mask.sum(dim=(0, 2)).clamp_min(1.0)
        if reduce == "mean":
            return (err * mask).sum(dim=(0, 2)) / denom
        if reduce == "median":
            out = []
            for h in range(err.shape[1]):
                sel = err[:, h][valid[:, h].bool()]
                out.append(sel.median() if sel.numel() else err.new_zeros(()))
            return torch.stack(out)
        raise ValueError(f"unknown reduce {reduce!r}")
    if reduce == "mean":
        return err.mean(dim=(0, 2))
    if reduce == "median":
        return err.median(dim=2).values.median(dim=0).values
    raise ValueError(f"unknown reduce {reduce!r}")


def error_growth_diagnostic(errors_by_horizon: Mapping[float, float]
                            ) -> dict[str, object]:
    """Per-horizon error ratios and an exponential error-growth rate.

    Fits ``log(error) = a + b * horizon`` by least squares over the strictly
    positive horizons; ``b`` is the growth rate per unit horizon.  Ratios are
    taken against the smallest horizon.  Robust to zero/negative errors (they
    are excluded from the fit and reported as ``None``).
    """
    items = sorted((float(h), float(e)) for h, e in errors_by_horizon.items())
    if not items:
        return {"horizons": [], "errors": [], "ratios": [], "growth_rate": None,
                "fit_r2": None, "fit_slope_per_s": None, "fit_intercept": None}
    horizons = [h for h, _ in items]
    errors = [e for _, e in items]
    e0 = errors[0]
    ratios = [None if e0 <= 0 else e / e0 for e in errors]
    pts = [(h, e) for h, e in items if e > 0]
    slope = intercept = r2 = None
    if len(pts) >= 2:
        xs = torch.tensor([h for h, _ in pts], dtype=torch.float64)
        ys = torch.tensor([e for _, e in pts], dtype=torch.float64).log()
        xm, ym = xs.mean(), ys.mean()
        denom = ((xs - xm) ** 2).sum()
        if float(denom) > 0.0:
            slope = float(((xs - xm) * (ys - ym)).sum() / denom)
            intercept = float(ym - slope * xm)
            pred = slope * xs + intercept
            ss_res = float(((ys - pred) ** 2).sum())
            ss_tot = float(((ys - ym) ** 2).sum())
            r2 = None if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot
    return {"horizons": horizons, "errors": errors, "ratios": ratios,
            "growth_rate": slope, "slope_per_horizon": slope,
            "fit_slope_per_s": slope, "fit_intercept": intercept, "fit_r2": r2}

# --------------------------------------------------------------------------- #
# (D) Loss combination hook (entry2 entrant mask + trafloss section terms)
# --------------------------------------------------------------------------- #
def combine_losses(base_loss: Tensor, *, conservation: Optional[Tensor] = None,
                   conservation_weight: float = 0.0,
                   stability: Optional[Tensor] = None,
                   stability_weight: float = 0.0,
                   extra_terms: Optional[Mapping[str, Tensor]] = None,
                   extra_weights: Optional[Mapping[str, float]] = None) -> Tensor:
    """Add the optional WM-S run-2 loss terms to a base loss.

    ``extra_terms`` is the clean merge point for the ``wms/trafloss`` section /
    traffic terms (they are added unchanged, weighted by ``extra_weights``).
    The ``wms/entry2`` contribution is already folded into ``conservation`` via
    its per-step entrant mask, so no term is implemented twice here.
    """
    total = base_loss
    if conservation is not None and conservation_weight:
        total = total + float(conservation_weight) * conservation
    if stability is not None and stability_weight:
        total = total + float(stability_weight) * stability
    for name, term in (extra_terms or {}).items():
        weight = 1.0 if extra_weights is None else float(extra_weights.get(name, 1.0))
        total = total + weight * term
    return total

