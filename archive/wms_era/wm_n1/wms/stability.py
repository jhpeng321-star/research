"""Run-2 long-horizon STABILITY features for the WM-S rollout.

Every feature here is an opt-in switch that defaults OFF.  With the flags off
the model and trainer are byte-identical to the pre-run-2 code: nothing in this
module is imported into a hot path unless the caller asks for it.  The module
contains four independent, unit-testable pieces plus two clean merge hooks:

* :class:`FeasibilityProjection` (item A) - differentiable post-step physics
  projection: min bumper gap, speed bounds, valid lane ids, no creation /
  removal.  It also logs how often each constraint was active.
* :func:`bounded_value` / :func:`bounded_residual` (item B) - tanh caps on the
  learned HDV acceleration residual and the lane-change logit adjustment.
* :func:`residual_amplification` / :func:`stability_regulariser` /
  :func:`apply_spectral_norm` (item B) - contractivity aids: the finite-
  difference amplification of a small state perturbation through the residual
  and an optional spectral-norm parametrisation of the residual net.
* :class:`ConservationTracker` (item D) - per-step vehicle-count bookkeeping
  ``on_road == initial + entered - exited`` with a conservation-violation
  metric and an optional loss.  It accepts a per-step entrant mask produced by
  the ``wms/entry2`` branch so the merge is clean.
* :func:`error_growth_diagnostic` (item D) - per-horizon error ratio and an
  exponential error-growth rate for the evaluation harness.
* :func:`combine_losses` (item D) - loss-combination point that accepts the
  section / traffic terms from the ``wms/trafloss`` branch unchanged.

The two external integrations (``wms/entry2`` entrant mask and ``wms/trafloss``
section terms) are *accepted*, not implemented here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

import torch
from torch import Tensor, nn

__all__ = [
    "ProjectionConfig",
    "FeasibilityProjection",
    "bounded_value",
    "bounded_residual",
    "residual_amplification",
    "stability_regulariser",
    "apply_spectral_norm",
    "ConservationConfig",
    "ConservationTracker",
    "ConservationResult",
    "per_horizon_error",
    "error_growth_diagnostic",
    "combine_losses",
]


# --------------------------------------------------------------------------- #
# (A) Differentiable physics feasibility projection after every rollout step
# --------------------------------------------------------------------------- #
@dataclass
class ProjectionConfig:
    """Constraints enforced by :class:`FeasibilityProjection`.

    ``min_gap_m`` is the bumper gap (leader rear to follower front) that must
    hold on every lane; ``v_max_mps``/``v_min_mps`` bound speed.  ``project_cavs``
    is OFF so a teacher-forced CAV keeps its recorded position exactly and only
    acts as an immovable obstacle for the learned HDVs.
    """

    min_gap_m: float = 2.0
    v_max_mps: float = 31.29
    v_min_mps: float = 0.0
    n_lanes: int = 3
    valid_lanes: tuple[int, ...] = (0, 1, 2)
    project_cavs: bool = False
    speed_tol_mps: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.min_gap_m < 0.0:
            raise ValueError("min_gap_m must be >= 0")
        if self.v_max_mps < self.v_min_mps:
            raise ValueError("v_max_mps must be >= v_min_mps")
        invalid = [int(x) for x in self.valid_lanes
                   if int(x) < 0 or int(x) >= self.n_lanes]
        if invalid:
            raise ValueError(f"valid_lanes outside [0, n_lanes): {invalid}")


class FeasibilityProjection:
    """Differentiable straight-through projection of a post-step state.

    ``__call__(s, v, lane, present, length, is_cav=None)`` returns projected
    ``s_m``/``v_mps``/``lane`` plus a per-step ``active`` dict of integer
    constraint counts.  Persistent totals are accumulated on ``self.stats`` and
    can be read/reset with :meth:`reset_stats` / :attr:`stats`.

    Gradients flow through the clamp/min operators (finite subgradients), so the
    projection can sit inside a training rollout.  ``present`` is never changed:
    the projection creates and removes no vehicle.
    """

    def __init__(self, config: Optional[ProjectionConfig] = None) -> None:
        self.config = config or ProjectionConfig()
        self.reset_stats()

    # -- stats ---------------------------------------------------------------
    def reset_stats(self) -> None:
        self.stats: dict[str, float] = {
            "steps": 0.0,
            "gap_before": 0.0,
            "gap_active": 0.0,
            "s_clamped": 0.0,
            "speed_high": 0.0,
            "speed_low": 0.0,
            "invalid_lane": 0.0,
            "present_changed": 0.0,
        }

    # -- call ----------------------------------------------------------------
    def __call__(self, s: Tensor, v: Tensor, lane: Tensor, present: Tensor,
                 length: Optional[Tensor] = None, *,
                 is_cav: Optional[Tensor] = None) -> dict[str, Tensor | dict]:
        cfg = self.config
        present = present.bool()
        if length is None:
            length = torch.full_like(s, 5.0)
        length = length.to(s.dtype)
        lane = lane.long()

        if is_cav is None:
            is_cav = torch.zeros_like(present)
        else:
            is_cav = is_cav.bool()
        projectable = present & (bool(cfg.project_cavs) | ~is_cav)

        # -- valid lane ids --------------------------------------------------
        lane_ok = (lane >= 0) & (lane < int(cfg.n_lanes))
        invalid_lane = present & ~lane_ok
        lane_proj = torch.where(invalid_lane,
                                lane.clamp(0, int(cfg.n_lanes) - 1), lane)

        # -- speed bounds (smooth clamps) ------------------------------------
        v_proj = v
        speed_high = present & (v > cfg.v_max_mps + cfg.speed_tol_mps)
        speed_low = present & (v < cfg.v_min_mps - cfg.speed_tol_mps)
        if bool(cfg.project_cavs):
            v_proj = v.clamp(cfg.v_min_mps, cfg.v_max_mps)
        else:
            v_clamped = v.clamp(cfg.v_min_mps, cfg.v_max_mps)
            v_proj = torch.where(projectable, v_clamped, v)

        # -- lane-wise min-gap projection ------------------------------------
        s_proj, gap_before, s_clamped = self._project_gaps(
            s, length, lane_proj, present, projectable, cfg.min_gap_m,
            n_lanes=int(cfg.n_lanes))

        active = {
            "gap_before": int(gap_before),
            "gap_active": int(s_clamped),
            "s_clamped": int(s_clamped),
            "speed_high": int(speed_high.sum()),
            "speed_low": int(speed_low.sum()),
            "invalid_lane": int(invalid_lane.sum()),
            "present_changed": 0,
        }
        self.stats["steps"] += 1.0
        for key, value in active.items():
            self.stats[key] += float(value)
        return {"s_m": s_proj, "v_mps": v_proj, "lane": lane_proj, "active": active}

    @staticmethod
    def _project_gaps(s: Tensor, length: Tensor, lane: Tensor, present: Tensor,
                      projectable: Tensor, min_gap: float, n_lanes: int = 3
                      ) -> tuple[Tensor, int, int]:
        """Push followers back until the leader gap is >= ``min_gap``.

        Vectorised per ``(row, lane)`` group: within a lane the vehicle order is
        sorted front-to-back and the front-to-back recurrence

            y[0] = x[0];  y[j] = min(x[j], y[j-1] - L[j-1] - min_gap)   (projectable)
                          y[j] = x[j]                                  (anchor)

        is evaluated in closed form as a **segmented prefix minimum** ending at
        the last anchor (a non-projectable vehicle or the lane leader):

            y[j] = min_{i in [anchor(j)..j]} (x[i] + S[i]) - S[j]

        with ``S`` the prefix sum of ``L + min_gap``.  This removes the previous
        per-vehicle Python loop and its host ``bool(...)`` syncs entirely; the
        result is built with gathers so every ``s`` keeps its autograd history.
        Returns ``(s_projected, gap_violations_before, s_clamped)`` exactly as the
        reference loop (parity-tested to 1e-6).
        """
        b, k = s.shape
        out = s.clone()
        device, dtype = s.device, s.dtype
        gap_before_total = torch.zeros((), dtype=torch.long, device=device)
        s_clamped_total = torch.zeros((), dtype=torch.long, device=device)
        thr_gap = float(min_gap) - 1.0e-6
        gap_floor = float(min_gap)
        inf = torch.full((), float("inf"), dtype=dtype, device=device)
        for row in range(int(b)):
            for ln in range(int(n_lanes)):
                sel = present[row] & (lane[row] == ln)
                sel_idx = torch.nonzero(sel, as_tuple=False).flatten()
                if sel_idx.numel() <= 1:
                    continue
                order = sel_idx[torch.argsort(s[row, sel_idx])]   # rearmost -> front
                rev = torch.flip(order, [0])                      # front -> back
                x = s[row, rev]
                length_rev = length[row, rev]
                m = int(x.numel())
                c = length_rev[1:] + gap_floor
                S = torch.zeros(m, dtype=dtype, device=device)
                if m > 1:
                    S[1:] = torch.cumsum(c, dim=0)
                t = x + S
                proj = projectable[row, rev]
                anchor = ~proj
                anchor[0] = True                                  # front is the leader
                idx = torch.arange(m, device=device)
                aidx = torch.where(anchor, idx,
                                   torch.full_like(idx, -1))
                anchor_start = torch.cummax(aidx, dim=0).values   # last anchor <= j
                mask = ((idx.unsqueeze(0) <= idx.unsqueeze(1))
                        & (idx.unsqueeze(0) >= anchor_start.unsqueeze(1)))
                tmin = torch.where(mask, t.unsqueeze(0).expand(m, m), inf).amin(dim=1)
                y = tmin - S
                out[row, rev] = torch.where(proj, y, x)
                gaps = x[:-1] - x[1:] - length_rev[:-1]           # front vs behind
                gap_before_total = gap_before_total + (gaps < thr_gap).sum()
                if m > 1:
                    # Same comparison grouping as the reference loop
                    # (``out[lead] - length[lead] - min_gap``), so the diagnostic
                    # clamp count is not inflated by the prefix-sum rounding.
                    cap = (y[:-1] - length_rev[:-1]) - gap_floor
                    s_clamped_total = s_clamped_total + (
                        proj[1:] & (cap < x[1:] - 1.0e-9)).sum()
        return out, int(gap_before_total), int(s_clamped_total)

    @staticmethod
    def _project_gaps_reference(s: Tensor, length: Tensor, lane: Tensor,
                                present: Tensor, projectable: Tensor, min_gap: float
                                ) -> tuple[Tensor, int, int]:
        """Original per-vehicle loop kept only as the parity reference."""
        b, k = s.shape
        out = s.clone()
        gap_before = 0
        s_clamped = 0
        for row in range(b):
            for ln in torch.unique(lane[row][present[row]]).tolist():
                idx = torch.nonzero(present[row] & (lane[row] == ln),
                                    as_tuple=False).flatten()
                if idx.numel() <= 1:
                    continue
                order = idx[torch.argsort(s[row, idx])]
                gaps = (s[row, order[1:]] - s[row, order[:-1]]
                        - length[row, order[1:]])
                gap_before += int((gaps < min_gap - 1.0e-6).sum())
                lead = order[-1]
                for p in range(order.numel() - 2, -1, -1):
                    cur = order[p]
                    if bool(projectable[row, cur]):
                        cap = out[row, lead] - length[row, lead] - min_gap
                        projected = torch.minimum(s[row, cur], cap)
                        if bool(projected < s[row, cur] - 1.0e-9):
                            s_clamped += 1
                        out[row, cur] = projected
                        lead = cur
                    else:
                        lead = cur
        return out, gap_before, s_clamped

    def summary(self) -> dict[str, float]:
        """Rates (per step / per projected vehicle) derived from the totals."""
        steps = max(self.stats["steps"], 1.0)
        total_active = (self.stats["gap_active"] + self.stats["speed_high"]
                        + self.stats["speed_low"] + self.stats["invalid_lane"])
        return {**self.stats, "steps": self.stats["steps"],
                "gap_active_rate": self.stats["gap_active"] / steps,
                "speed_active_rate": (self.stats["speed_high"]
                                      + self.stats["speed_low"]) / steps,
                "invalid_lane_rate": self.stats["invalid_lane"] / steps,
                "any_active_rate": total_active / steps}


# --------------------------------------------------------------------------- #
# (B) Bounded (tanh) learned correction
# --------------------------------------------------------------------------- #
def bounded_value(x: Tensor, cap: Optional[float]) -> Tensor:
    """Smoothly cap ``x`` to ``(-cap, cap)`` with ``tanh``; ``None`` is a no-op."""
    if cap is None:
        return x
    if cap <= 0.0:
        raise ValueError("cap must be > 0")
    return float(cap) * torch.tanh(x / float(cap))


def bounded_residual(delta: Tensor, cap_mps2: Optional[float],
                     lane_delta: Optional[Tensor] = None,
                     lane_cap: Optional[float] = None):
    """Cap the acceleration residual and, optionally, the lane logit delta.

    With both caps ``None`` the inputs are returned unchanged (byte-identical
    to the un-capped model).
    """
    accel = bounded_value(delta, cap_mps2)
    if lane_delta is None:
        return accel
    return accel, bounded_value(lane_delta, lane_cap)


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


def new_model_projection(config: Optional[ProjectionConfig] = None):
    """Small factory kept so callers never import the class name by accident."""
    return FeasibilityProjection(config)
