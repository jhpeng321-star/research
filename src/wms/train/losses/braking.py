"""Braking-response terms: asymmetric speed, braking-step weighting, backward propagation.

Offline diagnosis (DIAG120) found that long-horizon blow-ups are missed jams and
systematic under-braking: the error starts at one vehicle and propagates backward
car to car with a 1.0-1.5 s follower lag.  Each term is flag-gated; all masks and
multipliers are computed from detached values so gradients flow only through the
predicted quantity.

``asym_speed``   elements where the prediction is too fast get total weight ALPHA;
                 the logged term is ``(ALPHA - 1)`` x the too-fast Huber mass.
``decel_weight`` HDV steps whose recorded acceleration is below ``-A`` get total
                 weight W; the logged term is ``(W - 1)`` x the (asym-weighted) mass.
``propagation``  for every follower whose recorded same-lane leader brakes harder
                 than ``-A``, predicted speed and acceleration are matched to the
                 recording over the next S seconds (vectorised, no host sync).

The recorded acceleration at state ``t+1`` labels the ``t -> t+1`` transition.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import LossContext, LossTerm, TermValue, UnrollTargets

#: Pairwise leader-search tensor budget (elements) per time chunk.
_PAIRWISE_BUDGET = 4_000_000


@dataclass(frozen=True)
class BrakingParams:
    """Hyper-parameters of the braking terms (all default OFF)."""

    asym_alpha: float = 1.0
    decel_weight: float = 1.0
    decel_threshold_mps2: float = 3.0
    prop_weight: float = 0.0
    prop_window_s: float = 3.0
    prop_max_gap_m: Optional[float] = None
    prop_accel_huber_mps2: float = 1.0
    dt_s: float = 0.5

    def __post_init__(self) -> None:
        if not all(math.isfinite(float(x)) for x in vars(self).values() if x is not None):
            raise ValueError("braking hyper-parameters must be finite")
        if float(self.asym_alpha) < 1.0:
            raise ValueError("asym_alpha must be >= 1 (1 = off)")
        if float(self.decel_weight) < 1.0:
            raise ValueError("decel_weight must be >= 1 (1 = off)")
        if float(self.decel_threshold_mps2) <= 0.0:
            raise ValueError("decel_threshold_mps2 must be > 0 (a deceleration magnitude)")
        if float(self.prop_weight) < 0.0:
            raise ValueError("prop_weight must be >= 0 (0 = off)")
        if float(self.prop_window_s) <= 0.0 or float(self.dt_s) <= 0.0:
            raise ValueError("prop_window_s and dt_s must be positive")
        if (self.prop_max_gap_m is not None and float(self.prop_max_gap_m) <= 0.0) \
                or float(self.prop_accel_huber_mps2) <= 0.0:
            raise ValueError("prop_max_gap_m and prop_accel_huber_mps2 must be positive")

    @property
    def asym_on(self) -> bool:
        return float(self.asym_alpha) > 1.0

    @property
    def decel_on(self) -> bool:
        return float(self.decel_weight) > 1.0

    @property
    def prop_on(self) -> bool:
        return float(self.prop_weight) > 0.0

    @property
    def any_on(self) -> bool:
        return self.asym_on or self.decel_on or self.prop_on

    @property
    def window_steps(self) -> int:
        return max(1, int(round(float(self.prop_window_s) / float(self.dt_s))))

    @property
    def log_terms(self) -> tuple[str, ...]:
        """Loss-term names this config adds to the train log / loss share."""
        return tuple(name for name, on in (("asym_speed", self.asym_on),
                                           ("decel_weight", self.decel_on),
                                           ("propagation", self.prop_on)) if on)

    @classmethod
    def from_flat(cls, flat: dict, dt_s: float = 0.5) -> "BrakingParams":
        """Build from the legacy flat keys (receipt / config dict), defaults for absent keys."""
        gap = flat.get("prop_max_gap_m")
        return cls(asym_alpha=float(flat.get("asym_speed_alpha", 1.0)),
                   decel_weight=float(flat.get("decel_weight", 1.0)),
                   decel_threshold_mps2=float(flat.get("decel_threshold_mps2", 3.0)),
                   prop_weight=float(flat.get("propagation_weight", 0.0)),
                   prop_window_s=float(flat.get("prop_window_s", 3.0)),
                   prop_max_gap_m=None if gap is None else float(gap), dt_s=float(dt_s))


def asym_multiplier(pred: Tensor, target: Tensor, alpha: float) -> Tensor:
    """``alpha`` where ``pred > target`` (too fast / too far ahead) else ``1``."""
    return 1.0 + (float(alpha) - 1.0) * (pred.detach() > target).to(pred.dtype)


def braking_step_mask(a_rec: Tensor, valid: Tensor, threshold: float) -> Tensor:
    """Valid HDV steps whose recorded acceleration is below ``-threshold``."""
    return valid.bool() & (a_rec < -float(threshold))


def decel_multiplier(a_rec: Tensor, valid: Tensor, weight: float, threshold: float) -> Tensor:
    """``weight`` on braking steps, ``1`` elsewhere (float, detached)."""
    mask = braking_step_mask(a_rec, valid, threshold)
    return 1.0 + (float(weight) - 1.0) * mask.to(a_rec.dtype)


@torch.no_grad()
def leader_brake_events(s: Tensor, lane: Tensor, valid: Tensor, length: Tensor,
                        a_next: Tensor, *, threshold: float, max_gap_m: Optional[float] = None,
                        budget: int = _PAIRWISE_BUDGET) -> Tensor:
    """Per follower slot and transition: does its recorded leader brake hard?

    ``s, lane, valid, length`` are ``[B, T+1, K]`` recorded sequences; ``a_next``
    is the recorded acceleration of the ``t -> t+1`` transition (``[B, T, K]``).
    Returns ``[B, T, K]`` bool: the slot's nearest same-lane vehicle ahead
    (optionally within ``max_gap_m``) is still present at ``t + 1`` and brakes
    harder than ``-threshold``.  Pairwise search runs per time chunk (bounded memory).
    """
    b_n, t1, k_n = s.shape
    t_n = t1 - 1
    chunk = max(1, int(budget // max(1, b_n * k_n * k_n)))
    out = []
    for t0 in range(0, t_n, chunk):
        t1_ = min(t_n, t0 + chunk)
        st, ln, vd, le = (x[:, t0:t1_] for x in (s, lane, valid, length))
        ds = st.unsqueeze(-2) - st.unsqueeze(-1)            # [B,c,i,j] = s_j - s_i
        same = ln.unsqueeze(-2) == ln.unsqueeze(-1)
        cand = same & vd.unsqueeze(-2) & vd.unsqueeze(-1) & (ds > 0)
        dist = torch.where(cand, ds, torch.full_like(ds, float("inf")))
        dmin, jdx = dist.min(dim=-1)                          # nearest leader j
        has = torch.isfinite(dmin)
        gap = dmin - torch.gather(le, -1, jdx)
        a_lead = torch.gather(a_next[:, t0:t1_], -1, jdx)
        still = torch.gather(valid[:, t0 + 1:t1_ + 1], -1, jdx)
        close = torch.ones_like(has) if max_gap_m is None else gap <= float(max_gap_m)
        out.append(has & still & close & (a_lead < -float(threshold)))
    if not out:
        return torch.zeros(b_n, 0, k_n, dtype=torch.bool, device=s.device)
    return torch.cat(out, dim=1)


@torch.no_grad()
def propagation_window(events: Tensor, window_steps: int) -> Tensor:
    """Transitions within ``window_steps`` AFTER an onset (``t+1 .. t+L``), via ``cumsum``."""
    b_n, t_n, k_n = events.shape
    csum = torch.cat((torch.zeros(b_n, 1, k_n, dtype=torch.int32, device=events.device),
                      events.to(torch.int32).cumsum(dim=1)), dim=1)   # csum[u]=sum ev[<u]
    u = torch.arange(t_n, device=events.device)
    lo = (u - int(window_steps)).clamp_min(0)
    return (csum[:, u] - csum[:, lo]) > 0


def propagation_terms(pred_speed: Tensor, pred_accel: Tensor, rec_speed: Tensor,
                      rec_accel: Tensor, mask: Tensor, weights: Tensor, denom: Tensor, *,
                      speed_delta: float, accel_delta: float) -> Tensor:
    """Masked, step-weighted Huber of follower speed and acceleration (0-d, no sync)."""
    spd = F.huber_loss(pred_speed, rec_speed, delta=float(speed_delta), reduction="none")
    acc = F.huber_loss(pred_accel, rec_accel, delta=float(accel_delta), reduction="none")
    m = mask.to(weights.dtype)
    return ((spd + acc) * weights * m).sum() / denom


def realized_acceleration(speeds: Tensor, initial_speed: Tensor, dt_s: float) -> Tensor:
    """Acceleration of the actual predicted speed transitions (speed floor, projection, noise)."""
    previous = torch.cat((initial_speed.unsqueeze(1), speeds[:, :-1]), dim=1)
    return (speeds - previous) / float(dt_s)


@dataclass
class BrakingTargets:
    """Unroll-constant recorded braking data shared by the three braking terms."""

    mask: Tensor                      # HDV, non-CAV supervised steps [B, n, K]
    a_rec: Tensor                     # recorded transition acceleration [B, n, K]
    brake_steps: Tensor               # count of recorded braking steps
    prop_mask: Optional[Tensor]       # propagation window mask or None


def prepare_braking(params: BrakingParams, targets: UnrollTargets) -> BrakingTargets:
    """Compute (once per unroll) the recorded braking masks; cached on ``targets``."""
    cached = targets.prepared.get("braking")
    if cached is not None:
        return cached
    n = targets.n_steps
    states = targets.states
    mask = targets.hdv_mask & ~targets.is_cav[:, 1:n + 1]
    a_rec = states[:, 1:n + 1, :, 5]
    brake_steps = (mask & (a_rec < -float(params.decel_threshold_mps2))).sum()
    prop_mask = None
    if params.prop_on:
        events = leader_brake_events(
            states[:, :n + 1, :, 0], states[:, :n + 1, :, 1:4].argmax(-1),
            targets.loss_valid[:, :n + 1], states[:, :n + 1, :, 6], a_rec,
            threshold=float(params.decel_threshold_mps2), max_gap_m=params.prop_max_gap_m)
        prop_mask = propagation_window(events, params.window_steps) & mask
    result = BrakingTargets(mask, a_rec, brake_steps, prop_mask)
    targets.prepared["braking"] = result
    return result


class _BrakingTerm(LossTerm):
    """Base: shares the per-unroll targets and the per-chunk weighted base losses."""

    def __init__(self, params: BrakingParams, *, position_weight: float,
                 speed_weight: float) -> None:
        self.params = params
        self.position_weight = float(position_weight)
        self.speed_weight = float(speed_weight)

    def prepare(self, targets: UnrollTargets) -> None:
        prepare_braking(self.params, targets)

    def _chunk(self, ctx: LossContext) -> dict:
        """Weighted base losses and asym multipliers of this chunk (cached on ``ctx``)."""
        cache = ctx.cache.setdefault("braking", {})
        if not cache:
            plan = ctx.targets.prepared["braking"]
            m = plan.mask[:, ctx.c0:ctx.c1].to(ctx.w.dtype)
            base_p = self.position_weight * ctx.pos_huber * ctx.w * m
            base_s = self.speed_weight * ctx.spd_huber * ctx.w * m
            if self.params.asym_on:
                m_ap = asym_multiplier(ctx.pred["positions"], ctx.rec_pos, self.params.asym_alpha)
                m_as = asym_multiplier(ctx.pred["speeds"], ctx.rec_spd, self.params.asym_alpha)
            else:
                m_ap = m_as = None
            cache.update(base_p=base_p, base_s=base_s, m_ap=m_ap, m_as=m_as)
        return cache


class AsymSpeedLoss(_BrakingTerm):
    name = "asym_speed"

    def __call__(self, ctx: LossContext) -> Optional[TermValue]:
        c = self._chunk(ctx)
        value = (((c["m_ap"] - 1.0) * c["base_p"]).sum()
                 + ((c["m_as"] - 1.0) * c["base_s"]).sum()) / ctx.targets.denom
        return TermValue(value)


class DecelWeightLoss(_BrakingTerm):
    name = "decel_weight"

    def __call__(self, ctx: LossContext) -> Optional[TermValue]:
        c = self._chunk(ctx)
        plan = ctx.targets.prepared["braking"]
        m_d = decel_multiplier(plan.a_rec[:, ctx.c0:ctx.c1], plan.mask[:, ctx.c0:ctx.c1],
                               self.params.decel_weight, self.params.decel_threshold_mps2) - 1.0
        wp = c["base_p"] if c["m_ap"] is None else c["m_ap"] * c["base_p"]
        ws = c["base_s"] if c["m_as"] is None else c["m_as"] * c["base_s"]
        return TermValue(((m_d * wp).sum() + (m_d * ws).sum()) / ctx.targets.denom)


class PropagationLoss(_BrakingTerm):
    name = "propagation"

    def __call__(self, ctx: LossContext) -> Optional[TermValue]:
        plan = ctx.targets.prepared["braking"]
        sl = slice(ctx.c0, ctx.c1)
        speeds = ctx.pred["speeds"]
        realised = realized_acceleration(speeds, ctx.current["v_mps"], ctx.targets.dt_s)
        value = float(self.params.prop_weight) * propagation_terms(
            speeds, realised, ctx.rec_spd, plan.a_rec[:, sl], plan.prop_mask[:, sl],
            ctx.w, ctx.targets.denom, speed_delta=ctx.targets.speed_delta_mps,
            accel_delta=float(self.params.prop_accel_huber_mps2))
        return TermValue(value)


def required_log_terms(flat_config: dict) -> tuple[str, ...]:
    """Braking terms a run's log must contain, from a config or TRAIN-receipt ``config`` dict."""
    return BrakingParams.from_flat(flat_config).log_terms
