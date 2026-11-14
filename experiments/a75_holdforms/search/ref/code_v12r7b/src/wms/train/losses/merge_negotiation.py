"""V10NEGOT / R5YIELD merge-negotiation auxiliary losses.

``MergeNegotiationLossConfig`` / ``MergeNegotiationTerm`` are the frozen V10NEGOT
matched non-yield negatives (kept for old receipts).  The R5YIELD block adds three
default-off terms (all weights 0 = never built, so an existing checkpoint is
bit-identical):

``yield_huber``
    Huber regression of the learned follower-yield acceleration on the **recorded**
    follower acceleration (minus the model's physical follower acceleration) for lane-1
    followers within ``yield_window_s`` of a recorded merge.  Matched non-yield controls
    (same gap geometry, no merge in the window) are kept as additional regression
    samples; no merge-logit likelihood is applied here.

``ramp_adapt_huber``
    Huber regression of the gap-conditioned ramp longitudinal residual on the recorded
    pre-merge ramp acceleration (minus the IDM core), plus a gap-evolution term that
    compares the integrated predicted gap at the recorded merge with the eventual
    recorded gap.  The physical IDM core is never replaced.

``safety_bce``
    BCE on the interaction/safety head's conflict logit, and a Huber term on its
    predicted required follower braking.  Labels are the recorded post-insertion
    follower response at recorded merge events, calibrated with recorded non-merge
    controls.  The head's conflict probability is exposed (detached) as an extra merge
    decision input; it is not converted to a hand-set decision threshold.

All masks and targets are detached recorded data; only the model predictions carry
gradient.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import LossContext, LossTerm, TermValue, UnrollTargets

TERM_NAME = "merge_negotiation"


@dataclass(frozen=True)
class MergeNegotiationLossConfig:
    """Weight and the matching geometry / yield threshold."""

    weight: float = 0.0
    yield_decel_mps2: float = -0.5
    behind_m: float = 40.0
    dend_m: float = 60.0
    merge_lane: int = 2
    target_lane: int = 1
    #: Time-chunk width of the pairwise gap matrix (bounds the [B, chunk, K, K] peak).
    pair_chunk_steps: int = 50

    def __post_init__(self) -> None:
        if not (math.isfinite(float(self.weight)) and self.weight >= 0.0):
            raise ValueError("merge negotiation weight must be finite and non-negative")
        if self.behind_m <= 0.0 or self.dend_m <= 0.0:
            raise ValueError("merge negotiation matching distances must be positive")
        if int(self.pair_chunk_steps) < 1:
            raise ValueError("merge negotiation pair chunk size must be >= 1")


@dataclass(frozen=True)
class MergeYieldLossConfig:
    """R5YIELD follower-yield, ramp-adaptation and interaction/safety terms (all off by 0)."""

    yield_huber: float = 0.0
    ramp_adapt_huber: float = 0.0
    safety_bce: float = 0.0
    yield_window_s: float = 5.0
    gap_window_s: float = 5.0
    safety_window_s: float = 3.0
    yield_huber_delta_mps2: float = 0.5
    gap_huber_delta_m: float = 2.0
    gap_weight: float = 0.5
    brake_huber_delta_mps2: float = 1.0
    safety_brake_weight: float = 0.5
    conflict_decel_mps2: float = 1.5
    yield_decel_mps2: float = -0.5
    control_weight: float = 0.5
    safety_control_weight: float = 0.25
    behind_m: float = 40.0
    dend_m: float = 60.0
    merge_lane: int = 2
    target_lane: int = 1
    pair_chunk_steps: int = 50

    def __post_init__(self) -> None:
        weights = (self.yield_huber, self.ramp_adapt_huber, self.safety_bce)
        if not all(math.isfinite(float(x)) and float(x) >= 0.0 for x in weights):
            raise ValueError("merge_yield weights must be finite and non-negative")
        windows = (self.yield_window_s, self.gap_window_s, self.safety_window_s)
        if not all(math.isfinite(float(x)) and 0.0 < float(x) <= 15.0 for x in windows):
            raise ValueError("merge_yield windows must be finite and in (0, 15] seconds")
        positive = (self.yield_huber_delta_mps2, self.gap_huber_delta_m,
                    self.brake_huber_delta_mps2)
        if not all(math.isfinite(float(x)) and float(x) > 0.0 for x in positive):
            raise ValueError("merge_yield Huber deltas must be finite and positive")
        if not (math.isfinite(float(self.gap_weight)) and float(self.gap_weight) >= 0.0
                and math.isfinite(float(self.safety_brake_weight))
                and float(self.safety_brake_weight) >= 0.0):
            raise ValueError("merge_yield gap / brake weights must be non-negative")
        if not (math.isfinite(float(self.control_weight)) and float(self.control_weight) >= 0.0
                and math.isfinite(float(self.safety_control_weight))
                and float(self.safety_control_weight) >= 0.0):
            raise ValueError("merge_yield control weights must be non-negative")
        if not (math.isfinite(float(self.conflict_decel_mps2))
                and float(self.conflict_decel_mps2) > 0.0):
            raise ValueError("merge_yield conflict_decel_mps2 must be positive")
        if not (math.isfinite(float(self.yield_decel_mps2))
                and float(self.yield_decel_mps2) < 0.0):
            raise ValueError("merge_yield yield_decel_mps2 must be negative")
        if self.behind_m <= 0.0 or self.dend_m <= 0.0:
            raise ValueError("merge_yield matching distances must be positive")
        if int(self.pair_chunk_steps) < 1:
            raise ValueError("merge_yield pair chunk size must be >= 1")

    @property
    def enabled(self) -> bool:
        return (float(self.yield_huber) > 0.0 or float(self.ramp_adapt_huber) > 0.0
                or float(self.safety_bce) > 0.0)

    @staticmethod
    def window_steps(seconds: float, dt_s: float) -> int:
        return max(1, int(round(float(seconds) / float(dt_s))))


def _lane(kin: Tensor) -> Tensor:
    onehot = kin[..., 1:4]
    lane = onehot.argmax(-1)
    return torch.where(onehot.sum(-1) > 0, lane, torch.full_like(lane, -1))


def _nearest_behind_gap(s: Tensor, target: Tensor, behind_m: float,
                        chunk_steps: int) -> tuple[Tensor, Tensor]:
    """Nearest target-lane follower (``gap_min``, column index) per row.

    ``s`` / ``target`` are ``[B, T, K]``; the output is ``[B, T, K]``.  The pairwise
    ``s_i - s_j`` tensor is materialised ``chunk_steps`` time steps at a time, so the peak
    is ``[B, min(T, chunk), K, K]`` instead of ``[B, T, K, K]``.  The reduction is over the
    target columns, which chunking does not reorder, so results are bit-identical.
    """
    b, t, k = s.shape
    gaps: list[Tensor] = []
    idxs: list[Tensor] = []
    for t0 in range(0, t, chunk_steps):
        sc, tc = s[:, t0:t0 + chunk_steps], target[:, t0:t0 + chunk_steps]
        ds = sc.unsqueeze(-1) - sc.unsqueeze(-2)                   # [B, chunk, K, K]
        behind = tc.unsqueeze(-2) & (ds >= 0.0) & (ds <= float(behind_m))
        gap = torch.where(behind, ds, torch.full_like(ds, float("inf")))
        gap_min, idx = gap.min(-1)
        gaps.append(gap_min)
        idxs.append(idx)
    if not gaps:
        empty = s.new_empty((b, 0, k))
        return empty, torch.empty((b, 0, k), dtype=torch.long, device=s.device)
    return torch.cat(gaps, dim=1), torch.cat(idxs, dim=1)


def _premerge_mask(merge_event: Tensor, window: int) -> Tensor:
    """``[B, T, K]`` steps in the ``window`` steps before a recorded merge event."""
    b, n, k = merge_event.shape
    out = torch.zeros_like(merge_event)
    for lag in range(1, min(int(window), max(n - 1, 0)) + 1):
        out[:, :n - lag] |= merge_event[:, lag:]
    return out


def _gather_time(x: Tensor, t_idx: Tensor) -> Tensor:
    """``x[b, t_idx[b, k], k]`` -> ``[B, K]``."""
    b, n, k = x.shape
    b_idx = torch.arange(b, device=x.device).unsqueeze(1)
    k_idx = torch.arange(k, device=x.device).unsqueeze(0)
    return x[b_idx, t_idx.clamp(0, max(n - 1, 0)), k_idx]


def _future_min(resp: Tensor, idx: Tensor, window: int) -> Tensor:
    """Minimum recorded response over the next ``window`` steps (``[B, T, K]``)."""
    b, n, k = resp.shape
    values: list[Tensor] = []
    for lag in range(int(window)):
        t_idx = torch.arange(n, device=resp.device) + lag
        valid = t_idx < n
        t_idx = t_idx.clamp_max(max(n - 1, 0))
        current = resp.gather(1, t_idx.view(1, n, 1).expand(b, n, k))
        current = current.gather(-1, idx.clamp(0, max(k - 1, 0)))
        values.append(torch.where(valid.view(1, n, 1), current,
                                  torch.full_like(current, float("inf"))))
    return torch.stack(values, -1).min(-1).values


def _plan(config: MergeYieldLossConfig, targets: UnrollTargets) -> dict:
    """Recorded masks, gap/response targets and counts (cached per unroll)."""
    key = ("merge_yield", float(config.yield_window_s), float(config.gap_window_s),
           float(config.safety_window_s), float(config.behind_m), float(config.dend_m),
           float(config.conflict_decel_mps2), float(config.yield_decel_mps2),
           int(config.merge_lane), int(config.target_lane), int(config.pair_chunk_steps))
    cached = targets.prepared.get("merge_yield")
    if cached is not None and cached.get("key") == key:
        return cached
    n = int(targets.n_steps)
    if n <= 0:
        result = {"key": key, "empty": True, "n_premerge": 0.0, "n_pos": 0.0,
                  "n_neg": 0.0, "n_safety_pos": 0.0, "n_safety_neg": 0.0,
                  "n_conflict": 0.0, "n_gap": 0.0}
        targets.prepared["merge_yield"] = result
        return result
    rec = targets.states[:, :n + 1].float()
    k = rec.shape[2]
    lanes, lane_valid = _lane(rec), rec[..., 1:4].sum(-1) > 0
    valid = (targets.loss_valid[:, :n + 1, :k].bool()
             & ~targets.is_cav[:, :n + 1, :k].bool() & lane_valid)
    s, v, a = rec[..., 0], rec[..., 4], rec[..., 5]
    ramp = valid[:, :n] & (lanes[:, :n] == int(config.merge_lane))
    end = 75.44
    near = ramp & ((end - s[:, :n]) < float(config.dend_m))
    merge_event = (ramp & valid[:, 1:n + 1]
                   & (lanes[:, 1:n + 1] == int(config.target_lane)))
    target = valid[:, :n] & (lanes[:, :n] == int(config.target_lane))
    gap_min, idx = _nearest_behind_gap(
        s[:, :n], target, float(config.behind_m), int(config.pair_chunk_steps))
    has = torch.isfinite(gap_min) & near
    # Missing followers must not leak non-finite gap/response values into masked terms.
    gap_min = torch.where(torch.isfinite(gap_min), gap_min, torch.zeros_like(gap_min))
    safe_idx = idx.clamp(0, max(k - 1, 0))
    f_acc_next = a[:, 1:n + 1].gather(-1, safe_idx)
    f_v = v[:, :n].gather(-1, safe_idx)
    f_pos = s[:, :n].gather(-1, safe_idx)
    pre_y = _premerge_mask(merge_event, config.window_steps(
        config.yield_window_s, float(targets.dt_s)))
    pre_g = _premerge_mask(merge_event, config.window_steps(
        config.gap_window_s, float(targets.dt_s)))
    yield_mask = has & pre_y
    control_mask = has & ~pre_y
    gap_mask = has & pre_g
    has_merge = merge_event.any(dim=1)
    merge_time = merge_event.float().argmax(dim=1)
    safe_mt = merge_time.clamp(0, max(n - 1, 0))
    s_r_m = _gather_time(s[:, :n], safe_mt)
    s_f_m = _gather_time(f_pos, safe_mt)
    gap_star = (s_f_m - s_r_m).clamp_min(0.0)
    time = torch.arange(n, device=s.device).view(1, n, 1)
    horizon = merge_time.unsqueeze(1) - time
    horizon_s = horizon.clamp_min(1.0).float() * float(targets.dt_s)
    gap_steps = config.window_steps(config.gap_window_s, float(targets.dt_s))
    gap_mask = gap_mask & has_merge.unsqueeze(1) & (horizon >= 1) & (horizon <= gap_steps)
    safety_steps = config.window_steps(config.safety_window_s, float(targets.dt_s))
    resp_min = _future_min(a[:, 1:n + 1], safe_idx, safety_steps)
    resp_min = torch.where(torch.isfinite(resp_min), resp_min, torch.zeros_like(resp_min))
    required_brake = (-resp_min).clamp_min(0.0)
    conflict = resp_min < -float(config.conflict_decel_mps2)
    safety_pos = merge_event & has
    safety_neg = control_mask
    safety_mask = safety_pos | safety_neg
    result = {
        "key": key, "empty": False,
        "yield_mask": yield_mask.detach(), "control_mask": control_mask.detach(),
        "ramp_mask": near.detach(), "gap_mask": gap_mask.detach(),
        "safety_mask": safety_mask.detach(), "safety_pos": safety_pos.detach(),
        "safety_neg": safety_neg.detach(),
        "f_idx": safe_idx.detach(), "f_acc_next": f_acc_next.detach(),
        "ramp_acc_next": a[:, 1:n + 1].detach(),
        "gap_min": gap_min.detach(), "f_v": f_v.detach(),
        "ramp_v": v[:, :n].detach(), "gap_star": gap_star.detach(),
        "horizon_s": horizon_s.detach(),
        "required_brake": required_brake.detach(), "conflict": conflict.detach(),
        "n_premerge": float(yield_mask.sum()), "n_pos": float(
            (yield_mask & (f_acc_next < float(config.yield_decel_mps2))).sum()),
        "n_neg": float(control_mask.sum()), "n_gap": float(gap_mask.sum()),
        "n_safety_pos": float(safety_pos.sum()), "n_safety_neg": float(safety_neg.sum()),
        "n_conflict": float((safety_pos & conflict).sum()),
    }
    targets.prepared["merge_yield"] = result
    return result


def _physics(ctx: LossContext) -> Optional[Tensor]:
    return ctx.pred.get("physics_accel_mps2")


class MergeNegotiationTerm(LossTerm):
    """BCE on the follower-yield label + a merge-delta hinge on matched negatives."""

    name = TERM_NAME

    def __init__(self, cfg: MergeNegotiationLossConfig) -> None:
        self.cfg = cfg

    def prepare(self, targets: UnrollTargets) -> None:
        cfg, n = self.cfg, targets.n_steps
        with torch.no_grad():
            rec = targets.states[:, :n + 1].float()
            k = rec.shape[2]
            lanes = _lane(rec)
            valid = targets.loss_valid[:, :n + 1, :k].bool() & ~targets.is_cav[:, :n + 1, :k]
            s, a = rec[..., 0], rec[..., 5]
            ramp = valid[:, :n] & (lanes[:, :n] == cfg.merge_lane)
            end = 75.44
            near = ramp & ((end - s[:, :n]) < float(cfg.dend_m))
            target = valid[:, :n] & (lanes[:, :n] == cfg.target_lane)
            gap_min, idx = _nearest_behind_gap(
                s[:, :n], target, cfg.behind_m, int(cfg.pair_chunk_steps))
            has = torch.isfinite(gap_min) & near
            safe_idx = idx.clamp(0, max(k - 1, 0))
            f_acc = a[:, 1:n + 1].gather(-1, safe_idx)                 # next-step accel
            yield_label = (f_acc < float(cfg.yield_decel_mps2)) & has
            merged = (lanes[:, 1:n + 1] == cfg.target_lane) & ramp
            targets.prepared["merge_negotiation"] = {
                "valid": has, "yield": yield_label, "merged": merged,
                "n_valid": float(has.sum()), "n_pos": float(yield_label.sum()),
                "n_merge_pos": float((yield_label & merged).sum()),
                "n_neg": float((has & ~yield_label).sum())}

    def __call__(self, ctx: LossContext) -> Optional[TermValue]:
        cfg, prep = self.cfg, ctx.targets.prepared.get("merge_negotiation")
        pred = ctx.pred
        logit = pred.get("merge_negotiation_yield_logit")
        delta = pred.get("merge_negotiation_merge_delta")
        if prep is None or logit is None or delta is None:
            return None
        c0, c1 = ctx.c0, ctx.c1
        k = prep["valid"].shape[-1]
        logit = logit[:, :, :k]
        delta = delta[:, :, :k]
        valid = prep["valid"][:, c0:c1]
        y = prep["yield"][:, c0:c1].to(logit.dtype)
        merged = prep["merged"][:, c0:c1]
        if not bool(valid.any()):
            return None
        bce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
        bce = torch.where(valid, bce, torch.zeros_like(bce)).sum() / valid.sum().clamp_min(1)
        neg = valid & ~prep["yield"][:, c0:c1] & ~merged
        pos = valid & prep["yield"][:, c0:c1] & merged
        hinge = (torch.where(neg, F.softplus(delta), torch.zeros_like(delta)).sum()
                 + torch.where(pos, F.softplus(-delta), torch.zeros_like(delta)).sum())
        hinge = hinge / valid.sum().clamp_min(1)
        value = cfg.weight * (bce + 0.5 * hinge)
        return TermValue(value, log=bce.detach())


class YieldHuberTerm(LossTerm):
    """Huber follower-yield correction against recorded pre-merge follower acceleration."""

    name = "yield_huber"

    def __init__(self, cfg: MergeYieldLossConfig,
                 pred_key: str = "merge_negotiation_yield_accel",
                 name: Optional[str] = None) -> None:
        self.cfg = cfg
        self.pred_key = str(pred_key)
        if name is not None:
            self.name = str(name)

    def prepare(self, targets: UnrollTargets) -> None:
        _plan(self.cfg, targets)

    def __call__(self, ctx: LossContext) -> Optional[TermValue]:
        if float(self.cfg.yield_huber) <= 0.0:
            return None
        prep = _plan(self.cfg, ctx.targets)
        physics = _physics(ctx)
        pred = ctx.pred.get(self.pred_key)
        if prep.get("empty") or pred is None or physics is None:
            return None
        c0, c1, k = ctx.c0, ctx.c1, prep["f_idx"].shape[-1]
        # ``ctx.pred`` holds only this truncated-BPTT chunk while ``prep`` targets are on
        # the absolute unroll clock: index the targets, never the chunk-local prediction.
        idx = prep["f_idx"][:, c0:c1]
        physics = physics[:, :, :k]
        target = prep["f_acc_next"][:, c0:c1] - physics.gather(-1, idx)
        error = F.huber_loss(pred[:, :, :k], target,
                             delta=float(self.cfg.yield_huber_delta_mps2),
                             reduction="none")
        weight = (prep["yield_mask"][:, c0:c1].to(error.dtype)
                  + float(self.cfg.control_weight)
                  * prep["control_mask"][:, c0:c1].to(error.dtype))
        denom = weight.sum().clamp_min(1.0)
        if float(weight.sum()) <= 0.0:
            zero = torch.zeros((), device=error.device, dtype=error.dtype)
            return TermValue(zero, log=zero)
        value = (error * weight).sum() / denom
        return TermValue(float(self.cfg.yield_huber) * value, log=value.detach())


class RampAdaptHuberTerm(LossTerm):
    """Huber ramp longitudinal residual + integrated gap evolution to the recorded gap."""

    name = "ramp_adapt_huber"

    def __init__(self, cfg: MergeYieldLossConfig) -> None:
        self.cfg = cfg

    def prepare(self, targets: UnrollTargets) -> None:
        _plan(self.cfg, targets)

    def __call__(self, ctx: LossContext) -> Optional[TermValue]:
        if float(self.cfg.ramp_adapt_huber) <= 0.0:
            return None
        prep = _plan(self.cfg, ctx.targets)
        physics = _physics(ctx)
        pred = ctx.pred.get("merge_negotiation_ramp_delta")
        if prep.get("empty") or pred is None or physics is None:
            return None
        c0, c1, k = ctx.c0, ctx.c1, prep["ramp_mask"].shape[-1]
        # Chunk-local predictions, absolute-clock targets (same convention as the recorded
        # chunk terms): ``pred`` / ``physics`` already cover exactly ``[c0, c1)``.
        pred = pred[:, :, :k]
        physics = physics[:, :, :k]
        delta = pred
        ramp_mask = prep["ramp_mask"][:, c0:c1].to(delta.dtype)
        target = prep["ramp_acc_next"][:, c0:c1] - physics
        accel_err = F.huber_loss(delta, target,
                                 delta=float(self.cfg.yield_huber_delta_mps2),
                                 reduction="none")
        accel_value = (accel_err * ramp_mask).sum() / ramp_mask.sum().clamp_min(1.0)
        idx = prep["f_idx"][:, c0:c1]
        physics_f = physics.gather(-1, idx)
        horizon = prep["horizon_s"][:, c0:c1]
        gap_pred = (prep["gap_min"][:, c0:c1]
                    + (prep["f_v"][:, c0:c1] - prep["ramp_v"][:, c0:c1]) * horizon
                    + 0.5 * (physics_f - (physics + delta)) * horizon.square())
        gap_target = prep["gap_star"].unsqueeze(1).expand_as(gap_pred)
        gap_err = F.huber_loss(gap_pred, gap_target,
                               delta=float(self.cfg.gap_huber_delta_m), reduction="none")
        gap_mask = prep["gap_mask"][:, c0:c1].to(gap_err.dtype)
        gap_value = (gap_err * gap_mask).sum() / gap_mask.sum().clamp_min(1.0)
        value = accel_value + float(self.cfg.gap_weight) * gap_value
        return TermValue(float(self.cfg.ramp_adapt_huber) * value, log=value.detach())


class SafetyBCETerm(LossTerm):
    """Conflict BCE + required-braking Huber on recorded post-insertion responses."""

    name = "safety_bce"

    def __init__(self, cfg: MergeYieldLossConfig) -> None:
        self.cfg = cfg

    def prepare(self, targets: UnrollTargets) -> None:
        _plan(self.cfg, targets)

    def __call__(self, ctx: LossContext) -> Optional[TermValue]:
        if float(self.cfg.safety_bce) <= 0.0:
            return None
        prep = _plan(self.cfg, ctx.targets)
        logit = ctx.pred.get("merge_negotiation_conflict_logit")
        brake = ctx.pred.get("merge_negotiation_required_brake")
        if prep.get("empty") or logit is None or brake is None:
            return None
        c0, c1, k = ctx.c0, ctx.c1, prep["safety_pos"].shape[-1]
        logit = logit[:, :, :k]
        brake = brake[:, :, :k]
        positive = prep["safety_pos"][:, c0:c1].to(logit.dtype)
        negative = prep["safety_neg"][:, c0:c1].to(logit.dtype)
        weight = positive + float(self.cfg.safety_control_weight) * negative
        if float(weight.sum()) <= 0.0:
            zero = torch.zeros((), device=logit.device, dtype=logit.dtype)
            return TermValue(zero, log=zero)
        label = prep["conflict"][:, c0:c1].to(logit.dtype)
        bce = F.binary_cross_entropy_with_logits(logit, label, reduction="none")
        bce_value = (bce * weight).sum() / weight.sum().clamp_min(1.0)
        brake_err = F.smooth_l1_loss(brake, prep["required_brake"][:, c0:c1],
                                     beta=float(self.cfg.brake_huber_delta_mps2),
                                     reduction="none")
        brake_value = (brake_err * weight).sum() / weight.sum().clamp_min(1.0)
        value = bce_value + float(self.cfg.safety_brake_weight) * brake_value
        return TermValue(float(self.cfg.safety_bce) * value, log=bce_value.detach())


def merge_negotiation_terms(cfg: MergeNegotiationLossConfig) -> list[MergeNegotiationTerm]:
    """The legacy auxiliary term (empty when its weight is 0)."""
    return [MergeNegotiationTerm(cfg)] if cfg.weight > 0.0 else []


def merge_yield_terms(cfg: MergeYieldLossConfig) -> list[LossTerm]:
    """The enabled R5YIELD terms, in a fixed order (empty when every weight is 0)."""
    terms: list[LossTerm] = []
    if float(cfg.yield_huber) > 0.0:
        terms.append(YieldHuberTerm(cfg))
    if float(cfg.ramp_adapt_huber) > 0.0:
        terms.append(RampAdaptHuberTerm(cfg))
    if float(cfg.safety_bce) > 0.0:
        terms.append(SafetyBCETerm(cfg))
    return terms


__all__ = ["TERM_NAME", "MergeNegotiationLossConfig", "MergeYieldLossConfig",
           "MergeNegotiationTerm", "YieldHuberTerm", "RampAdaptHuberTerm",
           "SafetyBCETerm", "merge_negotiation_terms", "merge_yield_terms"]
