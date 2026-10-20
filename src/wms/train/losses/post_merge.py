"""R5MERGE (2026-10-20): post-merge acceleration-consistency terms.

The blow-up diagnosis (`BLOWUP_DIAG.md`, section R3) found just-merged ramp HDVs in
lane 1 held at 1-4 m/s on a free lane by a learned residual of -3.6..-16 m/s^2 while
the IDM core is +1.7..+2.2 m/s^2, 2-10 s after the 2->1 merge; the spike precedes the
lane-1 phantom queues.  `WAVE.md` adds that the residual attention head is saturated
at its -1.5 m/s^2 tanh tail, so the event cannot be learned from the head output.

These terms are the cheap learned fix, both default OFF (weight 0 = never built, so an
existing checkpoint is bit-identical):

``post_merge`` (``weight``)
    Huber error between the predicted acceleration and the recorded acceleration of
    every recorded HDV that changed lane 2->1 within the last ``window_s`` seconds
    (default 10 s).  The window and the lane transition are read from the recording.

``post_merge_residual`` (``residual_weight``)
    One-sided penalty ``relu(-raw_residual)`` on the same steps when the predicted IDM
    term is positive and the recorded acceleration is non-negative: the learned
    residual may not brake a car that SUMO accelerates.

Every mask and target is detached recorded data; only the predicted acceleration and
the raw residual carry gradient.  No parameters are added and both weights default to
0.  ``window_s`` is restricted to 2-20 s (the diagnosed events live 2-10 s after the
merge).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import LossContext, LossTerm, TermValue, UnrollTargets


@dataclass(frozen=True)
class PostMergeAccelConfig:
    """Hyper-parameters of the post-merge consistency terms (all default OFF)."""

    weight: float = 0.0
    residual_weight: float = 0.0
    window_s: float = 10.0
    huber_delta_mps2: float = 0.5
    merge_lane: int = 2
    target_lane: int = 1

    def __post_init__(self) -> None:
        if not all(math.isfinite(float(x)) for x in (
                self.weight, self.residual_weight, self.window_s,
                self.huber_delta_mps2)):
            raise ValueError("post_merge hyper-parameters must be finite")
        if float(self.weight) < 0.0 or float(self.residual_weight) < 0.0:
            raise ValueError("post_merge weights must be >= 0 (0 = off)")
        if not 2.0 <= float(self.window_s) <= 20.0:
            raise ValueError("post_merge.window_s must be in [2, 20] seconds")
        if float(self.huber_delta_mps2) <= 0.0:
            raise ValueError("post_merge.huber_delta_mps2 must be positive")
        if int(self.merge_lane) == int(self.target_lane) \
                or int(self.merge_lane) < 0 or int(self.target_lane) < 0:
            raise ValueError("post_merge lanes must be distinct and non-negative")

    @property
    def enabled(self) -> bool:
        return float(self.weight) > 0.0 or float(self.residual_weight) > 0.0

    def window_steps_for(self, dt_s: float) -> int:
        return max(1, int(round(float(self.window_s) / float(dt_s))))

    @property
    def log_terms(self) -> tuple[str, ...]:
        return tuple(name for name, on in (("post_merge", self.weight > 0.0),
                                           ("post_merge_residual",
                                            self.residual_weight > 0.0)) if on)


def _recorded_lanes(targets: UnrollTargets) -> tuple[Tensor, Tensor]:
    """Recorded lane index and lane validity ``[B, n_steps + 1, K]``."""
    onehot = targets.states[..., 1:4]
    lane = onehot.argmax(-1)
    return lane, onehot.sum(-1) > 0


def _lane_changes(targets: UnrollTargets, config: PostMergeAccelConfig) -> Tensor:
    """``[B, n, K]`` transitions from ``merge_lane`` to ``target_lane`` at step t.

    Step ``t`` is the post-step recorded lane of the t-th chunk step (``states[t+1]``);
    step 0 compares against the t0 lane, so a change already visible at the first
    recorded step is counted too.  A junction (undefined lane) never creates an event.
    """
    lanes, valid = _recorded_lanes(targets)
    n = targets.n_steps
    post_lane, post_valid = lanes[:, 1:n + 1], valid[:, 1:n + 1]
    initial_lane, initial_valid = lanes[:, 0], valid[:, 0]
    prev_lane = torch.cat((initial_lane.unsqueeze(1), post_lane[:, :-1]), dim=1)
    prev_valid = torch.cat((initial_valid.unsqueeze(1), post_valid[:, :-1]), dim=1)
    return (post_valid & prev_valid
            & (prev_lane == int(config.merge_lane))
            & (post_lane == int(config.target_lane)))


@torch.no_grad()
def post_merge_mask(targets: UnrollTargets, config: PostMergeAccelConfig) -> Tensor:
    """``[B, n_steps, K]`` recorded HDVs that changed 2->1 within the last window."""
    n = targets.n_steps
    window = config.window_steps_for(float(targets.dt_s))
    changes = _lane_changes(targets, config)
    recent = torch.zeros_like(changes)
    for lag in range(window + 1):
        if lag == 0:
            recent |= changes
        elif lag < n:
            recent[:, lag:] |= changes[:, :-lag]
    supervised = (targets.hdv_mask & ~targets.is_cav[:, 1:n + 1]
                  & targets.loss_valid[:, 1:n + 1])
    return recent & supervised


def _plan(config: PostMergeAccelConfig, targets: UnrollTargets) -> dict:
    """The recent-merge mask and its size (cached on the unroll targets)."""
    key = ("post_merge", float(config.window_s), int(config.merge_lane),
           int(config.target_lane))
    cached = targets.prepared.get("post_merge")
    if cached is not None and cached.get("key") == key:
        return cached
    mask = post_merge_mask(targets, config)
    result = {"key": key, "mask": mask, "count": int(mask.sum()),
              "window_steps": config.window_steps_for(float(targets.dt_s)),
              "dt_s": float(targets.dt_s)}
    targets.prepared["post_merge"] = result
    return result


def _pred_accel(ctx: LossContext) -> Tensor:
    """Predicted acceleration of every chunk step ``[B, len, K]``."""
    accel = ctx.pred.get("accel")
    if accel is not None:
        return accel
    sequence = torch.cat((ctx.current["v_mps"].unsqueeze(1), ctx.pred["speeds"]), dim=1)
    return (sequence[:, 1:] - sequence[:, :-1]) / float(ctx.targets.dt_s)


def _recorded_accel(ctx: LossContext) -> Tensor:
    return ctx.targets.states[:, ctx.c0 + 1:ctx.c1 + 1, :, 5].detach()


def _phases(ctx: LossContext) -> tuple[Tensor, Tensor]:
    """Predicted IDM term and recorded acceleration of every chunk step."""
    physics = ctx.pred.get("physics_accel_mps2")
    if physics is None:
        physics = torch.zeros_like(_recorded_accel(ctx))
    return physics, _recorded_accel(ctx)


class PostMergeAccelLoss(LossTerm):
    """Huber acceleration consistency of recently merged ramp HDVs."""

    name = "post_merge"

    def __init__(self, config: PostMergeAccelConfig) -> None:
        self.config = config

    def prepare(self, targets: UnrollTargets) -> None:
        _plan(self.config, targets)

    def __call__(self, ctx: LossContext) -> Optional[TermValue]:
        if float(self.config.weight) <= 0.0:
            return None
        plan = _plan(self.config, ctx.targets)
        mask = plan["mask"][:, ctx.c0:ctx.c1].to(ctx.pred["speeds"].dtype)
        if float(mask.sum()) <= 0.0:
            zero = torch.zeros((), device=mask.device, dtype=mask.dtype)
            return TermValue(zero, log=zero)
        error = F.smooth_l1_loss(_pred_accel(ctx), _recorded_accel(ctx),
                                 beta=float(self.config.huber_delta_mps2),
                                 reduction="none")
        value = (error * mask).sum() / mask.sum().clamp_min(1.0)
        return TermValue(float(self.config.weight) * value, log=value)


class PostMergeResidualLoss(LossTerm):
    """One-sided penalty on residual braking where SUMO accelerates."""

    name = "post_merge_residual"

    def __init__(self, config: PostMergeAccelConfig) -> None:
        self.config = config

    def prepare(self, targets: UnrollTargets) -> None:
        _plan(self.config, targets)

    def __call__(self, ctx: LossContext) -> Optional[TermValue]:
        if float(self.config.residual_weight) <= 0.0:
            return None
        residual = ctx.raw_residual
        if residual is None:
            return None
        plan = _plan(self.config, ctx.targets)
        physics, a_rec = _phases(ctx)
        active = (plan["mask"][:, ctx.c0:ctx.c1]
                  & (physics > 0.0) & (a_rec >= 0.0))
        active = active.to(residual.dtype)
        penalty = torch.relu(-residual)
        if float(active.sum()) <= 0.0:
            zero = torch.zeros((), device=penalty.device, dtype=penalty.dtype)
            return TermValue(zero, log=zero)
        value = (penalty * active).sum() / active.sum().clamp_min(1.0)
        return TermValue(float(self.config.residual_weight) * value, log=value)


def post_merge_terms(config: PostMergeAccelConfig) -> list[LossTerm]:
    """The enabled post-merge terms, in a fixed order (empty when both weights are 0)."""
    terms: list[LossTerm] = []
    if float(config.weight) > 0.0:
        terms.append(PostMergeAccelLoss(config))
    if float(config.residual_weight) > 0.0:
        terms.append(PostMergeResidualLoss(config))
    return terms


def prepare_post_merge(config: PostMergeAccelConfig, targets: UnrollTargets) -> dict:
    """Public prepare hook (also used by the tests)."""
    return _plan(config, targets)


__all__ = ["PostMergeAccelConfig", "PostMergeAccelLoss", "PostMergeResidualLoss",
           "post_merge_mask", "post_merge_terms", "prepare_post_merge"]
