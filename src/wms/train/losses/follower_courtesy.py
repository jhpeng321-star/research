"""R10B learned pre-merge follower courtesy: recorded-state supervision of the courtesy head.

Rows: every lane-1 HDV transition t -> t+1 (continuous slot, HDV at both rows, lane 1 at both rows)
whose follower has a ramp partner at the step START t (``courtesy_applies``, the model's own pairing:
nearest ramp-lane car whose target follower it is; no distance window).  Target: the recorded
acceleration of row t+1 (SUMO rows are post-step, as in the cut-in loss).  Prediction:

    a_hat = plant_clamp( a_pre(t).detach() + head(inputs(t)) )

where ``a_pre`` is the full model's acceleration WITHOUT the courtesy offset at the recorded state
(IDM core with the encoded driver settings + gated learned residual + every other channel).  The
frozen, state-conditioned model is therefore the matched control at the identical state (a learned
expectation fitted on all TRAIN rows, no hand-set cells): the head can only explain what the model
without courtesy does not, and on partner rows where SUMO followers do not slow (follower alongside
or already overlapping, ramp car far from the end, fast free gaps, far partners) the same loss anchors
it at ~0.  Lane-1 cars without a partner have an identically zero offset and contribute nothing.

Only the head is trained by this term (``a_pre`` is detached; the replay runs without gradient).
Recorded future quantities enter only as the TARGET.  No event window, threshold or label is used.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import TermValue, UnrollContext, UnrollTerm, UnrollTargets
from .cutin_reaction import _recorded, one_step_replay

TERM_NAME = "follower_courtesy"
#: Replay outputs the term needs (shared with the cut-in replay when both terms are on).
REPLAY_KEYS = ("courtesy_inputs", "courtesy_applies", "accel_pre_courtesy_mps2")
#: ``targets.prepared`` key under which the cut-in replay leaves the shared outputs.
SHARED_KEY = "follower_courtesy_replay"


@dataclass(frozen=True)
class FollowerCourtesyConfig:
    weight: float = 0.0
    huber_delta_mps2: float = 1.0
    target_lane: int = 1

    def __post_init__(self) -> None:
        if not (math.isfinite(float(self.weight)) and self.weight >= 0):
            raise ValueError("follower_courtesy.weight must be finite and >= 0")
        if not (math.isfinite(float(self.huber_delta_mps2)) and self.huber_delta_mps2 > 0):
            raise ValueError("follower_courtesy.huber_delta_mps2 must be finite and positive")

    @property
    def enabled(self) -> bool:
        return self.weight > 0


def courtesy_rows(targets: UnrollTargets, applies: Tensor, target_lane: int = 1) -> Tensor:
    """``[B, n, K]`` supervised rows: lane-1 HDV transitions whose follower has a partner at t."""
    _, lane, _, _, _, hdv = _recorded(targets)
    if applies.shape != hdv.shape:
        raise ValueError("courtesy_applies must be [B, n_steps, K]")
    return (hdv & (lane[:, :-1] == target_lane) & (lane[:, 1:] == target_lane)
            & applies.bool())


def replay_outputs(model: Any, targets: UnrollTargets) -> dict[str, Tensor]:
    """Detached replay outputs (shared cut-in replay when present, else a no-grad replay)."""
    shared = targets.prepared.get(SHARED_KEY)
    if shared is not None and all(key in shared for key in REPLAY_KEYS):
        return {**shared, "_shared": True}
    with torch.no_grad():
        _, collected = one_step_replay(model, targets, collect=REPLAY_KEYS)
    missing = [key for key in REPLAY_KEYS if key not in collected]
    if missing:
        raise RuntimeError(f"follower courtesy replay is missing model outputs {missing}; "
                           "is model.follower_courtesy enabled?")
    return {**collected, "_shared": False}


class FollowerCourtesyLoss(UnrollTerm):
    name = TERM_NAME
    differentiable = True

    def __init__(self, config: FollowerCourtesyConfig) -> None:
        self.config = config
        self.last_stats: dict[str, Any] = {}

    def __call__(self, ctx: UnrollContext) -> TermValue:
        zero = torch.zeros((), device=ctx.targets.states.device,
                           requires_grad=torch.is_grad_enabled())
        if not self.config.enabled:
            return TermValue(zero, valid_count=0)
        if not torch.is_grad_enabled():
            # Validation / forward-only probes: training-only term (as the cut-in term).
            self.last_stats = {}
            return TermValue(zero, valid_count=0)
        model, targets = ctx.model, ctx.targets
        if getattr(model, "follower_courtesy", None) is None:
            raise RuntimeError("loss.follower_courtesy needs model.follower_courtesy = true")
        if targets.n_steps == 0:
            self.last_stats = {"n_rows": 0, "n_partner_rows": 0}
            return TermValue(zero, valid_count=0)
        out = replay_outputs(model, targets)
        rows = courtesy_rows(targets, out["courtesy_applies"], self.config.target_lane)
        n_rows = int(rows.sum())
        self.last_stats = {"n_rows": n_rows,
                           "n_partner_rows": int(out["courtesy_applies"].bool().sum()),
                           "shared_replay": int(bool(out["_shared"]))}
        if not n_rows:
            return TermValue(zero, valid_count=0)
        inputs = out["courtesy_inputs"][rows]
        pre = out["accel_pre_courtesy_mps2"][rows].detach()
        target = targets.states[:, 1:targets.n_steps + 1, :, 5].detach()[rows]
        offset = model.follower_courtesy_offset(inputs, live=True)
        pred = model.follower_courtesy_total(pre, offset)
        value = F.huber_loss(pred, target, delta=self.config.huber_delta_mps2)
        with torch.no_grad():
            residual = target - pre
            self.last_stats.update(
                loss_raw=float(value.detach()),
                head_mean=float(offset.mean()), head_min=float(offset.min()),
                head_max=float(offset.max()),
                base_residual_mean=float(residual.mean()))
        return TermValue(self.config.weight * value, log=value, valid_count=n_rows)


def follower_courtesy_terms(config: FollowerCourtesyConfig) -> list[UnrollTerm]:
    return [FollowerCourtesyLoss(config)] if config.enabled else []


__all__ = ["FollowerCourtesyConfig", "FollowerCourtesyLoss", "REPLAY_KEYS", "SHARED_KEY",
           "TERM_NAME", "courtesy_rows", "follower_courtesy_terms", "replay_outputs"]
