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

R11 ``contrast = true`` (matched-negative contrast; R11_DESIGN.md).  The round-10b target makes the
head absorb every error of the base model on partner rows, including errors that are generic to the
STATE (the base also mis-predicts unpartnered lane-1 cars in the same state).  With the contrast the
target becomes

    target_i = a_rec_i - r_hat(phi_i),     r_hat = kernel-matched mean of (a_rec - a_model) over the
                                                    UNPARTNERED lane-1 HDV rows of the same batch

so ``head_i -> (a_rec_i - a_pre_i) - r_hat(phi_i)``: the part of the partner row's residual that a
state-matched unpartnered car does not have (the adjacency effect only).  ``phi`` are the follower's own
recorded state at t (speed, time headway, closing speed, distance to the lane end; R9_YIELD's covariates);
the matching is a Gaussian kernel in batch-standardised ``phi`` with Scott's bandwidth
``h = n_controls^(-1/(d+4))`` and one pseudo-row at the batch control mean (smooth shrinkage where
support is thin).  No bin, cell, window or threshold; the controls' recorded accelerations enter only
the target.
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
    # R11 matched-negative contrast (default False = round 10b)
    contrast: bool = False
    merge_lane: int = 2
    zone_end_m: float = 75.44
    zone_start_m: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.contrast, bool):
            raise ValueError("follower_courtesy.contrast must be a boolean")
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


#: Number of own-state matching features (see :func:`matching_features`).
MATCH_F = 4
#: Shrinkage: one pseudo-row at the batch control mean (documented rule, not tuned).
PRIOR_ROWS = 1.0


@torch.no_grad()
def matching_features(kin: Tensor, valid: Tensor, config: FollowerCourtesyConfig,
                      chunk: int = 32) -> Tensor:
    """``[B, n, K, MATCH_F]`` own-state descriptors of every slot at recorded state t (t < n).

    The covariates of R9_YIELD's matched expectation (own speed x time headway x closing speed x
    position) as continuous coordinates: v/10; asinh(time headway to the own-lane leader in s) (99 s
    without a leader); closing speed to it / 5 m/s (0 without); asinh((lane end - s) / 20 m).  Smooth
    maps of the recorded state; used ONLY to match controls in the loss target, never as a model
    input.  (A merge-zone occupancy coordinate was tried offline and dropped: it leaves ~1 effective
    control per partner row in the last 2 s before a merge, so the target falls back to the batch mean.)
    """
    b, n1, k = kin.shape[:3]
    n = n1 - 1
    out = []
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        x = kin[:, lo:hi].reshape(b * (hi - lo), k, kin.shape[-1]).float()
        pres = valid[:, lo:hi].reshape(b * (hi - lo), k).bool()
        onehot = x[..., 1:4]
        lane = torch.where(onehot.sum(-1) > 0, onehot.argmax(-1), torch.full_like(onehot[..., 0],
                                                                               -1).long())
        s, v, length = x[..., 0], x[..., 4], x[..., 6]
        same = (lane.unsqueeze(-1) == lane.unsqueeze(-2)) & pres.unsqueeze(-1) & pres.unsqueeze(-2)
        d = s.unsqueeze(-2) - s.unsqueeze(-1)                     # [r, i, l] = s_l - s_i
        ahead = same & (d > 0)
        key = torch.where(ahead, d, torch.full_like(d, float("inf")))
        dlead, lead = key.min(-1)
        has = torch.isfinite(dlead)
        lead = torch.where(has, lead, torch.zeros_like(lead))
        gap = dlead - length.gather(-1, lead)
        # time headway as in R9_YIELD's matched expectation (gap / max(v, 0.5 m/s); 99 s without leader)
        headway = torch.where(has, gap / v.clamp_min(0.5), torch.full_like(s, 99.0))
        closing = torch.where(has, v - v.gather(-1, lead), torch.zeros_like(v))
        feats = torch.stack((v / 10.0, torch.asinh(headway), closing / 5.0,
                             torch.asinh((config.zone_end_m - s) / 20.0)), -1)
        out.append(torch.nan_to_num(feats).reshape(b, hi - lo, k, MATCH_F))
    if not out:
        return kin.new_zeros(b, 0, k, MATCH_F)
    return torch.cat(out, 1)


@torch.no_grad()
def matched_control_residual(phi_rows: Tensor, phi_ctrl: Tensor, resid_ctrl: Tensor,
                             chunk: int = 512) -> tuple[Tensor, Tensor]:
    """Kernel-matched mean control residual for every row and its effective control count.

    Gaussian kernel in coordinates standardised over rows and controls together, Scott's bandwidth
    ``h = n_ctrl^(-1/(d+4))``, one pseudo-row at the control mean (:data:`PRIOR_ROWS`).
    """
    n_ctrl, dim = phi_ctrl.shape
    if n_ctrl == 0:
        zeros = phi_rows.new_zeros(phi_rows.shape[0])
        return zeros, zeros
    pooled = torch.cat((phi_rows, phi_ctrl), 0)
    mu, sd = pooled.mean(0), pooled.std(0, unbiased=False).clamp_min(1e-6)
    zr, zc = (phi_rows - mu) / sd, (phi_ctrl - mu) / sd
    h = float(n_ctrl) ** (-1.0 / (dim + 4.0))
    mean_c = resid_ctrl.mean()
    est, neff = [], []
    for part in zr.split(chunk):
        w = torch.exp(-0.5 * torch.cdist(part, zc).square() / (h * h))
        sw = w.sum(-1)
        est.append((w @ resid_ctrl + PRIOR_ROWS * mean_c) / (sw + PRIOR_ROWS))
        neff.append(sw)
    return torch.cat(est), torch.cat(neff)


def lane1_rows(targets: UnrollTargets, target_lane: int = 1) -> Tensor:
    """``[B, n, K]`` every continuous lane-1 HDV transition (partnered or not)."""
    _, lane, _, _, _, hdv = _recorded(targets)
    return hdv & (lane[:, :-1] == target_lane) & (lane[:, 1:] == target_lane)


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
        rec_all = targets.states[:, 1:targets.n_steps + 1, :, 5].detach()
        target = rec_all[rows]
        recorded = target
        if self.config.contrast:
            target = target - self._control_shift(model, targets, out, rows, rec_all)
        offset = model.follower_courtesy_offset(inputs, live=True)
        pred = model.follower_courtesy_total(pre, offset)
        value = F.huber_loss(pred, target, delta=self.config.huber_delta_mps2)
        with torch.no_grad():
            residual = recorded - pre
            self.last_stats.update(
                loss_raw=float(value.detach()),
                head_mean=float(offset.mean()), head_min=float(offset.min()),
                head_max=float(offset.max()),
                base_residual_mean=float(residual.mean()))
        return TermValue(self.config.weight * value, log=value, valid_count=n_rows)

    @torch.no_grad()
    def _control_shift(self, model: Any, targets: UnrollTargets, out: dict, rows: Tensor,
                       rec_all: Tensor) -> Tensor:
        """R11: kernel-matched control residual ``r_hat`` for every supervised partner row."""
        ctrl = lane1_rows(targets, self.config.target_lane) & ~out["courtesy_applies"].bool()
        kin, _, valid, _, _, _ = _recorded(targets)
        phi = matching_features(kin, valid, self.config)
        pre_all = out["accel_pre_courtesy_mps2"].detach()
        # Controls have no offset: their model acceleration is the plant-clamped a_pre.
        resid = rec_all - model.follower_courtesy_total(pre_all, torch.zeros_like(pre_all))
        resid_ctrl = resid[ctrl].float()
        finite = torch.isfinite(resid_ctrl)
        shift, neff = matched_control_residual(phi[rows].float(), phi[ctrl][finite].float(),
                                               resid_ctrl[finite])
        self.last_stats.update(
            contrast_n_controls=int(finite.sum()),
            contrast_shift_mean=float(shift.mean()) if shift.numel() else 0.0,
            contrast_neff_median=float(neff.median()) if neff.numel() else 0.0,
            contrast_control_resid_mean=float(resid_ctrl[finite].mean()) if bool(finite.any())
            else 0.0)
        return shift.to(rec_all.dtype)


def follower_courtesy_terms(config: FollowerCourtesyConfig) -> list[UnrollTerm]:
    return [FollowerCourtesyLoss(config)] if config.enabled else []


__all__ = ["FollowerCourtesyConfig", "FollowerCourtesyLoss", "MATCH_F", "PRIOR_ROWS", "REPLAY_KEYS",
           "SHARED_KEY", "TERM_NAME", "courtesy_rows", "follower_courtesy_terms", "lane1_rows",
           "matched_control_residual", "matching_features", "replay_outputs"]
