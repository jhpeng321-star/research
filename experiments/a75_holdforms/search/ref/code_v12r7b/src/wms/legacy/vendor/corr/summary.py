# ruff: noqa
# VENDORED from output/wd01/wm_n1/corr/summary.py (last changed in commit a8bfb8d; copied at baseline bf7d4cb).
# Copied byte-for-byte apart from this header.
# Do not edit: behaviour is frozen evidence; WP-C only mirrors it.
"""P5 CAV-motion summary: the ONLY correction-head input.

Strategy-independent summary of the CAVs' PLANNED motion, produced by the
controllers during the rollout (not labels, not a strategy name).  Fixed-size
feature vector per world, built from the rollout CAV trajectory.

The input signature is frozen by ``INTERFACES_V2.md`` and admits no strategy,
action or CAV-flag channel: ``(positions, speeds, accel, cav_mask)``.  The 16
raw features (``SUMMARY_RAW_F``) are deterministic functions of that CAV
trajectory only, so two different controllers that produce the same trajectory
yield identical summaries.  Grouped as:

0    active CAV fraction of the slot budget
1-6  per-CAV speed mean/std/min/max/p10/p90 over the horizon
7-11 per-CAV acceleration mean/|mean|/min/max/std (min = hardest braking)
12-15 per-CAV downstream displacement mean/std/min/max
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

# Raw summary feature count before projection.
SUMMARY_RAW_F = 16


@dataclass
class CorrConfig:
    summary_dim: int = 64
    d_model: int = 256
    n_tau: int = 5
    n_outcome_targets: int = 3
    n_quantiles: int = 19


def _flat_mask(cav_mask: Tensor, n_time: int, batch: int, slots: int,
               ref: Tensor) -> Tensor:
    """Coerce ``cav_mask`` to a float ``[T, B, K]`` CAV weighting.

    A ``[B, K]`` flag is expanded over time (the same CAV set at every step);
    a ``[T, B, K]`` mask is used ELEMENTWISE, so a slot can contribute to the
    summary only on the steps it is marked alive.  (The previous ``any(dim=0)``
    collapse let a slot that departed mid-window contribute all of its
    post-departure steps; the per-step mask is what the corpus window needs.)
    """
    m = cav_mask
    if m.dim() == 3:
        if tuple(m.shape) != (n_time, batch, slots):
            raise ValueError(
                f"cav_mask must be [T,B,K]={n_time,batch,slots}; "
                f"got {tuple(m.shape)}")
        return m.to(ref.dtype)
    if tuple(m.shape) != (batch, slots):
        raise ValueError(
            f"cav_mask must be [B,K]={batch,slots} or [T,B,K]; got {tuple(m.shape)}")
    return m.to(ref.dtype).unsqueeze(0).expand(n_time, batch, slots)


def _masked_percentile(values: Tensor, weights: Tensor, q: float) -> Tensor:
    """Per-world ``q``-quantile of ``values`` over the weighted slots.

    ``values``/``weights`` are ``[T, B, K]`` with ``weights`` in {0,1}.
    Empty worlds return 0.  A small per-world Python loop keeps this exact and
    deterministic across torch versions.
    """
    t, b, k = values.shape
    flat_v = values.reshape(t * b, k)
    flat_w = weights.reshape(t * b, k)
    out = values.new_zeros(b)
    for world in range(b):
        rows = slice(world, t * b, b)
        vw = flat_w[rows]
        sel = vw > 0
        if not bool(sel.any()):
            continue
        vals = flat_v[rows][sel].to(torch.float32)
        out[world] = torch.quantile(vals, q, interpolation="linear")
    return out


def _masked_extreme(values: Tensor, weights: Tensor, *, largest: bool) -> Tensor:
    """Per-world masked min (``largest=False``) / max over the CAV slots."""
    huge = torch.finfo(values.dtype).max
    filled = torch.where(weights > 0, values, torch.full_like(values, -huge
                                                              if largest else huge))
    reduced = filled.amax(dim=(0, 2)) if largest else filled.amin(dim=(0, 2))
    has = weights.sum(dim=(0, 2)) > 0
    return torch.where(has, reduced, torch.zeros_like(reduced))


def _masked_mean(values: Tensor, weights: Tensor) -> Tensor:
    num = (values * weights).sum(dim=(0, 2))
    den = weights.sum(dim=(0, 2)).clamp(min=1.0)
    return num / den


def _masked_std(values: Tensor, weights: Tensor, mean: Tensor) -> Tensor:
    centred = values - mean.unsqueeze(0).unsqueeze(-1)
    var = (centred * centred * weights).sum(dim=(0, 2))
    den = weights.sum(dim=(0, 2)).clamp(min=1.0)
    return (var / den).clamp(min=0.0).sqrt()


def cav_motion_summary(positions: Tensor, speeds: Tensor, accel: Tensor,
                       cav_mask: Tensor) -> Tensor:
    """Controller-produced CAV trajectory -> ``float [B, SUMMARY_RAW_F]``.

    ``positions/speeds/accel`` are ``[T, B, K]`` (rollout, CAV slots); ``cav_mask
    [B, K]`` selects the CAVs.  No strategy name or action label is accepted.
    """
    if positions.dim() != 3 or speeds.shape != positions.shape or \
            accel.shape != positions.shape:
        raise ValueError("positions/speeds/accel must all be [T,B,K] and equal")
    positions = positions.to(torch.float32)
    speeds = speeds.to(torch.float32)
    accel = accel.to(torch.float32)
    t, b, k = positions.shape
    w = _flat_mask(cav_mask, t, b, k, positions)

    n_cav = w[0].sum(dim=1)
    sp_mean = _masked_mean(speeds, w)
    sp_std = _masked_std(speeds, w, sp_mean)
    sp_min = _masked_extreme(speeds, w, largest=False)
    sp_max = _masked_extreme(speeds, w, largest=True)
    sp_p10 = _masked_percentile(speeds, w, 0.10)
    sp_p90 = _masked_percentile(speeds, w, 0.90)

    a_mean = _masked_mean(accel, w)
    a_abs_mean = _masked_mean(accel.abs(), w)
    a_min = _masked_extreme(accel, w, largest=False)
    a_max = _masked_extreme(accel, w, largest=True)
    a_std = _masked_std(accel, w, a_mean)

    disp = positions[-1] - positions[0]
    dw = w[0]
    d_mean = _masked_mean(disp.unsqueeze(0), dw.unsqueeze(0))
    d_std = _masked_std(disp.unsqueeze(0), dw.unsqueeze(0), d_mean)
    d_min = _masked_extreme(disp.unsqueeze(0), dw.unsqueeze(0), largest=False)
    d_max = _masked_extreme(disp.unsqueeze(0), dw.unsqueeze(0), largest=True)

    active = n_cav > 0
    feats = [
        n_cav / float(k),
        torch.where(active, sp_mean, torch.zeros_like(sp_mean)),
        torch.where(active, sp_std, torch.zeros_like(sp_std)),
        torch.where(active, sp_min, torch.zeros_like(sp_min)),
        torch.where(active, sp_max, torch.zeros_like(sp_max)),
        torch.where(active, sp_p10, torch.zeros_like(sp_p10)),
        torch.where(active, sp_p90, torch.zeros_like(sp_p90)),
        torch.where(active, a_mean, torch.zeros_like(a_mean)),
        torch.where(active, a_abs_mean, torch.zeros_like(a_abs_mean)),
        torch.where(active, a_min, torch.zeros_like(a_min)),
        torch.where(active, a_max, torch.zeros_like(a_max)),
        torch.where(active, a_std, torch.zeros_like(a_std)),
        d_mean, d_std, d_min, d_max,
    ]
    return torch.stack(feats, dim=-1)


class CavMotionSummary(nn.Module):
    """Projects the raw CAV-motion summary to ``summary_dim``."""

    def __init__(self, cfg=None) -> None:
        super().__init__()
        self.cfg = cfg or CorrConfig()
        self.proj = nn.Sequential(nn.Linear(SUMMARY_RAW_F, self.cfg.d_model),
                                  nn.GELU(),
                                  nn.Linear(self.cfg.d_model,
                                            self.cfg.summary_dim))

    def forward(self, summary_raw: Tensor) -> Tensor:
        """``[B, SUMMARY_RAW_F]`` -> ``[B, summary_dim]``."""
        if summary_raw.shape[-1] != SUMMARY_RAW_F:
            raise ValueError(
                f"summary_raw must have {SUMMARY_RAW_F} features; got "
                f"{summary_raw.shape[-1]}")
        return self.proj(summary_raw)
