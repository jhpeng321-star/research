"""Opt-in causal driver histories, shared by TRAIN windows and evaluation.

Native samples include t0. H seconds requests floor(H / stride) + 1
samples, anchored at t0; missing run-start/entry samples are front masked.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from wms.core.constants import LONG_HISTORY_F  # noqa: F401  (single source of truth)


def history_spec(history_s=None, history_stride_s=1.0, dt_s=0.5):
    """Return (native window ticks, stride ticks, sampled window ticks)."""
    if history_s is None:
        return 10, 1, 10
    for name, value in (("history_s", history_s), ("history_stride_s", history_stride_s),
                        ("dt_s", dt_s)):
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    stride = float(history_stride_s) / float(dt_s)
    if stride < 1 or abs(stride - round(stride)) > 1e-6:
        raise ValueError("history_stride_s must be a positive multiple of dt_s")
    native = int(math.floor(float(history_s) / float(dt_s) + 1e-8)) + 1
    stride = int(round(stride))
    return native, stride, (native - 1) // stride + 1


def long_history_features(kin: Tensor, mask: Tensor, target: Tensor,
                          present_now: Tensor, *, history_s,
                          history_stride_s=1.0, dt_s=0.5):
    """Featurise only supplied native ticks <= t0 into [K,H,9] + mask.

    The first six channels retain the rolling-history convention: v/30, a/3,
    lane one-hot and gap/100. Extras are displacement from t0/250,
    leader-relative speed/30 and a leader-observed indicator. Leaders may have
    exited by t0: use each historical tick's observed neighbours, not only the
    current driver's identity-continuous slots. Continuity is ANDed across ALL
    native ticks before downsampling, preventing slot reuse between samples.
    """
    native, stride, want = history_spec(history_s, history_stride_s, dt_s)
    if kin.ndim != 3 or kin.shape[-1] != 7 or mask.shape != kin.shape[:2]:
        raise ValueError("history kin/mask must be [T,K,7]/[T,K]")
    t, k = kin.shape[:2]
    if t < 1 or target.shape != (t - 1, k) or present_now.shape != (k,):
        raise ValueError("history continuity/present shape mismatch or empty history")
    lo = max(0, t - native)
    kin, mask, target = kin[lo:], mask[lo:].bool(), target[lo:].bool()
    t = kin.shape[0]
    # Sanitise unobserved values BEFORE arithmetic (even NaN padding is inert).
    kin = torch.where(mask.unsqueeze(-1), kin, torch.zeros_like(kin))
    same = torch.ones(t, k, dtype=torch.bool, device=kin.device)
    for j in range(t - 2, -1, -1):
        same[j] = same[j + 1] & target[j]
    valid = same & mask & present_now.bool().unsqueeze(0)
    indices = torch.arange(t - 1, -1, -stride, device=kin.device).flip(0)
    sample = kin[indices]
    seen = mask[indices]
    sample_valid = valid[indices]
    features = kin.new_zeros(len(indices), k, LONG_HISTORY_F)
    for i in range(len(indices)):
        row = sample[i]
        lane = row[:, 1:4].argmax(-1)
        ds = row[:, 0].unsqueeze(0) - row[:, 0].unsqueeze(1)
        candidates = (seen[i].unsqueeze(0) & seen[i].unsqueeze(1)
                      & (lane.unsqueeze(0) == lane.unsqueeze(1)) & (ds > 0))
        distances = torch.where(candidates, ds, torch.full_like(ds, float("inf")))
        distance, leader = distances.min(-1)
        has = torch.isfinite(distance)
        gap = torch.where(has, distance - row[leader, 6],
                          torch.full_like(distance, 250.0)).clamp(0.3, 250.0)
        dv = torch.where(has, row[leader, 4] - row[:, 4], torch.zeros_like(gap))
        features[i] = torch.cat((row[:, 4:5] / 30, row[:, 5:6] / 3,
                                 row[:, 1:4], gap[:, None] / 100,
                                 (row[:, 0:1] - kin[-1, :, 0:1]) / 250,
                                 dv[:, None] / 30, has[:, None].to(kin.dtype)), -1)
    out = kin.new_zeros(k, want, LONG_HISTORY_F)
    out_mask = torch.zeros(k, want, dtype=torch.bool, device=kin.device)
    use = len(indices)
    out[:, -use:] = torch.where(sample_valid.unsqueeze(-1), features,
                               torch.zeros_like(features)).transpose(0, 1)
    out_mask[:, -use:] = sample_valid.transpose(0, 1)
    return out, out_mask


def history_step_features(kin: Tensor, gap: Tensor) -> Tensor:
    """Per-step motion features ``[..., K, HISTORY_F]`` from kinematics and gap."""
    return torch.cat((kin[..., 4:5] / 30.0, kin[..., 5:6] / 3.0, kin[..., 1:4],
                      (gap.clamp(0.0, 250.0) / 100.0).unsqueeze(-1)), dim=-1)
