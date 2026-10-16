"""Rolling motion-history conventions of the WM-S rollout."""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

# Rolling history (last 5 s at 0.5 s), per-step feature width and long-history width are
# defined once in ``wms.core.constants`` and re-exported here.
from wms.core.constants import HISTORY_F, HISTORY_H, LONG_HISTORY_F  # noqa: F401


def history_spec(history_s: Optional[float] = None, history_stride_s: float = 1.0,
                 dt_s: float = 0.5) -> tuple[int, int, int]:
    """Return ``(native window ticks, stride ticks, sampled window ticks)``.

    Native samples include t0.  ``history_s`` seconds request
    ``floor(history_s / dt) + 1`` samples anchored at t0.
    """
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


def history_step_features(kin: Tensor, gap: Tensor) -> Tensor:
    """Per-step motion features ``[..., K, HISTORY_F]`` from kinematics and gap."""
    return torch.cat((kin[..., 4:5] / 30.0, kin[..., 5:6] / 3.0, kin[..., 1:4],
                      (gap.clamp(0.0, 250.0) / 100.0).unsqueeze(-1)), dim=-1)


def initial_history(feat: Tensor, present: Tensor, length: int) -> tuple[Tensor, Tensor]:
    """Empty rolling buffer whose newest slot holds the t0 features."""
    history = torch.zeros(*feat.shape[:2], length, feat.shape[-1],
                          dtype=feat.dtype, device=feat.device)
    history[:, :, -1] = feat
    mask = torch.zeros(*feat.shape[:2], length, dtype=torch.bool, device=feat.device)
    mask[:, :, -1] = present
    return history, mask


def push_history(history: Tensor, mask: Tensor, feat: Tensor,
                 present: Tensor) -> tuple[Tensor, Tensor]:
    """Drop the oldest slot and append the newest features."""
    return (torch.cat((history[:, :, 1:], feat.unsqueeze(2)), dim=2),
            torch.cat((mask[:, :, 1:], present.unsqueeze(-1)), dim=2))


def initial_lane_age(present: Tensor, dtype: torch.dtype,
                     driver_history: Optional[Tensor],
                     driver_history_mask: Optional[Tensor], *, long_history: bool,
                     stride_steps: Optional[int]) -> Tensor:
    """Steps since the last lane change of every slot, from the pre-t0 driver history.

    Slots without an observed change (or without history) get the sentinel 999.
    ``stride_steps`` converts history samples to physics steps for long histories.
    """
    lane_age = torch.full_like(present, 999, dtype=dtype)
    if driver_history is None or driver_history_mask is None \
            or (long_history and driver_history.shape[-2] <= 1):
        return lane_age
    hist_lane = driver_history[..., 2:5].argmax(-1)
    valid_pair = (driver_history_mask[..., 1:].bool() & driver_history_mask[..., :-1].bool())
    changed = (hist_lane[..., 1:] != hist_lane[..., :-1]) & valid_pair
    index = torch.arange(changed.shape[-1], device=present.device)
    latest = torch.where(changed, index, -1).amax(-1)
    since = changed.shape[-1] - latest - 1
    if long_history:
        since = since * stride_steps
    return torch.where(latest >= 0, since.to(dtype), lane_age)
