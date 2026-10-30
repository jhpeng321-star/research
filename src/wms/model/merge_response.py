"""Observable spatial discharge field; new-head-only, opt-in WM2 correction.

No driver types, future commands, event labels, or behavioural thresholds are inputs.
The RBF grid is an architectural resolution, scaled by the measured road geometry.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .entries import ENTRY_S_M, SECTION_MERGE_END_M


class MergeResponseField(nn.Module):
    def __init__(self, hidden: int = 32, basis: int = 4) -> None:
        super().__init__()
        if hidden < 1 or basis < 2:
            raise ValueError("positive hidden width and at least two spatial bases required")
        self.basis = basis
        self.net = nn.Sequential(nn.Linear(8 + basis * 9, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def features(self, kin: Tensor, present: Tensor, is_cav: Tensor) -> Tensor:
        """[B,K,F]; padding is removed before arithmetic, including padded NaNs."""
        kin = torch.where(present[..., None], kin, torch.zeros_like(kin))
        s, v, a = kin[..., 0], kin[..., 4], kin[..., 5]
        lane = kin[..., 1:4]
        length = kin[..., 6]
        start, end = min(ENTRY_S_M), SECTION_MERGE_END_M
        span = end - start
        centers = torch.linspace(start, end, self.basis, device=kin.device, dtype=kin.dtype)
        weight = torch.exp(-0.5 * ((s[..., None] - centers) / (span / (self.basis - 1))) ** 2)
        weight = weight[..., None] * lane[..., None, :] * present[..., None, None]
        count = weight.sum(1)
        # log-count and speed moments do not divide by the padding-slot count.
        mean_v = (weight * v[..., None, None]).sum(1) / count.clamp_min(1)
        cav_flux = (weight * (v * is_cav.to(v.dtype))[..., None, None]).sum(1)
        pooled = torch.stack((torch.log1p(count), mean_v / 31.29, cav_flux / 31.29), -1)
        pooled = pooled.flatten(1)[:, None].expand(-1, s.shape[1], -1)
        # Front-neighbour identity is a physical selector, not a behavioural veto.
        delta = s[:, None, :] - s[:, :, None]
        same = (lane[:, :, None] * lane[:, None, :]).sum(-1) > 0
        mask = same & present[:, :, None] & present[:, None, :] & (delta > 0)
        distance = torch.where(mask, delta, torch.full_like(delta, float("inf")))
        front_distance, index = distance.min(-1)
        has_front = torch.isfinite(front_distance)
        gap = torch.where(has_front, front_distance - length.gather(1, index),
                          torch.full_like(s, span))
        dv = torch.where(has_front, v - v.gather(1, index), torch.zeros_like(v))
        ego = torch.cat((((s - end) / span)[..., None], (v / 31.29)[..., None],
                         (a / 31.29)[..., None], lane, (gap / span)[..., None],
                         (dv / 31.29)[..., None]), -1)
        return torch.cat((ego, pooled), -1)

    def forward(self, kin: Tensor, present: Tensor, is_cav: Tensor) -> Tensor:
        raw = self.net(self.features(kin, present, is_cav)).squeeze(-1)
        # Geometry scope, no speed/jam threshold; CAV command ownership is untouched.
        applies = present & ~is_cav.bool() & (kin[..., 0] <= SECTION_MERGE_END_M)
        return torch.where(applies, raw, torch.zeros_like(raw))


def freeze_except_response(model: nn.Module) -> list[str]:
    """Exact optimizer allowlist; no legacy parameter is accidentally re-enabled."""
    if getattr(model, "merge_response", None) is None:
        raise ValueError("merge_response must be explicitly enabled")
    names = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("merge_response."))
        if parameter.requires_grad:
            names.append(name)
    return names
