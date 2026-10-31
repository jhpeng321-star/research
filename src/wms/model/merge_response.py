"""Observable spatial discharge field; new-head-only, opt-in WM2 correction.

No driver types, future commands, event labels, or behavioural thresholds are inputs.
The RBF grid is an architectural resolution, scaled by the measured road geometry.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .entries import ENTRY_S_M, SECTION_MERGE_END_M, SECTION_MERGE_START_M


def refined_spatial_layout(basis: int, merge_bases: int, ramp_bases: int, *,
                           device=None, dtype=None):
    """Centres and per-centre widths of the geometry-concentrated spatial basis.

    S-WMFIX-C fallback 2 (WMFIX_REVIEW claim 1: 4 uniform bases with sigma
    ~907 m cannot resolve the 75 m merge).  The coarse backbone (``basis``
    uniform centres over the observed full upstream span) is kept unchanged;
    ``merge_bases`` extra centres concentrate over the observed merge section
    [0, 75.44] m and ``ramp_bases`` over the observed ramp acceleration lane
    (ramp entrance -378.66 m to the merge nose).  Geometry only -- entrance
    positions and section boundaries of the observed network -- never a
    speed/jam/hold behavioural threshold.  Each width is the nearest-centre
    distance, which reproduces the uniform grid's sigma exactly and scales
    every centre to its local resolution; identical endpoints across groups
    (e.g. the merge end shared with the backbone) are deduplicated.
    """
    start, end = min(ENTRY_S_M), SECTION_MERGE_END_M
    groups = [torch.linspace(start, end, basis, device=device, dtype=dtype)]
    if merge_bases:
        groups.append(torch.linspace(SECTION_MERGE_START_M, SECTION_MERGE_END_M,
                                     merge_bases, device=device, dtype=dtype))
    if ramp_bases:
        groups.append(torch.linspace(ENTRY_S_M[2], SECTION_MERGE_START_M,
                                     ramp_bases, device=device, dtype=dtype))
    centres = torch.unique(torch.cat(groups))
    distances = (centres[:, None] - centres[None, :]).abs().fill_diagonal_(float("inf"))
    return centres, distances.min(-1).values


class MergeResponseField(nn.Module):
    def __init__(self, hidden: int = 32, basis: int = 4,
                 merge_bases: int = 0, ramp_bases: int = 0) -> None:
        super().__init__()
        if hidden < 1 or basis < 2:
            raise ValueError("positive hidden width and at least two spatial bases required")
        for name, count in (("merge_bases", merge_bases), ("ramp_bases", ramp_bases)):
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
            if count == 1:
                raise ValueError(f"{name} must be 0 (off) or >= 2 (a resolvable region)")
        self.basis = basis
        self.merge_bases, self.ramp_bases = int(merge_bases), int(ramp_bases)
        self.refined = bool(self.merge_bases or self.ramp_bases)
        self.centres = (len(refined_spatial_layout(basis, self.merge_bases,
                                                   self.ramp_bases)[0])
                        if self.refined else basis)
        self.net = nn.Sequential(nn.Linear(8 + self.centres * 9, hidden), nn.SiLU(),
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
        if self.refined:
            centres, widths = refined_spatial_layout(self.basis, self.merge_bases,
                                                     self.ramp_bases, device=kin.device,
                                                     dtype=kin.dtype)
            weight = torch.exp(-0.5 * ((s[..., None] - centres) / widths) ** 2)
        else:
            centres = torch.linspace(start, end, self.basis, device=kin.device, dtype=kin.dtype)
            weight = torch.exp(-0.5 * ((s[..., None] - centres) / (span / (self.basis - 1))) ** 2)
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
