"""Part B losses (DESIGN_PARTS_A_B.md §3.5).

Gaussian NLL on R_native and dR; pairwise ranking hinge across candidate
plans of the same branch point (hinge on the sign of dR_i - dR_j);
near-zero dR targets are up-weighted because decision flips happen there.
"""
from __future__ import annotations

import torch


def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor,
                 target: torch.Tensor, weight: torch.Tensor | None = None
                 ) -> torch.Tensor:
    """mean over batch of 0.5*(logvar + (target-mean)^2/exp(logvar))."""
    nll = 0.5 * (logvar + (target - mean) ** 2 / logvar.exp())
    if weight is not None:
        return (nll * weight).sum() / weight.sum().clamp(min=1e-8)
    return nll.mean()


def near_zero_weight(target: torch.Tensor, width: float = 2.0,
                     boost: float = 1.0) -> torch.Tensor:
    """w(y) = 1 + boost * exp(-(y/width)^2): 1 far from zero, 1+boost at 0."""
    return 1.0 + boost * torch.exp(-(target / width) ** 2)


def pairwise_ranking_hinge(dr_mean: torch.Tensor, dr_target: torch.Tensor,
                           group_ids: torch.Tensor | None = None,
                           margin: float = 1.0) -> torch.Tensor:
    """Ranking hinge over candidate plans within each branch point.

    For pairs (i, j) in the same group with sign s = sign(y_i - y_j)
    (+/-1; ties skipped): loss = relu(margin - s * (m_i - m_j)).
    group_ids None = one group.  Returns mean over valid pairs (0.0 if no
    valid pair).
    """
    n = dr_mean.shape[0]
    if n < 2:
        return dr_mean.new_zeros(())
    if group_ids is None:
        group_ids = torch.zeros(n, dtype=torch.long, device=dr_mean.device)
    same = group_ids.unsqueeze(0) == group_ids.unsqueeze(1)
    iu, ju = torch.triu_indices(n, n, offset=1)
    same = same[iu, ju]
    iu, ju = iu[same], ju[same]
    if iu.numel() == 0:
        return dr_mean.new_zeros(())
    sign = torch.sign(dr_target[iu] - dr_target[ju])
    keep = sign != 0
    if not bool(keep.any()):
        return dr_mean.new_zeros(())
    iu, ju, sign = iu[keep], ju[keep], sign[keep]
    viol = margin - sign * (dr_mean[iu] - dr_mean[ju])
    return torch.relu(viol).mean()


def partb_loss(pred: torch.Tensor, target: torch.Tensor,
               group_ids: torch.Tensor | None = None,
               margin: float = 1.0, rank_weight: float = 0.3,
               near_zero: bool = True) -> dict:
    """Combined loss.  pred [B,4] = (R_mean, R_logvar, dR_mean, dR_logvar);
    target [B,2] = (R_native, dR).  Returns dict of scalar tensors with a
    detached 'total_item' mean for logging."""
    r_mean, r_lv = pred[:, 0], pred[:, 1]
    dr_mean, dr_lv = pred[:, 2], pred[:, 3]
    w = near_zero_weight(target[:, 1]) if near_zero else None
    nll_r = gaussian_nll(r_mean, r_lv, target[:, 0])
    nll_dr = gaussian_nll(dr_mean, dr_lv, target[:, 1], weight=w)
    rank = pairwise_ranking_hinge(dr_mean, target[:, 1], group_ids,
                                  margin=margin)
    total = nll_r + nll_dr + rank_weight * rank
    return dict(total=total,
                nll_r=nll_r.detach(), nll_dr=nll_dr.detach(),
                rank=rank.detach(),
                total_item=total.detach())
