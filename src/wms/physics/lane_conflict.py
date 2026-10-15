"""Joint lane assignment feasibility, without changing lane-choice logits."""
from __future__ import annotations

import torch
from torch import Tensor


def resolve_lane_conflicts(s: Tensor, length: Tensor, lane: Tensor, proposed: Tensor,
                           present: Tensor, movable: Tensor, margin_m: float) -> Tensor:
    """Admit changes front-to-back, checking the post-motion bumper intervals.

    Start with every movable vehicle in its source lane and fixed (teacher/CAV)
    vehicles at their supplied destinations. Rejected changes therefore keep their
    source space reserved, including for later candidates. Ties use slot order.
    This is a conservative geometric arbitration: no logits, speeds or positions
    are changed, and disjoint lane changes remain admissible.
    """
    resolved = torch.where(movable, lane, proposed).clone()
    order = torch.argsort(s, dim=-1, descending=True, stable=True)
    slots = torch.arange(s.shape[-1], device=s.device)
    for batch in range(s.shape[0]):
        for index in order[batch]:
            if not (present[batch, index] and movable[batch, index]
                    and proposed[batch, index] != lane[batch, index]):
                continue
            target = proposed[batch, index]
            other = (present[batch] & (slots != index) & (resolved[batch] == target))
            hit = ((s[batch] - length[batch] < s[batch, index] + margin_m)
                   & (s[batch] > s[batch, index] - length[batch, index] - margin_m))
            if not (other & hit).any():
                resolved[batch, index] = target
    return resolved
