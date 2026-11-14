"""Differentiable physics feasibility projection applied after a rollout step.

Enforces a minimum bumper gap per lane, speed bounds and valid lane ids without
creating or removing vehicles.  Gradients flow through the clamp/min operators,
so the projection can sit inside a training rollout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

#: Slack used when counting gap violations (m).
_GAP_TOL_M = 1.0e-6
#: Minimum reduction (m) for a position to count as clamped.
_CLAMP_TOL_M = 1.0e-9


@dataclass
class ProjectionConfig:
    """Constraints enforced by :class:`FeasibilityProjection`.

    ``min_gap_m`` is the bumper gap that must hold on every lane;
    ``v_max_mps``/``v_min_mps`` bound speed.  ``project_cavs`` is off so a
    teacher-forced CAV keeps its recorded position and only acts as an immovable
    obstacle for the learned HDVs.
    """

    min_gap_m: float = 2.0
    v_max_mps: float = 31.29
    v_min_mps: float = 0.0
    n_lanes: int = 3
    valid_lanes: tuple[int, ...] = (0, 1, 2)
    project_cavs: bool = False
    speed_tol_mps: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.min_gap_m < 0.0:
            raise ValueError("min_gap_m must be >= 0")
        if self.v_max_mps < self.v_min_mps:
            raise ValueError("v_max_mps must be >= v_min_mps")
        invalid = [int(x) for x in self.valid_lanes
                   if int(x) < 0 or int(x) >= self.n_lanes]
        if invalid:
            raise ValueError(f"valid_lanes outside [0, n_lanes): {invalid}")


_STAT_KEYS = ("steps", "gap_before", "gap_active", "s_clamped", "speed_high",
              "speed_low", "invalid_lane", "present_changed")


class FeasibilityProjection:
    """Projection of a post-step state onto the feasible set.

    ``__call__(s, v, lane, present, length, is_cav=None)`` returns the projected
    ``s_m``/``v_mps``/``lane`` plus a per-step ``active`` dict of integer
    constraint counts.  Running totals are kept in :attr:`stats`.
    """

    def __init__(self, config: Optional[ProjectionConfig] = None) -> None:
        self.config = config or ProjectionConfig()
        self.reset_stats()

    def reset_stats(self) -> None:
        self.stats: dict[str, float] = {key: 0.0 for key in _STAT_KEYS}

    def __call__(self, s: Tensor, v: Tensor, lane: Tensor, present: Tensor,
                 length: Optional[Tensor] = None, *,
                 is_cav: Optional[Tensor] = None) -> dict[str, Tensor | dict]:
        cfg = self.config
        present = present.bool()
        if length is None:
            length = torch.full_like(s, 5.0)
        length = length.to(s.dtype)
        lane = lane.long()
        is_cav = torch.zeros_like(present) if is_cav is None else is_cav.bool()
        projectable = present & (bool(cfg.project_cavs) | ~is_cav)

        lane_ok = (lane >= 0) & (lane < int(cfg.n_lanes))
        invalid_lane = present & ~lane_ok
        lane_proj = torch.where(invalid_lane, lane.clamp(0, int(cfg.n_lanes) - 1), lane)

        speed_high = present & (v > cfg.v_max_mps + cfg.speed_tol_mps)
        speed_low = present & (v < cfg.v_min_mps - cfg.speed_tol_mps)
        v_clamped = v.clamp(cfg.v_min_mps, cfg.v_max_mps)
        v_proj = v_clamped if cfg.project_cavs else torch.where(projectable, v_clamped, v)

        s_proj, gap_before, s_clamped = project_gaps(
            s, length, lane_proj, present, projectable, cfg.min_gap_m,
            n_lanes=int(cfg.n_lanes))

        active = {"gap_before": int(gap_before), "gap_active": int(s_clamped),
                  "s_clamped": int(s_clamped), "speed_high": int(speed_high.sum()),
                  "speed_low": int(speed_low.sum()),
                  "invalid_lane": int(invalid_lane.sum()), "present_changed": 0}
        self.stats["steps"] += 1.0
        for key, value in active.items():
            self.stats[key] += float(value)
        return {"s_m": s_proj, "v_mps": v_proj, "lane": lane_proj, "active": active}

    def summary(self) -> dict[str, float]:
        """Totals plus rates per step."""
        steps = max(self.stats["steps"], 1.0)
        total_active = (self.stats["gap_active"] + self.stats["speed_high"]
                        + self.stats["speed_low"] + self.stats["invalid_lane"])
        return {**self.stats,
                "gap_active_rate": self.stats["gap_active"] / steps,
                "speed_active_rate": (self.stats["speed_high"]
                                      + self.stats["speed_low"]) / steps,
                "invalid_lane_rate": self.stats["invalid_lane"] / steps,
                "any_active_rate": total_active / steps}


def project_gaps(s: Tensor, length: Tensor, lane: Tensor, present: Tensor,
                 projectable: Tensor, min_gap: float, n_lanes: int = 3
                 ) -> tuple[Tensor, int, int]:
    """Push followers back until every leader gap is at least ``min_gap``.

    Within a ``(row, lane)`` group the vehicles are ordered front to back and the
    recurrence ``y[0] = x[0]``; ``y[j] = min(x[j], y[j-1] - L[j-1] - min_gap)`` for
    projectable vehicles (anchors keep ``y[j] = x[j]``) is evaluated in closed form
    as a segmented prefix minimum ending at the last anchor::

        y[j] = min_{i in [anchor(j)..j]} (x[i] + S[i]) - S[j]

    with ``S[0] = 0`` and ``S[j] = sum(L[:j] + min_gap)``: each offset
    uses the preceding leader's length.  The result is built with gathers so every
    position keeps its autograd history.

    Returns ``(projected_s, gap_violations_before, n_clamped)``.
    """
    out = s.clone()
    gap_before_total = torch.zeros((), dtype=torch.long, device=s.device)
    clamped_total = torch.zeros((), dtype=torch.long, device=s.device)
    for row in range(int(s.shape[0])):
        for lane_id in range(int(n_lanes)):
            members = torch.nonzero(present[row] & (lane[row] == lane_id),
                                    as_tuple=False).flatten()
            if members.numel() <= 1:
                continue
            front_to_back = torch.flip(members[torch.argsort(s[row, members])], [0])
            y, violations, clamped = _project_lane(
                s[row, front_to_back], length[row, front_to_back],
                projectable[row, front_to_back], float(min_gap))
            out[row, front_to_back] = y
            gap_before_total = gap_before_total + violations
            clamped_total = clamped_total + clamped
    return out, int(gap_before_total), int(clamped_total)


def _project_lane(x: Tensor, length: Tensor, projectable: Tensor, min_gap: float
                  ) -> tuple[Tensor, Tensor, Tensor]:
    """Projection of one lane given positions ``x`` ordered front to back."""
    m = int(x.numel())
    device, dtype = x.device, x.dtype
    prefix = torch.zeros(m, dtype=dtype, device=device)
    prefix[1:] = torch.cumsum(length[:-1] + min_gap, dim=0)
    shifted = x + prefix
    anchor = ~projectable
    anchor[0] = True                                  # the front vehicle leads
    idx = torch.arange(m, device=device)
    last_anchor = torch.cummax(torch.where(anchor, idx, torch.full_like(idx, -1)),
                               dim=0).values
    in_segment = (idx.unsqueeze(0) <= idx.unsqueeze(1)) \
        & (idx.unsqueeze(0) >= last_anchor.unsqueeze(1))
    inf = torch.full((), float("inf"), dtype=dtype, device=device)
    seg_min = torch.where(in_segment, shifted.unsqueeze(0).expand(m, m), inf).amin(dim=1)
    y = seg_min - prefix
    projected = torch.where(projectable, y, x)
    gaps = x[:-1] - x[1:] - length[:-1]               # front vehicle vs the one behind
    violations = (gaps < min_gap - _GAP_TOL_M).sum()
    # Same comparison grouping as the per-vehicle reference loop so the
    # diagnostic count is not inflated by prefix-sum rounding.
    cap = (y[:-1] - length[:-1]) - min_gap
    clamped = (projectable[1:] & (cap < x[1:] - _CLAMP_TOL_M)).sum()
    return projected, violations, clamped
