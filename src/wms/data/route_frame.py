"""Lossless longitudinal route coordinate for the recorded WM-S scene.

The pack frame ``s`` ("model frame", ``traffic_shepherd/c4/geometry.py``) joins the
upstream edges, the merge edge and the downstream edge on ONE line and collapses
every internal junction lane to zero length.  Vehicles inside a junction have no
lane one-hot and their ``s`` is frozen at the last on-lane value.  Consequences,
both measured on TRAIN episodes:

* the collapsed frame shortens every bumper-to-bumper distance that spans a
  junction (3.00 m main / 2.93 m ramp junction before the merge edge, 8.00 m
  junction after it), and
* a vehicle inside a junction appears to stand still although it keeps its speed.

The SUMO lane-change safety check works on true route distances, so the physical
support of a recorded merge must be evaluated in the route frame
``x = s + shift``.  The shift is exact and causal:

* on-lane rows: the constant of the edge the vehicle is on (below);
* junction rows: SUMO's Euler position update ``x_t = x_{t-1} + v_t * dt`` from the
  previous row (checked on 13,503 TRAIN junction crossings: the dead-reckoned
  position equals the first on-lane row after the junction to < 1e-5 m).

``x`` uses the merge-edge start as origin.  Junction lengths are the lengths of the
internal lanes in ``assets/research/highway_merge_shepherd_long/osm.net.xml.gz``
(``:1780739017_0_0`` 2.93 m ramp, ``:1780739017_1_*`` 3.00 m main,
``:13638027693_0_*`` 8.00 m downstream); the test-suite re-reads them from the
(unmodified) network.  Only the observed present row and its predecessor are used.
"""
from __future__ import annotations

import torch
from torch import Tensor

#: length (m) of the internal junction lane in front of the merge edge, ramp / main.
JUNCTION_BEFORE_RAMP_M = 2.93
JUNCTION_BEFORE_MAIN_M = 3.00
#: length (m) of the internal junction lane behind the merge edge (both main lanes).
JUNCTION_AFTER_M = 8.00
#: float32 pack-frame tolerance at the merge-edge end (75.44 -> 75.440002).
EDGE_END_TOL_M = 1.0e-4
#: slack on the Euler-consistency of a dead-reckoned junction position.
JUNCTION_TOL_M = 1.0e-3
#: below this travelled distance a repeated ``s`` is a standing vehicle, not a junction row.
MIN_MOTION_M = 1.0e-3


def static_shift(s: Tensor, lane: Tensor, zone_start_m: float, zone_end_m: float) -> Tensor:
    """Route-frame minus pack-frame offset of an ON-LANE row (``s`` is the pack value).

    ``s < zone_start`` is the upstream edge (the junction in front of the merge edge
    lies between it and ``x = 0``); the merge edge ``[zone_start, zone_end]`` is the
    origin; ``s > zone_end`` on a mainline lane is the downstream edge, which starts
    after the 8 m junction.  The ramp lane ends at ``zone_end`` and never continues.
    """
    ramp = lane == 2
    before = torch.where(ramp, torch.full_like(s, -JUNCTION_BEFORE_RAMP_M),
                         torch.full_like(s, -JUNCTION_BEFORE_MAIN_M))
    after = torch.full_like(s, JUNCTION_AFTER_M)
    zero = torch.zeros_like(s)
    return torch.where(s < zone_start_m, before,
                       torch.where((s > zone_end_m + EDGE_END_TOL_M) & ~ramp, after, zero))


def junction_span(x: Tensor, lane: Tensor, zone_start_m: float, zone_end_m: float
                  ) -> tuple[Tensor, Tensor, Tensor]:
    """``(has_junction, start, end)`` of the internal lane a vehicle at route ``x`` is on or about to enter.

    Before the merge edge (``x <= zone_start``): the junction in front of it
    (3.00 m main / 2.93 m ramp, ending at the merge-edge start).  On the merge edge a
    mainline vehicle faces the 8 m junction behind the edge; the ramp lane dead-ends.
    """
    ramp = lane == 2
    before_len = torch.where(ramp, torch.full_like(x, JUNCTION_BEFORE_RAMP_M),
                             torch.full_like(x, JUNCTION_BEFORE_MAIN_M))
    pre = x <= zone_start_m
    post = (x > zone_start_m) & (x <= zone_end_m + JUNCTION_AFTER_M + JUNCTION_TOL_M) & ~ramp
    start = torch.where(pre, zone_start_m - before_len, torch.full_like(x, zone_end_m))
    end = torch.where(pre, torch.full_like(x, zone_start_m),
                      torch.full_like(x, zone_end_m + JUNCTION_AFTER_M))
    return pre | post, start, end


@torch.no_grad()
def route_shift(states: Tensor, present: Tensor, continuity: Tensor | None, *,
                dt_s: float, zone_start_m: float = 0.0, zone_end_m: float = 75.44,
                return_junction: bool = False):
    """Per-row ``x - s`` for causal recorded states ``[B, T, K, 7]`` (float64 maths).

    With ``return_junction`` also returns the ``[B, T, K]`` mask of dead-reckoned
    (internal junction lane) rows.

    ``states`` must be the causal scene (lane one-hot carried across junction rows).
    A row is a junction row when the identity is continuous, ``s`` is bit-identical
    to the previous row, the speed moves the vehicle (``v*dt`` > 1 mm) AND the Euler
    position ``x_{t-1} + v_t dt`` lies inside the internal lane the vehicle can be on
    (past its start, not beyond its end; a vehicle standing mid-edge is never one);
    such a row is dead-reckoned with ``x_t = x_{t-1} + v_t dt``.  ``continuity[:, t]`` certifies the
    identity between rows ``t`` and ``t+1`` (``None``: every consecutive present pair).
    Rows that are not present return 0.
    """
    b, t, k, _ = states.shape
    s = states[..., 0].double()
    v = states[..., 4].double()
    lane = states[..., 1:4].argmax(-1)
    present = present.bool()
    static = static_shift(s, lane, zone_start_m, zone_end_m)
    x = s + static
    shift = static.clone()
    junction = torch.zeros(b, t, k, dtype=torch.bool, device=states.device)
    if continuity is None:
        continuity = torch.ones(b, max(t - 1, 0), k, dtype=torch.bool, device=states.device)
    for i in range(1, t):
        linked = continuity[:, i - 1].bool() & present[:, i - 1] & present[:, i]
        reckoned = x[:, i - 1] + v[:, i] * dt_s
        has, start, end = junction_span(x[:, i - 1], lane[:, i], zone_start_m, zone_end_m)
        # A chain that is already inside an internal lane stays there while it stands
        # still (v can be 0 inside a junction); a NEW chain needs real motion.
        moving = (v[:, i] * dt_s > MIN_MOTION_M) | junction[:, i - 1]
        stale = (linked & (states[:, i, :, 0] == states[:, i - 1, :, 0]) & moving & has
                 & (reckoned > start) & (reckoned <= end + JUNCTION_TOL_M))
        x[:, i] = torch.where(stale, reckoned, x[:, i])
        shift[:, i] = x[:, i] - s[:, i]
        junction[:, i] = stale
    shift = torch.where(present, shift, torch.zeros_like(shift)).to(states.dtype)
    return (shift, junction) if return_junction else shift
