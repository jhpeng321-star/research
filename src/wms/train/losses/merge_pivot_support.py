"""Recorded merges whose physical support the P1 contract cannot represent.

P1 evaluates the decision geometry on the pre-change snapshot (one longitudinal
update, pre-change lanes, single commit).  Two recorded phenomena can make an
event unsupported under P1 although SUMO executed it legally; both are decided
from the recorded SUCCESSOR row, which is a LABEL-SIDE diagnostic only and is never
a model input:

``same_step_vacated_leader``
    SUMO changes lanes sequentially, front-most vehicle first.  A target-lane
    vehicle that is AHEAD of the ego and leaves the target lane in the same step is
    already gone when the ego decides, but its pre-change membership still blocks the
    ego under P1.  (All 8 such TRAIN events: the overlapping vehicle is ahead.)
``junction_internal_target_vehicle``
    Every body that still intersects the ego (route frame, recorded successor) is a
    vehicle inside an internal junction lane; SUMO's lane-change safety check did not
    treat it as blocking.

An event is *explained* only when it is vetoed by the model geometry, the recorded
successor (route frame, pre-change lane membership) really has intersecting bodies,
and EVERY intersecting body is one of the two cases above.  An event whose recorded
successor is clear is a failure of the model's own one-step motion (class a) and is
never explained away.  The identical function feeds the likelihood, the fitter and
the support audit.
"""
from __future__ import annotations

import torch
from torch import Tensor

REASONS = ("same_step_vacated_leader", "junction_internal_target_vehicle")


@torch.no_grad()
def classify_vetoed_events(geo: dict[str, Tensor], event: Tensor, *, target_lane: int
                           ) -> dict[str, Tensor]:
    """Per-event masks ``[B, n, K]`` (all False off vetoed events).

    ``geo`` is ``build_recorded_geometry`` output including its ``label_*`` successor
    rows.  ``event`` marks recorded ramp->target transitions of modelled vehicles.
    Returns ``vetoed``, ``recorded_clear`` (successor bodies do not intersect: the
    veto comes from the model's own motion), ``vacated``, ``junction`` and
    ``explained``.
    """
    vetoed = event.bool() & ~geo["eligible"].bool()
    out = {key: torch.zeros_like(vetoed) for key in
           ("recorded_clear", "vacated", "junction", "explained")}
    out["vetoed"] = vetoed
    x, length = geo["label_succ_x"], geo["label_succ_length"]
    for b, t in torch.nonzero(vetoed.any(-1)).tolist():
        egos = torch.nonzero(vetoed[b, t]).flatten()
        xe, le = x[b, t, egos].unsqueeze(-1), length[b, t, egos].unsqueeze(-1)
        xo, lo = x[b, t].unsqueeze(0), length[b, t].unsqueeze(0)
        others = (geo["pre_target"][b, t].bool() & geo["label_linked"][b, t].bool()).unsqueeze(0)
        others = others & (torch.arange(x.shape[-1], device=x.device) != egos.unsqueeze(-1))
        hit = others & ((torch.minimum(xe, xo) - torch.maximum(xe - le, xo - lo)) > 0.0)
        left = (geo["label_succ_lane"][b, t] != target_lane).unsqueeze(0)
        vacated = left & (xo > xe)
        junction = geo["label_pre_junction"][b, t].bool().unsqueeze(0)
        some = hit.any(-1)
        out["recorded_clear"][b, t, egos] = ~some
        out["vacated"][b, t, egos] = some & (~hit | vacated).all(-1)
        out["junction"][b, t, egos] = some & (~hit | junction).all(-1)
        out["explained"][b, t, egos] = some & (~hit | vacated | junction).all(-1)
    return out


def sequence_rows(excluded_event: Tensor, risk: Tensor, reset: Tensor) -> Tensor:
    """Rows ``[B, n, K]`` of the risk sequences that end in an excluded event.

    Walks back from the event while the identity is continuous (``reset[:, i + 1]``
    starts a new identity) and the vehicle stays in risk.  Dropping these rows from
    ``risk``/``event`` removes the whole sequence, never just its terminal event.
    """
    n = excluded_event.shape[1]
    rows = torch.zeros_like(excluded_event)
    carry = torch.zeros_like(excluded_event[:, 0])
    for i in range(n - 1, -1, -1):
        linked = ~reset[:, i + 1].bool() if i + 1 < n else torch.ones_like(carry)
        carry = (carry & linked & risk[:, i].bool()) | excluded_event[:, i]
        rows[:, i] = carry
    return rows
