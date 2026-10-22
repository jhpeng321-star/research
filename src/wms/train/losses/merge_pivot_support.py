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


# ---------------------------------------------------------------------------
# Counted exclusion of sequences made physically impossible by the WM's own motion
# (owner decision 2026-10-22).  ONE function, used identically by the training loss
# (MergePivotTerm.prepare), the offline fitter and the support audit.
#
# A risk chain (one car, one identity) is excluded WHOLE, counted and reported,
# when its exact likelihood is zero for every parameter value:
#   * the recorded event step has zero passage support under the WM one-step geometry
#     (E_T = 0: vetoed, or a present side with zero clearance), or
#   * its survival is exactly zero before the event/censoring: an earlier step of the
#     chain (inside the window, or inside the F2 pre-window prefix) was a certain
#     passage (eligible, no neighbour on either side, c = +inf) that did not end in
#     the recorded event.
# No threshold, floor, padding or tolerance is involved; nothing else is excluded.
# ---------------------------------------------------------------------------
CLASSES = ("event_vetoed_explained", "event_vetoed_wm_motion", "event_vetoed_other",
           "event_zero_clearance", "prefix_survival_zero", "wait_survival_zero")
_NONE = 99


@torch.no_grad()
def boundary_support(geo: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    """Parameter-independent ``(dead, atom)`` masks of ``FirstPassageNet.boundary``.

    ``dead``: ``c == -inf`` for every parameter value (ineligible, nonfinite speed, or a
    present side without positive finite clearance).  ``atom``: ``c == +inf`` (eligible
    with no neighbour on either side: certain immediate passage).
    """
    eligible = geo["eligible"].bool() & torch.isfinite(geo["v"])
    dead_side = torch.zeros_like(eligible)
    any_side = torch.zeros_like(eligible)
    for side in ("f", "l"):
        has = geo[f"has_{side}"].bool()
        gap, other = geo[f"gap_{side}"], geo[f"v_{side}"]
        valid = has & torch.isfinite(gap) & (gap > 0.0) & torch.isfinite(other)
        dead_side = dead_side | (has & ~valid)
        any_side = any_side | has
    return ~eligible | dead_side, eligible & ~any_side


def lifecycle_rows(risk: Tensor, event: Tensor, reset: Tensor) -> tuple[Tensor, Tensor]:
    """``births, live`` ``[B, n, K]``: the scored-chain bookkeeping of ``MergePivotTerm``.

    ``live`` = rows of a chain that is started, not yet terminal and in risk (the event
    row included).  A chain ends at a risk exit, at its event, or at an identity reset;
    later re-entries of the same identity are not scored.
    """
    seen = torch.zeros_like(risk[:, 0])
    terminal = torch.zeros_like(seen)
    births, live = [], []
    for i in range(risk.shape[1]):
        seen = seen & ~reset[:, i]
        terminal = terminal & ~reset[:, i]
        terminal = terminal | (seen & ~risk[:, i])
        births.append(risk[:, i] & ~seen)
        live.append(risk[:, i] & ~terminal)
        seen = seen | risk[:, i]
        terminal = terminal | event[:, i]
    return torch.stack(births, 1), torch.stack(live, 1)


def _continues(live: Tensor, event: Tensor, reset: Tensor) -> Tensor:
    """``cont[:, i]``: the chain row ``i`` continues into row ``i + 1`` of the same chain."""
    cont = torch.zeros_like(live)
    cont[:, :-1] = live[:, :-1] & live[:, 1:] & ~reset[:, 1:].bool() & ~event[:, :-1].bool()
    return cont


def prefix_survival_zero(histories, like: Tensor) -> Tensor:
    """``[B, K]``: the certified F2 pre-window prefix contains a certain passage.

    Such a car survived a step whose passage probability was one, so the F2
    conditioning denominator (prefix survival) is exactly zero.  ``histories`` is the
    ``build_pre_window_history`` / ``certify_pre_window`` list (``None`` = no prefix).
    """
    out = torch.zeros_like(like, dtype=torch.bool)
    for b, history in enumerate(histories):
        if history is None:
            continue
        _, atom = boundary_support(history["geometry"])
        hit = (atom & history["history_mask"]).any(1) & history["certified"]
        out[b, :hit.shape[-1]] = hit[0]
    return out


@torch.no_grad()
def counted_exclusion(geo: dict[str, Tensor], live: Tensor, event: Tensor, reset: Tensor, *,
                      target_lane: int, prefix_zero: Tensor | None = None) -> dict[str, Tensor]:
    """The shared counted exclusion; see the section comment above.

    ``live``/``event``/``reset`` are ``[B, n, K]`` (``lifecycle_rows`` bookkeeping, recorded
    ramp->target events of modelled vehicles, identity resets).  ``prefix_zero`` is the
    optional ``[B, K]`` ``prefix_survival_zero`` mask of chains that begin at row 0.
    Returns ``rows`` (chain rows to remove), ``tail`` (``rows`` plus later rows of the same
    identity, so a re-entry is not born as a new chain), ``starts`` (first row of each
    excluded chain), ``cls`` (class index into ``CLASSES`` at ``starts``, -1 elsewhere).
    A chain with several causes is counted once under the first cause listed in CLASSES.
    """
    live, event, reset = live.bool(), event.bool(), reset.bool()
    ev = event & live
    verdict = classify_vetoed_events(geo, ev, target_lane=target_lane)
    dead, atom = boundary_support(geo)
    code = torch.full(live.shape, _NONE, dtype=torch.int16, device=live.device)

    def put(mask, index):
        nonlocal code
        code = torch.where(mask, torch.minimum(code, torch.full_like(code, index)), code)

    vetoed = verdict["vetoed"]
    put(verdict["explained"], 0)
    put(verdict["recorded_clear"] & vetoed, 1)
    put(vetoed & ~verdict["explained"] & ~verdict["recorded_clear"], 2)
    put(ev & dead & ~vetoed, 3)
    if prefix_zero is not None:
        first = torch.zeros_like(live)
        first[:, 0] = live[:, 0] & prefix_zero.bool()
        put(first, 4)
    put(live & atom & ~event, 5)
    cont = _continues(live, event, reset)
    n = live.shape[1]
    big = torch.full_like(code[:, 0], _NONE)
    acc = code.clone()
    for i in range(1, n):
        acc[:, i] = torch.minimum(code[:, i], torch.where(cont[:, i - 1], acc[:, i - 1], big))
    full = acc.clone()
    for i in range(n - 2, -1, -1):
        full[:, i] = torch.minimum(acc[:, i], torch.where(cont[:, i], full[:, i + 1], big))
    rows = live & (full < _NONE)
    starts = rows.clone()
    starts[:, 1:] = rows[:, 1:] & ~cont[:, :-1]
    tail = rows.clone()
    for i in range(1, n):
        tail[:, i] = tail[:, i] | (tail[:, i - 1] & ~reset[:, i])
    cls = torch.where(starts, full.long(), torch.full_like(full, -1, dtype=torch.long))
    return dict(rows=rows, tail=tail, starts=starts, cls=cls, event=ev & rows,
                cls_rows=torch.where(rows, full.long(), torch.full_like(full, -1, dtype=torch.long)))


def exclusion_counts(res: dict[str, Tensor], scored_starts: Tensor | None = None) -> dict[str, int]:
    """Plain-int counts of excluded chains (cars) by class; ``scored_*`` restricts to scored cars."""
    out = {"excluded_cars": int(res["starts"].sum())}
    for index, name in enumerate(CLASSES):
        out[name] = int((res["cls"] == index).sum())
    if scored_starts is not None:
        hit = res["starts"] & scored_starts.bool()
        out["excluded_scored_cars"] = int(hit.sum())
        for index, name in enumerate(CLASSES):
            out["scored_" + name] = int(((res["cls"] == index) & hit).sum())
    return out
