"""Frozen wall-law v2.2 ``merge-release`` — the P1 reference implementation.

This module mirrors the registered law so the batched Idea A world needs no
session server: the constants, the proposer gate, the hold cap and the
HOLD-only ``merge_release`` handback are transcribed from the FROZEN patched
copies ``output/wd01/walls_design/merge_release/copies/post`` (branch
``zcode/wd01``; registration ``REGISTRATION_WALL_V22.md``, design
``DESIGN_V22_MERGE_RELEASE.md``, LF e7fd3536).  Values are law constants:
never tuned on outcomes.

Frozen sources of each piece (post copies):

* constants                — ``common.py:120-127`` / ``wall_group.py:79-88``
* ``min_proposal_distance_m`` (D_min = 70 * v_wall) — ``wall_request.py``
* ``hold_cap_s``           — ``wall_request.py``
* ``dmin_unreachable`` refusal record — ``wall_request.py:_pairs_from_candidates``
* 2-abreast span/lane gate — ``wall_request.py`` (pair on the entry edge,
  two distinct lanes, span <= 75 m)
* ``merge_release`` (HOLD only, once, handback native) — ``wall_group.py``
* ``hold_cap`` log row     — ``wall_driver.py:hold_cap_row``

Road geometry (cumulative ``s_road``, edge-aware; identical to
:mod:`ctrl.wall_hook`): entry 1191554636 = 2644.2 m, merge 1487606179 =
75.4 m, exit 1487606178 = 477.9 m; MERGE START = 2644.2 m, MERGE END =
2719.6 m, NETWORK END = 3197.5 m.  In the batched merge-relative world
``s_road = s_m + MERGE_START_S``.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
from torch import Tensor

from .config import WALL_LAW_V22
from .wall_hook import MergeReleaseWallLaw

# ------------------------------------------------------------ law constants

#: wall-v2.2-merge-release (== ctrl.config.WALL_LAW_V22).
LAW_VERSION = WALL_LAW_V22

#: cumulative s_road of the merge start (entry edge 1191554636 length).
MERGE_START_S = 2644.2
#: cumulative s_road of the merge end (merge edge 1487606179, 75.4 m).
MERGE_END_S = 2719.6
#: cumulative s_road of the network end (exit edge 1487606178, 477.9 m).
NETWORK_END_S = 3197.5
#: owner binding: release point X = 0 m before the merge start.
MERGE_RELEASE_X_M = 0.0
#: release point on s_road: MERGE_START_S - X = 2644.2 m.
MERGE_RELEASE_POINT_S = MERGE_START_S - MERGE_RELEASE_X_M
#: minimal meaningful hold: one 60 s planner decision interval = the
#: smallest registered fixed-arm draw (common.py:126).
MIN_MEANINGFUL_HOLD_S = 60.0
#: a-priori J4 assembly time: the n=131 assembled median, mechanics only
#: (common.py:127).
ASSEMBLY_TIME_ASSUMED_S = 10.0
#: fixed-arm hold draws, recorded AS DRAWN (common.py:71).
FIXED_ARM_DRAWS_S = (60.0, 120.0, 240.0)
#: request span cap: two members across the two mainline lanes (2026-09-30
#: A2 revert of the 6 m tightening).
REQUEST_SPAN_CAP_M = 75.0
#: geometric comparisons fire at ``POINT - tol`` (wall_group.py uses 1e-9).
GEOM_TOL = 1e-9
#: D_min speed factor: X + v * (t_assemble + H_min) = 70 * v.
DMIN_SPEED_FACTOR_S = ASSEMBLY_TIME_ASSUMED_S + MIN_MEANINGFUL_HOLD_S


def _check_wall_speed(wall_speed_mps: float) -> float:
    speed = float(wall_speed_mps)
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("wall_speed_mps must be finite and positive")
    return speed


# ------------------------------------------------------- proposer (a priori)

def min_proposal_distance_m(wall_speed_mps: float) -> float:
    """The v2.2 proposer minimum ``D_min`` (owner binding, a priori):

        D_min = X + v_wall * t_assemble + v_wall * H_min
              = 0 + v_wall * (10 s + 60 s) = 70 * v_wall   [m]

    A request whose LEAD member is closer to the merge start than D_min
    cannot assemble AND hold meaningfully before the merge start, so it is
    refused at the proposal (never assembled into a forced release).
    """
    speed = _check_wall_speed(wall_speed_mps)
    return round(float(MERGE_RELEASE_X_M) + speed * DMIN_SPEED_FACTOR_S, 6)


def dmin_refusal(lead_s_road_m: float, wall_speed_mps: float) -> Optional[Dict]:
    """The ``dmin_unreachable`` refusal record, or ``None`` when reachable.

    Gate (frozen): the LEAD member's distance to the merge start
    (``MERGE_START_S - s_road(lead)``) must be ``>= D_min`` at the request;
    a span-feasible pair failing only this gate is refused AT THE PROPOSAL
    with the lead distance, D_min and wall speed logged.
    """
    speed = _check_wall_speed(wall_speed_mps)
    lead_s = float(lead_s_road_m)
    lead_distance = float(MERGE_START_S) - lead_s
    dmin = min_proposal_distance_m(speed)
    if lead_distance >= float(dmin) - GEOM_TOL:
        return None
    return {
        "reason": "dmin_unreachable",
        "lead_s_road_m": round(lead_s, 6),
        "lead_distance_to_merge_m": round(lead_distance, 6),
        "dmin_m": dmin,
        "wall_speed_mps": speed,
    }


def check_pair(lanes: List[int], positions_s_road_m: List[float],
               wall_speed_mps: float) -> Dict:
    """The v2.2 proposal gate for a candidate wall PAIR (2 abreast).

    Frozen order (``wall_request._pairs_from_candidates``): the two members
    sit in DISTINCT lanes (one CAV per mainline lane, 2 abreast), their
    span ``|s_a - s_b|`` is within :data:`REQUEST_SPAN_CAP_M`, and the LEAD
    member passes the D_min gate.  Returns ``{"feasible": bool,
    "refusal": dict | None, ...}``; the D_min refusal record is exactly
    :func:`dmin_refusal` (reason ``dmin_unreachable``).
    """
    speed = _check_wall_speed(wall_speed_mps)
    if len(lanes) != 2 or len(positions_s_road_m) != 2:
        raise ValueError("a wall pair has exactly two members")
    span = abs(float(positions_s_road_m[0]) - float(positions_s_road_m[1]))
    lead_s = max(float(p) for p in positions_s_road_m)
    out = {
        "lanes": [int(ln) for ln in lanes],
        "span_m": round(span, 6),
        "lead_s_road_m": round(lead_s, 6),
        "lead_distance_to_merge_m": round(float(MERGE_START_S) - lead_s, 6),
        "wall_speed_mps": speed,
        "dmin_m": min_proposal_distance_m(speed),
        "feasible": True,
        "refusal": None,
    }
    if len(set(int(ln) for ln in lanes)) != 2:
        out["feasible"] = False
        out["refusal"] = {"reason": "not_2_abreast",
                          "lanes": out["lanes"]}
        return out
    if span > float(REQUEST_SPAN_CAP_M) + GEOM_TOL:
        out["feasible"] = False
        out["refusal"] = {"reason": "span_exceeds_cap",
                          "span_m": out["span_m"],
                          "span_cap_m": float(REQUEST_SPAN_CAP_M)}
        return out
    refusal = dmin_refusal(lead_s, speed)
    if refusal is not None:
        out["feasible"] = False
        out["refusal"] = refusal
    return out


# ------------------------------------------------------- hold cap (fixed arm)

def hold_cap_s(hold_s: float, lead_s_road_m: float,
               wall_speed_mps: float) -> float:
    """The v2.2 hold cap (owner binding: the hold ALWAYS ends at the merge
    start at the latest): the registered draw ``hold_s`` capped by the time
    the lead member needs to reach the release point at the wall speed,

        hold_cap = min(H, max(0, 2644.2 - s_lead) / v_wall)   [s]

    The draw stays recorded AS DRAWN; this is the a-priori effective hold
    the session's ``merge_release`` enforces on the live positions.
    """
    speed = _check_wall_speed(wall_speed_mps)
    to_release = max(0.0, float(MERGE_RELEASE_POINT_S) - float(lead_s_road_m))
    return round(min(float(hold_s), to_release / speed), 6)


def hold_cap_record(*, wall_id: str, timestamp_s: float, requested_hold_s: float,
                    wall_speed_mps: float, lead_s_road_m: float,
                    effective_hold_s: Optional[float] = None,
                    capped_at_merge: Optional[bool] = None) -> Dict:
    """The realized ``hold_cap`` log row (frozen ``wall_driver.hold_cap_row``
    field set), logged ONCE per wall right after its terminal event.
    ``effective_hold_s`` / ``capped_at_merge`` are engine-realized values;
    when ``capped_at_merge`` is not given it is derived as
    ``hold_cap_s(requested) < requested``.  ``a_priori_effective_hold_s``
    is :func:`hold_cap_s` — computed before any outcome is read."""
    a_priori = hold_cap_s(requested_hold_s, lead_s_road_m, wall_speed_mps)
    if capped_at_merge is None:
        capped_at_merge = a_priori < round(float(requested_hold_s), 6)
    if effective_hold_s is None:
        effective_hold_s = a_priori
    return {
        "kind": "hold_cap",
        "wall_id": str(wall_id),
        "timestamp_s": round(float(timestamp_s), 6),
        "requested_hold_s": float(requested_hold_s),
        "effective_hold_s": round(float(effective_hold_s), 6),
        "capped_at_merge": bool(capped_at_merge),
        "cap_point_s": float(MERGE_RELEASE_POINT_S),
        "a_priori_effective_hold_s": a_priori,
        "lead_s_road_at_release_m": round(float(lead_s_road_m), 6),
        "wall_speed_mps": float(wall_speed_mps),
        "source": "cav_report_frames",
    }


# ------------------------------------------------- merge-release (HOLD, once)

def merge_release_event(*, wall_id: int, timestamp_s: float) -> Dict:
    """The frozen ``merge_release`` wall event (wall_group.py, verbatim
    fields): a RELEASE kind, never an abort; members hand back to native."""
    return {
        "kind": "wall_event",
        "wall_id": int(wall_id),
        "event": "merge_release",
        "timestamp_s": round(float(timestamp_s), 6),
        "release_kind": "merge_release",
        "release_point_s": float(MERGE_RELEASE_POINT_S),
        "merge_start_s": float(MERGE_START_S),
        "handback": "native",
        "law_version": LAW_VERSION,
    }


class MergeReleaseSession:
    """Engine-owned per-(batch-row, wall-id) merge-release state.

    The stateless rule (:meth:`MergeReleaseWallLaw.release_mask`) says
    WHICH slots are released at this tick; this session adds the frozen
    "fires exactly once" bookkeeping: the first tick whose HOLD members'
    lead reaches the release point latches the wall, emits ONE
    :func:`merge_release_event` into :attr:`events`, and never fires for
    that (row, wall) again.  Pure torch, deterministic, no RNG.
    """

    def __init__(self) -> None:
        self.events: List[Dict] = []
        self._fired: set = set()
        self._law = MergeReleaseWallLaw()

    @property
    def fired(self) -> set:
        """``{(batch_row, wall_id), ...}`` that have merge-released."""
        return set(self._fired)

    def lead_s_road_m(self, s_road_m: Tensor, group: Tensor,
                      row: int, wall_id: int) -> float:
        """Max member s_road of one (row, wall) at the last :meth:`update`."""
        g = group.to(torch.int64)
        pos = s_road_m.to(torch.float64)
        members = (g == int(wall_id))
        if g.dim() > 1:
            sel = members[int(row)]
            vals = pos[int(row)][sel]
        else:
            vals = pos[members]
        return float(vals.max().item())

    def update(self, now_s: float, s_road_m: Tensor, group: Tensor,
               hold: Tensor) -> Tensor:
        """One control tick: -> released mask (same shape as ``group``),
        emitting the merge_release event once per newly released wall."""
        released = self._law.release_mask(s_road_m, group, hold)
        flat = released.reshape(-1)
        if bool(flat.any()):
            gid = group.to(torch.int64).reshape(-1)
            rows = torch.nonzero(flat, as_tuple=False).reshape(-1)
            seen = set()
            for idx in rows.tolist():
                key = (int(idx), int(gid[idx].item()))
                if key in self._fired or key in seen:
                    continue
                seen.add(key)
                self.events.append(merge_release_event(
                    wall_id=key[1], timestamp_s=now_s))
            self._fired |= seen
        return released

    def reset(self) -> None:
        """Forget fired walls and events (new episode)."""
        self.events.clear()
        self._fired.clear()
