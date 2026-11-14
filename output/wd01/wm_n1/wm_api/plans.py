"""Candidate-plan parsing: the PLANNED SCHEDULE, never an arm name.

Owner requirement (ACTION_DESCRIPTOR_REQUIREMENT_20261001.md): the action
descriptor is the planned schedule — the per-CAV controller for each 30 s
segment of the scoring horizon — plus the wall hold, speed and cap.  "Damp
60 s then native" and "damp 570 s" are different inputs.

Plan schema (api_version 1.0), one candidate plan per list entry::

    {
      "actions": {
        "<cav_id>": "native" | "damping" | "pi",
        "<cav_id>": [{"time_s": 0.0, "action": "damping"},
                     {"time_s": 60.0, "action": "native"}]
      },
      "wall": {"wall_id": 3, "speed_mps": 6.5, "hold_s": 120.0,
               "members": ["cav0", "cav1"], "onset_s": 0.0}
    }

* A per-CAV value is either one static action or a list of schedule events
  (``{vehicle_id, time_s, action}`` with the vehicle implied; the event
  times are relative to ``t0_s``).  Segment ``k`` (covering
  ``[t0 + 30k, t0 + 30(k+1))``) runs the latest event at or before its
  start, default ``native`` — the ``switch_schedule.json`` convention.
* ``wall`` is the wall-v2.2 GROUP membership + params: the member CAVs run
  the wall law from ``onset_s`` for ``hold_s`` (HOLD phase from t0;
  assembly is not modelled by the scoring engine) and the v2.2 merge
  release / hold cap still apply inside the mover.  A wall member must not
  also carry its own schedule.  Feasibility gates (2-abreast, span,
  D_min = 70 * v_wall) are REPORTED per plan (``gates``), not refused:
  the mover still commands only feasible wall slots.
* CAVs absent from ``actions`` stay ``native`` everywhere.  Actions are
  canonicalised through the frozen alias table (``pi-external-u`` -> ``pi``,
  ``wall-v2`` -> ``wall``, ...); anything outside {native, damping, pi,
  wall} is refused (unknown action).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Optional

from .errors import PlanSchemaError, UnknownActionError
from .state import StateSnapshot

#: Canonical per-CAV action codes (ctrl.config.ACTION_CODES; frozen).
ACTION_CODES = {"none": -1, "native": 0, "damping": 1, "pi": 2, "wall": 3}

#: Plan-tensor action token order (features/constants.py ACTIONS).
PLAN_ACTIONS = ("native", "damping", "pi", "wall", "unknown")

#: Schedule-event action aliases -> canonical token (mirrored read-only from
#: evalh.idea_a.ACTION_ALIASES; pinned by a contract test).
ACTION_ALIASES = {
    "native": "native", "damping": "damping", "pi": "pi",
    "pi-external-u": "pi", "fleet-pi": "pi", "fleet-pi-external-u": "pi",
    "wall": "wall", "wall-v2": "wall", "unknown": "unknown",
}

#: The actions a plan may name (the trained CAV alphabet; never "unknown").
PLANNABLE_ACTIONS = ("native", "damping", "pi", "wall")

DEFAULT_ACTION = "native"
SEGMENT_S = 30.0


def canon_action(action) -> str:
    """Canonical plan action token for a schedule event action string."""
    key = str(action).strip().lower()
    return ACTION_ALIASES.get(key, key)


def _check_plannable(token: str, where: str) -> str:
    if token not in PLANNABLE_ACTIONS:
        raise UnknownActionError(
            f"{where}: unknown action {token!r}; the scoring API plans only "
            f"{PLANNABLE_ACTIONS} (aliases "
            f"{sorted(set(ACTION_ALIASES) - set(PLAN_ACTIONS))} are accepted)")
    return token


def _events_tokens(events, n_segments: int, where: str) -> list:
    """Per-segment canonical tokens (latest event <= segment start)."""
    evs = []
    for e in events or ():
        if isinstance(e, Mapping):
            if "time_s" not in e or "action" not in e:
                raise PlanSchemaError(
                    f"{where}: schedule events need 'time_s' and 'action'")
            evs.append((float(e["time_s"]), canon_action(e["action"])))
        else:
            ts, act = e
            evs.append((float(ts), canon_action(act)))
    evs.sort(key=lambda x: x[0])
    tokens = []
    for k in range(n_segments):
        t = SEGMENT_S * k
        act = DEFAULT_ACTION
        for tt, aa in evs:
            if tt <= t + 1e-9:
                act = aa
            else:
                break
        tokens.append(_check_plannable(act, where))
    return tokens


def _static_tokens(token: str, n_segments: int, where: str) -> list:
    return [_check_plannable(canon_action(token), where)] * n_segments


@dataclass
class ResolvedPlan:
    """One validated candidate plan (the planned schedule, ready to roll)."""

    index: int
    n_segments: int
    segment_tokens: list                  # [k][cav_id] -> canonical token
    segment_codes: list                   # [k][cav_id] -> ACTION_CODES int
    wall: Optional[dict] = None           # wall membership + params
    gates: dict = field(default_factory=dict)
    descriptor: object = None             # evalh.idea_a.ActionDescriptor

    @property
    def wall_members(self) -> tuple:
        return tuple(self.wall["members"]) if self.wall else ()

    def describe(self) -> str:
        """Compact human-readable schedule digest (evalh descriptor shape)."""
        d = self.descriptor
        return d.describe() if d is not None else ""


def _wall_block(plan: Mapping, state: StateSnapshot, n_segments: int,
                plan_index: int) -> Optional[dict]:
    wall = plan.get("wall")
    if wall is None:
        return None
    if not isinstance(wall, Mapping):
        raise PlanSchemaError(f"plan {plan_index}: 'wall' must be an object")
    members = wall.get("members")
    if not members:
        raise PlanSchemaError(
            f"plan {plan_index}: wall membership needs a non-empty "
            "'members' list")
    known_cav = {str(state.ids[i]) for i in range(len(state.ids))
                 if bool(state.is_cav[i]) and int(state.lane_phys[i]) >= 0}
    members = [str(m) for m in members]
    if len(set(members)) != len(members):
        raise PlanSchemaError(
            f"plan {plan_index}: duplicate wall member(s) in {members}")
    unknown = [m for m in members if m not in known_cav]
    if unknown:
        raise PlanSchemaError(
            f"plan {plan_index}: wall members not CAVs of the state: "
            f"{unknown}")
    speed = float(wall.get("speed_mps", 0.0) or 0.0)
    if not math.isfinite(speed) or speed <= 0:
        raise PlanSchemaError(
            f"plan {plan_index}: wall speed_mps must be finite and > 0")
    hold = float(wall.get("hold_s", 0.0) or 0.0)
    if not math.isfinite(hold) or hold <= 0:
        raise PlanSchemaError(
            f"plan {plan_index}: wall hold_s must be finite and > 0")
    onset = float(wall.get("onset_s", 0.0) or 0.0)
    return dict(wall_id=int(wall.get("wall_id", plan_index)),
                speed_mps=speed, hold_s=hold, members=members, onset_s=onset)


def resolve_plan(plan, state: StateSnapshot, n_segments: int,
                 plan_index: int = 0) -> ResolvedPlan:
    """Validate one candidate plan against ``state``; resolve its schedule."""
    if isinstance(plan, str):
        plan = {"actions": {}, "default_action": plan}
    if not isinstance(plan, Mapping):
        raise PlanSchemaError(
            f"plan {plan_index}: a candidate plan must be a mapping "
            "({'actions': {...}, 'wall': {...}})")
    actions = plan.get("actions") or {}
    if not isinstance(actions, Mapping):
        raise PlanSchemaError(
            f"plan {plan_index}: 'actions' must be an object keyed by CAV id")
    wall = _wall_block(plan, state, n_segments, plan_index)
    members = set(wall["members"]) if wall else set()
    known_ids = {str(v) for v in state.ids}
    cav_ids = {str(state.ids[i]) for i in range(len(state.ids))
               if bool(state.is_cav[i])}
    seg_tokens = [dict() for _ in range(n_segments)]
    for cid, spec in actions.items():
        cid = str(cid)
        if cid not in known_ids:
            raise PlanSchemaError(
                f"plan {plan_index}: unknown vehicle id {cid!r} (state ids: "
                f"{sorted(known_ids)})")
        if cid not in cav_ids:
            raise PlanSchemaError(
                f"plan {plan_index}: {cid!r} is not a CAV; only CAVs are "
                "plannable (HDVs keep IDM and receive no commands)")
        if cid in members:
            raise PlanSchemaError(
                f"plan {plan_index}: wall member {cid!r} must not also "
                "carry its own schedule")
        where = f"plan {plan_index} cav {cid}"
        if isinstance(spec, Mapping) and "schedule" in spec:
            tokens = _events_tokens(spec["schedule"], n_segments, where)
        elif isinstance(spec, (list, tuple)):
            tokens = _events_tokens(spec, n_segments, where)
        else:
            tokens = _static_tokens(spec, n_segments, where)
        for k in range(n_segments):
            seg_tokens[k][cid] = tokens[k]
    if wall is not None:
        onset_seg = int(round(wall["onset_s"] / SEGMENT_S))
        n_hold = max(1, int(round(wall["hold_s"] / SEGMENT_S)))
        for k in range(n_segments):
            active = onset_seg <= k < onset_seg + n_hold
            for cid in wall["members"]:
                seg_tokens[k][cid] = "wall" if active else DEFAULT_ACTION
    seg_codes = [{cid: ACTION_CODES[tok] for cid, tok in seg.items()}
                 for seg in seg_tokens]
    resolved = ResolvedPlan(
        index=int(plan_index), n_segments=int(n_segments),
        segment_tokens=seg_tokens, segment_codes=seg_codes, wall=wall)
    resolved.gates = wall_gate_records(resolved, state)
    resolved.descriptor = _descriptor_of(resolved)
    return resolved


def _descriptor_of(resolved: ResolvedPlan):
    """The evalh planned-schedule descriptor of this plan (reporting only)."""
    from evalh.idea_a import WALL_VEHICLE_ID, ActionDescriptor

    segments = []
    for cid in sorted({c for seg in resolved.segment_tokens
                       for c in seg}):
        segments.append((cid, tuple(resolved.segment_tokens[k].get(
            cid, DEFAULT_ACTION) for k in range(resolved.n_segments))))
    kw = {}
    if resolved.wall is not None:
        onset_seg = int(round(resolved.wall["onset_s"] / SEGMENT_S))
        n_hold = max(1, int(round(resolved.wall["hold_s"] / SEGMENT_S)))
        wall_segs = tuple(
            "wall" if onset_seg <= k < onset_seg + n_hold else DEFAULT_ACTION
            for k in range(resolved.n_segments))
        segments = [s for s in segments
                    if s[0] != WALL_VEHICLE_ID]
        segments.append((WALL_VEHICLE_ID, wall_segs))
        kw = dict(wall_hold_s=float(resolved.wall["hold_s"]),
                  wall_speed=float(resolved.wall["speed_mps"]),
                  wall_id=int(resolved.wall["wall_id"]))
    return ActionDescriptor(
        segments=tuple(sorted(segments)),
        wall_cap=0.0, **kw)


def wall_gate_records(resolved: ResolvedPlan, state: StateSnapshot) -> dict:
    """A-priori wall-v2.2 proposal-gate records (reported, never tuning).

    Frozen gates (ctrl.wall_v22): the 2-abreast + span pair gate and the
    ``D_min = 70 * v_wall`` reachability gate on the LEAD member, plus the
    a-priori effective hold cap.  A missing/infeasible record means the
    live session would refuse the proposal at the wall request stage.
    """
    from ctrl.wall_v22 import (REQUEST_SPAN_CAP_M, check_pair, dmin_refusal,
                               hold_cap_s)

    wall = resolved.wall
    if wall is None:
        return {}
    pos = {str(state.ids[i]): (int(state.lane_phys[i]),
                               float(state.s_down_m[i]))
           for i in range(len(state.ids))}
    members = [(cid,) + pos[cid] for cid in wall["members"]]
    members.sort(key=lambda r: -r[2])
    merge_start = 2644.2
    lead_s_road = members[0][2] + merge_start
    out = {"lead_id": members[0][0],
           "lead_s_road_m": round(lead_s_road, 6),
           "n_members": len(members)}
    dmin = dmin_refusal(lead_s_road, wall["speed_mps"])
    out["dmin_refusal"] = dmin
    out["hold_cap_s"] = hold_cap_s(wall["hold_s"], lead_s_road,
                                   wall["speed_mps"])
    if len(members) == 2:
        out["pair_gate"] = check_pair([m[1] for m in members],
                                      [m[2] + merge_start for m in members],
                                      wall["speed_mps"])
    else:
        out["pair_gate"] = {"feasible": False,
                            "refusal": {"reason": "pair_size",
                                        "n_members": len(members)},
                            "span_cap_m": REQUEST_SPAN_CAP_M}
    return out
