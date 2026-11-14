"""Part B v0.4 wall tokens (BRIEF_PARTB_WALL_ACTION_INPUTS_DRAFT.md item 2).

Per-wall planner state at decision time t, reconstructed ONLY from the wall
pilot's per-branch ``wall_events.jsonl`` (schema:
campaign_walls_v1/wall_driver.py -- kinds wall_request / wall_event
{assembly_start | assembled | released | safety_abort} / not_feasible /
wall_handback) plus the request-time fields of ``branch_record.json``.

Phase vocabulary (frozen with WALL_RAW): requested, assembling, holding,
released, aborted.  A wall token exists from its request tick through its
terminal tick (released/aborted) and is dropped afterwards; a
``not_feasible`` arm never produces a wall.

LEAKAGE RULE: every query uses only events with ``timestamp_s <= t``.
Outcome fields of branch_record.json (outcome, assembled_at_s, released_at_s,
aborted_at_s, abort_reason) are NEVER read here -- they describe the episode
END.  Shuffling all events after t therefore cannot change any feature at t
(asserted by test).

Member geometry (mean s_down, lane span) comes only from the MEMBER CAVs'
self-report rows at t (the positions dict passed by the caller, restricted
to the request's member list); with a roster given, a non-CAV member is a
loud contract error -- no HDV-derived wall feature exists.

Planner-owned field: ``planned_hold_remaining`` is the planner's own
decision state (it knows hold_s and the assembly completion it caused), so
it is allowed as input and flagged as PLANNER_OWNED here and in the field
table.  Pure offline reconstruction; zero simulation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# wall phase vocabulary (token slot order -- frozen with WALL_RAW)
WALL_PHASES = ("requested", "assembling", "holding", "released", "aborted")
# phases in which members are locked to the wall (before handback)
ACTIVE_WALL_PHASES = ("requested", "assembling", "holding")

# wall token field table (order == part4/partB/featurize.py WALL layout)
WALL_TOKEN_FIELDS = (
    "phase_onehot5",                # requested/assembling/holding/released/
                                    #   aborted
    "wall_speed_mps/40",            # planner-chosen commanded speed
    "time_in_phase_s/600",
    "planned_hold_remaining_s/600", # PLANNER_OWNED state (flagged; allowed)
    "n_members/4",
    "members_mean_s_down_m/1000",   # member CAV self-reports at t only
    "members_lane_span/2",          # max(lane)-min(lane) of present members
)

# branch_record.json fields this module may read: request-time planner
# state ONLY.  Everything else (the outcome block) is forbidden.
ALLOWED_RECORD_FIELDS = frozenset({
    "wall_id", "members", "wall_speed_mps", "hold_s", "request_s", "timing",
    "speed_factor"})
FORBIDDEN_RECORD_FIELDS = frozenset({
    "outcome", "assembled_at_s", "released_at_s", "aborted_at_s",
    "abort_reason"})

EPS = 1e-9


class WallTokenError(ValueError):
    """A wall-event contract violation; the build fails loudly."""


def _load_events(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


class _WallEp:
    """One wall's event stream, sorted by timestamp."""

    def __init__(self, wall_id: str):
        self.wall_id = wall_id
        self.request: dict | None = None
        self.assembly_start: float | None = None
        self.assembled: float | None = None
        self.terminal: tuple[str, float] | None = None   # (phase, ts)
        self.not_feasible: float | None = None

    def observe(self, ev: dict) -> None:
        kind = ev.get("kind")
        ts = float(ev["timestamp_s"])
        if kind == "wall_request":
            if self.request is not None:
                raise WallTokenError(
                    f"wall {self.wall_id}: duplicate request")
            self.request = ev
        elif kind == "wall_event":
            e = ev.get("event")
            if e == "assembly_start":
                self.assembly_start = ts
            elif e == "assembled":
                self.assembled = ts
            elif e in ("released", "safety_abort"):
                if self.terminal is not None:
                    raise WallTokenError(
                        f"wall {self.wall_id}: finished twice")
                self.terminal = ("released" if e == "released"
                                 else "aborted", ts)
            else:
                raise WallTokenError(
                    f"wall {self.wall_id}: unknown wall_event {e!r}")
        elif kind == "not_feasible":
            self.not_feasible = ts
        # wall_handback rows are derived duplicates of the terminal event:
        # never double-counted here

    def members(self) -> tuple:
        if self.request is not None and self.request.get("members"):
            return tuple(self.request["members"])
        return ()

    def speed_mps(self) -> float:
        if self.request is not None:
            return float(self.request.get("wall_speed_mps") or 0.0)
        return 0.0

    def hold_s(self) -> float:
        if self.request is not None:
            return float(self.request.get("hold_s") or 0.0)
        return 0.0


class WallEpisodeSet:
    """All wall episodes of one branch, queryable at any decision time."""

    def __init__(self, episodes: dict[str, _WallEp],
                 record_request_fields: dict):
        self._eps = episodes
        self._record = record_request_fields

    # ------------------------------------------------------------ build --
    @classmethod
    def from_branch_dir(cls, run_dir) -> "WallEpisodeSet":
        """Read wall_events.jsonl (+ branch_record.json request fields)
        from an arm directory (root or root/run)."""
        root = Path(run_dir)
        ev_path = None
        for cand in (root / "wall_events.jsonl", root / "run" /
                     "wall_events.jsonl"):
            if cand.is_file():
                ev_path = cand
                break
        eps: dict[str, _WallEp] = {}
        if ev_path is None:
            rec: dict = {}
        else:
            for ev in _load_events(ev_path):
                wid = ev.get("wall_id")
                if wid is None:
                    raise WallTokenError(f"{ev_path}: event without "
                                         f"wall_id: kind={ev.get('kind')!r}")
                eps.setdefault(str(wid), _WallEp(str(wid))).observe(ev)
            rec = {}
        rp = None
        for cand in (root / "branch_record.json", root / "run" /
                     "branch_record.json"):
            if cand.is_file():
                rp = cand
                break
        if rp is not None:
            raw = json.loads(rp.read_text(encoding="utf-8"))
            if raw.get("wall_id") is not None:
                # request-time planner fields ONLY; the outcome block is
                # never bound into this module's state
                rec = {k: raw[k] for k in ALLOWED_RECORD_FIELDS
                       if k in raw}
        return cls(eps, rec)

    # ------------------------------------------------------------ query --
    def walls_at(self, t: float, positions: dict | None = None,
                 cav_roster=None) -> list[dict]:
        """Planner-known wall states at t (events <= t only).

        positions: {vid: (s_down_m, lane)} from the member CAVs'
        self-report rows at t (only member vids are consulted).
        cav_roster: designated-CAV vid set; every member must be in it
        (loud error otherwise -- wall members are CAVs by construction).
        """
        t = float(t)
        out = []
        for wid, ep in sorted(self._eps.items()):
            st = self._state_at(ep, t)
            if st is None:
                continue
            members = ep.members()
            if cav_roster is not None:
                hdv = [m for m in members if str(m) not in
                       {str(v) for v in cav_roster}]
                if hdv:
                    raise WallTokenError(
                        f"wall {wid}: non-CAV member(s) {hdv} -- wall "
                        "features are CAV-self-report only (no HDV input)")
            s_vals, lanes = [], []
            for m in members:
                p = (positions or {}).get(str(m))
                if p is None:
                    continue                     # member absent at t
                s, ln = float(p[0]), int(p[1])
                s_vals.append(s)
                if ln >= 0:
                    lanes.append(ln)
            mean_s = float(np.mean(s_vals)) if s_vals else 0.0
            span = float(max(lanes) - min(lanes)) \
                if len(set(lanes)) >= 2 else 0.0
            out.append(dict(st, wall_id=wid, members=members,
                            n_members=len(members), mean_s_down_m=mean_s,
                            lane_span=span))
        return out

    def _state_at(self, ep: _WallEp, t: float) -> dict | None:
        """Phase/speed/timing state from events with ts <= t; None once the
        wall is gone (not_feasible arms never exist; terminal phases show
        only at their own tick, then the wall is dropped)."""
        if ep.not_feasible is not None and ep.request is None:
            return None
        if ep.request is None or ep.request["timestamp_s"] > t + EPS:
            return None                          # not requested yet at t
        if ep.terminal is not None:
            phase, ts = ep.terminal
            if t > ts + EPS:
                return None                      # wall handed back: dropped
            if t >= ts - EPS:                    # terminal phase: own tick
                return dict(phase=phase, speed_mps=ep.speed_mps(),
                            time_in_phase_s=0.0,
                            planned_hold_remaining_s=0.0)
        if ep.assembled is not None and t >= ep.assembled - EPS:
            phase = "holding"
            hold = ep.hold_s()
            start = ep.assembled
            hold_rem = max(0.0, hold - (t - ep.assembled))
        elif ep.assembly_start is not None and \
                t >= ep.assembly_start - EPS:
            phase = "assembling"
            start = ep.assembly_start
            hold_rem = ep.hold_s()               # full planned hold remains
        else:
            phase = "requested"
            start = float(ep.request["timestamp_s"])
            hold_rem = ep.hold_s()
        return dict(
            phase=phase, speed_mps=ep.speed_mps(),
            time_in_phase_s=max(0.0, t - start),
            planned_hold_remaining_s=hold_rem)

    def member_flags(self, vids, t: float) -> dict:
        """{vid: wall-state dict} for the CAVs locked by a wall active at
        t (requested/assembling/holding).  Released/aborted members are
        handed back: they are NOT flagged at t > terminal tick."""
        active = [w for w in self.walls_at(t)
                  if w["phase"] in ("requested", "assembling", "holding")]
        out = {}
        for w in active:
            for m in w["members"]:
                out[str(m)] = w
        return out

    def __len__(self) -> int:
        return len(self._eps)
