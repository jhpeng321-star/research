"""CORPUS-side fix for the amendment-2 J4 ``member_left_network`` wall runs.

The frozen v1 featurizer (``features/featurizer.py:287``) passes the per-tick
present-CAV roster into ``partB_v0data.wall_tokens_v04.WallEpisodeSet.walls_at``,
which raises ``WallTokenError: non-CAV member(s)`` whenever a wall member is
missing from that tick's roster while the wall is still visible.  Two real J4
situations hit this at a NON-terminal tick (18 walls-pilot chunk-4 runs):

1. **Road-end latency (15 runs).**  A member reaches the registered road end
   and leaves the network 1-3 physics ticks BEFORE the wall driver notices
   (``safety_abort`` ``member_left_network``, ``original_reason``
   ``member_frame_missing``).  Per the registered amendment-2 disclosure the
   episode is a road-end event; the departed member carries no positions, so
   the member geometry degrades exactly like the existing "member absent at
   t" path while the recorded event timeline (holding -> aborted flash at
   the recorded terminal tick) is kept.
2. **Present-but-unrostered member (3 dense runs).**  The member is still in
   the network and CAV-reported, but the featurizer's fixed 128-slot pool
   dropped it at crowded ticks (>128 concurrent vehicles), so it is missing
   from both ``positions`` and the roster.  Its last reported member state is
   carried (it contributes no new geometry that tick).

A member id that is PRESENT in the physics row and was NEVER CAV-reported
anywhere in the run is a genuine non-CAV member (data bug) and still fails
loudly, as does any missing member of a run without physics evidence.

This module installs an idempotent, corpus-owned monkeypatch on
``WallEpisodeSet`` (the frozen v1 featurizer and the frozen
``wall_tokens_v04`` module are NOT edited): members of a visible wall that
are missing from the roster are tolerated at the wall's terminal tick
(existing behaviour), and at other visible ticks only when the run's
``physics_steps.jsonl`` (mechanics rows only: ``vehicles`` / ``cav_ids``)
shows the member either absent from the network at t (road-end case) or
CAV-reported somewhere in the run (report/slot-gap case).  Every other
active-phase miss still fails loudly.  ``n_members`` keeps the planner-known
count.

Downstream, ``features/idea_a.apply_road_end_relabel`` (amendment 2) records
the ``member_left_network -> road_end`` relabel and
``corpus/idea_a.write_idea_a_channels`` stores the relabel flag and the
aborted-token count in ``IDEA_A_DONE.json``; run receipts keep their
original labels.
"""
from __future__ import annotations

import json
from bisect import bisect_left
from pathlib import Path

_TICK_TOL = 1e-6

_PATCHED = False


def _load_member_facts(run_dir, members: tuple) -> dict | None:
    """Per-member physics facts from one streaming pass (members only).

    Returns ``{vid: (times_in_network_sorted, cav_ever)}`` or ``None`` when
    no physics file exists.  Reads ONLY the mechanics row fields
    ``time_s`` / ``vehicles`` (keys) / ``cav_ids``.
    """
    root = Path(run_dir)
    path = None
    for cand in (root / "run" / "physics_steps.jsonl",
                 root / "physics_steps.jsonl"):
        if cand.is_file():
            path = cand
            break
    if path is None:
        return None
    want = {str(m) for m in members}
    times: dict[str, list] = {m: [] for m in want}
    cav_ever: dict[str, bool] = {m: False for m in want}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            t = float(r["time_s"])
            vehs = r.get("vehicles") or {}
            cav = r.get("cav_ids") or ()
            for m in want:
                if m in vehs:
                    times[m].append(t)
                if not cav_ever[m] and m in cav:
                    cav_ever[m] = True
    return {m: (times[m], cav_ever[m]) for m in want}


def _in_network(fact, t: float) -> bool:
    times = fact[0]
    i = bisect_left(times, t - _TICK_TOL)
    return i < len(times) and abs(times[i] - t) <= _TICK_TOL


def install() -> bool:
    """Patch ``WallEpisodeSet`` for J4 road-end / report-gap members.

    Idempotent: returns True iff this call installed the patch.
    """
    global _PATCHED
    if _PATCHED:
        return False
    from features.actions_adapter import WallEpisodeSet  # partB_v0data module
    import wall_tokens_v04 as WT

    orig_at = WallEpisodeSet.walls_at
    orig_from = WallEpisodeSet.from_branch_dir.__func__

    def from_branch_dir(cls, run_dir):
        eps = orig_from(cls, run_dir)
        try:
            eps._j4_run_dir = run_dir
        except AttributeError:                 # pragma: no cover
            pass
        return eps

    def _tolerated(self, t: float, roster: set) -> set:
        """Members of walls visible at t to add to the roster for this call."""
        eps = getattr(self, "_eps", {})
        visible = [ep for ep in eps.values()
                   if ep.request is not None
                   and float(ep.request["timestamp_s"]) <= t + WT.EPS
                   and (ep.terminal is None
                        or t <= float(ep.terminal[1]) + WT.EPS)]
        missing = {str(m) for ep in visible for m in ep.members()
                   if str(m) not in roster}
        if not missing:
            return set()
        # terminal ticks tolerate unconditionally (existing behaviour)
        term_now = any(
            ep.terminal is not None and abs(t - float(ep.terminal[1]))
            <= WT.EPS for ep in visible)
        if term_now:
            return missing
        facts = getattr(self, "_j4_facts", ...)
        if facts is ...:
            members = tuple({m for ep in eps.values() for m in ep.members()})
            run_dir = getattr(self, "_j4_run_dir", None)
            facts = (None if run_dir is None
                     else _load_member_facts(run_dir, members))
            try:
                self._j4_facts = facts
            except AttributeError:             # pragma: no cover
                pass
        if facts is None:
            return set()                       # no evidence: stay loud
        tol: set = set()
        for ep in visible:
            for m in ep.members():
                m = str(m)
                if m in roster or m in tol:
                    continue
                fact = facts.get(m)
                if fact is None:
                    continue                   # unknown id: stay loud
                if not _in_network(fact, t):
                    tol.add(m)                 # left the network (road end)
                elif fact[1]:
                    tol.add(m)                 # CAV report/slot gap: carry
                # present, never CAV-reported: genuine non-CAV -> loud
        return tol

    def walls_at(self, t, positions=None, cav_roster=None):
        if cav_roster is None:
            return orig_at(self, t, positions=positions, cav_roster=None)
        roster = {str(v) for v in cav_roster}
        tol = _tolerated(self, float(t), roster)
        if not tol:
            return orig_at(self, t, positions=positions, cav_roster=roster)
        return orig_at(self, t, positions=positions,
                       cav_roster=roster | tol)

    walls_at.__doc__ = orig_at.__doc__
    walls_at.__name__ = orig_at.__name__
    WallEpisodeSet.walls_at = walls_at
    WallEpisodeSet.from_branch_dir = classmethod(from_branch_dir)
    _PATCHED = True
    return True


def installed() -> bool:
    return _PATCHED
