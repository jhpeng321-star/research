"""Importable LCMOVES lane-change + speed families for other agents (REGIME), zero-SUMO at import time.

The families and the executor are the exact S-LCMOVES code (lcm_plan builders, lcm_core.Arm 'prog' role);
this module only wraps them.  CAV-only: the caller must pass only CAV ids in the snapshot; Executor asserts it.

    import sys; sys.path.insert(0, "<...>/runs/active/C4-V7FT-20261017/lcmoves/code")
    import lcm_families as F
    snap = F.snapshot(api, cav_ids)                       # libsumo/traci module, at a whole second
    members = F.propose(snap, "lane_redistribute")        # None if the family is infeasible now
    ex = F.Executor(api, members, cav_ids)                # then once per whole second (el = 0, 1, 2, ...):
    ex.step(el)                                           # before simulationStep(); ex.active() -> False when done
    ex.release_all(el)                                    # optional early stop (restores lcMode / tau / speed)

Families (FAMILIES): wall_fill, rolling_wall, gom_1, gom_multi (gap_open_merge), lane_redistribute,
free_program (seeded, needs sid/i), r_program (the matched random program; seeded, needs sid/i).
Snapshot format: dict(cavs=[{id, edge, lane, x, v}], veh_up=[[lane, x, v], ...]) with x = metres from the start of
edge 'up' (wz3to2; drop at X_B = 2696).  Only lane_redistribute reads veh_up (local lane counts within 100 m).
Control conventions as in the probe: 1 s control, slowDown(v + a, 1 s) with a in [-dmax, amax 1.0], CAV
speedMode 27 (set by the caller), lane requests changeLane(target, 2 s) every second while active (persistent hold),
lcMode 513 while commanded, everything restored on release.
"""
from __future__ import annotations

import lcm_core as C
import lcm_plan as P

X_B = P.X_B
FAMILIES = {
    "wall_fill": P.cands_wall_fill,
    "rolling_wall": P.cands_rolling_wall,
    "gom_1": P.cands_gom_1,
    "gom_multi": P.cands_gom_multi,
    "lane_redistribute": P.cands_lane_redistribute,
}
SAMPLED = ("free_program", "r_program")


def snapshot(api, cav_ids):
    """Geometry snapshot (CAV rows + all 'up' vehicles as [lane, x, v]) from a running libsumo/traci simulation."""
    s = C.snapshot(api, frozenset(cav_ids))
    return dict(cavs=s["cavs"], veh_up=s["veh_up"], approach_speed=s["approach_speed"])


def feasible(snap):
    """{family: bool} for the deterministic families at this snapshot."""
    return {k: bool(fn(snap)) for k, fn in FAMILIES.items()}


def propose(snap, family, sid="regime", i=0, exclude_sets=None):
    """Member specs of the best-geometry move of `family` (or the i-th seeded sampled program), else None."""
    if family in FAMILIES:
        return P._best(FAMILIES[family](snap))
    if family == "free_program":
        progs = P.free_programs(snap, sid, n=i + 1)
        return progs[i][1] if len(progs) > i else None
    if family == "r_program":
        progs = P.r_programs(snap, sid, i + 1, set(exclude_sets or ()))
        return progs[i][1] if len(progs) > i else None
    raise KeyError(family)


class Executor:
    """Runs one move's member programs with lcm_core.Arm (identical semantics to the S-LCMOVES branches)."""

    def __init__(self, api, members, cav_ids):
        for m in members:
            assert m["id"] in cav_ids, ("non-CAV member", m["id"])   # HDVs are never commanded
        self.api = api
        self.arm = C.Arm(api, members)
        ids = [m["id"] for m in members]
        for m in members:
            p = m["params"]
            ids += list(p.get("track_ref") or [])
            for k in ("anchor", "partner", "release_on_partner_merge"):
                if p.get(k):
                    ids.append(p[k])
        self.track = list(dict.fromkeys(ids))

    def step(self, el):
        live = set(self.api.vehicle.getIDList())
        pos = C._positions(self.api, self.track, live)
        self.arm.control(int(el), live, pos)

    def active(self):
        return any(not m.get("released") for m in self.arm.members)

    def release_all(self, el, why="external"):
        live = set(self.api.vehicle.getIDList())
        for m in self.arm.members:
            if not m.get("released") and m["id"] in live:
                self.arm._release(m, int(el), why)
            m["released"] = True

    @property
    def log(self):
        return dict(issued=self.arm.issued, commands=self.arm.commands, events=self.arm.events,
                    lc_done={m["id"]: m.get("lc_done") for m in self.arm.members},
                    release_el={m["id"]: m.get("release_el") for m in self.arm.members})
