"""WD-01-PA2G-CNT1 entrance-counter arrivals injector (registered
a-priori mechanism, REGISTRATION_PA2G_CNT1.md section 3; NOT tuned).

Two pieces:

- the COUNTER READER: truth-proxy entrance counts from the physics
  rows' ``departed_ids``, classified by each vehicle's first physical
  row edge (mainline entry edge 1191554636 -> "main", ramp entry edge
  11553048#1 -> "ramp"; both verified against the preserved network:
  two-lane motorway start / single-lane 378.66 m on-ramp, neither has a
  feeder edge).  Counts for a window start w use ONLY departures in
  (w - W, w] with W = 60 s -- departures after w are filtered out
  BEFORE counting, and the sentinel test proves the output is
  invariant to them (the "no count after w" registered test).

- the INJECTOR: a driver wrapped around a PartAV2GEngine that inserts
  generic HDVs at the entrance edges' model-world s = 0 following the
  registered a-priori schedule.  No vType, no CAV flag, no penetration
  and no future schedule is ever read (the engine is vType-free by
  construction).  The engine's ``insert_vehicle`` hook is the ONLY
  entry point; with the injector absent the nominal path is untouched
  (byte-identity tests).

Registered mechanism (all values a priori):
- rate lambda_e = count_e(w - 60, w] / 60 per entrance, held constant
  over (w, w + H];
- deterministic evenly spaced schedule: vehicle k of entrance e is due
  at w + (k - 0.5)/lambda_e while due <= w + H;
- placement at s = 0 of the entrance edge; mainline alternates model
  lanes 0/1 round-robin STARTING WITH LANE 1 (right); ramp uses its
  single lane (model lane 2);
- insertion speed v_ins = min(lane speed limit at s = 0, speed of the
  nearest leader in the target lane);
- insertion condition: IDM safe gap gap >= s0 + v_ins * T with the
  nominal s0 = 2 m, T = 1.5 s (read from params, not hard-coded);
  otherwise the vehicle waits in a FIFO virtual entrance queue and is
  retried each step (strict FIFO: a blocked head blocks its queue).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .engine import PartAV2GEngine, _seg_limit1

MAIN_ENTRY_EDGE = "1191554636"   # mainline entry boundary (network start)
RAMP_ENTRY_EDGE = "11553048#1"   # ramp entry boundary (single lane)
ENTRANCES = ("main", "ramp")
W_COUNTER_S = 60.0

EDGE_TO_ENTRANCE = {
    MAIN_ENTRY_EDGE: "main",
    RAMP_ENTRY_EDGE: "ramp",
}


# ---------------------------------------------------------- counter reader

def classify_departure(road_id: str) -> Optional[str]:
    """Entrance of a departed vehicle from its first physical row edge;
    None for any edge that is not a registered entrance (the runner
    asserts there are none)."""
    return EDGE_TO_ENTRANCE.get(str(road_id).strip())


def read_departures(physics_path: Path) -> dict:
    """One streaming pass over a truth physics log: departures as
    (time_s, entrance) pairs plus cross-check counters.

    Returns dict with:
      departures: sorted [(t, entrance)] for every departed vehicle
      other_edge: departures on an unregistered edge (must stay empty)
      repeat_ids: vehicles listed twice in departed_ids (must stay empty)
      bc_main / bc_ramp: summed boundary_crossings (sanity cross-check)
    """
    deps: List[Tuple[float, str]] = []
    seen = set()
    other_edge = 0
    repeat_ids = 0
    bc_main = bc_ramp = 0
    with open(physics_path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            t = float(row["time_s"])
            bc = row.get("boundary_crossings") or {}
            bc_main += int(bc.get("mainline_entry", 0))
            bc_ramp += int(bc.get("ramp_entry", 0))
            for vid in row.get("departed_ids") or []:
                if vid in seen:
                    repeat_ids += 1
                    continue
                seen.add(vid)
                veh = (row.get("vehicles") or {}).get(vid) or {}
                ent = classify_departure(veh.get("road_id", ""))
                if ent is None:
                    other_edge += 1
                    continue
                deps.append((t, ent))
    deps.sort()
    return dict(departures=deps, other_edge=other_edge,
                repeat_ids=repeat_ids, bc_main=bc_main, bc_ramp=bc_ramp)


def counts_from_departures(departures: Sequence[Tuple[float, str]],
                           w: float, W: float = W_COUNTER_S) -> Dict[str, int]:
    """Entrance counts over the half-open (w - W, w].

    FUTURE-READ GUARD: departures with time > w are excluded BEFORE any
    counting (the registered "no count after w" test feeds sentinel
    future rows and asserts the output does not move).
    """
    out = {e: 0 for e in ENTRANCES}
    for t, ent in departures:
        if t > w + 1e-9:            # after w: never read
            continue
        if w - W + 1e-9 < t <= w + 1e-9:
            out[ent] += 1
    return out


# --------------------------------------------------------------- schedule

def insertion_schedule(counts: Mapping[str, int], w: float, H: float,
                       W: float = W_COUNTER_S) -> List[dict]:
    """The registered deterministic schedule: for entrance e with
    lambda_e = count_e / W, vehicle k is due at w + (k - 0.5)/lambda_e
    while the due time <= w + H.  Returns the due list sorted by
    (due, entrance, k) -- mainline first on exact ties."""
    sched: List[dict] = []
    for e in ENTRANCES:
        c = int(counts.get(e, 0))
        if c <= 0:
            continue
        lam = c / float(W)
        k = 1
        while True:
            due = w + (k - 0.5) / lam
            if due > w + H + 1e-9:
                break
            sched.append(dict(entrance=e, k=k, due=due))
            k += 1
    sched.sort(key=lambda d: (d["due"], ENTRANCES.index(d["entrance"]),
                              d["k"]))
    return sched


def lane_for(entrance: str, k: int) -> int:
    """Registered placement: mainline alternates model lanes 0/1
    round-robin starting with lane 1 (right); the ramp uses its single
    lane (model lane 2)."""
    if entrance == "main":
        return 1 if k % 2 == 1 else 0
    if entrance == "ramp":
        return 2
    raise ValueError(f"unknown entrance {entrance!r}")


def injected_vid(entrance: str, k: int) -> str:
    """Stable injected-vehicle identity (labelled separately in raw
    rows; never collides with truth ids)."""
    return ("cntM" if entrance == "main" else "cntR") + f"{k:04d}"


# --------------------------------------------------------------- injector

class ArrivalInjector:
    """Drives a PartAV2GEngine with the registered arrivals (engine hook
    user).  ``run(n_steps)`` steps the engine itself so the empty-engine
    shortcut in ``run()`` cannot kill pending insertions.

    Queue-delay definition (identical on both sides, stated in the
    report): a vehicle accrues delay only from its FIRST PHYSICAL step,
    i.e. once inserted; waiting in the virtual entrance queue accrues
    none -- the truth counts delay from the physics rows the same way
    (a SUMO vehicle waiting at insertion does not exist in the physics
    log until inserted)."""

    def __init__(self, engine: PartAV2GEngine, schedule: Sequence[dict]):
        self.engine = engine
        self.t0 = float(engine.t0)
        self.dt = float(engine.dt)
        self.schedule = list(schedule)
        self._queues: Dict[str, List[dict]] = {e: [] for e in ENTRANCES}
        self.inserted: List[dict] = []       # {vid, entrance, k, due, step}
        self.blocked_steps = 0
        self._next = 0                       # cursor into schedule

    def _due_step(self, due: float) -> int:
        """First snapshot index j whose time t0 + j*dt >= due (the step
        the vehicle must be inserted before)."""
        return max(1, int(math.ceil((due - self.t0) / self.dt - 1e-9)))

    def _pump(self, j: int) -> None:
        """Move due schedule entries into the FIFO queues, then retry
        each queue HEAD (strict FIFO)."""
        while self._next < len(self.schedule) and \
                self._due_step(self.schedule[self._next]["due"]) <= j:
            item = self.schedule[self._next]
            self._queues[item["entrance"]].append(item)
            self._next += 1
        for e in ENTRANCES:
            q = self._queues[e]
            if q and self._try_insert(q[0], j):
                q.pop(0)

    def _try_insert(self, item: dict, j: int) -> bool:
        """Registered insertion rule at s = 0 of the entrance lane:
        v_ins = min(lane limit, nearest-leader speed); require the IDM
        safe gap gap >= s0 + v_ins * T; else stay queued."""
        eng = self.engine
        lane = lane_for(item["entrance"], item["k"])
        leader = -1
        for r in range(len(eng.names)):
            if eng.lane[r] == lane and \
                    (leader < 0 or eng.s[r] < eng.s[leader]):
                leader = r
        limit = _seg_limit1(lane, 0.0, eng.p)
        if leader < 0:
            v_ins = limit
        else:
            v_ins = min(limit, float(eng.v[leader]))
            gap = float(eng.s[leader]) - float(eng.length[leader])
            s0 = float(eng.idm_p["s0_m"])
            T = float(eng.idm_p["T_s"])
            if gap < s0 + v_ins * T:
                self.blocked_steps += 1
                return False
        vid = injected_vid(item["entrance"], item["k"])
        eng.insert_vehicle(vid, lane, 0.0, v_ins)
        self.inserted.append(dict(vid=vid, entrance=item["entrance"],
                                  k=item["k"], due=item["due"],
                                  step=j, lane=lane, v_ins=v_ins))
        return True

    def run(self, n_steps: int) -> "ArrivalInjector":
        eng = self.engine
        for j in range(eng.step_i + 1, n_steps + 1):
            self._pump(j)
            eng.step()
        return self

    def summary(self, n_steps: int) -> dict:
        """Bookkeeping for the raw rows: inserted within the window,
        still queued at w + H, per-entrance counts."""
        by_e = {e: 0 for e in ENTRANCES}
        for rec in self.inserted:
            if rec["step"] <= n_steps:
                by_e[rec["entrance"]] += 1
        queued = {e: len(q) for e, q in self._queues.items()}
        return dict(inserted_total=len(self.inserted),
                    inserted=by_e, still_queued=queued,
                    blocked_steps=self.blocked_steps)
