"""Total-delay accounting, WD-01 part 4 (PREDICT_CONTRACT_v1.md §4.1).

Pure numpy.  One convention set, used identically for truth logs
(delay.truth), for the contract trajectory layout (delay.contract_io) and
by future Part B label building:

Time discretisation (interval-ending convention)
  A record at tick time t_k certifies the vehicle occupied the section
  during the physics step (t_k - dt, t_k]; a counted record therefore
  contributes dt, truncated to the window at its right edge t0+horizon.
  A vehicle present at t0 contributes records at ticks in (t0, t0+H]
  (the t0 record is the anchor, not an interval -- the pre-t0 step is
  outside the window).  A vehicle inserted inside the window has its
  insertion record (first appearance) excluded as well: insertion is a
  point event, so its on-section time is (t_last - t_insertion) exactly,
  the same quantity the pair-sum of advances covers.  Residual boundary
  error is bounded by one physics step (0.5 s) and is zero for vehicles
  present at t0 and for insertees still on the section at the horizon.

Free-flow time
  Integrated per physics step as advance_m / free-flow speed of the step,
  where advance is the longitudinal advance along the vehicle's PATH:
  within a physical lane |ds|; across a lane change the within-edge
  position difference (chain-offset jumps at the merge edge removed);
  across a junction bridge the shared boundary s makes ds exact.  The
  junction internal length is zero in the physical coordinate (topology
  convention), so junction steps contribute ~0 advance and ~0 free-flow
  time.  Contract default speed = SUMO lane speed limit (free_flow_ref,
  vType-agnostic); oracle variant = limit x vType speedFactor
  (free_flow_oracle, diagnostic only).

Entry waiting
  From the SCHEDULED departure (requests.rou.xml of the regenerated run,
  or the contract's schedule) matched by vehicle id to the ACTUAL
  insertion record: wait = clip(min(t_ins, t0+H) - max(t_dep, t0), 0, inf).
  Vehicles inserted at or before t0 contribute 0 (their wait predates the
  window); vehicles still uninserted at t0+H contribute the full
  truncated wait (queued at the entrance past the horizon).  Insertees
  with no schedule entry are counted unmatched and contribute 0 waiting.

Totals (vehicle-seconds)
  total_delay_ref  = (time_on_section - free_flow_ref)  + wait_main + wait_ramp
  total_delay_oracle = (time_on_section - free_flow_oracle) + wait_main + wait_ramp
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import (ENTRANCE_MAINLINE, ENTRANCE_NONE, ENTRANCE_RAMP,
                       JUNCTION, SectionGeometry)

EPS = 1e-9


@dataclass
class VehicleTrace:
    """One vehicle's records inside a window, ascending in t (defensively
    truncated by the caller or by vehicle_delay)."""
    vid: str
    t: np.ndarray                 # (n,) float64, on the physics grid
    lane: np.ndarray              # (n,) int; JUNCTION = -1 allowed
    s: np.ndarray                 # (n,) float64, physical-lane s
    vehicle_type: int = -1        # vType label (oracle variant); -1 unknown

    def __post_init__(self):
        n = len(self.t)
        assert len(self.lane) == n and len(self.s) == n
        if n > 1:
            assert np.all(np.diff(self.t) > 0), "records must ascend in t"


@dataclass
class VehicleDelay:
    """Per-vehicle delay components inside one window (diagnostics; the
    contract ships totals only)."""
    vid: str
    present_at_t0: bool
    inserted_in_window: bool
    time_on_section_s: float
    free_flow_ref_s: float
    free_flow_oracle_s: float
    traversed_m: float
    n_records: int
    n_junction_records: int
    n_boundary_change_steps: int = 0
    gapped: bool = False


@dataclass
class WaitRecord:
    vid: str
    entrance: int                       # ENTRANCE_MAINLINE / ENTRANCE_RAMP
    t_dep: float
    t_ins: float                        # np.inf if never inserted by horizon
    wait_s: float
    matched: bool                       # False: insertee absent from schedule
    queued_at_horizon: bool = False     # scheduled but not inserted by t0+H


@dataclass
class WindowDelay:
    t0: float
    horizon_s: float
    vehicles: list = field(default_factory=list)   # VehicleDelay
    waits: list = field(default_factory=list)      # WaitRecord
    diagnostics: dict = field(default_factory=dict)

    def totals(self) -> dict:
        ton = sum(v.time_on_section_s for v in self.vehicles)
        ffr = sum(v.free_flow_ref_s for v in self.vehicles)
        ffo = sum(v.free_flow_oracle_s for v in self.vehicles)
        trav = sum(v.traversed_m for v in self.vehicles)
        w_main = sum(w.wait_s for w in self.waits
                     if w.entrance == ENTRANCE_MAINLINE)
        w_ramp = sum(w.wait_s for w in self.waits
                     if w.entrance == ENTRANCE_RAMP)
        t = dict(
            t0=self.t0, horizon_s=self.horizon_s,
            n_vehicles=len(self.vehicles),
            n_waiting_only=sum(1 for w in self.waits if w.wait_s > EPS),
            n_queued_at_horizon=sum(1 for w in self.waits
                                    if w.queued_at_horizon),
            time_on_section_s=ton, free_flow_ref_s=ffr,
            free_flow_oracle_s=ffo, traversed_m=trav,
            travel_delay_ref_s=ton - ffr,
            travel_delay_oracle_s=ton - ffo,
            wait_mainline_s=w_main, wait_ramp_s=w_ramp,
            total_delay_ref_s=ton - ffr + w_main + w_ramp,
            total_delay_oracle_s=ton - ffo + w_main + w_ramp,
        )
        t.update(self.diagnostics)
        return t


# ------------------------------------------------------------ vehicles ----

def vehicle_delay(trace: VehicleTrace, t0: float, horizon_s: float,
                  geom: SectionGeometry, dt: float,
                  speed_factors: dict | None = None,
                  strict: bool = True) -> VehicleDelay:
    """Per-vehicle (time on section, free-flow time, traversed length).

    `trace` may contain records outside [t0, t0+H]; everything outside is
    truncated here.  `speed_factors` maps vType label -> speedFactor for
    the oracle variant (default factor 1.0 for unknown labels).
    """
    t0, t1 = float(t0), float(t0) + float(horizon_s)
    keep = (trace.t >= t0 - EPS) & (trace.t <= t1 + EPS)
    t = np.asarray(trace.t, dtype=np.float64)[keep]
    lane = np.asarray(trace.lane, dtype=np.int64)[keep]
    s = np.asarray(trace.s, dtype=np.float64)[keep]
    n = len(t)
    if n == 0:
        raise ValueError(f"vehicle {trace.vid!r} has no records in window")

    present_at_t0 = abs(t[0] - t0) <= EPS
    inserted_in_window = not present_at_t0
    # The first kept record is a POINT (the t0 anchor, or the insertion
    # event), not an interval; every later record certifies one step.
    # Both cases drop exactly the first record: n-1 intervals.
    counted = n - 1
    ton = counted * dt

    ffr = ffo = adv_sum = 0.0
    n_junction = int((lane == JUNCTION).sum())
    n_boundary_change = 0
    gapped = bool(np.any(np.diff(t) > dt + EPS))
    sf = 1.0
    if speed_factors is not None and trace.vehicle_type in speed_factors:
        sf = float(speed_factors[trace.vehicle_type])
    v_cap = 1.5 * dt * 45.0        # physically plausible advance cap
    for i in range(n - 1):
        adv, attr_lane, attr_s, kind = _pair_advance(
            geom, int(lane[i]), s[i], int(lane[i + 1]), s[i + 1])
        if kind == "boundary_change":
            # lane change + chain-boundary crossing inside one step.  The
            # advance is the chain-continuity decomposition (to the end of
            # a's segment, then from the start of b's segment); a change
            # completed inside a's own edge instead telescopes to the raw
            # ds, so the residual error is bounded by the chain-offset
            # difference (1.45 m on WM-30 ~ 0.05 s of free-flow time).
            n_boundary_change += 1
            if strict and not (-EPS <= adv <= v_cap):
                raise ValueError(
                    f"vehicle {trace.vid!r}: implausible boundary step "
                    f"{lane[i]}@{s[i]:.2f} -> {lane[i+1]}@{s[i+1]:.2f} "
                    f"(advance {adv:.2f} m)")
        vlim = (geom.default_speed if attr_lane is None
                else geom.speed_at(attr_lane, attr_s))
        ffr += adv / vlim
        ffo += adv / (vlim * sf)
        adv_sum += adv
    return VehicleDelay(
        vid=trace.vid, present_at_t0=present_at_t0,
        inserted_in_window=inserted_in_window,
        time_on_section_s=ton, free_flow_ref_s=ffr, free_flow_oracle_s=ffo,
        traversed_m=adv_sum, n_records=n, n_junction_records=n_junction,
        n_boundary_change_steps=n_boundary_change, gapped=gapped)


def _pair_advance(geom: SectionGeometry, lane_a: int, s_a: float,
                  lane_b: int, s_b: float):
    """(advance_m, attr_lane|None, attr_s, kind) for one step.

    attr_lane/attr_s pick the lane and position whose speed limit applies
    (junction records defer to the adjacent physical record; a pair with
    no physical endpoint uses the geometry default).  kind is one of
    "physical", "junction", "lane_change", "boundary_change".
    """
    a_junc, b_junc = lane_a == JUNCTION, lane_b == JUNCTION
    if a_junc and b_junc:
        return abs(s_b - s_a), None, None, "junction"
    if a_junc or b_junc:
        lane_phys = lane_b if a_junc else lane_a
        s_own = s_b if a_junc else s_a
        s_other = s_a if a_junc else s_b
        return abs(s_other - s_own), lane_phys, \
            0.5 * (s_own + s_other), "junction"
    # both physical
    if lane_a == lane_b:
        return s_b - s_a, lane_a, 0.5 * (s_a + s_b), "physical"
    found = geom.change_segments(lane_a, s_a, lane_b, s_b)
    if found is not None:                        # lane change within an edge
        seg_a, seg_b = found
        adv = (s_b - seg_b.s0) - (s_a - seg_a.s0)
        return adv, lane_b, s_b, "lane_change"
    # lane change + chain-boundary crossing inside one step: decompose as
    # (end of a's segment) + (start of b's segment) -- exact for changes
    # at the boundary, bounded elsewhere by the chain-offset difference.
    seg_a = max(geom.segments_near(lane_a, s_a), key=lambda g: g.s1)
    seg_b = min(geom.segments_near(lane_b, s_b), key=lambda g: g.s0)
    adv = (seg_a.s1 - s_a) + (s_b - seg_b.s0)
    return adv, lane_b, s_b, "boundary_change"


# ------------------------------------------------------------- waiting ----

def waiting_component(schedule: dict, insertion: dict, t0: float,
                      horizon_s: float) -> tuple[list, dict]:
    """Entry waiting at the mainline entrance and the ramp.

    schedule:  {vid: (t_dep_s, entrance)}  scheduled departures
               (entrance ENTRANCE_MAINLINE=0 / ENTRANCE_RAMP=1); entries
               with t_dep beyond the horizon are irrelevant and may be
               omitted by the caller.
    insertion: {vid: t_ins_s} actual insertions INSIDE (t0, t0+H]
               (a vehicle inserted outside the window is equivalent to
               absent: its wait truncates to 0 or to the full horizon).

    Returns (list[WaitRecord], diagnostics).  A record with matched=False
    is an insertee absent from the schedule (contributes 0 waiting).
    """
    t0, t1 = float(t0), float(t0) + float(horizon_s)
    out = []
    seen = set()
    for vid, (t_dep, entrance) in schedule.items():
        t_dep = float(t_dep)
        if t_dep > t1 + EPS:
            continue
        t_ins = float(insertion.get(vid, np.inf))
        wait = min(t_ins, t1) - max(t_dep, t0)
        wait = float(min(max(wait, 0.0), horizon_s))
        queued_at_horizon = (t_ins > t1 + EPS)
        out.append(WaitRecord(vid=vid, entrance=int(entrance), t_dep=t_dep,
                              t_ins=t_ins, wait_s=wait, matched=True,
                              queued_at_horizon=queued_at_horizon))
        seen.add(vid)
    unmatched = [vid for vid in insertion if vid not in seen]
    for vid in unmatched:                      # count, contribute nothing
        out.append(WaitRecord(vid=vid, entrance=ENTRANCE_NONE,
                              t_dep=float("nan"),
                              t_ins=float(insertion[vid]), wait_s=0.0,
                              matched=False))
    diag = dict(n_unmatched_sched_id=len(unmatched),
                n_schedule_candidates=len(seen))
    return out, diag


def window_totals(vehicles: list, waits: list, t0: float, horizon_s: float,
                  diagnostics: dict | None = None) -> dict:
    wd = WindowDelay(t0=t0, horizon_s=horizon_s, vehicles=list(vehicles),
                     waits=list(waits), diagnostics=dict(diagnostics or {}))
    return wd.totals()
