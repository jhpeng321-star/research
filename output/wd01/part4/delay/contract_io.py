"""Contract trajectory-layout adapter (part 4).

Turns PREDICT_CONTRACT_v1.md §4.3 trajectory arrays -- per vehicle
[T, 4] with columns (t_s, s_m, lane_id, speed_mps), NaN rows forming a
trailing suffix (step 0 always real; lane ids 0/1/2) -- plus the arrivals
draw rows (t_s, entrance, speed_mps, is_cav, vehicle_type) and a schedule
{vid: (t_dep_s, entrance)} into the same accounting as the truth adapter.

`trajectories` holds vehicles present at t0 (tracked; step 0 real).
`arrived_trajectories` holds vehicles inserted inside the window (NaN
prefix up to the insertion step, real rows after, NaN suffix from the
exit step on) -- the trajectories a perfect simulator or a Part A rollout
would emit for the arrival-draw vehicles.  Arrival DRAW rows carry no
vehicle ids (contract §3.1), so insertion identities come from the
trajectory arrays; the rows are used as a count-level cross-check.

The adapter TOLERATES lane id -1 (junction) rows as an extension of the
layout: the accounting bridges junction records exactly as the truth
adapter does (delay.core._pair_advance).
"""
from __future__ import annotations

import numpy as np

from . import core
from .geometry import ENTRANCE_MAINLINE, ENTRANCE_RAMP, SectionGeometry

ENTRANCE_NAMES = {"mainline": ENTRANCE_MAINLINE, "ramp": ENTRANCE_RAMP,
                  ENTRANCE_MAINLINE: ENTRANCE_MAINLINE,
                  ENTRANCE_RAMP: ENTRANCE_RAMP}


def _entrance_code(e) -> int:
    code = ENTRANCE_NAMES.get(e)
    if code is None:
        raise ValueError(f"entrance label {e!r} not in {{mainline, ramp}}")
    return code


def _finite_rows(arr: np.ndarray, vid: str) -> np.ndarray:
    """Rows up to the first non-finite (t, s, lane) row; NaN must be a
    trailing suffix, and lane -1 (junction) rows count as finite."""
    arr = np.asarray(arr, dtype=np.float64)
    bad = ~np.isfinite(arr[:, :3]).all(axis=1)
    if bad.any():
        first = int(np.argmax(bad))
        if not bad[first:].all():
            raise ValueError(f"vehicle {vid!r}: NaN rows must be a trailing "
                             "suffix (contract §8.6)")
        return arr[:first]
    return arr


def traces_from_contract(trajectories: dict | None,
                         arrived_trajectories: dict | None, t0: float):
    """Contract arrays -> (present-at-t0 traces, in-window traces, insertion).

    insertion = {vid: t_ins} for in-window vehicles (first real row).
    """
    traces_t0, traces_win, insertion = [], [], {}
    for vid, arr in (trajectories or {}).items():
        rows = _finite_rows(arr, vid)
        if len(rows) == 0:
            raise ValueError(f"tracked vehicle {vid!r} has no real rows "
                             "(contract: step 0 is always real)")
        if abs(rows[0, 0] - t0) > core.EPS:
            raise ValueError(f"tracked vehicle {vid!r} does not start at t0")
        traces_t0.append(core.VehicleTrace(
            vid=str(vid), t=rows[:, 0], lane=rows[:, 2].astype(np.int64),
            s=rows[:, 1]))
    for vid, arr in (arrived_trajectories or {}).items():
        full = np.asarray(arr, dtype=np.float64)
        head_nan = ~np.isfinite(full[:, :3]).all(axis=1)
        real = np.flatnonzero(~head_nan)
        if len(real) == 0:
            continue                       # never inserted within the window
        first = int(real[0])               # NaN prefix until insertion
        rows = _finite_rows(full[first:], vid)
        insertion[str(vid)] = float(rows[0, 0])
        traces_win.append(core.VehicleTrace(
            vid=str(vid), t=rows[:, 0], lane=rows[:, 2].astype(np.int64),
            s=rows[:, 1]))
    return traces_t0, traces_win, insertion


def delay_from_contract_layout(
        t0: float, horizon_s: float, geometry: SectionGeometry,
        trajectories: dict | None = None,
        arrived_trajectories: dict | None = None,
        arrivals: dict | None = None,
        schedule: dict | None = None,
        entered_before_t0: dict | set | None = None,
        dt: float = 1.0, speed_factors: dict | None = None,
        strict: bool = True) -> core.WindowDelay:
    """Total-delay accounting from contract-layout inputs.

    trajectories:        {vid: [T,4]} tracked vehicles, present at t0
    arrived_trajectories:{vid: [T,4]} inserted in (t0, t0+H] (NaN prefix)
    arrivals:            optional draw rows {"t_s": (n,), "entrance": (n,)}
                         -- used only for the count-level diagnostic
    schedule:            {vid: (t_dep_s, entrance)}; entry waiting needs
                         it (waiting = actual insertion - scheduled
                         departure, matched by id, truncated at t0+H)
    entered_before_t0:   ids (dict-keys or set) of vehicles that already
                         entered at or before t0 but have no trajectory in
                         the window (entered AND left before t0, or queued
                         entries already served).  Without them, scheduled
                         entries absent from the trajectories would count
                         as queued at the horizon.
    """
    traces_t0, traces_win, insertion = traces_from_contract(
        trajectories, arrived_trajectories, t0)
    diagnostics = dict(n_arrival_rows=int(len(arrivals["t_s"]))
                       if arrivals is not None and "t_s" in (arrivals or {})
                       else None)
    t1 = t0 + horizon_s
    # vehicles present at t0 count as already inserted for the waiting
    # account (their entry wait predates the window; t_ins = t0 truncates
    # it to 0 and keeps them out of the queued-at-horizon count)
    ins_all = dict(insertion)
    for trace in traces_t0:
        ins_all[str(trace.vid)] = t0
    for vid in (entered_before_t0 or {}):
        ins_all.setdefault(str(vid), t0)
    sched_w = {vid: (float(td), _entrance_code(ent))
               for vid, (td, ent) in (schedule or {}).items()
               if float(td) <= t1 + core.EPS}
    ins_w = ins_all
    waits, wdiag = core.waiting_component(sched_w, ins_w, t0, horizon_s)
    veh, diags = [], dict(n_gapped=0, n_junction_records=0,
                          n_boundary_change_steps=0)
    for trace in traces_t0 + traces_win:
        v = core.vehicle_delay(trace, t0, horizon_s, geometry, dt,
                               speed_factors=speed_factors, strict=strict)
        veh.append(v)
        diags["n_gapped"] += int(v.gapped)
        diags["n_junction_records"] += v.n_junction_records
        diags["n_boundary_change_steps"] += v.n_boundary_change_steps
    return core.WindowDelay(t0=t0, horizon_s=horizon_s, vehicles=veh,
                            waits=waits,
                            diagnostics={**diagnostics, **diags, **wdiag})


# ------------------------------------------- truth -> contract-layout view ----

def contract_view_of_truth(rd, t0: float, horizon_s: float):
    """Re-express a truth window in the contract's trajectory layout
    (1 s sampling; NaN trailing suffix; NaN prefix for insertees), for
    adapter cross-checks.  Junction rows keep lane -1 (see module doc).

    Returns (trajectories, arrived_trajectories, arrivals, schedule).
    """
    dt_c = 2 * rd.dt                       # 0.5 s records -> 1 s rows
    k0 = int(round((t0 - rd.window_t0) / rd.dt))
    kH = k0 + int(round(horizon_s / rd.dt))
    t1 = rd.window_t0 + kH * rd.dt
    T = (kH - k0) // 2 + 1
    tracks, arrived = {}, {}
    for vi in range(len(rd.vid_table)):
        lo, hi = rd.start[vi], rd.end[vi]
        a = int(np.searchsorted(rd.tick[lo:hi], k0, "left"))
        b = int(np.searchsorted(rd.tick[lo:hi], kH, "right"))
        if b <= a:
            continue
        ticks = rd.tick[lo + a:lo + b]
        # 1 s grid anchored at the vehicle's first in-window tick (t0 for
        # tracked vehicles, the insertion tick for insertees): kept rows
        # map 1:1 onto grid rows, so dt=1 s accounting stays exact; the
        # residual error is the trailing half-step before an exit.
        anchor = int(ticks[0])
        keep = ((ticks - anchor) % 2) == 0
        ticks = ticks[keep]
        vid = str(rd.vid_table[vi])
        Ti = (kH - anchor) // 2 + 1
        arr = np.full((Ti, 4), np.nan)
        tt = rd.window_t0 + ticks * rd.dt
        kk = ((ticks - anchor) // 2).astype(np.int64)
        arr[kk, 0] = tt
        arr[kk, 1] = rd.s[lo + a:lo + b][keep]
        arr[kk, 2] = rd.lane[lo + a:lo + b][keep]
        arr[kk, 3] = 0.0                   # speed column unused here
        if anchor <= k0:
            tracks[vid] = arr
        else:
            arrived[vid] = arr
    t_ins = rd.window_t0 + rd.ar_tick.astype(np.float64) * rd.dt
    m = (t_ins > t0 + core.EPS) & (t_ins <= t1 + core.EPS)
    arrivals = dict(
        t_s=t_ins[m],
        entrance=np.where(rd.ar_entrance[m] == 0, "mainline", "ramp"),
        speed_mps=rd.ar_v[m],
        is_cav=np.zeros(int(m.sum()), dtype=np.int8),
        vehicle_type=rd.ar_type[m])
    # ids inserted at or before t0 (they must not be counted as queued)
    entered = {str(rd.vid_table[vi])
               for vi in np.flatnonzero(rd.b_tick.astype(np.int64) <= k0)}
    from .truth import load_schedule, root_run_dir
    schedule = {vid: (sd[0], _entrance_code(sd[1]))
                for vid, sd in load_schedule(root_run_dir(rd.root)).items()
                if sd[0] <= t1 + core.EPS}
    return tracks, arrived, arrivals, schedule, entered
