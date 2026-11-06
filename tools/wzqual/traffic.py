"""Traffic series at the wz3to2 lane drop computed from a :class:`~tools.wzqual.traj.Traj`.

Every quantity uses the steps AFTER t0 with ``t0 < time <= t1`` (the SUMO recorder accumulates
after each physics step, so J on ``[t0, t0+H)`` is the sum over those steps).

* team delay J (veh-s) = sum_steps dt * [sum_on-road max(0, 1 - v/31.29) + backlog] - the
  newmap2/coopkinds label definition; ``cohort`` restricts the on-road term to the vehicles
  present at t0 (entries-off backends), ``include_backlog`` adds the source term when known;
* approach-zone speed in 10 s bins (breakdown signal), queue count / closing-lane queue /
  queue extent at the drop (bin means), discharge throughput across the detector;
* per-section mean speed and density in 30 s bins (traffic-level error).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from . import frame as F
from .traj import Traj


def _after(traj: Traj, t0: float, t1: float) -> np.ndarray:
    """Row mask of steps with t0 < time <= t1."""
    t = traj.time_s[traj.step]
    return (t > t0 + 1e-9) & (t <= t1 + 1e-9)


def _steps_after(traj: Traj, t0: float, t1: float) -> np.ndarray:
    return np.flatnonzero((traj.time_s > t0 + 1e-9) & (traj.time_s <= t1 + 1e-9))


def team_delay(traj: Traj, t0: float, t1: float, *, cohort: Optional[np.ndarray] = None,
               include_backlog: bool = True) -> float:
    m = _after(traj, t0, t1)
    if cohort is not None:
        m &= np.isin(traj.vid, cohort)
    loss = np.clip(1.0 - traj.v[m] / F.SPEED_LIMIT, 0.0, None)
    total = traj.dt * float(loss.sum())
    if include_backlog and traj.backlog is not None:
        b = traj.backlog[_steps_after(traj, t0, t1)]
        total += traj.dt * float(np.nansum(b))
    return total


def delay_series(traj: Traj, t0: float, t1: float, *, bin_s: float = 1.0,
                 cohort: Optional[np.ndarray] = None, include_backlog: bool = True) -> np.ndarray:
    """Per-bin team delay on [t0, t1) (bin b covers (t0 + b*bin, t0 + (b+1)*bin])."""
    n = int(round((t1 - t0) / bin_s))
    out = np.zeros(n)
    m = _after(traj, t0, t1)
    if cohort is not None:
        m &= np.isin(traj.vid, cohort)
    t = traj.time_s[traj.step[m]]
    b = np.clip(np.ceil((t - t0) / bin_s - 1e-9).astype(int) - 1, 0, n - 1)
    np.add.at(out, b, traj.dt * np.clip(1.0 - traj.v[m] / F.SPEED_LIMIT, 0.0, None))
    if include_backlog and traj.backlog is not None:
        ks = _steps_after(traj, t0, t1)
        bb = np.clip(np.ceil((traj.time_s[ks] - t0) / bin_s - 1e-9).astype(int) - 1, 0, n - 1)
        np.add.at(out, bb, traj.dt * np.nan_to_num(traj.backlog[ks]))
    return out


def _bins(traj: Traj, t0: float, t1: float, bin_s: float):
    n = int(round((t1 - t0) / bin_s))
    m = _after(traj, t0, t1)
    t = traj.time_s[traj.step]
    b = np.ceil((t - t0) / bin_s - 1e-9).astype(int) - 1
    return n, m, b


def zone_speed_bins(traj: Traj, t0: float, t1: float, *, bin_s: float = 10.0,
                    zone=F.APPROACH_ZONE_S) -> np.ndarray:
    """Mean speed of vehicle-steps inside ``zone`` per bin (NaN = zone empty)."""
    n, m, b = _bins(traj, t0, t1, bin_s)
    m &= (traj.s >= zone[0]) & (traj.s <= zone[1])
    num = np.zeros(n)
    den = np.zeros(n)
    np.add.at(num, b[m], traj.v[m])
    np.add.at(den, b[m], 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def queue_bins(traj: Traj, t0: float, t1: float, *, bin_s: float = 10.0) -> dict:
    """Bin means of: queued vehicles (v < 5 m/s) in the queue zone (all lanes, closing lane)
    and queue extent (m upstream of the drop of the furthest queued vehicle, 0 if none)."""
    n, m, b = _bins(traj, t0, t1, bin_s)
    ks = _steps_after(traj, t0, t1)
    steps_per_bin = np.zeros(n)
    kb = np.ceil((traj.time_s[ks] - t0) / bin_s - 1e-9).astype(int) - 1
    np.add.at(steps_per_bin, kb, 1.0)
    slow = m & (traj.v < F.QUEUE_SPEED)
    inz = slow & (traj.s >= F.QUEUE_ZONE_S[0]) & (traj.s <= F.QUEUE_ZONE_S[1])
    count = np.zeros(n)
    np.add.at(count, b[inz], 1.0)
    closing = np.zeros(n)
    cl = inz & (traj.lane == F.CLOSING_LANE)
    np.add.at(closing, b[cl], 1.0)
    ext_rows = slow & (traj.s >= F.QUEUE_EXTENT_RANGE_S[0]) & (traj.s <= F.QUEUE_EXTENT_RANGE_S[1])
    per_step_ext = np.zeros(traj.n_steps)
    if ext_rows.any():
        mins = np.full(traj.n_steps, np.inf)
        np.minimum.at(mins, traj.step[ext_rows], traj.s[ext_rows])
        per_step_ext = np.where(np.isfinite(mins), F.DROP_S - mins, 0.0)
    extent = np.zeros(n)
    np.add.at(extent, kb, per_step_ext[ks])
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.maximum(steps_per_bin, 1)
        return dict(count=count / d, closing=closing / d, extent_m=extent / d)


def crossings(traj: Traj, t0: float, t1: float, *, detector_s: float = F.DETECTOR_S) -> np.ndarray:
    """Times of detector crossings (consecutive rows of one vehicle with s_k < det <= s_k+1)."""
    order = np.lexsort((traj.step, traj.vid))
    vid, step, s = traj.vid[order], traj.step[order], traj.s[order]
    same = (vid[1:] == vid[:-1]) & (step[1:] == step[:-1] + 1)
    cross = same & (s[:-1] < detector_s) & (s[1:] >= detector_s)
    t = traj.time_s[step[1:][cross]]
    return t[(t > t0 + 1e-9) & (t <= t1 + 1e-9)]


def throughput_vph(traj: Traj, t0: float, t1: float, *, detector_s: float = F.DETECTOR_S) -> float:
    return 3600.0 * len(crossings(traj, t0, t1, detector_s=detector_s)) / max(t1 - t0, 1e-9)


def section_bins(traj: Traj, t0: float, t1: float, *, bin_s: float = 30.0) -> dict:
    """Per section: mean speed (NaN if empty) and density veh/km/lane per bin."""
    n, m, b = _bins(traj, t0, t1, bin_s)
    ks = _steps_after(traj, t0, t1)
    steps_per_bin = np.zeros(n)
    np.add.at(steps_per_bin, np.ceil((traj.time_s[ks] - t0) / bin_s - 1e-9).astype(int) - 1, 1.0)
    out = {}
    for name, (a, z) in F.SECTIONS_S.items():
        mm = m & (traj.s >= a) & (traj.s < z)
        num, den = np.zeros(n), np.zeros(n)
        np.add.at(num, b[mm], traj.v[mm])
        np.add.at(den, b[mm], 1.0)
        lanes = 2.0 if name == "downstream" else 3.0
        with np.errstate(invalid="ignore", divide="ignore"):
            out[name] = dict(speed=np.where(den > 0, num / np.maximum(den, 1), np.nan),
                             density=den / np.maximum(steps_per_bin, 1) / ((z - a) / 1000.0) / lanes)
    return out


def lane_changes(traj: Traj, t0: float, t1: float) -> dict:
    """Lane-change counts by WM lane transition 'a->b' (consecutive rows of one vehicle) on
    (t0, t1] and the vehicle-hours observed (rate = count / veh_h).  Junction folding: a closing-lane
    vehicle that passes the drop is counted as 2->1 (it must leave the closing lane)."""
    order = np.lexsort((traj.step, traj.vid))
    vid, step, lane = traj.vid[order], traj.step[order], traj.lane[order]
    t = traj.time_s[step]
    same = (vid[1:] == vid[:-1]) & (step[1:] == step[:-1] + 1) & (t[1:] > t0 + 1e-9) & (t[1:] <= t1 + 1e-9)
    ch = same & (lane[1:] != lane[:-1])
    out: dict = {}
    for a, b in zip(lane[:-1][ch], lane[1:][ch]):
        key = f"{int(a)}->{int(b)}"
        out[key] = out.get(key, 0) + 1
    veh_h = float(same.sum()) * traj.dt / 3600.0
    return dict(counts=out, veh_h=veh_h)
