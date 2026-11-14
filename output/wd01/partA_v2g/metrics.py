"""WD-01 partA_v2g metrics: the Part A metrics over TRACKED vehicles.

Conventions follow the frozen project definitions (p3lib / act_lib /
part3 engine):
- window delay: sum over 0.5 s steps of (DT - ds/limit) per vehicle,
  ds = decrease of d = downstream progress (act_lib.delay_step);
- TTC: same-lane follower->leader pairs, gap = s_lead - s_foll
  (engine.ttc_stats convention, applied identically to predictions and
  truth so the comparison is like-for-like); exposure = pair-ticks * DT;
- hard braking: vehicle-steps with a < -3 m/s^2 (a = dv/DT);
- collisions: bumper gap < 0 after an update (engine events);
- lane changes by type: merge (2->1), discretionary left (1->0),
  discretionary right (0->1).
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .engine import segment_limits

HARD_BRAKE_MPS2 = -3.0


def delay_step(prev_lane: int, prev_s: float, cur_lane: int, cur_s: float,
               params: Mapping) -> float:
    """(DT - ds/limit) for one step; ds = decrease of d (downstream
    progress); limit at the PREVIOUS position (act_lib convention)."""
    off = np.asarray(params["geometry"]["lane_offsets_m"])
    dt = float(params["dt_s"])
    lim = float(segment_limits(np.array([int(prev_lane)]),
                               np.array([float(prev_s)], dtype=np.float64),
                               params)[0])
    ds = (off[prev_lane] - prev_s) - (off[cur_lane] - cur_s)
    return dt - ds / lim


def window_delay(snaps: Sequence[Mapping[str, Tuple]], params: Mapping,
                 ids: Optional[Sequence[str]] = None) -> float:
    """Total tracked-vehicle delay over the snapshot sequence (steps
    1..len-1; junction rows lane < 0 skipped).  ids=None means ALL
    vehicles seen anywhere in the window (the missing-arrivals curve's
    all-vehicles view, arrivals included)."""
    total = 0.0
    if ids is None:
        tracked = set()
        for sn in snaps:
            tracked.update(sn)
    else:
        tracked = set(ids)
    for j in range(1, len(snaps)):
        prev, cur = snaps[j - 1], snaps[j]
        for vid in tracked:
            a, b = prev.get(vid), cur.get(vid)
            if a is None or b is None:
                continue
            if a[0] < 0 or b[0] < 0:
                continue
            total += delay_step(a[0], a[1], b[0], b[1], params)
    return total


def ttc_tick(lane: np.ndarray, s: np.ndarray, v: np.ndarray
             ) -> Tuple[int, int, float]:
    """Same-lane follower->leader TTC at one tick (engine.ttc_stats
    convention): (n_lt15, n_lt3, min_ttc or inf)."""
    lt15 = lt3 = 0
    m = float("inf")
    for li in (0, 1, 2):
        ids = np.where(lane == li)[0]
        if len(ids) < 2:
            continue
        o = ids[np.argsort(-s[ids], kind="stable")]
        for j in range(len(o) - 1):
            l_, f = int(o[j]), int(o[j + 1])
            gap = s[l_] - s[f]
            closing = v[f] - v[l_]
            if gap > 0 and closing > 0:
                t = gap / closing
                lt3 += t < 3.0
                lt15 += t < 1.5
                m = min(m, t)
    return lt15, lt3, m


def ttc_exposure(snaps: Sequence[Mapping[str, Tuple]], params: Mapping,
                 ids: Optional[Sequence[str]] = None
                 ) -> Tuple[float, float, bool]:
    """(ttc_lt3_exposure_s, min_ttc_s or nan, has_closing_pair) over the
    window (steps 1..len-1, tracked vehicles only)."""
    dt = float(params["dt_s"])
    tracked = set(snaps[0]) if ids is None else set(ids)
    expo = 0
    mins: List[float] = []
    closing = False
    for j in range(1, len(snaps)):
        snap = {vid: snaps[j][vid] for vid in tracked
                if vid in snaps[j] and snaps[j][vid][0] >= 0}
        if len(snap) < 2:
            continue
        lane = np.array([snap[v][0] for v in snap], dtype=np.int64)
        s = np.array([snap[v][1] for v in snap], dtype=np.float64)
        v = np.array([snap[v][2] for v in snap], dtype=np.float64)
        _, lt3, m = ttc_tick(lane, s, v)
        expo += lt3
        if np.isfinite(m):
            closing = True
            mins.append(float(m))
    return (expo * dt,
            float(min(mins)) if mins else float("nan"),
            closing)


def hard_brake_s(snaps: Sequence[Mapping[str, Tuple]], params: Mapping,
                 ids: Optional[Sequence[str]] = None) -> float:
    """Tracked vehicle-seconds with a < -3 m/s^2 (per step, a = dv/DT
    carried in the snapshot tuple slot 3)."""
    dt = float(params["dt_s"])
    tracked = set(snaps[0]) if ids is None else set(ids)
    n = 0
    for j in range(1, len(snaps)):
        for vid in tracked:
            st = snaps[j].get(vid)
            if st is None or st[0] < 0:
                continue
            if st[3] < HARD_BRAKE_MPS2:
                n += 1
    return n * dt


def lane_change_counts(truth_rows: Sequence[Mapping[str, Tuple]]
                       ) -> Dict[str, int]:
    """Truth lane changes by type from consecutive model-lane rows."""
    out = {"merge": 0, "disc_left": 0, "disc_right": 0}
    for j in range(1, len(truth_rows)):
        prev, cur = truth_rows[j - 1], truth_rows[j]
        for vid, st in cur.items():
            p = prev.get(vid)
            if p is None or p[0] < 0 or st[0] < 0 or p[0] == st[0]:
                continue
            if p[0] == 2 and st[0] == 1:
                out["merge"] += 1
            elif p[0] == 1 and st[0] == 0:
                out["disc_left"] += 1
            elif p[0] == 0 and st[0] == 1:
                out["disc_right"] += 1
    return out


def traj_rmse(pred_snaps: Sequence[Mapping[str, Tuple]],
              truth_snaps: Sequence[Mapping[str, Tuple]],
              ids: Sequence[str]) -> Dict[str, float]:
    """Id-matched RMSE of s and v over steps 1..H (both present, truth
    lane >= 0), plus match counts."""
    se_s = se_v = 0.0
    n = 0
    for j in range(1, min(len(pred_snaps), len(truth_snaps))):
        trow = truth_snaps[j]
        prow = pred_snaps[j]
        for vid in ids:
            p, t = prow.get(vid), trow.get(vid)
            if p is None or t is None or t[0] < 0 or p[0] < 0:
                continue
            se_s += (p[1] - t[1]) ** 2
            se_v += (p[2] - t[2]) ** 2
            n += 1
    return {"s_rmse": float(np.sqrt(se_s / n)) if n else float("nan"),
            "v_rmse": float(np.sqrt(se_v / n)) if n else float("nan"),
            "n_matched": n}


def snaps_list(engine) -> List[Mapping[str, Tuple]]:
    """Engine snapshots as a plain list (drops the __t__ marker)."""
    n = max(k for k in engine.snapshots if k != "__t__")
    return [{k: v for k, v in engine.snapshots[j].items() if k != "__t__"}
            for j in range(n + 1)]


# ---------------------------------------------- one-pass array form (eval)

class WindowSeries:
    """Snapshot series as [T, n] arrays for one id set, so every
    window metric is a numpy slice instead of a dict walk.  Semantics
    identical to window_delay / ttc_exposure / hard_brake_s /
    lane_change_counts (pinned by test_metrics consistency)."""

    def __init__(self, snaps: Sequence[Mapping[str, Tuple]],
                 ids: Optional[Sequence[str]] = None):
        if ids is None:
            idset = set()
            for sn in snaps:
                idset.update(sn)
            ids = sorted(idset)
        else:
            ids = sorted(set(ids))
        self.ids = ids
        idx = {v: i for i, v in enumerate(ids)}
        T, n = len(snaps), len(ids)
        self.lane = np.full((T, n), -1, dtype=np.int64)
        self.s = np.full((T, n), np.nan)
        self.v = np.full((T, n), np.nan)
        self.a = np.full((T, n), np.nan)
        for j, sn in enumerate(snaps):
            for vid, st in sn.items():
                i = idx.get(vid)
                if i is not None:
                    self.lane[j, i], self.s[j, i], self.v[j, i], \
                        self.a[j, i] = st[0], st[1], st[2], st[3]

    def stats(self, j_hi: int, params: Mapping) -> dict:
        """Window metrics over steps 1..j_hi."""
        dt = float(params["dt_s"])
        lane = self.lane[:j_hi + 1]
        s = self.s[:j_hi + 1]
        v = self.v[:j_hi + 1]
        a = self.a[:j_hi + 1]
        present = lane >= 0
        # delay (steps 1..j_hi, both endpoints present)
        both = present[1:] & present[:-1]
        off = np.asarray(params["geometry"]["lane_offsets_m"])[lane[:-1]]
        lim = segment_limits(lane[:-1].reshape(-1), s[:-1].reshape(-1),
                             params).reshape(s[:-1].shape)
        ds = (off - s[:-1]) - (np.asarray(
            params["geometry"]["lane_offsets_m"])[lane[1:]] - s[1:])
        delay = float(np.sum(np.where(both, dt - ds / lim, 0.0)))
        # ttc / hard brake / lane changes per step
        expo = 0
        mins = []
        closing = False
        hard = 0
        lc = {"merge": 0, "disc_left": 0, "disc_right": 0}
        for j in range(1, j_hi + 1):
            m = present[j]
            if m.sum() >= 2:
                _, lt3, mn = ttc_tick(lane[j][m], s[j][m], v[j][m])
                expo += lt3
                if np.isfinite(mn):
                    closing = True
                    mins.append(float(mn))
            hard += int(np.sum(m & (a[j] < HARD_BRAKE_MPS2)))
            if j >= 1:
                prev_m = present[j - 1]
                bothj = m & prev_m
                to = np.where(bothj & (lane[j] != lane[j - 1]))[0]
                for i in to:
                    p, c = int(lane[j - 1, i]), int(lane[j, i])
                    if p == 2 and c == 1:
                        lc["merge"] += 1
                    elif p == 1 and c == 0:
                        lc["disc_left"] += 1
                    elif p == 0 and c == 1:
                        lc["disc_right"] += 1
        return dict(delay=delay, expo3=expo * dt,
                    min_ttc=float(min(mins)) if mins else float("nan"),
                    closing=closing, hard=hard * dt, lc=lc)

    def rmse(self, other: "WindowSeries", j_hi: int,
             offsets: Optional[Sequence[float]] = None) -> dict:
        """Id-matched position/speed RMSE over steps 1..j_hi (both
        present).  Position error is on the SHARED d axis (d =
        OFFSETS[lane] - s): per-lane s offsets differ by up to 2265 m,
        so raw s would explode at every lane change."""
        if offsets is None:
            offsets = (2645.67, 2644.22, 378.66)
        off = np.asarray(offsets, dtype=np.float64)
        m = (self.lane[1:j_hi + 1] >= 0) & (other.lane[1:j_hi + 1] >= 0)
        d_self = off[self.lane[1:j_hi + 1]] - self.s[1:j_hi + 1]
        d_other = off[other.lane[1:j_hi + 1]] - other.s[1:j_hi + 1]
        dd = (d_self - d_other)[m]
        dv = (self.v[1:j_hi + 1] - other.v[1:j_hi + 1])[m]
        n = int(m.sum())
        return {"s_rmse": float(np.sqrt((dd ** 2).sum() / n)) if n
                else float("nan"),
                "v_rmse": float(np.sqrt((dv ** 2).sum() / n)) if n
                else float("nan"),
                "n_matched": n}
