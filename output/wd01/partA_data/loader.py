"""Part A v0 NATIVE dataset loader (WD-01 part 2C).

windows(root, t0, horizon_s): full state at t0 (all vehicles), the future
truth trajectories at the 0.5 s step (births appear, exits disappear),
and the arrivals list in the PREDICT_CONTRACT_v1 explicit format
(t_s, entrance, speed_mps, is_cav, vehicle_type).

neighbour_features(...): the design §2.2 six neighbour slots (leader and
follower of the own lane, the left lane, the right lane), each with
(present, gap capped 250 m, rel speed, is_cav).  Adjacency respects the
WM-30 topology: main_left<->main_right<->accel; accel has no right
neighbour and is never adjacent to main_left.  Leader/follower search uses
the merge-relative d = OFFSET[lane] - s, comparable across lanes.

The vType label (`type_label`) is SUPERVISION ONLY and appears NOWHERE in
the input-feature outputs (enforced by tests/test_inputs.py).
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pdl as L  # noqa: E402

ADJACENT = {0: {"left": None, "right": 1},
            1: {"left": 0, "right": 2},
            2: {"left": 1, "right": None}}

# Explicit input-feature contract (vType label deliberately absent).
EGO_FEATURES = ("v", "a", "lane", "in_junction")
SLOT_NAMES = ("own_leader", "own_follower", "left_leader", "left_follower",
              "right_leader", "right_follower")
SLOT_FEATURES = ("present", "gap", "rel_speed", "is_cav")


@lru_cache(maxsize=8)
def load_root(root, registration_sha256=None):
    """All arrays of data/root_<root>.npz as a dict.

    Holdout roots require registration_sha256 (see holdout_block.py)."""
    from holdout_block import assert_holdout_allowed
    assert_holdout_allowed(root, registration_sha256)
    path = L.DATA / f"root_{root}.npz"
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def roots():
    return sorted(p.stem[5:] for p in L.DATA.glob("root_*.npz"))


def offsets():
    """OFFSET[lane] for d = OFFSET - s (from the topology, cached)."""
    return _offsets()


@lru_cache(maxsize=1)
def _offsets():
    topo, _, _ = L.topology()
    return {L.LANE_IDS[name]: float(topo.offsets[name])
            for name in L.LANE_IDS}


def _at(load, tick):
    """Row indices of one tick, grouped for fast access."""
    idx = np.where(load["tick"] == tick)[0]
    return idx


def window_ticks(t0, horizon_s):
    """Tick indices k for t in (t0, t0 + horizon_s] on the 0.5 s grid,
    plus the t0 tick k.  Raises if outside the stored window."""
    dt = L.DT
    k0 = int(round((t0 - L.WINDOW_T0) / dt))
    kk = int(round(horizon_s / dt))
    assert abs(t0 - (L.WINDOW_T0 + k0 * dt)) < 1e-6, "t0 off the 0.5 s grid"
    assert abs(horizon_s - kk * dt) < 1e-6, "horizon off the 0.5 s grid"
    assert 0 <= k0 and k0 + kk < L.N_TICKS, \
        f"window [{t0}, {t0 + horizon_s}] outside stored [60, 600]"
    return k0, kk


def windows(root, t0, horizon_s):
    """Full state at t0 + future truth at 0.5 s + arrivals.

    Returns dict:
      t0        float
      horizon_s float
      state:    dict of t0 arrays over ALL vehicles at t0:
                (vid, id, lane, s, d, v, a, is_cav, in_junction)
      truth:    dict with tick k=1..kk arrays (vid, lane, s, d, v, a,
                is_cav, in_junction) for every vehicle alive at that tick
                (births appear, exits/absent vehicles are simply missing);
                plus `new_vid` (vid indices first seen inside the window)
                and `gone_vid` (vid indices whose trajectory ends inside
                the window, with their end tick).
      arrivals: contract-format rows with t in (t0, t0+horizon]:
                (t_s, entrance 0=mainline/1=ramp, speed_mps, is_cav,
                 vehicle_type label -- SUPERVISION side info)
    """
    load = load_root(root)
    k0, kk = window_ticks(t0, horizon_s)
    state_idx = _at(load, k0)
    off = _offsets()
    off_arr = np.array([off[0], off[1], off[2], np.nan])

    def rows_of(idx):
        lane = load["lane"][idx]
        s = load["s"][idx]
        d = off_arr[np.where(lane >= 0, lane, 3)] - s
        return dict(vid=load["vid"][idx],
                    id=load["vid_table"][load["vid"][idx]],
                    lane=lane, s=s, d=d,
                    v=load["v"][idx], a=load["a"][idx],
                    is_cav=load["is_cav"][idx],
                    in_junction=load["in_junction"][idx])

    truth_ticks = []
    for k in range(k0 + 1, k0 + kk + 1):
        idx = _at(load, k)
        r = rows_of(idx)
        r["tick"] = k
        truth_ticks.append(r)
    ticks_all = load["tick"].astype(np.int64)
    vids_all = load["vid"].astype(np.int64)
    order = np.argsort(vids_all, kind="stable")
    v_sorted = vids_all[order]
    t_sorted = ticks_all[order]
    starts = np.flatnonzero(np.r_[True, v_sorted[1:] != v_sorted[:-1]])
    end_tick = np.maximum.reduceat(t_sorted, starts)
    win = (ticks_all > k0) & (ticks_all <= k0 + kk)
    seen_vids = np.unique(vids_all[win])
    base_vids = np.unique(vids_all[ticks_all <= k0])
    new_vid = np.setdiff1d(seen_vids, base_vids)
    gone_vid = np.array([vi for vi in base_vids
                         if end_tick[np.searchsorted(v_sorted[starts], vi)]
                         <= k0 + kk], dtype=np.int64)

    ar = (load["ar_tick"] > k0) & (load["ar_tick"] <= k0 + kk)
    arrivals = dict(
        t_s=L.WINDOW_T0 + load["ar_tick"][ar] * L.DT,
        entrance=load["ar_entrance"][ar], speed_mps=load["ar_v"][ar],
        is_cav=load["ar_cav"][ar], vehicle_type=load["ar_type"][ar])
    return dict(t0=t0, horizon_s=horizon_s, state=rows_of(state_idx),
                truth=truth_ticks, new_vid=new_vid, gone_vid=gone_vid,
                arrivals=arrivals)


def neighbour_features(root, t0):
    """The 6 neighbour slots for every vehicle at t0.

    Returns a dict of float32 arrays shaped [n_vehicles, 6] for each of
    present / gap / rel_speed / is_cav, with slots ordered as SLOT_NAMES.
    Junction vehicles (lane=-1) get all-absent slots.  gap is capped at
    250 m; rel_speed = v_other - v_self (positive = other is faster).
    Leader/follower search uses the merge-relative d, comparable across
    lanes; on the own lane leader = nearest strictly ahead (smaller d).
    """
    load = load_root(root)
    k0, _ = window_ticks(t0, 0.5)
    rows = np.where(load["tick"] == k0)[0]
    n = len(rows)
    lane = load["lane"][rows].astype(np.int64)
    s = load["s"][rows].astype(np.float64)
    spd = load["v"][rows].astype(np.float64)
    cav = load["is_cav"][rows].astype(np.int8)
    off = _offsets()
    off_lane = np.array([off[0], off[1], off[2], np.nan])
    d = off_lane[np.where(lane >= 0, lane, 3)] - s

    present = np.zeros((n, 6), dtype=np.float32)
    gaps = np.full((n, 6), L.GAP_MAX, dtype=np.float32)
    rel = np.zeros((n, 6), dtype=np.float32)
    slot_cav = np.zeros((n, 6), dtype=np.float32)
    slot_col = {name: j for j, name in enumerate(SLOT_NAMES)}

    by_lane = {li: np.where(lane == li)[0] for li in (-1, 0, 1, 2)}

    def fill(i, c, j):
        present[i, c] = 1.0
        gaps[i, c] = min(abs(d[i] - d[j]), L.GAP_MAX)
        rel[i, c] = spd[j] - spd[i]
        slot_cav[i, c] = cav[j]

    for i in range(n):
        li = int(lane[i])
        if li < 0:
            continue
        same = by_lane[li]
        order = same[np.argsort(d[same])]        # d ascending: front first
        pos = int(np.searchsorted(d[order], d[i]))
        if pos > 0:                              # nearest strictly ahead
            fill(i, slot_col["own_leader"], order[pos - 1])
        if pos < len(order) - 1:                 # nearest strictly behind
            fill(i, slot_col["own_follower"], order[pos + 1])
        for side in ("left", "right"):
            other = ADJACENT[li][side]
            if other is None or len(by_lane[other]) == 0:
                continue
            oth = by_lane[other]
            doth = d[oth]
            ahead_m = doth <= d[i]
            if ahead_m.any():                    # nearest ahead = max d
                fill(i, slot_col[f"{side}_leader"],
                     oth[ahead_m][np.argmax(doth[ahead_m])])
            behind_m = ~ahead_m
            if behind_m.any():                   # nearest behind = min d
                fill(i, slot_col[f"{side}_follower"],
                     oth[behind_m][np.argmin(doth[behind_m])])
    return dict(present=present, gap=gaps, rel_speed=rel, is_cav=slot_cav)


def list_roots_by_split():
    man = L.load_json(HERE / "MANIFEST.json") if (HERE / "MANIFEST.json")\
        .exists() else None
    if man is None:
        return {}
    out = {"train": [], "holdout": []}
    for name, r in man["roots"].items():
        out[r["split"]].append(name)
    return out
