"""S-COOPMEAS behaviour classifier: named cooperative CAV behaviours from SUMO trajectories.

Controller-agnostic: the input is a trajectory table of every vehicle (time, id, s, v, a, lane)
plus the controller's own CAV list (CAV vs HDV identity); nothing about driver types is read.
Model frame (``traffic_shepherd.c4.geometry``): lane 0 = main left, 1 = main right,
2 = ramp/acceleration lane, -1 = junction internal lane; ``s`` merge-relative (merge zone
0..75.44 m, upstream negative).

Per CAV and 1 Hz sample (integer times only):

* ``gap``   bumper gap to the same-lane leader (5 m nominal length), capped at 200 m;
* ``du``    ``v_ref - v`` with ``v_ref`` = leader speed if the leader is within 100 m, else
            the speed limit (31.29 m/s);  ``u`` = mean of ``du`` over the 5 most recent modelled
            samples ("voluntary slowness": positive = slower than the traffic ahead allows);
* ``sdL / sdC / sdF`` = std over the 20 most recent samples of the leader's / own / follower's
            speed (leader and follower within 100 m on every one of those samples);
            ``rF = sdF / sdL`` (follower speed variation per unit of upstream variation).

Behaviours (thresholds ``theta_*`` from the NATIVE calibration distribution, frozen in a
registered JSON before any treatment result is looked at; see :func:`derive_thresholds`):

* ``multi_slow``  >= 2 CAVs on lanes 0/1 with s in [-1500, 75.44] m, each u > theta_u, within
                  150 m of each other along s, >= 3 consecutive s (union of members);
                  subtype ``side_by_side`` if some pair is in different lanes within 30 m;
* ``metering``    one CAV on lanes 0/1 with s in [-800, -100] m and u > theta_u for >= 5 s;
* ``gap_used``    a ramp car merges (lane 2 -> 1) directly ahead of a CAV (new follower in
                  lane 1 within 100 m) and that CAV's max u over the 8 s before the merge
                  > theta_O;  (``merge_ahead_cav`` counts every ramp merge ahead of a CAV);
* ``vacate``      a CAV changes lane 1 -> 0 with s in [-400, 0] m; subtype ``ramp_present``
                  if a ramp-lane vehicle is within 100 m along s at that time;
* ``damping``     a CAV with leader and follower present, sdL >= sigma_min and
                  rF < theta_rF for >= 3 consecutive s.

Events are counted when they START inside the measurement window; exposure = CAV-seconds on
a modelled lane inside the window.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

SPEED_LIMIT_MPS = 31.29
MERGE_S1_M = 75.44
LEN_M = 5.0
GAP_CAP_M = 200.0
LEAD_NEAR_M = 100.0
U_SAMPLES = 5
SD_SAMPLES = 20
BEHAVIOURS = ("multi_slow", "metering", "gap_used", "vacate", "damping")
#: priority for the attribution replays (traffic_shepherd.c4.coopmeas)
ATTRIBUTION_PRIORITY = ("gap_used", "multi_slow", "metering", "vacate", "damping")
QUANTILES = {"theta_u": 0.95, "sigma_min": 0.50, "theta_rF": 0.05, "theta_O": 0.90}
SNIPPET_LEN, SNIPPET_STRIDE = 10, 5
SNIPPET_FEATURES = ("v_mean", "v_std", "a_mean", "a_min", "u_mean", "u_max", "gap_mean",
                    "lane_changes", "lane_end", "s_start_km", "n_cav_near", "u_cav_near",
                    "n_ramp_near", "sdF", "sdL")


# ===================================================================== table + features
def to_frame(t, vid, s, v, a, lane, vids, cav_set) -> pd.DataFrame:
    """Trajectory arrays (``vid`` = index into ``vids``) -> 1 Hz table with a ``cav`` flag."""
    df = pd.DataFrame({"t": np.asarray(t, np.float64), "vid": np.asarray(vids, object)[np.asarray(vid)],
                       "s": np.asarray(s, np.float64), "v": np.asarray(v, np.float64),
                       "a": np.asarray(a, np.float64), "lane": np.asarray(lane, np.int64)})
    df = df[np.abs(df.t - np.round(df.t)) < 1e-6].copy()
    df["t"] = np.round(df.t).astype(np.int64)
    df["cav"] = df.vid.isin(set(cav_set))
    return df.drop_duplicates(["t", "vid"], keep="last").reset_index(drop=True)


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Leader/follower, u, rolling speed std (all vehicles; rows on lanes 0-2 only)."""
    d = df[df.lane >= 0].sort_values(["t", "lane", "s"]).reset_index(drop=True)
    grp = d.groupby(["t", "lane"], sort=False)
    d["lead_s"] = grp.s.shift(-1)
    d["lead_v"] = grp.v.shift(-1)
    d["foll_s"] = grp.s.shift(1)
    d["foll_v"] = grp.v.shift(1)
    gap = d.lead_s - d.s - LEN_M
    d["gap"] = gap.fillna(GAP_CAP_M).clip(upper=GAP_CAP_M)
    near = gap < LEAD_NEAR_M
    d["v_ref"] = np.where(near, d.lead_v, SPEED_LIMIT_MPS)
    d["du"] = d.v_ref - d.v
    fgap = d.s - d.foll_s - LEN_M
    d["lead_v_near"] = d.lead_v.where(near)
    d["foll_v_near"] = d.foll_v.where(fgap < LEAD_NEAR_M)
    d = d.sort_values(["vid", "t"]).reset_index(drop=True)
    g = d.groupby("vid", sort=False)
    d["u"] = g.du.transform(lambda x: x.rolling(U_SAMPLES, min_periods=U_SAMPLES).mean())
    d["sdL"] = g.lead_v_near.transform(lambda x: x.rolling(SD_SAMPLES, min_periods=SD_SAMPLES).std())
    d["sdF"] = g.foll_v_near.transform(lambda x: x.rolling(SD_SAMPLES, min_periods=SD_SAMPLES).std())
    d["sdC"] = g.v.transform(lambda x: x.rolling(SD_SAMPLES, min_periods=SD_SAMPLES).std())
    d["rF"] = d.sdF / d.sdL.where(d.sdL > 1e-6)
    d["prev_lane"] = g.lane.shift(1)
    d["prev_t"] = g.t.shift(1)
    return d


def ramp_merges(d: pd.DataFrame) -> pd.DataFrame:
    """Ramp merges: a vehicle's last ramp-lane sample is followed (<= 3 s) by a lane-1 sample.
    Returns one row per merge: vid, t_m (first lane-1 time), s_m, follower vid in lane 1."""
    out = []
    for vid, x in d.groupby("vid", sort=False):
        lanes, ts = x.lane.to_numpy(), x.t.to_numpy()
        idx = np.flatnonzero((lanes[:-1] == 2) & (lanes[1:] == 1) & (ts[1:] - ts[:-1] <= 3))
        for j in idx:
            out.append((vid, int(ts[j + 1]), float(x.s.iloc[j + 1])))
    if not out:
        return pd.DataFrame(columns=["vid", "t_m", "s_m", "foll", "foll_cav", "foll_gap"])
    m = pd.DataFrame(out, columns=["vid", "t_m", "s_m"])
    lane1 = d[d.lane == 1]
    foll, fcav, fgap = [], [], []
    by_t = {t: x for t, x in lane1.groupby("t", sort=False)}
    for r in m.itertuples():
        x = by_t.get(r.t_m)
        behind = None if x is None else x[x.s < r.s_m]
        if behind is None or behind.empty:
            foll.append(None), fcav.append(False), fgap.append(math.inf)
            continue
        b = behind.loc[behind.s.idxmax()]
        foll.append(b.vid), fcav.append(bool(b.cav)), fgap.append(float(r.s_m - b.s - LEN_M))
    m["foll"], m["foll_cav"], m["foll_gap"] = foll, fcav, fgap
    return m


def merge_opening(d: pd.DataFrame, merges: pd.DataFrame) -> pd.DataFrame:
    """Merges ahead of a CAV (follower CAV within 100 m) with O = max u of that CAV over
    [t_m - 8, t_m]."""
    rows = []
    cav_u = d[d.cav][["vid", "t", "u", "s"]]
    for r in merges.itertuples():
        if not r.foll_cav or not (r.foll_gap < LEAD_NEAR_M):
            continue
        x = cav_u[(cav_u.vid == r.foll) & (cav_u.t >= r.t_m - 8) & (cav_u.t <= r.t_m)]
        o = float(x.u.max()) if x.u.notna().any() else math.nan
        rows.append({"merger": r.vid, "cav": r.foll, "t_m": r.t_m, "s_m": r.s_m, "O": o})
    return pd.DataFrame(rows, columns=["merger", "cav", "t_m", "s_m", "O"])


def _runs(ts: np.ndarray, min_len: int) -> list:
    """Maximal runs of consecutive integer times (diff 1) of length >= min_len -> [(t0, t1)]."""
    if ts.size == 0:
        return []
    ts = np.unique(ts)
    breaks = np.flatnonzero(np.diff(ts) != 1)
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [ts.size - 1]])
    return [(int(ts[a]), int(ts[b])) for a, b in zip(starts, ends) if b - a + 1 >= min_len]


# ===================================================================== thresholds
def derive_thresholds(feature_tables: list, window: tuple) -> dict:
    """Native calibration -> registered thresholds (quantiles in ``QUANTILES``)."""
    t0, t1 = window
    u, sdl, rows_rf, opens = [], [], [], []
    for d in feature_tables:
        w = d[(d.t >= t0) & (d.t < t1) & d.cav]
        main = w[w.lane.isin([0, 1]) & (w.s >= -1500) & (w.s <= MERGE_S1_M)]
        u.append(main.u.dropna().to_numpy())
        ok = w[w.sdL.notna() & w.sdF.notna()]
        sdl.append(ok.sdL.to_numpy())
        rows_rf.append(ok[["sdL", "rF"]])
        mo = merge_opening(d, ramp_merges(d))
        mo = mo[(mo.t_m >= t0) & (mo.t_m < t1)]
        opens.append(mo.O.dropna().to_numpy())
    u, sdl, opens = (np.concatenate(x) if x else np.zeros(0) for x in (u, sdl, opens))
    sigma_min = float(np.quantile(sdl, QUANTILES["sigma_min"]))
    rf = pd.concat(rows_rf)
    rf = rf[rf.sdL >= sigma_min].rF.dropna().to_numpy()
    th = {"theta_u": float(np.quantile(u, QUANTILES["theta_u"])),
          "sigma_min": sigma_min,
          "theta_rF": float(np.quantile(rf, QUANTILES["theta_rF"])),
          "theta_O": float(np.quantile(opens, QUANTILES["theta_O"])),
          "quantiles": QUANTILES, "window_s": list(window),
          "n": {"u": int(u.size), "sdL": int(sdl.size), "rF": int(rf.size), "O": int(opens.size)}}
    return th


# ===================================================================== detection
def detect(d: pd.DataFrame, th: dict, window: tuple) -> dict:
    """Events of every behaviour starting in ``[t0, t1)``; returns {"events": [...],
    "exposure_cav_s": ..., "merges": n, "merge_ahead_cav": n}."""
    t0, t1 = window
    cav = d[d.cav]
    ev = []
    # multi_slow
    flag = cav[cav.lane.isin([0, 1]) & (cav.s >= -1500) & (cav.s <= MERGE_S1_M) & (cav.u > th["theta_u"])]
    pair_t, members, sbs = [], {}, {}
    for t, x in flag.groupby("t", sort=True):
        if len(x) < 2:
            continue
        s, ln, vid = x.s.to_numpy(), x.lane.to_numpy(), x.vid.to_numpy()
        ds = np.abs(s[:, None] - s[None, :])
        iu = np.triu_indices(len(x), 1)
        close = ds[iu] <= 150.0
        if not close.any():
            continue
        pair_t.append(int(t))
        a, b = iu[0][close], iu[1][close]
        members[int(t)] = set(vid[a]) | set(vid[b])
        sbs[int(t)] = bool(np.any((ln[a] != ln[b]) & (ds[a, b] <= 30.0)))
    for r0, r1 in _runs(np.asarray(pair_t), 3):
        mem = set().union(*(members[t] for t in range(r0, r1 + 1) if t in members))
        side = any(sbs.get(t, False) for t in range(r0, r1 + 1))
        ev.append({"type": "multi_slow", "subtype": "side_by_side" if side else "staggered",
                   "cavs": sorted(mem), "t0": r0, "t1": r1})
    # metering
    met = cav[cav.lane.isin([0, 1]) & (cav.s >= -800) & (cav.s <= -100) & (cav.u > th["theta_u"])]
    for vid, x in met.groupby("vid", sort=False):
        for r0, r1 in _runs(x.t.to_numpy(), 5):
            ev.append({"type": "metering", "subtype": "", "cavs": [vid], "t0": r0, "t1": r1})
    # gap_used
    merges = ramp_merges(d)
    mo = merge_opening(d, merges)
    n_merge = int(((merges.t_m >= t0) & (merges.t_m < t1)).sum()) if len(merges) else 0
    n_ahead = int(((mo.t_m >= t0) & (mo.t_m < t1)).sum()) if len(mo) else 0
    for r in mo.itertuples():
        if r.O > th["theta_O"]:
            ev.append({"type": "gap_used", "subtype": "", "cavs": [r.cav], "t0": int(r.t_m) - 8,
                       "t1": int(r.t_m), "t_merge": int(r.t_m), "merger": r.merger})
    # vacate (lane 1 -> 0 with <= 3 s junction gap)
    lc = cav[(cav.prev_lane == 1) & (cav.lane == 0) & (cav.t - cav.prev_t <= 3)
             & (cav.s >= -400) & (cav.s <= 0)]
    ramp = d[d.lane == 2]
    ramp_by_t = {t: x.s.to_numpy() for t, x in ramp.groupby("t", sort=False)}
    for r in lc.itertuples():
        rs = ramp_by_t.get(r.t, np.zeros(0))
        present = bool(np.any(np.abs(rs - r.s) <= 100.0))
        ev.append({"type": "vacate", "subtype": "ramp_present" if present else "no_ramp",
                   "cavs": [r.vid], "t0": int(r.t), "t1": int(r.t)})
    # damping
    dm = cav[(cav.sdL >= th["sigma_min"]) & (cav.rF < th["theta_rF"])]
    for vid, x in dm.groupby("vid", sort=False):
        for r0, r1 in _runs(x.t.to_numpy(), 3):
            ev.append({"type": "damping", "subtype": "", "cavs": [vid], "t0": r0, "t1": r1})
    # window filter: event start (merge time for gap_used) inside [t0, t1)
    keep = []
    for e in ev:
        start = e.get("t_merge", e["t0"])
        if t0 <= start < t1:
            keep.append(e)
    exp = cav[(cav.t >= t0) & (cav.t < t1) & (cav.lane >= 0)]
    return {"events": keep, "exposure_cav_s": int(len(exp)), "merges": n_merge,
            "merge_ahead_cav": n_ahead, "window": [int(t0), int(t1)]}


def pick_attribution(events: list, k: int = 2) -> list:
    """Up to ``k`` events with distinct CAV sets, by ATTRIBUTION_PRIORITY then start time."""
    order = sorted(events, key=lambda e: (ATTRIBUTION_PRIORITY.index(e["type"]), e["t0"]))
    out, seen = [], []
    for e in order:
        s = frozenset(e["cavs"])
        if any(s == x for x in seen):
            continue
        out.append(e)
        seen.append(s)
        if len(out) >= k:
            break
    return out


# ===================================================================== snippets
def snippets(d: pd.DataFrame, window: tuple) -> pd.DataFrame:
    """10 s CAV snippets (stride 5 s, all 10 samples on lanes 0-2) inside ``window``."""
    t0, t1 = window
    per_t = {t: x for t, x in d.groupby("t", sort=False)}
    near = {}
    for t, x in per_t.items():
        s, c, ln, u = x.s.to_numpy(), x.cav.to_numpy(), x.lane.to_numpy(), x.u.to_numpy()
        vids = x.vid.to_numpy()
        cs, cu = s[c], np.nan_to_num(u[c])
        rs = s[ln == 2]
        for i in np.flatnonzero(c):
            dd = np.abs(cs - s[i])
            m = (dd <= 150.0) & (dd > 0)
            near[(t, vids[i])] = (int(m.sum()), float(cu[m].mean()) if m.any() else 0.0,
                                  int(np.sum(np.abs(rs - s[i]) <= 100.0)))
    rows = []
    for vid, x in d[d.cav].groupby("vid", sort=False):
        x = x.set_index("t")
        for start in range(int(t0), int(t1) - SNIPPET_LEN + 1, SNIPPET_STRIDE):
            ts = range(start, start + SNIPPET_LEN)
            if not all(t in x.index for t in ts):
                continue
            y = x.loc[list(ts)]
            nb = np.array([near.get((t, vid), (0, 0.0, 0)) for t in ts], np.float64)
            lanes = y.lane.to_numpy()
            rows.append({"vid": vid, "t0": start,
                         "v_mean": y.v.mean(), "v_std": y.v.std(), "a_mean": y.a.mean(),
                         "a_min": y.a.min(), "u_mean": np.nan_to_num(y.u.mean()),
                         "u_max": np.nan_to_num(y.u.max()), "gap_mean": y.gap.mean(),
                         "lane_changes": int(np.sum(lanes[1:] != lanes[:-1])),
                         "lane_end": int(lanes[-1]), "s_start_km": float(np.clip(y.s.iloc[0], -2000, 200)) / 1000.0,
                         "n_cav_near": nb[:, 0].mean(), "u_cav_near": nb[:, 1].mean(),
                         "n_ramp_near": nb[:, 2].mean(),
                         "sdF": np.nan_to_num(y.sdF.iloc[-1]), "sdL": np.nan_to_num(y.sdL.iloc[-1])})
    return pd.DataFrame(rows, columns=["vid", "t0", *SNIPPET_FEATURES])


def kmeans(x: np.ndarray, k: int, *, seed: int = 0, n_init: int = 10, iters: int = 200):
    """Plain k-means (k-means++ init, best of ``n_init``) -> (centroids, labels, inertia)."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        c = [x[rng.integers(len(x))]]
        for _ in range(1, k):
            d2 = np.min(((x[:, None, :] - np.asarray(c)[None]) ** 2).sum(-1), 1)
            c.append(x[rng.choice(len(x), p=d2 / d2.sum())])
        c = np.asarray(c)
        for _ in range(iters):
            lab = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(-1), 1)
            new = np.asarray([x[lab == j].mean(0) if np.any(lab == j) else c[j] for j in range(k)])
            if np.allclose(new, c):
                break
            c = new
        lab = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(-1), 1)
        inertia = float(((x - c[lab]) ** 2).sum())
        if best is None or inertia < best[2]:
            best = (c, lab, inertia)
    return best


def load_traj(path) -> pd.DataFrame:
    """``traj_*.npz`` written by traffic_shepherd.c4.coopmeas -> 1 Hz table."""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return to_frame(z["t"], z["vid"], z["s"], z["v"], z["a"], z["lane"], meta["vids"], meta["cavs"])


def dump_json(obj, path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def window_events_summary(det: dict) -> dict:
    out = {b: 0 for b in BEHAVIOURS}
    for e in det["events"]:
        out[e["type"]] += 1
    return out


__all__ = ["to_frame", "features", "derive_thresholds", "detect", "pick_attribution",
           "snippets", "kmeans", "load_traj", "BEHAVIOURS", "SNIPPET_FEATURES"]
