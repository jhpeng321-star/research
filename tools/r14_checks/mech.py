# ruff: noqa: E741, F841  (verbatim copy of the COMBO300 / LONGDIAG analysis code)
"""S-LONGDIAG mechanism metrics from compact 1 s trajectories (longdiag_roll.py output).

Per file (one pair x arm x variant, or the SUMO stream) and per 60 s interval:
  delay by region (wm_delay_check definition, 1 s grid, relu(1-v/31.29)), vehicle counts by region,
  flows (veh/h): merge-exit crossings of s=75.44 on lanes 0/1 (bottleneck discharge), crossings of s=-800 into up_near,
  crossings of s=0 (out of up_near), ramp merges (lane 2 -> 0/1), upstream entries (first seen s<-2400), ramp entries,
  queue: slow (v<10 m/s) mainline vehicles in up_near / far_up, spillback tail (most upstream congested 25 m cell
  connected to the merge, gaps <= 100 m), discharge flow while a queue is present (>=3 slow cars in [-200, 75.44)).
"""
from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np

FREE_V = 31.29
ROAD_END = 553.33
M1 = 75.44
UPN = -800.0
RAMP_MIN = -378.66
SLOW = 10.0
EDGES = [0, 60, 120, 180, 240, 300]


def regions(S, L):
    on = np.isfinite(S) & (S < ROAD_END)
    main = on & ((L == 0) | (L == 1))
    ramp = on & (L == 2) & (S >= RAMP_MIN) & (S < M1)
    up = main & (S >= UPN) & (S < 0)
    far = main & (S < UPN)
    mrg = main & (S >= 0) & (S < M1)
    dn = main & (S >= M1)
    return {"far_up": far, "up_near": up, "ramp": ramp, "merge": mrg, "downstream": dn}


def tail_series(S, L, V):
    """per second: spillback tail s (NaN if no queue at the merge) using 25 m cells, mainline lanes pooled."""
    T = S.shape[0]
    out = np.full(T, np.nan)
    cells = np.arange(-2650, M1 + 25, 25.0)
    for t in range(T):
        s, l, v = S[t], L[t], V[t]
        ok = np.isfinite(s) & ((l == 0) | (l == 1)) & (s < M1) & (v < SLOW)
        if not ok.any():
            continue
        ci = np.floor((s[ok] + 2650) / 25).astype(int)
        cong = np.zeros(len(cells), bool)
        cong[np.clip(ci, 0, len(cells) - 1)] = True
        # start from the merge end: need a congested cell within [-100, 75.44)
        top = int(np.floor((M1 - 1e-6 + 2650) / 25))
        lo = int(np.floor((-100 + 2650) / 25))
        if not cong[lo:top + 1].any():
            continue
        j = max(i for i in range(lo, top + 1) if cong[i])
        tail = j
        gap = 0
        i = j
        while i > 0:
            i -= 1
            if cong[i]:
                tail = i
                gap = 0
            else:
                gap += 1
                if gap > 4:
                    break
        out[t] = cells[tail]
    return out


def wave_speed(S, L, V, x0=-1600.0, x1=0.0, dx=50.0, sep=200.0, maxlag=80):
    """median backward propagation speed (m/s, negative = upstream) of speed disturbances on the mainline approach:
    cell-mean speed series (50 m cells, lanes pooled); for cell pairs 200 m apart the lag (1..80 s) maximising the
    correlation of the downstream cell's series with the upstream cell's later series; only cells that see v<15."""
    T = S.shape[0]
    edges = np.arange(x0, x1 + dx, dx)
    nc = len(edges) - 1
    F = np.full((T, nc), np.nan)
    main = np.isfinite(S) & ((L == 0) | (L == 1))
    for t in range(T):
        m = main[t] & (S[t] >= x0) & (S[t] < x1)
        if not m.any():
            continue
        ci = ((S[t][m] - x0) // dx).astype(int)
        sums = np.bincount(ci, V[t][m], nc)
        cnt = np.bincount(ci, None, nc)
        with np.errstate(invalid="ignore", divide="ignore"):
            F[t] = np.where(cnt > 0, sums / cnt, np.nan)
    # fill gaps per cell by interpolation over time
    for c in range(nc):
        y = F[:, c]
        ok = np.isfinite(y)
        if ok.sum() > 10:
            F[:, c] = np.interp(np.arange(T), np.where(ok)[0], y[ok])
    k = int(sep // dx)
    lags = []
    for c in range(k, nc):
        down, up = F[:, c], F[:, c - k]
        if not (np.isfinite(down).all() and np.isfinite(up).all()):
            continue
        if min(down.min(), up.min()) > 15.0:
            continue
        a = down - down.mean()
        best, bl = -1.0, None
        for lag in range(1, maxlag + 1):
            x, y = a[:-lag], (up - up.mean())[lag:]
            den = np.sqrt((x * x).sum() * (y * y).sum())
            if den <= 0:
                continue
            r = (x * y).sum() / den
            if r > best:
                best, bl = r, lag
        if bl is not None and best > 0.5:
            lags.append(bl)
    return float(-sep / np.median(lags)) if lags else np.nan


def crossings(S, L, x, lanes=(0, 1)):
    """per second index t (t -> t+1): number of vehicles crossing s=x upward, on lanes at t+1 in ``lanes``."""
    a, b = S[:-1], S[1:]
    lb = L[1:]
    ok = np.isfinite(a) & np.isfinite(b) & (a < x) & (b >= x) & np.isin(lb, lanes)
    return ok.sum(1)


def metrics(path):
    d = np.load(path, allow_pickle=False)
    return metrics_arrays(d["S"].astype(float), d["L"].astype(int), d["V"].astype(float))


def metrics_arrays(S, L, V):
    T = S.shape[0]
    R = regions(S, L)
    dl = np.where(np.isfinite(V), np.maximum(0.0, 1.0 - V / FREE_V), 0.0)
    fin = np.isfinite(S)
    # appearance events (slot reuse safe): finite now and absent before, or a backward jump > 500 m
    prev = np.vstack([np.zeros((1, S.shape[1]), bool), fin[:-1]])
    back = np.vstack([np.zeros((1, S.shape[1]), bool), fin[1:] & fin[:-1] & (S[1:] < S[:-1] - 500.0)])
    app = fin & (~prev | back)
    app[0] = False
    up_app = (app & (S < -2400)).sum(1)
    ramp_app = (app & (L == 2) & (S < -300)).sum(1)
    x_m1 = crossings(S, L, M1)
    x_0 = crossings(S, L, 0.0)
    x_up = crossings(S, L, UPN)
    rm = ((L[:-1] == 2) & np.isin(L[1:], (0, 1)) & np.isfinite(S[1:])).sum(1)
    tail = tail_series(S, L, V)
    slow_mrg = ((R["merge"] | (R["up_near"] & (S >= -200))) & (V < SLOW)).sum(1)
    res = {"n_ids": int(S.shape[1])}
    for i in range(5):
        a, b = EDGES[i], EDGES[i + 1]
        sl = slice(a, b)  # seconds a..b-1 (delay on the 1 s grid, as the metric)
        row = {}
        for r, m in R.items():
            row[f"delay_{r}"] = float((dl[sl] * m[sl]).sum())
            row[f"n_{r}"] = float(m[sl].sum(1).mean())
            row[f"slow_{r}"] = float((m[sl] & (V[sl] < SLOW)).sum(1).mean())
            vv = np.where(m[sl], V[sl], np.nan)
            row[f"v_{r}"] = float(np.nanmean(vv)) if np.isfinite(vv).any() else np.nan
        row["q_m1"] = float(x_m1[a:b].sum() * 60.0)          # veh/h over the 60 s interval
        row["q_0"] = float(x_0[a:b].sum() * 60.0)
        row["q_up"] = float(x_up[a:b].sum() * 60.0)
        row["q_rampmerge"] = float(rm[a:b].sum() * 60.0)
        row["q_upentry"] = float(up_app[a:b].sum() * 60.0)
        row["q_rampentry"] = float(ramp_app[a:b].sum() * 60.0)
        qmask = slow_mrg[a:b] >= 3
        row["queue_share"] = float(qmask.mean())
        row["q_m1_queued"] = float(x_m1[a:b][qmask].sum() / max(qmask.sum(), 1) * 3600.0) if qmask.sum() >= 10 else np.nan
        tl = tail[a:b]
        row["spill"] = float(np.nanmean(np.where(np.isfinite(tl), -tl, 0.0)))  # m upstream of s=0 (0 if no queue)
        row["spill_max"] = float(np.nanmax(np.where(np.isfinite(tl), -tl, 0.0)))
        res[i] = row
    # backward front speed: slope of the tail while it grows (least squares over seconds with a queue), m/s
    t = np.arange(T, dtype=float)
    ok = np.isfinite(tail)
    res["tail"] = tail
    res["wave_c"] = wave_speed(S, L, V)
    res["front_speed"] = float(np.polyfit(t[ok], tail[ok], 1)[0]) if ok.sum() > 30 else np.nan
    return res


def load_all(dirs):
    files = []
    for dd in dirs:
        files += glob.glob(os.path.join(dd, "p*_*.npz"))
    out = {}
    for f in sorted(files):
        m = re.match(r"p(\d+)_(ref|act)_(\w+?)(?:_k(\d+))?\.npz$", os.path.basename(f))
        if not m:
            continue
        idx, arm, var = int(m.group(1)), m.group(2), m.group(3)
        try:
            out[(idx, arm, var)] = (f, metrics(f))
        except Exception as e:  # noqa: BLE001
            print("bad", f, e, file=sys.stderr)
    return out


if __name__ == "__main__":
    import pickle
    res = load_all(sys.argv[1:-1])
    pickle.dump(res, open(sys.argv[-1], "wb"))
    print(len(res))
