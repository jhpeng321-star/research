"""S-R14 D2 diagnosis: WHEN and WHERE ramp cars merge, WM vs SUMO (compact 1 s trajectories).

Per run file (COMBO300 / roll_r14 format) and the paired SUMO stream:
  merge events (lane 2 -> 0/1): position s at the last ramp tick, distance to the lane end, speed,
  time since the car entered the merge zone (first s >= 0 on lane 2), time since first seen on the ramp,
  standing time in the zone before the merge, accepted bumper gaps (new leader / new follower on lane 1);
  per-tick in-zone ramp rows: empirical 1 s merge hazard by distance-to-end bin, split by whether the
  lane-1 gap is "open" under one fixed DIAGNOSTIC reference (textbook gs, T=1.0 s, b=4.5 m/s^2; identical
  for WM and SUMO, never used by the model).
Diagnostic cutoffs here (bins, reference law) are analysis-only.

  python d2_merge_timing.py DIR [--arms 13E,13EL] [--tags ref,act]  -> markdown + DIR/d2_merge_timing.json
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np

M1, LV = 75.44, 5.0
BINS = [0.0, 15.0, 30.0, 45.0, 60.0, 75.44]       # distance to lane end (m), analysis bins
T_REF, B_REF = 1.0, 4.5                           # diagnostic reference law only


def load(path):
    d = np.load(path, allow_pickle=False)
    return d["S"].astype(float), d["L"].astype(int), d["V"].astype(float)


def need(vb, vf):
    return T_REF * vb + np.maximum(0.0, vb * vb / (2 * B_REF) - vf * vf / (2 * B_REF))


def lane1_gaps(S, L, V, t, j):
    on = np.isfinite(S[t]) & (L[t] == 1)
    s = S[t, j]
    ah, bh = np.flatnonzero(on & (S[t] > s)), np.flatnonzero(on & (S[t] <= s))
    gl = vl = gf = vf = np.nan
    if len(ah):
        k = ah[np.argmin(S[t, ah])]
        gl, vl = S[t, k] - LV - s, V[t, k]
    if len(bh):
        k = bh[np.argmax(S[t, bh])]
        gf, vf = s - LV - S[t, k], V[t, k]
    return gl, vl, gf, vf


def analyse(S, L, V):
    T, N = S.shape
    fin = np.isfinite(S)
    ev, rows = [], []
    for j in range(N):
        on2 = fin[:, j] & (L[:, j] == 2)
        if not on2.any():
            continue
        first_seen = int(np.flatnonzero(on2)[0])
        zone = on2 & (S[:, j] >= 0) & (S[:, j] <= M1)
        z0 = int(np.flatnonzero(zone)[0]) if zone.any() else None
        merged_at = None
        for t in range(T - 1):
            if on2[t] and fin[t + 1, j] and L[t + 1, j] in (0, 1):
                merged_at = t
                break
        tz = np.flatnonzero(zone)
        for t in tz:
            if merged_at is not None and t > merged_at:
                break
            if t + 1 >= T:
                break
            gl, vl, gf, vf = lane1_gaps(S, L, V, t, j)
            v = V[t, j]
            ok = ((np.isnan(gl) or gl >= need(v, vl)) and (np.isnan(gf) or gf >= need(vf, v)))
            rows.append((M1 - S[t, j], bool(ok), merged_at == t, v))
        if merged_at is not None and S[merged_at, j] >= 0:
            t = merged_at
            gl, vl, gf, vf = lane1_gaps(S, L, V, t, j)
            stand = int(((V[:, j] < 0.5) & zone)[: t + 1].sum())
            ev.append(dict(s=S[t, j], dist=M1 - S[t, j], v=V[t, j], t_zone=(t - z0) if z0 is not None else np.nan,
                           t_seen=t - first_seen, stand=stand, gl=gl, gf=gf, left_censored=first_seen == 0))
    return ev, rows


def summarize(evs, rows):
    out = {"n_merges": len(evs)}
    if evs:
        a = {k: np.array([e[k] for e in evs], float) for k in ("dist", "v", "t_zone", "t_seen", "stand", "gl", "gf")}
        for k, x in a.items():
            x = x[np.isfinite(x)]
            out[f"{k}_med"] = float(np.median(x)) if len(x) else np.nan
        out["share_dist_lt15"] = float((a["dist"] < 15).mean())
        out["share_dist_gt45"] = float((a["dist"] > 45).mean())
        out["share_standing_merge"] = float((a["v"] < 0.5).mean())
    if rows:
        r = np.array([(d, o, m, v) for d, o, m, v in rows], float)
        haz = {}
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            for ok in (1.0, 0.0):
                sel = (r[:, 0] >= lo) & (r[:, 0] < hi) & (r[:, 1] == ok)
                n = int(sel.sum())
                haz[f"{lo:.0f}-{hi:.0f}m_{'open' if ok else 'closed'}"] = (
                    [float(r[sel, 2].mean()), n] if n else [np.nan, 0])
        out["hazard_by_dist"] = haz
        out["open_share"] = float(r[:, 1].mean())
        out["rows"] = int(len(r))
    return out


def main():
    d = sys.argv[1]
    arms = sys.argv[sys.argv.index("--arms") + 1].split(",") if "--arms" in sys.argv else None
    tags = sys.argv[sys.argv.index("--tags") + 1].split(",") if "--tags" in sys.argv else ["ref", "act"]
    pool: dict = {}
    for f in sorted(glob.glob(os.path.join(d, "p*_*.npz"))):
        m = re.match(r"p(\d+)_(ref|act)_(\w+?)(?:_k(\d+))?\.npz$", os.path.basename(f))
        if not m or m.group(2) not in tags:
            continue
        arm = m.group(3)
        if arms and arm not in arms and arm != "sumo":
            continue
        ev, rows = analyse(*load(f))
        e, r = pool.setdefault(arm, ([], []))
        e.extend(ev)
        r.extend(rows)
    res = {arm: summarize(*v) for arm, v in pool.items()}
    with open(os.path.join(d, "d2_merge_timing.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    order = ["sumo"] + [a for a in res if a != "sumo"]
    keys = ["n_merges", "dist_med", "share_dist_gt45", "share_dist_lt15", "v_med", "share_standing_merge",
            "t_zone_med", "t_seen_med", "stand_med", "gl_med", "gf_med", "open_share"]
    print("| metric | " + " | ".join(order) + " |\n|---|" + "---|" * len(order))
    for k in keys:
        print(f"| {k} | " + " | ".join(f"{res[a].get(k, np.nan):.3g}" for a in order) + " |")
    hk = list(res["sumo"].get("hazard_by_dist", {}))
    for k in hk:
        print(f"| haz {k} | " + " | ".join(
            "{:.3f} (n={})".format(*res[a].get("hazard_by_dist", {}).get(k, [np.nan, 0])) for a in order) + " |")


if __name__ == "__main__":
    main()
