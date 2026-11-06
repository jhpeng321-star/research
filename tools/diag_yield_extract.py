"""S-YIELD (2026-10-23) analysis-only: per-(time, lane-1 vehicle) table with the nearest waiting
ramp car, from TRAIN recorded episodes (r9_train).  No SUMO, no model, no truth tables.

Columns of the sample table (float32):
  ep, t, vid, k, cav, s, v, a, gapL, dvL, vL (own lane-1 leader; gap in m, dvL = vL - v),
  hasL, dA (offset s_R - s_F of nearest waiting ramp car AHEAD or alongside-ahead, NaN none), vR, aR, waitR,
  dendR, tf (1 if F is the nearest lane-1 vehicle behind that ramp car), nRamp (ramp-lane cars
  within [s_F-30, s_F+150]), dB (offset s_F - s_R of nearest waiting ramp car behind, NaN none),
  nzone (number of waiting ramp cars on the merge edge), pen, dem.
Waiting ramp car = lane 2 and 0 <= s <= 75.44 (merge edge).  Positions use the pack frame
(junction rows have no lane and are ignored as neighbours).
"""
from __future__ import annotations
import glob, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT = "E:/agent-work/c4-v10pfit/data/r9_train"
OUT = "E:/agent-work/diag_yield_out"
END = 75.44
L = 5.0
COLS = ["ep", "t", "vid", "k", "cav", "s", "v", "a", "gapL", "dvL", "vL", "hasL", "dA", "vR", "aR",
        "waitR", "dendR", "tf", "nRamp", "dB", "nzone", "pen", "dem", "vidR", "lane"]


def load(path):
    k = np.load(path + "/veh_kin.npy"); m = np.load(path + "/veh_mask.npy").astype(bool)
    vid = np.load(path + "/vid.npy"); cav = np.load(path + "/is_cav.npy").astype(bool)
    t = np.load(path + "/time_s.npy"); meta = json.load(open(path + "/episode.json", encoding="utf-8-sig"))
    return k, m, vid, cav, t, meta


def lanes(k):
    oh = k[..., 1:4]
    return np.where(oh.sum(-1) > 0, oh.argmax(-1), -1)


def one(args):
    path, ei, tstart = args
    k, m, vid, cav, tt, meta = load(path)
    lane = lanes(k); s = k[..., 0].astype(np.float64); v = k[..., 4].astype(np.float64); a = k[..., 5].astype(np.float64)
    pen = float(meta.get("penetration", np.nan)); dem = float(meta.get("demand_vph", np.nan))
    return rows_from_arrays(s, v, a, lane, m, vid, cav, tt, ei, pen, dem, tstart)


def rows_from_arrays(s, v, a, lane, m, vid, cav, tt, ei, pen, dem, tstart):
    T, K = m.shape
    first_zone = {}
    rows = []
    for ti in range(T):
        if tt[ti] < tstart:
            # still track waiting start
            pass
        pm = m[ti]
        ramp_w = pm & (lane[ti] == 2) & (s[ti] >= 0) & (s[ti] <= END + 1e-3)
        ramp_any = pm & (lane[ti] == 2)
        for kk in np.flatnonzero(ramp_w):
            first_zone.setdefault(int(vid[ti, kk]), tt[ti])
        if tt[ti] < tstart or ti % 2:
            continue
        rw = np.flatnonzero(ramp_w); o2 = np.argsort(s[ti, rw]); rw = rw[o2]; sr = s[ti, rw]
        ra = np.flatnonzero(ramp_any); sra = np.sort(s[ti, ra])
        nz = len(rw)
        for ln in (1, 0):
            l1 = np.flatnonzero(pm & (lane[ti] == ln))
            if len(l1) == 0:
                continue
            o = np.argsort(s[ti, l1]); l1 = l1[o]; s1 = s[ti, l1]
            tfmap = {}
            if ln == 1:
                for j, r in enumerate(rw):
                    idx = np.searchsorted(s1, sr[j], side="right") - 1
                    if idx >= 0:
                        tfmap.setdefault(int(l1[idx]), []).append(j)
            for i, f in enumerate(l1):
                sf = s1[i]
                if sf < -150 or sf > END + 5:
                    continue
                if i + 1 < len(l1):
                    gl = s1[i + 1] - L - sf; vl = v[ti, l1[i + 1]]; hasl = 1.0
                else:
                    gl = np.nan; vl = np.nan; hasl = 0.0
                ja = np.searchsorted(sr, sf, side="left")
                if ja < nz:
                    r = rw[ja]; dA = sr[ja] - sf; vR = v[ti, r]; aR = a[ti, r]
                    wR = tt[ti] - first_zone[int(vid[ti, r])]; dend = END - sr[ja]; vidR = int(vid[ti, r])
                    tf = 1.0 if (int(f) in tfmap and ja in tfmap[int(f)]) else 0.0
                else:
                    dA = np.nan; vR = np.nan; aR = np.nan; wR = np.nan; dend = np.nan; tf = 0.0; vidR = -1
                dB = sf - sr[ja - 1] if ja - 1 >= 0 else np.nan
                nr = int(((sra >= sf - 30) & (sra <= sf + 150)).sum())
                rows.append((ei, tt[ti], vid[ti, f], f, float(cav[ti, f]), sf, v[ti, f], a[ti, f], gl,
                             (vl - v[ti, f]) if hasl else np.nan, vl, hasl, dA, vR, aR, wR, dend, tf, nr, dB, nz, pen, dem, vidR, ln))
    return np.array(rows, dtype=np.float64)


def main():
    tstart = 300.0
    paths = []
    for src in ("v5_native", "genrand_plain"):
        paths += sorted(glob.glob(f"{ROOT}/{src}/episodes/*"))
    jobs = [(p, i, tstart) for i, p in enumerate(paths)]
    with open(OUT + "/episode_index.json", "w") as f:
        json.dump(paths, f)
    with ProcessPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 16) as ex:
        res = list(ex.map(one, jobs))
    tab = np.concatenate([r for r in res if len(r)])
    np.save(OUT + "/samples.npy", tab.astype(np.float32))
    print(tab.shape, COLS)


if __name__ == "__main__":
    main()
