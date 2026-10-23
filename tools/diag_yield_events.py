"""S-YIELD: event-aligned traces of merge events (TRAIN recorded episodes). Analysis-only.

For each ramp car R that goes lane 2 -> lane 1 (t_m = first lane-1 row, t_m >= 300 s) store a window of
OFFS rows (0.5 s) around t_m for: R (s, v, a), the lane-1 follower F* at t_m (nearest lane-1 vehicle
behind R, s_F <= s_R), F*'s own lane-1 leader (gap, v), and the ramp-lane vehicles nearby.
Fields per (event, offset): see FIELDS.
"""
from __future__ import annotations
import glob, json, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, "tools")
from diag_yield_extract import load, lanes, END, L, ROOT, OUT

PRE, POST = 80, 24            # rows before / after t_m
OFFS = np.arange(-PRE, POST + 1)
FIELDS = ["sF", "vF", "aF", "laneF", "gapL", "vL", "hasL", "sR", "vR", "aR", "laneR", "gapF", "waitR"]


def events_of(args):
    path, ei = args
    k, m, vid, cav, tt, meta = load(path)
    lane = lanes(k); s = k[..., 0].astype(np.float64); v = k[..., 4].astype(np.float64); a = k[..., 5].astype(np.float64)
    return events_from_arrays(s, v, a, lane, m, vid, cav, tt, ei, float(meta.get("penetration", np.nan)),
                              float(meta.get("demand_vph", np.nan)), 300.0, PRE, POST)


def events_from_arrays(s, v, a, lane, m, vid, cav, tt, ei, pen, dem, tmin, PRE, POST):
    OFFS = np.arange(-PRE, POST + 1)
    T, K = m.shape
    ev = []; meta_rows = []
    # find merges
    same = m[1:] & m[:-1] & (vid[1:] == vid[:-1])
    mg = (lane[:-1] == 2) & (lane[1:] == 1) & same
    ti_arr, k_arr = np.nonzero(mg)
    # zone entry time per vid
    first_zone = {}
    inzone = m & (lane == 2) & (s >= 0) & (s <= END + 1e-3)
    for ti in range(T):
        for kk in np.flatnonzero(inzone[ti]):
            first_zone.setdefault(int(vid[ti, kk]), tt[ti])
    # vid -> (slot) lookup per time via dict built lazily
    for ti0, kr in zip(ti_arr, k_arr):
        tm = ti0 + 1
        if tt[tm] < tmin or tm - PRE < 0 or tm + POST >= T:
            continue
        cand = np.flatnonzero(m[tm] & (lane[tm] == 1) & (np.arange(K) != kr) & (s[tm] <= s[tm, kr]))
        if len(cand) == 0:
            continue
        f = cand[np.argmax(s[tm, cand])]
        vf = vid[tm, f]; vr = vid[tm, kr]
        arr = np.full((len(OFFS), len(FIELDS)), np.nan)
        for oi, o in enumerate(OFFS):
            ti = tm + o
            # find slots of F* and R at row ti
            fs = np.flatnonzero(m[ti] & (vid[ti] == vf)); rs = np.flatnonzero(m[ti] & (vid[ti] == vr))
            if len(fs):
                ff = fs[0]
                arr[oi, 0] = s[ti, ff]; arr[oi, 1] = v[ti, ff]; arr[oi, 2] = a[ti, ff]; arr[oi, 3] = lane[ti, ff]
                lead = np.flatnonzero(m[ti] & (lane[ti] == 1) & (s[ti] > s[ti, ff]) & (np.arange(K) != ff))
                if len(lead) and lane[ti, ff] == 1:
                    j = lead[np.argmin(s[ti, lead])]
                    arr[oi, 4] = s[ti, j] - L - s[ti, ff]; arr[oi, 5] = v[ti, j]; arr[oi, 6] = 1
                else:
                    arr[oi, 6] = 0
            if len(rs):
                rr = rs[0]
                arr[oi, 7] = s[ti, rr]; arr[oi, 8] = v[ti, rr]; arr[oi, 9] = a[ti, rr]; arr[oi, 10] = lane[ti, rr]
                arr[oi, 12] = tt[ti] - first_zone.get(int(vr), tt[ti])
                if len(fs):
                    arr[oi, 11] = s[ti, rr] - L - s[ti, fs[0]]
        ev.append(arr)
        meta_rows.append((ei, tt[tm], vf, vr, float(cav[tm, f]), float(cav[tm, kr]), pen, dem))
    return np.array(ev).reshape(-1, len(OFFS), len(FIELDS)), np.array(meta_rows)


def main():
    paths = []
    for src in ("v5_native", "genrand_plain"):
        paths += sorted(glob.glob(f"{ROOT}/{src}/episodes/*"))
    with ProcessPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 16) as ex:
        res = list(ex.map(events_of, [(p, i) for i, p in enumerate(paths)]))
    ev = np.concatenate([r[0] for r in res if len(r[0])]); me = np.concatenate([r[1] for r in res if len(r[1])])
    np.savez(OUT + "/events.npz", ev=ev, meta=me, offs=OFFS, fields=np.array(FIELDS))
    print(ev.shape, me.shape)


if __name__ == "__main__":
    main()
