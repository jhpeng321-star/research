"""S-YIELD Q3: per-vehicle pre-merge-zone car-following features (behaviour only; no vType / truth files).
For every HDV (episode, vid) that is on lane 0/1 with s in [-1500, -60) store aggregated features of its own
past car-following: time headway stats, accel/brake style, speed level; then (separately) the encounter yield score.
"""
import glob, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, "tools")
from diag_yield_extract import load, lanes, ROOT, OUT, L

FEATS = ["n", "thw_med", "thw_p20", "apos_p90", "aneg_p10", "v_p90", "v_mean", "dvgain", "jerk_sd", "tte_first", "tfirst"]


def one(args):
    path, ei = args
    k, m, vid, cav, tt, meta = load(path)
    lane = lanes(k); s = k[..., 0].astype(np.float64); v = k[..., 4].astype(np.float64); a = k[..., 5].astype(np.float64)
    T, K = m.shape
    rows = []
    for ti in range(T):
        if tt[ti] < 120:
            continue
        for ln in (0, 1):
            idx = np.flatnonzero(m[ti] & (lane[ti] == ln))
            if len(idx) == 0:
                continue
            o = np.argsort(s[ti, idx]); idx = idx[o]; ss = s[ti, idx]
            gap = np.full(len(idx), np.nan); vl = np.full(len(idx), np.nan)
            gap[:-1] = ss[1:] - L - ss[:-1]; vl[:-1] = v[ti, idx[1:]]
            sel = (ss >= -1500) & (ss < -60) & (~cav[ti, idx])
            for j in np.flatnonzero(sel):
                f = idx[j]
                rows.append((ei, vid[ti, f], tt[ti], ss[j], v[ti, f], a[ti, f], gap[j], vl[j]))
    return np.array(rows, dtype=np.float64)


def main():
    paths = []
    for src in ("v5_native", "genrand_plain"):
        paths += sorted(glob.glob(f"{ROOT}/{src}/episodes/*"))
    with ProcessPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 16) as ex:
        res = list(ex.map(one, [(p, i) for i, p in enumerate(paths)]))
    tab = np.concatenate([r for r in res if len(r)])
    np.save(OUT + "/driver_rows.npy", tab.astype(np.float32))
    print(tab.shape)


if __name__ == "__main__":
    main()
