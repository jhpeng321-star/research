"""S-RAMPSPEED (R16) initialization of the learned ramp speed-adaptation scalars from TRAIN rows.

Recorded SUMO TRAIN-role episodes only (``v5_native``; a world is TRAIN only if it is TRAIN in
every v5 manifest, ``wms.data.c4_pairs.world_roles``); evaluation seeds 964000-964199 and restricted
paths are refused.  No WM rollout, no SUMO, no vType/truth files.

Rows: every ramp HDV (lane 2, ``0 <= s <= 75.44`` m, not a CAV) at 1 s sampling after the 300 s
warm-up whose identity is present on the next row, with a lane-1 vehicle at or ahead of its front.
Target: the recorded acceleration of the next row.  Physics: the WM baseline for a ramp car (IDM
with the same-lane leader or the standing lane-end obstacle, bounded by -b) and the R16 virtual
target-lane leader term, both with the driver population CENTRE of the 13E checkpoint (learned
encoder centre, physical mode; no per-driver labels).  Fitted by grid least squares:

    pred = a_base + sigmoid(omega) * sigmoid((D - d_end) / s0) * min(a_v(kappa) - a_base, 0)

The minimiser is written as the module INITIALIZATION (``src/wms/model/ramp_speed_adapt_init.json``);
training re-learns all three scalars.  Also writes descriptive TRAIN statistics.

  python ramp_adapt_init_fit.py --data-root L5/data --ckpt CK --out OUT.json [--workers 4]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

END, START, DT = 75.44, 0.0, 0.5
ROW0 = 600                      # 300 s warm-up (0.5 s rows)
FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1", "hetdata_truth",
             "truth", "vtype")
EXCLUDED = range(964000, 964200)
V5 = ("v5_native", "genrand_plain", "genrand_branch", "pairs300")
ACCEL_MIN, ACCEL_MAX = -9.0, 3.0


def guard(path) -> str:
    s = str(path).replace("\\", "/").lower()
    if any(t in s for t in FORBIDDEN):
        raise SystemExit(f"refusing restricted path {path}")
    if any(int(n) in EXCLUDED for n in re.findall(r"\d+", s) if len(n) == 6):
        raise SystemExit(f"refusing evaluation-seed path {path}")
    return str(path)


def idm(v, vl, gap, v0, T, s0, a, b):
    closing = v - vl
    desired = s0 + np.clip(v * T + v * closing / (2.0 * np.sqrt(a * b)), 0.0, None)
    acc = a * (1.0 - (v / v0) ** 4 - (desired / np.maximum(gap, 0.3)) ** 2)
    return np.maximum(np.clip(acc, ACCEL_MIN, ACCEL_MAX), -b)


def episode(args):
    path, = args
    d = Path(guard(path))
    kin = np.load(d / "veh_kin.npy", mmap_mode="r")
    mask = np.load(d / "veh_mask.npy", mmap_mode="r") > 0
    cav = np.load(d / "is_cav.npy", mmap_mode="r")
    vid = np.load(d / "vid.npy", mmap_mode="r")
    T = kin.shape[0]
    out = []
    merges = []
    for t in range(ROW0, T - 2, 2):
        k = np.asarray(kin[t])
        pres = np.asarray(mask[t])
        oh = k[:, 1:4]
        lane = np.where(oh.sum(-1) > 0, oh.argmax(-1), -1)
        s, v, L = k[:, 0], k[:, 4], k[:, 6]
        nxt_same = np.asarray(mask[t + 1]) & (np.asarray(vid[t + 1]) == np.asarray(vid[t]))
        ramp = pres & (lane == 2) & (np.asarray(cav[t]) == 0) & (s >= START) & (s <= END) & nxt_same
        if not ramp.any():
            continue
        ds = s[None, :] - s[:, None]                         # [i, j] = s_j - s_i
        n = len(s)
        eye = np.eye(n, dtype=bool)
        own = pres[None, :] & (lane[None, :] == lane[:, None]) & (ds > 0) & ~eye
        key = np.where(own, ds, np.inf)
        oi = key.argmin(-1)
        has_o = np.isfinite(key.min(-1))
        gap_o = np.where(has_o, ds[np.arange(n), oi] - L[oi], 1e3)
        v_o = np.where(has_o, v[oi], v)
        end_gap = np.maximum(END - s, 0.3)
        obstacle = end_gap < gap_o
        gap_b = np.where(obstacle, end_gap, gap_o)
        v_b = np.where(obstacle, 0.0, v_o)
        tgt = pres[None, :] & (lane[None, :] == 1) & (ds >= 0) & ~eye
        tkey = np.where(tgt, ds, np.inf)
        ti = tkey.argmin(-1)
        has_t = np.isfinite(tkey.min(-1))
        gap_t = ds[np.arange(n), ti] - L[ti]
        rec = np.asarray(kin[t + 2, :, 5])                  # accel over the next 1 s row pair
        rec1 = np.asarray(kin[t + 1, :, 5])
        lane_next = np.asarray(kin[t + 2, :, 1:4]).argmax(-1)
        for i in np.nonzero(ramp)[0]:
            out.append((s[i], v[i], gap_b[i], v_b[i], float(has_t[i]),
                        gap_t[i] if has_t[i] else np.nan, v[ti[i]] if has_t[i] else np.nan,
                        0.5 * (rec[i] + rec1[i]), float(lane_next[i] != 2)))
    # descriptive: merge events (lane 2 -> 1) of HDVs after warm-up
    oh_all = np.asarray(kin[:, :, 1:4])
    lane_all = np.where(oh_all.sum(-1) > 0, oh_all.argmax(-1), -1)
    m_all = np.asarray(mask)
    vid_all = np.asarray(vid)
    cav_all = np.asarray(cav)
    for t in range(ROW0, T - 1):
        ev = (m_all[t] & m_all[t + 1] & (lane_all[t] == 2) & (lane_all[t + 1] == 1)
              & (vid_all[t + 1] == vid_all[t]) & (cav_all[t] == 0))
        for i in np.nonzero(ev)[0]:
            merges.append((END - float(kin[t, i, 0]), float(kin[t, i, 4])))
    return np.asarray(out, np.float64).reshape(-1, 9), np.asarray(merges, np.float64).reshape(-1, 2)


def population_centre(ckpt: str) -> dict:
    import torch
    payload = torch.load(guard(ckpt), map_location="cpu", weights_only=True)
    sd = payload["model"]
    lo, hi = sd["driver_encoder.physical_lower"], sd["driver_encoder.physical_upper"]
    vals = (lo + (hi - lo) * torch.sigmoid(sd["driver_encoder.centre_logits"])).tolist()
    return dict(zip(("v0", "T", "s0", "a", "b"), vals)), hashlib.sha256(Path(ckpt).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if not 1 <= a.workers <= 4:
        raise SystemExit("workers 1..4")
    if os.name == "posix":
        os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
    root = Path(guard(a.data_root))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from wms.data.c4_pairs import world_roles
    docs = [json.loads((root / s / "C4_EPISODES_MANIFEST.json").read_text()) for s in V5]
    roles, _ = world_roles(docs)
    eps = [str(root / "v5_native" / e["dir"]) for e in docs[0]["episodes"]
           if roles.get(e["world"]) == "train" and int(e["seed"]) not in EXCLUDED]
    centre, ck_sha = population_centre(a.ckpt)
    with ProcessPoolExecutor(a.workers) as ex:
        parts = list(ex.map(episode, [(p,) for p in eps], chunksize=4))
    rows = np.concatenate([p[0] for p in parts])
    merges = np.concatenate([p[1] for p in parts])
    s, v, gap_b, v_b, has_t, gap_t, v_t, rec, merged = rows.T
    c = centre
    a_base = idm(v, v_b, gap_b, c["v0"], c["T"], c["s0"], c["a"], c["b"])
    sel = (has_t > 0) & np.isfinite(rec)
    d_end = END - s
    base_err = rec - a_base
    best = None
    grid = {"lookahead_m": np.arange(5.0, 155.0, 5.0), "strength": np.arange(0.05, 1.0, 0.05),
            "headway_scale": np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0])}
    for kappa in grid["headway_scale"]:
        a_v = idm(v[sel], v_t[sel], gap_t[sel], c["v0"], kappa * c["T"], c["s0"], c["a"], c["b"])
        dlt = np.minimum(a_v - a_base[sel], 0.0)
        for D in grid["lookahead_m"]:
            g = 1.0 / (1.0 + np.exp(-(D - d_end[sel]) / c["s0"]))
            for w in grid["strength"]:
                err = rec[sel] - (a_base[sel] + w * g * dlt)
                mse = float(np.mean(err ** 2))
                if best is None or mse < best[0]:
                    best = (mse, float(D), float(w), float(kappa))
    mse0 = float(np.mean(base_err[sel] ** 2))
    D, w, kappa = best[1:]
    # bootstrap by episode would need ids; report row-level split-half stability instead
    half = np.arange(sel.sum()) % 2 == 0
    a_v = idm(v[sel], v_t[sel], gap_t[sel], c["v0"], kappa * c["T"], c["s0"], c["a"], c["b"])
    g = 1.0 / (1.0 + np.exp(-(D - d_end[sel]) / c["s0"]))
    pred = a_base[sel] + w * g * np.minimum(a_v - a_base[sel], 0.0)
    rel = v[sel] - v_t[sel]
    bins = [0, 15, 30, 45, 60, 75.44]
    by = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (d_end[sel] >= lo) & (d_end[sel] < hi)
        by[f"{lo:.0f}-{hi:.0f}m"] = {
            "n": int(m.sum()), "rec_mean": float(rec[sel][m].mean()) if m.any() else None,
            "base_mean": float(a_base[sel][m].mean()) if m.any() else None,
            "fit_mean": float(pred[m].mean()) if m.any() else None,
            "v_minus_vlead1_med": float(np.median(rel[m])) if m.any() else None,
            "rmse_base": float(np.sqrt(np.mean((rec[sel][m] - a_base[sel][m]) ** 2))) if m.any() else None,
            "rmse_fit": float(np.sqrt(np.mean((rec[sel][m] - pred[m]) ** 2))) if m.any() else None}
    doc = {"study": "S-RAMPSPEED R16 init fit (TRAIN rows, numpy, no rollout)",
           "checkpoint_sha256": ck_sha, "population_centre": centre, "episodes": len(eps),
           "rows_ramp_zone": int(len(rows)), "rows_with_lane1_leader": int(sel.sum()),
           "fit": {"lookahead_m": D, "strength": w, "headway_scale": kappa},
           "rmse_base": float(np.sqrt(mse0)), "rmse_fit": float(np.sqrt(best[0])),
           "rmse_fit_half0": float(np.sqrt(np.mean((rec[sel][half] - pred[half]) ** 2))),
           "rmse_base_half0": float(np.sqrt(np.mean(base_err[sel][half] ** 2))),
           "by_distance_to_end": by,
           "sumo_train_merges": {"n": int(len(merges)),
                                 "dist_before_end_q": np.quantile(merges[:, 0], [.1, .25, .5, .75, .9]).tolist(),
                                 "share_standstill_v_lt_0p5": float((merges[:, 1] < 0.5).mean()),
                                 "speed_med": float(np.median(merges[:, 1]))},
           "grid": {k: [float(x) for x in g_] for k, g_ in grid.items()}}
    Path(a.out).write_text(json.dumps(doc, indent=1))
    print(json.dumps({k: doc[k] for k in ("fit", "rmse_base", "rmse_fit", "rows_with_lane1_leader",
                                          "sumo_train_merges")}, indent=1))


if __name__ == "__main__":
    main()
