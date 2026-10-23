"""S-BASEDIAG Q2 (recorded SUMO TRAIN data only, numpy, no model): does treating the waiting ramp car
as a virtual leader of its lane-1 target follower explain the follower's recorded acceleration?

For every lane-1 HDV row (1 s sampling, t >= 300 s, continuous id, lane 1 at t and t+1):
  * own lane-1 leader (gap, speed), the nearest invisible (junction, one-hot = 0) vehicle ahead;
  * the geometric ramp partner (the WM's pairing: the row is the nearest lane-1 car with s <= s_ramp
    of some lane-2 car; nearest such ramp car), its bumper gap, speed, distance to lane end, time to
    its merge (diagnostic only);
  * IDM with the own leader (``a_own``) and with the partner as a leader at its current position
    (``a_virt``), population parameter sets (WM physics default; SUMO-population medians);
  * the recorded acceleration of the next row.
Writes ``<out>.npz`` (row table) for the analysis script.  Diagnosis only: nothing here is a model
rule or a fitted model parameter.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

END = 75.44


def idm(v, vl, gap, a, b, T, s0, v0=33.0, delta=4, amin=-9.0, amax=3.0):
    closing = v - vl
    desired = s0 + np.clip(v * T + v * closing / (2.0 * np.sqrt(a * b)), 0.0, None)
    acc = a * (1.0 - (v / v0) ** delta - (desired / np.maximum(gap, 0.3)) ** 2)
    return np.clip(acc, amin, amax)


PSETS = {  # (a, b, T, s0)
    "wm": (2.49, 4.2, 1.135, 2.6),          # WM physics default (physics.idm())
    "sumo": (2.6, 2.02, 1.19, 2.0),         # LCSTATE.md section 9 population medians (accel = SUMO default 2.6)
}


def episode(path):
    d = Path(path)
    kin = np.load(d / "veh_kin.npy", mmap_mode="r")
    mask = np.load(d / "veh_mask.npy", mmap_mode="r")
    cav = np.load(d / "is_cav.npy", mmap_mode="r")
    vid = np.load(d / "vid.npy", mmap_mode="r")
    T = kin.shape[0]
    rows = []
    onehot_all = np.asarray(kin[..., 1:4])
    lane_all = np.where(onehot_all.sum(-1) > 0, onehot_all.argmax(-1), -1)
    pres_all = np.asarray(mask) > 0
    # time of merge per (vid): first step a lane-2 vid appears in lane 1
    merge_step = {}
    for t in range(1, T):
        m = pres_all[t] & pres_all[t - 1] & (lane_all[t] == 1) & (lane_all[t - 1] == 2) \
            & (vid[t] == vid[t - 1])
        for k in np.nonzero(m)[0]:
            merge_step.setdefault((int(vid[t, k])), t)
    for t in range(600, T - 1, 2):
        k_t = np.asarray(kin[t])
        pres = pres_all[t]
        lane = lane_all[t]
        s, v, a, L = k_t[:, 0], k_t[:, 4], k_t[:, 5], k_t[:, 6]
        main = pres & (lane == 1)
        ramp = pres & (lane == 2)
        inv = pres & (lane == -1)
        cont = main & pres_all[t + 1] & (lane_all[t + 1] == 1) & (vid[t + 1] == vid[t]) & (cav[t] == 0)
        if not cont.any():
            continue
        ds = s[None, :] - s[:, None]                      # [i, j] = s_j - s_i
        big = np.inf
        lead_key = np.where(main[None, :] & (ds > 0), ds, big)
        li = lead_key.argmin(-1)
        has_l = np.isfinite(lead_key.min(-1))
        gap_l = np.where(has_l, ds[np.arange(len(s)), li] - L[li], 1e3)
        v_l = np.where(has_l, v[li], v)
        inv_key = np.where(inv[None, :] & (ds > 0), ds, big)
        inv_d = inv_key.min(-1)
        # partner: target follower of each ramp car j = nearest lane-1 i with s_i <= s_j
        beh = np.where(ramp[None, :] & main[:, None] & (ds >= 0), ds, big)   # [i, j]
        f_of_j = beh.argmin(0)
        has_f = np.isfinite(beh.min(0))
        own = np.zeros_like(beh, bool)
        jj = np.nonzero(has_f)[0]
        own[f_of_j[jj], jj] = True
        pkey = np.where(own, ds, big)
        pj = pkey.argmin(-1)
        has_p = np.isfinite(pkey.min(-1)) & main
        any_wait = bool((ramp & (s >= 0) & (s <= END)).any())
        near = ((ramp & (s >= 0) & (s <= END))[None, :] & (ds <= 30) & (ds >= -20)).any(-1)
        for i in np.nonzero(cont)[0]:
            j = pj[i]
            hp = bool(has_p[i])
            ms = merge_step.get(int(vid[t, j]), None) if hp else None
            tm = (ms - t) * 0.5 if (ms is not None and ms > t) else np.nan
            rows.append((s[i], v[i], a[i], kin[t + 1, i, 5], gap_l[i], v_l[i], float(has_l[i]),
                         inv_d[i] if np.isfinite(inv_d[i]) else 1e3, float(hp),
                         s[j] - L[j] - s[i] if hp else np.nan, v[j] if hp else np.nan,
                         a[j] if hp else np.nan, END - s[j] if hp else np.nan, tm,
                         float((not hp) and any_wait and not near[i]), t * 0.5))
    return np.asarray(rows, np.float32)


COLS = ("s", "v", "a_now", "rec", "gap_l", "v_l", "has_l", "inv_d", "has_p", "gap_r", "v_r", "a_r",
        "dend", "tmerge", "c3", "time")


def main():
    root, out, workers = sys.argv[1], sys.argv[2], int(sys.argv[3])
    eps = []
    for src in ("v5_native", "genrand_plain"):
        doc = json.loads((Path(root) / src / "C4_EPISODES_MANIFEST.json").read_text())
        eps += [str(Path(root) / src / e["dir"]) for e in doc["episodes"]]
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(episode, eps))
    ep_id = np.concatenate([np.full(len(p), i, np.int32) for i, p in enumerate(parts)])
    tab = np.concatenate(parts)
    np.savez_compressed(out, ep=ep_id, **{c: tab[:, i] for i, c in enumerate(COLS)})
    print("episodes", len(eps), "rows", len(tab))


if __name__ == "__main__":
    main()
