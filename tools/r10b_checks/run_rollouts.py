"""R10B checks, rollout phase: per (state, lane seed) roll every arm on CPU and save compact per-run tables to SHM.

  python run_rollouts.py --repo REPO --states S.json.gz --var r9=CK --var r10b=CK --var headzero=CK \
      --workers 24 --k 2 --steps 240 --out DIR [--limit N]

Per run and arm (``DIR/<arm>/s<idx>_k<sample>.npz``): lane-1 HDV rows (diag_yield_extract.rows_from_arrays), merge-event
windows (events_from_arrays, +-S-YIELD definitions), the courtesy head table (rows where the head applies), the exact-zero
check on every other row, partner-switch jerk and a CAV-replay error.  ``sumo`` = the recorded future of the same t0 roster.

R12 seed replication: ``--seed-offset N`` (default: absent = unchanged, process-global torch RNG) rolls sample k with
seed / lane seed k + N AND seeds the global torch RNG (the first-passage merge draws' stream keys) with
seed_of(state, k + N) before every arm's roll: reproducible, common random numbers across arms.  File names keep the
sample index k, so a replication pass must write to its own ``--out``.
R12 ``--lane-changes`` (default off = unchanged files): every run file (and the ``sumo`` reference) gets ``lc01`` =
[HDV mainline lane changes 0<->1, all-vehicle 0<->1 changes, HDV vehicle-steps] from the full lane arrays."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import DT, add_paths  # noqa: E402

PRE, POST = 40, 24
SPECS = []
OUTDIR = ""
REPO = ""
SEED_OFFSET = None
LANE_CHANGES = False


def lane_changes_01(lane, mask, is_cav):
    """[HDV 0<->1 lane changes, all-vehicle 0<->1 changes, HDV vehicle-steps] of one trajectory [T, n]."""
    li = np.where(np.isfinite(lane), np.rint(lane), -1)
    m = np.asarray(mask, bool)
    ok = m[:-1] & m[1:] & np.isin(li[:-1], (0, 1)) & np.isin(li[1:], (0, 1)) & (li[:-1] != li[1:])
    hdv = ~np.asarray(is_cav, bool)[None, :]
    return np.array([float((ok & hdv).sum()), float(ok.sum()), float((m & hdv).sum())], np.float64)


def seed_of(idx: int, seed: int) -> int:
    """Global torch seed of one (state, sample seed) roll (R12 --seed-offset; identical for every arm)."""
    return (1_000_003 * int(idx) + 7919 * int(seed) + 12345) % (2 ** 31 - 1)


def _init(specs, repo, outdir, seed_offset=None, lane_changes=False):
    global SPECS, OUTDIR, REPO, SEED_OFFSET, LANE_CHANGES
    SPECS, OUTDIR, REPO = specs, outdir, repo
    LANE_CHANGES = bool(lane_changes)
    SEED_OFFSET = None if seed_offset is None else int(seed_offset)
    add_paths(repo)
    import rollout_worker as rw
    rw.init_arms(specs, "off")


def summarize_traj(pos, lane, mask, v, is_cav, t0, idx, pen, dem):
    import rollout_worker as rw
    from diag_yield_events import events_from_arrays
    from diag_yield_extract import rows_from_arrays
    s, vv, a, ln, m, vid, cav, tt = rw.traj_arrays(pos, lane, mask, v, is_cav, t0)
    rows = rows_from_arrays(s, vv, a, ln, m, vid, cav, tt, idx, pen, dem, tt[0] + DT)
    if len(rows):
        keep = (rows[:, 24] == 1) & (rows[:, 4] == 0)
        rows = rows[keep]
    ev, me = events_from_arrays(s, vv, a, ln, m, vid, cav, tt, idx, pen, dem, tt[0], PRE, POST)
    return rows.astype(np.float32), ev.astype(np.float32), me.astype(np.float64), a, ln, m, cav


def head_tables(res, a_model_unused, physics):
    import rollout_worker as rw
    calls = res["calls"]
    n = res["n"]
    T = res["pos"].shape[0]
    has, partner = rw.pairs_per_step(res["pos"], res["lane"], res["mask"], res["v"], physics)
    acc = np.full((T, n), np.nan, np.float32)
    tab, nonapp_max, n_calls = [], 0.0, 0
    for c in calls:
        k = c["k"]
        if k < 0 or k >= T:
            continue
        n_calls += 1
        acc[k] = c["accel"][:n]
        if "offset" in c:
            app = c["applies"][:n]
            off = c["offset"][:n]
            if (~app).any():
                nonapp_max = max(nonapp_max, float(np.abs(off[~app]).max()))
            sl = np.flatnonzero(app)
            if len(sl):
                x = c["x"][:n][sl][:, :10]   # the 10 round-10b inputs (a version-2 head appends 3; layout kept)
                tab.append(np.column_stack([np.full(len(sl), k), sl, off[sl], c["pre"][:n][sl], c["accel"][:n][sl],
                                            partner[k][sl], x]))
    tab = np.concatenate(tab).astype(np.float32) if tab else np.zeros((0, 16), np.float32)
    return tab, nonapp_max, n_calls, acc, has, partner


def jerk_stats(acc, has, partner, lane, mask, is_cav):
    """|d accel|/dt of the model acceleration: at ramp-partner switches vs all lane-1 HDV rows."""
    T, n = acc.shape
    hdv = ~np.asarray(is_cav, bool)[None, :]
    ok = mask[:-1] & mask[1:] & (np.rint(lane[:-1]) == 1) & (np.rint(lane[1:]) == 1) & hdv
    da = np.abs(acc[1:] - acc[:-1]) / DT
    allj = da[ok & np.isfinite(da)]
    sw = ok & has[:-1] & has[1:] & (partner[:-1] != partner[1:]) & np.isfinite(da)
    return (np.asarray(da[sw], np.float32),
            np.asarray(np.percentile(allj, [50, 95, 99]) if len(allj) else [np.nan] * 3, np.float32),
            int(len(allj)))


def work(payload):
    import rollout_worker as rw
    idx, rec, sample, steps = payload
    out_msgs = []
    first = True
    for name, _ in SPECS:
        if SEED_OFFSET is None:
            res = rw.roll_once(name, rec, sample, steps)
        else:
            import torch
            torch.manual_seed(seed_of(idx, sample + SEED_OFFSET))
            res = rw.roll_once(name, rec, sample + SEED_OFFSET, steps)
        p, _cap = rw.G["p"][name]
        physics = p.model.physics
        pen = float(res["pen"] or np.nan)
        dem = float(res["dem"] or np.nan)
        rows, ev, me, a, ln, m, cav = summarize_traj(res["pos"], res["lane"], res["mask"], res["v"], res["is_cav"],
                                                    res["t0"], idx, pen, dem)
        tab, nonapp_max, n_calls, acc, has, partner = head_tables(res, None, physics)
        jsw, jall, njall = jerk_stats(acc, has, partner, res["lane"], res["mask"], res["is_cav"])
        cav_sel = res["is_cav"][None, :] & res["rec_mask"] & res["mask"]
        cav_err = float(np.abs(res["pos"] - res["rec_pos"])[cav_sel].mean()) if cav_sel.any() else float("nan")
        d = os.path.join(OUTDIR, name)
        os.makedirs(d, exist_ok=True)
        extra = {"lc01": lane_changes_01(res["lane"], res["mask"], res["is_cav"])} if LANE_CHANGES else {}
        np.savez_compressed(os.path.join(d, f"s{idx:03d}_k{sample}.npz"), rows=rows, ev=ev, me=me, head=tab,
                            jerk_switch=jsw, jerk_all=jall, n_jerk_all=njall,
                            misc=np.array([nonapp_max, n_calls, steps, cav_err, res["pos"].shape[1]], np.float64),
                            **extra)
        out_msgs.append((name, len(rows), len(ev), len(tab), nonapp_max, n_calls, cav_err))
        if first and sample == 0:
            rows, ev, me, *_ = summarize_traj(res["rec_pos"], res["rec_lane"], res["rec_mask"], res["rec_v"],
                                              res["is_cav"], res["t0"], idx, pen, dem)
            d = os.path.join(OUTDIR, "sumo")
            os.makedirs(d, exist_ok=True)
            np.savez_compressed(os.path.join(d, f"s{idx:03d}_k0.npz"), rows=rows, ev=ev, me=me,
                                head=np.zeros((0, 16), np.float32), jerk_switch=np.zeros(0, np.float32),
                                jerk_all=np.zeros(3, np.float32), n_jerk_all=0, misc=np.zeros(5),
                                **({"lc01": lane_changes_01(res["rec_lane"], res["rec_mask"], res["is_cav"])}
                                   if LANE_CHANGES else {}))
        first = False
    return idx, sample, out_msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--states", required=True)
    ap.add_argument("--var", action="append", default=[])
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1, help="use every Nth state")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed-offset", type=int, default=None,
                    help="R12 seed replication: sample k uses seed k + offset and a seeded torch RNG (default: unchanged)")
    ap.add_argument("--lane-changes", action="store_true",
                    help="R12 read: store lc01 (mainline 0<->1 lane changes) in every run file (default off)")
    a = ap.parse_args()
    specs = [tuple(v.split("=", 1)) for v in a.var]
    with gzip.open(a.states, "rt") as f:
        recs = json.load(f)
    recs = recs[::a.stride]
    if a.limit:
        recs = recs[: a.limit]
    tasks = [(i, r, s, a.steps) for i, r in enumerate(recs) for s in range(a.k)]
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    summary = []
    if a.workers <= 1:
        _init(specs, a.repo, a.out, a.seed_offset, a.lane_changes)
        for t in tasks:
            summary.append(work(t))
            print(summary[-1], flush=True)
    else:
        from multiprocessing import Pool
        with Pool(a.workers, initializer=_init, initargs=(specs, a.repo, a.out, a.seed_offset, a.lane_changes)) as pool:
            for j, r in enumerate(pool.imap_unordered(work, tasks, chunksize=1)):
                summary.append(r)
                if (j + 1) % 8 == 0:
                    print(f"{j+1}/{len(tasks)} {time.time()-t0:.0f}s", flush=True)
    done = {"n_states": len(recs), "k": a.k, "steps": a.steps, "specs": specs, "seconds": time.time() - t0}
    if a.seed_offset is not None:
        done["seed_offset"] = a.seed_offset
    json.dump(done, open(os.path.join(a.out, "ROLLOUT_DONE.json"), "w"))
    print(f"done {time.time()-t0:.0f}s runs {len(summary)}", flush=True)


if __name__ == "__main__":
    main()
