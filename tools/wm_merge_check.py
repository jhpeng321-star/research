"""WM-vs-SUMO merge-bottleneck check for one or more WM-S checkpoints (generalises
runs/active/C4-DIAG-20261016/wm_discharge.py).

Rolls every checkpoint 120 s from the recorded unseen-wall states (default
E:/agent-work/c4-v6merge/_ref/states_h60/wall_same.json; arms native and wall; CAVs teacher-forced
from the recorded arm; entries OFF) and compares, for HDVs of the t0 roster:

* merge-zone HDV mean speed (snapshot) in cells [-400,0), [0,75.44), [75.44,300), [300,600) m at
  30 / 60 / 120 s;
* discharge: HDVs first seen upstream of s = 75.44 m that cross it within 120 s (all lanes);
* queue: HDVs slower than 5 m/s upstream of the merge at 120 s.

Prints per model and arm the mean WM, SUMO and WM-SUMO (bootstrap 95 % CI over states), and for
every pair of models the paired difference WM_A - WM_B and |err_A| - |err_B| (negative = A closer
to SUMO), bootstrap 95 % CI over states.

  PYTHONPATH="src;." python tools/wm_merge_check.py \
      --ckpt v5m=E:/research-dense/runs/active/C4-V5-20261014/fetched/c4_wm_v5/wm_s_best.pt \
      --ckpt v7ft=/path/wm_s_best.pt@AS_IS --mode ON --every 2 --workers 4 --json out.json

``--ckpt NAME=path[@MODE]`` (repeatable); MODE overrides ``--mode`` for that checkpoint
(ON = merge_end_wait + mainline_merge_yield on = v5m physics, OFF, AS_IS = physics.json).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import wm_eval_common as C  # noqa: E402

DEFAULT_SET = "E:/agent-work/c4-v6merge/_ref/states_h60/wall_same.json"
BINS = [(-400.0, 0.0), (0.0, C.MERGE_END_M), (C.MERGE_END_M, 300.0), (300.0, 600.0)]
HZ = (30.0, 60.0, 120.0)
G: dict = {}


def stats(pos, vel, msk, hdv) -> dict:
    """pos/vel/msk [T+1,K] (row i = time i*0.5 s, row 0 = t0) -> metric dict."""
    out = {}
    for hz in HZ:
        for lo, hi in BINS:
            out[f"v{int(hz)}_{int(lo)}"] = C.snapshot_speed(pos, vel, msk, hdv, hz, lo, hi)
    out["discharge_120"] = C.discharge(pos, msk, hdv, 120.0)
    i = int(round(120 / C.DT_S))
    out["queue_120"] = int((msk[i] & hdv & (pos[i] < 0) & (vel[i] < 5.0)).sum())
    return out


def init(ckpts, device, threads):
    G["models"] = {name: C.load_predictor(path, mode, device, threads)
                   for name, path, mode in ckpts}


def work(payload):
    idx, rec, arms = payload
    res = {"idx": idx, "world": rec.get("world"), "arms": {}}
    for arm in arms:
        if arm not in rec.get("arms", []):
            continue
        entry = {}
        for name, (p, cap) in G["models"].items():
            t = time.time()
            r = C.roll(p, cap, rec, arm, 120.0)
            if r is None:
                break
            hdv = ~r["is_cav"]
            entry.setdefault("sumo", stats(r["rec_pos"], r["rec_vel"], r["rec_msk"], hdv))
            entry[name] = stats(r["wm_pos"], r["wm_vel"], r["wm_msk"], hdv)
            entry.setdefault("sec", {})[name] = time.time() - t
        else:
            res["arms"][arm] = entry
    return res


def summarize(results, names, arms, n_boot=2000):
    out = {}
    for arm in arms:
        rows = [r["arms"][arm] for r in results if arm in r["arms"]]
        if not rows:
            continue
        keys = list(rows[0]["sumo"].keys())
        a = {"n_states": len(rows), "models": {}, "pairs": {}}
        s = {k: np.array([x["sumo"][k] for x in rows], float) for k in keys}
        for name in names:
            w = {k: np.array([x[name][k] for x in rows], float) for k in keys}
            a["models"][name] = {k: {"wm": float(np.nanmean(np.where(np.isfinite(s[k]), w[k], np.nan))),
                                     "sumo": float(np.nanmean(np.where(np.isfinite(w[k]), s[k], np.nan))),
                                     "err": C.bootstrap_mean(w[k] - s[k], n_boot)}
                                 for k in keys}
        for na, nb in itertools.combinations(names, 2):
            wa = {k: np.array([x[na][k] for x in rows], float) for k in keys}
            wb = {k: np.array([x[nb][k] for x in rows], float) for k in keys}
            a["pairs"][f"{na}-{nb}"] = {
                k: {"wm_diff": C.bootstrap_mean(wa[k] - wb[k], n_boot),
                    "abs_err_diff": C.bootstrap_mean(np.abs(wa[k] - s[k]) - np.abs(wb[k] - s[k]),
                                                     n_boot)}
                for k in keys}
        out[arm] = a
    return out


def print_summary(summ, names):
    for arm, a in summ.items():
        print(f"\n== arm {arm}: {a['n_states']} states  (mean WM | SUMO | WM-SUMO [95% CI])")
        keys = list(next(iter(a["models"].values())).keys())
        for name in names:
            print(f"-- {name}")
            for k in keys:
                m = a["models"][name][k]
                print(f"  {k:16s} WM {m['wm']:7.2f}  SUMO {m['sumo']:7.2f}  "
                      f"WM-SUMO {C.fmt_ci(m['err'])}")
        for pair, d in a["pairs"].items():
            print(f"-- paired {pair}: WM_A-WM_B | |err_A|-|err_B| (neg = A closer to SUMO)")
            for k in keys:
                print(f"  {k:16s} {C.fmt_ci(d[k]['wm_diff'])} | {C.fmt_ci(d[k]['abs_err_diff'])}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", action="append", required=True, help="NAME=path[@MODE]")
    ap.add_argument("--mode", default="ON", help="default physics mode ON/OFF/AS_IS")
    ap.add_argument("--set", default=DEFAULT_SET, help="wms_eval_states file")
    ap.add_argument("--arms", default="native,wall")
    ap.add_argument("--every", type=int, default=2, help="use every N-th state")
    ap.add_argument("--limit", type=int, default=None, help="max states (smoke)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=1, help="torch threads per worker")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--json", default=None, help="write per-state results + summary here")
    a = ap.parse_args(argv)
    ckpts = [C.parse_ckpt(s, a.mode.upper()) for s in a.ckpt]
    names = [c[0] for c in ckpts]
    arms = a.arms.split(",")
    from wms.eval.normalize import iter_windows
    recs = [(i, r, arms) for i, r in enumerate(iter_windows(C.guard(a.set)))
            if i % a.every == 0]
    if a.limit:
        recs = recs[:a.limit]
    print(f"[merge_check] {len(recs)} states, models {ckpts}, device {a.device}", flush=True)
    t = time.time()
    if a.workers <= 1:
        init(ckpts, a.device, a.threads)
        results = [work(x) for x in recs]
    else:
        with Pool(a.workers, initializer=init, initargs=(ckpts, a.device, a.threads)) as pool:
            results = list(pool.imap(work, recs, chunksize=1))
    print(f"[merge_check] rollouts done in {time.time() - t:.0f} s", flush=True)
    summ = summarize(results, names, arms, a.bootstrap)
    print_summary(summ, names)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        with open(C.guard(a.json), "w", encoding="utf-8") as fh:
            json.dump({"schema": "wm_merge_check/1", "set": a.set, "every": a.every,
                       "ckpts": [{"name": n, "path": p, "mode": m} for n, p, m in ckpts],
                       "summary": summ, "states": results}, fh, indent=1)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
