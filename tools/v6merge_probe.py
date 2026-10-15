"""Merge-zone physics probe: one WM-S checkpoint rolled out 120 s on recorded wall states.

Compares physics modes that differ only in ``PhysicsConfig.merge_end_wait`` /
``mainline_merge_yield`` (OFF both False, ON both True, WAIT only wait, YIELD only yield).
Rollout = probe_walls.py variant "E": estimated drivers, residual on, no overrides, entries off,
CAVs teacher-forced from the recorded arm.  CPU only, no SUMO.

  cd E:/agent-work/c4-v6merge2
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
    /c/Users/jpeng/miniconda3/envs/rl_driving/python.exe tools/v6merge_probe.py \
    --modes OFF,ON --every 4 --workers 2 --out _probe_out/run1

Metrics (per state, then aggregated over states):
  * HDV |ds| = |predicted - recorded position| of non-CAV vehicles present in both, at 30/60/120 s;
    per-state median.
  * blow-up (rule of runs/active/C4-V5-20261014/an_walls.py): per-state median |ds| at 60 s > 50 m
    AND at 120 s > 100 m (states with no matched HDV at either horizon are excluded).
  * lane-end wait (an_endmerge.py): HDVs recorded on lane 2 at t0; time of the first step at which
    the vehicle is present and not on lane 2 (inf = never within the horizon), model vs SUMO.
    Cohort STOPPED: s0 > 60 m and v0 < 1 m/s.  Cohort ALL: s0 >= 0.  Pooled over states.
  * lane-2 overshoot: predicted vehicles present on lane 2 with s > 75.44 m (step-vehicle count
    and distinct vehicles).  Should be 0 when merge_end_wait is on.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pickle
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402

FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1")
DEFAULT_CKPT = "E:/research-dense/runs/active/C4-V5-20261014/fetched/c4_wm_v5/wm_s_best.pt"
DEFAULT_SET = "E:/agent-work/c4-v6merge/_ref/states_h60/wall_same.json"
HORIZONS = (30.0, 60.0, 120.0)
LANE_END_M = 75.44
# mode -> (merge_end_wait, mainline_merge_yield)
MODES = {"OFF": (False, False), "ON": (True, True), "WAIT": (True, False), "YIELD": (False, True)}
FIELDS = ("merge_end_wait", "mainline_merge_yield")
G: dict = {}


def guard(p):
    s = str(p).replace("\\", "/").lower()
    if any(t in s for t in FORBIDDEN):
        raise SystemExit(f"restricted path {p}")
    return p


# ----------------------------------------------------------------------------- physics switch
def physics_holders(model):
    """Every (owner, attr) in the model graph that holds a PhysicsConfig instance."""
    from wms.physics.config import PhysicsConfig
    found = []
    seen = set()

    def visit(obj):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        for name, val in list(vars(obj).items()):
            if isinstance(val, PhysicsConfig):
                found.append((obj, name))
        for m in getattr(obj, "_modules", {}).values():
            if m is not None:
                visit(m)

    visit(model)
    return found


def set_mode(model, mode):
    """Swap the model's physics for a copy with the mode's two switches; replaces every reference."""
    from wms.physics.config import PhysicsConfig
    have = PhysicsConfig.__dataclass_fields__
    wait, yld = MODES[mode]
    want = dict(zip(FIELDS, (wait, yld)))
    missing = [f for f in FIELDS if f not in have]
    if mode != "OFF" and missing:
        raise SystemExit(f"mode {mode} needs PhysicsConfig fields {missing}, which do not exist yet")
    changes = {k: v for k, v in want.items() if k in have}
    holders = physics_holders(model)
    if not holders:
        raise SystemExit("no PhysicsConfig reference found on the model")
    base = G.setdefault("base_physics", model.physics)
    new = dataclasses.replace(base, **changes)
    for owner, name in holders:
        setattr(owner, name, new)
    assert all(getattr(o, n) is new for o, n in physics_holders(model))
    return {"holders": [f"{type(o).__name__}.{n}" for o, n in holders], "applied": changes}


# ----------------------------------------------------------------------------- worker
def init(ckpt, modes, threads):
    torch.set_num_threads(threads)
    from wms.eval.predictor import RolloutFlags, WmsPredictor
    p = WmsPredictor.from_checkpoint(ckpt, RolloutFlags(entries=False, entries_v2=False))
    model = p.model
    G["p"] = p
    G["modes"] = modes
    cap = {}
    orig = model.rollout

    def rollout(state, **kw):
        cap["last"] = orig(state, **kw)
        return cap["last"]

    model.rollout = rollout
    G["cap"] = cap


def first_change(lanes, mask):
    idx = np.where(mask & (lanes != 2))[0]
    return (idx[0] + 1) * 0.5 if idx.size else np.inf


def work(payload):
    idx, rec, arm = payload
    from wms.eval.rollout_inputs import recorded_teacher, t0_vehicles
    p, cap = G["p"], G["cap"]
    t0, ids, kin, is_cav = t0_vehicles(rec)
    n_steps = 240
    teacher, present_seq, _ = recorded_teacher(rec, t0, ids, kin, n_steps, arm, p.backend.tools)
    tr = teacher[0].numpy()            # [T+1,K,4] (s, lane, v, a)
    rp = present_seq[0].numpy()        # [T+1,K]
    hdv = ~is_cav
    nid = len(ids)
    res = dict(idx=idx, world=rec.get("world"), source_arm=rec.get("source_arm"), t0=t0,
               n_veh=nid, n_hdv=int(hdv.sum()), modes={})
    for mode in G["modes"]:
        info = set_mode(p.model, mode)
        t_start = time.time()
        p.predict_states(rec, {**rec.get("cav_controllers", {}), "arm": arm}, 120.0, 0)
        sec = time.time() - t_start
        roll = cap["last"]
        pos = roll["positions"][0, :, :nid].numpy()      # [T,K], row i-1 <-> time i*0.5
        lan = roll["lanes"][0, :, :nid].numpy()
        msk = roll["masks"][0, :, :nid].numpy()
        ds_abs, n_match = {}, {}
        for hz in HORIZONS:
            i = int(round(hz / 0.5))
            m = rp[i] & msk[i - 1] & hdv
            n_match[str(int(hz))] = int(m.sum())
            ds_abs[str(int(hz))] = (float(np.median(np.abs(pos[i - 1][m] - tr[i, :, 0][m])))
                                    if m.any() else float("nan"))
        # lane-end wait
        stopped_m, stopped_r, all_m, all_r = [], [], [], []
        for k in range(nid):
            if is_cav[k] or int(tr[0, k, 1]) != 2:
                continue
            s0, v0 = tr[0, k, 0], tr[0, k, 2]
            pm = first_change(lan[:, k], msk[:, k])
            rr = first_change(tr[1:, k, 1].astype(int), rp[1:, k])
            if s0 > 60 and v0 < 1:
                stopped_m.append(pm)
                stopped_r.append(rr)
            if s0 >= 0:
                all_m.append(pm)
                all_r.append(rr)
        over = msk & (lan == 2) & (pos > LANE_END_M)       # [T,K]
        res["modes"][mode] = dict(
            seconds=sec, ds_med=ds_abs, n_match=n_match, info=info,
            wait_stopped=(stopped_m, stopped_r), wait_all=(all_m, all_r),
            over_hdv_steps=int(over[:, hdv].sum()), over_hdv_veh=int(over[:, hdv].any(0).sum()),
            over_all_steps=int(over.sum()), over_all_veh=int(over.any(0).sum()))
    return res


# ----------------------------------------------------------------------------- summary
def blown(ds):
    a, b = ds["60"], ds["120"]
    if not (np.isfinite(a) and np.isfinite(b)):
        return None
    return bool(a > 50 and b > 100)


def _wait_line(model_w, sumo_w):
    if not len(model_w):
        return "n=0"
    mw, sw = np.array(model_w), np.array(sumo_w)
    return (f"n={len(mw)} med {np.median(mw):.1f}/{np.median(sw):.1f} "
            f">30s {np.mean(mw > 30):.2f}/{np.mean(sw > 30):.2f}")


def summarize(results, modes):
    rows, out = [], {}
    for mode in modes:
        ms = [r["modes"][mode] for r in results]
        d = {h: np.array([m["ds_med"][h] for m in ms]) for h in ("30", "60", "120")}
        bl = [blown(m["ds_med"]) for m in ms]
        den = sum(b is not None for b in bl)
        pool = lambda key, j: [x for m in ms for x in m[key][j]]  # noqa: E731
        ws_m, ws_r = pool("wait_stopped", 0), pool("wait_stopped", 1)
        wa_m, wa_r = pool("wait_all", 0), pool("wait_all", 1)
        out[mode] = dict(
            n_states=len(ms), blowups=sum(b for b in bl if b), blow_den=den,
            ds_median={h: float(np.nanmedian(v)) for h, v in d.items()},
            ds_mean60=float(np.nanmean(d["60"])),
            wait_stopped=dict(n=len(ws_m), model_median=float(np.median(ws_m)) if ws_m else None,
                              sumo_median=float(np.median(ws_r)) if ws_r else None,
                              model_frac_gt30=float(np.mean(np.array(ws_m) > 30)) if ws_m else None,
                              sumo_frac_gt30=float(np.mean(np.array(ws_r) > 30)) if ws_r else None),
            wait_all=dict(n=len(wa_m), model_median=float(np.median(wa_m)) if wa_m else None,
                          sumo_median=float(np.median(wa_r)) if wa_r else None,
                          model_frac_gt30=float(np.mean(np.array(wa_m) > 30)) if wa_m else None,
                          sumo_frac_gt30=float(np.mean(np.array(wa_r) > 30)) if wa_r else None),
            over_hdv_steps=sum(m["over_hdv_steps"] for m in ms),
            over_hdv_veh=sum(m["over_hdv_veh"] for m in ms),
            over_all_steps=sum(m["over_all_steps"] for m in ms),
            sec_per_state=float(np.mean([m["seconds"] for m in ms])))
        rows.append((mode, out[mode], _wait_line(ws_m, ws_r), _wait_line(wa_m, wa_r)))
    return rows, out


def print_table(rows):
    print("\nmode | n | blowups | med|ds| 30/60/120 s | mean60 | "
          "STOPPED lane-end wait model/SUMO | ALL lane-2 wait model/SUMO | "
          "lane2>75.44m hdv steps(veh)/all steps | s/state")
    for mode, o, ws, wa in rows:
        d = o["ds_median"]
        print(f"{mode} | {o['n_states']} | {o['blowups']}/{o['blow_den']} | "
              f"{d['30']:.2f}/{d['60']:.2f}/{d['120']:.2f} | {o['ds_mean60']:.2f} | {ws} | {wa} | "
              f"{o['over_hdv_steps']}({o['over_hdv_veh']})/{o['over_all_steps']} | "
              f"{o['sec_per_state']:.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--set", default=DEFAULT_SET)
    ap.add_argument("--arm", default="wall")
    ap.add_argument("--limit", type=int, default=None, help="first N states (after --every)")
    ap.add_argument("--every", type=int, default=1, help="keep every K-th state")
    ap.add_argument("--modes", default="OFF,ON")
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon", type=float, default=120.0)
    ap.add_argument("--workers", type=int, default=1, help="1 = serial (2 torch threads); max 2")
    a = ap.parse_args()
    for x in (a.ckpt, a.set, a.out):
        guard(x)
    modes = [m.strip().upper() for m in a.modes.split(",") if m.strip()]
    bad = [m for m in modes if m not in MODES]
    if bad or not modes:
        raise SystemExit(f"unknown modes {bad}; choose from {list(MODES)}")
    if a.horizon != 120.0:
        raise SystemExit("only --horizon 120 is supported (metrics are defined at 30/60/120 s)")
    if a.workers not in (1, 2):
        raise SystemExit("--workers must be 1 or 2")
    if any(m != "OFF" for m in modes):
        from wms.physics.config import PhysicsConfig
        miss = [f for f in FIELDS if f not in PhysicsConfig.__dataclass_fields__]
        if miss:
            raise SystemExit(f"PhysicsConfig lacks {miss}; only --modes OFF is possible for now")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from wms.eval.normalize import iter_windows
    recs = (r for i, r in enumerate(iter_windows(a.set)) if i % a.every == 0)
    payloads = []
    for r in recs:
        payloads.append((len(payloads), r, a.arm))
        if a.limit is not None and len(payloads) >= a.limit:
            break
    threads = 2 if a.workers == 1 else 1
    t_all = time.time()
    results = []
    if a.workers == 1:
        init(a.ckpt, modes, threads)
        for pl in payloads:
            results.append(work(pl))
            print(f"state {pl[0]} {results[-1]['world']} "
                  + " ".join(f"{m}:{results[-1]['modes'][m]['seconds']:.1f}s" for m in modes)
                  + f" total {time.time() - t_all:.0f}s", flush=True)
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers, initializer=init,
                                          initargs=(a.ckpt, modes, threads)) as pool:
            for i, res in enumerate(pool.imap(work, payloads, chunksize=1)):
                results.append(res)
                print(f"state {i} {res['world']} total {time.time() - t_all:.0f}s", flush=True)
    rows, summ = summarize(results, modes)
    print_table(rows)
    with open(out / "results.pkl", "wb") as fh:
        pickle.dump(results, fh, protocol=4)
    with open(out / "summary.json", "w") as fh:
        json.dump(dict(args=vars(a), summary=summ), fh, indent=1)
    print("done", len(results), "states", f"{time.time() - t_all:.0f}s", "->", out)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    main()
