"""WD-01-PA2G-A2 (registered analysis, zero simulation): the Part A v2
general model error re-run on the chunk-4 TRUE fleet-native truth, with
the engine churn fix and the wired CAV path.

Reuses REGISTRATION_PA2G_EVAL_A.md with the truth source changed
(REGISTRATION_PA2G_EVAL_A2.md):

- TRUTH: chunk-4 `<world>__native_base` runs (ALL CAVs native from
  t = 0, speedMode 27): `E:/research-dense/runs/campaign_stage1_chunk4/
  <world>__native_base/run/physics_steps.jsonl`.  CAV identity comes
  from the run's own physics rows (`cav_ids`), never from
  insertions.jsonl / dispatcher_rows.jsonl.
- Only runs whose summary.json exists AND whose RUN_RECEIPT.json shows
  exit_status 'complete' AND whose physics log reaches the run duration
  are used (TRAIN worlds of results_plan_chunk4/WORLD_PLAN.json).
- Windows: every 30 s over (t0, t0+540], t0 from the plan; H in
  {10, 30, 60, 120} kept when the log covers w+H.
- CAVs: zone-state vehicles present in the run's cav_ids carry
  is_cav=True and the engine runs with the native command path
  (cavctrl.CavControllerLoop, all-native actions -> no commands; the
  partA_ctrl native path is engaged every control tick).
- Same comparators (v1 p3o replica, const-speed persistence),
  missing-arrivals curve, n_free_road_assumed stats, descriptive T/a
  sweep (no selection).  Raw rows are saved BEFORE aggregation.
- Labelled variant: `v2g_guards_on` re-runs the fixed v2g engine with
  the PA2G-A-era stability guards (cooldown_s 2.0, lc_min_speed_mps
  1.0) on every k-th world -- a labelled design-deviation comparison,
  NEVER part of the headline (owner design-deviation rule).
- TRAIN only: every opened run file is recorded and asserted to belong
  to a TRAIN world; CAL/VALIDATION/HOLDOUT are refused by split before
  any run file is opened (asserted again at close-out).

Usage:
    python eval_a2.py [--stratified N] [--workers N] [--sweep-every K]
                      [--guards-on-every K] [--no-p3o] [--smoke]
--workers N splits the selected worlds over N processes (<= 4).
--data-root / RESEARCH_DENSE_ROOT overrides the truth root (default
E:/research-dense).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
WD01 = HERE.parent
for _p in (str(WD01), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_P3D_ROOT = os.environ.get("WD01_DATA_ROOT", "").strip() or \
    str(Path("E:/rd-wt/wd01/output/wd01"))
os.environ.setdefault("WD01_DATA_ROOT", _P3D_ROOT)

import wdpaths  # noqa: E402

DATA_ROOT = Path(os.environ.get("WD01_DATA_ROOT_TRUTH", "").strip() or
                 wdpaths.RESEARCH_DENSE_ROOT)

from partA_v2g import metrics as MX  # noqa: E402
from partA_v2g.engine import (PartAV2GEngine, free_road_flags,  # noqa: E402
                              load_params, param_set_id)

from partA_v2g import eval_a as EA  # noqa: E402

ANALYSIS_ID = "WD-01-PA2G-A2"
CHUNK = 4
RUNS_DIR = "campaign_stage1_chunk4"
PLAN = DATA_ROOT / "output" / "campaign_stage1_v3" / \
    "results_plan_chunk4" / "WORLD_PLAN.json"
GUARDS_ON = dict(cooldown_s=2.0, lc_min_speed_mps=1.0)


def results_dir(smoke: bool) -> Path:
    if smoke:
        return HERE / "results" / "eval_a2_smoke"
    # WD-01-PA2G-LC1: the labelled post-fix re-run writes to its own
    # directory (EVAL_A2_OUT_DIR=eval_a2_post_lc1); the frozen A2 rows
    # are never overwritten.
    override = os.environ.get("EVAL_A2_OUT_DIR", "").strip()
    return HERE / "results" / (override if override else "eval_a2")


# ------------------------------------------------------------- worlds

def _run_paths(world: str, root: Path) -> dict:
    d = root / "runs" / RUNS_DIR / f"{world}__native_base"
    return dict(dir=d, run=d / "run",
                physics=d / "run" / "physics_steps.jsonl",
                summary=d / "run" / "summary.json",
                receipt=d / "RUN_RECEIPT.json")


def _last_time(physics: Path) -> float:
    import os
    with open(physics, "rb") as fh:
        fh.seek(max(-400000, -os.path.getsize(physics)), 2)
        rows = fh.read().splitlines()
    return float(json.loads(rows[-1])["time_s"])


def load_train_worlds(root: Path = DATA_ROOT,
                      opened: Optional[set] = None) -> List[dict]:
    """TRAIN worlds of the chunk-4 plan whose __native_base run shows
    completion.  Splits are read from the plan BEFORE any run file is
    opened; CALIBRATION/HOLDOUT worlds are never touched."""
    opened = EA.OPENED_RUN_FILES if opened is None else opened
    plan = EA._open_json(PLAN)
    out = []
    for w in plan["worlds"]:
        if w.get("split") != "TRAIN":
            continue
        entry = dict(w)
        entry["chunk"] = CHUNK
        m = re.search(r"w\d+m(\d+)p(\d+)t(\d+)", w["world"])
        entry["demand"] = int(m.group(1)) if m else -1
        entry["penetration"] = int(m.group(2)) / 100.0 if m else \
            float("nan")
        out.append(entry)
    EA.TRAIN_WORLD_NAMES.update(w["world"] for w in out)
    # completion screening (open run files for TRAIN worlds only)
    usable = []
    for entry in out:
        pp = _run_paths(entry["world"], root)
        if not pp["physics"].exists() or not pp["summary"].exists():
            entry["skip_reason"] = "native_base run files missing"
            continue
        opened.add(str(pp["summary"]))
        summary = json.loads(pp["summary"].read_text(encoding="utf-8"))
        status = None
        if pp["receipt"].exists():
            opened.add(str(pp["receipt"]))
            status = json.loads(
                pp["receipt"].read_text(encoding="utf-8")).get("exit_status")
        dur = float(summary["duration_s"])
        t_end = _last_time(pp["physics"])
        if status == "complete" and abs(t_end - dur) <= 1.0:
            entry["native_run"] = pp["physics"]
            entry["t_end_s"] = t_end
            usable.append(entry)
        else:
            entry["skip_reason"] = f"receipt={status} t_end={t_end} dur={dur}"
    usable.sort(key=lambda w: w["world"])
    return usable


def stratified_subset(worlds: List[dict], n: int) -> List[dict]:
    """Deterministic round-robin over the (demand, penetration) cells:
    each cell keeps its own name-sorted queue; one world is taken per
    cell per pass until n worlds are collected."""
    cells: Dict[Tuple[int, float], List[dict]] = defaultdict(list)
    for w in sorted(worlds, key=lambda w: w["world"]):
        cells[(w["demand"], w["penetration"])].append(w)
    order = sorted(cells)
    queues = {c: list(rows) for c, rows in cells.items()}
    picked = []
    while len(picked) < n and any(queues[c] for c in order):
        for c in order:
            if queues[c] and len(picked) < n:
                picked.append(queues[c].pop(0))
    return picked


# ------------------------------------------------------- CAV wiring

def flag_cavs(vehicles: dict, cav_ids: set) -> dict:
    """Zone-state vehicles present in the run's CAV identity set carry
    is_cav=True (the run's own physics rows are the only CAV source)."""
    out = {}
    for vid, rec in vehicles.items():
        rec = dict(rec)
        rec["is_cav"] = vid in cav_ids
        out[vid] = rec
    return out


def native_control_hook(t0: float, params: Mapping, cav_ids: set):
    """The native command path: a CavControllerLoop with all-native
    actions and no walls (ideal frames of the predicted state; no
    command is issued, the PI state advances every tick -- the
    contract's native semantics)."""
    from partA_v2g.cavctrl import CavControllerLoop
    return CavControllerLoop(
        t0=t0, params=params,
        per_cav_actions={vid: "native" for vid in cav_ids},
        walls=[], cav_state_hint={}, switch_native_at=None)


# ---------------------------------------------------------- windows

def window_rows(params, log, entry, w, replica, hook_of, sweep_params=None,
                emit_hs=EA.HS, emit_const=True):
    """eval_a.window_rows with the CAV wiring: is_cav flags + the native
    hook are attached to the v2g (and guards_on) rollouts."""
    truth_w = log.truth_at(w)
    j0 = int(round((w - log.t0) / 0.5))
    inj = getattr(log, "inj", {})
    vehicles, free_true = EA.tracked_state(truth_w, params,
                                           inj_at=inj.get(j0))
    assert vehicles, f"empty zone state at w={w}"
    vehicles = flag_cavs(vehicles, log.cav_ids)
    vids = sorted(vehicles)
    zone = params["rsu_zone_default"]["per_lane_s"]
    llen = params["geometry"]["lane_lengths_m"]
    free_pred = free_road_flags(vehicles, zone, llen)
    n_roll = int(round(EA.H_MAX / EA.DT))
    rows: List[dict] = []

    def emit(model, snaps, coll_by_step, H):
        n_h = int(round(H / EA.DT))
        pred = _series(model, snaps)
        rmse = pred.rmse(_truth_tr, n_h)
        st_p = pred.stats(n_h, params)
        st_t = _truth_tr.stats(n_h, params)
        st_all = _truth_all.stats(n_h, params)
        arr = EA.arrivals_between(log, w, H, set(vids))
        vsteps = sum(len(snaps[jj]) for jj in range(1, n_h + 1))
        coll = sum(coll_by_step.get(jj, 0) for jj in range(1, n_h + 1))
        rows.append(dict(
            world=entry["world"], chunk=entry["chunk"], w=w, H=H,
            model=model, penetration=entry["penetration"],
            demand=entry["demand"],
            n_tracked=len(vids),
            n_cav=len([v for v in vids if v in log.cav_ids]),
            n_free_road_true=len(free_true),
            n_free_road_pred=len(free_pred),
            n_arrivals=len(arr),
            s_rmse=rmse["s_rmse"], v_rmse=rmse["v_rmse"],
            n_matched=rmse["n_matched"],
            delay_pred=st_p["delay"], delay_true_tracked=st_t["delay"],
            delay_err=st_p["delay"] - st_t["delay"],
            delay_true_all=st_all["delay"],
            ttc3_pred=st_p["expo3"], ttc3_true=st_t["expo3"],
            ttc3_err=st_p["expo3"] - st_t["expo3"],
            hard_pred=st_p["hard"], hard_true=st_t["hard"],
            hard_err=st_p["hard"] - st_t["hard"],
            lc_pred=st_p["lc"], lc_true=st_t["lc"],
            collisions_pred=coll, vehicle_steps_pred=vsteps,
        ))

    j_roll = n_roll
    truth_snaps_all = [log.rows.get(j0 + jj, {}) for jj in range(j_roll + 1)]
    _truth_tr = MX.WindowSeries(truth_snaps_all, vids)
    _truth_all = MX.WindowSeries(truth_snaps_all, None)
    _series_cache: Dict[str, MX.WindowSeries] = {}

    def _series(model, snaps):
        if model not in _series_cache:
            _series_cache[model] = MX.WindowSeries(snaps, vids)
        return _series_cache[model]

    def hs_ok():
        return [H for H in emit_hs if w + H <= log.t_end + 1e-9]

    # ---- v2g with the wired CAV path (or one labelled variant)
    p = params if sweep_params is None else sweep_params
    eng = PartAV2GEngine(p, t0=w, vehicles=vehicles,
                         control_hook=hook_of(w, vehicles))
    t_a = time.perf_counter()
    eng.run(n_roll)
    t_engine = time.perf_counter() - t_a
    snaps = MX.snaps_list(eng)
    model_name = "v2g" if sweep_params is None else "v2g_sweep"
    for H in hs_ok():
        emit(model_name, snaps, eng.collisions_by_step, H)
    # ---- constant-speed persistence (main pass only: sweep/guards
    # variants share the identical row, no duplicates are emitted)
    if emit_const:
        cs = EA.const_speed_rollout(params, vehicles, n_roll)
        for H in hs_ok():
            emit("const_speed", cs, {}, H)
    # ---- v1 p3o replica (with the logged arrivals of the window)
    if replica is not None:
        births: Dict[int, dict] = {}
        seen = set(vids)
        for jj in range(1, n_roll + 1):
            rowj = log.rows.get(j0 + jj, {})
            for vid, st in rowj.items():
                if vid not in seen and st[0] >= 0:
                    births.setdefault(jj, {})[vid] = dict(
                        lane=st[0], s=st[1], v=st[2],
                        is_cav=vid in log.cav_ids,
                        _vtype=log.types.get(vid, 1))
                seen.add(vid)
        v0 = {vid: dict(rec, is_cav=vid in log.cav_ids,
                        _vtype=log.types.get(vid, 1))
              for vid, rec in vehicles.items()}
        for vid, rec in v0.items():
            if not rec["is_cav"]:
                u = EA.crn_uniform(0, vid, 0, "vtype")
                rec["_vtype"] = int(np.searchsorted(
                    np.cumsum([0.1, 0.7, 0.2]), u, "left"))
        snaps_p, _lc, coll_p, _vs = replica.rollout(w, v0, births, n_roll)
        snaps_p = [snaps_p.get(jj, {}) for jj in range(n_roll + 1)]
        for H in hs_ok():
            emit("v1_p3o_replica", snaps_p, coll_p, H)
    return rows, t_engine


# ---------------------------------------------------------- workers

def _worker(args):
    """One worker: own subset of worlds, own shard file.  Returns
    (shard_path, opened_files, per-world stats)."""
    rank, entries, params, opts = args
    out_dir = results_dir(opts["smoke"])
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard = shard_dir / f"shard_{rank}.jsonl"
    opened = set()
    EA.OPENED_RUN_FILES.clear()
    EA.OPENED_RUN_FILES.update(opened)
    EA.MAPPER = EA.ModelMapper()
    replica = None if opts["no_p3o"] else EA.V1P3OReplica()
    guard_params = None
    if opts["guards_on_every"]:
        guard_params = json.loads(json.dumps(params))
        guard_params["mobil"].update(GUARDS_ON)
    rows_written = 0
    stats = []
    with open(shard, "w", encoding="utf-8") as raw:
        for wi, entry in enumerate(entries):
            tw = time.perf_counter()
            try:
                log = EA.NativeLog(entry)
            except FileNotFoundError as e:
                print(f"  [w{rank}] {entry['world']}: SKIP ({e})",
                      flush=True)
                continue
            opened.update(EA.OPENED_RUN_FILES)
            rows_world = 0
            for k in range(1, opts.get("max_windows", 18) + 1):
                w = log.t0 + 30.0 * k
                if w > log.t_end + 1e-9:
                    break
                rows, _t = window_rows(
                    params, log, entry, w, replica,
                    hook_of=lambda ww, vehs: native_control_hook(
                        ww, params, set(vehs) & log.cav_ids))
                for r in rows:
                    raw.write(json.dumps(r) + "\n")
                rows_world += len(rows)
            if guard_params is not None and opts["guards_on_every"] and \
                    wi % opts["guards_on_every"] == 0:
                for k in range(1, opts.get("max_windows", 18) + 1):
                    w = log.t0 + 30.0 * k
                    if w > log.t_end + 1e-9:
                        break
                    rows, _t = window_rows(
                        guard_params, log, entry, w, None,
                        hook_of=lambda ww, vehs: native_control_hook(
                            ww, guard_params, set(vehs) & log.cav_ids),
                        emit_const=False)
                    for r in rows:
                        r["model"] = "v2g_guards_on"
                        raw.write(json.dumps(r) + "\n")
                    rows_world += len(rows)
            if opts["sweep_every"] and (wi % opts["sweep_every"] == 0):
                for T in EA.SWEEP_T:
                    for a in EA.SWEEP_A:
                        if abs(T - 1.5) < 1e-9 and abs(a - 1.0) < 1e-9:
                            continue
                        sp = json.loads(json.dumps(params))
                        sp["idm"]["T_s"] = T
                        sp["idm"]["a_max_mps2"] = a
                        for k in range(1, opts.get("max_windows", 18) + 1):
                            w = log.t0 + 30.0 * k
                            if w > log.t_end + 1e-9:
                                break
                            rows, _t = window_rows(
                                params, log, entry, w, None,
                                hook_of=lambda ww, vehs: native_control_hook(
                                    ww, params, set(vehs) & log.cav_ids),
                                sweep_params=sp, emit_hs=(EA.SWEEP_H,),
                                emit_const=False)
                            for r in rows:
                                r["T_s"] = T
                                r["a_max"] = a
                                raw.write(json.dumps(r) + "\n")
                            rows_world += len(rows)
            raw.flush()
            opened.update(EA.OPENED_RUN_FILES)
            stats.append(dict(world=entry["world"], rows=rows_world,
                              wall_s=round(time.perf_counter() - tw, 1)))
            print(f"  [w{rank}] [{wi + 1}/{len(entries)}] "
                  f"{entry['world']}: {rows_world} rows, "
                  f"{time.perf_counter() - tw:.1f}s", flush=True)
    side = shard.with_suffix(".opened.json")
    side.write_text(json.dumps(sorted(opened)), encoding="utf-8")
    return str(shard), sorted(opened), stats


def aggregate(df):
    """Registered aggregations from the RAW rows (same definitions as
    eval_a.aggregate, plus the labelled guards_on model kept separate)."""
    base = df[df.model.isin(["v2g", "const_speed", "v1_p3o_replica"])]
    per_mh = {}
    for (m, H), g in base.groupby(["model", "H"]):
        per_mh[f"{m}|H{int(H)}"] = dict(
            n=len(g),
            s_rmse_mean=float(g.s_rmse.mean()),
            v_rmse_mean=float(g.v_rmse.mean()),
            delay_err_mean=float(g.delay_err.mean()),
            delay_err_mae=float(g.delay_err.abs().mean()),
            ttc3_err_mean=float(g.ttc3_err.mean()),
            ttc3_err_mae=float(g.ttc3_err.abs().mean()),
            hard_err_mean=float(g.hard_err.mean()),
            hard_err_mae=float(g.hard_err.abs().mean()),
            lc_merge_pred=int(g.lc_pred.apply(lambda d: d["merge"]).sum()),
            lc_merge_true=int(g.lc_true.apply(lambda d: d["merge"]).sum()),
            lc_disc_pred=int(g.lc_pred.apply(
                lambda d: d["disc_left"] + d["disc_right"]).sum()),
            lc_disc_true=int(g.lc_true.apply(
                lambda d: d["disc_left"] + d["disc_right"]).sum()),
            collisions_pred=int(g.collisions_pred.sum()),
            vehicle_steps_pred=int(g.vehicle_steps_pred.sum()),
            collisions_per_1000=float(
                g.collisions_pred.sum() /
                max(g.vehicle_steps_pred.sum() / 1000.0, 1e-9)),
            n_free_road_true_mean=float(g.n_free_road_true.mean()),
            n_free_road_pred_mean=float(g.n_free_road_pred.mean()),
            n_tracked_mean=float(g.n_tracked.mean()),
            n_cav_mean=float(g.n_cav.mean()),
        )
    go = df[df.model == "v2g_guards_on"]
    guards_on = {}
    for H, g in go.groupby("H"):
        guards_on[f"H{int(H)}"] = dict(
            n=len(g),
            s_rmse_mean=float(g.s_rmse.mean()),
            delay_err_mean=float(g.delay_err.mean()),
            lc_disc_pred=int(g.lc_pred.apply(
                lambda d: d["disc_left"] + d["disc_right"]).sum()),
            collisions_per_1000=float(
                g.collisions_pred.sum() /
                max(g.vehicle_steps_pred.sum() / 1000.0, 1e-9)))
    curve = {}
    tr = df[df.model == "v2g"]
    for H, g in tr.groupby("H"):
        curve[f"H{int(H)}"] = dict(
            delay_true_tracked_mean=float(g.delay_true_tracked.mean()),
            delay_true_all_mean=float(g.delay_true_all.mean()),
            arrivals_mean=float(g.n_arrivals.mean()),
            untracked_delay_share=float(
                (g.delay_true_all - g.delay_true_tracked).sum()
                / g.delay_true_all.sum()),
        )
    sweep = []
    sw = df[df.model == "v2g_sweep"]
    if len(sw) and "T_s" in sw:
        for (T, a), g in sw.groupby(["T_s", "a_max"]):
            sweep.append(dict(T_s=float(T), a_max=float(a), n=len(g),
                              s_rmse_mean=float(g.s_rmse.mean()),
                              v_rmse_mean=float(g.v_rmse.mean()),
                              delay_err_mean=float(g.delay_err.mean())))
    by_pen = {}
    for (pen, H), g in tr.groupby(["penetration", "H"]):
        by_pen[f"p{pen}|H{int(H)}"] = dict(
            n=len(g), s_rmse_mean=float(g.s_rmse.mean()),
            delay_err_mean=float(g.delay_err.mean()))
    by_dem = {}
    for (dem, H), g in tr.groupby(["demand", "H"]):
        by_dem[f"m{int(dem)}|H{int(H)}"] = dict(
            n=len(g), s_rmse_mean=float(g.s_rmse.mean()),
            delay_err_mean=float(g.delay_err.mean()))
    return dict(per_model_h=per_mh, guards_on_labelled=guards_on,
                missing_arrivals=curve, sweep=sweep,
                by_penetration=by_pen, by_demand=by_dem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stratified", type=int, default=0,
                    help="deterministic (demand, penetration) round-robin "
                    "subset of the usable TRAIN worlds")
    ap.add_argument("--workers", type=int, default=1,
                    help="world-parallel processes (<= 4; SUMO-free but "
                    "CPU-bound scalar loops)")
    ap.add_argument("--sweep-every", type=int, default=4)
    ap.add_argument("--guards-on-every", type=int, default=4,
                    help="labelled v2g_guards_on variant every k-th "
                    "world (0=off)")
    ap.add_argument("--no-p3o", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="1 world, 2 windows, sweep off -> "
                    "results/eval_a2_smoke (wiring check only)")
    ap.add_argument("--data-root", type=str, default="",
                    help="truth root (default E:/research-dense)")
    args = ap.parse_args()
    if args.data_root:
        globals()["DATA_ROOT"] = Path(args.data_root)
        globals()["PLAN"] = globals()["DATA_ROOT"] / "output" / \
            "campaign_stage1_v3" / "results_plan_chunk4" / "WORLD_PLAN.json"
    n_workers = max(1, min(4, args.workers))
    t_start = time.perf_counter()
    params = load_params()
    EA.MAPPER = EA.ModelMapper()
    worlds = load_train_worlds(globals()["DATA_ROOT"])
    n_usable = len(worlds)
    skipped = [w for w in EA.TRAIN_WORLD_NAMES
               if w not in {e["world"] for e in worlds}]
    if args.stratified and args.stratified < len(worlds):
        worlds = stratified_subset(worlds, args.stratified)
    max_windows = 2 if args.smoke else 18
    if args.smoke:
        worlds = worlds[:1]
    print(f"{ANALYSIS_ID}: usable TRAIN native_base runs "
          f"{n_usable} (skipped {len(skipped)}: no/incomplete run); "
          f"running {len(worlds)} worlds, workers {n_workers}, "
          f"p3o {'ON' if not args.no_p3o else 'OFF'}, sweep every "
          f"{args.sweep_every}, guards_on every {args.guards_on_every}",
          flush=True)
    if args.stratified:
        cells = defaultdict(list)
        for w in worlds:
            cells[(w["demand"], w["penetration"])].append(w["world"])
        print("stratified cells (demand, penetration -> worlds):")
        for c in sorted(cells):
            print(f"  {c}: {len(cells[c])} {sorted(cells[c])}", flush=True)

    out_dir = results_dir(args.smoke)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_rows.jsonl"

    if n_workers == 1:
        results = [_worker((0, worlds, params, dict(
            smoke=args.smoke, no_p3o=args.no_p3o,
            sweep_every=args.sweep_every,
            guards_on_every=args.guards_on_every,
            max_windows=max_windows)))]
    else:
        ctx = mp.get_context("spawn")
        chunks = [worlds[i::n_workers] for i in range(n_workers)]
        with ctx.Pool(n_workers) as pool:
            results = pool.map(_worker, [
                (i, chunks[i], params, dict(
                    smoke=args.smoke, no_p3o=args.no_p3o,
                    sweep_every=args.sweep_every,
                    guards_on_every=args.guards_on_every,
                    max_windows=max_windows))
                for i in range(n_workers)])
    # merge shards in the assigned world order
    world_rank = {}
    for rank, (_shard, _opened, stats) in enumerate(results):
        for st in stats:
            world_rank[st["world"]] = rank
    shard_rows = {}
    for shard, _opened, _stats in results:
        with open(shard, encoding="utf-8") as fh:
            shard_rows[shard] = fh.readlines()
    n_rows = 0
    with open(raw_path, "w", encoding="utf-8") as out:
        for w in worlds:
            rank = world_rank.get(w["world"])
            if rank is None:
                continue
            shard = results[rank][0]
            for line in shard_rows[shard]:
                r = json.loads(line)
                if r["world"] == w["world"]:
                    out.write(line if line.endswith("\n") else line + "\n")
                    n_rows += 1
    opened_all = sorted(set().union(*[set(o) for _s, o, _st in results]))
    # the parent's screening opens (per-run summary.json/receipt of TRAIN
    # worlds + the plan) must be recorded too; workers only ever see the
    # physics logs
    opened_all = sorted(set(opened_all) | set(EA.OPENED_RUN_FILES))
    EA.OPENED_RUN_FILES.clear()
    EA.OPENED_RUN_FILES.update(opened_all)
    EA.assert_train_only()
    (out_dir / "opened_run_files_full.json").write_text(
        json.dumps(opened_all, indent=1), encoding="utf-8")
    import pandas as pd
    df = pd.read_json(raw_path, lines=True)
    summary = aggregate(df)
    meta = dict(
        analysis_id=ANALYSIS_ID,
        status="completed",
        truth="chunk-4 <world>__native_base runs (true fleet-native, "
              "speedMode 27); CAV ids from the physics rows' cav_ids",
        data_root=str(DATA_ROOT),
        plan=str(PLAN),
        n_train_worlds_total=len(EA.TRAIN_WORLD_NAMES),
        n_usable=n_usable,
        skipped_worlds=sorted(skipped),
        stratified=args.stratified,
        world_selection=("stratified round-robin over (demand, "
                         "penetration) cells" if args.stratified
                         else "all usable"),
        n_worlds_run=len(worlds),
        workers=n_workers,
        windows_per_world=max_windows,
        param_set_id=param_set_id(),
        guards_on_variant="labelled design-deviation comparison "
                          "(cooldown 2.0, lc_min_speed 1.0), every "
                          f"{args.guards_on_every}-th world",
        wall_clock_s=time.perf_counter() - t_start,
        sweep_every=args.sweep_every,
        opened_run_files=opened_all,
        assert_train_only="passed",
        summary=summary,
    )
    (out_dir / "summary.json").write_text(
        json.dumps(meta, indent=1, default=str), encoding="utf-8")
    print("per-model H rows:")
    for k, v in summary["per_model_h"].items():
        print(" ", k, "n", v["n"], "s_rmse",
              round(v["s_rmse_mean"], 3), "delay_err",
              round(v["delay_err_mean"], 3), "disc_lc",
              v["lc_disc_pred"], "/", v["lc_disc_true"], "coll/1000",
              round(v["collisions_per_1000"], 2))
    print("DONE", ANALYSIS_ID, f"wall {meta['wall_clock_s']:.0f}s, "
          f"raw rows {n_rows} -> {raw_path}")


if __name__ == "__main__":
    main()
