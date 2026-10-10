"""WD-01-PA2G-CNT1 (registered analysis, zero simulation): Part A v2g
plus an entrance-counter arrivals injector (v2g-CNT) vs the tracked-only
v2g, const-speed and the v1 p3o replica (LOGGED arrivals = oracle-
arrival reference, not deployable) on the 32 PA2G-A2 TRAIN worlds.

Implements REGISTRATION_PA2G_CNT1.md exactly:
- counter channel: departed_ids of the truth physics rows classified by
  the vehicle's first physical row edge (mainline 1191554636 -> "main",
  ramp 11553048#1 -> "ramp"); counts ONLY over (w - 60, w]; future rows
  are filtered before counting (sentinel test in
  tests/test_cnt1_injector.py).
- injector: arrivals.py (a-priori schedule, lane alternation from lane
  1, v_ins = min(limit, leader speed), IDM safe-gap s0 + v*T FIFO
  virtual queue).  Injected vehicles are generic HDVs of the nominal
  parameter set, labelled cntM*/cntR* in raw rows.
- TRAIN only: splits are read from the chunk-4 WORLD_PLAN BEFORE any
  run file is opened; only the 32 panel (TRAIN) worlds' files are ever
  opened; the opened-file list is asserted at close-out.
- models, paired on the SAME windows: v2g_cnt, v2g, const_speed,
  v1_p3o_replica.  Raw rows are saved BEFORE aggregation.
- zero SUMO/TraCI; <= 2 worker processes at BelowNormal priority.

Usage:
    python eval_cnt1.py [--smoke] [--preflight] [--workers N]
--preflight: byte-identity gate BEFORE the panel: with the injector
OFF the runner's v2g/const_speed rows must equal the UNTOUCHED
eval_a2.window_rows harness output on the first 2 panel worlds (all
shared fields exactly), and an EMPTY-schedule injector must reproduce
the v2g rollout bit-for-bit.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

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

from partA_v2g import arrivals as ARR  # noqa: E402
from partA_v2g import metrics as MX  # noqa: E402
from partA_v2g.engine import (PartAV2GEngine, free_road_flags,  # noqa: E402
                              load_params, param_set_id)

from partA_v2g import eval_a2 as A2  # noqa: E402
from partA_v2g import eval_a as EA  # noqa: E402

ANALYSIS_ID = "WD-01-PA2G-CNT1"
RUNS_DIR = "campaign_stage1_chunk4"
PLAN = DATA_ROOT / "output" / "campaign_stage1_v3" / \
    "results_plan_chunk4" / "WORLD_PLAN.json"
PANEL = HERE / "results" / "lc1_a2_worlds32.json"
DT = 0.5
HS = (10, 30, 60, 120)
H_MAX = max(HS)
W_COUNTER = ARR.W_COUNTER_S


def results_dir(smoke: bool) -> Path:
    return HERE / "results" / ("eval_cnt1_smoke" if smoke else "eval_cnt1")


def set_below_normal() -> None:
    """BelowNormal priority for this process (children inherit; the
    workers also call this explicitly)."""
    try:
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass


# ------------------------------------------------------------- worlds

def load_panel_worlds(root: Path = DATA_ROOT,
                      opened: Optional[set] = None) -> List[dict]:
    """The 32 PA2G-A2 TRAIN worlds: splits are read from the chunk-4
    plan BEFORE any run file is opened; a panel world whose plan split
    is not TRAIN aborts (never opened).  Completion screening opens
    summary/receipt of TRAIN panel worlds only."""
    opened = EA.OPENED_RUN_FILES if opened is None else opened
    plan = EA._open_json(PLAN)
    panel = json.loads(PANEL.read_text(encoding="utf-8"))
    by_name = {w["world"]: w for w in plan["worlds"]}
    out = []
    for name in panel:
        w = by_name.get(name)
        assert w is not None, f"panel world {name} not in the chunk-4 plan"
        assert w.get("split") == "TRAIN", \
            f"panel world {name} split={w.get('split')!r} != TRAIN"
        entry = dict(w)
        entry["chunk"] = 4
        import re
        m = re.search(r"w\d+m(\d+)p(\d+)t(\d+)", w["world"])
        entry["demand"] = int(m.group(1)) if m else -1
        entry["penetration"] = int(m.group(2)) / 100.0 if m else \
            float("nan")
        out.append(entry)
    EA.TRAIN_WORLD_NAMES.update(e["world"] for e in out)
    # completion screening (TRAIN panel run files only)
    usable = []
    for entry in out:
        pp = A2._run_paths(entry["world"], root)
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
        t_end = A2._last_time(pp["physics"])
        if status == "complete" and abs(t_end - dur) <= 1.0:
            entry["native_run"] = pp["physics"]
            entry["t_end_s"] = t_end
            usable.append(entry)
        else:
            entry["skip_reason"] = f"receipt={status} t_end={t_end} dur={dur}"
    assert len(usable) == len(panel), \
        f"panel incompleteness: {len(usable)}/{len(panel)} usable"
    usable.sort(key=lambda w: w["world"])
    return usable


def assert_panel_only() -> None:
    """Close-out assertion: every opened run file belongs to a TRAIN
    panel world (EA.assert_train_only does the TRAIN part)."""
    EA.assert_train_only()
    import re
    panel = set(json.loads(PANEL.read_text(encoding="utf-8")))
    bad = []
    for p in EA.OPENED_RUN_FILES:
        m = re.search(r"(w\S+?)__native", p)
        if m and m.group(1) not in panel:
            bad.append(p)
    assert not bad, f"non-panel run files opened: {bad}"


# ------------------------------------------------------------- window

def window_rows_cnt(params, log, entry, w, deps, replica,
                    injector_on=True, emit_hs=HS):
    """All four model rows for one window (plus the truth metrics).
    With injector_on the v2g engine is driven by the registered
    arrivals injector (schedule from the entrance counts at w)."""
    truth_w = log.truth_at(w)
    j0 = int(round((w - log.t0) / 0.5))
    inj = getattr(log, "inj", {})
    vehicles, free_true = EA.tracked_state(truth_w, params,
                                           inj_at=inj.get(j0))
    assert vehicles, f"empty zone state at w={w}"
    vehicles = A2.flag_cavs(vehicles, log.cav_ids)
    vids = sorted(vehicles)
    counts = ARR.counts_from_departures(deps, w, W_COUNTER)
    zone = params["rsu_zone_default"]["per_lane_s"]
    llen = params["geometry"]["lane_lengths_m"]
    free_pred = free_road_flags(vehicles, zone, llen)
    n_roll = int(round(H_MAX / DT))
    rows: List[dict] = []

    truth_snaps_all = [log.rows.get(j0 + jj, {}) for jj in range(n_roll + 1)]
    _truth_tr = MX.WindowSeries(truth_snaps_all, vids)
    _truth_all = MX.WindowSeries(truth_snaps_all, None)
    cache: Dict[tuple, MX.WindowSeries] = {}

    def _series(model, snaps, ids):
        key = (model, "tr" if ids is not None else "all")
        if key not in cache:
            cache[key] = MX.WindowSeries(snaps, ids)
        return cache[key]

    def emit(model, snaps, coll_by_step, H, inserted=None, n_sched=0):
        n_h = int(round(H / DT))
        pred = _series(model, snaps, vids)
        pred_all = _series(model, snaps, None)
        rmse = pred.rmse(_truth_tr, n_h)
        st_p = pred.stats(n_h, params)
        st_t = _truth_tr.stats(n_h, params)
        st_pa = pred_all.stats(n_h, params)
        st_ta = _truth_all.stats(n_h, params)
        arr = EA.arrivals_between(log, w, H, set(vids))
        vsteps = sum(len(snaps[jj]) for jj in range(1, n_h + 1))
        coll = sum(coll_by_step.get(jj, 0) for jj in range(1, n_h + 1))
        n_inj = 0 if inserted is None else \
            sum(1 for rec in inserted if rec["step"] <= n_h)
        rows.append(dict(
            world=entry["world"], chunk=entry["chunk"], w=w, H=H,
            model=model, penetration=entry["penetration"],
            demand=entry["demand"],
            n_tracked=len(vids),
            n_cav=len([v for v in vids if v in log.cav_ids]),
            n_free_road_true=len(free_true),
            n_free_road_pred=len(free_pred),
            n_arrivals=len(arr),
            counter=dict(counts),
            n_injected=n_inj,
            n_injected_scheduled=n_sched if inserted is not None else 0,
            s_rmse=rmse["s_rmse"], v_rmse=rmse["v_rmse"],
            n_matched=rmse["n_matched"],
            delay_pred=st_p["delay"], delay_true_tracked=st_t["delay"],
            delay_err=st_p["delay"] - st_t["delay"],
            delay_pred_all=st_pa["delay"],
            delay_true_all=st_ta["delay"],
            delay_all_err=st_pa["delay"] - st_ta["delay"],
            ttc3_pred=st_p["expo3"], ttc3_true=st_t["expo3"],
            ttc3_err=st_p["expo3"] - st_t["expo3"],
            ttc3_pred_all=st_pa["expo3"], ttc3_true_all=st_ta["expo3"],
            ttc3_all_err=st_pa["expo3"] - st_ta["expo3"],
            hard_pred=st_p["hard"], hard_true=st_t["hard"],
            hard_err=st_p["hard"] - st_t["hard"],
            hard_pred_all=st_pa["hard"], hard_true_all=st_ta["hard"],
            hard_all_err=st_pa["hard"] - st_ta["hard"],
            lc_pred=st_p["lc"], lc_true=st_t["lc"],
            collisions_pred=coll, vehicle_steps_pred=vsteps,
        ))

    def hs_ok():
        return [H for H in emit_hs if w + H <= log.t_end + 1e-9]

    # ---- plain v2g (tracked-only comparator; the A2 nominal path)
    p = params
    eng = PartAV2GEngine(p, t0=w, vehicles=vehicles,
                         control_hook=A2.native_control_hook(
                             w, p, set(vids) & log.cav_ids))
    eng.run(n_roll)
    snaps = MX.snaps_list(eng)
    for H in hs_ok():
        emit("v2g", snaps, eng.collisions_by_step, H)
    # ---- v2g-CNT (this variant: the registered arrivals injector on a
    # fresh identical engine; with injector_on=False the variant is not
    # emitted at all -- the preflight mode uses that for the
    # byte-identity gate)
    if injector_on:
        schedule = ARR.insertion_schedule(counts, w, H_MAX, W_COUNTER)
        eng_cnt = PartAV2GEngine(p, t0=w, vehicles=vehicles,
                                 control_hook=A2.native_control_hook(
                                     w, p, set(vids) & log.cav_ids))
        injector = ARR.ArrivalInjector(eng_cnt, schedule)
        injector.run(n_roll)
        snaps_cnt = MX.snaps_list(eng_cnt)
        for H in hs_ok():
            emit("v2g_cnt", snaps_cnt, eng_cnt.collisions_by_step, H,
                 inserted=injector.inserted, n_sched=len(schedule))
    # ---- constant-speed persistence
    cs = EA.const_speed_rollout(params, vehicles, n_roll)
    for H in hs_ok():
        emit("const_speed", cs, {}, H)
    # ---- v1 p3o replica (LOGGED true arrivals = oracle-arrival ref)
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
    return rows


# ------------------------------------------------------------ workers

def _worker(args):
    rank, entries, params, opts = args
    set_below_normal()
    out_dir = results_dir(opts["smoke"])
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard = shard_dir / f"shard_{rank}.jsonl"
    opened = set()
    EA.OPENED_RUN_FILES.clear()
    EA.OPENED_RUN_FILES.update(opened)
    EA.MAPPER = EA.ModelMapper()
    replica = None if opts["no_p3o"] else EA.V1P3OReplica()
    rows_written = 0
    stats = []
    with open(shard, "w", encoding="utf-8") as raw:
        for wi, entry in enumerate(entries):
            tw = time.perf_counter()
            log = EA.NativeLog(entry)
            opened.update(EA.OPENED_RUN_FILES)
            deps = ARR.read_departures(entry["native_run"])
            assert deps["other_edge"] == 0, \
                f"{entry['world']}: departures on unregistered edges"
            assert deps["repeat_ids"] == 0
            n_main = sum(1 for _t, e in deps["departures"] if e == "main")
            n_ramp = sum(1 for _t, e in deps["departures"] if e == "ramp")
            assert n_main == deps["bc_main"] and n_ramp == deps["bc_ramp"], \
                f"{entry['world']}: departure/counter mismatch " \
                f"{(n_main, n_ramp)} vs {(deps['bc_main'], deps['bc_ramp'])}"
            rows_world = 0
            for k in range(1, opts.get("max_windows", 18) + 1):
                w = log.t0 + 30.0 * k
                if w > log.t_end + 1e-9:
                    break
                rows = window_rows_cnt(params, log, entry, w,
                                       deps["departures"], replica)
                for r in rows:
                    raw.write(json.dumps(r) + "\n")
                rows_world += len(rows)
            raw.flush()
            opened.update(EA.OPENED_RUN_FILES)
            stats.append(dict(world=entry["world"], rows=rows_world,
                              departures_main=n_main,
                              departures_ramp=n_ramp,
                              wall_s=round(time.perf_counter() - tw, 1)))
            print(f"  [w{rank}] [{wi + 1}/{len(entries)}] "
                  f"{entry['world']}: {rows_world} rows, "
                  f"{time.perf_counter() - tw:.1f}s", flush=True)
    side = shard.with_suffix(".opened.json")
    side.write_text(json.dumps(sorted(opened)), encoding="utf-8")
    return str(shard), sorted(opened), stats


# ---------------------------------------------------------- preflight

def preflight(params, worlds) -> dict:
    """Registered byte-identity gate (injector OFF): the runner's v2g /
    const_speed rows must equal the UNTOUCHED eval_a2.window_rows
    output on the first 2 panel worlds (all shared fields exactly), and
    an EMPTY-schedule injector must reproduce the v2g rollout exactly."""
    EA.MAPPER = EA.ModelMapper()
    out = []
    for entry in worlds[:2]:
        log = EA.NativeLog(entry)
        deps = ARR.read_departures(entry["native_run"])
        for k in (1, 2):
            w = log.t0 + 30.0 * k
            mine = window_rows_cnt(params, log, entry, w,
                                   deps["departures"], None,
                                   injector_on=False)
            theirs_rows, _t_engine = A2.window_rows(
                params, log, entry, w, None,
                hook_of=lambda ww, vehs: A2.native_control_hook(
                    ww, params, set(vehs) & log.cav_ids))
            assert len(mine) == len(theirs_rows)
            for r_mine, r_theirs in zip(mine, theirs_rows):
                assert r_mine["model"] == r_theirs["model"]
                for key, val in r_theirs.items():
                    assert r_mine[key] == val, \
                        f"preflight mismatch {entry['world']} w={w} " \
                        f"{r_mine['model']} {key}: {r_mine[key]!r} != {val!r}"
            # empty-schedule injector degenerates to the plain v2g path:
            # deps=[] -> zero counts -> no due vehicle at any step
            mine_cnt_empty = window_rows_cnt(params, log, entry, w,
                                             [], None, injector_on=True)
            cnt_v2g = [r for r in mine_cnt_empty if r["model"] == "v2g_cnt"]
            off_v2g = [r for r in mine if r["model"] == "v2g"]
            assert len(cnt_v2g) == len(off_v2g) and cnt_v2g
            for r_cnt, r_off in zip(cnt_v2g, off_v2g):
                for key in ("s_rmse", "v_rmse", "n_matched", "delay_pred",
                            "delay_err", "delay_all_err", "ttc3_err",
                            "hard_err", "collisions_pred",
                            "vehicle_steps_pred", "lc_pred", "lc_true"):
                    assert r_cnt[key] == r_off[key], \
                        f"empty-injector mismatch {entry['world']} w={w} " \
                        f"{key}: {r_cnt[key]!r} != {r_off[key]!r}"
            out.append(dict(world=entry["world"], w=w,
                            rows=len(mine), ok=True))
            print(f"  preflight {entry['world']} w={w}: "
                  f"{len(mine)} rows identical", flush=True)
    return dict(passed=True, checks=out)


# ------------------------------------------------------------- aggregate

def aggregate(df):
    """Basic per-model x H aggregation from the RAW rows (bootstrap CIs
    and the decision rule live in cnt1_tables.py, run afterwards)."""
    out = {}
    for (m, H), g in df.groupby(["model", "H"]):
        out[f"{m}|H{int(H)}"] = dict(
            n=len(g),
            s_rmse_mean=float(g.s_rmse.mean()),
            v_rmse_mean=float(g.v_rmse.mean()),
            delay_err_mean=float(g.delay_err.mean()),
            delay_err_mae=float(g.delay_err.abs().mean()),
            delay_all_err_mean=float(g.delay_all_err.mean()),
            delay_all_err_mae=float(g.delay_all_err.abs().mean()),
            delay_true_all_mean=float(g.delay_true_all.mean()),
            delay_all_mae_pct=100.0 * float(g.delay_all_err.abs().mean())
            / float(g.delay_true_all.mean()),
            ttc3_all_err_mean=float(g.ttc3_all_err.mean()),
            hard_all_err_mean=float(g.hard_all_err.mean()),
            collisions_pred=int(g.collisions_pred.sum()),
            vehicle_steps_pred=int(g.vehicle_steps_pred.sum()),
            collisions_per_1000=float(
                g.collisions_pred.sum() /
                max(g.vehicle_steps_pred.sum() / 1000.0, 1e-9)),
            n_arrivals_true_mean=float(g.n_arrivals.mean()),
            n_injected_mean=float(g.n_injected.mean()) if "n_injected" in g
            else 0.0,
        )
    return dict(per_model_h=out)


# ------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-p3o", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="1 world, 2 windows -> results/eval_cnt1_smoke")
    ap.add_argument("--preflight", action="store_true",
                    help="byte-identity gate only (first 2 worlds), "
                    "no panel rows written")
    ap.add_argument("--data-root", type=str, default="")
    args = ap.parse_args()
    if args.data_root:
        globals()["DATA_ROOT"] = Path(args.data_root)
        globals()["PLAN"] = globals()["DATA_ROOT"] / "output" / \
            "campaign_stage1_v3" / "results_plan_chunk4" / "WORLD_PLAN.json"
    set_below_normal()
    t_start = time.perf_counter()
    params = load_params()
    print(f"{ANALYSIS_ID}: params {param_set_id()}", flush=True)
    worlds = load_panel_worlds(globals()["DATA_ROOT"])
    print(f"panel TRAIN worlds usable: {len(worlds)}", flush=True)
    if args.preflight:
        res = preflight(params, worlds)
        out_dir = results_dir(args.smoke)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "preflight.json").write_text(
            json.dumps(res, indent=1), encoding="utf-8")
        print("PREFLIGHT PASSED", flush=True)
        return
    n_workers = max(1, min(2, args.workers))
    max_windows = 2 if args.smoke else 18
    if args.smoke:
        worlds = worlds[:1]
    print(f"running {len(worlds)} worlds x {max_windows} windows, "
          f"workers {n_workers}, p3o {'ON' if not args.no_p3o else 'OFF'}",
          flush=True)
    out_dir = results_dir(args.smoke)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_rows.jsonl"
    if n_workers == 1:
        results = [_worker((0, worlds, params, dict(
            smoke=args.smoke, no_p3o=args.no_p3o,
            max_windows=max_windows)))]
    else:
        ctx = mp.get_context("spawn")
        chunks = [worlds[i::n_workers] for i in range(n_workers)]
        with ctx.Pool(n_workers) as pool:
            results = pool.map(_worker, [
                (i, chunks[i], params, dict(
                    smoke=args.smoke, no_p3o=args.no_p3o,
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
    opened_all = sorted(set(opened_all) | set(EA.OPENED_RUN_FILES) |
                        {str(PLAN), str(PANEL)})
    EA.OPENED_RUN_FILES.clear()
    EA.OPENED_RUN_FILES.update(opened_all)
    assert_panel_only()
    (out_dir / "opened_run_files_full.json").write_text(
        json.dumps(opened_all, indent=1), encoding="utf-8")
    import pandas as pd
    df = pd.read_json(raw_path, lines=True)
    summary = aggregate(df)
    meta = dict(
        analysis_id=ANALYSIS_ID,
        status="completed",
        registration="output/wd01/REGISTRATION_PA2G_CNT1.md",
        registration_lf_sha256=
        "1bfbce27c7ada42a680fc807344c21ab8303fa8b03e469c8d8705c4e1c213116",
        truth="chunk-4 <world>__native_base physics_steps.jsonl; counter "
              "channel = departed_ids classified by first physical row "
              f"edge (main {ARR.MAIN_ENTRY_EDGE}, "
              f"ramp {ARR.RAMP_ENTRY_EDGE}); counts only over "
              f"(w - {W_COUNTER}, w]",
        data_root=str(DATA_ROOT),
        plan=str(PLAN),
        panel=str(PANEL),
        n_worlds_run=len(worlds),
        workers=n_workers,
        windows_per_world=max_windows,
        param_set_id=param_set_id(),
        models=["v2g_cnt", "v2g", "const_speed", "v1_p3o_replica"],
        wall_clock_s=time.perf_counter() - t_start,
        n_raw_rows=n_rows,
        opened_run_files=opened_all,
        assert_panel_only="passed",
        departures_per_world={st["world"]: dict(
            main=st["departures_main"], ramp=st["departures_ramp"])
            for _s, _o, sts in results for st in sts},
        summary=summary,
    )
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=1, default=str), encoding="utf-8")
    print("per-model H rows:")
    for k, v in summary["per_model_h"].items():
        print(" ", k, "n", v["n"], "s_rmse", round(v["s_rmse_mean"], 3),
              "delay_all_err", round(v["delay_all_err_mean"], 3),
              "MAE", round(v["delay_all_err_mae"], 3),
              "coll/1000", round(v["collisions_per_1000"], 3))
    print("DONE", ANALYSIS_ID, f"wall {meta['wall_clock_s']:.0f}s, "
          f"raw rows {n_rows} -> {raw_path}")


if __name__ == "__main__":
    main()
