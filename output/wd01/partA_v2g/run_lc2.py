"""WD-01-PA2G-LC2 runner: nominal + registered structural LC variants.

Executes REGISTRATION_PA2G_LC2.md on the SAME 32 PA2G-A2 TRAIN worlds
(results/lc1_a2_worlds32.json), 18 windows each (w = t0+30k), H in
{10, 30, 60, 120}:

  nominal  the registered v2g set, flags OFF (byte-identity enforced)
  va       V-a persistence  (T_persist 2.0 s = 4 steps)
  vb       V-b manoeuvre time (T_lc 4.0 s, discretionary only)
  vc       V-c strategic early merge (zero threshold + b_merge 1.0)
  vabc     all three combined

Zero SUMO.  TRAIN only: the plan split is asserted per world BEFORE any
run file is opened; every opened run file is recorded and asserted at
close-out (`assert_train_only`); the opened-file list is saved.

PRE-FLIGHT byte-identity gate: before the panel, the nominal path (all
flags OFF) is re-run on the FIRST TWO panel worlds and every 'v2g' row
must be byte-identical (canonical json) to the PRE-CHANGE reference
recorded by lc2_nominal_ref.py before the variant implementation touched
engine.py.  Any mismatch aborts the run.

Raw rows are saved BEFORE aggregation:
  lc2_rows.jsonl           eval_a2.emit schema, model = variant name
  lc2_merge_outcomes.jsonl per tracked accel-lane vehicle (dwell, merge
                           s, per-H merge/stop flags; diag_lc1_pred
                           conventions)
  lc2_opened_run_files.json, lc2_run_meta.json

Usage: python run_lc2.py [--workers N <= 3] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
WD01 = HERE.parent
for _p in (str(WD01), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import wdpaths  # noqa: E402
DATA_ROOT = Path(wdpaths.RESEARCH_DENSE_ROOT)

from partA_v2g import metrics as MX  # noqa: E402
from partA_v2g import eval_a as EA  # noqa: E402
from partA_v2g import eval_a2 as EA2  # noqa: E402
from partA_v2g import variants as VAR  # noqa: E402
from partA_v2g.engine import (PartAV2GEngine, free_road_flags,  # noqa: E402
                              load_params, param_set_id)

ANALYSIS_ID = "WD-01-PA2G-LC2"
DT = 0.5
HS = (10, 30, 60, 120)
H_MAX = 120
RESULTS = HERE / "results" / "lc2"
N_REF_WORLDS = 2
REF = RESULTS / "nominal_ref_prechange.jsonl"
EARLY_EDGE_S = 415.0   # DIAGNOSIS_LC1 section 2 early-edge convention
VARIANTS = ("nominal", "va", "vb", "vc", "vabc")


def load_panel_entries() -> List[dict]:
    """The 32-world TRAIN panel; split asserted before any file open."""
    worlds = json.loads(
        (HERE / "results" / "lc1_a2_worlds32.json").read_text())
    plan = json.loads(EA2.PLAN.read_text(encoding="utf-8"))
    meta = {w["world"]: w for w in plan["worlds"]}
    entries = []
    for name in worlds:
        e = dict(meta[name])
        assert e.get("split") == "TRAIN", f"{name} is not TRAIN"
        e["chunk"] = 4
        m = re.search(r"w\d+m(\d+)p(\d+)t(\d+)", name)
        e["demand"] = int(m.group(1)) if m else -1
        e["penetration"] = int(m.group(2)) / 100.0 if m else float("nan")
        e["native_run"] = DATA_ROOT / "runs" / "campaign_stage1_chunk4" / \
            f"{name}__native_base" / "run" / "physics_steps.jsonl"
        entries.append(e)
    return entries


def lc2_window_rows(params, log, entry, w, variant_name):
    """One window, one variant: the v2g branch of eval_a2.window_rows
    (replica None, const off), same emit schema and key order."""
    truth_w = log.truth_at(w)
    j0 = int(round((w - log.t0) / 0.5))
    inj = getattr(log, "inj", {})
    vehicles, free_true = EA.tracked_state(truth_w, params,
                                           inj_at=inj.get(j0))
    assert vehicles, f"empty zone state at w={w}"
    vehicles = EA2.flag_cavs(vehicles, log.cav_ids)
    vids = sorted(vehicles)
    zone = params["rsu_zone_default"]["per_lane_s"]
    llen = params["geometry"]["lane_lengths_m"]
    free_pred = free_road_flags(vehicles, zone, llen)
    n_roll = int(round(H_MAX / DT))
    cav_ids = set(vids) & log.cav_ids
    hook = EA2.native_control_hook(w, params, cav_ids)
    eng = PartAV2GEngine(params, t0=w, vehicles=vehicles,
                         control_hook=hook)
    eng.run(n_roll)
    snaps = MX.snaps_list(eng)
    truth_snaps_all = [log.rows.get(j0 + jj, {}) for jj in range(n_roll + 1)]
    truth_tr = MX.WindowSeries(truth_snaps_all, vids)
    truth_all = MX.WindowSeries(truth_snaps_all, None)
    pred = MX.WindowSeries(snaps, vids)
    rows: List[dict] = []
    merge_rows: List[dict] = []
    for H in [h for h in HS if w + h <= log.t_end + 1e-9]:
        n_h = int(round(H / DT))
        rmse = pred.rmse(truth_tr, n_h)
        st_p = pred.stats(n_h, params)
        st_t = truth_tr.stats(n_h, params)
        st_all = truth_all.stats(n_h, params)
        arr = EA.arrivals_between(log, w, H, set(vids))
        vsteps = sum(len(snaps[jj]) for jj in range(1, n_h + 1))
        coll = sum(eng.collisions_by_step.get(jj, 0)
                   for jj in range(1, n_h + 1))
        rows.append(dict(
            world=entry["world"], chunk=entry["chunk"], w=w, H=H,
            model=variant_name, penetration=entry["penetration"],
            demand=entry["demand"],
            n_tracked=len(vids),
            n_cav=len(cav_ids),
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
    merge_rows.extend(lc2_merge_outcomes(
        eng, snaps, vids, w, entry, variant_name,
        float(params["merge"]["zone_s1_m"])))
    return rows, merge_rows


def lc2_merge_outcomes(eng, snaps, vids, w, entry, variant_name, s1z):
    """Per tracked accel-lane vehicle at w: merge step/time/s (pre-change
    lane-2 s), dead-end stop, dwell; per-H flags.  Mirrors the
    diag_lc1_pred.merge_outcomes conventions (entry = first lane-2 snap,
    stop = s >= s1z-0.51 with v < 0.1 while still on lane 2)."""
    out = []
    for vid in vids:
        p0 = snaps[0].get(vid)
        if p0 is None or p0[0] != 2:
            continue
        entry_step = None
        for j in range(len(snaps)):
            st = snaps[j].get(vid)
            if st is not None and st[0] == 2:
                entry_step = j
                break
        merge_step = merge_s = stop_step = None
        for j in range(1, len(snaps)):
            st = snaps[j].get(vid)
            if st is None:
                continue
            if st[0] == 1 and merge_step is None:
                merge_step = j
                prev = snaps[j - 1].get(vid)
                merge_s = round(float(prev[1]), 2) if prev is not None \
                    else None
                break
            if st[0] == 2 and stop_step is None and \
                    st[1] >= s1z - 0.51 and st[2] < 0.1:
                stop_step = j
        dwell = (merge_step - entry_step) * DT \
            if (merge_step is not None and entry_step is not None) else None
        out.append(dict(
            world=entry["world"], w=w, variant=variant_name, vid=vid,
            s0=round(float(p0[1]), 2), v0=round(float(p0[2]), 2),
            entry_step=entry_step,
            entry_t=round(w + entry_step * DT, 3)
            if entry_step is not None else None,
            merge_step=merge_step,
            merge_t=round(w + merge_step * DT, 3)
            if merge_step is not None else None,
            merge_s=merge_s,
            stop_step=stop_step,
            stop_t=round(w + stop_step * DT, 3)
            if stop_step is not None else None,
            merged_in={f"H{H}": bool(merge_step is not None and
                                     merge_step * DT <= H) for H in HS},
            stop_in={f"H{H}": bool(stop_step is not None and
                                   stop_step * DT <= H) for H in HS},
            dwell_s=dwell,
            dwell_in={f"H{H}": (dwell if
                                merge_step is not None and
                                entry_step is not None and
                                merge_step * DT <= H else None)
                      for H in HS},
        ))
    return out


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True)


def check_identity(params, entries) -> None:
    """PRE-FLIGHT gate: nominal (flags OFF) must reproduce the
    pre-change reference rows byte-identically on the first two panel
    worlds."""
    assert REF.exists(), f"missing pre-change reference {REF}"
    ref_lines = REF.read_text(encoding="utf-8").splitlines()
    ref = {}
    for line in ref_lines:
        r = json.loads(line)
        ref[(r["world"], r["w"], r["H"])] = _canon(r)
    EA.MAPPER = EA.ModelMapper()
    n_cmp = 0
    for entry in entries[:N_REF_WORLDS]:
        log = EA.NativeLog(entry)
        for k in range(1, 19):
            w = log.t0 + 30.0 * k
            if w > log.t_end + 1e-9:
                break
            rows, _mo = lc2_window_rows(
                VAR.variant_params(params, "nominal"), log, entry, w,
                "v2g")
            for r in rows:
                key = (r["world"], r["w"], r["H"])
                assert key in ref, f"reference missing {key}"
                got = _canon(r)
                assert got == ref[key], \
                    f"NOMINAL BYTE-IDENTITY FAILED at {key}:\n ref={ref[key]}\n got={got}"
                n_cmp += 1
    print(f"byte-identity vs pre-change reference: {n_cmp} rows EQUAL "
          f"({[e['world'] for e in entries[:N_REF_WORLDS]]})", flush=True)


def _worker(args):
    rank, entries, base_params, out_dir = args
    EA.MAPPER = EA.ModelMapper()
    opened = set()
    shard_dir = Path(out_dir) / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    f_rows = open(shard_dir / f"shard_{rank}.rows.jsonl", "w",
                  encoding="utf-8")
    f_mrg = open(shard_dir / f"shard_{rank}.merge.jsonl", "w",
                 encoding="utf-8")
    var_params = {name: VAR.variant_params(base_params, name)
                  for name in VARIANTS}
    stats = []
    for wi, entry in enumerate(entries):
        t_a = time.perf_counter()
        log = EA.NativeLog(entry)
        opened.update(EA.OPENED_RUN_FILES)
        EA.OPENED_RUN_FILES.clear()
        n_w = 0
        for k in range(1, 19):
            w = log.t0 + 30.0 * k
            if w > log.t_end + 1e-9:
                break
            for name in VARIANTS:
                rows, mrows = lc2_window_rows(var_params[name], log,
                                              entry, w, name)
                for r in rows:
                    f_rows.write(json.dumps(r) + "\n")
                for r in mrows:
                    f_mrg.write(json.dumps(r) + "\n")
            n_w += 1
        stats.append(dict(world=entry["world"], windows=n_w,
                          wall_s=round(time.perf_counter() - t_a, 1)))
        print(f"  [w{rank}] [{wi + 1}/{len(entries)}] {entry['world']}: "
              f"{n_w} windows x {len(VARIANTS)} variants, "
              f"{time.perf_counter() - t_a:.1f}s", flush=True)
    f_rows.close()
    f_mrg.close()
    return (str(shard_dir / f"shard_{rank}.rows.jsonl"),
            str(shard_dir / f"shard_{rank}.merge.jsonl"),
            sorted(opened), stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--smoke", action="store_true",
                    help="identity gate + first world only (wiring check)")
    args = ap.parse_args()
    n_workers = max(1, min(3, args.workers))
    t_start = time.perf_counter()
    out_dir = HERE / "results" / ("lc2_smoke" if args.smoke else "lc2")
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = load_panel_entries()
    EA.OPENED_RUN_FILES.clear()
    EA.TRAIN_WORLD_NAMES.clear()
    EA.TRAIN_WORLD_NAMES.update(e["world"] for e in entries)
    if args.smoke:
        entries = entries[:1]
    params = load_params()
    print(f"{ANALYSIS_ID}: {len(entries)} TRAIN worlds x {len(VARIANTS)} "
          f"configs, workers {n_workers}, param_set_id "
          f"{param_set_id()[:12]}, variant cfg sha "
          f"{VAR.variant_config_sha256()[:12]}", flush=True)
    check_identity(params, entries)
    if n_workers == 1:
        results = [_worker((0, entries, params, out_dir))]
    else:
        ctx = mp.get_context("spawn")
        chunks = [entries[i::n_workers] for i in range(n_workers)]
        with ctx.Pool(n_workers) as pool:
            results = pool.map(_worker, [(i, chunks[i], params, out_dir)
 for i in range(n_workers)])
    # merge shards in the assigned world order
    world_rank = {}
    for rank, (_fr, _fm, _o, stats) in enumerate(results):
        for st in stats:
            world_rank[st["world"]] = rank
    shard_rows: Dict[str, List[str]] = {}
    for fr, fm, _o, _s in results:
        shard_rows[fr] = open(fr, encoding="utf-8").readlines()
        shard_rows[fm] = open(fm, encoding="utf-8").readlines()
    n_rows = n_mrg = 0
    with open(out_dir / "lc2_rows.jsonl", "w", encoding="utf-8") as out, \
            open(out_dir / "lc2_merge_outcomes.jsonl", "w",
                 encoding="utf-8") as mout:
        for entry in entries:
            rank = world_rank.get(entry["world"])
            if rank is None:
                continue
            fr, fm = results[rank][0], results[rank][1]
            for line in shard_rows[fr]:
                r = json.loads(line)
                if r["world"] == entry["world"]:
                    out.write(line if line.endswith("\n") else line + "\n")
                    n_rows += 1
            for line in shard_rows[fm]:
                r = json.loads(line)
                if r["world"] == entry["world"]:
                    mout.write(line if line.endswith("\n") else line + "\n")
                    n_mrg += 1
    opened_all = sorted(set().union(*[set(o) for _f, _g, o, _s in results]))
    opened_all = sorted(set(opened_all) | set(EA.OPENED_RUN_FILES) |
                        {str(EA2.PLAN)})
    EA.OPENED_RUN_FILES.clear()
    EA.OPENED_RUN_FILES.update(opened_all)
    EA.assert_train_only()
    (out_dir / "lc2_opened_run_files.json").write_text(
        json.dumps(opened_all, indent=1), encoding="utf-8")
    meta = dict(
        analysis_id=ANALYSIS_ID,
        registration="output/wd01/REGISTRATION_PA2G_LC2.md (LF-sha256 "
                     "725e0a876c554d7b)",
        status="completed",
        variants=list(VARIANTS),
        variant_flags={n: VAR.enabled_flags(VAR.variant_params(params, n))
                       for n in VARIANTS},
        variant_config_sha256=VAR.variant_config_sha256(),
        param_set_id=param_set_id(),
        params_nominal_unchanged=True,
        worlds=len(entries),
        workers=n_workers,
        wall_clock_s=time.perf_counter() - t_start,
        byte_identity_gate="passed (nominal flags-off == pre-change "
                           "reference on the first 2 panel worlds)",
        raw_rows=n_rows, merge_rows=n_mrg,
        opened_run_files=opened_all,
        assert_train_only="passed",
    )
    (out_dir / "lc2_run_meta.json").write_text(
        json.dumps(meta, indent=1), encoding="utf-8")
    print(f"DONE {ANALYSIS_ID}: {n_rows} rows, {n_mrg} merge rows, "
          f"wall {meta['wall_clock_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
