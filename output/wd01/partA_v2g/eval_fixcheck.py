"""WD-01-PA2G-A2 fix-attribution run (labelled, zero simulation).

The FIXED v2g engine on the PA2G-A chunk-2/3 `<world>__native` TRAIN
truth, with the SAME CAV treatment as PA2G-A (no is_cav flags, no
control hook), so the only difference vs the saved PA2G-A v2g rows is
the engine fix.  4 worlds, round-robin over (demand, penetration)
cells; v2g nominal (guards OFF) + the labelled `v2g_fixed_guards_on`
variant (PA2G-A-era guards re-enabled).  No p3o, no const-speed (their
rows are unchanged by the fix and already saved in results/eval_a).

TRAIN only: reuses eval_a.load_train_worlds/assert_train_only.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict

HERE = Path(__file__).resolve().parent
WD01 = HERE.parent
for _p in (str(WD01), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import partA_v2g.eval_a as EA  # noqa: E402
from partA_v2g.engine import PartAV2GEngine, load_params, param_set_id  # noqa: E402

ANALYSIS_ID = "WD-01-PA2G-A2-fixcheck"
RESULTS = HERE / "results" / "eval_fixcheck"
GUARDS_ON = dict(cooldown_s=2.0, lc_min_speed_mps=1.0)


def window_rows_fixed(params, log, entry, w, emit_hs=EA.HS):
    """PA2G-A's window exactly (no CAV flags, no hook), fixed engine."""
    truth_w = log.truth_at(w)
    j0 = int(round((w - log.t0) / 0.5))
    inj = getattr(log, "inj", {})
    vehicles, free_true = EA.tracked_state(truth_w, params,
                                           inj_at=inj.get(j0))
    assert vehicles, f"empty zone state at w={w}"
    vids = sorted(vehicles)
    zone = params["rsu_zone_default"]["per_lane_s"]
    llen = params["geometry"]["lane_lengths_m"]
    free_pred = EA.free_road_flags(vehicles, zone, llen)
    n_roll = int(round(EA.H_MAX / EA.DT))
    rows = []
    truth_snaps_all = [log.rows.get(j0 + jj, {}) for jj in range(n_roll + 1)]
    _tr = EA.MX.WindowSeries(truth_snaps_all, vids)
    _all = EA.MX.WindowSeries(truth_snaps_all, None)

    def emit(model, snaps, coll_by_step, H):
        n_h = int(round(H / EA.DT))
        series = EA.MX.WindowSeries(snaps, vids)
        rmse = series.rmse(_tr, n_h)
        st_p = series.stats(n_h, params)
        st_t = _tr.stats(n_h, params)
        st_all = _all.stats(n_h, params)
        arr = EA.arrivals_between(log, w, H, set(vids))
        vsteps = sum(len(snaps[jj]) for jj in range(1, n_h + 1))
        coll = sum(coll_by_step.get(jj, 0) for jj in range(1, n_h + 1))
        rows.append(dict(
            world=entry["world"], chunk=entry["chunk"], w=w, H=H,
            model=model, penetration=entry["penetration"],
            demand=entry["demand"], n_tracked=len(vids),
            n_free_road_true=len(free_true),
            n_free_road_pred=len(free_pred), n_arrivals=len(arr),
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
            collisions_pred=coll, vehicle_steps_pred=vsteps))

    eng = PartAV2GEngine(params, t0=w, vehicles=vehicles)
    eng.run(n_roll)
    snaps = EA.MX.snaps_list(eng)
    for H in emit_hs:
        if w + H <= log.t_end + 1e-9:
            emit("v2g_fixed", snaps, eng.collisions_by_step, H)
    gp = json.loads(json.dumps(params))
    gp["mobil"].update(GUARDS_ON)
    eng2 = PartAV2GEngine(gp, t0=w, vehicles=vehicles)
    eng2.run(n_roll)
    snaps2 = EA.MX.snaps_list(eng2)
    for H in emit_hs:
        if w + H <= log.t_end + 1e-9:
            emit("v2g_fixed_guards_on", snaps2, eng2.collisions_by_step, H)
    return rows


def main():
    n_pick = 4
    t0w = time.perf_counter()
    params = load_params()
    EA.MAPPER = EA.ModelMapper()
    worlds = EA.load_train_worlds()
    # deterministic strided spread over the sorted 98 TRAIN worlds
    # (96 // 4 = stride 24 -> one world per quarter of the list, so the
    # 4 worlds span different demand/penetration cells)
    stride = max(1, len(worlds) // n_pick)
    picked = worlds[::stride][:n_pick]
    print(f"{ANALYSIS_ID}: worlds {[w['world'] for w in picked]}",
          flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    raw = open(RESULTS / "raw_rows.jsonl", "w", encoding="utf-8")
    n_rows = 0
    for entry in picked:
        log = EA.NativeLog(entry)
        tw = time.perf_counter()
        for k in range(1, 19):
            w = log.t0 + 30.0 * k
            if w > log.t_end + 1e-9:
                break
            for r in window_rows_fixed(params, log, entry, w):
                raw.write(json.dumps(r) + "\n")
                n_rows += 1
        print(f"  {entry['world']}: done {time.perf_counter() - tw:.1f}s",
              flush=True)
    raw.close()
    EA.assert_train_only()
    summary = dict(
        analysis_id=ANALYSIS_ID,
        status="completed",
        purpose="fix attribution: the FIXED engine on the PA2G-A chunk-2/3 "
                "truth with PA2G-A's CAV treatment (no is_cav flags), so "
                "the only delta vs the saved PA2G-A v2g rows is the engine "
                "fix",
        worlds=[w["world"] for w in picked],
        param_set_id=param_set_id(),
        guards_on_variant=dict(GUARDS_ON),
        raw_rows=n_rows,
        opened_run_files=sorted(EA.OPENED_RUN_FILES),
        assert_train_only="passed",
        wall_clock_s=time.perf_counter() - t0w,
    )
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8")
    print(f"DONE {ANALYSIS_ID}: {n_rows} rows, "
          f"wall {summary['wall_clock_s']:.0f}s")


if __name__ == "__main__":
    main()
