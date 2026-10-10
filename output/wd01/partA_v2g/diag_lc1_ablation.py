"""WD-01-PA2G-LC1 descriptive ablations (registration section 3.4).

LABELLED, DESCRIPTIVE ONLY -- NOT SELECTION, NOT ADOPTED, no change to
params_nominal.json (deep-copied overrides in memory only):

  nominal            the registered v2g set (baseline on the same runs)
  politeness_0.2     mobil.politeness_p 0.2
  politeness_1.0     mobil.politeness_p 1.0
  bias0_symmetric    mobil.keep_right_bias_mps2 0.0 (symmetric MOBIL;
                     the nominal 0.3 IS the asymmetric keep-right form)
  threshold_0.3      mobil.accel_threshold_mps2 0.3
  urgency_late       merge.zone_len_m 75.44 -> 37.22 (urgency onset moves
                     from the merge-edge start s=378.66 to mid-edge
                     s=416.88; candidacy window unchanged)

All 32 worlds x 18 windows; per-window LC counts over the tracked cohort
(counting identical to truth), executed-event rows, s_RMSE and delay_err
at H10/H120 for context.  TRAIN only; zero SUMO.
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
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
from partA_v2g.engine import PartAV2GEngine, load_params  # noqa: E402

ANALYSIS_ID = "WD-01-PA2G-LC1-ablation"
DT = 0.5
HS = (10, 30, 60, 120)
H_MAX = 120
RESULTS = HERE / "results" / "lc1"

VARIANTS: List[Dict] = [
    dict(name="nominal", overrides={}),
    dict(name="politeness_0.2",
         overrides={"mobil": {"politeness_p": 0.2}}),
    dict(name="politeness_1.0",
         overrides={"mobil": {"politeness_p": 1.0}}),
    dict(name="bias0_symmetric",
         overrides={"mobil": {"keep_right_bias_mps2": 0.0}}),
    dict(name="threshold_0.3",
         overrides={"mobil": {"accel_threshold_mps2": 0.3}}),
    dict(name="urgency_late",
         overrides={"merge": {"zone_len_m": 37.22}}),
]


def apply_overrides(params: dict, overrides: Dict[str, Dict]) -> dict:
    p = copy.deepcopy(params)
    for sect, kv in overrides.items():
        p[sect].update(kv)
    return p


def _worker(args):
    rank, entries, base_params = args
    EA.MAPPER = EA.ModelMapper()
    opened = set()
    f_rows = open(RESULTS / "ablation_rows.jsonl", "a", encoding="utf-8")
    f_ev = open(RESULTS / "ablation_events.jsonl", "a", encoding="utf-8")
    n_roll = int(round(H_MAX / DT))
    stats = []
    for wi, entry in enumerate(entries):
        t_a = time.perf_counter()
        log = EA.NativeLog(entry)
        opened.update(EA.OPENED_RUN_FILES)
        EA.OPENED_RUN_FILES.clear()
        for var in VARIANTS:
            params = apply_overrides(base_params, var["overrides"])
            for k in range(1, 19):
                w = log.t0 + 30.0 * k
                if w > log.t_end + 1e-9:
                    break
                j0 = int(round((w - log.t0) / 0.5))
                truth_w = log.truth_at(w)
                vehicles, _free = EA.tracked_state(
                    truth_w, params, inj_at=log.inj.get(j0))
                if not vehicles:
                    continue
                vids = sorted(vehicles)
                cav_ids = set(vids) & log.cav_ids
                hook = EA2.native_control_hook(w, params, cav_ids)
                eng = PartAV2GEngine(params, t0=w, vehicles=vehicles,
                                     control_hook=hook)
                eng.run(n_roll)
                snaps = MX.snaps_list(eng)
                truth_snaps = [log.rows.get(j0 + jj, {})
                               for jj in range(n_roll + 1)]
                tr = MX.WindowSeries(truth_snaps, vids)
                pr = MX.WindowSeries(snaps, vids)
                counts_by_H = {}
                rmse_by_H = {}
                for H in HS:
                    if w + H > log.t_end + 1e-9:
                        continue
                    n_h = int(round(H / DT))
                    st_p = pr.stats(n_h, params)
                    counts_by_H[f"H{H}"] = st_p["lc"]
                    rmse_by_H[f"H{H}"] = dict(
                        s_rmse=pr.rmse(tr, n_h)["s_rmse"],
                        delay_err=st_p["delay"] - tr.stats(n_h,
                                                           params)["delay"])
                f_rows.write(json.dumps(dict(
                    world=entry["world"], w=w, variant=var["name"],
                    n_tracked=len(vids), lc=counts_by_H,
                    metrics=rmse_by_H)) + "\n")
                for j in range(1, len(snaps)):
                    prev, cur = snaps[j - 1], snaps[j]
                    for vid in vids:
                        p_, c_ = prev.get(vid), cur.get(vid)
                        if p_ is None or c_ is None or p_[0] < 0 or \
                                c_[0] < 0 or p_[0] == c_[0]:
                            continue
                        key = {(2, 1): "merge", (1, 0): "disc_left",
                               (0, 1): "disc_right"}.get((p_[0], c_[0]))
                        if key:
                            f_ev.write(json.dumps(dict(
                                world=entry["world"], w=w,
                                variant=var["name"], vid=vid, step=j,
                                key=key, s=round(p_[1], 2),
                                is_cav=vid in cav_ids)) + "\n")
        stats.append(dict(world=entry["world"],
                          wall_s=round(time.perf_counter() - t_a, 1)))
        print(f"  [w{rank}] [{wi + 1}/{len(entries)}] {entry['world']}: "
              f"{time.perf_counter() - t_a:.1f}s", flush=True)
    f_rows.close()
    f_ev.close()
    return sorted(opened), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    n_workers = max(1, min(4, args.workers))
    t_start = time.perf_counter()
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name in ("ablation_rows.jsonl", "ablation_events.jsonl"):
        (RESULTS / name).unlink(missing_ok=True)
    worlds = json.loads(
        (HERE / "results" / "lc1_a2_worlds32.json").read_text())
    EA.OPENED_RUN_FILES.clear()
    EA.TRAIN_WORLD_NAMES.clear()
    plan = json.loads(EA2.PLAN.read_text(encoding="utf-8"))
    meta = {w["world"]: w for w in plan["worlds"]}
    entries = []
    for name in worlds:
        e = dict(meta[name])
        assert e.get("split") == "TRAIN"
        e["chunk"] = 4
        e["native_run"] = DATA_ROOT / "runs" / "campaign_stage1_chunk4" / \
            f"{name}__native_base" / "run" / "physics_steps.jsonl"
        entries.append(e)
    EA.TRAIN_WORLD_NAMES.update(worlds)
    print(f"{ANALYSIS_ID}: {len(entries)} worlds x {len(VARIANTS)} "
          f"labelled variants, workers {n_workers}", flush=True)
    params = load_params()
    if n_workers == 1:
        results = [_worker((0, entries, params))]
    else:
        ctx = mp.get_context("spawn")
        chunks = [entries[i::n_workers] for i in range(n_workers)]
        with ctx.Pool(n_workers) as pool:
            results = pool.map(_worker, [(i, chunks[i], params)
                                         for i in range(n_workers)])
    opened_all = sorted(set().union(*[set(o) for o, _s in results]))
    EA.OPENED_RUN_FILES.update(opened_all)
    EA.assert_train_only()
    (RESULTS / "ablation_opened_run_files.json").write_text(
        json.dumps(opened_all, indent=1), encoding="utf-8")
    (RESULTS / "ablation_meta.json").write_text(json.dumps(dict(
        analysis_id=ANALYSIS_ID,
        variants=[v["name"] for v in VARIANTS],
        overrides=[v["overrides"] for v in VARIANTS],
        worlds=len(entries), workers=n_workers,
        wall_clock_s=time.perf_counter() - t_start,
        label="DESCRIPTIVE ONLY -- NOT ADOPTED, NOT SELECTED; "
              "params_nominal.json untouched")), encoding="utf-8")
    print("DONE", ANALYSIS_ID,
          f"wall {time.perf_counter() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
