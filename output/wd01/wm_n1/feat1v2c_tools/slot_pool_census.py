"""FEAT1 v2c step 5: slot-pool census (read-only).

For every shard under ``D:/rd-staging/wmn1_feat/shards_v2`` read
``meta.json`` (featurizer drop counters) and the small ``veh_mask.npy``
array and report:

* ``dropped_vehicle_ticks`` = ``n_dropped_slot_steps`` (the featurizer's
  ``n_dropped_concurrent`` counter: vehicle-ticks dropped because more
  than ``K_max=128`` vehicles were present);
* ``vehicle_ticks`` = occupied-slot total (``veh_mask`` sum);
* ``ticks_pool_full`` = ticks with all 128 slots occupied (strict
  upper-bound indicator set of drop-affected ticks: a drop requires the
  pool to be full, but a full pool need not have dropped).

Counts only - no outcomes, no labels, no simulation.  Writes
``SLOT_POOL_CENSUS.json`` + ``SLOT_POOL_CENSUS.md`` in ``wm_n1``.

Usage (cwd output/wd01/wm_n1, WD_FORBID_D=1):
  python -X utf8 feat1v2c_tools/slot_pool_census.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

WM_N1 = Path(__file__).resolve().parents[1]
if str(WM_N1) not in sys.path:
    sys.path.insert(0, str(WM_N1))
SHARDS = Path("D:/rd-staging/wmn1_feat/shards_v2")
OUT_JSON = WM_N1 / "SLOT_POOL_CENSUS.json"
OUT_MD = WM_N1 / "SLOT_POOL_CENSUS.md"
K_MAX = 128


def _loading_penetration(world: str) -> tuple:
    m = re.match(r"^w\d+m(\d+)p(\d+)t(\d+)$", str(world))
    if not m:
        return ("?", "?")
    return (f"m{m.group(1)}", f"p{m.group(2)}")


def main() -> int:
    import os

    if os.environ.get("WD_FORBID_D", "").strip() != "1":
        raise SystemExit("WD_FORBID_D must be 1")
    from corpus import _util as U

    U.activate_d_output_override("D:/rd-staging/wmn1_feat")

    index = json.loads((SHARDS / "INDEX.json").read_text(encoding="utf-8"))
    runs = index.get("runs", {})
    keys = sorted(runs)
    per_run = []
    for key in keys:
        rec = runs[key]
        dirp = Path(rec["shard_dir"])
        meta = json.loads((dirp / "meta.json").read_text(encoding="utf-8"))
        n_slots = int(meta.get("n_slots") or K_MAX)
        vm = np.load(dirp / "veh_mask.npy", mmap_mode="r")
        occ = np.asarray(vm).sum(axis=1)
        vehicle_ticks = int(occ.sum())
        ticks_pool_full = int((occ == n_slots).sum())
        n_ticks = int(meta.get("n_ticks") or occ.shape[0])
        assert occ.shape[0] == n_ticks, f"{key}: veh_mask ticks != n_ticks"
        loading, pen = _loading_penetration(rec.get("world") or
                                            meta.get("world"))
        per_run.append(dict(
            key=key, batch=meta.get("batch") or rec.get("batch"),
            split=meta.get("split") or rec.get("split"),
            loading=loading, penetration=pen,
            n_ticks=n_ticks, n_slots=n_slots,
            n_vehicles=int(meta.get("n_vehicles") or 0),
            n_dropped_slots=int(meta.get("n_dropped_slots") or 0),
            dropped_vehicle_ticks=int(
                meta.get("n_dropped_slot_steps") or 0),
            vehicle_ticks=vehicle_ticks,
            ticks_pool_full=ticks_pool_full))

    def agg(rows):
        n = len(rows)
        dropped = sum(r["dropped_vehicle_ticks"] for r in rows)
        veh = sum(r["vehicle_ticks"] for r in rows)
        full = sum(r["ticks_pool_full"] for r in rows)
        ticks = sum(r["n_ticks"] for r in rows)
        n_dropped_runs = sum(1 for r in rows
                             if r["dropped_vehicle_ticks"] > 0)
        return dict(n_runs=n, runs_with_drops=n_dropped_runs,
                    dropped_vehicle_ticks=dropped,
                    vehicle_ticks=veh,
                    drop_share_of_vehicle_ticks=(
                        round(dropped / veh, 6) if veh else None),
                    ticks_total=ticks, ticks_pool_full=full,
                    pool_full_tick_share=(
                        round(full / ticks, 6) if ticks else None))

    def group(field):
        out = {}
        for r in per_run:
            out.setdefault(str(r[field]), []).append(r)
        return {k: agg(v) for k, v in sorted(out.items())}

    overall = agg(per_run)
    report = dict(
        schema="wm_n1_slot_pool_census/1",
        k_max=K_MAX,
        source=str(SHARDS),
        metric_definitions=dict(
            dropped_vehicle_ticks="meta.json n_dropped_slot_steps "
                                  "(featurizer n_dropped_concurrent): "
                                  "vehicle-ticks dropped because more than "
                                  "K_max=128 vehicles were present",
            vehicle_ticks="sum of veh_mask (rostered vehicle-ticks)",
            ticks_pool_full="ticks with all 128 slots occupied (upper-bound "
                            "indicator set of drop-affected ticks; a drop "
                            "requires a full pool, a full pool need not "
                            "have dropped)",
            counts_only="no outcome/label data read or reported"),
        overall=overall,
        by_batch=group("batch"),
        by_loading=group("loading"),
        by_penetration=group("penetration"),
        by_loading_penetration={},
        runs=per_run)
    pairs = {}
    for r in per_run:
        pairs.setdefault(f"{r['loading']}/{r['penetration']}", []).append(r)
    report["by_loading_penetration"] = {k: agg(v)
                                        for k, v in sorted(pairs.items())}

    top = sorted(per_run, key=lambda r: -r["dropped_vehicle_ticks"])[:20]
    assert not OUT_JSON.exists(), f"{OUT_JSON} already exists"
    OUT_JSON.write_text(json.dumps(report, indent=1), encoding="utf-8")

    L = []
    L.append("# SLOT-POOL CENSUS (FEAT1 pack_v2c, WM-N1 shards_v2)\n")
    L.append("Read-only census over every featurized run: how many "
             "vehicle-ticks the featurizer dropped because more than "
             f"K_max={K_MAX} concurrent vehicles were present (frozen "
             "128-slot pool), from the shard ``meta.json`` drop counters "
             "(``n_dropped_slot_steps`` = featurizer "
             "``n_dropped_concurrent``). Counts only; no outcomes.\n")
    L.append(f"- runs: {overall['n_runs']} "
             f"(runs with drops: {overall['runs_with_drops']})")
    L.append(f"- dropped vehicle-ticks (total): "
             f"{overall['dropped_vehicle_ticks']:,}")
    L.append(f"- rostered vehicle-ticks (total): "
             f"{overall['vehicle_ticks']:,}")
    if overall["drop_share_of_vehicle_ticks"] is not None:
        L.append(f"- dropped / rostered vehicle-ticks: "
                 f"{overall['drop_share_of_vehicle_ticks']:.6%}")
    L.append(f"- ticks total: {overall['ticks_total']:,}; ticks with a full "
             f"128-slot pool: {overall['ticks_pool_full']:,} "
             f"({overall['pool_full_tick_share']:.6%} of ticks, upper-bound "
             "indicator set of drop-affected ticks)\n")
    L.append("## By batch\n")
    L.append("| batch | runs | runs w/ drops | dropped veh-ticks | "
             "rostered veh-ticks | drop share | pool-full ticks |")
    L.append("|---|---|---|---|---|---|---|")
    for k, v in report["by_batch"].items():
        L.append(f"| {k} | {v['n_runs']} | {v['runs_with_drops']} | "
                 f"{v['dropped_vehicle_ticks']:,} | "
                 f"{v['vehicle_ticks']:,} | "
                 f"{v['drop_share_of_vehicle_ticks']:.6%} | "
                 f"{v['ticks_pool_full']:,} |")
    L.append("\n## By loading (mainline demand, mNNNN)\n")
    L.append("| loading | runs | runs w/ drops | dropped veh-ticks | "
             "rostered veh-ticks | drop share | pool-full ticks |")
    L.append("|---|---|---|---|---|---|---|")
    for k, v in report["by_loading"].items():
        L.append(f"| {k} | {v['n_runs']} | {v['runs_with_drops']} | "
                 f"{v['dropped_vehicle_ticks']:,} | "
                 f"{v['vehicle_ticks']:,} | "
                 f"{v['drop_share_of_vehicle_ticks']:.6%} | "
                 f"{v['ticks_pool_full']:,} |")
    L.append("\n## By CAV penetration (pNN)\n")
    L.append("| penetration | runs | runs w/ drops | dropped veh-ticks | "
             "rostered veh-ticks | drop share | pool-full ticks |")
    L.append("|---|---|---|---|---|---|---|")
    for k, v in report["by_penetration"].items():
        L.append(f"| {k} | {v['n_runs']} | {v['runs_with_drops']} | "
                 f"{v['dropped_vehicle_ticks']:,} | "
                 f"{v['vehicle_ticks']:,} | "
                 f"{v['drop_share_of_vehicle_ticks']:.6%} | "
                 f"{v['ticks_pool_full']:,} |")
    L.append("\n## By loading x penetration\n")
    L.append("| loading/penetration | runs | runs w/ drops | "
             "dropped veh-ticks | rostered veh-ticks | drop share |")
    L.append("|---|---|---|---|---|---|")
    for k, v in report["by_loading_penetration"].items():
        L.append(f"| {k} | {v['n_runs']} | {v['runs_with_drops']} | "
                 f"{v['dropped_vehicle_ticks']:,} | "
                 f"{v['vehicle_ticks']:,} | "
                 f"{v['drop_share_of_vehicle_ticks']:.6%} |")
    L.append("\n## Top 20 runs by dropped vehicle-ticks\n")
    L.append("| run key | n_ticks | n_vehicles | dropped veh-ticks | "
             "rostered veh-ticks | pool-full ticks |")
    L.append("|---|---|---|---|---|---|")
    for r in top:
        L.append(f"| {r['key']} | {r['n_ticks']:,} | {r['n_vehicles']:,} | "
                 f"{r['dropped_vehicle_ticks']:,} | "
                 f"{r['vehicle_ticks']:,} | {r['ticks_pool_full']:,} |")
    L.append("\nPer-run counts (counts only) are in "
             "`SLOT_POOL_CENSUS.json` under `runs`.\n")
    assert not OUT_MD.exists(), f"{OUT_MD} already exists"
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(json.dumps(dict(n_runs=overall["n_runs"],
                          runs_with_drops=overall["runs_with_drops"],
                          dropped_vehicle_ticks=overall[
                              "dropped_vehicle_ticks"],
                          vehicle_ticks=overall["vehicle_ticks"],
                          ticks_pool_full=overall["ticks_pool_full"],
                          out_json=str(OUT_JSON), out_md=str(OUT_MD)),
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
