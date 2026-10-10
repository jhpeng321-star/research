"""FEAT3 stage-2 step 1: per-shard verify + slot-pool census V3 (read-only).

1. ``features.writer.verify_index`` over every shard of
   ``D:/rd-staging/wmn1_feat/shards_v3`` (DONE.json output-sha verification,
   the featurizer's own resume check) plus a per-shard meta self-description
   check: schema ``wm_n1_features_format_v3/1``, ``format == "v3"``,
   ``k_max == 192``.
2. Slot-pool census at K_max=192 from the shard ``meta.json`` drop counters
   (``n_dropped_slot_steps`` = featurizer ``n_dropped_concurrent``), summed
   by batch / loading / penetration (+ loading x penetration).  Counts only;
   no outcomes, labels or summaries are read.  Writes
   ``SLOT_POOL_CENSUS_V3.json`` + ``SLOT_POOL_CENSUS_V3.md`` in ``wm_n1``
   and compares the measured residual drop share with the A5 census
   residual estimate (pooled 0.001399, ``KCENSUS_RESULT.md``).

Usage (cwd output/wd01/wm_n1, WD_FORBID_D=1):
  python -X utf8 census_kmax_tools/shard_verify_census_v3.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

WM_N1 = Path(__file__).resolve().parents[1]
if str(WM_N1) not in sys.path:
    sys.path.insert(0, str(WM_N1))
SHARDS = Path("D:/rd-staging/wmn1_feat/shards_v3")
OUT_JSON = WM_N1 / "SLOT_POOL_CENSUS_V3.json"
OUT_MD = WM_N1 / "SLOT_POOL_CENSUS_V3.md"
K_MAX = 192
EXPECTED_K_MAX = 192
EXPECTED_FORMAT = "v3"
EXPECTED_SCHEMA = "wm_n1_features_format_v3/1"
A5_RESIDUAL_ESTIMATE = 0.001399      # KCENSUS_RESULT.md pooled residual @K=192
A5_RESIDUAL_DROPPED = 57157
A5_RESIDUAL_ROSTERED = 40855639


def main() -> int:
    if os.environ.get("WD_FORBID_D", "").strip() != "1":
        raise SystemExit("WD_FORBID_D must be 1")
    from corpus import _util as U

    U.activate_d_output_override("D:/rd-staging/wmn1_feat")
    from features import writer as W

    index = json.loads((SHARDS / "INDEX.json").read_text(encoding="utf-8"))
    runs = index.get("runs", {})
    keys = sorted(runs)
    print(f"[verify] shard verify over {len(keys)} registered shards ...")
    t0 = time.monotonic()
    rep = W.verify_index(SHARDS, index)
    print(f"[verify] ok={rep['ok']} n_runs={rep['n_runs']} "
          f"errors={len(rep['errors'])} wall={time.monotonic() - t0:.1f}s")
    for e in rep["errors"][:20]:
        print(f"[verify] ERROR {e}")
    if not rep["ok"]:
        raise SystemExit(f"shard verify FAILED ({len(rep['errors'])} errors)")

    per_run = []
    meta_bad = []
    t1 = time.monotonic()
    for i, key in enumerate(keys):
        rec = runs[key]
        dirp = Path(rec["shard_dir"])
        meta = json.loads((dirp / "meta.json").read_text(encoding="utf-8"))
        if (meta.get("schema") != EXPECTED_SCHEMA
                or meta.get("format") != EXPECTED_FORMAT
                or int(meta.get("k_max") or 0) != EXPECTED_K_MAX):
            meta_bad.append(f"{key}: schema={meta.get('schema')!r} "
                            f"format={meta.get('format')!r} "
                            f"k_max={meta.get('k_max')!r}")
        n_slots = int(meta.get("n_slots") or 0)
        vm = np.load(dirp / "veh_mask.npy", mmap_mode="r")
        occ = np.asarray(vm).sum(axis=1)
        vehicle_ticks = int(occ.sum())
        ticks_pool_full = int((occ == K_MAX).sum())
        n_ticks = int(meta.get("n_ticks") or occ.shape[0])
        assert occ.shape[0] == n_ticks, f"{key}: veh_mask ticks != n_ticks"
        world = rec.get("world") or meta.get("world")
        m = re.match(r"^w\d+m(\d+)p(\d+)t(\d+)$", str(world))
        loading = f"m{m.group(1)}" if m else "?"
        pen = f"p{m.group(2)}" if m else "?"
        per_run.append(dict(
            key=key, batch=meta.get("batch") or rec.get("batch"),
            split=meta.get("split") or rec.get("split"),
            loading=loading, penetration=pen,
            n_ticks=n_ticks, n_slots=n_slots, k_max=K_MAX,
            n_vehicles=int(meta.get("n_vehicles") or 0),
            n_dropped_slots=int(meta.get("n_dropped_slots") or 0),
            dropped_vehicle_ticks=int(
                meta.get("n_dropped_slot_steps") or 0),
            vehicle_ticks=vehicle_ticks,
            ticks_pool_full=ticks_pool_full))
        if (i + 1) % 500 == 0:
            print(f"[census] {i + 1}/{len(keys)} "
                  f"wall={time.monotonic() - t1:.1f}s")
    if meta_bad:
        for e in meta_bad[:20]:
            print(f"[meta] MISMATCH {e}")
        raise SystemExit(f"meta format/k_max check FAILED ({len(meta_bad)})")
    print(f"[meta] all {len(keys)} shards self-describe "
          f"schema={EXPECTED_SCHEMA} format={EXPECTED_FORMAT} "
          f"k_max={EXPECTED_K_MAX}")
    print(f"[census] veh_mask scan done wall={time.monotonic() - t1:.1f}s")

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
    pairs = {}
    for r in per_run:
        pairs.setdefault(f"{r['loading']}/{r['penetration']}", []).append(r)
    report = dict(
        schema="wm_n1_slot_pool_census_v3/1",
        k_max=K_MAX,
        source=str(SHARDS),
        shard_verify=dict(ok=rep["ok"], n_runs=rep["n_runs"],
                          n_errors=len(rep["errors"])),
        metric_definitions=dict(
            dropped_vehicle_ticks="meta.json n_dropped_slot_steps "
                                  "(featurizer n_dropped_concurrent): "
                                  "vehicle-ticks dropped because more than "
                                  "K_max=192 vehicles were present",
            vehicle_ticks="sum of veh_mask (rostered vehicle-ticks)",
            ticks_pool_full="ticks with all 192 slots occupied (upper-bound "
                            "indicator set of drop-affected ticks; a drop "
                            "requires a full pool, a full pool need not "
                            "have dropped)",
            counts_only="no outcome/label data read or reported"),
        a5_residual_estimate=dict(
            source="KCENSUS_RESULT.md pooled residual drop share at K=192",
            dropped_vehicle_ticks=A5_RESIDUAL_DROPPED,
            rostered_vehicle_ticks=A5_RESIDUAL_ROSTERED,
            share=A5_RESIDUAL_ESTIMATE),
        measured_residual_share=round(
            overall["dropped_vehicle_ticks"] / overall["vehicle_ticks"], 6)
        if overall["vehicle_ticks"] else None,
        overall=overall,
        by_batch=group("batch"),
        by_loading=group("loading"),
        by_penetration=group("penetration"),
        by_loading_penetration={k: agg(v) for k, v in sorted(pairs.items())},
        runs=per_run)
    if overall["vehicle_ticks"]:
        measured = overall["dropped_vehicle_ticks"] / overall["vehicle_ticks"]
        report["a5_residual_delta"] = round(measured - A5_RESIDUAL_ESTIMATE, 6)

    assert not OUT_JSON.exists(), f"{OUT_JSON} already exists"
    OUT_JSON.write_text(json.dumps(report, indent=1), encoding="utf-8")

    L = []
    L.append("# SLOT-POOL CENSUS V3 (FEAT3 pack_v3, WM-N1 shards_v3, K_max=192)\n")
    L.append("Read-only census over every featurized run: how many "
             "vehicle-ticks the featurizer still dropped because more than "
             f"K_max={K_MAX} concurrent vehicles were present (192-slot "
             "pool), from the shard ``meta.json`` drop counters "
             "(``n_dropped_slot_steps`` = featurizer "
             "``n_dropped_concurrent``). Counts only; no outcomes.\n")
    L.append(f"- shard verify: ok={rep['ok']} over {rep['n_runs']} shards "
             f"(DONE.json output-sha verification; all shards self-describe "
             f"format={EXPECTED_FORMAT} k_max={EXPECTED_K_MAX})")
    L.append(f"- runs: {overall['n_runs']} "
             f"(runs with drops: {overall['runs_with_drops']})")
    L.append(f"- dropped vehicle-ticks (total): "
             f"{overall['dropped_vehicle_ticks']:,}")
    L.append(f"- rostered vehicle-ticks (total): "
             f"{overall['vehicle_ticks']:,}")
    if overall["drop_share_of_vehicle_ticks"] is not None:
        L.append(f"- dropped / rostered vehicle-ticks (measured residual "
                 f"@K=192): {overall['drop_share_of_vehicle_ticks']:.6%}")
    L.append(f"- ticks total: {overall['ticks_total']:,}; ticks with a full "
             f"192-slot pool: {overall['ticks_pool_full']:,} "
             f"({overall['pool_full_tick_share']:.6%} of ticks, upper-bound "
             "indicator set of drop-affected ticks)")
    L.append(f"- A5 census residual estimate (pooled, 300-run sample): "
             f"{A5_RESIDUAL_ESTIMATE:.6%} "
             f"({A5_RESIDUAL_DROPPED:,} / {A5_RESIDUAL_ROSTERED:,}); "
             f"delta measured - estimate: "
             f"{report.get('a5_residual_delta', 0):+.6%}\n")
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
    L.append("\nPer-run counts (counts only) are in "
             "`SLOT_POOL_CENSUS_V3.json` under `runs`.\n")
    assert not OUT_MD.exists(), f"{OUT_MD} already exists"
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(json.dumps(dict(shard_verify_ok=rep["ok"],
                          n_runs=overall["n_runs"],
                          runs_with_drops=overall["runs_with_drops"],
                          dropped_vehicle_ticks=overall[
                              "dropped_vehicle_ticks"],
                          vehicle_ticks=overall["vehicle_ticks"],
                          ticks_pool_full=overall["ticks_pool_full"],
                          measured_residual_share=report[
                              "measured_residual_share"],
                          a5_estimate=A5_RESIDUAL_ESTIMATE,
                          out_json=str(OUT_JSON), out_md=str(OUT_MD)),
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
