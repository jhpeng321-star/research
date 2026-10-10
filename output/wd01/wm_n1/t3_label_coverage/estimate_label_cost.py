"""T3 label-coverage cost estimate (TRAIN only, no writes, no simulation).

For every pack_v3 TRAIN item that has zero manifest ``outcome_rows`` this
counts the label rows a schema/2 sidecar extension would need
(``decision_t`` from the shard x taus available at ``data_end``), sums them
per (batch, arm), and times an actual label computation (run trace build +
``v0lib.window_label``) for two sample sources plus one wall sample.  It
never opens a label parquet other than to measure bytes/row, and it writes
nothing.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("E:/rd-wt/imp-window/output/wd01")
sys.path.insert(0, str(ROOT / "partB_v0data"))
sys.path.insert(0, str(ROOT))
import v0lib as V  # noqa: E402
import v04a_sources as S  # noqa: E402

PACK = Path("D:/rd-staging/wmn1_feat/pack_v3")
STAGING = Path("D:/rd-staging/wmn1_feat")
SIDE_PARQUET = STAGING / "labels_fix1_20261003T141730Z" / "train_outcomes.parquet"


def rows_for(t0, data_end, decision_t):
    t0 = float(t0)
    n = 0
    for t in decision_t:
        t = float(t)
        for tau in V.TAUS:
            if tau <= float(data_end) - t + 1e-9:
                n += 1
    return n


def main() -> int:
    man = json.loads((PACK / "manifest.json").read_text(encoding="utf-8"))
    idx = json.loads((PACK / "INDEX.json").read_text(encoding="utf-8"))
    runs = {(r.get("batch") or r["chunk"], r["world"], r["arm"]): r
            for r in idx["runs"]}
    items = [dict(i) for i in man["items"]
             if str(i.get("split", "")).upper() == "TRAIN"]

    per_group: dict = {}
    total_rows = 0
    total_items = 0
    for it in items:
        if any(int(p) >= 0 for p in it.get("outcome_rows", [])):
            continue
        batch = it.get("batch") or it["chunk"]
        key = (str(batch), str(it["arm"]))
        r = runs.get((batch, it["world"], it["arm"]))
        n = 0
        if r:
            sd = Path(str(r["shard_dir"]))
            m = json.loads((sd / "meta.json").read_text(encoding="utf-8"))
            dt = np.load(sd / "decision_t.npy")
            n = rows_for(m["t0_s"], m["data_end"], dt)
        g = per_group.setdefault(key, dict(items=0, rows=0))
        g["items"] += 1
        g["rows"] += n
        total_rows += n
        total_items += 1

    # bytes/row of the delivered /2 sidecar (TRAIN rows only; value shape)
    import pandas as pd
    side = pd.read_parquet(SIDE_PARQUET)
    bytes_per_row = SIDE_PARQUET.stat().st_size / max(len(side), 1)

    # ---- timed samples (metadata/timing only) --------------------------
    samples = []
    sample_specs = [
        ("chunk1", "w8411m2700p02t240", "fixed1", "fixed1", "base",
         "branch"),
        ("chunk1", "w8411m2700p02t240", "native", "native", "base",
         "branch"),
    ]
    for chunk, world, label, arm_sub, base_sub, typ in sample_specs:
        r = runs[(chunk, world, arm_sub)]
        sd = Path(str(r["shard_dir"]))
        m = json.loads((sd / "meta.json").read_text(encoding="utf-8"))
        t0 = float(m["t0_s"])
        dt = np.load(sd / "decision_t.npy")
        runs_root = S.RUNS[chunk]
        base_dir = runs_root / f"{world}__{base_sub}"
        arm_dir = runs_root / f"{world}__{arm_sub}"
        sc: dict = {}
        t = time.perf_counter()
        rt_base = V.build_run_trace(world, "base", base_dir, t0,
                                    t_ins_pre={}, sched_cache=sc)
        t_base = time.perf_counter() - t
        t = time.perf_counter()
        t_pre = V.build_t_ins_pre(rt_base, t0)
        rt = V.build_run_trace(world, "native", arm_dir, t0,
                               t_ins_pre=t_pre, sched_cache=sc,
                               base_arm_dir=base_dir)
        t_arm = time.perf_counter() - t
        t = time.perf_counter()
        n = 0
        for tt in dt:
            skip = abs(float(tt) - t0) < 1e-9 and typ == "base_like"
            for tau in V.TAUS:
                if tau <= rt.data_end - float(tt) + 1e-9:
                    V.window_label(rt, float(tt), tau, skip_first_tick=skip)
                    n += 1
        t_lab = time.perf_counter() - t
        samples.append(dict(
            chunk=chunk, world=world, arm=arm_sub, kind=label,
            base_trace_s=round(t_base, 2), arm_trace_s=round(t_arm, 2),
            label_loop_s=round(t_lab, 3), n_rows=n,
            total_incremental_s=round(t_arm + t_lab, 2),
            raw_physics_bytes=Path(
                m["input_physics"]).stat().st_size if Path(
                m["input_physics"]).is_file() else 0))

    # wall sample: no v0.4a base trace on the wall root
    wall_root = Path("E:/research-dense/runs/campaign_walls_pilot_chunk4")
    wr = ("walls_pilot_chunk4", "w8659m2700p10t420", "w-lo-early")
    r = runs[wr]
    sd = Path(str(r["shard_dir"]))
    m = json.loads((sd / "meta.json").read_text(encoding="utf-8"))
    t0 = float(m["t0_s"])
    dt = np.load(sd / "decision_t.npy")
    arm_dir = wall_root / f"{wr[1]}__{wr[2]}"
    sc = {}
    t = time.perf_counter()
    rt = V.build_run_trace(wr[1], "native", arm_dir, t0, t_ins_pre={},
                           sched_cache=sc, base_arm_dir=None)
    t_arm = time.perf_counter() - t
    t = time.perf_counter()
    n = 0
    for tt in dt:
        for tau in V.TAUS:
            if tau <= rt.data_end - float(tt) + 1e-9:
                V.window_label(rt, float(tt), tau, skip_first_tick=False)
                n += 1
    t_lab = time.perf_counter() - t
    samples.append(dict(
        chunk="chunk4", world=wr[1], arm=wr[2], kind="wall",
        base_trace_s=None, arm_trace_s=round(t_arm, 2),
        label_loop_s=round(t_lab, 3), n_rows=n,
        total_incremental_s=round(t_arm + t_lab, 2),
        raw_physics_bytes=Path(
            m["input_physics"]).stat().st_size if Path(
            m["input_physics"]).is_file() else 0))

    out = dict(
        n_zero_row_items=total_items,
        est_total_label_rows=total_rows,
        sidecar_bytes_per_row=round(bytes_per_row, 2),
        est_sidecar_extension_bytes=int(total_rows * bytes_per_row),
        est_sidecar_extension_mib=round(
            total_rows * bytes_per_row / 1024 / 1024, 2),
        per_group=[dict(batch=k[0], arm=k[1], items=v["items"],
                        est_rows=v["rows"]) for k, v in sorted(
                            per_group.items())],
        timed_samples=samples)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
