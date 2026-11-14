"""v0.2 sanity checks over the consolidated V0 dataset (ZERO SIMULATION).

1. Chunk-2 row contract: every chunk-2 label/TD row must carry
   chunk_id='chunk2', cav_speed_mode=27, pre_fix=False (post-fix fleet).
2. Chunk-2 speed-mode resolution census: how many sources took the
   receipt, the observed physics values, or the --default-speed-mode 27
   fallback ('config'/'config-default').
3. Quick sanity table (printed, one-line note only): TRAIN worlds,
   NATIVE source -- mean/median total delay label per demand at tau=30
   and tau=300, chunk 2 vs chunk 1.

Writes results/sanity_v02.json; exit 1 on any contract violation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TAUS = (30.0, 300.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args(argv)
    out = Path(args.out)
    man = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))

    def entries(chunk):
        return {k: e for k, e in man["worlds"].items()
                if e["chunk_id"] == chunk}

    # ---- 1+2: chunk-2 contract + speed-mode census --------------------
    c2 = entries("chunk2")
    census = {}
    n_rows = 0
    bad = []
    for key, e in sorted(c2.items()):
        census[e["speed_mode_source"]] = \
            census.get(e["speed_mode_source"], 0) + 1
        for rel in [f for f in e["files"] if f.endswith(".parquet")]:
            df = pd.read_parquet(out / rel)
            n_rows += len(df)
            for col, want in (("chunk_id", "chunk2"), ("cav_speed_mode", 27),
                              ("pre_fix", False)):
                if not df[col].eq(want).all():
                    bad.append((key, rel, col,
                                df[col].unique()[:4].tolist()))
    n_fallback = census.get("config", 0) + census.get("config-default", 0)
    print(f"chunk-2 sources: {len(c2)}; speed-mode resolution census: "
          f"{census} -> fallback (config default 27): {n_fallback}")
    print(f"chunk-2 parquet rows checked: {n_rows}; contract violations "
          f"(chunk_id=='chunk2', cav_speed_mode==27, pre_fix==False on "
          f"EVERY row): {len(bad)}")
    for b in bad[:10]:
        print(f"  VIOLATION {b}")

    # ---- 3: sanity table (TRAIN, native, per demand) -------------------
    frames = []
    for chunk in ("chunk1", "chunk2"):
        for key, e in entries(chunk).items():
            if e["split"] != "TRAIN" or e["source"] != "native":
                continue
            rel = next(f for f in e["files"] if f.endswith("_labels.parquet"))
            df = pd.read_parquet(out / rel)
            frames.append(df[df["tau"].isin(TAUS)])
    lab = pd.concat(frames, ignore_index=True)
    g = lab.groupby(["chunk_id", "demand_veh_h", "tau"])[
        "total_delay_ref_s"].agg(["mean", "median"])

    print("\nsanity table -- TRAIN worlds, NATIVE source, total delay "
          "label (veh.s) per demand, chunk 2 vs chunk 1:")
    header = (f"{'demand':>7} | {'tau':>3} | {'c1_mean':>8} "
              f"{'c1_median':>9} | {'c2_mean':>8} {'c2_median':>9}")
    def cell(r):
        return f"{r['mean']:8.1f} {r['median']:9.1f}" if r is not None \
            else f"{'-':>8} {'-':>9}"

    print(header)
    print("-" * len(header))
    table = []
    for dem in sorted(lab["demand_veh_h"].unique()):
        for tau in TAUS:
            r1 = g.loc[("chunk1", dem, tau)] if (
                "chunk1", dem, tau) in g.index else None
            r2 = g.loc[("chunk2", dem, tau)] if (
                "chunk2", dem, tau) in g.index else None
            table.append(dict(
                demand_veh_h=int(dem), tau=int(tau),
                c1_mean=None if r1 is None else round(float(r1["mean"]), 1),
                c1_median=None if r1 is None else round(float(r1["median"]), 1),
                c2_mean=None if r2 is None else round(float(r2["mean"]), 1),
                c2_median=None if r2 is None else round(float(r2["median"]), 1)))
            print(f"{dem:7d} | {tau:3.0f} | {cell(r1)} | "
                  f"{cell(r2)}")
    print("note: descriptive only -- chunk 2 is the post-fix fleet "
          "(speedMode 27) on the WM-47 demand grid; differences mix "
          "fleet-fix and demand-mix effects and are not interpreted here.")

    (out / "results").mkdir(exist_ok=True)
    (out / "results" / "sanity_v02.json").write_text(json.dumps(dict(
        schema="wd01_partB_v0data_sanity/1",
        chunk2_speed_mode_census=census, chunk2_fallback_sources=n_fallback,
        chunk2_rows_checked=n_rows, chunk2_contract_violations=len(bad),
        sanity_train_native=table), indent=1), encoding="utf-8")
    print("\nsanity_v02.json written")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
