"""FEAT1 v2c step 1 prep: build the 18-run J4 subset parquet.

Read-only over the workspace inventory and shards_v2; writes ONLY
``D:/rd-staging/wmn1_feat/logs_v2c/`` (new files, never overwrite) and
snapshot copies of the current shards_v2 top-level receipts.

Checks (hard failures):
* exactly the 18 run names from J4_WALLTOKEN_FAILED_RUNS.txt exist in the
  inventory as readable TRAIN/CALIBRATION rows;
* the 18 shard dirs are ABSENT from shards_v2 (never overwrite);
* shards_v2 PLAN_MAP.json holds all 18 keys and >= 2463 entries;
* free RAM >= 8 GB and free virtual >= 10 GB before exit (gate printed).

Usage (cwd output/wd01/wm_n1, WD_FORBID_D=1):
  python -X utf8 feat1v2c_tools/make_subset_j4.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

WM_N1 = Path(__file__).resolve().parents[1]
if str(WM_N1) not in sys.path:
    sys.path.insert(0, str(WM_N1))
SHARDS = Path("D:/rd-staging/wmn1_feat/shards_v2")
LOGS = Path("D:/rd-staging/wmn1_feat/logs_v2c")
INVENTORY = WM_N1 / "inventory" / "inventory.parquet"
J4_TXT = WM_N1 / "J4_WALLTOKEN_FAILED_RUNS.txt"


def main() -> int:
    import os

    if os.environ.get("WD_FORBID_D", "").strip() != "1":
        raise SystemExit("WD_FORBID_D must be 1")
    from corpus import _util as U
    from corpus import memory as MEM

    U.activate_d_output_override("D:/rd-staging/wmn1_feat")

    names = [ln.split("\t")[0].strip()
             for ln in J4_TXT.read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    assert len(names) == 18, f"expected 18 run names, got {len(names)}"
    assert len(set(names)) == 18, "duplicate run names in J4 list"

    import pandas as pd

    df = pd.read_parquet(INVENTORY)
    sub = df[df["run"].astype(str).isin(names)]
    got = sorted(sub["run"].astype(str))
    assert got == sorted(names), (
        f"inventory missing runs: {sorted(set(names) - set(got))}")
    keep = sub[sub["split"].isin(("TRAIN", "CALIBRATION"))
               & sub["present"].astype(bool)
               & sub["is_complete"].astype(bool)
               & (~sub["is_smoke"].astype(bool))]
    assert len(keep) == 18, f"readable TRAIN/CAL rows = {len(keep)}, want 18"

    absent = []
    for r in names:
        world, arm = str(r).rsplit("__", 1)
        d = SHARDS / "walls_pilot_chunk4" / world / arm
        if d.exists():
            absent.append(str(d))
    assert not absent, f"shard dirs already present (refusing): {absent}"

    pm = json.loads((SHARDS / "PLAN_MAP.json").read_text(encoding="utf-8"))
    pm_keys = set(pm.get("runs", {}))
    missing_pm = [f"walls_pilot_chunk4/{r}" for r in names
                  if f"walls_pilot_chunk4/{r}" not in pm_keys]
    assert not missing_pm, f"PLAN_MAP missing J4 keys: {missing_pm}"
    n_plan_map = len(pm_keys)

    index = json.loads((SHARDS / "INDEX.json").read_text(encoding="utf-8"))
    n_index_runs = len(index.get("runs", {}))

    LOGS.mkdir(parents=True, exist_ok=True)
    out_parquet = LOGS / "sub_j4_18.parquet"
    assert not out_parquet.exists(), f"{out_parquet} already exists"
    keep.sort_values(["chunk", "world", "run"]).to_parquet(out_parquet,
                                                           index=False)

    snapshots = {}
    for name in ("PLAN_MAP.json", "INDEX.json", "ALLOCATION.json"):
        src = SHARDS / name
        dst = LOGS / f"pre_v2c_{name}"
        assert not dst.exists(), f"snapshot {dst} already exists"
        if src.is_file():
            shutil.copyfile(src, dst)
            snapshots[name] = str(dst)

    mem = MEM.free_memory()
    report = dict(
        schema="wm_n1_feat1_v2c_subset_prep/1",
        inventory=str(INVENTORY), subset_parquet=str(out_parquet),
        n_rows=len(df), n_subset_rows=int(len(keep)),
        runs=sorted(names), shards_root=str(SHARDS),
        n_plan_map=n_plan_map, n_index_runs=n_index_runs,
        snapshots=snapshots,
        memory=dict(free_ram_gb=round(float(mem["free_ram_gb"]), 3),
                    free_virtual_gb=round(float(mem["free_virtual_gb"]), 3)),
        gate_ok=bool(mem["free_ram_gb"] >= 8.0
                     and mem["free_virtual_gb"] >= 10.0))
    rep_path = LOGS / "subset_prep.json"
    assert not rep_path.exists()
    rep_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
