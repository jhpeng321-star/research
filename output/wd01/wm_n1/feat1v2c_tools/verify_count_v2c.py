"""FEAT1 v2c step 2: verified-count assertion for shards_v2.

* Recompute the featurize-all selection from the workspace inventory
  (readable TRAIN/CALIBRATION rows) and read
  ``inventory/non_featurized_traincal.json`` -> expected count and the
  printed arithmetic (readable inventory - non-featurized = expected).
* Verify EVERY leaf shard dir (batch/world/arm) with the frozen
  quarantine-scan verifier (v1 DONE.json output shas + Idea A receipt
  shas/bytes) - read-only, nothing is moved.
* Assert: verified leaf dirs == expected == INDEX.json runs.

Usage (cwd output/wd01/wm_n1, WD_FORBID_D=1):
  python -X utf8 feat1v2c_tools/verify_count_v2c.py
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WM_N1 = Path(__file__).resolve().parents[1]
if str(WM_N1) not in sys.path:
    sys.path.insert(0, str(WM_N1))
SHARDS = Path("D:/rd-staging/wmn1_feat/shards_v2")
NON_FEAT = WM_N1 / "inventory" / "non_featurized_traincal.json"
INVENTORY = WM_N1 / "inventory" / "inventory.parquet"
OUT = Path("D:/rd-staging/wmn1_feat/logs_v2c/assertion_v2c.json")


def main() -> int:
    import os

    if os.environ.get("WD_FORBID_D", "").strip() != "1":
        raise SystemExit("WD_FORBID_D must be 1")
    from corpus import _util as U

    U.activate_d_output_override("D:/rd-staging/wmn1_feat")

    import pandas as pd

    from feat1prof_tools.quarantine_scan_v3 import verify_dir

    df = pd.read_parquet(INVENTORY)
    n_inventory = int(len(df))
    keep = df[df["split"].isin(("TRAIN", "CALIBRATION"))
              & df["present"].astype(bool)
              & df["is_complete"].astype(bool)
              & (~df["is_smoke"].astype(bool))]
    n_selected = int(len(keep))
    non_feat = json.loads(NON_FEAT.read_text(encoding="utf-8"))
    rows = non_feat if isinstance(non_feat, list) else non_feat.get("runs",
                                                                    non_feat)
    n_non_featurized = len(rows) if hasattr(rows, "__len__") else int(rows)

    arm_dirs = []
    for batch in sorted(p for p in SHARDS.iterdir() if p.is_dir()):
        for world in sorted(p for p in batch.iterdir() if p.is_dir()):
            for arm in sorted(p for p in world.iterdir() if p.is_dir()):
                arm_dirs.append(arm)
    n_dirs = len(arm_dirs)

    n_ok = 0
    bad = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, d in enumerate(arm_dirs):
            verdict, _ = verify_dir(d, pool)
            if verdict == "ok":
                n_ok += 1
            else:
                bad.append(dict(dir=str(d), reason=verdict))
            if (i + 1) % 500 == 0:
                print(f"  verified {i + 1}/{n_dirs} ok={n_ok} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    index = json.loads((SHARDS / "INDEX.json").read_text(encoding="utf-8"))
    n_index = len(index.get("runs", {}))
    arithmetic = (f"{n_inventory} inventory rows - {n_non_featurized} "
                  f"non-featurized = {n_inventory - n_non_featurized} "
                  f"expected (select_runs readable TRAIN/CAL = "
                  f"{n_selected}); verified shard dirs = {n_ok}; "
                  f"INDEX.json runs = {n_index}")
    ok = (n_ok == n_dirs == n_index == n_selected
          and n_selected == n_inventory - n_non_featurized)
    report = dict(
        schema="wm_n1_feat1_v2c_count_assertion/1",
        shards_root=str(SHARDS),
        n_inventory_rows=n_inventory,
        n_non_featurized=n_non_featurized,
        n_selected=n_selected,
        n_leaf_dirs=n_dirs,
        n_verified_ok=n_ok,
        n_index_runs=n_index,
        arithmetic=arithmetic,
        ok=ok,
        bad=bad)
    assert not OUT.exists(), f"{OUT} already exists"
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(arithmetic)
    print(json.dumps({k: report[k] for k in
                      ("n_inventory_rows", "n_non_featurized", "n_selected",
                       "n_leaf_dirs", "n_verified_ok", "n_index_runs", "ok")},
                     indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
