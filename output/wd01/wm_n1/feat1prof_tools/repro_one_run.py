"""FEAT1 1-worker reproduction runner for a single inventory run.

Resolves one inventory row exactly like ``featurize_all`` does and executes
the identical worker code path (``FA._run_one``) in-process.  Used to
re-run each run that was in flight at a worker crash ALONE with one
worker: a run that reproducibly kills its worker will terminate this
process non-zero (or hard-crash it), which the calling loop records.

Usage (cwd output/wd01/wm_n1, WD_FORBID_D=1):

  python -X utf8 feat1prof_tools/repro_one_run.py \
      --inventory inventory/inventory.parquet \
      --data-root E:/research-dense \
      --out D:/rd-staging/wmn1_feat/shards_v2 \
      --batch walls_pilot_chunk4 --run w8658m2600p10t360__w-lo-early \
      --idea-fast
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

WM_N1 = Path(__file__).resolve().parents[1]
if str(WM_N1) not in sys.path:
    sys.path.insert(0, str(WM_N1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--idea-fast", action="store_true")
    a = ap.parse_args(argv)

    import os

    if os.environ.get("WD_FORBID_D", "").strip() != "1":
        raise SystemExit("WD_FORBID_D must be 1")
    from corpus import _util as U

    U.activate_d_output_override("D:/rd-staging/wmn1_feat")
    U.lower_priority_below_normal()

    import pandas as pd

    df = pd.read_parquet(a.inventory)
    sel = df[(df["batch"].astype(str) == a.batch)
             & (df["run"].astype(str) == a.run)]
    if len(sel) != 1:
        raise SystemExit(f"inventory lookup {a.batch}/{a.run}: "
                         f"{len(sel)} rows")
    row = {k: sel.iloc[0][k] for k in sel.columns}
    from corpus import featurize_all as FA

    job = FA._resolve_run(row, Path(a.data_root), Path(a.out), force=False)
    if FA.sealed_reason(job["chunk"], job["world"]):
        raise SystemExit("sealed world refused")
    job["idea_fast"] = bool(a.idea_fast)
    job["force"] = False
    print(f"repro-one: {U.run_key(job['batch'], job['world'], job['arm'])} "
          f"split={job['split']} idea_fast={job['idea_fast']}", flush=True)
    rec = FA._run_one(job)
    print(f"repro-one OK: {rec['key']} status={rec['status']} "
          f"n_ticks={rec['record'].get('n_ticks')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
