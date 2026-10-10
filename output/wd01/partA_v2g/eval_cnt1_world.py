"""WD-01-PA2G-CNT1 per-world runner (crash-isolated execution mode).

The panel machine reaps long-lived python processes (attempts 1-5 of
eval_cnt1.py: two detached multi-process runs and one in-process run
were hard-killed with empty stderr mid-panel; see results/eval_cnt1/
RUN*_KILLED_NOTE.txt).  This driver processes exactly ONE panel world
per invocation with the identical registered code path
(eval_cnt1.window_rows_cnt, all four models, 18 windows), writing
    results/eval_cnt1/worlds/<world>.jsonl        (raw rows)
    results/eval_cnt1/worlds/<world>.opened.json  (opened run files)
and SKIPS worlds whose row file already exists (idempotent resume).
A kill loses at most the current world and nothing result-informed is
ever re-decided.  Row counts, wall times and the per-world kill
history are disclosed in the report/ledger.

TRAIN guard: the split is checked from the chunk-4 plan before any run
file is opened (eval_cnt1.load_panel_worlds); close-out asserts the
opened-file union touches TRAIN panel worlds only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WD01 = HERE.parent
for _p in (str(WD01), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("WD01_DATA_ROOT",
                      str(Path("E:/rd-wt/wd01/output/wd01")))

from partA_v2g import arrivals as ARR  # noqa: E402
from partA_v2g.engine import load_params, param_set_id  # noqa: E402
from partA_v2g import eval_a as EA  # noqa: E402
from partA_v2g import eval_a2 as A2  # noqa: E402
from partA_v2g import eval_cnt1 as C1  # noqa: E402

EXPECTED_ROWS_PER_WORLD = 280  # 16 windows x 4 models x 4 H - 2 x 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    args = ap.parse_args()
    C1.set_below_normal()
    out_dir = HERE / "results" / "eval_cnt1" / "worlds"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"{args.world}.jsonl"
    if rows_path.exists():
        n = sum(1 for _ in open(rows_path, encoding="utf-8"))
        if n == EXPECTED_ROWS_PER_WORLD:
            print(f"SKIP {args.world}: complete ({n} rows)")
            return
    t0w = time.perf_counter()
    params = load_params()
    EA.MAPPER = EA.ModelMapper()
    EA.OPENED_RUN_FILES.clear()
    worlds = {w["world"]: w for w in C1.load_panel_worlds()}
    entry = worlds[args.world]
    replica = EA.V1P3OReplica()
    log = EA.NativeLog(entry)
    deps = ARR.read_departures(entry["native_run"])
    assert deps["other_edge"] == 0 and deps["repeat_ids"] == 0
    n_main = sum(1 for _t, e in deps["departures"] if e == "main")
    n_ramp = sum(1 for _t, e in deps["departures"] if e == "ramp")
    assert n_main == deps["bc_main"] and n_ramp == deps["bc_ramp"]
    rows_all = []
    for k in range(1, 19):
        w = log.t0 + 30.0 * k
        if w > log.t_end + 1e-9:
            break
        rows_all += C1.window_rows_cnt(params, log, entry, w,
                                       deps["departures"], replica)
    assert len(rows_all) == EXPECTED_ROWS_PER_WORLD, len(rows_all)
    tmp = rows_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows_all:
            fh.write(json.dumps(r) + "\n")
    tmp.replace(rows_path)
    (out_dir / f"{args.world}.opened.json").write_text(
        json.dumps(sorted(set(EA.OPENED_RUN_FILES)), indent=1),
        encoding="utf-8")
    print(f"OK {args.world}: {len(rows_all)} rows, "
          f"departures main={n_main} ramp={n_ramp}, "
          f"{time.perf_counter() - t0w:.1f}s")


if __name__ == "__main__":
    main()
