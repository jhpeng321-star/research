"""WD-01-PA2G-LC1 truth merge diagnostics from the RAW physics rows
(zero SUMO): for every vehicle on the accel lane at a window start
(plus, flagged, lane-2 arrivals within the window), per H:

- merge_t   first tick on a mainline physical lane (1487606179_1 /
            1487606178_*) after w;
- stop_t    first tick standing (v < 0.1 m/s) at the accel-lane dead end
            (1487606179_0 @ 75.44 m);
- dwell     merge_t - lane-2 entry (edge entry and merge-edge entry);
- never-merged within H.

TRAIN only; opened run files recorded and asserted at close-out.
"""
from __future__ import annotations

import argparse
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

from diag_lc1_audit import (DT, HS, OPENED, PHYS_OF, ZONE,  # noqa: E402
                            assert_train_only, load_worlds, phys_state)

ANALYSIS_ID = "WD-01-PA2G-LC1-truth-merge"
RUNS_DIR = "campaign_stage1_chunk4"
ACCEL_EDGE = "1487606179_0"
ACCEL_LEN = 75.44
MAIN_NEXT = {"1487606179_1", "1487606178_0"}
RESULTS = HERE / "results" / "lc1"
LANE2_IDS = {"11553048#1_0", ":1780739017_0_0", ACCEL_EDGE}


def parse_world(entry: dict) -> dict:
    world = entry["world"]
    t0 = float(entry["t0_s"])
    path = DATA_ROOT / "runs" / RUNS_DIR / f"{world}__native_base" / \
        "run" / "physics_steps.jsonl"
    OPENED.add(str(path))
    seq: Dict[str, List] = {}      # vid -> [(tick, edge_id, pos, v)]
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            i = line.find('"time_s": ')
            if i < 0:
                continue
            j = i + len('"time_s": ')
            t = float(line[j:line.find(",", j)])
            if t + 1e-9 < t0:
                continue
            row = json.loads(line)
            tick = int(round((t - t0) / DT))
            for vid, v in row["vehicles"].items():
                seq.setdefault(vid, []).append(
                    (tick, v["lane_id"], float(v["lane_position_m"]),
                     float(v["speed_mps"])))
    t_end = t0 + DT * max(tk for rows in seq.values() for tk, *_ in rows)
    out = []
    for kk in range(1, 19):
        w = t0 + 30.0 * kk
        if w > t_end + 1e-9:
            break
        j0 = int(round((w - t0) / DT))
        tracked = set()
        for vid, rows in seq.items():
            for tick, lid, pos, _v in rows:
                if tick == j0:
                    hit = PHYS_OF.get(lid)
                    if hit is not None:
                        phys, s = hit
                        if ZONE[phys][0] <= s < ZONE[phys][1]:
                            tracked.add(vid)
                    break
        j_hi120 = j0 + int(round(120 / DT))
        # cohort: on lane 2 at w (tracked) or entering lane 2 later
        for vid, rows in seq.items():
            at_w = vid in tracked and any(
                tk == j0 and lid == ACCEL_EDGE for tk, lid, _p, _v in rows)
            enters = False
            if not at_w:
                for tk, lid, _p, _v in rows:
                    if j0 < tk <= j_hi120 and lid == ACCEL_EDGE and \
                            tk > 0 and rows and True:
                        prev_lid = None
                        for tk2, lid2, _p2, _v2 in rows:
                            if tk2 == tk - 1:
                                prev_lid = lid2
                                break
                        if prev_lid != ACCEL_EDGE:
                            enters = True
                            break
            if not (at_w or enters):
                continue
            idx = {tk: i for i, (tk, *_r) in enumerate(rows)}
            # lane-2 episode info
            first_l2 = None
            first_edge = None
            for i2, (tk, lid, _p, _v) in enumerate(rows):
                if lid in LANE2_IDS:
                    first_l2 = tk if first_l2 is None else first_l2
                    if lid == ACCEL_EDGE and first_edge is None:
                        first_edge = tk
            merge_tk = stop_tk = None
            for tk, lid, pos, v in rows:
                if tk <= j0:
                    continue
                if merge_tk is None and lid in MAIN_NEXT:
                    merge_tk = tk
                    break
                if lid == ACCEL_EDGE and pos >= ACCEL_LEN - 0.01 and \
                        v < 0.1 and stop_tk is None:
                    stop_tk = tk
            rec = dict(world=world, w=w, vid=vid, at_w=at_w,
                       first_l2_t=(t0 + first_l2 * DT)
                       if first_l2 is not None else None,
                       first_edge_t=(t0 + first_edge * DT)
                       if first_edge is not None else None,
                       merge_t=(t0 + merge_tk * DT)
                       if merge_tk is not None else None,
                       stop_t=(t0 + stop_tk * DT)
                       if stop_tk is not None else None)
            for H in HS:
                if w + H > t_end + 1e-9:
                    continue
                lim = t0 + (j0 + int(round(H / DT))) * DT
                rec[f"merged_H{H}"] = bool(merge_tk is not None and
                                           t0 + merge_tk * DT <= lim)
                rec[f"stopped_H{H}"] = bool(stop_tk is not None and
                                            t0 + stop_tk * DT <= lim)
            out.append(rec)
    return out


def _worker(args):
    rank, entries = args
    out = []
    for e in entries:
        t_a = time.perf_counter()
        recs = parse_world(e)
        out.extend(recs)
        print(f"  [w{rank}] {e['world']}: {len(recs)} lane-2 outcomes, "
              f"{time.perf_counter() - t_a:.1f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    n_workers = max(1, min(4, args.workers))
    t_start = time.perf_counter()
    RESULTS.mkdir(parents=True, exist_ok=True)
    entries = load_worlds()
    names = {e["world"] for e in entries}
    print(f"{ANALYSIS_ID}: {len(entries)} TRAIN worlds", flush=True)
    if n_workers == 1:
        results = [_worker((0, entries))]
    else:
        ctx = mp.get_context("spawn")
        chunks = [entries[i::n_workers] for i in range(n_workers)]
        with ctx.Pool(n_workers) as pool:
            results = pool.map(_worker, [(i, chunks[i])
                                         for i in range(n_workers)])
    all_recs = [r for sub in results for r in sub]
    with open(RESULTS / "merge_outcomes_truth.jsonl", "w",
              encoding="utf-8") as fh:
        for r in all_recs:
            fh.write(json.dumps(r) + "\n")
    assert_train_only(names)
    (RESULTS / "truthmerge_opened_run_files.json").write_text(
        json.dumps(sorted(OPENED), indent=1), encoding="utf-8")
    print("DONE", ANALYSIS_ID,
          f"records {len(all_recs)}, "
          f"wall {time.perf_counter() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
