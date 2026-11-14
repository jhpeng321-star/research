"""T1a: rebuild all 88 per-run track arrays from the regenerated runs with the
original builder logic, then compare array-by-array against the surviving
tracks/*.npz.  Also caches per-tick states for T1b/T2.

Outputs:
  results/rebuilt_tracks/<split>/<tag>.npz   (rebuilt, same format as source)
  results/states_cache/<split>/<tag>.npz     (per-tick d,v,lane_group,vid)
  results/rebuild_compare.json               (88-row comparison table)

Read-only on the research-dense source repo (wdpaths.RESEARCH_DENSE_ROOT,
default E:/research-dense).  No simulation.
"""
import json
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wdpaths  # noqa: E402
import wd1lib as L  # noqa: E402

W = L.wm()
ARRAYS = ("v_self", "v_leader", "gap", "lane_group", "free", "action", "dv", "t")


def rebuild_one(row):
    t0 = time.time()
    rdir = Path(wdpaths.remap(row["run_dir"]))
    out = dict(tag=row["tag"], split=row["split"], pref=row["pref"])
    try:
        info = L.states_to_cache(rdir, L.RESULTS / "states_cache" /
                                 row["split"] / f"{row['tag']}.npz", w=W)
        states, _ = L.cache_to_states(L.RESULTS / "states_cache" /
                                      row["split"] / f"{row['tag']}.npz"), None
        pairs = W.build_pairs(states, row["action_idx"])
        rec = dict(n_pairs=len(pairs["dv"]),
                   n_free=int(pairs["free"].sum()),
                   n_vehicles_distinct=len(set(pairs["vid"].tolist())),
                   mean_abs_dv=float(np.abs(pairs["dv"]).mean()),
                   n_ticks=info["n_ticks"], missing_ticks=info["missing"])
        odir = L.RESULTS / "rebuilt_tracks" / row["split"]
        odir.mkdir(parents=True, exist_ok=True)
        meta = dict(tag=row["tag"], split=row["split"], arm=row["arm"],
                    action_idx=row["action_idx"], seed=row["seed"],
                    loading=row["loading"], n_pairs=rec["n_pairs"],
                    n_free=rec["n_free"], window=[W.TICK_LO, W.TICK_HI],
                    gap_max=W.GAP_MAX)
        np.savez_compressed(
            odir / f"{row['tag']}.npz",
            v_self=pairs["v_self"].astype(np.float32),
            v_leader=pairs["v_leader"].astype(np.float32),
            gap=pairs["gap"].astype(np.float32),
            lane_group=pairs["lane_group"].astype(np.int8),
            free=pairs["free"].astype(np.int8),
            action=pairs["action"].astype(np.int8),
            dv=pairs["dv"].astype(np.float32),
            t=pairs["t"].astype(np.int16),
            meta=np.array(json.dumps(meta)),
        )
        out.update(rec, seconds=round(time.time() - t0, 1), error=None)
    except Exception as e:  # noqa: BLE001 - record and continue
        out.update(error=f"{type(e).__name__}: {e}")
    return out


def compare_one(row):
    tag, split = row["tag"], row["split"]
    surv = np.load(L.WM / "tracks" / split / f"{tag}.npz", allow_pickle=True)
    reb = np.load(L.RESULTS / "rebuilt_tracks" / split / f"{tag}.npz",
                  allow_pickle=True)
    out = dict(tag=tag, split=split)
    meta_s = json.loads(str(surv["meta"]))
    meta_r = json.loads(str(reb["meta"]))
    out["meta_equal"] = all(meta_s.get(k) == meta_r.get(k) for k in
                            ("tag", "split", "arm", "action_idx", "seed",
                             "n_pairs", "n_free", "window"))
    arr_res = {}
    all_eq = True
    for k in ARRAYS:
        a, b = surv[k], reb[k]
        if a.shape != b.shape:
            arr_res[k] = dict(shape_a=list(a.shape), shape_b=list(b.shape),
                              exact=False, max_abs_diff=None)
            all_eq = False
            continue
        eq = np.array_equal(a, b)
        mad = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) \
            if a.size and not eq else (0.0 if eq else None)
        arr_res[k] = dict(n=a.size, exact=bool(eq), max_abs_diff=mad)
        all_eq &= bool(eq)
    out["arrays"] = arr_res
    out["all_exact"] = bool(all_eq and out["meta_equal"])
    out["n_pairs_surviving"] = int(len(surv["dv"]))
    return out


def main():
    t0 = time.time()
    m = L.load_json(L.RESULTS / "tag_run_map.json")
    rows = [r for r in m["rows"]]
    print(f"{len(rows)} runs; rebuilding with {min(8, len(rows))} workers",
          flush=True)
    with Pool(processes=8) as pool:
        recs = pool.map(rebuild_one, rows)
    errs = [r for r in recs if r["error"]]
    print(f"rebuilt {len(recs) - len(errs)}/{len(recs)}, {len(errs)} errors "
          f"({time.time() - t0:.0f}s)", flush=True)
    for r in errs:
        print("  ERROR:", r["tag"], r["error"])
    if errs:
        return 1

    cmp_rows = [compare_one(r) for r in rows]
    n_exact = sum(1 for r in cmp_rows if r["all_exact"])
    summary = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        n_runs=len(rows), n_exact=n_exact, n_not_exact=len(rows) - n_exact,
        source_of_truth=("runs/regenerated_v1(_retry1) prefixes; jsonl bytes "
                         "anchored via dataset sha256 == v2 MANIFEST"),
        worst_max_abs_diff=max(
            (max((a["max_abs_diff"] or 0.0) for a in r["arrays"].values()),
             r["tag"]) for r in cmp_rows),
        per_run=cmp_rows,
        rebuild_records=recs,
    )
    L.dump_json(L.RESULTS / "rebuild_compare.json", summary)
    mism = [r["tag"] for r in cmp_rows if not r["all_exact"]]
    print(f"exact-equal runs: {n_exact}/{len(rows)}")
    for t in mism[:10]:
        print("  NOT EXACT:", t)
    print(f"total {time.time() - t0:.0f}s -> results/rebuild_compare.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
