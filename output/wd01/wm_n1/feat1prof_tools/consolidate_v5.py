"""FEAT1 v5 consolidation: rebuild shared shard-tree artifacts + full verify.

The 6 independent ``--workers 1`` processes each wrote the shared
``ALLOCATION.json`` / ``PLAN_MAP.json`` / ``INDEX.json`` / ``SHA256SUMS`` /
``opened_paths.json`` at the shards root (atomic per write, last-writer-wins
across processes).  The per-run shard bytes + receipts are per-leaf-dir and
disjoint, so this tool rebuilds the shared artifacts deterministically:

1. verify EVERY leaf dir (v1 ``DONE.json`` output shas + ``IDEA_A_DONE.json``
   array shas/bytes) -- reusing the quarantine-scan verifier;
2. rebuild ``INDEX.json`` from the verified ``DONE.json`` records;
3. rebuild ``PLAN_MAP.json`` from the inventory (``select_runs`` +
   ``_resolve_run``, the identical metadata path the driver uses; no run
   physics file is opened);
4. rewrite ``SHA256SUMS`` over every registered shard file;
5. write ``consolidation.json`` with counts, J4 relabel sums, the
   no-DONE/bad lists and the expected-shards arithmetic inputs.

Usage (cwd output/wd01/wm_n1, WD_FORBID_D=1):

  python -X utf8 feat1prof_tools/consolidate_v5.py \
      --inventory inventory/inventory.parquet \
      --data-root E:/research-dense \
      --shards-root D:/rd-staging/wmn1_feat/shards_v2 \
      --logs-root D:/rd-staging/wmn1_feat/logs_v5
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WM_N1 = Path(__file__).resolve().parents[1]
if str(WM_N1) not in sys.path:
    sys.path.insert(0, str(WM_N1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--shards-root", required=True)
    ap.add_argument("--logs-root", required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)

    import os

    if os.environ.get("WD_FORBID_D", "").strip() != "1":
        raise SystemExit("WD_FORBID_D must be 1")
    from corpus import _util as U

    U.activate_d_output_override("D:/rd-staging/wmn1_feat")

    from feat1prof_tools import quarantine_scan_v3 as Q
    from corpus import featurize_all as FA
    from features import writer as W

    shards = Path(a.shards_root)
    arm_dirs = []
    for batch in sorted(shards.iterdir()):
        if not batch.is_dir():
            continue
        for world in sorted(batch.iterdir()):
            if not world.is_dir():
                continue
            for arm in sorted(world.iterdir()):
                if arm.is_dir():
                    arm_dirs.append(arm)
    print(f"consolidate: {len(arm_dirs)} arm dirs", flush=True)

    ok_keys: list = []
    bad: list = []
    j4_relabel_runs = 0
    j4_relabelled_total = 0
    idea_bytes = 0
    with ThreadPoolExecutor(max_workers=max(1, a.threads)) as pool:
        for i, dirp in enumerate(arm_dirs):
            verdict, _ = Q.verify_dir(dirp, pool)
            if verdict != "ok":
                bad.append(dict(key="/".join(
                    dirp.relative_to(shards).parts), reason=verdict))
                continue
            done = json.loads((dirp / "DONE.json").read_text(encoding="utf-8"))
            ok_keys.append(U.run_key(done.get("batch") or done["chunk"],
                                     done["world"], done["arm"]))
            idea = json.loads((dirp / "IDEA_A_DONE.json").read_text(
                encoding="utf-8"))
            if idea.get("amendment2_road_end_relabel"):
                j4_relabel_runs += 1
                j4_relabelled_total += int(idea.get("road_end_relabelled_n")
                                           or 0)
            idea_bytes += int(idea.get("arrays_bytes", 0))
            if (i + 1) % 500 == 0:
                print(f"  verified {i + 1}/{len(arm_dirs)}", flush=True)
    if bad:
        print(json.dumps(bad, indent=1))
        raise SystemExit(f"{len(bad)} unverified dirs present; consolidate "
                         f"refused (quarantine first)")

    from features import constants as C

    index = dict(schema=C.INDEX_SCHEMA, runs={}, pairs={}, totals={})
    for dirp in arm_dirs:
        dp = dirp / "DONE.json"
        if not dp.is_file():
            continue
        done = json.loads(dp.read_text(encoding="utf-8"))
        W.update_index(index, done)
    W.save_index(shards, index)
    sums = W.write_sums(shards, index)
    n_sums = sum(1 for _ in open(sums, encoding="utf-8"))

    # PLAN_MAP rebuild (metadata only; no run physics opened)
    readable = FA.select_runs(a.inventory)
    plan_map = {}
    for r in readable:
        job = FA._resolve_run(r, Path(a.data_root), shards, force=False)
        plan_map[U.run_key(job["batch"], job["world"], job["arm"])] = {
            "chunk": job["chunk"], "batch": job["batch"],
            "world": job["world"], "arm": job["arm"], "split": job["split"],
            "t0_s": job["plan_row"].get("t0_s"), "policy": job["policy"],
            "plan_path": job["plan_path"], "root_rel": job["root_rel"]}
    (shards / FA.PLAN_MAP_NAME).write_text(
        json.dumps({"runs": plan_map}, indent=1), encoding="utf-8")

    rep = dict(
        schema="wm_n1_feat1_v5_consolidation/1", created_utc=U.utc_now(),
        shards_root=str(shards), n_arm_dirs=len(arm_dirs),
        n_verified=len(ok_keys), n_index_runs=len(index["runs"]),
        n_plan_map=len(plan_map), sha256sums_lines=n_sums,
        sums_file=str(sums),
        index_totals=index["totals"],
        j4_relabel_runs=j4_relabel_runs,
        j4_relabelled_total=j4_relabelled_total,
        idea_a_arrays_bytes=idea_bytes, bad=bad)
    out = Path(a.logs_root) / "consolidation.json"
    out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in
                      ("n_arm_dirs", "n_verified", "n_index_runs",
                       "n_plan_map", "sha256sums_lines", "j4_relabel_runs",
                       "j4_relabelled_total")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
