"""FEAT1 v5 subset builder: remaining runs -> 6 disjoint inventory parquets.

After the v4b 10-worker pool was stopped (main-authorized) the shard tree is
quarantine-v4-verified: every remaining arm dir carries a sha-verified
``DONE.json`` AND ``IDEA_A_DONE.json``.  This tool recomputes

    remaining = readable TRAIN/CAL inventory runs - verified DONE keys

and splits ``remaining`` (sorted, then round-robin) into ``--n-subsets``
DISJOINT subsets written as inventory parquet files
``<logs-root>/subsets/sub<k>.parquet`` (same columns as the full inventory,
so ``corpus.featurize_all.select_runs`` and ``_resolve_run`` behave
identically).  It asserts disjointness + that the union is exactly
``remaining``, and writes ``LAUNCH.json`` (key lists per subset) next to the
parquets.  Read-only on the shard tree; never touches shard bytes.

Usage (cwd output/wd01/wm_n1, WD_FORBID_D=1):

  python -X utf8 feat1prof_tools/build_subsets_v5.py \
      --inventory inventory/inventory.parquet \
      --shards-root D:/rd-staging/wmn1_feat/shards_v2 \
      --logs-root D:/rd-staging/wmn1_feat/logs_v5 --n-subsets 6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WM_N1 = Path(__file__).resolve().parents[1]
if str(WM_N1) not in sys.path:
    sys.path.insert(0, str(WM_N1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--shards-root", required=True)
    ap.add_argument("--logs-root", required=True)
    ap.add_argument("--n-subsets", type=int, default=6)
    a = ap.parse_args(argv)

    import os

    if os.environ.get("WD_FORBID_D", "").strip() != "1":
        raise SystemExit("WD_FORBID_D must be 1")
    from corpus import _util as U

    U.activate_d_output_override("D:/rd-staging/wmn1_feat")

    import pandas as pd

    from corpus import featurize_all as FA

    df = pd.read_parquet(a.inventory)
    readable = FA.select_runs(a.inventory)
    key_of = {}
    row_of = {}
    for r in readable:
        arm = U.arm_suffix(r["run"], r.get("arm"))
        key = U.run_key(r["batch"], r["world"], arm)
        key_of[key] = (r["batch"], r["run"])
        row_of[(str(r["batch"]), str(r["run"]))] = key

    shards = Path(a.shards_root)
    done = set()
    n_dirs = 0
    for batch in shards.iterdir():
        if not batch.is_dir():
            continue
        for world in batch.iterdir():
            if not world.is_dir():
                continue
            for arm in world.iterdir():
                if not arm.is_dir():
                    continue
                n_dirs += 1
                dp = arm / "DONE.json"
                if not dp.is_file():
                    continue
                d = json.loads(dp.read_text(encoding="utf-8"))
                done.add(U.run_key(d.get("batch") or d["chunk"],
                                   d["world"], d["arm"]))
    remaining = sorted(set(key_of) - done)
    print(f"readable={len(key_of)} shard_dirs={n_dirs} verified_done={len(done)}"
          f" remaining={len(remaining)}", flush=True)

    subsets = [[] for _ in range(a.n_subsets)]
    for i, key in enumerate(remaining):
        subsets[i % a.n_subsets].append(key)

    subsets_root = Path(a.logs_root) / "subsets"
    subsets_root.mkdir(parents=True, exist_ok=True)
    launch = dict(
        schema="wm_n1_feat1_v5_launch/1",
        created_utc=U.utc_now(),
        inventory=str(Path(a.inventory).resolve()),
        shards_root=str(shards), n_readable=len(key_of),
        n_verified_done=len(done), n_remaining=len(remaining),
        n_subsets=a.n_subsets, subsets=[])
    seen: set = set()
    for k, keys in enumerate(subsets):
        dup = seen.intersection(keys)
        if dup:
            raise SystemExit(f"subset {k} overlaps earlier subsets: {dup}")
        seen.update(keys)
        pairs = [key_of[key] for key in keys]
        mask = pd.Series(False, index=df.index)
        for batch, run in pairs:
            mask |= (df["batch"].astype(str) == batch) & \
                    (df["run"].astype(str) == run)
        sub = df[mask]
        if len(sub) != len(keys):
            raise SystemExit(f"subset {k}: parquet rows {len(sub)} != "
                             f"keys {len(keys)}")
        out = subsets_root / f"sub{k}.parquet"
        sub.to_parquet(out, index=False)
        re_sel = FA.select_runs(str(out))
        re_keys = sorted(
            U.run_key(r["batch"], r["world"],
                      U.arm_suffix(r["run"], r.get("arm")))
            for r in re_sel)
        if re_keys != sorted(keys):
            raise SystemExit(f"subset {k}: re-read keys mismatch")
        print(f"sub{k}: {len(keys)} runs -> {out}", flush=True)
        launch["subsets"].append(dict(
            k=k, parquet=str(out), n_runs=len(keys),
            n_parquet_rows=int(len(sub)), keys=keys))
    if len(seen) != len(remaining):
        raise SystemExit("subset union != remaining")
    (Path(a.logs_root) / "LAUNCH.json").write_text(
        json.dumps(launch, indent=1), encoding="utf-8")
    print(f"LAUNCH.json written; union check OK "
          f"({len(seen)} == remaining {len(remaining)})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
