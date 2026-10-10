"""Build a read-only SUBSET of the A11 ``pack_v3`` corpus.

The subset keeps ONLY the runs referenced by the ``lpwo_v3`` pack_v3
train/val window lists.  Whole runs are copied (per-tick
``[tick_start, tick_end)`` and per-decision ``[dec_start, dec_end)`` slices)
so every kept window -- including its pre-t0 history -- decodes identically to
the full pack.  Arrays, ``INDEX.json`` and ``manifest.json`` follow the exact
pack format the loader validates; ``outcomes.parquet`` keeps only the rows the
kept items reference, remapped to pack-local positions.

RAM stays bounded: source arrays are memory-mapped (``mmap_mode="r"``) and the
destination arrays are written through ``np.lib.format.open_memmap`` one run at
a time (a single run slice is < 20 MB).  No source byte is ever modified.

Usage (from the repo root, PYTHONPATH=output/wd01/wm_n1):

    python _out/r3_packv3_subset/build_subset.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
WM_N1 = REPO / "output" / "wd01" / "wm_n1"
if str(WM_N1) not in sys.path:
    sys.path.insert(0, str(WM_N1))

INDEX_SCHEMA = "wm_n1_corpus_pack/2"
MANIFEST_SCHEMA = "wm_n1_pack/1"
K_MAX = 192
FORMAT = "v3"
DEFAULT_SOURCE = r"D:\rd-staging\wmn1_feat\pack_v3"
DEFAULT_OUT = r"D:\rd-staging\pack_v3_r3_subset"
DEFAULT_WORK = r"E:\rd-wt\ws-r3-jam\_out\r3_packv3_subset"
DEFAULT_TRAIN = r"E:\rd-wt\ws-r3-jam\_out\lpwo_v3\windows_train_noPI_noWall.json"
DEFAULT_VAL = r"E:\rd-wt\ws-r3-jam\_out\lpwo_v3\windows_val_noPI_noWall.json"
DEFAULT_IDENTITIES = r"E:\rd-wt\ws-r3-jam\_out\sidecars_packv3_v2\identities"
EXPECT_TRAIN = 294
EXPECT_VAL = 47
MIN_FREE_GIB = 25.0


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_key_of(item: dict) -> str:
    batch = item.get("batch") or item["chunk"]
    return f"{batch}/{item['world']}__{item['arm']}"


def pack_v3_keys(list_path, pack="pack_v3"):
    data = load_json(list_path)
    wins = [w for w in data["windows"] if str(w.get("pack")) == pack]
    return data, wins


def write_filtered_list(data: dict, wins: list, out_path: Path) -> None:
    """A pack_v3-only window list with the same schema the loader expects."""
    out = dict(data)
    out["windows"] = wins
    out["n_windows"] = len(wins)
    out["pack_names"] = ["pack_v3"]
    by_batch: dict = {}
    by_policy: dict = {}
    for w in wins:
        by_batch[str(w.get("batch") or w.get("chunk"))] = \
            by_batch.get(str(w.get("batch") or w.get("chunk")), 0) + 1
        by_policy[str(w.get("policy"))] = \
            by_policy.get(str(w.get("policy")), 0) + 1
    counts = dict(data.get("counts") or {})
    counts.update({"by_pack": {"pack_v3": len(wins)}, "total": len(wins),
                   "by_batch": by_batch, "by_policy": by_policy,
                   "n_worlds": len({w["world"] for w in wins})})
    out["counts"] = counts
    out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")


def rmtree_checked(path: Path) -> None:
    """Remove only a plain directory (never a reparse point/junction)."""
    if os.path.islink(path) or (hasattr(os.path, "isjunction")
                                and os.path.isjunction(path)):
        raise SystemExit(f"refusing to remove link/junction {path}")
    shutil.rmtree(path)


def build(args) -> int:
    source = Path(args.source)
    out = Path(args.out)
    work = Path(args.work)
    if not source.is_dir():
        raise SystemExit(f"source pack missing: {source}")
    free_initial = shutil.disk_usage(str(out.parent)).free
    if free_initial / float(1 << 30) < float(args.min_free_gib):
        raise SystemExit(
            f"REFUSING: only {free_initial / float(1 << 30):.2f} GiB free at "
            f"{out.parent} (< {args.min_free_gib} GiB)")
    if out.exists():
        if not args.force:
            raise SystemExit(
                f"output {out} already exists; refusing to overwrite "
                "(pass --force to rebuild)")
        rmtree_checked(out)
    free_before = shutil.disk_usage(str(out.parent)).free
    free_gib = free_before / float(1 << 30)
    if free_gib < float(args.min_free_gib):
        raise SystemExit(
            f"REFUSING: only {free_gib:.2f} GiB free at {out.parent} after "
            f"cleanup (< {args.min_free_gib} GiB)")
    print(f"free_space_before_bytes={free_before} ({free_gib:.2f} GiB) "
          f"[initial={free_initial}]")
    work.mkdir(parents=True, exist_ok=True)

    train_data, train_wins = pack_v3_keys(args.window_train)
    val_data, val_wins = pack_v3_keys(args.window_val)
    n_train, n_val = len(train_wins), len(val_wins)
    print(f"pack_v3 windows: train={n_train} val={n_val}")
    if n_train != EXPECT_TRAIN or n_val != EXPECT_VAL:
        raise SystemExit(
            f"window-count mismatch: train {n_train} != {EXPECT_TRAIN} or "
            f"val {n_val} != {EXPECT_VAL}; stopping")
    train_keys = {str(w["key"]) for w in train_wins}
    val_keys = {str(w["key"]) for w in val_wins}
    kept_keys = sorted(train_keys | val_keys)
    print(f"distinct kept runs = {len(kept_keys)}")

    index = load_json(source / "INDEX.json")
    manifest = load_json(source / "manifest.json")
    arrays_spec = manifest["arrays"]
    run_by_key = {str(r["key"]): r for r in index["runs"]}
    missing = [k for k in kept_keys if k not in run_by_key]
    if missing:
        raise SystemExit(f"INDEX lacks run keys: {missing[:5]}")

    # kept items (exactly one per run) in original-manifest order
    items = manifest["items"]
    item_by_key: dict = {}
    for pos, raw in enumerate(items):
        item_by_key.setdefault(run_key_of(raw), []).append(pos)
    kept_items = []
    for key in kept_keys:
        positions = item_by_key.get(key, [])
        if len(positions) != 1:
            raise SystemExit(
                f"run {key} maps to {len(positions)} manifest items "
                "(expected exactly 1)")
        pos = positions[0]
        kept_items.append((pos, items[pos]))
    kept_items.sort(key=lambda pk: pk[0])
    if len(kept_items) != len(kept_keys):
        raise SystemExit("item/run count mismatch")
    # align runs to items by their run key (item order is manifest order)
    kept_runs = [run_by_key[run_key_of(raw)] for _, raw in kept_items]
    bad = [(r["key"], r["split"]) for r in kept_runs if r["split"] != "TRAIN"]
    if bad:
        raise SystemExit(f"kept runs not TRAIN: {bad[:5]}")

    # new contiguous offsets
    runs_meta = []
    tick_cursor = dec_cursor = 0
    for (_, raw), run in zip(kept_items, kept_runs):
        rec = dict(run)
        rec["_orig_tick_start"] = int(run["tick_start"])
        rec["_orig_dec_start"] = int(run["dec_start"])
        rec["tick_start"] = tick_cursor
        rec["tick_end"] = tick_cursor + int(run["n_ticks"])
        rec["dec_start"] = dec_cursor
        rec["dec_end"] = dec_cursor + int(run["n_decision"])
        tick_cursor = rec["tick_end"]
        dec_cursor = rec["dec_end"]
        runs_meta.append(rec)
    n_tick, n_dec = tick_cursor, dec_cursor
    print(f"subset ticks={n_tick} decisions={n_dec}")

    out.mkdir(parents=True, exist_ok=True)
    src_mm = {name: np.load(source / spec["file"], mmap_mode="r")
              for name, spec in arrays_spec.items()}
    dst_mm = {}
    arrays_shape: dict = {}
    for name, spec in arrays_spec.items():
        n0 = n_dec if name == "plan" else n_tick
        shape = (n0,) + tuple(int(x) for x in spec["shape"][1:])
        dtype = np.dtype(spec["dtype"])
        dst_mm[name] = np.lib.format.open_memmap(
            out / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
        arrays_shape[name] = (dtype, shape)

    for (_, _raw), rec in zip(kept_items, runs_meta):
        ots, ote = rec["_orig_tick_start"], rec["_orig_tick_start"] + \
            int(rec["n_ticks"])
        ods, ode = rec["_orig_dec_start"], rec["_orig_dec_start"] + \
            int(rec["n_decision"])
        ts, te = rec["tick_start"], rec["tick_end"]
        ds, de = rec["dec_start"], rec["dec_end"]
        for name in arrays_spec:
            if name == "plan":
                dst_mm[name][ds:de] = src_mm[name][ods:ode]
            else:
                dst_mm[name][ts:te] = src_mm[name][ots:ote]
    for name, arr in dst_mm.items():
        arr.flush()
    del dst_mm
    for arr in src_mm.values():
        del arr
    del src_mm

    # outcome rows: gather kept items' original rows, remap pack-local
    import pandas as pd

    frame = pd.read_parquet(source / "outcomes.parquet")
    parts, outcome_cursor = [], 0
    items_out = []
    for (orig_pos, raw), rec in zip(kept_items, runs_meta):
        new = dict(raw)
        new["tick"] = rec["tick_start"] + \
            (int(raw["tick"]) - rec["_orig_tick_start"])
        new["plan_row"] = rec["dec_start"] + \
            (int(raw["plan_row"]) - rec["_orig_dec_start"])
        orig_rows = [int(x) for x in raw["outcome_rows"]]
        new["outcome_rows"] = list(range(outcome_cursor,
                                         outcome_cursor + len(orig_rows)))
        outcome_cursor += len(orig_rows)
        if orig_rows:
            parts.append(frame.iloc[orig_rows])
        else:
            parts.append(frame.iloc[0:0])
        items_out.append(new)
    out_df = pd.concat(parts, ignore_index=True) if parts else frame.iloc[0:0]
    out_df.to_parquet(out / "outcomes.parquet", index=False)
    print(f"outcome rows kept = {len(out_df)}")

    arrays_manifest = {}
    arrays_index = {}
    sum_bytes = 0
    for name, spec in arrays_spec.items():
        dtype, shape = arrays_shape[name]
        nb = int(np.prod(shape)) * dtype.itemsize
        arrays_manifest[name] = {"file": f"{name}.npy", "dtype": str(dtype),
                                 "shape": list(shape)}
        arrays_index[name] = {"file": f"{name}.npy", "dtype": str(dtype),
                              "shape": list(shape), "n_bytes": nb}
        sum_bytes += nb

    manifest_out = {
        "schema": MANIFEST_SCHEMA,
        "k_max": K_MAX,
        "format": FORMAT,
        "arrays": arrays_manifest,
        "outcomes": {"file": "outcomes.parquet"},
        "items": items_out,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest_out, indent=1), encoding="utf-8")

    runs_out = []
    for rec in runs_meta:
        clean = {k: v for k, v in rec.items() if not k.startswith("_")}
        runs_out.append(clean)
    pairs_out = {}
    for pk, keys in (index.get("pairs") or {}).items():
        keep = [k for k in keys if k in set(kept_keys)]
        if keep:
            pairs_out[pk] = keep

    files = {}
    for name in arrays_spec:
        files[f"{name}.npy"] = sha256_file(out / f"{name}.npy")
    files["outcomes.parquet"] = sha256_file(out / "outcomes.parquet")
    files["manifest.json"] = sha256_file(out / "manifest.json")
    total_bytes = sum_bytes + (out / "outcomes.parquet").stat().st_size + \
        (out / "manifest.json").stat().st_size
    index_out = {
        "schema": INDEX_SCHEMA,
        "train_schema": MANIFEST_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_data_root": str(source),
        "manifest_file": "manifest.json",
        "outcomes_file": "outcomes.parquet",
        "n_runs": len(runs_out),
        "n_items": len(items_out),
        "n_ticks": n_tick,
        "n_decision": n_dec,
        "k_max": K_MAX,
        "arrays": arrays_index,
        "runs": runs_out,
        "pairs": pairs_out,
        "files": files,
        "totals": {"n_runs": len(runs_out), "n_items": len(items_out),
                   "n_ticks": n_tick, "n_decision": n_dec,
                   "bytes": total_bytes},
        "format": FORMAT,
    }
    (out / "INDEX.json").write_text(json.dumps(index_out, indent=1),
                                    encoding="utf-8")

    # SHA256SUMS over every artifact
    checksum_names = [f"{name}.npy" for name in arrays_spec]
    checksum_names += ["manifest.json", "INDEX.json", "outcomes.parquet"]
    lines = [f"{sha256_file(out / n)}  {n}" for n in checksum_names]
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # strict, self-contained identity sidecar: pack-local positions and a
    # remapped canonical vehicle map, so train.identity_sidecar.
    # load_identity_sidecar accepts the subset for a real training run (the
    # source sidecar binds to the full pack and cannot be used verbatim).
    from train import identity_sidecar as IS  # noqa: PLC0415

    ids_dir = out / "identities"
    ids_dir.mkdir(parents=True, exist_ok=True)
    real_dir = Path(args.identities)
    real_ids = np.load(real_dir / "identities.npy")
    real_pos = np.load(real_dir / "global_positions.npy")
    pos_index = {int(p): i for i, p in enumerate(real_pos)}
    copied = np.empty((len(kept_items), real_ids.shape[1]), dtype=np.int64)
    for i, (orig_pos, _raw) in enumerate(kept_items):
        if orig_pos not in pos_index:
            raise SystemExit(
                f"real identity sidecar lacks original manifest position "
                f"{orig_pos}")
        copied[i] = real_ids[pos_index[orig_pos]]
    used = sorted({int(x) for x in copied[copied >= 0]})
    remap = {old: new for new, old in enumerate(used)}
    new_ids = np.full_like(copied, -1)
    nz = copied >= 0
    flat = copied[nz]
    new_ids[nz] = [remap[int(x)] for x in flat]
    real_map = load_json(real_dir / "vehicle_map.json")
    new_map = [real_map[i] for i in used]
    np.save(ids_dir / "identities.npy", new_ids)
    np.save(ids_dir / "global_positions.npy",
            np.arange(len(kept_items), dtype=np.int64))
    (ids_dir / "vehicle_map.json").write_text(json.dumps(new_map),
                                              encoding="utf-8")

    plan = IS._plan(str(out))  # noqa: SLF001
    real_receipt = load_json(real_dir / IS.RECEIPT_NAME)
    src_by_key = {s["selection"]["key"]: s for s in real_receipt["sources"]}
    sources = []
    for sel in plan["selection"]:
        src = src_by_key[sel["key"]]
        sources.append({"selection": sel,
                        "done_sha256": src["done_sha256"],
                        "slot_index_sha256": src["slot_index_sha256"],
                        "slot_vid_sha256": src["slot_vid_sha256"]})
    id_files = {n: sha256_file(ids_dir / n) for n in
                (IS.IDS_NAME, IS.POSITIONS_NAME, IS.MAP_NAME)}
    id_receipt = {
        "schema": IS.SCHEMA,
        "split": "TRAIN",
        "pack": plan,
        "sources": sources,
        "n_vehicles": len(new_map),
        "namespace": ["chunk", "world", "vehicle_id"],
        "files": id_files,
    }
    id_receipt["receipt_sha256"] = hashlib.sha256(  # noqa: SLF001
        IS._canonical(id_receipt)).hexdigest()
    (ids_dir / IS.RECEIPT_NAME).write_text(json.dumps(id_receipt, indent=1),
                                           encoding="utf-8")

    train_out = work / "windows_train_pack_v3.json"
    val_out = work / "windows_val_pack_v3.json"
    write_filtered_list(train_data, train_wins, train_out)
    write_filtered_list(val_data, val_wins, val_out)

    receipt = {
        "schema": "wm_n1_pack_v3_subset/1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_pack": str(source),
        "subset_root": str(out),
        "source_manifest_sha256": sha256_file(source / "manifest.json"),
        "source_index_sha256": sha256_file(source / "INDEX.json"),
        "n_runs_kept": len(kept_keys),
        "n_items_kept": len(items_out),
        "n_train_windows": n_train,
        "n_val_windows": n_val,
        "n_ticks": n_tick,
        "n_decision": n_dec,
        "arrays": {name: {"dtype": str(arrays_shape[name][0]),
                          "shape": list(arrays_shape[name][1]),
                          "bytes": arrays_index[name]["n_bytes"]}
                   for name in arrays_spec},
        "outcomes_rows": int(len(out_df)),
        "total_bytes": int(total_bytes),
        "free_space_before_bytes": int(free_before),
        "free_space_initial_bytes": int(free_initial),
        "sha256": dict(list(files.items()) +
                       [("INDEX.json", sha256_file(out / "INDEX.json"))]),
        "identity_sidecar": str(ids_dir),
        "filtered_window_lists": [str(train_out), str(val_out)],
    }
    (work / "SUBSET_RECEIPT.json").write_text(
        json.dumps(receipt, indent=1), encoding="utf-8")
    print(f"subset written: {out}")
    print(f"total_bytes={total_bytes} ({total_bytes / float(1 << 30):.2f} GiB)")
    print(f"receipt: {work / 'SUBSET_RECEIPT.json'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--work", default=DEFAULT_WORK)
    ap.add_argument("--window-train", default=DEFAULT_TRAIN)
    ap.add_argument("--window-val", default=DEFAULT_VAL)
    ap.add_argument("--identities", default=DEFAULT_IDENTITIES)
    ap.add_argument("--min-free-gib", type=float, default=MIN_FREE_GIB)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    return build(args)


if __name__ == "__main__":
    sys.exit(main())
