"""Pack/convert: per-run ``features`` shards -> one packed memmap corpus.

The pack directory is the read-side contract owned by TRAIN
(``output/wd01/wm_n1/train/dataset.py``): ``manifest.json`` (schema
``wm_n1_pack/1``), the concatenated memmap arrays, and ``outcomes.parquet``.
CORPUS additionally writes its own richer ``INDEX.json`` (per-run offsets,
source paths, file hashes) and ``SHA256SUMS`` so :mod:`corpus.verify` and
:mod:`corpus.manifest` can audit the pack without re-opening the payload.

The split is never re-decided here; the pack embeds the ``plan_path`` recorded
by :mod:`corpus.featurize_all` (``PLAN_MAP.json``) so the upload manifest can
re-derive each world's split from its ``WORLD_PLAN.json`` without opening the
packed data.  ``convert`` re-packs an existing pack (optionally recasting
dtypes) and never reads raw physics or touches HOLDOUT/chunk-5.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import _util as U

PACK_SCHEMA = "wm_n1_corpus_pack/2"
TRAIN_SCHEMA = "wm_n1_pack/1"
INDEX_NAME = "INDEX.json"
MANIFEST_NAME = "manifest.json"
SUMS_NAME = "SHA256SUMS"
PLAN_MAP_NAME = "PLAN_MAP.json"
OUTCOMES_NAME = "outcomes.parquet"

# The arrays the TRAIN read-side contract (train/dataset.py) loads, plus the
# P9 Idea A channels (INTERFACES_V2 tensors table) appended by the FEAT1
# integrator patch; corpus.verify picks the extension up via this import.
TRAIN_ARRAYS = ("veh_step", "veh_mask", "plan", "wall", "wall_mask", "counter",
                "accel_next", "target_mask", "lane_next", "arrivals")
TRAIN_ARRAYS = TRAIN_ARRAYS + ("veh_kin", "hdv_base", "cav_plan_ch",
                               "event_tok", "event_mask")
TICK_ARRAYS = tuple(n for n in TRAIN_ARRAYS if n != "plan")


def _features():
    U.ensure_wm_n1_on_path()
    from features import constants as C
    from features import writer as W

    return C, W


def format_of_k_max(k_max: int):
    """Registered format name of a pack K_max (128 -> 'v2', 192 -> 'v3').

    Returns ``None`` for an unregistered K (legacy/synthetic fixtures); v2 is
    the default and is only recorded explicitly for a v3 (non-default) pack.
    """
    C, _ = _features()
    try:
        fmt = C.format_of_k_max(int(k_max))
    except ValueError:
        return None
    return None if fmt == C.DEFAULT_FORMAT else fmt


def _hash_many(paths, workers: int) -> list:
    paths = [Path(p) for p in paths]
    if workers <= 1 or len(paths) <= 1:
        return [U.sha256_file(p) for p in paths]
    with ThreadPoolExecutor(max_workers=min(workers, len(paths))) as ex:
        return list(ex.map(U.sha256_file, paths))


def _load_plan_map(data_root: Path) -> dict:
    p = Path(data_root) / PLAN_MAP_NAME
    if p.is_file():
        return U.read_json(p).get("runs", {})
    return {}


def _ordered_runs(index: dict, plan_map: dict) -> list:
    out = []
    for key in sorted(index.get("runs", {})):
        rec = dict(index["runs"][key])
        rec["key"] = key
        batch, tail = key.split("/", 1)
        world, arm = tail.rsplit("__", 1)
        rec.setdefault("world", world)
        rec.setdefault("arm", arm)
        rec.setdefault("chunk", chunk_of(batch))
        rec.setdefault("batch", batch)
        pm = plan_map.get(key, {})
        rec.setdefault("split", pm.get("split"))
        rec.setdefault("t0_s", pm.get("t0_s"))
        rec["plan_path"] = pm.get("plan_path")
        rec["root_rel"] = pm.get("root_rel")
        # policy is metadata: PLAN_MAP (inventory) first, then the shard
        # DONE/Index policy, never the arm suffix itself
        rec["policy"] = (pm.get("policy") or rec.get("policy")
                         or pm.get("arm") or arm)
        out.append(rec)
    return out


def chunk_of(batch: str) -> str:
    """Chunk label of a shard-key first component (batch name or label)."""
    from .featurize_all import chunk_label

    return chunk_label(batch)


def _n_decision() -> int:
    C, _ = _features()
    return int(C.N_DECISION)


def _shard_array(src: Path, name: str):
    return np.load(src / f"{name}.npy", mmap_mode="r")


def _write_manifest(out_dir: Path, arrays: dict, items: list, k_max: int,
                    format: str | None = None) -> dict:
    manifest = dict(
        schema=TRAIN_SCHEMA, k_max=int(k_max),
        arrays={n: dict(file=f"{n}.npy", dtype=str(arrays[n]["dtype"]),
                        shape=list(arrays[n]["shape"])) for n in TRAIN_ARRAYS},
        outcomes=dict(file=OUTCOMES_NAME),
        items=items,
    )
    if format is not None:
        # a non-default (v3) pack self-describes its slot-width format; the
        # v2 manifest bytes stay exactly as the frozen writer wrote them
        manifest["format"] = str(format)
    U.write_json_atomic(out_dir / MANIFEST_NAME, manifest)
    return manifest


def pack_shards(data_root, out_dir, *, workers: int = 4) -> dict:
    """Convert the per-run shard tree at ``data_root`` into ``out_dir``.

    Returns the corpus pack manifest (also written as ``INDEX.json``).
    """
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    U.refuse_unapproved_d_out(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, W = _features()
    index = W.load_index(data_root)
    plans = _load_plan_map(data_root)
    runs = _ordered_runs(index, plans)
    if not runs:
        raise ValueError(f"no shards in {data_root / 'INDEX.json'}")

    src = {r["key"]: Path(r["shard_dir"]) for r in runs}

    arrays: dict = {}
    for name in TRAIN_ARRAYS:
        dt = None
        tail = None
        total = 0
        for r in runs:
            a = _shard_array(src[r["key"]], name)
            if dt is None:
                dt, tail = a.dtype, tuple(a.shape[1:])
            elif a.dtype != dt or tuple(a.shape[1:]) != tail:
                raise ValueError(
                    f"array {name!r} shape/dtype mismatch at {r['key']}")
            total += int(a.shape[0])
        arrays[name] = dict(dtype=str(dt), shape=[total, *tail], n_bytes=0)

    targets = {n: np.lib.format.open_memmap(
        out_dir / f"{n}.npy", mode="w+", dtype=np.dtype(arrays[n]["dtype"]),
        shape=tuple(arrays[n]["shape"])) for n in TRAIN_ARRAYS}
    offsets = {n: 0 for n in TRAIN_ARRAYS}
    for r in runs:
        for name in TRAIN_ARRAYS:
            a = _shard_array(src[r["key"]], name)
            start = offsets[name]
            end = start + int(a.shape[0])
            targets[name][start:end] = a
            offsets[name] = end
            if name == "veh_mask":
                r["tick_start"], r["tick_end"] = start, end
                nz = np.flatnonzero(np.asarray(a).any(axis=1))
                r["snap_tick"] = start + (int(nz[0]) if nz.size else 0)
            if name == "plan":
                r["dec_start"], r["dec_end"] = start, end
                r["n_decision"] = int(end - start)
    for name in TRAIN_ARRAYS:
        targets[name].flush()
        del targets[name]

    # ---- outcomes: concatenate the per-run label rows ---------------------
    import pandas as pd

    frames = []
    for r in runs:
        o = src[r["key"]] / OUTCOMES_NAME
        if o.is_file():
            frames.append(pd.read_parquet(o))
        else:
            frames.append(pd.DataFrame())
    non_empty = [f for f in frames if len(f)]
    if non_empty:
        outcomes = pd.concat(non_empty, ignore_index=True)
    else:
        outcomes = pd.DataFrame()
    outcomes.to_parquet(out_dir / OUTCOMES_NAME, index=False)
    row_cursor = 0
    for r, f in zip(runs, frames):
        n = int(len(f))
        r["outcomes"] = OUTCOMES_NAME
        r["outcome_first"] = row_cursor
        r["n_outcome_rows"] = n
        row_cursor += n

    files = {f"{n}.npy": U.sha256_file(out_dir / f"{n}.npy")
             for n in TRAIN_ARRAYS}
    for n in TRAIN_ARRAYS:
        arrays[n]["n_bytes"] = int((out_dir / f"{n}.npy").stat().st_size)
    files[OUTCOMES_NAME] = U.sha256_file(out_dir / OUTCOMES_NAME)

    k_max = max(int(_shard_array(src[r["key"]], "veh_step").shape[1])
                for r in runs)
    fmt = format_of_k_max(k_max)
    items = [_train_item(r) for r in runs]
    _write_manifest(out_dir, arrays, items, k_max, format=fmt)
    files[MANIFEST_NAME] = U.sha256_file(out_dir / MANIFEST_NAME)

    pairs: dict = {}
    for r in runs:
        pk = W.pair_key(r["world"], float(r["t0_s"]))
        pairs.setdefault(pk, [])
        if r["key"] not in pairs[pk]:
            pairs[pk].append(r["key"])

    total_bytes = int(sum((out_dir / rel).stat().st_size for rel in files))
    pack = dict(
        schema=PACK_SCHEMA, train_schema=TRAIN_SCHEMA, created_utc=U.utc_now(),
        source_data_root=str(data_root.resolve()),
        manifest_file=MANIFEST_NAME, outcomes_file=OUTCOMES_NAME,
        n_runs=len(runs), n_items=len(items),
        n_ticks=sum(int(r["n_ticks"]) for r in runs),
        n_decision=sum(int(r.get("n_decision") or 0) for r in runs),
        k_max=int(k_max),
        arrays={n: dict(file=f"{n}.npy", **arrays[n]) for n in TRAIN_ARRAYS},
        runs=[_run_record(r) for r in runs], pairs=pairs, files=files,
        totals=dict(n_runs=len(runs), n_items=len(items),
                    n_ticks=sum(int(r["n_ticks"]) for r in runs),
                    n_decision=sum(int(r.get("n_decision") or 0)
                                   for r in runs),
                    bytes=total_bytes),
    )
    if fmt is not None:
        pack["format"] = fmt
    U.write_json_atomic(out_dir / INDEX_NAME, pack)
    _write_sums(out_dir, pack)
    return pack


def _train_item(r: dict) -> dict:
    first = int(r.get("outcome_first", 0))
    n = int(r.get("n_outcome_rows", 0))
    return dict(
        world=r["world"], arm=r["arm"], chunk=r["chunk"],
        batch=r.get("batch") or r["chunk"], split=r["split"],
        t0_s=float(r["t0_s"]),
        pair_key=f"{r['world']}@t0={float(r['t0_s']):g}",
        policy=r.get("policy") or r["arm"],
        tick=int(r.get("snap_tick", r["tick_start"])),
        plan_row=int(r["dec_start"]),
        outcome_rows=list(range(first, first + n)),
    )


def _run_record(r: dict) -> dict:
    keys = ("key", "chunk", "batch", "world", "arm", "split", "t0_s",
            "n_ticks",
            "n_decision", "tick_start", "tick_end", "dec_start", "dec_end",
            "shard_dir", "input_sha256", "plan_path", "root_rel", "policy",
            "outcomes", "n_outcome_rows")
    return {k: U.sanitize(r.get(k)) for k in keys}


def _write_sums(out_dir: Path, manifest: dict) -> Path:
    all_files = dict(manifest["files"])
    all_files[INDEX_NAME] = U.sha256_file(out_dir / INDEX_NAME)
    lines = [f"{all_files[rel]}  {rel}" for rel in sorted(all_files)]
    p = out_dir / SUMS_NAME
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def convert(in_dir, out_dir, *, workers: int = 4, recast=None) -> dict:
    """Re-pack an existing pack (optionally recast dtypes); no raw physics."""
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = U.read_json(in_dir / INDEX_NAME)
    if index.get("schema") != PACK_SCHEMA:
        raise ValueError(f"unexpected pack schema {index.get('schema')!r}")
    recast = recast or {}

    new_arrays = {}
    for name, spec in index["arrays"].items():
        a = np.load(in_dir / spec["file"], mmap_mode="r")
        dt = np.dtype(recast.get(name, spec["dtype"]))
        dst = np.lib.format.open_memmap(out_dir / spec["file"], mode="w+",
                                        dtype=dt, shape=a.shape)
        if dt == a.dtype:
            dst[...] = a
        else:
            block = 1 << 20
            for i in range(0, int(a.shape[0]), block):
                dst[i:i + block] = a[i:i + block]
        dst.flush()
        del dst, a
        new_arrays[name] = dict(spec, dtype=str(dt),
                                n_bytes=int((out_dir /
                                             spec["file"]).stat().st_size))

    files = {spec["file"]: U.sha256_file(out_dir / spec["file"])
             for spec in new_arrays.values()}
    out_rel = index.get("outcomes_file", OUTCOMES_NAME)
    dst = out_dir / out_rel
    dst.write_bytes((in_dir / out_rel).read_bytes())
    files[out_rel] = U.sha256_file(dst)

    manifest = U.read_json(in_dir / MANIFEST_NAME)
    for name, spec in manifest.get("arrays", {}).items():
        if name in new_arrays:
            spec["dtype"] = new_arrays[name]["dtype"]
            spec["shape"] = list(new_arrays[name]["shape"])
    U.write_json_atomic(out_dir / MANIFEST_NAME, manifest)
    files[MANIFEST_NAME] = U.sha256_file(out_dir / MANIFEST_NAME)

    new_index = dict(index)
    new_index.update(
        created_utc=U.utc_now(), arrays=new_arrays, files=files,
        source_pack=str(in_dir.resolve()),
        totals=dict(index["totals"],
                    bytes=int(sum((out_dir / rel).stat().st_size
                                  for rel in files))))
    U.write_json_atomic(out_dir / INDEX_NAME, new_index)
    _write_sums(out_dir, new_index)
    return new_index
