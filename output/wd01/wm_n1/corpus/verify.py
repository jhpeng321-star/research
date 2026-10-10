"""Packed-corpus verify CLI.

Re-checks every packed array and the offset index against ``INDEX.json``,
re-validates the train read-side ``manifest.json`` (schema, arrays, items), and
re-checks every ``SHA256SUMS`` entry — so a tampered array fails.  Source shard
``DONE.json`` shas are re-verified when the source shard tree is still
reachable.  Returns ``{"ok", "n_runs", "errors"}``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import _util as U
from .pack import (INDEX_NAME, MANIFEST_NAME, SUMS_NAME, TRAIN_ARRAYS,
                   TRAIN_SCHEMA, format_of_k_max)

FORBIDDEN_SPLITS = ("HOLDOUT", "TEST", "SEALED")
FORBIDDEN_CHUNKS = ("chunk5", "chunk_5", "chunk-5")
SOURCE_NPY_FILES = TRAIN_ARRAYS + ("slot_index", "decision_t")
_ITEM_REQUIRED = ("world", "arm", "chunk", "split", "t0_s", "pair_key",
                  "policy", "tick", "plan_row", "outcome_rows")
#: packed arrays whose axis 1 is the slot axis (K_max)
SLOT_AXIS_ARRAYS = ("veh_step", "veh_mask", "plan", "accel_next",
                    "target_mask", "lane_next", "veh_kin", "hdv_base",
                    "cav_plan_ch")


def _verify_k_contract(pack_dir: Path, index: dict, errors: list) -> int:
    """K_max contract: manifest/index agreement + slot-dim + format checks.

    Returns the pack K_max (0 when unknown).  A v3 pack must carry
    ``format="v3"`` with ``k_max == 192``; a pack whose K_max is a registered
    format K must not claim a disagreeing ``format``.
    """
    k_max = index.get("k_max")
    if k_max is None:
        errors.append(f"{INDEX_NAME}: missing k_max")
        return 0
    try:
        k_max = int(k_max)
    except (TypeError, ValueError):
        errors.append(f"{INDEX_NAME}: non-integer k_max {index.get('k_max')!r}")
        return 0
    fmt = format_of_k_max(k_max)
    claimed = index.get("format")
    if claimed is not None and claimed != fmt:
        errors.append(
            f"{INDEX_NAME}: format {claimed!r} disagrees with K_max {k_max} "
            f"(expected {fmt!r})")
    # slot axis of every slot-indexed packed array must equal k_max
    for name, spec in (index.get("arrays") or {}).items():
        if name not in SLOT_AXIS_ARRAYS:
            continue
        shape = spec.get("shape") or []
        if len(shape) >= 2 and int(shape[1]) != k_max:
            errors.append(
                f"array {name}: slot dim {shape[1]} != k_max {k_max}")
    # the TRAIN read-side manifest must agree
    man_p = pack_dir / MANIFEST_NAME
    if man_p.is_file():
        try:
            man = U.read_json(man_p)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{MANIFEST_NAME}: unreadable ({e})")
            return k_max
        if int(man.get("k_max", -1)) != k_max:
            errors.append(
                f"{MANIFEST_NAME}: k_max {man.get('k_max')!r} != "
                f"{INDEX_NAME} k_max {k_max}")
        if man.get("format") != index.get("format"):
            errors.append(
                f"{MANIFEST_NAME}: format {man.get('format')!r} != "
                f"{INDEX_NAME} format {index.get('format')!r}")
    return k_max


def _verify_source_done(shard_dir: Path, record: dict, errors: list) -> None:
    done_p = shard_dir / "DONE.json"
    if not done_p.is_file():
        return
    try:
        done = U.read_json(done_p)
    except Exception as e:  # noqa: BLE001
        errors.append(f"{record['key']}: unreadable DONE.json ({e})")
        return
    if done.get("input_sha256") and record.get("input_sha256") \
            and done["input_sha256"] != record["input_sha256"]:
        errors.append(f"{record['key']}: source input sha drift")
    for name, sha in (done.get("outputs") or {}).items():
        fname = f"{name}.npy" if name in SOURCE_NPY_FILES else name
        p = shard_dir / fname
        if not p.is_file():
            errors.append(f"{record['key']}: source {fname} missing")
        elif U.sha256_file(p) != sha:
            errors.append(f"{record['key']}: source {fname} sha mismatch")


def _verify_train_manifest(pack_dir: Path, index: dict, errors: list) -> int:
    """Validate the TRAIN read-side ``manifest.json``; return item count."""
    p = pack_dir / MANIFEST_NAME
    if not p.is_file():
        errors.append(f"missing {MANIFEST_NAME}")
        return 0
    manifest = U.read_json(p)
    if manifest.get("schema") != TRAIN_SCHEMA:
        errors.append(
            f"{MANIFEST_NAME}: schema {manifest.get('schema')!r} != "
            f"{TRAIN_SCHEMA!r}")
    marrays = manifest.get("arrays") or {}
    if set(marrays) != set(TRAIN_ARRAYS):
        errors.append(
            f"{MANIFEST_NAME}: arrays {sorted(marrays)} != {sorted(TRAIN_ARRAYS)}")
    for name, spec in marrays.items():
        fp = pack_dir / spec["file"]
        if not fp.is_file():
            errors.append(f"{MANIFEST_NAME}: {name} missing {spec['file']}")
            continue
        if name in (index.get("arrays") or {}):
            ix = index["arrays"][name]
            if list(spec["shape"]) != list(ix["shape"]):
                errors.append(f"{MANIFEST_NAME}: {name} shape disagrees with "
                              f"INDEX.json")
            if str(spec["dtype"]) != str(ix["dtype"]):
                errors.append(f"{MANIFEST_NAME}: {name} dtype disagrees with "
                              f"INDEX.json")
    out_rel = (manifest.get("outcomes") or {}).get("file", "outcomes.parquet")
    if not (pack_dir / out_rel).is_file():
        errors.append(f"{MANIFEST_NAME}: outcomes missing {out_rel}")

    n_ticks = int(index.get("n_ticks", 0))
    n_dec = int(index.get("n_decision", 0))
    items = manifest.get("items") or []
    seen_items: set = set()
    for it in items:
        missing = [k for k in _ITEM_REQUIRED if k not in it]
        if missing:
            errors.append(
                f"{MANIFEST_NAME}: item {it.get('world')}/{it.get('arm')} "
                f"misses {missing}")
            continue
        split = str(it.get("split", "")).upper()
        chunk = str(it.get("chunk", ""))
        if split in FORBIDDEN_SPLITS:
            errors.append(f"{MANIFEST_NAME}: item split {split!r} is sealed")
        if any(chunk.startswith(c) for c in FORBIDDEN_CHUNKS):
            errors.append(f"{MANIFEST_NAME}: item chunk {chunk!r} is sealed")
        # shard-key injectivity: (batch|chunk, world, arm) must be unique;
        # ``arm`` is the exact run-dir suffix so replicated plan arms
        # (fixed1/fixed2/switching1..3) never collapse into one item
        ikey = (str(it.get("batch") or chunk), str(it["world"]),
                str(it["arm"]))
        if ikey in seen_items:
            errors.append(
                f"{MANIFEST_NAME}: duplicate item {ikey} (arm must be the "
                f"exact run-dir suffix, policy {it.get('policy')!r} is "
                f"metadata only)")
        seen_items.add(ikey)
        tick = int(it["tick"])
        plan_row = int(it["plan_row"])
        if not 0 <= tick < max(1, n_ticks):
            errors.append(f"{MANIFEST_NAME}: item tick {tick} out of range")
        if not 0 <= plan_row < max(1, n_dec):
            errors.append(f"{MANIFEST_NAME}: item plan_row {plan_row} out of "
                          f"range")
    return len(items)


def verify_pack(pack_dir) -> dict:
    """Verify a packed corpus; ``ok`` is True iff there are no errors."""
    pack_dir = Path(pack_dir)
    errors: list = []
    index_p = pack_dir / INDEX_NAME
    if not index_p.is_file():
        return dict(ok=False, n_runs=0, n_items=0, n_arrays=0,
                    errors=[f"missing {INDEX_NAME}"])
    index = U.read_json(index_p)
    runs = index.get("runs", [])

    k_max = _verify_k_contract(pack_dir, index, errors)

    for name, spec in (index.get("arrays") or {}).items():
        p = pack_dir / spec["file"]
        if not p.is_file():
            errors.append(f"array {name}: missing {spec['file']}")
            continue
        if p.stat().st_size != int(spec["n_bytes"]):
            errors.append(f"array {name}: size {p.stat().st_size} != "
                          f"{spec['n_bytes']}")
        try:
            a = np.load(p, mmap_mode="r")
        except Exception as e:  # noqa: BLE001
            errors.append(f"array {name}: unreadable ({e})")
            continue
        if str(a.dtype) != spec["dtype"]:
            errors.append(f"array {name}: dtype {a.dtype} != {spec['dtype']}")
        if list(a.shape) != list(spec["shape"]):
            errors.append(f"array {name}: shape {list(a.shape)} != "
                          f"{spec['shape']}")
        del a
        want = (index.get("files") or {}).get(spec["file"])
        if want and U.sha256_file(p) != want:
            errors.append(f"array {name}: sha256 mismatch (tampered?)")

    tick_cursor = 0
    for r in sorted(runs, key=lambda x: x.get("tick_start", 0)):
        if r.get("tick_start") is None or r.get("tick_end") is None:
            errors.append(f"{r.get('key')}: missing tick offsets")
            continue
        if r.get("tick_start") != tick_cursor:
            errors.append(f"{r.get('key')}: tick_start {r.get('tick_start')} "
                          f"!= {tick_cursor}")
        if r.get("tick_end", 0) - r.get("tick_start", 0) != int(
                r.get("n_ticks", -1)):
            errors.append(f"{r.get('key')}: tick span != n_ticks")
        dec = int(r.get("dec_end", 0)) - int(r.get("dec_start", 0))
        if dec != int(r.get("n_decision", dec)):
            errors.append(f"{r.get('key')}: decision span {dec} != "
                          f"n_decision {r.get('n_decision')}")
        tick_cursor = int(r.get("tick_end", tick_cursor))
        shard = r.get("shard_dir")
        if shard and Path(shard).is_dir():
            _verify_source_done(Path(shard), r, errors)

    n_items = _verify_train_manifest(pack_dir, index, errors)

    sums_p = pack_dir / SUMS_NAME
    if sums_p.is_file():
        for line in sums_p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            sha, _, rel = line.partition("  ")
            p = pack_dir / rel
            if not p.is_file():
                errors.append(f"sums: missing {rel}")
            elif U.sha256_file(p) != sha:
                errors.append(f"sums: sha256 mismatch for {rel}")
    else:
        errors.append(f"missing {SUMS_NAME}")

    return dict(ok=not errors, n_runs=len(runs), n_items=n_items,
                n_arrays=len(index.get("arrays") or {}), k_max=k_max,
                format=index.get("format"),
                errors=errors)
