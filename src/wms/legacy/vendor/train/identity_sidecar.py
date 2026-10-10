# ruff: noqa
# VENDORED from output/wd01/wm_n1/train/identity_sidecar.py (last changed in commit f93ad35; copied at baseline bf7d4cb).
# Copied byte-for-byte apart from this header.
# Do not edit: behaviour is frozen evidence; WP-C only mirrors it.
"""Recover TRAIN snapshot vehicle identities; never a learned input channel.

This module is independent of dataset/model/label loaders. Building reads only
selected TRAIN shards' DONE.json, slot_vid.json and slot_index.npy. Loading needs
only pack metadata and this portable sidecar. No path from a receipt is opened.
Execution requires the separately reviewed access guard; this is not authority
to access a corpus. Original packs/shards are never changed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np

SCHEMA = "wm_n1_train_identity_sidecar/1"
RECEIPT_NAME = "TRAIN_IDENTITY_RECEIPT.json"
IDS_NAME = "identities.npy"
POSITIONS_NAME = "global_positions.npy"
MAP_NAME = "vehicle_map.json"
TRAIN_IDENTITIES_ENV = "WM_N1_TRAIN_IDENTITIES"
_BATCH_CHUNKS = {
    "chunk1": "chunk1", "chunk2": "chunk2", "chunk3": "chunk3",
    "chunk4": "chunk4", "chunk1_fix27": "chunk1_fix27",
    "truearms_chunk2": "chunk2", "truearms_chunk3": "chunk3",
    "walls_pilot_chunk4": "chunk4",
}
_RUN_FIELDS = ("key", "batch", "chunk", "world", "arm", "split", "t0_s",
               "tick_start", "tick_end", "n_ticks", "dec_start", "dec_end",
               "n_outcome_rows", "input_sha256", "shard_rel")


class IdentitySidecarError(PermissionError):
    """Identity provenance is incomplete or unsafe; refuse the operation."""


def _require(condition, message):
    if not condition:
        raise IdentitySidecarError(message)


def _int(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, f"invalid {name}")
    return value


def _digest(value):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
             "missing/invalid SHA256")
    return value


def _component(value):
    _require(isinstance(value, str) and value not in ("", ".", "..")
             and not re.search(r"[\\/:\x00-\x1f]", value)
             and value == value.strip() and not value.endswith("."),
             "invalid identity/path component")
    return value


def _protected(text):
    norm = re.sub(r"[\s_.-]+", "", str(text)).lower()
    return any(word in norm for word in
               ("holdout", "calibration", "calsplit", "chunks5", "chunk5",
                "j4b", "inc1", "sealed")) or norm == "cal"


def _path(path, *, kind, fresh=False):
    """Check lexical path and all ancestors, including Windows junctions.

    No resolve/stat follows a reparse point: inspect components from root to
    leaf with lstat first. Reject traversal even when it normalizes in-bounds.
    """
    raw = Path(path)
    _require(".." not in raw.parts and not any(_protected(p) for p in raw.parts),
             "protected/traversing path")
    absolute = Path(os.path.abspath(raw))
    for component in reversed((absolute, *absolute.parents)):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        _require(not stat.S_ISLNK(info.st_mode)
                 and not (getattr(info, "st_file_attributes", 0) & 0x400),
                 "symlink/reparse point refused")
    _require(absolute.resolve(strict=False) == absolute,
             "resolved path differs from literal path")
    if fresh:
        _require(not absolute.exists(), "output directory must be fresh")
    elif kind == "file":
        _require(absolute.is_file(), "required regular file missing")
    else:
        _require(absolute.is_dir(), "required directory missing")
    return absolute


def _file(root, name):
    _component(name)
    root = _path(root, kind="dir")
    path = _path(root / name, kind="file")
    _require(path.parent == root, "file escapes selected directory")
    return path


def _sha(path):
    path = _path(path, kind="file")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _no_duplicate_keys(pairs):
    out = {}
    for key, value in pairs:
        _require(key not in out, "duplicate JSON key")
        out[key] = value
    return out


def _json(path):
    path = _path(path, kind="file")
    try:
        return json.loads(path.read_text(encoding="utf-8"),
                          object_pairs_hook=_no_duplicate_keys)
    except (ValueError, UnicodeError) as exc:
        raise IdentitySidecarError("invalid JSON") from exc


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _write_json(path, obj):
    # Exclusive creation, no replacement/removal; partial failures survive.
    with path.open("xb") as stream:
        stream.write(_canonical(obj))


def _relative_shard(record):
    """Historical source paths are parsed lexically on every OS, never opened."""
    raw = record.get("shard_dir")
    _require(isinstance(raw, str) and raw, "missing recorded shard identity")
    portable = raw.replace("\\", "/")
    parts = PurePosixPath(portable).parts
    _require(".." not in parts and not any(_protected(p) for p in parts),
             "unsafe recorded shard identity")
    absolute = PurePosixPath(portable).is_absolute() or PureWindowsPath(raw).is_absolute()
    _require(len(parts) >= 3 if absolute else len(parts) == 3,
             "invalid recorded shard path")
    tail = parts[-3:]
    _require(tail[0] in (record["batch"], record["chunk"])
             and tail[1:] == (record["world"], record["arm"]),
             "recorded shard does not match run identity")
    return "/".join(_component(p) for p in tail)


def _metadata_identity(record):
    _require(isinstance(record, dict), "run/item must be an object")
    for name in ("batch", "chunk", "world", "arm"):
        _component(record.get(name))
    _require(record["batch"] in _BATCH_CHUNKS
             and _BATCH_CHUNKS[record["batch"]] == record["chunk"],
             "unknown or inconsistent batch/chunk")
    _require(record.get("split") in ("TRAIN", "CALIBRATION"),
             "unknown or forbidden split")
    _require(not any(_protected(record[n]) for n in ("world", "arm")),
             "protected run identity")
    t0 = record.get("t0_s")
    _require(type(t0) in (int, float) and math.isfinite(t0), "invalid t0_s")
    return record["batch"], record["world"], record["arm"]


def _plan(pack_dir):
    """Fully reconcile mixed metadata BEFORE any selected shard open."""
    pack = _path(pack_dir, kind="dir")
    mp, ip = _file(pack, "manifest.json"), _file(pack, "INDEX.json")
    manifest, index = _json(mp), _json(ip)
    _require(isinstance(manifest, dict) and isinstance(index, dict),
             "invalid pack metadata")
    _require(manifest.get("schema") == "wm_n1_pack/1"
             and index.get("schema") == "wm_n1_corpus_pack/2", "unknown pack schema")
    width = _int(manifest.get("k_max"), "k_max", 1)
    _require(index.get("k_max") == width, "INDEX/manifest width mismatch")
    fmt = manifest.get("format")
    _require(fmt == index.get("format") and fmt in (None, "v2", "v3"),
             "unknown/inconsistent format")
    _require(fmt is None or width == {"v2": 128, "v3": 192}[fmt],
             "format width mismatch")
    items, runs = manifest.get("items"), index.get("runs")
    _require(isinstance(items, list) and items and isinstance(runs, list) and runs,
             "missing manifest items/INDEX runs")
    by_identity, ordered = {}, []
    tick_cursor = row_cursor = dec_cursor = 0
    for raw in runs:
        rec = dict(raw) if isinstance(raw, dict) else {}
        identity = _metadata_identity(rec)
        _require(identity not in by_identity, "duplicate INDEX run")
        _require(rec.get("key") == f"{identity[0]}/{identity[1]}__{identity[2]}",
                 "missing/inconsistent INDEX run key")
        for name in ("tick_start", "tick_end", "n_ticks", "dec_start", "dec_end",
                     "n_outcome_rows"):
            _int(rec.get(name), name)
        _require(rec["tick_start"] == tick_cursor and rec["n_ticks"] > 0
                 and rec["tick_end"] == tick_cursor + rec["n_ticks"],
                 "invalid/overlapping tick ranges")
        _require(rec["dec_start"] == dec_cursor and rec["dec_end"] > dec_cursor,
                 "invalid decision ranges")
        _digest(rec.get("input_sha256"))
        rec["shard_rel"] = _relative_shard(rec)
        rec["outcome_first"] = row_cursor
        tick_cursor, dec_cursor = rec["tick_end"], rec["dec_end"]
        row_cursor += rec["n_outcome_rows"]
        by_identity[identity] = rec
        ordered.append(rec)
    # Width/tick validation uses descriptors only: no packed arrays are opened.
    for meta in (manifest, index):
        arrays = meta.get("arrays")
        _require(isinstance(arrays, dict), "missing array metadata")
        for name in ("veh_step", "veh_mask", "plan", "accel_next", "target_mask", "lane_next"):
            spec = arrays.get(name, {})
            shape = spec.get("shape", [])
            _require(isinstance(shape, list) and len(shape) >= 2
                     and all(type(x) is int and x >= 0 for x in shape)
                     and shape[1] == width
                     and shape[0] == (dec_cursor if name == "plan" else tick_cursor),
                     "array descriptor width/tick mismatch")
    selected, seen = [], set()
    for position, raw in enumerate(items):
        item = dict(raw) if isinstance(raw, dict) else {}
        # Older manifest writers use chunk as batch; only this documented fallback.
        item.setdefault("batch", item.get("chunk"))
        identity = _metadata_identity(item)
        _require(identity in by_identity and identity not in seen,
                 "unmatched/duplicate manifest identity")
        seen.add(identity)
        rec = by_identity[identity]
        for name in ("split", "chunk", "t0_s"):
            _require(item.get(name) == rec[name], "manifest/INDEX identity mismatch")
        tick = _int(item.get("tick"), "item tick")
        plan_row = _int(item.get("plan_row"), "item plan_row")
        _require(rec["tick_start"] <= tick < rec["tick_end"]
                 and rec["dec_start"] <= plan_row < rec["dec_end"],
                 "item tick/plan row outside its run")
        rows = item.get("outcome_rows")
        expected = list(range(rec["outcome_first"], rec["outcome_first"] + rec["n_outcome_rows"]))
        _require(isinstance(rows, list) and all(type(x) is int for x in rows)
                 and rows == expected, "item outcome coverage mismatch")
        _require(item.get("pair_key") == f"{rec['world']}@t0={float(rec['t0_s']):g}",
                 "item pair key mismatch")
        if rec["split"] == "TRAIN":
            selected.append(dict(position=position, tick=tick, plan_row=plan_row,
                                 **{name: rec[name] for name in _RUN_FIELDS}))
    _require(len(seen) == len(ordered) and selected, "missing run/item/TRAIN selection")
    return dict(manifest_sha256=_sha(mp), index_sha256=_sha(ip), k_max=width,
                n_items=len(items), selection=selected)


def _output(out_dir, pack_dir, shards_root):
    out = _path(out_dir, kind="dir", fresh=True)
    for tree in (pack_dir, shards_root):
        root = _path(tree, kind="dir")
        _require(out != root and root not in out.parents and out not in root.parents,
                 "output overlaps pack/source tree")
    _require(not any(re.sub(r"[_.-]", "", p).lower() in
                     ("archive", "archives", "frozen", "protocols", "baselines")
                     for p in out.parts), "output inside frozen tree")
    return out


def _source(root, selection, k_max):
    shard = _path(root / selection["shard_rel"], kind="dir")
    _require(root in shard.parents, "shard escapes source root")
    done_path = _file(shard, "DONE.json")
    done = _json(done_path)
    _metadata_identity(done)
    for name in ("batch", "chunk", "world", "arm", "split", "t0_s", "n_ticks",
                 "input_sha256", "n_outcome_rows"):
        _require(done.get(name) == selection[name], "DONE/source identity mismatch")
    _require(done.get("schema") == "wm_n1_features_index/1"
             and _relative_shard(done) == selection["shard_rel"],
             "DONE schema/path mismatch")
    # n_slots counts columns occupied at any tick, not distinct vehicles.
    # A reusable column can reference many entries in the run-local VID table.
    n_slots = _int(done.get("n_slots"), "DONE n_slots")
    _require(n_slots <= k_max, "DONE n_slots exceeds slot width")
    outputs = done.get("outputs")
    _require(isinstance(outputs, dict), "missing DONE outputs")
    idx, vids = _file(shard, "slot_index.npy"), _file(shard, "slot_vid.json")
    idx_sha, vids_sha = _sha(idx), _sha(vids)
    _require(idx_sha == _digest(outputs.get("slot_index"))
             and vids_sha == _digest(outputs.get("slot_vid.json")),
             "source identity hash mismatch")
    vid_list = _json(vids)
    _require(isinstance(vid_list, list)
             and all(isinstance(x, str) and x for x in vid_list)
             and len(set(vid_list)) == len(vid_list),
             "missing/duplicate vehicle map")
    return idx, vid_list, dict(done_sha256=_sha(done_path),
                               slot_index_sha256=idx_sha, slot_vid_sha256=vids_sha), n_slots


def build_identity_sidecar(pack_dir, shards_root, out_dir):
    """Build once from selected TRAIN sources into an exclusively fresh path.

    Canonical IDs enumerate sorted (chunk, world, actual VID) tuples, independent
    of run-local numeric indices. -1 denotes an absent slot. Receipt and partial
    output are retained on failure. No outcomes or other packed arrays are read.
    """
    out = _output(out_dir, pack_dir, shards_root)
    plan = _plan(pack_dir)  # Complete TRAIN selection before source access.
    root = _path(shards_root, kind="dir")
    out.mkdir(parents=True, exist_ok=False)
    _write_json(out / "BUILD_STARTED.json", dict(schema=SCHEMA, pack=plan))
    try:
        rows, sources, all_vehicles = [], [], set()
        for selection in plan["selection"]:
            idx_path, vids, source, n_slots = _source(root, selection, plan["k_max"])
            slots = np.load(_path(idx_path, kind="file"), allow_pickle=False, mmap_mode="r")
            _require(slots.dtype == np.dtype("int32")
                     and slots.shape == (selection["n_ticks"], plan["k_max"]),
                     "source slot array dtype/shape mismatch")
            _require(bool(np.all((slots >= -1) & (slots < len(vids)))),
                     "slot references missing vehicle identity")
            _require(int(np.any(slots >= 0, axis=0).sum()) == n_slots,
                     "DONE n_slots disagrees with occupied slot columns")
            for tick_row in slots:
                present = tick_row[tick_row >= 0]
                _require(len(np.unique(present)) == len(present),
                         "duplicate vehicle in source tick")
            row = slots[selection["tick"] - selection["tick_start"]]
            keys = [None if int(i) == -1 else
                    (selection["chunk"], selection["world"], vids[int(i)]) for i in row]
            all_vehicles.update(k for k in keys if k is not None)
            rows.append(keys)
            sources.append(dict(selection=selection, **source))
            del row, slots
        vehicle_map = sorted(all_vehicles)
        lookup = {key: i for i, key in enumerate(vehicle_map)}
        ids = np.asarray([[lookup[k] if k is not None else -1 for k in row]
                          for row in rows], dtype=np.int64)
        positions = np.asarray([s["position"] for s in plan["selection"]], dtype=np.int64)
        for name, array in ((IDS_NAME, ids), (POSITIONS_NAME, positions)):
            with (out / name).open("xb") as stream:
                np.save(stream, array, allow_pickle=False)
        _write_json(out / MAP_NAME, [list(k) for k in vehicle_map])
        receipt = dict(schema=SCHEMA, split="TRAIN", pack=plan, sources=sources,
                       n_vehicles=len(vehicle_map), namespace=["chunk", "world", "vehicle_id"],
                       files={name: _sha(_file(out, name))
                              for name in (IDS_NAME, POSITIONS_NAME, MAP_NAME)})
        receipt["receipt_sha256"] = hashlib.sha256(_canonical(receipt)).hexdigest()
        _write_json(out / RECEIPT_NAME, receipt)
        return receipt
    except Exception as exc:
        _write_json(out / "BUILD_FAILED.json", dict(schema=SCHEMA, error_type=type(exc).__name__,
                                                   message=str(exc)))
        raise


class IdentitySidecar:
    """Read-only snapshot IDs indexed by GLOBAL manifest position.

    ``vehicle_map`` (a list of ``(chunk, world, vehicle_id)`` tuples, canonical
    index = row position) is retained so a CONCATENATED corpus can re-project
    every pack's row IDs into ONE shared canonical index space: two packs carry
    independent local index spaces, so integer equality alone would not
    identify the same physical vehicle across packs.  ``None`` keeps the
    historical bytes-only object (only possible for TEST-ONLY injected
    loaders).  This is metadata, never a learned input channel.
    """

    def __init__(self, ids, positions, receipt_sha256, vehicle_map=None):
        # Immutable bytes backing prevents callers from re-enabling write flags.
        self._ids = np.frombuffer(ids.tobytes(), dtype=np.int64).reshape(ids.shape)
        self._rows = {int(p): i for i, p in enumerate(positions)}
        self.receipt_sha256 = receipt_sha256
        self.vehicle_map = (None if vehicle_map is None
                            else [tuple(k) for k in vehicle_map])

    def __getstate__(self):
        return (self._ids.tobytes(), self._ids.shape, self._rows,
                self.receipt_sha256, self.vehicle_map)

    def __setstate__(self, state):
        data, shape, self._rows, self.receipt_sha256, self.vehicle_map = state
        self._ids = np.frombuffer(data, dtype=np.int64).reshape(shape)

    def ids_for_item(self, global_manifest_position):
        position = _int(global_manifest_position, "global manifest position")
        _require(position in self._rows, "item lacks verified TRAIN identities")
        return self._ids[self._rows[position]]


def load_identity_sidecar(pack_dir, sidecar_dir):
    """Validate receipt/metadata/hashes before array parsing; no source access.

    Receipt digests detect corruption, not a malicious rewrite of an entire
    bundle. Deployment must externally pin its receipt SHA256 in the run record.
    All filenames are constants; receipt paths are never interpreted as paths.
    pack_dir must be the independently validated canonical pack directory.
    Deployment aliases must be resolved by an approved caller, never implicitly
    here; record the alias separately from the canonical target and metadata SHAs.
    """
    side = _path(sidecar_dir, kind="dir")
    receipt = _json(_file(side, RECEIPT_NAME))
    _require(isinstance(receipt, dict), "invalid identity receipt")
    payload = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    _require(hashlib.sha256(_canonical(payload)).hexdigest()
             == _digest(receipt.get("receipt_sha256")), "receipt digest mismatch")
    _require(set(receipt) == {"schema", "split", "pack", "sources", "n_vehicles",
                              "namespace", "files", "receipt_sha256"}
             and receipt["schema"] == SCHEMA and receipt["split"] == "TRAIN"
             and receipt["namespace"] == ["chunk", "world", "vehicle_id"],
             "unknown identity receipt metadata")
    plan = _plan(pack_dir)
    _require(receipt["pack"] == plan, "receipt does not bind this pack/selection")
    sources = receipt["sources"]
    _require(isinstance(sources, list) and len(sources) == len(plan["selection"]),
             "source coverage mismatch")
    for source, selection in zip(sources, plan["selection"]):
        _require(isinstance(source, dict) and set(source) == {
            "selection", "done_sha256", "slot_index_sha256", "slot_vid_sha256"}
            and source["selection"] == selection, "source identity mismatch")
        for name in ("done_sha256", "slot_index_sha256", "slot_vid_sha256"):
            _digest(source[name])
    count = _int(receipt["n_vehicles"], "n_vehicles")
    files = receipt["files"]
    _require(isinstance(files, dict) and set(files) == {IDS_NAME, POSITIONS_NAME, MAP_NAME},
             "unexpected sidecar files")
    for name in (IDS_NAME, POSITIONS_NAME, MAP_NAME):
        _require(_sha(_file(side, name)) == _digest(files[name]), "sidecar hash mismatch")
    mapping = _json(_file(side, MAP_NAME))
    _require(isinstance(mapping, list) and len(mapping) == count, "vehicle map size mismatch")
    keys = []
    allowed_namespaces = {(s["chunk"], s["world"]) for s in plan["selection"]}
    for entry in mapping:
        _require(isinstance(entry, list) and len(entry) == 3
                 and all(isinstance(x, str) and x for x in entry)
                 and tuple(entry[:2]) in allowed_namespaces, "invalid vehicle namespace")
        keys.append(tuple(entry))
    _require(keys == sorted(set(keys)), "vehicle map is duplicated/unsorted")
    ids = np.load(_file(side, IDS_NAME), allow_pickle=False)
    positions = np.load(_file(side, POSITIONS_NAME), allow_pickle=False)
    _require(ids.dtype == np.dtype("int64") and positions.dtype == np.dtype("int64")
             and ids.shape == (len(plan["selection"]), plan["k_max"])
             and positions.shape == (len(plan["selection"]),), "sidecar shape/dtype mismatch")
    _require(positions.tolist() == [s["position"] for s in plan["selection"]]
             and bool(np.all((ids >= -1) & (ids < count))), "sidecar position/ID mismatch")
    for row, selection in zip(ids, plan["selection"]):
        present = row[row >= 0]
        _require(len(np.unique(present)) == len(present), "duplicate sidecar vehicle")
        _require(all(keys[int(i)][:2] == (selection["chunk"], selection["world"])
                     for i in present), "sidecar row vehicle namespace mismatch")
    _require(set(int(x) for x in ids[ids >= 0]) == set(range(count)),
             "unreferenced/missing canonical vehicles")
    return IdentitySidecar(ids, positions, _sha(_file(side, RECEIPT_NAME)),
                           vehicle_map=keys)
