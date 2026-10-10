# ruff: noqa
# VENDORED from output/wd01/wm_n1/dspeed/identity.py (last changed in commit df6a02f; copied at baseline bf7d4cb).
# Only change: absolute imports of sibling modules re-pointed to this vendor package.
# Do not edit: behaviour is frozen evidence; WP-C only mirrors it.
"""A9: TRAIN identity sidecar for the extra (dspeed) pack, from physics.

The pack_v3 identity sidecar is built by :mod:`train.identity_sidecar` from the
feature shard tree (``DONE.json`` + ``slot_index.npy`` + ``slot_vid.json``).
The dspeed pack (``dspeed.build_pack``) streams featurization straight into the
pack and keeps only the light per-run alignment receipts
(``meta.json`` / ``slot_vid.json`` / ``ALIGN.json``) -- there is no
``slot_index.npy``.  This module therefore reconstructs each run's slot -> VID
row with the EXACT frozen code that featurized the pack
(:func:`features.featurizer.parse_run` over the run's
``physics_steps.jsonl.gz``) and emits the SAME sidecar object/schema
(``identities.npy`` / ``global_positions.npy`` / ``vehicle_map.json`` + a
receipt) with the SAME canonical-ID convention
(``(chunk, world, vehicle_id)`` sorted).

Provenance is fail-closed: the shard ``ANCHOR`` (ALIGN) must agree with the pack
INDEX selection on world/arm/chunk/split/t0/n_ticks and its recorded physics
SHA256 must equal the INDEX ``input_sha256``; the physics-derived ``slot_vid``
must equal the shard ``slot_vid.json`` and ``meta.json``.  No TRAIN-filtered
position is ever used: rows are keyed by the ORIGINAL pack manifest position.

Writes only under the caller-provided fresh output directory; the physics tree
is opened read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from wms.legacy.vendor.train import identity_sidecar as I

SCHEMA = I.SCHEMA
SOURCE = "physics"
_RECEIPT_KEYS = {"schema", "split", "pack", "sources", "n_vehicles", "namespace",
                 "source", "files", "receipt_sha256"}
_SOURCE_KEYS = {"selection", "physics_sha256", "slot_vid_sha256", "align_sha256"}


def _require(condition, message):
    if not condition:
        raise I.IdentitySidecarError(message)


def _physics(path):
    """Lexically safe external physics path (no reparse refusal: the campaign
    tree is a read-only external root, never a training output)."""
    raw = Path(path)
    _require(raw.is_absolute() and ".." not in raw.parts
             and not any(I._protected(p) for p in raw.parts),
             "unsafe recorded physics path")
    _require(raw.is_file(), "recorded physics file missing")
    return raw


def _string_list(value, message):
    _require(isinstance(value, list)
             and all(isinstance(x, str) and x for x in value)
             and len(set(value)) == len(value), message)
    return value


def _physics_row(root, selection, k_max):
    shard_dir = I._path(root.joinpath(*selection["shard_rel"].split("/")),
                        kind="dir")
    _require(root in shard_dir.parents, "shard escapes source root")
    align = I._json(I._file(shard_dir, "ALIGN.json"))
    meta = I._json(I._file(shard_dir, "meta.json"))
    vids = _string_list(I._json(I._file(shard_dir, "slot_vid.json")),
                        "missing/duplicate vehicle map")
    # ALIGN anchors the run identity to the pack INDEX selection.
    for name in ("world", "arm", "chunk", "split"):
        _require(align.get(name) == selection[name],
                 f"ALIGN/selection {name} mismatch")
    _require(abs(float(align.get("t0_s")) - float(selection["t0_s"])) <= 1e-9
             and int(align.get("n_ticks")) == int(selection["n_ticks"]),
             "ALIGN t0/n_ticks mismatch")
    _require(align.get("batch") == selection["batch"],
             "ALIGN/selection batch mismatch")
    _require(I._digest(align.get("physics_gz_sha256")) == selection["input_sha256"],
             "ALIGN physics SHA256 disagrees with the pack INDEX")
    _require(str(meta.get("world")) == selection["world"]
             and str(meta.get("arm")) == selection["arm"]
             and str(meta.get("chunk")) == selection["chunk"]
             and abs(float(meta.get("t0_s")) - float(selection["t0_s"])) <= 1e-9
             and int(meta.get("n_ticks")) == int(selection["n_ticks"])
             and _string_list(meta.get("slot_vid"),
                              "meta slot_vid invalid") == vids,
             "meta.json disagrees with the selection/slot_vid")
    physics = _physics(align["input_physics"])
    from features import featurizer as FZ

    parsed = FZ.parse_run(shard_dir, float(selection["t0_s"]), k_max=k_max,
                          physics_path=physics)
    _require(_string_list(parsed["slot_vid"], "parsed slot_vid invalid") == vids,
             "physics slot_vid disagrees with the shard slot_vid")
    _require(int(parsed["n_ticks"]) == int(selection["n_ticks"]),
             "physics tick count disagrees with the pack INDEX")
    slots = parsed["slot_index"]
    _require(slots.dtype == np.dtype("int32")
             and slots.shape == (int(selection["n_ticks"]), int(k_max)),
             "physics slot_index dtype/shape mismatch")
    _require(bool(np.all((slots >= -1) & (slots < len(vids)))),
             "slot references missing vehicle identity")
    for tick_row in slots:
        present = tick_row[tick_row >= 0]
        _require(len(np.unique(present)) == len(present),
                 "duplicate vehicle in physics tick")
    row = slots[int(selection["tick"]) - int(selection["tick_start"])]
    source = dict(physics_sha256=selection["input_sha256"],
                  slot_vid_sha256=I._sha(I._file(shard_dir, "slot_vid.json")),
                  align_sha256=I._sha(I._file(shard_dir, "ALIGN.json")))
    return row, vids, source


def _row_task(task):
    """Worker: one selection -> (canonical keys row, source hashes)."""
    root, selection, k_max = task
    row, vids, source = _physics_row(Path(root), selection, k_max)
    keys = [None if int(i) == -1 else
            (selection["chunk"], selection["world"], vids[int(i)])
            for i in row]
    return keys, source


def build_dspeed_identity_sidecar(pack_dir, shards_root, out_dir,
                                  workers: int = 1):
    """Build once from the frozen physics slot assignment into a fresh path."""
    out = I._output(out_dir, pack_dir, shards_root)
    plan = I._plan(pack_dir)
    root = I._path(shards_root, kind="dir")
    out.mkdir(parents=True, exist_ok=False)
    I._write_json(out / "BUILD_STARTED.json",
                  dict(schema=SCHEMA, source=SOURCE, pack=plan))
    try:
        tasks = [(str(root), sel, plan["k_max"]) for sel in plan["selection"]]
        if workers and int(workers) > 1:
            with Pool(int(workers)) as pool:
                results = pool.map(_row_task, tasks, chunksize=1)
        else:
            results = [_row_task(t) for t in tasks]
        all_vehicles = set()
        sources = []
        for selection, (keys, source) in zip(plan["selection"], results):
            all_vehicles.update(k for k in keys if k is not None)
            sources.append(dict(selection=selection, **source))
        rows = [keys for keys, _ in results]
        vehicle_map = sorted(all_vehicles)
        lookup = {key: i for i, key in enumerate(vehicle_map)}
        ids = np.asarray([[lookup[k] if k is not None else -1 for k in row]
                          for row in rows], dtype=np.int64)
        positions = np.asarray([s["position"] for s in plan["selection"]],
                               dtype=np.int64)
        for name, array in ((I.IDS_NAME, ids), (I.POSITIONS_NAME, positions)):
            with (out / name).open("xb") as stream:
                np.save(stream, array, allow_pickle=False)
        I._write_json(out / I.MAP_NAME, [list(k) for k in vehicle_map])
        receipt = dict(schema=SCHEMA, split="TRAIN", pack=plan, sources=sources,
                       n_vehicles=len(vehicle_map), source=SOURCE,
                       namespace=["chunk", "world", "vehicle_id"],
                       files={name: I._sha(I._file(out, name))
                              for name in (I.IDS_NAME, I.POSITIONS_NAME,
                                           I.MAP_NAME)})
        receipt["receipt_sha256"] = hashlib.sha256(
            I._canonical(receipt)).hexdigest()
        I._write_json(out / I.RECEIPT_NAME, receipt)
        return receipt
    except Exception as exc:
        I._write_json(out / "BUILD_FAILED.json",
                      dict(schema=SCHEMA, source=SOURCE,
                           error_type=type(exc).__name__, message=str(exc)))
        raise


def load_dspeed_identity_sidecar(pack_dir, sidecar_dir):
    """Validate a physics-sourced receipt/metadata/hashes; no source access.

    Symmetric to :func:`train.identity_sidecar.load_identity_sidecar`; the
    only difference is the per-source provenance keys (physics/align instead
    of DONE/slot_index).  Returns an :class:`I.IdentitySidecar` carrying the
    canonical ``vehicle_map`` for cross-pack ID re-projection.
    """
    side = I._path(sidecar_dir, kind="dir")
    receipt = I._json(I._file(side, I.RECEIPT_NAME))
    _require(isinstance(receipt, dict), "invalid identity receipt")
    payload = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    _require(hashlib.sha256(I._canonical(payload)).hexdigest()
             == I._digest(receipt.get("receipt_sha256")),
             "receipt digest mismatch")
    _require(set(receipt) == _RECEIPT_KEYS and receipt["schema"] == SCHEMA
             and receipt["split"] == "TRAIN" and receipt["source"] == SOURCE
             and receipt["namespace"] == ["chunk", "world", "vehicle_id"],
             "unknown identity receipt metadata")
    plan = I._plan(pack_dir)
    _require(receipt["pack"] == plan, "receipt does not bind this pack/selection")
    sources = receipt["sources"]
    _require(isinstance(sources, list) and len(sources) == len(plan["selection"]),
             "source coverage mismatch")
    for source, selection in zip(sources, plan["selection"]):
        _require(isinstance(source, dict) and set(source) == _SOURCE_KEYS
                 and source["selection"] == selection, "source identity mismatch")
        for name in ("physics_sha256", "slot_vid_sha256", "align_sha256"):
            I._digest(source[name])
    count = I._int(receipt["n_vehicles"], "n_vehicles")
    files = receipt["files"]
    _require(isinstance(files, dict)
             and set(files) == {I.IDS_NAME, I.POSITIONS_NAME, I.MAP_NAME},
             "unexpected sidecar files")
    for name in (I.IDS_NAME, I.POSITIONS_NAME, I.MAP_NAME):
        _require(I._sha(I._file(side, name)) == I._digest(files[name]),
                 "sidecar hash mismatch")
    mapping = I._json(I._file(side, I.MAP_NAME))
    _require(isinstance(mapping, list) and len(mapping) == count,
             "vehicle map size mismatch")
    keys = []
    allowed = {(s["chunk"], s["world"]) for s in plan["selection"]}
    for entry in mapping:
        _require(isinstance(entry, list) and len(entry) == 3
                 and all(isinstance(x, str) and x for x in entry)
                 and tuple(entry[:2]) in allowed, "invalid vehicle namespace")
        keys.append(tuple(entry))
    _require(keys == sorted(set(keys)), "vehicle map is duplicated/unsorted")
    ids = np.load(I._file(side, I.IDS_NAME), allow_pickle=False)
    positions = np.load(I._file(side, I.POSITIONS_NAME), allow_pickle=False)
    _require(ids.dtype == np.dtype("int64") and positions.dtype == np.dtype("int64")
             and ids.shape == (len(plan["selection"]), plan["k_max"])
             and positions.shape == (len(plan["selection"]),),
             "sidecar shape/dtype mismatch")
    _require(positions.tolist() == [s["position"] for s in plan["selection"]]
             and bool(np.all((ids >= -1) & (ids < count))),
             "sidecar position/ID mismatch")
    for row, selection in zip(ids, plan["selection"]):
        present = row[row >= 0]
        _require(len(np.unique(present)) == len(present),
                 "duplicate sidecar vehicle")
        _require(all(keys[int(i)][:2] == (selection["chunk"], selection["world"])
                     for i in present), "sidecar row vehicle namespace mismatch")
    _require(set(int(x) for x in ids[ids >= 0]) == set(range(count)),
             "unreferenced/missing canonical vehicles")
    return I.IdentitySidecar(ids, positions, I._sha(I._file(side, I.RECEIPT_NAME)),
                             vehicle_map=keys)


def load_extra_identity_sidecar(pack_dir, sidecar_dir):
    """Dispatch an extra-pack sidecar to the loader its receipt claims.

    The dspeed pack (``dspeed.build_pack``) carries physics-sourced identities
    (receipt ``source == "physics"``), while the damping-snap pack
    (``dspeed.build_damping``) reuses the standard shard-sourced
    :mod:`train.identity_sidecar` builder.  Both emit the same schema/objects;
    only the per-source provenance keys differ, so the receipt's ``source``
    selects the fail-closed validator.  An unknown/absent source uses the
    standard loader (whose provenance keys are the strict default).
    """
    from wms.legacy.vendor.train import identity_sidecar as I
    side = I._path(sidecar_dir, kind="dir")
    receipt = I._json(I._file(side, I.RECEIPT_NAME))
    if isinstance(receipt, dict) and receipt.get("source") == SOURCE:
        return load_dspeed_identity_sidecar(pack_dir, sidecar_dir)
    return I.load_identity_sidecar(pack_dir, sidecar_dir)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="dspeed.identity", description=__doc__)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--shards-root", dest="shards_root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int,
                        default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    args = parser.parse_args(argv)
    receipt = build_dspeed_identity_sidecar(args.pack, args.shards_root,
                                            args.out, workers=args.workers)
    print(json.dumps({"status": "ok", "n_vehicles": receipt["n_vehicles"],
                      "n_items": len(receipt["pack"]["selection"]),
                      "receipt_sha256": receipt["receipt_sha256"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
