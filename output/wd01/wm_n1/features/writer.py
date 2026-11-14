"""Memmap shard writer + JSON index + SHA256SUMS (brief writer item).

One directory per run under ``<data_root>/<chunk>/<world>/<arm>/``.  Large
tensors are written with ``numpy.lib.format.open_memmap`` (npy).  A shard is
complete iff ``DONE.json`` exists, records the input physics sha256 and the
sha256 of every output file, and those still verify.  Completed shards are
skipped (resumable by sha); a non-empty shard directory without a valid
DONE.json is never silently overwritten unless ``--force`` is given.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import constants as C

NPY_FILES = ("veh_step", "veh_mask", "slot_index", "plan", "decision_t",
             "wall", "wall_mask", "counter", "accel_next", "target_mask",
             "lane_next", "arrivals")
NON_NPY = ("slot_vid.json", "meta.json", "outcomes.parquet")


class WriterError(RuntimeError):
    """A writer contract violation (overwrite refusal)."""


def sha256_file(path, block=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def shard_dir(data_root, chunk, world, arm) -> Path:
    return Path(data_root) / str(chunk) / str(world) / str(arm)


def expected_bytes(n_ticks: int, n_slots: int, n_dec: int = C.N_DECISION,
                   has_outcomes: bool = True,
                   k_max: int = C.K_MAX) -> int:
    """Expected uncompressed shard bytes for one run (estimate mode).

    ``k_max`` is the format K (v2 = 128 default, v3 = 192).
    """
    K = int(k_max)
    total = 0
    total += n_ticks * K * C.VEH_F * 2            # veh_step float16
    total += n_ticks * K * 1                       # veh_mask uint8
    total += n_ticks * K * 4                       # slot_index int32
    total += n_dec * K * C.N_SEGMENTS * C.PLAN_F * 2
    total += n_dec * 8                             # decision_t float64
    total += n_ticks * C.MAX_WALLS * C.WALL_F * 2
    total += n_ticks * C.MAX_WALLS * 1
    total += n_ticks * C.N_ENTRANCES * C.COUNTER_BINS * 2
    total += n_ticks * K * 2                       # accel_next float16
    total += n_ticks * K * 1                       # target_mask
    total += n_ticks * K * 1                       # lane_next int8
    total += n_ticks * C.N_ENTRANCES * 2           # arrivals int16
    total += n_slots * 24                          # slot_vid json (approx)
    total += 8192 if has_outcomes else 2048        # outcomes/meta
    return int(total)


def _write_npy(path: Path, arr: np.ndarray) -> None:
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=arr.dtype,
                                   shape=arr.shape)
    mm[...] = arr
    mm.flush()
    del mm


def _verify_done(dirp: Path, done: dict) -> bool:
    for name in NPY_FILES:
        rel = done.get("outputs", {}).get(name)
        if rel is None:
            return False
        p = dirp / f"{name}.npy"
        if not p.is_file() or sha256_file(p) != rel:
            return False
    return True


def write_shard(shard: dict, data_root, *, force: bool = False) -> dict:
    """Write one run shard.  Returns a record; skips if already complete.

    The directory is ``<data_root>/<batch|chunk>/<world>/<arm>``; ``arm`` is
    the exact run-dir suffix, ``batch`` (when present in the shard meta)
    disambiguates cross-batch runs sharing world + arm.
    """
    meta = shard["meta"]
    dirp = shard_dir(data_root, meta.get("batch") or meta["chunk"],
                     meta["world"], meta["arm"])
    input_sha = sha256_file(meta["input_physics"])
    done_path = dirp / "DONE.json"

    if dirp.exists():
        if not force and done_path.is_file():
            try:
                done = json.loads(done_path.read_text(encoding="utf-8"))
            except Exception:
                done = None
            if done and done.get("input_sha256") == input_sha \
                    and _verify_done(dirp, done):
                return dict(record=done, status="skipped")
        if any(dirp.iterdir()) and not force:
            raise WriterError(
                f"shard {dirp} already exists and is not a verified "
                f"completed shard; refusing to overwrite (pass --force)")
    dirp.mkdir(parents=True, exist_ok=True)

    arrays = {k: shard[k] for k in
              ("veh_step", "veh_mask", "slot_index", "plan", "decision_t",
               "wall", "wall_mask", "counter", "accel_next", "target_mask",
               "lane_next", "arrivals")}
    for name, arr in arrays.items():
        _write_npy(dirp / f"{name}.npy", arr)

    (dirp / "slot_vid.json").write_text(
        json.dumps(shard["slot_vid"], indent=0), encoding="utf-8")
    meta_out = dict(meta)
    meta_out["schema"] = C.FORMAT_SCHEMAS.get(
        meta.get("format"), C.FORMAT_SCHEMA)
    meta_out["shapes"] = {k: list(np.asarray(v).shape) for k, v in
                          arrays.items()}
    meta_out["dtypes"] = {k: str(np.asarray(v).dtype) for k, v in
                          arrays.items()}
    (dirp / "meta.json").write_text(
        json.dumps(meta_out, indent=1, default=str), encoding="utf-8")
    outcomes = shard.get("outcomes")
    n_out = 0
    if outcomes is not None and len(outcomes) > 0:
        outcomes.to_parquet(dirp / "outcomes.parquet")
        n_out = int(len(outcomes))

    outputs = {}
    for name in NPY_FILES:
        outputs[name] = sha256_file(dirp / f"{name}.npy")
    outputs["slot_vid.json"] = sha256_file(dirp / "slot_vid.json")
    outputs["meta.json"] = sha256_file(dirp / "meta.json")
    if n_out:
        outputs["outcomes.parquet"] = sha256_file(dirp / "outcomes.parquet")

    sizes = {name: (dirp / name).stat().st_size for name in
             [f"{n}.npy" for n in NPY_FILES] + ["slot_vid.json", "meta.json"]}
    if n_out:
        sizes["outcomes.parquet"] = (dirp / "outcomes.parquet").stat().st_size
    total_bytes = sum(sizes.values())
    done = dict(
        schema=C.INDEX_SCHEMA, chunk=meta["chunk"],
        batch=meta.get("batch") or meta["chunk"],
        world=meta["world"], arm=meta["arm"], split=meta["split"],
        policy=meta.get("policy"), t0_s=meta["t0_s"],
        n_ticks=meta["n_ticks"], n_slots=meta["n_slots"],
        n_dropped_slots=meta["n_dropped_slots"],
        n_dropped_slot_steps=meta["n_dropped_slot_steps"],
        input_sha256=input_sha, shard_dir=str(dirp),
        outputs=outputs, sizes=sizes, total_bytes=int(total_bytes),
        n_outcome_rows=n_out,
        written_utc=datetime.now(timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"))
    done_path.write_text(json.dumps(done, indent=1), encoding="utf-8")
    return dict(record=done, status="written")


# --------------------------------------------------------------- index ----
def index_path(data_root) -> Path:
    return Path(data_root) / "INDEX.json"


def load_index(data_root) -> dict:
    p = index_path(data_root)
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return dict(schema=C.INDEX_SCHEMA, runs={}, pairs={}, totals={})


def pair_key(world: str, t0_s: float) -> str:
    return f"{world}@t0={t0_s:g}"


def update_index(index: dict, record: dict, pair_groups: bool = True) -> dict:
    key = (f"{record.get('batch') or record['chunk']}/"
           f"{record['world']}__{record['arm']}")
    index["runs"][key] = dict(
        chunk=record["chunk"],
        batch=record.get("batch") or record["chunk"],
        world=record["world"], arm=record["arm"],
        split=record["split"], policy=record.get("policy"),
        t0_s=record["t0_s"],
        n_ticks=record["n_ticks"], total_bytes=record["total_bytes"],
        shard_dir=record["shard_dir"], input_sha256=record["input_sha256"])
    if pair_groups:
        pk = pair_key(record["world"], record["t0_s"])
        index["pairs"].setdefault(pk, [])
        if key not in index["pairs"][pk]:
            index["pairs"][pk].append(key)
    index["totals"] = dict(
        n_runs=len(index["runs"]),
        total_bytes=int(sum(r["total_bytes"] for r in index["runs"].values())),
        updated_utc=datetime.now(timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"))
    return index


def save_index(data_root, index: dict) -> Path:
    p = index_path(data_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(index, indent=1, default=str), encoding="utf-8")
    return p


def write_sums(data_root, index: dict) -> Path:
    """SHA256SUMS over every shard file registered in the index."""
    lines = []
    for rec in sorted(index["runs"].values(),
                      key=lambda r: (r["chunk"], r["world"], r["arm"])):
        dirp = Path(rec["shard_dir"])
        for name in sorted(list((dirp / "DONE.json").parent.iterdir())):
            if name.name == "DONE.json":
                continue
            rel = name.relative_to(Path(data_root)).as_posix()
            lines.append(f"{sha256_file(name)}  {rel}")
    p = Path(data_root) / "SHA256SUMS"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def verify_index(data_root, index: dict) -> dict:
    """Verify every registered shard's outputs against its DONE.json."""
    bad = []
    for key, rec in index["runs"].items():
        dirp = Path(rec["shard_dir"])
        done_p = dirp / "DONE.json"
        if not done_p.is_file():
            bad.append(f"{key}: missing DONE.json")
            continue
        done = json.loads(done_p.read_text(encoding="utf-8"))
        if done.get("input_sha256") != rec.get("input_sha256"):
            bad.append(f"{key}: input sha drift")
        if not _verify_done(dirp, done):
            bad.append(f"{key}: output sha mismatch")
    return dict(ok=not bad, n_runs=len(index["runs"]), errors=bad)
