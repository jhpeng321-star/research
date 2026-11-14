"""OWNER-APPROVED damping-snap pack build.

Streams the 263 TRAIN ``base`` / ``native_base`` campaign runs (read-only)
through ``features.featurizer.build_shard`` with ``snapshot_at_t0=True`` and
packs them into one ``wm_n1_pack/1`` v3 memmap corpus under
``_out/pack_damping``.  Uses the SAME streaming pack path as
:mod:`dspeed.build_pack` (imported helpers / array contract) so the bytes are
pack-contract compatible.

Per-run shard receipts (``meta.json`` / ``slot_vid.json`` / ``ALIGN.json``),
plus the identity-sidecar source artifacts ``slot_index.npy`` / ``DONE.json``,
are written under ``_out/shards_damping/`` so
:func:`train.identity_sidecar.build_identity_sidecar` can bind a verified
sidecar to this pack without a second pack copy.

Item metadata carries ``policy`` (``base -> damping``, ``native_base ->
native``) and the additive marker ``damping_snap=True`` so outcome supervision
is ENABLED for these re-aligned copies while the pack_v3 origin-0 copies stay
owner-(b) excluded.

Usage::

    python -m dspeed.build_damping --workers 6
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np  # noqa: E402

from . import common as D  # noqa: E402
from . import damping as DM  # noqa: E402
from .build_pack import DTYPES, K, SNAP_OFFSET_S, TAILS, _bytes_for_run

HERE = Path(__file__).resolve().parent
_WM_N1 = HERE.parent
if str(_WM_N1) not in sys.path:
    sys.path.insert(0, str(_WM_N1))
sys.path.insert(0, str(_WM_N1.parent / "output"))

from corpus import pack as PK  # noqa: E402
from features import constants as C  # noqa: E402
from features import featurizer as FZ  # noqa: E402
from features.idea_a import idea_a_channels_from_shard  # noqa: E402

_T0W = time.time()


def _log(msg):
    print(f"[{time.time() - _T0W:7.1f}s] {msg}", flush=True)


def _scan_full(physics: Path, t0: float) -> dict:
    """Count ticks and validate the re-aligned snapshot of a FULL run.

    Only rows at/after ``t0 + 0.5 s`` are considered; the first such row must
    be exactly ``t0 + 0.5`` and the kept rows must be contiguous on the
    ``0.5 s`` grid (fail closed, exactly like the branch ``_scan``).
    """
    import gzip
    opener = gzip.open if str(physics).endswith(".gz") else open
    lo = float(t0) + SNAP_OFFSET_S
    n = 0
    first = last = prev = None
    with opener(physics, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            t = float(json.loads(line)["time_s"])
            if t < lo - 1e-6:
                continue
            if first is None:
                first = t
            elif prev is not None and abs(t - prev - FZ.C.DT_S) > 1e-6:
                raise DM.DampingError(
                    f"{physics}: rows {prev} -> {t} are non-contiguous; "
                    "refusing")
            prev = last = t
            n += 1
    if first is None:
        raise DM.DampingError(
            f"{physics}: no physics rows at/after t0+0.5 ({lo})")
    if abs(first - lo) > 1e-6:
        raise DM.DampingError(
            f"{physics}: first re-aligned row {first} != t0+0.5 ({lo}); "
            "misaligned, refusing")
    n_ticks = int(round((last - float(t0)) / FZ.C.DT_S)) + 1
    if n != n_ticks - 1:
        raise DM.DampingError(
            f"{physics}: {n} rows over [{first}, {last}] disagree with "
            f"n_ticks {n_ticks}; refusing")
    return dict(n_ticks=n_ticks, first=first, last=last)


def _scan_task(run):
    return run["chunk"], run["world"], run["arm"], _scan_full(run["physics"],
                                                              run["t0_s"])


def _featurize_run(run):
    shard = FZ.build_shard(
        run["chunk"], run["world"], run["arm"], run["run_dir"],
        split="TRAIN", plan_row=dict(t0_s=run["t0_s"]),
        outcomes_root=str(DM.EMPTY_OUTCOMES), base_history_run_dir=None,
        batch=run["chunk"], policy=run["policy"], format="v3",
        physics_path=run["physics"], snapshot_at_t0=True)
    channels = idea_a_channels_from_shard(shard, fast=True)
    return shard, channels


def _write_identity_artifacts(sd: Path, run: dict, info: dict, shard) -> None:
    """Write the receipts + ``slot_index.npy`` / ``DONE.json`` needed by the
    identity sidecar (source hashes, n_slots) without a second pack copy."""
    sd.mkdir(parents=True, exist_ok=True)
    meta = dict(shard["meta"])
    meta["shard_dir"] = str(sd)
    D.atomic_write_json(sd / "meta.json", meta)
    (sd / "slot_vid.json").write_text(
        json.dumps(list(shard["slot_vid"])), encoding="utf-8")
    slot_index = np.asarray(shard["slot_index"], np.int32)
    np.save(sd / "slot_index.npy", slot_index)
    n_slots = int(np.any(slot_index >= 0, axis=0).sum())
    idx_sha = D.sha256_file(sd / "slot_index.npy")
    vids_sha = D.sha256_file(sd / "slot_vid.json")
    D.atomic_write_json(sd / "ALIGN.json", dict(
        world=run["world"], arm=run["arm"], chunk=run["chunk"],
        batch=run["chunk"], split="TRAIN", t0_s=run["t0_s"],
        t_origin=float(meta["t_origin"]),
        first_physics_time=float(info["first"]),
        snapshot_offset_s=SNAP_OFFSET_S, n_ticks=int(meta["n_ticks"]),
        snapshot_at_t0=True, input_physics=str(run["physics"]),
        physics_gz_sha256=run["input_sha256"]))
    D.atomic_write_json(sd / "DONE.json", dict(
        schema="wm_n1_features_index/1", batch=run["chunk"],
        chunk=run["chunk"], world=run["world"], arm=run["arm"],
        split="TRAIN", t0_s=float(run["t0_s"]),
        n_ticks=int(meta["n_ticks"]), input_sha256=run["input_sha256"],
        n_outcome_rows=0, shard_dir=str(sd), n_slots=n_slots,
        outputs={"slot_index": idx_sha, "slot_vid.json": vids_sha}))


def build(workers: int = 6) -> dict:
    free = shutil.disk_usage(str(D.OUT)).free
    if free < DM.MIN_FREE_BYTES:
        raise DM.DampingError(
            f"E: free {free/1e9:.2f} GB < {DM.MIN_FREE_BYTES/1e9:.0f} GB "
            "floor; stop")
    runs = DM.enumerate_runs()
    _log(f"enumerated {len(runs)} damping runs")

    out = DM.PACK_ROOT
    for root in (out, DM.SHARDS_ROOT):
        if root.exists():
            shutil.rmtree(root)
    out.mkdir(parents=True)
    DM.SHARDS_ROOT.mkdir(parents=True)

    _log("pass 1/2: scanning full-run physics for re-aligned tick totals")
    totals: dict = {}
    with Pool(workers) as pool:
        for chunk, world, arm, info in pool.imap_unordered(_scan_task, runs, 1):
            totals[(chunk, world, arm)] = info
    free = shutil.disk_usage(str(D.OUT)).free
    budget = free - DM.MIN_FREE_BYTES - int(0.3 * (1 << 30))
    kept, used = [], 0
    for run in runs:
        b = _bytes_for_run(totals[(run["chunk"], run["world"],
                                   run["arm"])]["n_ticks"])
        if used + b > budget:
            break
        kept.append(run)
        used += b
    n_dropped = len(runs) - len(kept)
    if n_dropped:
        raise DM.DampingError(
            f"disk budget drops {n_dropped} damping runs; refusing a partial "
            "supplementary pack (E: free "
            f"{free/1e9:.2f} GB, need {used/1e9:.2f} GB)")
    runs = kept
    n_ticks = sum(totals[(r["chunk"], r["world"], r["arm"])]["n_ticks"]
                  for r in runs)
    _log(f"total ticks={n_ticks} runs={len(runs)} "
         f"(projected {used/1e9:.2f} GB, free {free/1e9:.2f} GB)")

    shapes = {}
    for name in PK.TRAIN_ARRAYS:
        head = len(runs) * C.N_DECISION if name == "plan" else n_ticks
        shapes[name] = (int(head), *TAILS[name])
    targets = {n: np.lib.format.open_memmap(
        out / f"{n}.npy", mode="w+", dtype=DTYPES[n], shape=shapes[n])
        for n in PK.TRAIN_ARRAYS}
    offsets = {n: 0 for n in PK.TRAIN_ARRAYS}
    dec_off = 0
    runs_meta = []

    def _featurized():
        if workers > 1:
            with Pool(workers) as pool:
                for run, (shard, ch) in zip(
                        runs, pool.imap(_featurize_run, runs, 1)):
                    yield run, shard, ch
        else:
            for run in runs:
                shard, ch = _featurize_run(run)
                yield run, shard, ch

    _log("pass 2/2: featurizing (snapshot_at_t0) + packing")
    for ri, (run, shard, ch) in enumerate(_featurized(), 1):
        n = int(shard["meta"]["n_ticks"])
        info = totals[(run["chunk"], run["world"], run["arm"])]
        if n != info["n_ticks"]:
            raise DM.DampingError(f"{run['run_dir']}: tick count drift")
        if abs(float(shard["meta"]["t_origin"]) - run["t0_s"]) > 1e-6:
            raise DM.DampingError(
                f"{run['run_dir']}: t_origin != t0 (snapshot misaligned)")
        tstart = offsets["veh_step"]
        vmask = np.asarray(shard["veh_mask"])
        nz = np.flatnonzero(vmask.any(axis=1))
        snap_local = int(nz[0]) if nz.size else 0
        arrays = dict(shard)
        arrays.update(ch)
        dec_start = dec_off
        plan_arr = np.asarray(shard["plan"])
        targets["plan"][dec_off:dec_off + plan_arr.shape[0]] = plan_arr
        for name in PK.TRAIN_ARRAYS:
            if name == "plan":
                continue
            a = np.asarray(arrays[name])
            st = offsets[name]
            targets[name][st:st + a.shape[0]] = a
            offsets[name] = st + a.shape[0]
        sd = DM.SHARDS_ROOT / run["chunk"] / run["world"] / run["arm"]
        _write_identity_artifacts(sd, run, info, shard)
        runs_meta.append(dict(
            key=f"{run['chunk']}/{run['world']}__{run['arm']}",
            chunk=run["chunk"], batch=run["chunk"], world=run["world"],
            arm=run["arm"], split="TRAIN", t0_s=run["t0_s"], n_ticks=n,
            n_decision=C.N_DECISION, tick_start=tstart,
            tick_end=tstart + n, dec_start=dec_start,
            dec_end=dec_start + C.N_DECISION, snap_tick=tstart + snap_local,
            shard_dir=str(sd), input_sha256=run["input_sha256"],
            policy=run["policy"], outcomes=PK.OUTCOMES_NAME,
            outcome_first=0, n_outcome_rows=0, n_decision_rows=0))
        dec_off += C.N_DECISION
        if ri % 25 == 0 or ri == len(runs):
            _log(f"  packed {ri}/{len(runs)}")

    for name in PK.TRAIN_ARRAYS:
        targets[name].flush()
        del targets[name]
    import pandas as pd
    pd.DataFrame().to_parquet(out / PK.OUTCOMES_NAME, index=False)

    arrays_spec = {}
    for name in PK.TRAIN_ARRAYS:
        arrays_spec[name] = dict(file=f"{name}.npy", dtype=str(DTYPES[name]),
                                 shape=list(shapes[name]),
                                 n_bytes=int((out / f"{name}.npy").stat().st_size))
    items = [PK._train_item(r) for r in runs_meta]
    for it in items:
        it["damping_snap"] = True
    PK._write_manifest(out, arrays_spec, items, K, format="v3")
    files = {f"{n}.npy": D.sha256_file(out / f"{n}.npy")
             for n in PK.TRAIN_ARRAYS}
    files[PK.OUTCOMES_NAME] = D.sha256_file(out / PK.OUTCOMES_NAME)
    files[PK.MANIFEST_NAME] = D.sha256_file(out / PK.MANIFEST_NAME)
    pairs: dict = {}
    for r in runs_meta:
        pk = f"{r['world']}@t0={float(r['t0_s']):g}"
        pairs.setdefault(pk, []).append(r["key"])
    total_bytes = int(sum((out / rel).stat().st_size for rel in files))
    pack = dict(
        schema=PK.PACK_SCHEMA, train_schema=PK.TRAIN_SCHEMA,
        created_utc=datetime.now(timezone.utc).isoformat(),
        source_data_root=str(DM.CAMPAIGN_ROOT), manifest_file=PK.MANIFEST_NAME,
        outcomes_file=PK.OUTCOMES_NAME, n_runs=len(runs), n_items=len(items),
        n_ticks=int(n_ticks), n_decision=int(dec_off), k_max=K, format="v3",
        snapshot_at_t0=True,
        arrays={n: dict(arrays_spec[n]) for n in PK.TRAIN_ARRAYS},
        runs=runs_meta, pairs=pairs, files=files,
        totals=dict(n_runs=len(runs), n_items=len(items), n_ticks=int(n_ticks),
                    n_decision=int(dec_off), bytes=total_bytes))
    D.atomic_write_json(out / PK.INDEX_NAME, pack)
    PK._write_sums(out, pack)
    D.atomic_write_json(out / "DAMPING_RECEIPT.json", dict(
        schema="wm_n1_damping_pack/1", n_runs=len(runs), n_ticks=int(n_ticks),
        k_max=K, format="v3", snapshot_at_t0=True,
        base_pack=str(DM.BASE_PACK), shards_root=str(DM.SHARDS_ROOT),
        e_free_at_build_bytes=int(free),
        index_sha256=D.sha256_file(out / PK.INDEX_NAME),
        manifest_sha256=D.sha256_file(out / PK.MANIFEST_NAME)))
    _log(f"damping pack: {len(runs)} runs, {n_ticks} ticks, "
         f"{total_bytes/1e9:.2f} GB")
    return dict(root=str(out), n_runs=len(runs), n_items=len(items),
                n_ticks=int(n_ticks), bytes=total_bytes, format="v3", k_max=K,
                manifest_sha256=D.sha256_file(out / PK.MANIFEST_NAME),
                index_sha256=D.sha256_file(out / PK.INDEX_NAME))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dspeed.build_damping")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    print(json.dumps(build(workers=args.workers), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
