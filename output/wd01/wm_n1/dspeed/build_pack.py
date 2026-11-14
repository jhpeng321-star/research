"""A9 steps 2-3: featurize the trajectory runs straight into one pack.

The frozen ``corpus.pack`` writer concatenates a *shard tree* into a pack,
which needs a second full copy of the payload on disk.  E: free space cannot
hold shards (~13 GB) + pack (~13 GB) at once and the brief's 8 GB free-space
floor is binding, so this module performs the SAME featurization + packing in
a single streaming pass: each run is featurized in memory
(``features.featurizer.build_shard`` with an explicit gzip physics path,
K_max 192 / v3) and its arrays are copied directly into the preallocated pack
memmaps.  The emitted ``manifest.json`` / ``INDEX.json`` / ``outcomes.parquet``
are byte-contract-compatible with :mod:`corpus.pack` (schema
``wm_n1_pack/1`` / ``wm_n1_corpus_pack/2``, array set = ``corpus.pack.
TRAIN_ARRAYS``, v3 format).

Small per-run alignment receipts (``meta.json``, ``slot_vid.json``,
``ALIGN.json``) are written under ``_out/shards_dspeed/`` so the overlay
builder can assert the ``t_origin == t0`` branch convention and the
``t0 + 0.5 s`` first physics row without the heavy shard arrays.

Admitted runs are exactly those with a verified label done marker
(:mod:`dspeed.labels`); re-running after the laptop shard finishes adds the
remaining runs (the pack root is rewritten atomically from the full admitted
set -- this is a derived artifact, not frozen evidence).

Usage::

    python -m dspeed.build_pack --workers 6
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
from . import labels as L  # noqa: E402

HERE = Path(__file__).resolve().parent
_WM_N1 = HERE.parent
if str(_WM_N1) not in sys.path:
    sys.path.insert(0, str(_WM_N1))
sys.path.insert(0, str(_WM_N1.parent / "output"))

from corpus import pack as PK  # noqa: E402
from features import constants as C  # noqa: E402
from features import featurizer as FZ  # noqa: E402
from features.idea_a import (CAV_PLAN_CH_F, EVENT_MAX,  # noqa: E402
                             EVENT_TOK_F, HDV_BASE_F, KIN_F,
                             idea_a_channels_from_shard)

K = 192
TAILS = {
    "veh_step": (K, 16), "veh_mask": (K,), "plan": (K, 19, 8),
    "wall": (4, 11), "wall_mask": (4,), "counter": (2, 30),
    "accel_next": (K,), "target_mask": (K,), "lane_next": (K,),
    "arrivals": (2,), "veh_kin": (K, KIN_F), "hdv_base": (K, HDV_BASE_F),
    "cav_plan_ch": (K, CAV_PLAN_CH_F), "event_tok": (EVENT_MAX, EVENT_TOK_F),
    "event_mask": (EVENT_MAX,),
}
DTYPES = {
    "veh_step": np.float16, "veh_mask": np.uint8, "plan": np.float16,
    "wall": np.float16, "wall_mask": np.uint8, "counter": np.float16,
    "accel_next": np.float16, "target_mask": np.uint8, "lane_next": np.int8,
    "arrivals": np.int16, "veh_kin": np.float16, "hdv_base": np.float16,
    "cav_plan_ch": np.float16, "event_tok": np.float16, "event_mask": np.uint8,
}
_T0W = time.time()
SNAP_OFFSET_S = 0.5


def _bytes_for_run(n_ticks: int) -> int:
    """Exact pack bytes of one run at K=192 (manifest array set)."""
    total = 0
    for name in PK.TRAIN_ARRAYS:
        head = C.N_DECISION if name == "plan" else int(n_ticks)
        total += int(np.prod((head, *TAILS[name]))) * np.dtype(
            DTYPES[name]).itemsize
    return total


def _log(msg):
    print(f"[{time.time() - _T0W:7.1f}s] {msg}", flush=True)


def _scan(physics: Path, t0: float) -> dict:
    """Count rows and validate the branch alignment (first row t0+0.5)."""
    import gzip
    opener = gzip.open if str(physics).endswith(".gz") else open
    n = 0
    first = last = None
    with opener(physics, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            t = float(json.loads(line)["time_s"])
            if first is None:
                first = t
            last = t
            n += 1
    if first is None:
        raise D.DspeedError(f"{physics}: no rows")
    if abs(first - (t0 + SNAP_OFFSET_S)) > 1e-6:
        raise D.DspeedError(
            f"{physics}: first physics row {first} != t0+0.5 ({t0 + 0.5}); "
            "misaligned, refusing")
    n_ticks = int(round((last - t0) / FZ.C.DT_S)) + 1
    # the branch grid carries k=0 (t0) with no row, so rows = n_ticks - 1;
    # a gap would show as rows < round((last-first)/dt)+1
    if n != int(round((last - first) / FZ.C.DT_S)) + 1:
        raise D.DspeedError(
            f"{physics}: {n} rows over [{first}, {last}] are non-contiguous; "
            "refusing")
    return dict(n_ticks=n_ticks, first=first, last=last)


def _scan_task(run):
    return run["world"], run["arm"], _scan(run["physics"], run["t0_s"])


def _featurize_run(run):
    shard = FZ.build_shard(
        run["chunk"], run["world"], run["arm"], run["run_dir"],
        split="TRAIN", plan_row=dict(t0_s=run["t0_s"]),
        outcomes_root=None, base_history_run_dir=run["base_dir"],
        batch=run["chunk"], policy="trajectory", format="v3",
        physics_path=run["physics"])
    channels = idea_a_channels_from_shard(shard, fast=True)
    return shard, channels


def build(workers: int = 6, limit_worlds: set | None = None) -> dict:
    runs = D.enumerate_runs()
    if limit_worlds:
        runs = [r for r in runs if r["world"] in limit_worlds]
    admitted = []
    for r in runs:
        if L.verify_done(D.LABELS_ROOT, r["chunk"], r["world"], r["arm"]):
            admitted.append(r)
    if not admitted:
        raise D.DspeedError("no admitted runs (labels first: dspeed.labels)")
    _log(f"admitted {len(admitted)}/{len(runs)} runs (verified label markers)")

    # derived artifact: remove the previous pack BEFORE measuring free space
    # so the disk-budget cap reflects the space this build will actually use.
    out = D.PACK_ROOT
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    _log("pass 1/2: scanning physics for tick totals + alignment")
    totals: dict = {}
    with Pool(workers) as pool:
        for world, arm, info in pool.imap_unordered(_scan_task, admitted, 1):
            totals[(world, arm)] = info
    # disk-budget cap: keep the mandated >=8 GiB free (owner constraint).
    free = shutil.disk_usage(str(D.OUT)).free
    budget = free - int(8 * (1 << 30)) - int(0.3 * (1 << 30))
    kept, used = [], 0
    for run in admitted:
        b = _bytes_for_run(totals[(run["world"], run["arm"])]["n_ticks"])
        if used + b > budget:
            break
        kept.append(run)
        used += b
    n_dropped_disk = len(admitted) - len(kept)
    if not kept:
        raise D.DspeedError(
            f"E: free {free/1e9:.2f} GB cannot hold even one run under the "
            "8 GB floor; stop")
    admitted = kept
    n_ticks = sum(totals[(r["world"], r["arm"])]["n_ticks"] for r in admitted)
    n_runs = len(admitted)
    _log(f"total ticks={n_ticks} runs={n_runs} "
         f"(disk-dropped {n_dropped_disk}; projected {used/1e9:.2f} GB, "
         f"free {free/1e9:.2f} GB)")

    shapes = {}
    for name in PK.TRAIN_ARRAYS:
        head = n_runs * C.N_DECISION if name == "plan" else n_ticks
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
                        admitted, pool.imap(_featurize_run, admitted, 1)):
                    yield run, shard, ch
        else:
            for run in admitted:
                shard, ch = _featurize_run(run)
                yield run, shard, ch

    _log("pass 2/2: featurizing + packing")
    for ri, (run, shard, ch) in enumerate(_featurized(), 1):
        n = int(shard["meta"]["n_ticks"])
        if n != totals[(run["world"], run["arm"])]["n_ticks"]:
            raise D.DspeedError(f"{run['run_dir']}: tick count drift")
        tstart = offsets["veh_step"]
        vmask = np.asarray(shard["veh_mask"])
        nz = np.flatnonzero(vmask.any(axis=1))
        snap_local = int(nz[0]) if nz.size else 0   # t0+0.5 (branch grid)
        arrays = dict(shard)
        arrays.update(ch)
        for name in PK.TRAIN_ARRAYS:
            if name == "plan":
                a = np.asarray(shard["plan"])
                targets[name][dec_off:dec_off + a.shape[0]] = a
            else:
                a = np.asarray(arrays[name])
                st = offsets[name]
                targets[name][st:st + a.shape[0]] = a
                if name != "plan":
                    offsets[name] = st + a.shape[0]
        # per-run alignment receipt (small; heavy arrays stay in the pack)
        sd = D.SHARDS_ROOT / run["chunk"] / run["world"] / run["arm"]
        sd.mkdir(parents=True, exist_ok=True)
        meta = dict(shard["meta"])
        meta["shard_dir"] = str(sd)
        D.atomic_write_json(sd / "meta.json", meta)
        (sd / "slot_vid.json").write_text(
            json.dumps(list(shard["slot_vid"])), encoding="utf-8")
        D.atomic_write_json(sd / "ALIGN.json", dict(
            world=run["world"], arm=run["arm"], chunk=run["chunk"],
            batch=run["chunk"], split="TRAIN", t0_s=run["t0_s"],
            t_origin=float(meta["t_origin"]),
            first_physics_time=totals[(run["world"], run["arm"])]["first"],
            snapshot_offset_s=SNAP_OFFSET_S, n_ticks=n,
            input_physics=str(run["physics"]),
            physics_gz_sha256=D.sha256_file(run["physics"])))
        runs_meta.append(dict(
            key=f"{run['chunk']}/{run['world']}__{run['arm']}",
            chunk=run["chunk"], batch=run["chunk"], world=run["world"],
            arm=run["arm"], split="TRAIN", t0_s=run["t0_s"], n_ticks=n,
            n_decision=C.N_DECISION, tick_start=tstart,
            tick_end=tstart + n, dec_start=dec_off,
            dec_end=dec_off + C.N_DECISION, snap_tick=tstart + snap_local,
            shard_dir=str(sd), input_sha256=D.sha256_file(run["physics"]),
            policy="trajectory", outcomes=PK.OUTCOMES_NAME,
            outcome_first=0, n_outcome_rows=0, n_decision_rows=0))
        dec_off += C.N_DECISION
        if ri % 50 == 0 or ri == n_runs:
            _log(f"  packed {ri}/{n_runs}")

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
        source_data_root=str(D.OUT), manifest_file=PK.MANIFEST_NAME,
        outcomes_file=PK.OUTCOMES_NAME, n_runs=n_runs, n_items=len(items),
        n_ticks=int(n_ticks), n_decision=int(dec_off), k_max=K,
        format="v3",
        arrays={n: dict(arrays_spec[n]) for n in PK.TRAIN_ARRAYS},
        runs=runs_meta, pairs=pairs, files=files,
        totals=dict(n_runs=n_runs, n_items=len(items), n_ticks=int(n_ticks),
                    n_decision=int(dec_off), bytes=total_bytes))
    D.atomic_write_json(out / PK.INDEX_NAME, pack)
    PK._write_sums(out, pack)
    D.atomic_write_json(out / "DSpeed_RECEIPT.json", dict(
        schema="wm_n1_dspeed_pack/1", n_runs=n_runs, n_ticks=int(n_ticks),
        n_dropped_disk=int(n_dropped_disk), e_free_at_build_bytes=int(free),
        k_max=K, format="v3", shards_root=str(D.SHARDS_ROOT),
        labels_manifest_sha256=D.sha256_file(D.LABELS_ROOT / "MANIFEST.json"),
        index_sha256=D.sha256_file(out / PK.INDEX_NAME),
        manifest_sha256=D.sha256_file(out / PK.MANIFEST_NAME),
        sha256sums_sha256=(D.sha256_file(out / PK.SUMS_NAME)
                           if (out / PK.SUMS_NAME).is_file() else None)))
    _log(f"pack: {n_runs} runs, {n_ticks} ticks, {total_bytes/1e9:.2f} GB")
    return dict(root=str(out), n_runs=n_runs, n_ticks=int(n_ticks),
                bytes=total_bytes, format="v3", k_max=K,
                manifest_sha256=D.sha256_file(out / PK.MANIFEST_NAME),
                index_sha256=D.sha256_file(out / PK.INDEX_NAME))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dspeed.build_pack")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit-worlds", default="")
    args = ap.parse_args(argv)
    limit = {w for w in args.limit_worlds.split(",") if w}
    rep = build(workers=args.workers, limit_worlds=limit or None)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
