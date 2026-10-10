"""P9 corpus passthrough for the Idea A featurizer channels.

Reads and writes the channels defined in :mod:`features.idea_a` through the
existing packed-corpus layout, without editing any frozen ``corpus`` or
``features`` module.  Per shard (one run) the five arrays
``veh_kin``/``hdv_base``/``cav_plan_ch``/``event_tok``/``event_mask`` are
written next to the v1 arrays together with an ``IDEA_A_DONE.json`` receipt
(the frozen ``DONE.json``/``meta.json`` are never touched, so the v1
resume-by-sha verification stays valid).

Ownership note (INTERFACES_V2 file table): this module is the ONLY P9 file
in ``corpus/``; ``corpus/featurize_all.py``, ``corpus/cli.py``,
``corpus/pack.py`` and ``corpus/verify.py`` are the landed CORPUS piece and
are not edited here.  The single-command ``python -m corpus featurize-all``
integration is therefore a documented two-line patch for the INTEGRATOR
(see ``corpus/README.md`` section "Idea A integration"); until it lands,
``python -m corpus.idea_a backfill`` adds the Idea A channels to an
existing shard tree resumably.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from . import _util as U

IDEA_A_SCHEMA = "wm_n1_features_idea_a/1"
IDEA_A_DONE_NAME = "IDEA_A_DONE.json"
IDEA_A_ARRAYS = ("veh_kin", "hdv_base", "cav_plan_ch", "event_tok",
                 "event_mask")
IDEA_A_DTYPES = {"veh_kin": "float16", "hdv_base": "float16",
                 "cav_plan_ch": "float16", "event_tok": "float16",
                 "event_mask": "uint8"}


def _features_idea_a():
    U.ensure_wm_n1_on_path()
    import features.idea_a as IA

    return IA


def _shard_dir_for(data_root, key: str) -> Path:
    chunk, tail = str(key).split("/", 1)
    world, arm = tail.rsplit("__", 1)
    return Path(data_root) / chunk / world / arm


def _sha(path: Path) -> str:
    return U.sha256_file(path)


def _valid_done(shard_dir: Path, *, params_sha: str) -> dict | None:
    """Return the receipt when the written Idea A channels still verify."""
    p = Path(shard_dir) / IDEA_A_DONE_NAME
    if not p.is_file():
        return None
    try:
        done = U.read_json(p)
    except Exception:  # noqa: BLE001
        return None
    if done.get("schema") != IDEA_A_SCHEMA:
        return None
    if done.get("baseline_params_lf_sha256") != params_sha:
        return None
    arrays = done.get("arrays") or {}
    for name in IDEA_A_ARRAYS:
        spec = arrays.get(name)
        if spec is None:
            return None
        f = Path(shard_dir) / f"{name}.npy"
        if not f.is_file() or _sha(f) != spec.get("sha256"):
            return None
    return done


def write_idea_a_channels(shard, out_dir, *, force: bool = False,
                          fast: bool = False) -> dict:
    """Write the Idea A channels for one featurized shard.

    ``shard`` is the dict built by ``features.featurizer.build_shard``;
    ``out_dir`` is the shard directory (``<data_root>/<chunk>/<world>/<arm>``).
    Skips when a valid :data:`IDEA_A_DONE_NAME` receipt is present
    (resumable by sha); ``force`` rebuilds.  ``fast=True`` selects the
    byte-identical fast baseline (``features.idea_a`` ``fast`` flag; FEAT1
    profiling speed-up) -- the written arrays are identical either way and
    the receipt schema is unchanged.  The frozen v1 ``DONE.json`` and
    ``meta.json`` are never modified.  Returns a record dict with the
    per-array sha256/sizes, the exact array byte count, the relabel
    metadata and the written receipt path.
    """
    IA = _features_idea_a()
    meta = dict(shard.get("meta") or {})
    n_ticks = int(meta.get("n_ticks") or
                  np.asarray(shard["veh_mask"]).shape[0])
    shard_dir = Path(out_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    params = IA.load_nominal_params()
    params_sha = IA.NOMINAL_PARAMS_LF_SHA256
    if not force:
        done = _valid_done(shard_dir, params_sha=params_sha)
        if done is not None:
            return dict(status="skipped", record=done,
                        receipt=str(shard_dir / IDEA_A_DONE_NAME),
                        arrays_bytes=int(done.get("arrays_bytes", 0)),
                        total_bytes=int(done.get("total_bytes", 0)))

    built = IA.idea_a_channels_from_shard(shard, params=params, fast=fast)
    arrays = {name: np.ascontiguousarray(built[name])
              for name in IDEA_A_ARRAYS}
    specs = IA.IdeaAChannelSpecs()
    shapes = specs.array_shapes(n_ticks, int(shard["veh_mask"].shape[1]))
    relabel_info = dict(built.get("relabel") or {})
    is_wall_arm = bool(np.asarray(shard["wall_mask"]).any())

    out: dict = {}
    for name, arr in arrays.items():
        if tuple(arr.shape) != tuple(shapes[name]):
            raise ValueError(f"{name}: shape {arr.shape} != {shapes[name]}")
        if str(arr.dtype) != IDEA_A_DTYPES[name]:
            raise ValueError(f"{name}: dtype {arr.dtype} != "
                             f"{IDEA_A_DTYPES[name]}")
        mm = np.lib.format.open_memmap(shard_dir / f"{name}.npy", mode="w+",
                                       dtype=arr.dtype, shape=arr.shape)
        mm[...] = arr
        mm.flush()
        del mm
        out[name] = dict(sha256=_sha(shard_dir / f"{name}.npy"),
                         bytes=int((shard_dir / f"{name}.npy").stat().st_size),
                         dtype=str(arr.dtype), shape=list(arr.shape))

    arrays_bytes = int(sum(spec["bytes"] for spec in out.values()))
    done = dict(
        schema=IDEA_A_SCHEMA,
        chunk=meta.get("chunk"),
        batch=meta.get("batch") or meta.get("chunk"),
        world=meta.get("world"),
        arm=meta.get("arm"), split=meta.get("split"),
        policy=meta.get("policy"),
        t0_s=meta.get("t0_s"), n_ticks=n_ticks,
        n_slots=int(shard["veh_mask"].shape[1]),
        channels=list(IA.IDEA_A_CHANNELS),
        baseline_params_lf_sha256=params_sha,
        arrays=out, arrays_bytes=arrays_bytes,
        amendment2_road_end_relabel=bool(is_wall_arm),
        road_end_relabelled_n=(int(relabel_info.get("n_aborted_tokens", 0))
                               if is_wall_arm else 0),
        written_utc=U.utc_now())
    if is_wall_arm:
        done["road_end_relabel_note"] = (
            f"{IA.ROAD_END_REASON_SOURCE} -> {IA.ROAD_END_REASON_RELABELLED} "
            "on J4 wall branches (amendment 2); the frozen phase vocabulary "
            "is unchanged and run receipts keep their original labels")
    rec_path = U.write_json_atomic(shard_dir / IDEA_A_DONE_NAME, done)
    total = arrays_bytes + rec_path.stat().st_size
    return dict(status="written", record=done, receipt=str(rec_path),
                arrays_bytes=arrays_bytes, total_bytes=int(total))


def ensure_idea_a_channels(chunk: str, world: str, arm: str, run_dir,
                           out_dir, *, split: str, plan_row: dict,
                           force: bool = False, fast: bool = False,
                           base_history_run_dir=None,
                           outcomes_root=None, batch=None,
                           policy=None) -> dict:
    """Build (or rebuild) one shard's v1 arrays and add the Idea A channels.

    Skips straight to the passthrough when the shard already verifies.
    ``arm`` is the exact run-dir arm suffix; ``batch`` keys the shard root
    component and ``policy`` is the plan-level policy label (metadata).
    """
    shard_dir = Path(out_dir)
    IA = _features_idea_a()
    if not force and _valid_done(
            shard_dir, params_sha=IA.NOMINAL_PARAMS_LF_SHA256) is not None:
        p = U.read_json(shard_dir / IDEA_A_DONE_NAME)
        return dict(status="skipped", record=p,
                    receipt=str(shard_dir / IDEA_A_DONE_NAME),
                    arrays_bytes=int(p.get("arrays_bytes", 0)),
                    total_bytes=int(p.get("total_bytes", 0)))

    from features.featurizer import build_shard
    from .j4_wall_fix import install as install_j4_wall_fix

    install_j4_wall_fix()
    shard = build_shard(chunk, world, arm, run_dir, split=split,
                        plan_row=plan_row, outcomes_root=outcomes_root,
                        base_history_run_dir=base_history_run_dir,
                        batch=batch, policy=policy)
    return write_idea_a_channels(shard, shard_dir, force=True, fast=fast)


def road_end_relabel_enabled(meta: dict, *, default: bool = False) -> bool:
    """Read the amendment-2 J4 road-end relabel flag from ``meta.json``."""
    from features.idea_a import ROAD_END_RELABEL_FLAG

    return bool(meta.get(ROAD_END_RELABEL_FLAG, default))


def verify_idea_a(data_root, *, index_path=None) -> dict:
    """Re-verify every Idea A receipt in a shard tree (shas + params)."""
    root = Path(data_root)
    index = U.read_json(index_path or root / "INDEX.json")
    IA = _features_idea_a()
    params_sha = IA.NOMINAL_PARAMS_LF_SHA256
    errors: list = []
    runs = index.get("runs") or {}
    for key, rec in runs.items():
        shard_dir = Path(rec.get("shard_dir") or _shard_dir_for(root, key))
        done_p = shard_dir / IDEA_A_DONE_NAME
        if not done_p.is_file():
            errors.append(f"{key}: missing {IDEA_A_DONE_NAME}")
            continue
        try:
            done = U.read_json(done_p)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{key}: unreadable {IDEA_A_DONE_NAME} ({e})")
            continue
        if done.get("schema") != IDEA_A_SCHEMA:
            errors.append(f"{key}: schema {done.get('schema')!r}")
        if done.get("baseline_params_lf_sha256") != params_sha:
            errors.append(f"{key}: baseline params sha drift")
        arrays = done.get("arrays") or {}
        for name in IDEA_A_ARRAYS:
            spec = arrays.get(name)
            f = shard_dir / f"{name}.npy"
            if spec is None:
                errors.append(f"{key}: {name} not in receipt")
                continue
            if not f.is_file():
                errors.append(f"{key}: {name}.npy missing")
                continue
            if f.stat().st_size != int(spec.get("bytes", -1)):
                errors.append(f"{key}: {name}.npy size mismatch")
            elif _sha(f) != spec.get("sha256"):
                errors.append(f"{key}: {name}.npy sha256 mismatch (tampered?)")
    return dict(ok=not errors, n_runs=len(runs), errors=errors)


class IdeaAPassthrough:
    """Read the Idea A channels of a shard tree (or of a single shard).

    ``pack_dir`` is either a featurized data root holding ``INDEX.json``
    (runs addressed by index ``i`` in sorted run-key order) or one shard
    directory (then only ``i == 0`` is valid).
    """

    def __init__(self, pack_dir, *, split: str = "TRAIN") -> None:
        self.pack_dir = Path(pack_dir)
        self.split = split
        self._runs: list = []
        index_p = self.pack_dir / "INDEX.json"
        if index_p.is_file():
            index = U.read_json(index_p)
            runs = index.get("runs") or {}
            self._runs = [dict(rec, key=key)
                          for key, rec in sorted(runs.items())
                          if str(rec.get("split") or "") == split
                          or split is None]
        else:
            done = self.pack_dir / IDEA_A_DONE_NAME
            if done.is_file():
                self._runs = [dict(
                    shard_dir=str(self.pack_dir),
                    key=f"{U.read_json(done).get('chunk')}/"
                        f"{U.read_json(done).get('world')}__"
                        f"{U.read_json(done).get('arm')}")]

    def __len__(self) -> int:
        return len(self._runs)

    def _dir(self, i: int) -> Path:
        if not 0 <= i < len(self._runs):
            raise IndexError(f"run index {i} out of range "
                             f"(n_runs={len(self._runs)})")
        rec = self._runs[i]
        if rec.get("shard_dir"):
            return Path(rec["shard_dir"])
        return _shard_dir_for(self.pack_dir, rec["key"])

    def has_idea_a_channels(self) -> bool:
        """True if every selected run carries verified P9 channels."""
        if not self._runs:
            return False
        IA = _features_idea_a()
        for i in range(len(self._runs)):
            if _valid_done(self._dir(i),
                           params_sha=IA.NOMINAL_PARAMS_LF_SHA256) is None:
                return False
        return True

    def get(self, i: int, channel: str, *, dtype=None):
        """Read one Idea A channel for run ``i`` (memmap; ``dtype`` casts)."""
        if channel not in IDEA_A_ARRAYS:
            raise KeyError(f"unknown Idea A channel {channel!r}; expected "
                           f"one of {IDEA_A_ARRAYS}")
        arr = np.load(self._dir(i) / f"{channel}.npy", mmap_mode="r")
        if dtype is not None and np.dtype(dtype) != arr.dtype:
            arr = np.asarray(arr, dtype=np.dtype(dtype))
        return arr


def backfill(data_root, *, workers: int = 1, force: bool = False,
             inventory_parquet=None, raw_data_root=None,
             fast: bool = False) -> dict:
    """Add the Idea A channels to every shard of a featurized data root.

    Uses ``corpus.featurize_all.select_runs`` + ``_resolve_run`` (the same
    split-before-open, sealed-world-refusing path as the v1 driver) to
    re-derive each job, then ``ensure_idea_a_channels`` per shard
    (resumable; the v1 arrays are rebuilt in memory only).  With the
    documented featurize-all patch in place this command is a no-op.

    ``raw_data_root`` is the ``--data-root`` the v1 featurize-all ran with
    (the raw runs root the inventory ``root_rel`` values anchor to; default
    the workspace root ``E:/research-dense``).
    """
    from . import featurize_all as FA

    U.lower_priority_below_normal()
    data_root = Path(data_root).resolve()
    out_root = data_root
    if raw_data_root is None:
        from features import splits as SPLIT

        raw_data_root = SPLIT.RD
    raw_data_root = Path(raw_data_root)
    inventory_parquet = inventory_parquet or (
        U.WM_N1_ROOT / "inventory" / "inventory.parquet")
    jobs = []
    n_v1 = 0
    for run in FA.select_runs(inventory_parquet):
        # shard key = batch / world__<run-dir arm suffix> (matches the
        # featurize-all writer); the plan-level arm is only metadata
        arm = U.arm_suffix(run["run"], run.get("arm"))
        key = U.run_key(run["batch"], run["world"], arm)
        shard_dir = _shard_dir_for(out_root, key)
        if not (shard_dir / "DONE.json").is_file():
            continue                      # v1 shard not featurized yet
        n_v1 += 1
        try:
            job = FA._resolve_run(run, raw_data_root, out_root, force=force)
        except FA.FeaturizeRefusal as e:  # pragma: no cover
            print(f"  [refused] {e}")
            continue
        except Exception as e:  # noqa: BLE001
            print(f"  [no-plan] {run['world']}: {e}")
            continue
        jobs.append((job, shard_dir))

    t0 = time.time()
    results = []
    for job, shard_dir in jobs:
        rec = ensure_idea_a_channels(
            job["chunk"], job["world"], job["arm"], job["run_dir"],
            shard_dir, split=job["split"], plan_row=job["plan_row"],
            force=force, fast=fast,
            base_history_run_dir=job.get("base_history_run_dir"),
            outcomes_root=job.get("outcomes_root"),
            batch=job.get("batch"), policy=job.get("policy"))
        results.append((job, rec))
        key = U.run_key(job["batch"], job["world"], job["arm"])
        print(f"  [{rec['status']:7s}] {key} arrays_bytes="
              f"{rec['arrays_bytes']}")
    secs = time.time() - t0
    n_written = sum(1 for _, r in results if r["status"] == "written")
    n_skipped = sum(1 for _, r in results if r["status"] == "skipped")
    print(f"idea-a backfill: v1_shards={n_v1} runs={len(results)} "
          f"written={n_written} skipped={n_skipped} workers={workers} "
          f"wall={secs:.1f}s")
    return dict(runs=len(results), written=n_written, skipped=n_skipped,
                wall_s=secs)


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="corpus.idea_a",
        description="P9 passthrough: add/verify the Idea A channels on a "
                    "featurized shard tree (resumable, BELOW_NORMAL, 1 "
                    "worker default)")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backfill")
    b.add_argument("--data-root", required=True)
    b.add_argument("--raw-data-root", dest="raw_data_root", default=None)
    b.add_argument("--inventory", default=None)
    b.add_argument("--workers", type=int, default=1)
    b.add_argument("--force", action="store_true")
    b.add_argument("--fast", action="store_true",
                   help="byte-identical fast baseline (FEAT1 profiling "
                        "speed-up); written arrays are identical")
    v = sub.add_parser("verify")
    v.add_argument("--data-root", required=True)
    v.add_argument("--allow-d-output-root", dest="allow_d_output_root",
                   default=None,
                   help="owner-approved override: permit D: reads ONLY "
                        "under this exact root (read-only verify)")
    args = p.parse_args(argv)
    if getattr(args, "allow_d_output_root", None):
        U.activate_d_output_override(args.allow_d_output_root)
    if args.cmd == "backfill":
        backfill(args.data_root, workers=args.workers, force=args.force,
                 inventory_parquet=args.inventory,
                 raw_data_root=args.raw_data_root, fast=args.fast)
        return 0
    rep = verify_idea_a(args.data_root)
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
