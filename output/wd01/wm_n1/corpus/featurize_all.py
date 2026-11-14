"""Featurize-all driver: inventory -> per-run shards.

Selects the readable TRAIN/CALIBRATION runs from ``inventory/inventory.parquet``
and calls the frozen ``features.featurizer.build_shard`` +
``features.writer.write_shard`` for each.  The split of every world is resolved
from its chunk ``WORLD_PLAN.json`` BEFORE any run file is opened; HOLDOUT,
excluded and chunk-5 worlds are refused.  Never opens ``dispatcher_rows.jsonl``
or ``insertions.jsonl``.

The driver enforces the owner constraints:

* ``workers`` default 4, hard maximum 6 — exceeded ONLY by the explicit
  owner-approved ``--max-workers-override N`` (recorded in
  ``ALLOCATION.json``), e.g. the FEAT1 v2 ``--workers 12
  --max-workers-override 12`` speed-up;
* a dynamic worker gate: a run above ``MEM.DYNAMIC_FALLBACK_WORKERS``
  drops the current batch to 8 workers while free virtual memory is
  < 8 GB or free RAM is < 6 GB (logged + recorded in the receipt);
* Windows BELOW_NORMAL priority class;
* a free-virtual-memory (>= 10 GB by default) and free-RAM gate before start
  and between worker batches (pause, then resume when recovered);
* a written ``ALLOCATION.json`` receipt (at start, then updated);
* a written ``opened_paths.json`` audit that must be clean.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

from . import _util as U
from . import memory as MEM

RUN_COLUMNS = (
    "batch", "chunk", "world", "split", "root_rel", "t0_s", "arm", "policy",
    "policy_source", "is_fleet", "schedule_arm", "duration_s", "present",
    "receipt_status", "is_complete", "physics_rows_written",
    "physics_file_bytes", "saved_state_id", "is_smoke", "mode", "run",
)

KEEP_SPLITS = ("TRAIN", "CALIBRATION")
CHUNK5_SEED_LO = 8707
CHUNK5_SEED_HI = 8736
PLAN_MAP_NAME = "PLAN_MAP.json"
ALLOCATION_NAME = "ALLOCATION.json"
OPENED_PATHS_NAME = "opened_paths.json"

# J4 walls-pilot branch arms branched from the stage-1 chunk-4 base runs
# (the plan's ``bases_root``), so their pre-restore counter history lives
# there -- not inside the walls-pilot root (FEAT1 blocker 3).
BASES_ROOT_REL = {
    "walls_pilot_chunk4": "runs/campaign_stage1_chunk4",
}

_BATCH_CHUNK_LABEL = {
    "chunk1": "chunk1",
    "chunk2": "chunk2",
    "chunk3": "chunk3",
    "chunk4": "chunk4",
    "chunk1_fix27": "chunk1_fix27",
    "truearms_chunk2": "chunk2",
    "truearms_chunk3": "chunk3",
    "walls_pilot_chunk4": "chunk4",
}

SEAL_MISSING = object()

PHYSICS_NAME = "physics_steps.jsonl"


class FeaturizeRefusal(PermissionError):
    """A sealed/excluded/HOLDOUT world reached the featurize-all driver."""


def load_source_override(map_path) -> dict:
    """Load a ``--source-override MAP.json`` (run key -> physics path).

    FEAT3 stage 1: the 73 chunk1 p40 branch arms whose raw runs live in the
    owner-approved lossless archive read their physics from the verified
    staging extract (``raw_chunk1_p40``) instead of the runs tree, which is
    never restored.  The map value must be an existing
    ``physics_steps.jsonl``; the effective run dir is its parent's parent
    (the arm root the featurizer opens).  Returns ``{run_key: run_dir}``.
    """
    if map_path is None:
        return {}
    raw = U.read_json(map_path)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            f"source-override {map_path}: expected a non-empty JSON object "
            f"mapping run key -> physics_steps.jsonl path")
    dirs: dict = {}
    for key, phys in raw.items():
        chunk, world, arm = U.split_run_key(str(key))
        p = Path(str(phys))
        if p.name != PHYSICS_NAME or not p.is_file():
            raise ValueError(
                f"source-override {key}: {p} is not an existing "
                f"{PHYSICS_NAME}")
        dirs[U.run_key(chunk, world, arm)] = str(p.parent.parent)
    return dirs


def _inventory_specs() -> dict:
    path = U.WM_N1_ROOT / "inventory" / "inventory_lib.py"
    if not path.is_file():
        return {}
    spec = importlib.util.spec_from_file_location("corpus_inv_lib", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {s["name"]: dict(s) for s in mod.PLAN_SPECS}


def chunk_label(batch: str) -> str:
    return _BATCH_CHUNK_LABEL.get(str(batch), str(batch))


def world_seed(world: str):
    m = re.match(r"^[a-zA-Z]*(\d+)", str(world))
    return int(m.group(1)) if m else None


def _load_chunk5_seal():
    try:
        import wdpaths

        rd = Path(wdpaths.RESEARCH_DENSE_ROOT)
    except Exception:  # noqa: BLE001
        rd = Path("E:/research-dense")
    for cand in (rd / "protocols" / "chunk5_test_seal.json",
                 U.WORKSPACE_ROOT / "protocols" / "chunk5_test_seal.json"):
        if cand.is_file():
            try:
                return json.loads(cand.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                return {"worlds": [], "seeds": []}
    return {"worlds": [], "seeds": []}


def sealed_reason(chunk: str, world: str, seal=SEAL_MISSING):
    """Non-empty reason string when (chunk, world) is a sealed TEST world."""
    if seal is SEAL_MISSING:
        seal = _load_chunk5_seal()
    label = chunk_label(chunk)
    if label in ("chunk5", "chunk_5", "chunk-5"):
        return f"chunk {label!r} is the sealed chunk-5 TEST split"
    seed = world_seed(world)
    if seed is not None and CHUNK5_SEED_LO <= seed <= CHUNK5_SEED_HI:
        return (f"world {world!r} (seed {seed}) is in the sealed chunk-5 block "
                f"{CHUNK5_SEED_LO}-{CHUNK5_SEED_HI}")
    sealed_worlds = {str(w) for w in (seal.get("worlds")
                                      or seal.get("sealed_worlds") or [])}
    if str(world) in sealed_worlds:
        return f"world {world!r} is listed in chunk5_test_seal.json"
    for key in ("seeds", "seed_block", "seed_range"):
        val = seal.get(key)
        if isinstance(val, (list, tuple)) and len(val) == 2 \
                and all(isinstance(x, int) for x in val):
            if val[0] <= seed <= val[1]:
                return f"world {world!r} (seed {seed}) sealed by {key}"
    return ""


def select_runs(inventory_parquet, *, data_root=None,
                splits=("TRAIN", "CALIBRATION")) -> list:
    """Return the present/complete, non-smoke runs of the requested splits.

    HOLDOUT worlds never appear in the readable inventory rows; they are also
    dropped by name.  Returns a list of plain run dicts (``RUN_COLUMNS``),
    sorted by ``(chunk, world, run)``.
    """
    import pandas as pd

    splits = tuple(str(s) for s in splits)
    df = pd.read_parquet(inventory_parquet)
    for col in ("split", "present", "is_complete", "is_smoke"):
        if col not in df.columns:
            raise ValueError(f"inventory is missing required column {col!r}")
    keep = df[df["split"].isin(splits)
              & df["present"].astype(bool)
              & df["is_complete"].astype(bool)
              & (~df["is_smoke"].astype(bool))]
    keep = keep[~keep["world"].astype(str).str.upper().str.startswith("HOLDOUT")]
    keep = keep.sort_values(["chunk", "world", "run"]).reset_index(drop=True)
    cols = [c for c in RUN_COLUMNS if c in keep.columns]
    return U.records_from_dataframe(keep[cols])


def _plan_path_for(batch: str, data_root: Path):
    specs = _inventory_specs()
    spec = specs.get(batch)
    if spec is None or not spec.get("plan_rel"):
        return None
    return Path(data_root) / spec["plan_rel"]


def _resolve_run(run: dict, data_root: Path, out_root: Path, *, force: bool,
                 exclusions=None, source_override=None):
    from features import splits as S

    batch = str(run["batch"])
    chunk = chunk_label(batch)
    world = str(run["world"])
    reason = sealed_reason(chunk, world)
    if reason:
        raise FeaturizeRefusal(f"featurize-all: {reason}")
    plan_path = _plan_path_for(batch, data_root)
    if plan_path is None or not Path(plan_path).is_file():
        plan_path = S.CHUNKS.get(chunk, {}).get("plan")
    try:
        plan = S.load_plan(chunk, plan_path)
        row = next(w for w in plan["worlds"] if w["world"] == world)
        split = S.decide_split(chunk, world, plan_path=plan_path,
                               exclusions=exclusions)
    except S.SplitRefusal:
        raise
    arm = U.arm_suffix(run["run"], run.get("arm"))
    root_rel = str(run["root_rel"])
    run_dir = data_root / root_rel / str(run["run"])
    if source_override:
        key = U.run_key(batch, world, arm)
        run_dir = Path(source_override.get(key, run_dir))
    base_hist = None
    if arm not in ("base", "native_base"):
        base_hist = str(_base_history_dir(run, row, data_root, root_rel))
    return dict(
        chunk=chunk, world=world, arm=arm, split=split, plan_row=row,
        run_dir=str(run_dir), data_root=str(out_root),
        outcomes_root=None, force=bool(force),
        base_history_run_dir=base_hist, plan_path=str(plan_path),
        root_rel=root_rel, run=str(run["run"]), batch=batch,
        # the plan-level policy label is metadata only; the shard key is the
        # exact run-dir arm suffix (world + '__' + arm, e.g. .../fixed1)
        policy=str(run.get("policy") or run.get("arm") or arm),
    )


def _base_history_dir(run: dict, plan_row: dict, data_root, root_rel: str):
    """Runs root holding a branch arm's shared pre-restore base run.

    Batch mapping first (J4 walls pilot -> ``runs/campaign_stage1_chunk4``),
    then the plan row's ``bases_root`` when it anchors under ``data_root``;
    falls back to the batch's own runs root.
    """
    batch = str(run["batch"])
    world = str(run["world"])
    if batch in BASES_ROOT_REL:
        return Path(data_root) / BASES_ROOT_REL[batch] / f"{world}__base"
    bases_root = str(plan_row.get("bases_root") or "")
    if bases_root:
        try:
            rel = Path(bases_root).resolve().relative_to(
                Path(data_root).resolve())
            return Path(data_root) / rel / f"{world}__base"
        except ValueError:                    # anchors elsewhere: ignore
            pass
    return Path(data_root) / root_rel / f"{world}__base"


def _run_one(job: dict) -> dict:
    U.ensure_wm_n1_on_path()
    U.apply_env_d_policy()
    from .j4_wall_fix import install as install_j4_wall_fix

    install_j4_wall_fix()
    from features import writer as W
    from features.featurizer import build_shard

    shard = build_shard(job["chunk"], job["world"], job["arm"], job["run_dir"],
                        split=job["split"], plan_row=job["plan_row"],
                        outcomes_root=job.get("outcomes_root"),
                        base_history_run_dir=job.get("base_history_run_dir"),
                        batch=job.get("batch"), policy=job.get("policy"),
                        format=job.get("format"))
    rec = W.write_shard(shard, job["data_root"], force=job["force"])
    from .idea_a import write_idea_a_channels
    write_idea_a_channels(shard, W.shard_dir(job["data_root"],
                           job["batch"], job["world"], job["arm"]),
                          fast=bool(job.get("idea_fast")))
    return dict(key=U.run_key(job["batch"], job["world"], job["arm"]),
                status=rec["status"], record=rec["record"])


def _key(job: dict) -> str:
    return U.run_key(job["batch"], job["world"], job["arm"])


def _run_batch(batch: list, workers: int, failures: list) -> list:
    from concurrent.futures.process import BrokenProcessPool

    # per-run start lines (flushed) so a crashing run is identifiable in the
    # log: a run with a [start] but no [end] line was in flight at the crash
    for j in batch:
        print(f"  [start] {_key(j)}", flush=True)
    if workers <= 1:
        out = []
        for j in batch:
            try:
                r = _run_one(j)
            except BrokenProcessPool:
                raise
            except Exception as exc:  # noqa: BLE001
                print(f"  [error] {_key(j)} {type(exc).__name__}: {exc}",
                      flush=True)
                failures.append(dict(key=_key(j), error=type(exc).__name__,
                                     detail=str(exc)[:500]))
                continue
            print(f"  [end] {r['key']} status={r['status']}", flush=True)
            out.append(r)
        return out
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_run_one, j): j for j in batch}
        out = []
        for fut in as_completed(futures):
            j = futures[fut]
            try:
                r = fut.result()
            except BrokenProcessPool:
                # the pool itself is dead: every remaining future in this
                # batch would fail the same way, so abort the whole driver
                print(f"  [error] {_key(j)} BrokenProcessPool "
                      f"(worker terminated abruptly; aborting)", flush=True)
                raise
            except Exception as exc:  # noqa: BLE001
                # a per-run failure (recorded as an explicit exclusion; the
                # rest of the corpus continues)
                exc_type = type(exc).__name__
                print(f"  [error] {_key(j)} {exc_type}: {exc}", flush=True)
                failures.append(dict(key=_key(j), error=exc_type,
                                     detail=str(exc)[:500]))
                continue
            print(f"  [end] {r['key']} status={r['status']}", flush=True)
            out.append(r)
        return out


def _chunks(seq, size):
    size = max(1, int(size))
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def featurize_all(inventory_parquet, data_root, out_root, *, workers: int = 4,
                  force: bool = False, min_free_virtual_gb=None,
                  min_free_ram_gb=None, pause_s=MEM.DEFAULT_PAUSE_S,
                  max_workers_override=None, probe=None,
                  idea_fast: bool = False, format: str | None = None,
                  source_override=None) -> dict:
    """Featurize every selected run into ``<out_root>/<chunk>/<world>/<arm>``.

    ``idea_fast=True`` selects the byte-identical fast Idea A baseline for
    every run (``features.idea_a`` ``fast`` flag; FEAT1 profiling speed-up,
    recorded in ALLOCATION.json).  ``format`` selects the slot-width format
    (v2 = 128 default, byte-reproducible; v3 = 192, amendment 5); the resolved
    ``format``/``k_max`` are recorded in ALLOCATION.json.  ``source_override``
    is an optional MAP.json (run key -> physics_steps.jsonl path) redirecting
    selected runs' raw source (FEAT3 stage 1 archive restore); the count and
    path are recorded in ALLOCATION.json.  Resumable by shard sha
    (``features.writer``); returns ``{"runs", "written", "skipped", "index",
    "receipt", "refused"}``.
    """
    U.ensure_wm_n1_on_path()
    from features import constants as FC

    fmt, k_max = FC.resolve_k_max(format=format)
    workers = int(workers)
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    override = (None if max_workers_override is None
                else int(max_workers_override))
    if workers > U.MAX_WORKERS and override != workers:
        raise ValueError(
            f"workers={workers} exceeds the hard maximum of {U.MAX_WORKERS}; "
            f"pass the explicit owner-approved override "
            f"--max-workers-override {workers}")

    U.refuse_unapproved_d_out(out_root)

    # The parent must resolve every module a worker result or exception
    # references: results are pickled by module reference, and an
    # unpicklable exception result (wall_tokens_v04.WallTokenError from a
    # contract-failing wall run) killed the whole pool twice
    # (2026-10-01 ~09:28 and ~11:01 local, BrokenProcessPool after a
    # ModuleNotFoundError inside the executor result reader).
    U.ensure_wm_n1_on_path()
    import features.actions_adapter  # noqa: F401  (puts partB on sys.path)

    data_root = Path(data_root).resolve()
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    runs = select_runs(inventory_parquet)
    override_dirs = load_source_override(source_override)
    jobs = [_resolve_run(r, data_root, out_root, force=force,
                         source_override=override_dirs) for r in runs]
    if override_dirs:
        job_keys = {U.run_key(j["batch"], j["world"], j["arm"])
                    for j in jobs}
        unmatched = sorted(set(override_dirs) - job_keys)
        if unmatched:
            raise ValueError(
                f"source-override: {len(unmatched)} keys match no selected "
                f"run (wrong chunk/world/arm, or split-excluded): "
                f"{unmatched[:5]}")
    for job in jobs:
        job["idea_fast"] = bool(idea_fast)
        job["format"] = fmt

    U.lower_priority_below_normal()

    receipt = dict(
        schema="wm_n1_corpus_featurize_allocation/1",
        status="started", started_utc=U.utc_now(),
        inventory=str(Path(inventory_parquet).resolve()),
        data_root=str(data_root), out_root=str(out_root),
        workers=workers, max_workers_override=override, force=bool(force),
        idea_fast=bool(idea_fast), format=fmt, k_max=int(k_max),
        source_override=(None if source_override is None
                         else str(source_override)),
        n_source_override=len(override_dirs),
        n_runs=len(jobs),
        runs=[U.run_key(j["batch"], j["world"], j["arm"]) for j in jobs],
    )
    U.write_json_atomic(out_root / ALLOCATION_NAME, receipt)

    memory_before = MEM.require_memory(
        min_free_virtual_gb=min_free_virtual_gb,
        min_free_ram_gb=min_free_ram_gb, probe=probe)
    receipt["memory_before"] = memory_before

    U.OpenedAudit.reset()
    U.OpenedAudit.start()
    results = []
    failures = []
    gate_events = []
    try:
        batches = list(_chunks(jobs, workers))
        for bi, batch in enumerate(batches):
            eff = workers
            if workers > MEM.DYNAMIC_FALLBACK_WORKERS:
                sample = MEM.free_memory(probe=probe)
                eff, dropped = MEM.dynamic_worker_gate(sample, workers=workers)
                if dropped:
                    event = dict(
                        batch_index=bi, utc=U.utc_now(),
                        requested_workers=workers, effective_workers=eff,
                        free_virtual_gb=round(float(
                            sample.get("free_virtual_gb", 0.0)), 3),
                        free_ram_gb=round(float(
                            sample.get("free_ram_gb", 0.0)), 3),
                        thresholds=dict(virtual_gb=MEM.DYNAMIC_VIRTUAL_GB,
                                        ram_gb=MEM.DYNAMIC_RAM_GB))
                    gate_events.append(event)
                    receipt["worker_gate_events"] = gate_events
                    U.write_json_atomic(out_root / ALLOCATION_NAME, receipt)
                    print(f"worker-gate: free_virtual="
                          f"{event['free_virtual_gb']:.2f}GB free_ram="
                          f"{event['free_ram_gb']:.2f}GB below thresholds "
                          f"({MEM.DYNAMIC_VIRTUAL_GB:.0f}/"
                          f"{MEM.DYNAMIC_RAM_GB:.0f} GB) -> batch {bi} runs "
                          f"with {eff} workers (requested {workers})",
                          flush=True)
            results.extend(_run_batch(batch, eff, failures))
            if bi < len(batches) - 1:
                MEM.wait_for_memory(
                    min_free_virtual_gb=min_free_virtual_gb,
                    min_free_ram_gb=min_free_ram_gb, pause_s=pause_s,
                    probe=probe)
    finally:
        U.OpenedAudit.stop()

    U.OpenedAudit.assert_clean()

    plan_map = {}
    for j in jobs:
        key = U.run_key(j["batch"], j["world"], j["arm"])
        plan_map[key] = {"chunk": j["chunk"], "batch": j["batch"],
                         "world": j["world"],
                         "arm": j["arm"], "split": j["split"],
                         "t0_s": j["plan_row"].get("t0_s"),
                         "policy": j.get("policy"),
                         "plan_path": j["plan_path"],
                         "root_rel": j["root_rel"]}
    U.write_json_atomic(out_root / PLAN_MAP_NAME, {"runs": plan_map})

    index = _update_index(out_root, results)
    n_written = sum(1 for r in results if r["status"] == "written")
    n_skipped = sum(1 for r in results if r["status"] == "skipped")

    (out_root / OPENED_PATHS_NAME).write_text(
        json.dumps({"count": len(U.OpenedAudit.paths),
                    "paths": U.OpenedAudit.paths}, indent=1),
        encoding="utf-8")

    receipt.update(status="complete", finished_utc=U.utc_now(),
                   n_written=n_written, n_skipped=n_skipped,
                   n_failed=len(failures), failed=failures,
                   worker_gate_events=gate_events,
                   n_gate_drops=len(gate_events),
                   memory_after=MEM.free_memory(probe=probe),
                   opened_paths=len(U.OpenedAudit.paths))
    U.write_json_atomic(out_root / ALLOCATION_NAME, receipt)

    for r in results:
        rec = r["record"]
        print(f"  [{r['status']:7s}] {r['key']} n_ticks={rec['n_ticks']} "
              f"bytes={rec['total_bytes']}")
    for f in failures:
        print(f"  [failed ] {f['key']} {f['error']}: {f['detail']}")
    if gate_events:
        print(f"worker-gate: {len(gate_events)}/{len(batches)} batches "
              f"dropped to {MEM.DYNAMIC_FALLBACK_WORKERS} workers")
    print(f"featurize-all: runs={len(results) + len(failures)} "
          f"written={n_written} skipped={n_skipped} failed={len(failures)} "
          f"workers={workers}")

    return dict(runs=len(results) + len(failures), written=n_written,
                skipped=n_skipped, failed=failures,
                index=index, receipt=str(out_root / ALLOCATION_NAME),
                plan_map=str(out_root / PLAN_MAP_NAME), refused=[])


def _update_index(out_root: Path, results: list) -> dict:
    U.ensure_wm_n1_on_path()
    from features import writer as W

    index = W.load_index(out_root)
    for r in results:
        W.update_index(index, r["record"])
    W.save_index(out_root, index)
    W.write_sums(out_root, index)
    return index
