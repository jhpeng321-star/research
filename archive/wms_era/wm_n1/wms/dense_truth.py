"""Dense per-run driver-parameter ground truth (WM-S2 contribution C2).

Finding (b) in ``_out/S2_READY.md``: the hetpack driver sidecar aligns truth only
at each run's single packed snapshot tick (the run's first non-empty tick, which
carries 1--3 vehicles), while ``scenario/hetdata_truth.csv`` has one row per
scheduled vehicle (order 1e3 per run).  Per-driver identification therefore has
almost no dense ground truth.

This module builds a DENSE sidecar over EVERY tick of a raw heterogeneous-driver
run, entirely as an offline CPU post-processing step:

* a per-run ``vehicle_id -> (params[11], class)`` table from
  ``run/scenario/hetdata_truth.csv`` (optionally ``.gz``);
* a per-tick ``slot -> vehicle_id`` map reconstructed from
  ``run/physics_steps.jsonl`` (optionally ``.gz``) with the SAME persistent-slot
  pool algorithm the WM-N1 featurizer uses (``features.featurizer.parse_run``),
  so slot indices match the packed windows (the pack stores no vehicle ids).

Lookup for any ``(run, tick, slot)``::

    slot_vehicle.npy[tick, slot]  ->  row index into vehicle_ids.json
    vehicle_params.npy[row]       ->  11 truth params (PARAM_ORDER)
    vehicle_class.npy[row]        ->  driver class index (-1 unknown)

CPU-only and memory-bounded: one run at a time, arrays written through
``np.lib.format.open_memmap``; a whole pack is never loaded.  This module never
touches CAL/HOLDOUT/sealed/chunk-5 data and is evaluation-only ground truth (it
is never an online input and never enters a training loss).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import multiprocessing as mp
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# Single source of truth for the 11 parameters and the audited driver classes.
from .driver_sidecar import DRIVER_CLASSES, PARAM_ORDER

DT_S = 0.5
DEFAULT_K_MAX = 192
SCHEMA = "wms_dense_driver_truth/1"
_SAFE = re.compile(r"[^0-9A-Za-z_.-]+")


class DenseTruthError(RuntimeError):
    """A dense-truth contract violation (fail closed)."""


# ------------------------------------------------------------------ helpers ---
def _digest(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _open_lines(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _safe_name(text: str) -> str:
    return _SAFE.sub("-", str(text)).strip("-") or "run"


def physics_path(run_dir: Path) -> Optional[Path]:
    """Locate ``physics_steps.jsonl`` in the kit (``run/``) or flat layout."""
    run_dir = Path(run_dir)
    for base in (run_dir, run_dir / "run"):
        for name in ("physics_steps.jsonl", "physics_steps.jsonl.gz"):
            candidate = base / name
            if candidate.is_file():
                return candidate
    return None


def truth_path(run_dir: Path) -> Optional[Path]:
    """Locate ``hetdata_truth.csv`` in the kit or flat layout."""
    run_dir = Path(run_dir)
    for base in (run_dir, run_dir / "run"):
        for rel in (("scenario", "hetdata_truth.csv"),
                    ("scenario", "hetdata_truth.csv.gz"),
                    ("hetdata_truth.csv",), ("hetdata_truth.csv.gz",)):
            candidate = base.joinpath(*rel)
            if candidate.is_file():
                return candidate
    return None


# ------------------------------------------------------------------- truth ----
def load_truth_table(path: Path) -> dict:
    """Parse ``hetdata_truth.csv`` into ``vehicle_id -> dict`` (params+class)."""
    table: dict = {}
    with _open_text(Path(path)) as fh:
        reader = csv.DictReader(fh)
        missing = [name for name in PARAM_ORDER
                   if name not in (reader.fieldnames or [])]
        if missing:
            raise DenseTruthError(
                f"{path}: truth CSV lacks parameter columns {missing}")
        for row in reader:
            vid = row.get("vehicle_id")
            if not vid:
                continue
            try:
                params = [float(row[name]) for name in PARAM_ORDER]
            except (TypeError, ValueError) as exc:
                raise DenseTruthError(
                    f"{path}: vehicle {vid!r} has a non-numeric parameter") from exc
            table[str(vid)] = {
                "driver_class": row.get("driver_class") or None,
                "params": params,
            }
    if not table:
        raise DenseTruthError(f"{path}: empty truth table")
    return table


# ------------------------------------------------------------- slot tracking --
def assign_slots(physics: Path, *, k_max: int = DEFAULT_K_MAX,
                 t_origin: Optional[float] = None):
    """Stream a run's physics log into the featurizer's per-tick slot map.

    Returns ``(t_origin, n_ticks, records, vids, n_dropped)`` where ``records`` is
    a list of ``(local_tick, slot, vid_row)`` and ``vids`` the ordered vehicle-id
    list (``vid_row`` indexes it).  The persistent-pool algorithm is byte-for-byte
    the same as :func:`features.featurizer.parse_run`: a continuing vehicle keeps
    its slot; a slot freed by a leaver is reused (lowest free index first) by a
    newcomer in the same or a later tick; a vehicle beyond ``k_max`` concurrent is
    dropped and counted.
    """
    import heapq

    vid_index: dict = {}
    vids: list = []
    free = list(range(int(k_max)))
    heapq.heapify(free)
    current: dict = {}
    records: list = []
    n_dropped = 0
    last_t: Optional[float] = None
    resolved_origin = t_origin

    with _open_lines(Path(physics)) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            t = float(row["time_s"])
            if resolved_origin is None:
                resolved_origin = round(t - 0.5, 6)
            if t <= resolved_origin + 1e-9:
                continue
            last_t = t
            k = int(round((t - resolved_origin) / DT_S))
            if abs(resolved_origin + k * DT_S - t) > 1e-6:
                raise DenseTruthError(
                    f"{physics}: off-grid tick t={t} (t_origin={resolved_origin})")
            vehs = row.get("vehicles") or {}
            for vid in [v for v in current if v not in vehs]:
                heapq.heappush(free, current.pop(vid))
            for vid, _vv in vehs.items():
                vid = str(vid)
                slot = current.get(vid)
                if slot is None:
                    if not free:
                        n_dropped += 1
                        continue
                    slot = current[vid] = heapq.heappop(free)
                vi = vid_index.get(vid)
                if vi is None:
                    vi = vid_index[vid] = len(vids)
                    vids.append(vid)
                records.append((k, slot, vi))

    if last_t is None or resolved_origin is None:
        raise DenseTruthError(f"{physics}: no rows after t_origin")
    n_ticks = int(round((last_t - resolved_origin) / DT_S)) + 1
    return float(resolved_origin), n_ticks, records, vids, n_dropped


def _scatter_records(target: np.ndarray, records) -> None:
    """Vectorised ``target[k, slot] = vi`` for every ``(k, slot, vi)`` record.

    WM-S2 speed fix: one fancy-index scatter replaces the per-record Python loop
    over every (tick, vehicle) assignment in the dense-truth builder.  Each
    ``(k, slot)`` pair is assigned at most once (a slot holds a single vehicle),
    so the scatter is exact; parity with the loop is asserted in
    ``tests/test_wms_s2_vectorization_parity.py``.
    """
    if not records:
        return
    rec = np.asarray(records, dtype=np.int64)
    target[rec[:, 0], rec[:, 1]] = rec[:, 2]


# ------------------------------------------------------------------- build ----
def build_run(run_dir: Path, out_dir: Path, *, k_max: int = DEFAULT_K_MAX
              ) -> dict:
    """Build one run's dense tables under ``out_dir`` (created if absent)."""
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    physics = physics_path(run_dir)
    truth = truth_path(run_dir)
    if physics is None:
        raise DenseTruthError(f"{run_dir}: no physics_steps.jsonl(.gz)")
    if truth is None:
        raise DenseTruthError(f"{run_dir}: no scenario/hetdata_truth.csv(.gz)")

    table = load_truth_table(truth)
    t_origin, n_ticks, records, vids, n_dropped = assign_slots(
        physics, k_max=k_max)

    out_dir.mkdir(parents=True, exist_ok=True)
    slot_vehicle = np.lib.format.open_memmap(
        out_dir / "slot_vehicle.npy", mode="w+", dtype=np.int32,
        shape=(int(n_ticks), int(k_max)))
    slot_vehicle[...] = -1
    _scatter_records(slot_vehicle, records)
    slot_vehicle.flush()
    del slot_vehicle

    n_veh = len(vids)
    params = np.full((n_veh, len(PARAM_ORDER)), np.nan, dtype=np.float32)
    classes = np.full(n_veh, -1, dtype=np.int8)
    n_with_truth = 0
    for vi, vid in enumerate(vids):
        got = table.get(vid)
        if got is None:
            continue
        params[vi] = got["params"]
        n_with_truth += 1
        cls = got["driver_class"]
        classes[vi] = DRIVER_CLASSES.index(cls) if cls in DRIVER_CLASSES else -1
    np.save(out_dir / "vehicle_params.npy", params, allow_pickle=False)
    np.save(out_dir / "vehicle_class.npy", classes, allow_pickle=False)
    (out_dir / "vehicle_ids.json").write_text(
        json.dumps(vids, indent=0), encoding="utf-8")

    files = {}
    for name in ("slot_vehicle.npy", "vehicle_params.npy",
                 "vehicle_class.npy", "vehicle_ids.json"):
        files[name] = _digest(out_dir / name)
    return {
        "n_ticks": int(n_ticks), "k_max": int(k_max), "t0_s": float(t_origin),
        "n_vehicles": int(n_veh),         "n_vehicles_with_truth": int(n_with_truth),
        "n_rows_with_any_truth": int(np.sum(np.isfinite(params).any(axis=1))),
        "n_slot_assignments": int(len(records)),
        "n_dropped_concurrent": int(n_dropped),
        "physics_sha256": _digest(Path(physics)),
        "truth_sha256": _digest(Path(truth)),
        "files": files,
    }


def discover_runs(runs_root: Path, *, splits: Optional[Sequence[str]] = None,
                  limit: Optional[int] = None) -> list:
    """Discover completed kit-layout runs (``<run_id>/attempt-*/run``)."""
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        raise DenseTruthError(f"runs root not found: {runs_root}")
    wanted = None if not splits else {str(s).upper() for s in splits}
    records: list = []
    for run_root in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        attempts = sorted((a for a in run_root.iterdir()
                           if a.is_dir() and a.name.startswith("attempt-")),
                          key=lambda a: a.name)
        chosen = None
        for attempt in reversed(attempts):
            receipt = attempt / "receipt.json"
            if not receipt.is_file():
                continue
            try:
                data = json.loads(receipt.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            if data.get("status") == "complete":
                chosen = (attempt, data)
                break
        if chosen is None:
            continue
        attempt, data = chosen
        meta = data.get("run") or {}
        world, controller = meta.get("world"), meta.get("controller")
        if not world or not controller:
            continue
        split = str(meta.get("split", "")).upper()
        if wanted is not None and split not in wanted:
            continue
        run_dir = attempt / "run"
        if physics_path(run_dir) is None or truth_path(run_dir) is None:
            continue
        records.append({
            "run_id": meta.get("run_id") or run_root.name,
            "split": split, "world": str(world), "arm": str(controller),
            "run_dir": str(run_dir),
            "run_key": f"{world}__{controller}",
        })
    if limit is not None:
        records = records[:int(limit)]
    if not records:
        raise DenseTruthError(f"no completed runs with truth under {runs_root}")
    records.sort(key=lambda r: (r["world"], r["arm"], r["run_id"]))
    return records


def _build_one(task) -> dict:
    record, out_root, k_max = task
    out_dir = Path(out_root) / _safe_name(record["run_key"])
    built = build_run(Path(record["run_dir"]), out_dir, k_max=k_max)
    return {**record, "dir": _safe_name(record["run_key"]), **built}


def build_dense_truth(runs_root: Path, out_root: Path, *,
                      k_max: int = DEFAULT_K_MAX, workers: int = 1,
                      splits: Optional[Sequence[str]] = None,
                      limit: Optional[int] = None, force: bool = False) -> dict:
    """Build the dense sidecar for every discovered run (parallel by run)."""
    out_root = Path(out_root)
    records = discover_runs(runs_root, splits=splits, limit=limit)
    tasks = [(r, str(out_root), int(k_max)) for r in records]
    results: list = []
    if workers <= 1:
        for task in tasks:
            results.append(_build_one(task))
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=int(workers)) as pool:
            for res in pool.imap(_build_one, tasks, 1):
                results.append(res)
    results.sort(key=lambda r: (r["world"], r["arm"], r["run_id"]))
    lines = [json.dumps(r, sort_keys=True) for r in results]
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "runs.jsonl").write_text("\n".join(lines) + "\n",
                                         encoding="utf-8")
    receipt = {
        "schema": SCHEMA, "runs_root": str(Path(runs_root).resolve()),
        "k_max": int(k_max), "n_runs": len(results),
        "splits": sorted({r["split"] for r in results}),
        "n_vehicles": int(sum(r["n_vehicles"] for r in results)),
        "n_vehicles_with_truth": int(
            sum(r["n_vehicles_with_truth"] for r in results)),
        "n_slot_assignments": int(sum(r["n_slot_assignments"] for r in results)),
        "n_dropped_concurrent": int(
            sum(r["n_dropped_concurrent"] for r in results)),
        "runs_file": "runs.jsonl",
    }
    (out_root / "DENSE_TRUTH_RECEIPT.json").write_text(
        json.dumps(receipt, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    sums = [f"{_digest(out_root / 'runs.jsonl')}  runs.jsonl",
            f"{_digest(out_root / 'DENSE_TRUTH_RECEIPT.json')}  "
            f"DENSE_TRUTH_RECEIPT.json"]
    (out_root / "SHA256SUMS").write_text("\n".join(sums) + "\n",
                                         encoding="utf-8")
    return receipt


# ------------------------------------------------------------------ loader ----
@dataclass
class DenseDriverTruth:
    """Read-side view of a dense-truth directory (arrays mmapped on demand)."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        receipt = self.root / "DENSE_TRUTH_RECEIPT.json"
        if not receipt.is_file():
            raise DenseTruthError(f"not a dense-truth dir: {self.root}")
        self.receipt = json.loads(receipt.read_text(encoding="utf-8"))
        self._runs: dict = {}
        runs_file = self.root / self.receipt.get("runs_file", "runs.jsonl")
        for line in runs_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                self._runs[(row["world"], row["arm"])] = row

    @property
    def run_keys(self) -> list:
        return sorted(self._runs)

    def run(self, world: str, arm: str) -> dict:
        key = (str(world), str(arm))
        if key not in self._runs:
            raise DenseTruthError(f"dense truth has no run {key}")
        return self._runs[key]

    def lookup(self, world: str, arm: str, tick: int, slot: int):
        """``(vehicle_id, params[11], class_index)`` or ``None`` for an empty slot."""
        row = self.run(world, arm)
        base = self.root / row["dir"]
        slot_vehicle = np.load(base / "slot_vehicle.npy", mmap_mode="r",
                               allow_pickle=False)
        if not (0 <= tick < slot_vehicle.shape[0]) or not (0 <= slot < slot_vehicle.shape[1]):
            raise DenseTruthError(
                f"(tick={tick}, slot={slot}) outside {world}/{arm}")
        vid_row = int(slot_vehicle[tick, slot])
        if vid_row < 0:
            return None
        vids = json.loads((base / "vehicle_ids.json").read_text(encoding="utf-8"))
        params = np.load(base / "vehicle_params.npy", mmap_mode="r",
                         allow_pickle=False)
        classes = np.load(base / "vehicle_class.npy", mmap_mode="r",
                          allow_pickle=False)
        return vids[vid_row], np.asarray(params[vid_row]), int(classes[vid_row])

    def slot_vehicle(self, world: str, arm: str) -> np.ndarray:
        row = self.run(world, arm)
        return np.load(self.root / row["dir"] / "slot_vehicle.npy",
                       mmap_mode="r", allow_pickle=False)


# ------------------------------------------------------------------ CLI -------
def _cmd_build(args) -> int:
    receipt = build_dense_truth(Path(args.runs_root), Path(args.out),
                                k_max=int(args.k_max), workers=int(args.workers),
                                splits=args.split, limit=args.limit,
                                force=bool(args.force))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def _cmd_describe(args) -> int:
    sidecar = DenseDriverTruth(args.out)
    out = dict(sidecar.receipt)
    out["run_keys"] = sidecar.run_keys[:20]
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def _cmd_verify(args) -> int:
    sidecar = DenseDriverTruth(args.out)
    root = sidecar.root
    bad = []
    for (world, arm), row in sorted(sidecar._runs.items()):
        base = root / row["dir"]
        for name, want in (row.get("files") or {}).items():
            path = base / name
            if not path.is_file() or _digest(path) != want:
                bad.append(f"{world}/{arm}/{name}")
    for rel in ("runs.jsonl", "DENSE_TRUTH_RECEIPT.json"):
        p = root / rel
        if not p.is_file():
            bad.append(rel)
    report = {"schema": SCHEMA, "out": str(root), "n_runs": len(sidecar._runs),
              "n_bad": len(bad), "bad": bad[:20]}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not bad else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Dense driver truth (WM-S2)")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="build the dense sidecar over raw runs")
    b.add_argument("--runs-root", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--k-max", type=int, default=DEFAULT_K_MAX)
    b.add_argument("--workers", type=int, default=1)
    b.add_argument("--split", action="append", default=None,
                   help="restrict to a split (repeatable); default all")
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--force", action="store_true")
    b.set_defaults(func=_cmd_build)
    d = sub.add_parser("describe", help="print the dense-truth receipt")
    d.add_argument("--out", required=True)
    d.set_defaults(func=_cmd_describe)
    v = sub.add_parser("verify", help="re-hash every per-run array")
    v.add_argument("--out", required=True)
    v.set_defaults(func=_cmd_verify)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
