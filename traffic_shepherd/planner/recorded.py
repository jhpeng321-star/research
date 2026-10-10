"""Offline planner test on generated campaign data (PL-02a). Zero simulation.

Loads the WM-44/WM-41 campaign runs (READ-ONLY) and answers predict() queries
from the RECORDED branch outcomes, so the planner's decision rules can be
evaluated offline before any perfect-sim SUMO experiment.

Inputs (per output/planner_v1/BRIEF_PL02a.md and its AMENDMENT; every path
and field name configurable through :class:`Layout`):
- runs_root/<world>__<arm>/ directories; the BASE arm keeps the frozen
  evaluate layout (physics/telemetry/scenario under ``<run>/run/``), restored
  arms keep physics/branch_record/arrivals/outcome/receipt at ``<run>/``.
- WORLD_PLAN.json (results_plan_wm44/): world metadata (demand, penetration,
  split, seed) and per-branch ``restores_state.from_run`` (the base run),
  ``native_sibling_run`` and ``branch_seed`` -- this is the root->branches
  index (the amendment: no separate index file).
- RUN_RECEIPT.json marks a run COMPLETE; partial data is tolerated: it is
  reported in ``CampaignIndex.skipped`` and never silently mixed in.

DECLARED DEFINITIONS (also rendered into REPORT.md by the runner):
- WINDOW: per root, the common comparison window is
  ``[t0, t0 + min(coverage over compared arms)]`` where coverage is the
  arm's recorded post-t0 length (chunk-1: base covers 542 s post-t0,
  restored arms 600 s -> common 542 s unless the reconstructor extends the
  base).
- DELAY (per vehicle, over the window): time on section (sum of 0.5 s
  physics-step coverage) - free-flow time (section_len / free_flow_mps)
  plus entry wait. Entry wait = first physics appearance - scheduled depart
  (requests.rou.xml @depart) for vehicles scheduled from t0 on; a vehicle
  scheduled in-window but never inserted by window end accrues wait to the
  window end. arrivals.jsonl is the EXIT list (not used for waits).
- SAFETY (declared stricter than the contract: the window is the FULL
  horizon, not just Part A): collision (physics colliding_ids or
  outcome_record.collisions), min closing TTC per lane and has_closing_pair
  from physics rows (gap / positive closing speed; the closing_ttc
  semantics), TTC<3 s exposure (vehicle-seconds), hard-brake
  vehicle-seconds (a < -3 m/s^2). OPT-IN TRUE GAP (gap #18, PL-EXP1b):
  ``true_gap_lengths`` (vehicle id -> length from the scenario vType
  definitions) switches the TTC gap to the bumper-to-bumper gap
  leader_pos - leader_length - follower_pos; the DEFAULT front-bumper
  difference is unchanged and the true-gap mode fails closed on any
  vehicle whose length is not declared (TrueGapLengthError).
- TELEPORTS are not logged by the campaign: the detector flags (a) a
  reappearing vehicle whose position jumps more than ``max_position_jump_m``
  between consecutive physics sightings, and (b) a vehicle that disappears
  before window end without an arrivals exit while still short of the exit
  band (``exit_band_m``). ``teleport_policy``: "detect" (default; a detected
  teleport invalidates the cell) or "fail_closed_all" (invalidate everything
  until explicit teleport logging exists).
- PLAN IDENTITY: an arm's constant plan is its switch_schedule t0 per-CAV
  payload; realised dispatcher/physics actions are cross-checked for fixed
  arms (mismatch invalidates the cell). Switching arms are schedules, not
  v1 constant plans: headroom only, never planner candidates.
- RecordedBranchWorldModel emits contract v1.1 predictions (M=1, K=1) and
  FAILS CLOSED for any plan that is not a recorded constant plan
  (cell_valid=False).
"""

import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .interface import Prediction
from .paths import rd_wt_root, research_dense_root

__all__ = [
    "BranchOutcome",
    "CampaignIndex",
    "DelayProvider",
    "Layout",
    "PhysicsScan",
    "RecordedBranchWorldModel",
    "Root",
    "RunPaths",
    "SUMO_DEFAULT_VEHICLE_LENGTH_M",
    "Tick",
    "TrueGapLengthError",
    "atomic_write_json",
    "branch_outcome_from_dict",
    "branch_outcome_to_dict",
    "default_delay",
    "get_scan",
    "layout_fingerprint",
    "load_json",
    "load_jsonl",
    "scan_run_physics",
    "sha256_file",
    "true_gap_lengths_from_rou",
    "vtype_lengths_from_rou",
]

CONTRACT_VERSION_V1_1 = "wd01_predict/1.1"


# --------------------------------------------------------------------- io
def load_json(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file's bytes (cache identity for receipts and code)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping) -> None:
    """Write JSON crash-atomically: tmp file in the same directory, fsync,
    then os.replace (rename) over the target. A crash mid-write leaves the
    previous cache entry (or none) intact, never a half-written entry."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():  # only still there if the replace never ran
            tmp.unlink()


def layout_fingerprint(layout: "Layout") -> str:
    """sha256 over every Layout field (canonical JSON) -- part of the cache
    key: any definition change (thresholds, provider config, edge lengths)
    invalidates cached outcomes."""
    payload = {f.name: getattr(layout, f.name) for f in fields(layout)}
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------- layout
@dataclass(frozen=True)
class Layout:
    """Every path/field name the loader depends on (AMENDMENT: configurable)."""

    arm_sep: str = "__"
    receipt_name: str = "RUN_RECEIPT.json"
    receipt_status_key: str = "exit_status"
    # PL-02a-S: acceptable receipt statuses (default: only "complete").
    # The pilot-r3 verify run carries "failed" purely for the missing t0
    # tick while its EXACTNESS_VERDICT shows divergence_count == 0.
    receipt_status_ok_values: Optional[Tuple[str, ...]] = None
    # restored arms may keep their files in a subdirectory (pilot r3: "run")
    restored_file_subdir: str = ""
    # PL-02a-S: the pilot-r3 data has base+verify only; when False, roots
    # without a native sibling are still indexed (headroom/planner analyses
    # are simply not runnable on them -- declared).
    require_native_sibling: bool = True
    receipt_complete_value: str = "complete"
    branch_record_name: str = "branch_record.json"
    outcome_name: str = "outcome_record.json"
    arrivals_name: str = "arrivals.jsonl"
    physics_name: str = "physics_steps.jsonl"
    base_physics_subdir: str = "run"
    scenario_rou_relpaths: Tuple[str, ...] = ("run/scenario/requests.rou.xml",)
    switch_schedule_name: str = "switch_schedule.json"
    dispatcher_rows_name: str = "dispatcher_rows.jsonl"
    schedule_events_key: str = "events"
    # PL-02a CHUNK-1: the schedule payload may live NESTED under a container
    # key (chunk 1 nests it under "schedule" inside switch_schedule.json),
    # uses "time_s" per-vehicle event rows (not "t_s"), and carries the t0
    # per-CAV payload as an "assignments" map.  All three are Layout keys.
    schedule_container_key: Optional[str] = None
    # Constant-plan rule: a schedule is constant iff its events carry exactly
    # ONE distinct time (chunk-1 fixed arms emit one ROW per CAV at t0).
    # Per-vehicle dispatcher telemetry (chunk 1: rows of
    # {timestamp_s, vehicle_id, action, ...}) for the realised-action
    # cross-check; "map" is the pilot's {time_s, actions: {...}} rows.
    dispatcher_rows_format: str = "map"
    # chunk-1 branch_record declares the arm as branch_type (native/pi/fixed/
    # switching) instead of suffix_controller; branch_type_actions maps a
    # branch type to its constant action ("pi" -> "pi-external-u").
    branch_type_key: str = "branch_type"
    branch_type_actions: Optional[Mapping[str, str]] = None
    # per-row list field naming the CAV universe (chunk-1 rows carry
    # "cav_ids" even when cav_actions is empty)
    row_cav_list_key: str = "cav_ids"
    # PL-02a PILOT: branch_record field declaring an arm's constant
    # controller when no switch_schedule exists
    branch_controller_key: str = "suffix_controller"
    # arms that exist only to verify the restore (same-action continuation
    # of the base): integrity checks, never constant-plan candidates
    integrity_arm_names: Tuple[str, ...] = ("verify",)
    schedule_time_key: str = "t_s"
    schedule_plan_key: str = "per_cav"
    # physics row field names
    row_time_key: str = "time_s"
    row_cav_key: str = "cav_actions"
    row_vehicles_key: str = "vehicles"
    row_colliding_key: str = "colliding_ids"
    veh_lane_key: str = "lane_id"
    # Optional road/edge lengths by road_id: let the built-in provider bridge
    # an edge change's advance ((len(a) - pos_a) + pos_b). Absent lengths ->
    # the step's advance is skipped and counted (declared diagnostic).
    road_length_by_id: Optional[Mapping[str, float]] = None
    veh_pos_key: str = "lane_position_m"
    veh_speed_key: str = "speed_mps"
    veh_accel_key: str = "acceleration_mps2"
    # outcome_record cross-check field names (absent -> cross-check skipped)
    outcome_delay_key: str = "time_on_section_s"
    outcome_collisions_key: str = "collisions"
    # definition constants
    physics_dt_s: float = 0.5
    section_len_m: float = 1000.0
    free_flow_mps: float = 30.0
    ttc_threshold_s: float = 3.0
    hard_brake_mps2: float = -3.0
    max_position_jump_m: float = 25.0
    exit_band_m: float = 30.0
    teleport_policy: str = "detect"  # "detect" | "fail_closed_all"
    crosscheck_delay_tol_s: float = 5.0
    onset_speed_share: float = 0.6  # congestion onset: lane mean speed below
    onset_persist_s: float = 4.0  # this share of free-flow for this long
    # PL-02a-S3: per-arm exit identity.  Teleport detection must use THAT
    # arm's own exits: the run's arrivals.jsonl when present, else the arm's
    # own physics rows -- per-row ``row_arrived_key`` ids (named exits) plus
    # a last-sighting fallback inside the exit zone (position >=
    # section_len_m - exit_band_m on ``exit_zone_road_ids``; None = any
    # road).  A sibling's exit list is never borrowed: it is valid only for
    # identical-action continuations and wrong for native/pi branches.
    row_arrived_key: str = "arrived_ids"
    arrivals_vehicle_key: str = "vehicle_id"
    exit_zone_road_ids: Optional[Tuple[str, ...]] = None
    # PL-02a-S3: t0 insertion state from the saved-state manifest.  t0 comes
    # from ``manifest_t0_key`` (branch_point_t0_s), never hard-coded; every
    # id in physics_row_at_t0[manifest_live_key] was inserted at/before t0
    # and gets ZERO entry wait in every arm (contract §8.14), including
    # restored arms whose first physics row is t0 + dt.
    saved_state_subdir: str = "saved_state"
    manifest_name: str = "STATE_MANIFEST.json"
    manifest_t0_key: str = "branch_point_t0_s"
    manifest_t0_row_key: str = "physics_row_at_t0"
    manifest_live_key: str = "live_ids"
    # PL-02a-S3: a "failed" receipt is accepted ONLY when
    # ``accept_failed_receipt_if_exact`` is set AND the run's
    # EXACTNESS_VERDICT.json shows the exactness gate itself passed
    # (divergence_count == exactness_max_divergence_count, no extra ticks,
    # and every tick missing from the restore is <= the receipt's t0 -- i.e.
    # the only failure is the t0-tick window boundary).  Otherwise the run
    # is excluded and counted in CampaignIndex.skipped.
    accept_failed_receipt_if_exact: bool = False
    exactness_verdict_name: str = "EXACTNESS_VERDICT.json"
    exactness_max_divergence_count: int = 0


# ------------------------------------------------------------------ index
@dataclass
class RunPaths:
    """One complete run directory and the files it provides."""

    directory: Path
    arm: str
    is_base: bool
    receipt: dict
    world_meta: dict = field(default_factory=dict)
    branch_record: Optional[dict] = None
    physics_paths: Tuple[Path, ...] = ()
    arrivals_path: Optional[Path] = None
    outcome_path: Optional[Path] = None
    schedule_path: Optional[Path] = None
    dispatcher_path: Optional[Path] = None
    rou_paths: Tuple[Path, ...] = ()
    # PL-02a-S3: the base arm's saved-state manifest (t0 insertion truth)
    manifest_path: Optional[Path] = None
    # PL-02a-C1: memoized streaming scan (ONE physics parse per run per
    # process, shared by every metric and both delay providers).
    scan_cache: Optional["PhysicsScan"] = field(default=None, repr=False, compare=False)


@dataclass
class Root:
    """One branch point: the base run plus its recorded restored branches."""

    label: str
    t0_s: float
    base: RunPaths
    branches: Dict[str, RunPaths]
    meta: dict = field(default_factory=dict)
    # PL-02a-S3: ids inserted at/before t0 per the base's saved-state
    # manifest (physics_row_at_t0.live_ids); empty when no manifest exists.
    t0_present_ids: frozenset = frozenset()


def _first_existing(base: Path, relpaths: Sequence[str]) -> Tuple[Path, ...]:
    return tuple(base / rel for rel in relpaths if (base / rel).is_file())


def _single_file(directory: Path, name: str) -> Optional[Path]:
    candidate = directory / name
    return candidate if candidate.is_file() else None


def _arrivals_path(run_dir: Path, file_dir: Path, layout: Layout) -> Optional[Path]:
    """The run's OWN arrivals file, checked at the run root first and then
    in the restored file subdir (pilot r3 keeps it under <run>/run/)."""
    for directory in (run_dir, file_dir):
        candidate = directory / layout.arrivals_name
        if candidate.is_file():
            return candidate
    return None


class CampaignIndex:
    """Scans the runs root, keeps COMPLETE runs, groups them into roots.

    Tolerates partial data: incomplete or unreadable runs are listed in
    ``skipped`` with a reason and never silently mixed into a root."""

    def __init__(
        self,
        runs_root: Path,
        world_plan_path: Optional[Path] = None,
        layout: Optional[Layout] = None,
        run_prefixes: Optional[Sequence[str]] = None,
    ):
        self.runs_root = Path(runs_root)
        self.world_plan_path = None if world_plan_path is None else Path(world_plan_path)
        self.layout = layout or Layout()
        # PL-02a-S4: optional scan scope.  When set, only run directories
        # whose name starts with one of these prefixes are considered;
        # everything else is OUT OF SCOPE (not indexed, not counted as
        # skipped) -- e.g. a live stage writing sibling runs into the same
        # root must not even be read.
        self.run_prefixes = (
            tuple(str(p) for p in run_prefixes) if run_prefixes is not None else None
        )
        self.roots: Dict[str, Root] = {}
        self.roots_seen: set = set()  # incl. partial roots (reported, unused)
        self.skipped: List[str] = []

    # ------------------------------------------------------------ helpers
    def _complete(self, run_dir: Path) -> Optional[dict]:
        receipt_path = run_dir / self.layout.receipt_name
        if not receipt_path.is_file():
            return None
        try:
            receipt = load_json(receipt_path)
        except (OSError, ValueError) as exc:
            self.skipped.append(f"{run_dir.name}: unreadable receipt {exc!r}")
            return None
        status = receipt.get(self.layout.receipt_status_key)
        ok_values = self.layout.receipt_status_ok_values or (self.layout.receipt_complete_value,)
        if status not in ok_values:
            if (
                status == "failed"
                and self.layout.accept_failed_receipt_if_exact
                and self._failed_receipt_is_exact(run_dir, receipt)
            ):
                return receipt
            self.skipped.append(
                f"{run_dir.name}: receipt status {status!r} not in {sorted(ok_values)}"
            )
            return None
        return receipt

    def _failed_receipt_is_exact(self, run_dir: Path, receipt: Mapping) -> bool:
        """Whether a 'failed' receipt may pass the exactness gate: the run's
        EXACTNESS_VERDICT.json exists with divergence_count within the
        configured maximum, no extra ticks, and every tick missing from the
        restore <= the receipt's t0 -- i.e. the ONLY failure is the t0-tick
        window boundary.  Anything else stays excluded (and counted)."""
        subdirs = [run_dir]
        if self.layout.restored_file_subdir:
            subdirs.insert(0, run_dir / self.layout.restored_file_subdir)
        verdict = None
        for directory in subdirs:
            candidate = directory / self.layout.exactness_verdict_name
            if candidate.is_file():
                verdict = self._optional_json(candidate, run_dir.name)
                break
        if verdict is None:
            return False
        if int(verdict.get("divergence_count", -1)) != self.layout.exactness_max_divergence_count:
            return False
        if verdict.get("extra_in_verify"):
            return False
        t0 = _receipt_t0_s(receipt, self.layout)
        if t0 is None:
            return False
        missing = [float(m) for m in verdict.get("missing_in_verify") or ()]
        return all(m <= t0 + 1e-9 for m in missing)

    def _world_plan_meta(self) -> Dict[str, dict]:
        """run name -> world/arm metadata from WORLD_PLAN.json (amendment:
        the root->branches index comes from here + branch_record)."""
        meta: Dict[str, dict] = {}
        if self.world_plan_path is None or not self.world_plan_path.is_file():
            return meta
        plan = load_json(self.world_plan_path)
        for world in plan.get("worlds", []):
            label = world.get("world")
            shared = {
                "world": label,
                "loading_veh_h": world.get("loading_veh_h"),
                "penetration": world.get("penetration"),
                "split": world.get("split"),
                "seed": world.get("seed"),
            }
            base = world.get("base_arm") or {}
            base_run = base.get("run")
            if base_run:
                meta[base_run] = {
                    **shared,
                    "arm": "base",
                    "is_base": True,
                    "controller_name": base.get("controller_name"),
                    "restores_from": base_run,
                    "t0_s": float(
                        (base.get("saves_state") or {}).get("at_s", world.get("t0_s", 60.0))
                    ),
                }
            for branch in world.get("branches", []):
                run = branch.get("run")
                if run:
                    meta[run] = {
                        **shared,
                        "arm": branch.get("arm"),
                        "is_base": False,
                        "restores_from": (branch.get("restores_state") or {}).get("from_run"),
                        "native_sibling_run": branch.get("native_sibling_run"),
                        "branch_seed": branch.get("branch_seed"),
                        "t0_s": float(
                            (branch.get("restores_state") or {}).get(
                                "at_s", world.get("t0_s", 60.0)
                            )
                        ),
                    }
        return meta

    # -------------------------------------------------------------- build
    def build(self) -> "CampaignIndex":
        layout = self.layout
        run_meta = self._world_plan_meta()
        runs: Dict[str, RunPaths] = {}
        if not self.runs_root.is_dir():
            self.skipped.append(f"runs root missing: {self.runs_root}")
            return self
        for run_dir in sorted(self.runs_root.iterdir()):
            if not run_dir.is_dir():
                continue
            if self.run_prefixes is not None and not run_dir.name.startswith(self.run_prefixes):
                continue  # out of scope: not indexed, not counted
            receipt = self._complete(run_dir)
            if receipt is None:
                continue
            meta = run_meta.get(run_dir.name, {})
            # PL-02a CHUNK-1: the run-dir suffix IS the arm identity
            # (fixed1/fixed2/switching1-3; the chunk-1 world plan labels
            # arms by BRANCH TYPE "fixed"/"switching", which would collide);
            # the plan meta arm is only the fallback for suffixed roots.
            arm = str(run_dir.name.rsplit(layout.arm_sep, 1)[-1] or meta.get("arm"))
            is_base = bool(meta.get("is_base")) or arm == "base"
            file_dir = run_dir
            if not is_base and layout.restored_file_subdir:
                file_dir = run_dir / layout.restored_file_subdir
            physics_dir = (run_dir / layout.base_physics_subdir) if is_base else file_dir
            record = RunPaths(
                directory=run_dir,
                arm=arm,
                is_base=is_base,
                receipt=receipt,
                world_meta=meta,
                branch_record=(
                    None
                    if is_base
                    else self._optional_json(file_dir / layout.branch_record_name, run_dir.name)
                ),
                physics_paths=_first_existing(physics_dir, (layout.physics_name,)),
                arrivals_path=_arrivals_path(run_dir, file_dir, layout),
                outcome_path=(
                    run_dir / layout.outcome_name
                    if (run_dir / layout.outcome_name).is_file()
                    else (
                        file_dir / layout.outcome_name
                        if file_dir != run_dir and (file_dir / layout.outcome_name).is_file()
                        else None
                    )
                ),
                schedule_path=(
                    run_dir / layout.switch_schedule_name
                    if (run_dir / layout.switch_schedule_name).is_file()
                    else (
                        file_dir / layout.switch_schedule_name
                        if file_dir != run_dir
                        and (file_dir / layout.switch_schedule_name).is_file()
                        else None
                    )
                ),
                dispatcher_path=(
                    run_dir / layout.dispatcher_rows_name
                    if (run_dir / layout.dispatcher_rows_name).is_file()
                    else (
                        file_dir / layout.dispatcher_rows_name
                        if file_dir != run_dir
                        and (file_dir / layout.dispatcher_rows_name).is_file()
                        else None
                    )
                ),
                rou_paths=_first_existing(run_dir, layout.scenario_rou_relpaths),
                # PL-02a-S3: the BASE arm carries the saved-state manifest
                # (t0 insertion truth); restored arms never have one.
                manifest_path=(
                    _single_file(run_dir / layout.saved_state_subdir, layout.manifest_name)
                    if is_base
                    else None
                ),
            )
            runs[run_dir.name] = record

        for run_name, record in runs.items():
            meta = record.world_meta
            root_label = str(
                meta.get("restores_from")
                or run_name.rsplit(layout.arm_sep, 1)[0] + layout.arm_sep + "base"
            )
            self.roots_seen.add(root_label)
            base_run = runs.get(root_label)
            if base_run is None or not base_run.is_base:
                self.skipped.append(f"{run_name}: base run {root_label!r} missing/incomplete")
                continue
            if record.arm != "native":
                prefix = run_name.rsplit(layout.arm_sep, 1)[0]
                native_run = meta.get("native_sibling_run") or prefix + layout.arm_sep + "native"
                if not native_run or native_run not in runs:
                    if layout.require_native_sibling:
                        self.skipped.append(
                            f"{run_name}: native sibling {native_run!r} missing/incomplete"
                        )
                        continue
                    # require_native_sibling=False: index the root anyway
                    # (headroom/planner analyses need the sibling; declared)
            root = self.roots.setdefault(
                root_label,
                Root(
                    label=root_label,
                    t0_s=float(meta.get("t0_s", 60.0)),
                    base=base_run,
                    branches={},
                    meta={
                        key: base_run.world_meta.get(key)
                        for key in ("loading_veh_h", "penetration", "split", "seed")
                    },
                ),
            )
            if not record.is_base:
                root.branches[record.arm] = record
            # PL-02a-S3: t0 and the t0-present set come from the BASE's
            # saved-state manifest when one exists (branch_point_t0_s /
            # physics_row_at_t0.live_ids) -- never from a hard-coded 60.0.
            if root.base.manifest_path is not None:
                manifest_t0_s, manifest_ids = t0_state_from_manifest(
                    root.base.manifest_path, layout
                )
                root.t0_s = manifest_t0_s
                root.t0_present_ids = manifest_ids
        return self

    def _optional_json(self, path: Path, label: str) -> Optional[dict]:
        if not path.is_file():
            return None
        try:
            return load_json(path)
        except (OSError, ValueError) as exc:
            self.skipped.append(f"{label}: unreadable {path.name} {exc!r}")
            return None


# ------------------------------------------------------- streaming scan
class Tick:
    """One physics step, parsed ONCE into compact parallel arrays.

    ``vids/roads/lanes`` keep the row's own dict order and raw values so
    every consumer reproduces the dict-row iteration semantics exactly;
    ``pos/speed/accel`` are numpy float64 arrays; ``lane_codes`` maps each
    distinct raw lane value to a small int for per-tick vectorisation."""

    __slots__ = (
        "time",
        "dt",
        "vids",
        "roads",
        "lanes",
        "lane_codes",
        "pos",
        "speed",
        "accel",
        "colliding",
        "arrived",
        "cav_actions",
        "cav_ids",
    )

    def __init__(
        self,
        time,
        dt,
        vids,
        roads,
        lanes,
        lane_codes,
        pos,
        speed,
        accel,
        colliding,
        arrived,
        cav_actions,
        cav_ids,
    ):
        self.time = time
        self.dt = dt
        self.vids = vids
        self.roads = roads
        self.lanes = lanes
        self.lane_codes = lane_codes
        self.pos = pos
        self.speed = speed
        self.accel = accel
        self.colliding = colliding
        self.arrived = arrived
        self.cav_actions = cav_actions
        self.cav_ids = cav_ids


class PhysicsScan:
    """A run's whole physics_steps.jsonl, streamed and parsed exactly once.

    ``ticks`` covers every recorded row (window filtering happens later, on
    the in-memory arrays -- never a re-read); ``first_time``/``last_time``
    are the min/max row times; ``cav_ids_all`` unions every row's
    cav_actions keys and cav_ids list (the CAV universe of the run)."""

    __slots__ = ("ticks", "times", "first_time", "last_time", "cav_ids_all")

    def __init__(self, ticks: List[Tick]):
        self.ticks = ticks
        self.times = np.asarray([tick.time for tick in ticks], dtype=float)
        self.first_time = float(self.times.min()) if ticks else None
        self.last_time = float(self.times.max()) if ticks else None
        ids: set = set()
        for tick in ticks:
            ids.update(tick.cav_actions)
            ids.update(tick.cav_ids)
        self.cav_ids_all = tuple(sorted(str(vid) for vid in ids))


def _tick_from_row(row: dict, layout: Layout, lane_codes: Dict[Any, int]) -> Tick:
    vehicles = row.get(layout.row_vehicles_key) or {}
    vids = list(vehicles.keys())
    roads: List[str] = []
    lanes: List[Any] = []
    codes = np.empty(len(vids), dtype=np.int32)
    pos = np.empty(len(vids), dtype=float)
    speed = np.empty(len(vids), dtype=float)
    accel = np.empty(len(vids), dtype=float)
    for i, state in enumerate(vehicles.values()):
        roads.append(str(state.get("road_id", "")))
        lane = state.get(layout.veh_lane_key)
        lanes.append(lane)
        codes[i] = lane_codes.setdefault(lane, len(lane_codes))
        pos[i] = float(state.get(layout.veh_pos_key, 0.0))
        speed[i] = float(state.get(layout.veh_speed_key, 0.0))
        accel[i] = float(state.get(layout.veh_accel_key, 0.0))
    return Tick(
        time=float(row[layout.row_time_key]),
        dt=float(row.get(layout.physics_dt_s, layout.physics_dt_s)),
        vids=vids,
        roads=roads,
        lanes=lanes,
        lane_codes=codes,
        pos=pos,
        speed=speed,
        accel=accel,
        colliding=tuple(row.get(layout.row_colliding_key) or ()),
        arrived=tuple(row.get(layout.row_arrived_key) or ()),
        cav_actions=tuple((row.get(layout.row_cav_key) or {}).keys()),
        cav_ids=tuple(str(vid) for vid in row.get(layout.row_cav_list_key) or ()),
    )


def scan_run_physics(run: RunPaths, layout: Layout) -> PhysicsScan:
    """Stream every physics_steps.jsonl of the run ONCE (line by line, no
    repeated re-reads per metric or provider) into a :class:`PhysicsScan`."""
    ticks: List[Tick] = []
    lane_codes: Dict[Any, int] = {}
    for path in run.physics_paths:
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if line:
                    ticks.append(_tick_from_row(json.loads(line), layout, lane_codes))
    return PhysicsScan(ticks)


def get_scan(run: RunPaths, layout: Layout) -> PhysicsScan:
    """The run's scan, parsed on first use and memoized on the run record
    (both delay providers and every metric share the single parse)."""
    scan = run.scan_cache
    if scan is None:
        scan = scan_run_physics(run, layout)
        run.scan_cache = scan
    return scan


def ticks_from_rows(rows: Sequence[dict], layout: Layout) -> List[Tick]:
    """Compat adapter: dict physics rows -> ticks (tests and old callers)."""
    lane_codes: Dict[Any, int] = {}
    return [_tick_from_row(row, layout, lane_codes) for row in rows]


def window_ticks(ticks: Sequence[Tick], start: float, end: float) -> List[Tick]:
    """Ticks with start - 1e-9 <= time <= end + 1e-9 (the branch_outcome
    window rule, applied to parsed ticks instead of re-read rows)."""
    return [tick for tick in ticks if start - 1e-9 <= tick.time <= end + 1e-9]


_TIME_VALUE = r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
_TIME_RE_CACHE: Dict[str, "re.Pattern[str]"] = {}


def probe_run_times(run: RunPaths, layout: Layout) -> Tuple[float, float]:
    """(first, last) row time WITHOUT a full JSON parse: the first and last
    lines are parsed exactly (physics rows are time-ordered) and a cheap
    regex sweep verifies per-line monotone time order; any violation falls
    back to the exact full scan (memoized, so the parse is never paid
    twice). Used to derive a root's common window before deciding cache
    recomputes."""
    pattern = _TIME_RE_CACHE.get(layout.row_time_key)
    if pattern is None:
        pattern = re.compile(rf'"{re.escape(layout.row_time_key)}"\s*:\s*{_TIME_VALUE}')
        _TIME_RE_CACHE[layout.row_time_key] = pattern
    paths = [Path(p) for p in run.physics_paths]
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            lines = stream.readlines()
    parseable = False
    first = last = None
    if paths:
        for text in (lines[0] if lines else "", lines[-1] if lines else ""):
            if text.strip():
                try:
                    row = json.loads(text)
                    first = float(row[layout.row_time_key])
                    parseable = True
                except (ValueError, KeyError, TypeError):
                    parseable = False
                break
        for text in reversed(lines):
            if text.strip():
                try:
                    last = float(json.loads(text)[layout.row_time_key])
                except (ValueError, KeyError, TypeError):
                    parseable = False
                break
    monotone = parseable
    previous = None
    for text in lines:
        match = pattern.search(text)
        if match is None:
            continue
        value = float(match.group(1))
        if previous is not None and value < previous:
            monotone = False
            break
        previous = value
    if monotone and first is not None and last is not None:
        return first, last
    scan = get_scan(run, layout)
    return scan.first_time, scan.last_time


# --------------------------------------------------------------- outcomes
@dataclass
class BranchOutcome:
    """One arm's delay/safety computation over the declared window."""

    arm: str
    window_s: float
    coverage_s: float
    t0_s: float
    delay_s: float
    entry_wait_s: float
    n_vehicles: int
    collision: bool
    min_ttc_s: Optional[float]
    has_closing_pair: bool
    ttc_lt3_exposure_s: float
    hard_brake_s: float
    n_teleports: int
    delay_provider: str = "builtin"
    edge_change_steps_total: int = 0
    edge_change_steps_bridged: int = 0
    edge_change_steps_skipped: int = 0
    onset_s: Optional[float] = None
    realised_plan: Optional[Dict[str, str]] = None
    is_constant: bool = True
    # PL-02a PILOT: where the constant plan came from --
    # "schedule" | "declared" (branch_record.suffix_controller) | "none"
    realised_source: str = "none"
    invalid_reasons: List[str] = field(default_factory=list)
    # PL-02a-S3: where this arm's teleport-exit ids came from and their count
    exits_source: str = "none"
    n_exits: int = 0
    # PL-02a-C1: the run's CAV universe over ALL rows (cav_actions keys +
    # cav_ids lists) -- lets cache-only report stages rebuild the recorded
    # plan candidates without touching physics files.
    cav_ids_all: Tuple[str, ...] = ()
    # T-A (PL-EXP1b addendum 2): the SAME exposure at the PRIMARY 1.5 s
    # threshold, computed alongside the frozen 3 s one. Additive default
    # keeps cached payloads and every existing caller unchanged.
    ttc_lt1p5_exposure_s: float = 0.0

    @property
    def cell_valid(self) -> bool:
        return not self.invalid_reasons

    def to_dict(self) -> dict:
        """JSON-safe dict of every field (cache serialisation)."""
        payload = asdict(self)
        payload["cav_ids_all"] = list(self.cav_ids_all)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping) -> "BranchOutcome":
        known = {f.name for f in fields(cls)}
        data = {key: value for key, value in payload.items() if key in known}
        data["cav_ids_all"] = tuple(data.get("cav_ids_all") or ())
        if data.get("realised_plan") is not None:
            data["realised_plan"] = {str(k): str(v) for k, v in data["realised_plan"].items()}
        data["invalid_reasons"] = list(data.get("invalid_reasons") or [])
        return cls(**data)


DelayProvider = Callable[[Sequence[Any], float, float, Layout, Mapping], Tuple[float, float, int]]


def branch_outcome_to_dict(outcome: BranchOutcome) -> dict:
    """JSON-safe cache payload for one BranchOutcome."""
    return outcome.to_dict()


def branch_outcome_from_dict(payload: Mapping) -> BranchOutcome:
    """Rebuild a BranchOutcome from its cache payload (exact round-trip)."""
    return BranchOutcome.from_dict(payload)


#: WD-01 part-4 delay library (read-only reference): the DEFAULT
#: DelayProvider. Lazy import through a configurable search path; the
#: built-in provider remains as the cross-check diagnostic.
#: Search root resolved at import from paths.rd_wt_root() (env RD_WT_ROOT,
#: E:/rd-wt default).
P4_SEARCH_PATHS = (str(rd_wt_root() / "wd01c" / "output" / "wd01"),)
P4_DELAY_MODULE = "part4.delay.core"
P4_GEOMETRY_MODULE = "part4.delay.geometry"

#: The frozen network the p4 WM-30 geometry cites. The sibling library's own
#: WM30_NET default is a stale machine path (a removed D: junction), so both
#: call sites below resolve the SAME net through the shared research-dense
#: assets root (env RESEARCH_DENSE_ROOT, E: default -- see paths.py).


def _p4_net_path() -> str:
    return str(
        research_dense_root() / "assets" / "research"
        / "highway_merge_shepherd_long" / "osm.net.xml.gz"
    )


class P4DelayUnavailable(RuntimeError):
    """The WD-01 part-4 delay library is not importable from the configured
    search paths (precise reason carried in the message)."""


def road_lengths_from_p4(search_paths=P4_SEARCH_PATHS, geometry=None) -> Dict[str, float]:
    """SUMO edge id -> lane length, read from the WD-01 p4 geometry chains
    (read-only citation of the frozen network's lane table).

    Feeds ``Layout.road_length_by_id`` so the builtin provider can BRIDGE
    edge-change advances instead of skipping them: a skipped step removes
    real travel from the free-flow integral and biases the delay HIGH.
    Edges shared by several physical lanes take the first chain's length
    (WM-30 per-lane differences are <= 1.45 m -- the declared bridge
    residual)."""
    _core, geometry_module = load_p4_delay(search_paths)
    geom = (
        geometry if geometry is not None
        else geometry_module.wm30_geometry(net_path=_p4_net_path())
    )
    lengths: Dict[str, float] = {}
    for _lane, segs in geom.segments.items():
        for seg in segs:
            lengths.setdefault(str(seg.edge), float(seg.s1) - float(seg.s0))
    return lengths


def load_p4_delay(search_paths=P4_SEARCH_PATHS):
    """Import (core, geometry) of the p4 delay library, lazily.

    Temporarily inserts the search paths into sys.path; the imported
    modules stay cached (their submodule names are part4.*)."""
    import sys

    added = [str(p) for p in search_paths if str(p) not in sys.path]
    for path in added:
        sys.path.insert(0, path)
    try:
        core = __import__(f"{P4_DELAY_MODULE}", fromlist=("core",))
        geometry = __import__(P4_GEOMETRY_MODULE, fromlist=("geometry",))
    except ImportError as exc:
        raise P4DelayUnavailable(
            f"WD-01 p4 delay library ({P4_DELAY_MODULE}) not importable "
            f"from search paths {list(search_paths)}: {exc!r}"
        ) from exc
    finally:
        for path in added:
            sys.path.remove(path)
    return core, geometry


def p4_delay_provider(search_paths=P4_SEARCH_PATHS, geometry=None, strict=True) -> DelayProvider:
    """Default provider built on the WD-01 part-4 delay library
    (advance-based free-flow, section 8.14 entry waits). Raises
    :class:`P4DelayUnavailable` when the library is absent."""
    core, geometry_module = load_p4_delay(search_paths)
    geom = (
        geometry if geometry is not None
        else geometry_module.wm30_geometry(net_path=_p4_net_path())
    )
    lane_chains = getattr(geometry_module, "WM30_CHAINS", None)

    def provider(rows, t0_s, window_s, layout, departs, t0_present_ids=frozenset()):
        return p4_delay(
            rows,
            t0_s,
            window_s,
            layout,
            departs,
            core,
            geom,
            strict=strict,
            t0_present_ids=t0_present_ids,
            lane_chains=lane_chains,
        )

    return provider


# WM-30 SUMO lane id -> physical lane (0 main_left / 1 main_right / 2 accel),
# from the WD-01 part-4 geometry chains (read-only citation; the p2C physics
# rows carry SUMO lane ids, not physical ints).
WM30_LANE_TO_PHYSICAL = {
    "1191554636_1": 0,
    "1487606179_2": 0,
    "1487606178_1": 0,
    "1191554636_0": 1,
    "1487606179_1": 1,
    "1487606178_0": 1,
    "11553048#1_0": 2,
    "1487606179_0": 2,
}


def physical_lane(lane_value) -> int:
    """SUMO lane id (or int) -> WM-30 physical lane; unknown/junction -> -1.
    Note: int() accepts '1_0' as 10 (underscore literals), so SUMO lane ids
    must go through the explicit mapping, never int()."""
    text = str(lane_value)
    if text.isdigit():
        return int(text)
    return WM30_LANE_TO_PHYSICAL.get(text, -1)


def _chain_tables(geom, chains=None):
    """Chain-coordinate tables for the p4 adapter (PL-02a-S3).

    The campaign physics rows are per-SUMO-edge: ``lane_position_m`` resets
    at every chain edge and internal junction edges appear as road ids.
    Feeding those raw positions into the p4 chain-coordinate primitives
    produces phantom hundreds-of-metre "junction" advances (measured on
    pilot r3: n=1036, mean 290 m), which was the dominant term of the old
    p4-vs-builtin gap.  The fix maps every record onto the physical lane's
    chain first: s_chain = seg.s0 + lane_position_m.

    Returns (lane_id -> physical lane, (physical lane, edge) -> (s0, len));
    ``chains`` is the geometry module's WM30_CHAINS.  Records whose edge is
    not part of a chain (synthetic tests, other nets) keep the raw position
    on the physical lane, preserving the pre-S3 behaviour."""
    lane_phys: Dict[str, int] = {}
    if chains:
        for lane, (_name, lane_ids) in chains.items():
            for lane_id in lane_ids:
                lane_phys[str(lane_id)] = int(lane)
    if not lane_phys:
        lane_phys = dict(WM30_LANE_TO_PHYSICAL)
    chain_seg: Dict[Tuple[int, str], Tuple[float, float]] = {}
    for lane, segs in geom.segments.items():
        for seg in segs:
            chain_seg[(int(lane), str(seg.edge))] = (
                float(seg.s0),
                float(seg.s1) - float(seg.s0),
            )
    return lane_phys, chain_seg


def p4_delay(
    rows,
    t0_s,
    window_s,
    layout,
    departs,
    core,
    geom,
    strict=True,
    t0_present_ids=frozenset(),
    lane_chains=None,
):
    """Adapt the recorded physics rows to the p4 accounting:
    VehicleTrace per vehicle -> vehicle_delay -> waiting_component (§8.14)
    -> window_totals (advance-based free-flow reference).

    PL-02a-S3: records are mapped onto the physical lane's CHAIN coordinate
    before the trace is built (see :func:`_chain_tables`); junction-internal
    records (no chain position) are resolved to the nearest resolved
    neighbour -- the junction's internal length is ~0 in the p4 convention.

    PL-02a-C1: ``rows`` may be dict rows (compat) or pre-parsed
    :class:`Tick` objects from the run's single streaming scan; both feed
    the identical trace construction in identical order."""
    lane_phys, chain_seg = _chain_tables(geom, lane_chains)
    ticks = rows if (rows and isinstance(rows[0], Tick)) else ticks_from_rows(rows, layout)
    window_end = t0_s + window_s
    kept = [tick for tick in ticks if t0_s - 1e-9 <= tick.time <= window_end + 1e-9]
    traces: Dict[str, dict] = {}
    first_seen: Dict[str, float] = {}
    for tick in kept:
        now = tick.time
        for i, veh in enumerate(tick.vids):
            trace = traces.setdefault(veh, {"t": [], "lane": [], "s": [], "raw": []})
            road = tick.roads[i]
            phys = lane_phys.get(str(tick.lanes[i]))
            if phys is None:
                phys = physical_lane(tick.lanes[i])
            seg = chain_seg.get((phys, road)) if phys != core.JUNCTION else None
            pos = float(tick.pos[i])
            if seg is not None:
                s0, seg_len = seg
                s_chain: Optional[float] = s0 + min(max(pos, 0.0), seg_len)
            elif phys != core.JUNCTION and phys in geom.segments:
                s_chain = pos  # not a chain edge: keep the raw position
            else:
                s_chain = None  # junction-internal: resolved by neighbours
            trace["t"].append(now)
            trace["lane"].append(phys)
            trace["s"].append(s_chain)
            trace["raw"].append(pos)
            if veh not in first_seen or now < first_seen[veh]:
                first_seen[veh] = now
    for trace in traces.values():
        s_vals = trace["s"]
        last = None
        for i, val in enumerate(s_vals):  # forward fill from the previous
            if val is None:
                s_vals[i] = last
            else:
                last = val
        nxt = None
        for i in range(len(s_vals) - 1, -1, -1):  # then from the next
            if s_vals[i] is None:
                s_vals[i] = nxt
            else:
                nxt = s_vals[i]
        if s_vals[0] is None:
            # no chain edge anywhere in the trace (synthetic/other net):
            # legacy behaviour -- raw per-edge positions on junction records
            trace["s"] = trace["raw"]
    traces = {vid: data for vid, data in traces.items() if data["s"][0] is not None}
    vehicles = []
    for vid in sorted(traces):
        data = traces[vid]
        trace = core.VehicleTrace(
            vid=vid,
            t=np.asarray(data["t"], dtype=float),
            lane=np.asarray(data["lane"], dtype=np.int64),
            s=np.asarray(data["s"], dtype=float),
        )
        vehicles.append(
            core.vehicle_delay(trace, t0_s, window_s, geom, layout.physics_dt_s, strict=strict)
        )
    schedule = {
        vid: (float(depart), core.ENTRANCE_MAINLINE)
        for vid, depart in departs.items()
        if depart < window_end
    }
    # §8.14: vehicles present at t0 were inserted at/before t0 -> no wait
    insertion_map = dict(first_seen)
    for vid in t0_present_ids or (()):
        insertion_map.setdefault(vid, t0_s)
    waits, _ = core.waiting_component(schedule, insertion_map, t0_s, window_s)
    totals = core.window_totals(vehicles, waits, t0_s, window_s)
    edge_diag = {
        "edge_change_steps_total": sum(v.n_boundary_change_steps for v in vehicles),
        "edge_change_steps_bridged": sum(v.n_boundary_change_steps for v in vehicles),
        "edge_change_steps_skipped": 0,
    }
    # PL-02a-S3: per-term sub-totals for the p4-vs-builtin decomposition
    edge_diag.update(
        {
            "time_on_section_s": float(totals["time_on_section_s"]),
            "free_flow_ref_s": float(totals["free_flow_ref_s"]),
            "advance_m": float(totals["traversed_m"]),
            "wait_mainline_s": float(totals["wait_mainline_s"]),
            "wait_ramp_s": float(totals["wait_ramp_s"]),
        }
    )
    return (
        float(totals["total_delay_ref_s"]),
        float(totals["wait_mainline_s"] + totals["wait_ramp_s"]),
        int(totals["n_vehicles"]),
        edge_diag,
    )


def _read_departs(rou_paths: Sequence[Path]) -> Dict[str, float]:
    """vehicle id -> scheduled depart (requests.rou.xml @depart)."""
    departs: Dict[str, float] = {}
    for path in rou_paths:
        try:
            root = ET.parse(path).getroot()
        except (OSError, ET.ParseError):
            continue
        for vehicle in root.iter("vehicle"):
            vid = vehicle.get("id")
            depart = vehicle.get("depart")
            if vid and depart is not None:
                try:
                    departs[vid] = float(depart)
                except ValueError:
                    continue
    return departs


#: SUMO's default passenger vType length (metres). Named, documented
#: constant for the opt-in TRUE-GAP TTC mode (gap #18, PL-EXP1b): a vType
#: without a ``length`` attribute in the scenario's requests.rou.xml has
#: this length. The current campaign vTypes (normal_car, cautious_car,
#: aggressive_car) declare no length, so this constant applies to them.
SUMO_DEFAULT_VEHICLE_LENGTH_M = 5.0


#: T-A (PL-EXP1b addendum 2): the PRIMARY relative-TTC exposure threshold
#: (seconds). Computed alongside the frozen 3 s ``layout.ttc_threshold_s``
#: exposure everywhere the safety dict is built; never replaces it.
TTC_EXPOSURE_PRIMARY_S = 1.5


class TrueGapLengthError(ValueError):
    """Fail-closed error of the opt-in TRUE-GAP TTC mode: a vehicle's
    type_id (or the vehicle id itself) cannot be resolved to a declared
    length from the run's scenario route files. The DEFAULT
    front-bumper-difference TTC never raises this."""


def vtype_lengths_from_rou(rou_paths: Sequence[Path]) -> Dict[str, float]:
    """vType id -> declared length (m) from the scenario route files
    (``<vType id=... length=...>``). A vType WITHOUT a ``length`` attribute
    gets :data:`SUMO_DEFAULT_VEHICLE_LENGTH_M` (SUMO's passenger default;
    declared, not guessed)."""
    lengths: Dict[str, float] = {}
    for path in rou_paths:
        try:
            root = ET.parse(path).getroot()
        except (OSError, ET.ParseError):
            continue
        for vtype in root.iter("vType"):
            type_id = vtype.get("id")
            if not type_id:
                continue
            raw = vtype.get("length")
            try:
                lengths[type_id] = (
                    SUMO_DEFAULT_VEHICLE_LENGTH_M
                    if raw is None
                    else float(raw)
                )
            except ValueError:
                continue
    return lengths


def true_gap_lengths_from_rou(rou_paths: Sequence[Path]) -> Dict[str, float]:
    """vehicle id -> length (m) for the TRUE-GAP TTC, resolved through the
    vehicle's ``type`` attribute against the scenario's vType definitions.
    A vehicle WITHOUT a ``type`` attribute is the SUMO default passenger
    type (:data:`SUMO_DEFAULT_VEHICLE_LENGTH_M`). FAILS CLOSED: a vehicle
    whose type_id is not declared in the route files raises
    :class:`TrueGapLengthError` (a vehicle absent from the route files is
    simply not in the map; the consumer raises on it)."""
    lengths = vtype_lengths_from_rou(rou_paths)
    per_vehicle: Dict[str, float] = {}
    for path in rou_paths:
        try:
            root = ET.parse(path).getroot()
        except (OSError, ET.ParseError):
            continue
        for vehicle in root.iter("vehicle"):
            vid = vehicle.get("id")
            if not vid:
                continue
            type_id = vehicle.get("type")
            if type_id is None:
                per_vehicle[vid] = SUMO_DEFAULT_VEHICLE_LENGTH_M
                continue
            length = lengths.get(type_id)
            if length is None:
                raise TrueGapLengthError(
                    f"vehicle {vid!r} references type_id {type_id!r}, which is not "
                    f"declared as a vType in {path} (true-gap TTC fails closed)"
                )
            per_vehicle[vid] = length
    return per_vehicle


def _receipt_t0_s(receipt: Mapping, layout: Layout) -> Optional[float]:
    """The receipt's t0: restores_state.at_s, else branch_point_t0_s."""
    restored = receipt.get("restores_state") or {}
    for key in ("at_s", layout.manifest_t0_key):
        if key in restored:
            return float(restored[key])
    if layout.manifest_t0_key in receipt:
        return float(receipt[layout.manifest_t0_key])
    return None


def t0_state_from_manifest(manifest_path: Path, layout: Layout) -> Tuple[float, frozenset]:
    """(t0_s, ids inserted at/before t0) from the saved-state manifest.

    t0 is ``manifest[manifest_t0_key]`` (branch_point_t0_s) -- never a
    hard-coded 60.0 -- and the inserted ids are
    ``manifest[manifest_t0_row_key][manifest_live_key]`` (physics_row_at_t0
    .live_ids)."""
    manifest = load_json(manifest_path)
    t0 = float(manifest[layout.manifest_t0_key])
    row = manifest.get(layout.manifest_t0_row_key) or {}
    ids = frozenset(str(v) for v in row.get(layout.manifest_live_key) or ())
    return t0, ids


def _exits_from_rows(
    rows: Union[Sequence[dict], Sequence[Tick]], layout: Layout
) -> Tuple[set, str]:
    """Per-arm exit ids from the arm's OWN physics rows (a sibling's exit
    list is never borrowed -- it is valid only for identical-action
    continuations and wrong for native/pi branches).

    Named exits: the ids in each row's ``row_arrived_key`` (arrived_ids).
    Position fallback: a vehicle whose LAST sighting sits in the exit zone
    (position >= section_len_m - exit_band_m on ``exit_zone_road_ids``;
    None = any road).  Returns (exit ids, source) with source one of
    {"physics_rows", "derived_positions", "none"}. Accepts dict rows
    (compat) or parsed ticks."""
    ticks = rows if (rows and isinstance(rows[0], Tick)) else ticks_from_rows(rows, layout)
    named: set = set()
    last_pos: Dict[str, float] = {}
    last_road: Dict[str, str] = {}
    for tick in ticks:
        for vid in tick.arrived:
            named.add(str(vid))
        for i, veh in enumerate(tick.vids):
            last_pos[veh] = float(tick.pos[i])
            last_road[veh] = tick.roads[i]
    floor_pos = layout.section_len_m - layout.exit_band_m
    roads = layout.exit_zone_road_ids
    zone = {
        veh
        for veh, pos in last_pos.items()
        if pos >= floor_pos and (roads is None or last_road[veh] in roads)
    }
    if named:
        return named | zone, "physics_rows"
    if zone:
        return zone, "derived_positions"
    return set(), "none"


def default_delay(
    rows: Sequence[dict],
    t0_s: float,
    window_s: float,
    layout: Layout,
    departs: Mapping[str, float],
    t0_present_ids: Optional[set] = None,
) -> Tuple[float, float, int, Dict[str, int]]:
    """Built-in DelayProvider (CROSS-CHECK diagnostic; the default is the
    WD-01 p4 provider). Contract §4.1 + §8.14 semantics:

    - time on section: sum of 0.5 s physics-step coverage inside the window;
    - free-flow time: ADVANCE-BASED (PL-02a2) - the distance actually
      travelled inside the window divided by ``free_flow_mps`` - NOT a
      full-section free-flow subtraction per vehicle. A lane_position_m
      RESET at an edge boundary must not count as travel: deltas are summed
      only between consecutive rows on the SAME road_id+lane_id, and for a
      road change the advance is bridged from the configured edge lengths
      ((len(a) - pos_a) + pos_b) when ``layout.road_length_by_id`` knows
      both roads, else the step is SKIPPED and counted in the returned
      diagnostic (declared);
    - entry waits (§8.14): wait = clip(min(t_insert, window end)
      - max(t_depart, t0), 0, inf). A vehicle scheduled before t0 but not
      inserted at t0 accrues wait FROM t0 until insertion or window end; a
      vehicle inserted at or before t0 contributes no wait; a vehicle
      scheduled in-window and never inserted accrues to window end.

    Returns (delay_s, entry_wait_s, n_vehicles_present, diagnostics) with
    diagnostics {"edge_change_steps_skipped": int,
    "edge_change_steps_bridged": int, "edge_change_steps": int}.

    PL-02a-C1: ``rows`` may be dict rows (compat) or pre-parsed
    :class:`Tick` objects from the run's single streaming scan; iteration
    order and arithmetic are identical in both forms."""
    diagnostics = {
        "edge_change_steps_skipped": 0,
        "edge_change_steps_bridged": 0,
        "edge_change_steps_total": 0,
    }
    ticks = rows if (rows and isinstance(rows[0], Tick)) else ticks_from_rows(rows, layout)
    end = t0_s + window_s
    presence: Dict[str, float] = {}
    first_seen: Dict[str, float] = {}
    last_spot: Dict[str, Tuple[str, str, float]] = {}
    advance_m = 0.0
    for tick in ticks:
        now = tick.time
        step_start = max(now, t0_s)
        step_end = min(now + tick.dt, end)
        if step_end <= step_start:
            continue
        dt = step_end - step_start
        for i, veh in enumerate(tick.vids):
            presence[veh] = presence.get(veh, 0.0) + dt
            road = tick.roads[i]
            lane = tick.lanes[i]
            pos = float(tick.pos[i])
            previous = last_spot.get(veh)
            if previous is None:
                first_seen[veh] = now  # insertion record: a point, no step
            else:
                prev_road, prev_lane, prev_pos = previous
                diagnostics["edge_change_steps_total"] += 1
                if road == prev_road:
                    # same road (a lane change keeps the edge coordinate)
                    advance_m += abs(pos - prev_pos)
                else:
                    # edge change: bridge with the declared edge lengths
                    lengths = layout.road_length_by_id or {}
                    len_a = lengths.get(prev_road)
                    len_b = lengths.get(road)
                    if len_a is not None and len_b is not None:
                        advance_m += (float(len_a) - prev_pos) + pos
                        diagnostics["edge_change_steps_bridged"] += 1
                    else:
                        diagnostics["edge_change_steps_skipped"] += 1
            last_spot[veh] = (road, lane, pos)
    waits = 0.0
    present_at_t0 = set(t0_present_ids or ())
    for veh, depart in departs.items():
        if depart >= end:
            continue
        floor = max(depart, t0_s)
        if veh in present_at_t0:
            continue  # PL-02a-S2: inserted at/before t0 -> no entry wait
        seen = first_seen.get(veh)
        waits += (end - floor) if seen is None else max(0.0, seen - floor)
    delay = sum(presence.values()) - advance_m / layout.free_flow_mps + waits
    # PL-02a-S3: per-term sub-totals for the p4-vs-builtin decomposition
    # (the returned delay is exactly time_on_section - free_flow + waits).
    diagnostics["time_on_section_s"] = sum(presence.values())
    diagnostics["free_flow_s"] = advance_m / layout.free_flow_mps
    diagnostics["advance_m"] = advance_m
    diagnostics["entry_wait_s"] = waits
    diagnostics["n_vehicles"] = len(presence)
    return delay, waits, len(presence), diagnostics


def _safety_from_rows(rows: Sequence[dict], layout: Layout,
                      true_gap_lengths: Optional[Mapping[str, float]] = None) -> dict:
    """min closing TTC per lane, has_closing_pair, TTC<3 exposure
    (vehicle-seconds), hard-brake seconds, collision flag (colliding_ids).
    Dict-row compat wrapper for :func:`_safety_from_ticks`."""
    return _safety_from_ticks(ticks_from_rows(rows, layout), layout, true_gap_lengths)


def _safety_from_ticks(ticks: Sequence[Tick], layout: Layout,
                       true_gap_lengths: Optional[Mapping[str, float]] = None) -> dict:
    """Vectorised safety metrics: per tick, vehicles are sorted by lane
    position ONCE (stable lexsort by (lane code, -position)) and the
    follower->leader closing TTC, TTC<threshold exposure and hard-brake
    seconds are computed with numpy. Same pairs, same arithmetic and same
    minima as the old per-vehicle dict loop.

    TRUE GAP (opt-in, gap #18 / PL-EXP1b): when ``true_gap_lengths``
    (vehicle id -> length in m, from the run's scenario vType definitions)
    is given, the pair gap is the BUMPER-TO-BUMPER gap
    ``leader_pos - leader_length - follower_pos`` -- what SUMO's
    getLeader/getMinGap report. The DEFAULT (``None``) keeps the historic
    front-bumper-difference gap (``leader_pos - follower_pos``), which
    includes the leader's length and biases TTC HIGH; default behaviour is
    unchanged. A leader id missing from the mapping raises
    :class:`TrueGapLengthError` (fail closed, never imputed)."""
    min_ttc: Optional[float] = None
    has_pair = False
    exposure = 0.0
    # T-A: the primary 1.5 s exposure accumulator (see _safety_from_ticks)
    exposure_lt1p5 = 0.0
    hard_brake = 0.0
    collisions: set = set()
    for tick in ticks:
        dt = tick.dt
        hard_brake += dt * int(np.count_nonzero(tick.accel < layout.hard_brake_mps2))
        if tick.vids:
            # front-first order within each lane, lanes grouped (one sort)
            order = np.lexsort((-tick.pos, tick.lane_codes))
            codes = tick.lane_codes[order]
            boundaries = np.flatnonzero(np.diff(codes)) + 1
            starts = np.concatenate(([0], boundaries))
            ends = np.concatenate((boundaries, [codes.size]))
            for start, end in zip(starts, ends):
                if end - start < 2:
                    continue  # a lone vehicle has no leader
                idx = order[start:end]
                leader_pos = tick.pos[idx[:-1]]
                follower_pos = tick.pos[idx[1:]]
                gap = leader_pos - follower_pos
                if true_gap_lengths is not None:
                    leader_lengths = np.empty(gap.shape, dtype=float)
                    for j, leader_i in enumerate(idx[:-1]):
                        length = true_gap_lengths.get(tick.vids[leader_i])
                        if length is None:
                            raise TrueGapLengthError(
                                f"vehicle {tick.vids[leader_i]!r} has no declared "
                                "length (true-gap TTC fails closed)"
                            )
                        leader_lengths[j] = float(length)
                    gap = gap - leader_lengths
                closing = tick.speed[idx[1:]] - tick.speed[idx[:-1]]
                mask = closing > 1e-6
                if not mask.any():
                    continue
                ttc = np.maximum(gap, 0.0)[mask] / closing[mask]
                has_pair = True
                tick_min = float(ttc.min())
                min_ttc = tick_min if min_ttc is None else min(min_ttc, tick_min)
                exposure += dt * int(np.count_nonzero(ttc < layout.ttc_threshold_s))
                # T-A: the SAME pair set at the PRIMARY 1.5 s threshold
                # (always computed; the frozen 3 s meaning is unchanged)
                exposure_lt1p5 += dt * int(np.count_nonzero(ttc < TTC_EXPOSURE_PRIMARY_S))
        collisions.update(tick.colliding)
    return {
        "min_ttc_s": min_ttc,
        "has_closing_pair": has_pair,
        "ttc_lt3_exposure_s": exposure,
        "ttc_lt1p5_exposure_s": exposure_lt1p5,
        "hard_brake_s": hard_brake,
        "collisions": sorted(collisions),
    }


def per_lane(vehicles: Mapping[str, Mapping], layout: Layout, lane) -> List[str]:
    """Vehicle ids on ``lane`` ordered front-first (descending position)."""
    on_lane = [
        (vid, float(state[layout.veh_pos_key]))
        for vid, state in vehicles.items()
        if state.get(layout.veh_lane_key) == lane
    ]
    return [vid for vid, _ in sorted(on_lane, key=lambda item: -item[1])]


def _teleports(
    rows: Union[Sequence[dict], Sequence[Tick]],
    layout: Layout,
    exit_ids: set,
    window_end: float,
) -> Tuple[int, List[str]]:
    """PL-02a-S2: PER-VEHICLE teleport candidates (each vehicle counts once).

    A vehicle is a teleport candidate iff it disappears from the physics
    rows before the window end AND it is NOT in the run's exit list AND its
    last position is not on a network-exit edge (short of the exit band);
    OR a position jump beyond ``max_position_jump_m`` between consecutive
    rows on the SAME road_id+lane_id (lane_position resets at road
    boundaries, so cross-road deltas are never jumps).

    Returns (n_candidates, candidate ids sorted). Accepts dict rows (compat)
    or parsed ticks."""
    ticks = rows if (rows and isinstance(rows[0], Tick)) else ticks_from_rows(rows, layout)
    last_pos: Dict[str, float] = {}
    last_time: Dict[str, float] = {}
    last_road: Dict[str, str] = {}
    last_lane: Dict[Any, Any] = {}
    jumpers: set = set()
    for tick in ticks:
        now = tick.time
        for i, veh in enumerate(tick.vids):
            pos = float(tick.pos[i])
            road = tick.roads[i]
            lane = tick.lanes[i]
            if veh in last_pos:
                same_lane = last_road[veh] == road and last_lane[veh] == lane
                if same_lane and abs(pos - last_pos[veh]) > layout.max_position_jump_m:
                    jumpers.add(veh)
            last_pos[veh] = pos
            last_time[veh] = now
            last_road[veh] = road
            last_lane[veh] = lane
    vanished = {
        veh
        for veh, seen_until in last_time.items()
        if seen_until < window_end - 1e-9
        and veh not in exit_ids
        and last_pos[veh] < layout.section_len_m - layout.exit_band_m
    }
    candidates = sorted(jumpers | vanished)
    return len(candidates), candidates


def _onset_s(rows: Union[Sequence[dict], Sequence[Tick]], layout: Layout) -> Optional[float]:
    """Declared congestion onset: the first in-window time at which some
    lane's mean speed stays below onset_speed_share * free-flow for
    onset_persist_s of consecutive physics rows; None if never. Accepts
    dict rows (compat) or parsed ticks."""
    ticks = rows if (rows and isinstance(rows[0], Tick)) else ticks_from_rows(rows, layout)
    free_flow = layout.free_flow_mps
    threshold = layout.onset_speed_share * free_flow
    run_start = None
    for tick in ticks:
        n_lanes = int(tick.lane_codes.max()) + 1 if tick.vids else 0
        slow = False
        if n_lanes:
            sums = np.bincount(tick.lane_codes, weights=tick.speed, minlength=n_lanes)
            counts = np.bincount(tick.lane_codes, minlength=n_lanes)
            means = sums[counts > 0] / counts[counts > 0]
            slow = bool((means < threshold).any())
        if slow:
            run_start = tick.time if run_start is None else run_start
            if tick.time - run_start >= layout.onset_persist_s:
                return run_start
        else:
            run_start = None
    return None


def branch_outcome(
    run: RunPaths,
    *,
    t0_s: float,
    window_s: float,
    layout: Optional[Layout] = None,
    delay_provider: Optional[DelayProvider] = None,
    departs: Optional[Mapping[str, float]] = None,
    provider_name: str = "builtin",
    window_start_s: Optional[float] = None,
    exit_ids: Optional[set] = None,
    t0_present_ids: Optional[set] = None,
    true_gap_lengths: Optional[Mapping[str, float]] = None,
) -> BranchOutcome:
    """One arm's delay/safety computation over [t0, t0 + window_s].

    PL-02a-C1: the run's physics files are streamed and parsed exactly once
    (memoized on the run record); the window is applied to the parsed ticks
    and BOTH delay providers, safety, teleports, onset and exits consume
    the same parsed arrays.

    ``true_gap_lengths`` (vehicle id -> length in m) opts into the TRUE
    bumper-to-bumper TTC gap (see :func:`_safety_from_ticks`); ``None``
    keeps the default front-bumper-difference TTC unchanged."""
    layout = layout or Layout()
    scan = get_scan(run, layout)
    window_start = t0_s if window_start_s is None else float(window_start_s)
    ticks = window_ticks(scan.ticks, window_start, window_start + window_s)
    provider = delay_provider or default_delay
    departs = departs if departs is not None else _read_departs(run.rou_paths)
    delay, waits, n_vehicles, edge_diag = provider(
        ticks,
        t0_s=window_start,
        window_s=window_s,
        layout=layout,
        departs=departs,
        t0_present_ids=t0_present_ids,
    )
    safety = _safety_from_ticks(ticks, layout, true_gap_lengths)
    # PL-02a-S3: per-arm exit identity.  An explicit ``exit_ids`` override
    # wins (tests/back-compat); otherwise the run's OWN arrivals file, and
    # otherwise the arm's own physics rows (arrived_ids + exit-zone rule).
    # A sibling's exit list is never borrowed.
    if exit_ids is not None:
        exits, exits_source = set(exit_ids), "explicit"
    elif run.arrivals_path is not None:
        exits = {
            str(row.get(layout.arrivals_vehicle_key))
            for row in load_jsonl(run.arrivals_path)
            if row.get(layout.arrivals_vehicle_key)
        }
        exits_source = "arrivals_file"
    else:
        exits, exits_source = _exits_from_rows(ticks, layout)
    n_candidates, candidates = _teleports(ticks, layout, exits, t0_s + window_s)
    invalid: List[str] = []
    if candidates:
        invalid.append("teleport candidates: " + ",".join(candidates))
    if layout.teleport_policy == "fail_closed_all":
        invalid.append("teleport_policy=fail_closed_all")
    collisions = set(safety["collisions"])
    # PL-02a-C2: the outcome-record ``collisions`` field has TWO recorded
    # forms -- an id LIST (wm44 branch outcome) and a scalar COUNT (wm44
    # base outcome: {"collisions": 0}). The old reader wrapped ANY non-None
    # scalar as a one-element id list, so every base run recording a zero
    # collision count was flagged collision=True ("0" became a collision
    # id) even when its physics rows carry no colliding_ids at all (real
    # case: w8407m2400p02t360__base, min closing TTC 2.20 s, 0 colliding
    # rows). A count of 0 means NO collision; a POSITIVE count means
    # collisions happened whose ids are not recorded -> fail closed.
    outcome_count_note: Optional[str] = None
    if run.outcome_path is not None:
        record = load_json(run.outcome_path)
        recorded_collisions = record.get(layout.outcome_collisions_key)
        if isinstance(recorded_collisions, (list, tuple)):
            collisions.update(str(c) for c in recorded_collisions)
        elif isinstance(recorded_collisions, str):
            if recorded_collisions:
                collisions.add(recorded_collisions)
        elif isinstance(recorded_collisions, (int, float)) and not isinstance(
            recorded_collisions, bool
        ):
            if float(recorded_collisions) > 0:
                outcome_count_note = (
                    f"outcome_record collision count {recorded_collisions:g} > 0 "
                    "(collision ids not recorded; failed closed)"
                )
                collisions.add(str(recorded_collisions))
        recorded_delay = record.get(layout.outcome_delay_key)
        if (
            isinstance(recorded_delay, (int, float))
            and abs(float(recorded_delay) - delay) > layout.crosscheck_delay_tol_s
        ):
            invalid.append(
                f"cross-check: outcome delay {recorded_delay} vs recomputed "
                f"{delay:.3f} beyond tolerance {layout.crosscheck_delay_tol_s}s"
            )
    collision = bool(collisions)
    if outcome_count_note is not None:
        invalid.append(outcome_count_note)
    if collision:
        invalid.append("collision in window")

    realised: Optional[Dict[str, str]] = None
    is_constant = True
    realised_source = "none"
    if run.schedule_path is not None:
        schedule = load_json(run.schedule_path)
        if layout.schedule_container_key:
            # PL-02a CHUNK-1: the payload lives nested (switch_schedule.json
            # carries run/arm/branch_seed wrappers around the schedule)
            schedule = schedule.get(layout.schedule_container_key) or {}
        events = sorted(
            schedule.get(layout.schedule_events_key, []),
            key=lambda e: float(e.get(layout.schedule_time_key, 0.0)),
        )
        if events:
            # constant iff exactly ONE distinct event time (chunk-1 fixed
            # arms emit one ROW per CAV, all at t0). A schedule with many
            # event times is a controller TIMELINE: the arm is headroom-only
            # -- UNLESS branch_type maps the arm to a constant action
            # (chunk-1 native/pi: their timeline logs controller handoffs,
            # but the arm IS a declared constant plan; the dispatcher
            # cross-check applies to FIXED arms only).
            is_constant = len({float(e.get(layout.schedule_time_key, 0.0)) for e in events}) == 1
            payload = events[0].get(layout.schedule_plan_key) or {}
            if not payload and layout.schedule_plan_key == "assignments" and is_constant:
                # chunk-1 per-VEHICLE event rows: rebuild the t0 payload from
                # the events themselves (fixed arms emit one row per CAV, all
                # at the same time)
                payload = {
                    e.get("vehicle_id", e.get(layout.schedule_time_key)): e.get("action")
                    for e in events
                    if abs(float(e.get(layout.schedule_time_key, 0.0)) - t0_s) <= 1e-9
                }
            realised = {str(c): str(a) for c, a in payload.items()}
            realised_source = "schedule"
            if realised is not None and run.dispatcher_path is not None and is_constant:
                invalid.extend(_dispatcher_mismatches(run, layout, realised, t0_s))
        if not is_constant:
            # declared constant plan from branch_type (native/pi); switching
            # types are unmapped -> headroom only, never a candidate
            branch_type = (run.branch_record or {}).get(layout.branch_type_key)
            controller = (layout.branch_type_actions or {}).get(branch_type)
            if controller:
                realised = {cav: str(controller) for cav in _cav_ids_from_rows(ticks, layout)}
                is_constant = True
                realised_source = "declared"
    elif run.branch_record is not None:
        # PL-02a PILOT: no switch_schedule exists (the pilot predates
        # dispatcher payload logs), so the constant plan is the arm's
        # DECLARED controller: branch_record[suffix_controller] applied to
        # every CAV seen in the run's physics rows.  The realised-action
        # cross-check is N/A here (physics cav_actions log control
        # ACTIVITY, not the plan payload); the verify arm's exactness
        # verdict is the integrity gate instead.
        controller = run.branch_record.get(layout.branch_controller_key)
        if not controller:
            # chunk-1 branch_record carries branch_type (native/pi/...)
            # instead of suffix_controller
            branch_type = run.branch_record.get(layout.branch_type_key)
            controller = (layout.branch_type_actions or {}).get(branch_type)
        cav_ids = _cav_ids_from_rows(ticks, layout)
        if controller:
            realised = {cav: str(controller) for cav in cav_ids}
            realised_source = "declared"
    return BranchOutcome(
        arm=run.arm,
        delay_provider=provider_name,
        window_s=window_s,
        coverage_s=window_s,
        t0_s=t0_s,
        delay_s=delay,
        entry_wait_s=waits,
        n_vehicles=n_vehicles,
        collision=collision,
        min_ttc_s=safety["min_ttc_s"],
        has_closing_pair=safety["has_closing_pair"],
        ttc_lt3_exposure_s=safety["ttc_lt3_exposure_s"],
        ttc_lt1p5_exposure_s=safety["ttc_lt1p5_exposure_s"],
        hard_brake_s=safety["hard_brake_s"],
        n_teleports=n_candidates,
        edge_change_steps_total=int(edge_diag.get("edge_change_steps_total", 0)),
        edge_change_steps_bridged=int(edge_diag.get("edge_change_steps_bridged", 0)),
        edge_change_steps_skipped=int(edge_diag.get("edge_change_steps_skipped", 0)),
        onset_s=_onset_s(ticks, layout),
        realised_plan=realised,
        is_constant=is_constant,
        realised_source=realised_source,
        invalid_reasons=invalid,
        exits_source=exits_source,
        n_exits=len(exits),
        cav_ids_all=scan.cav_ids_all,
    )


def _dispatcher_mismatches(
    run: RunPaths, layout: Layout, realised: Mapping[str, str], t0_s: float
) -> List[str]:
    """Realised-action cross-check for fixed arms, in BOTH telemetry
    formats: "map" rows ({time_s, actions: {cav: action}}) and chunk-1
    "per_vehicle" rows ({timestamp_s, vehicle_id, action, ...}).  Returns
    the mismatch reasons (empty = the realised actions match the t0
    payload)."""
    invalid: List[str] = []
    for row in load_jsonl(run.dispatcher_path):
        if layout.dispatcher_rows_format == "per_vehicle":
            now = float(row.get("timestamp_s", row.get(layout.row_time_key, 0.0)))
            if now < t0_s - 1e-9:
                continue
            cav = str(row.get("vehicle_id"))
            action = row.get("action")
            if cav in realised and realised[cav] not in (None, str(action)):
                invalid.append(f"realised action differs from t0 payload for {cav!r} at t={now}")
        else:
            if float(row.get(layout.row_time_key, 0.0)) < t0_s - 1e-9:
                continue
            for cav, action in (row.get("actions") or {}).items():
                if realised.get(cav) not in (None, str(action)):
                    invalid.append(
                        f"realised action differs from t0 payload for "
                        f"{cav!r} at t={row.get(layout.row_time_key)}"
                    )
                    break
    return invalid


def _cav_ids_from_rows(
    rows: Union[Sequence[dict], Sequence[Tick]], layout: Layout
) -> Tuple[str, ...]:
    """CAV ids seen in the given rows: the ``row_cav_key`` (cav_actions)
    keys plus each row's ``row_cav_list_key`` (cav_ids) list -- chunk-1
    all-native arms log NO cav_actions entries, so the per-row list is
    their only CAV universe."""
    ticks = rows if (rows and isinstance(rows[0], Tick)) else ticks_from_rows(rows, layout)
    ids: set = set()
    for tick in ticks:
        ids.update(str(vid) for vid in tick.cav_actions)
        ids.update(tick.cav_ids)
    return tuple(sorted(ids))


def _cav_ids(run: RunPaths, layout: Layout) -> Tuple[str, ...]:
    """CAV ids of a run over ALL its physics rows: cav_actions keys plus
    cav_ids lists (chunk-1 arms log empty cav_actions, so the per-row list
    is their only CAV universe -- without it the recorded native/damping
    plan candidates would be empty). Served from the run's single scan."""
    return get_scan(run, layout).cav_ids_all


def first_row_time(run: RunPaths, layout: Layout) -> Optional[float]:
    """Earliest physics row time of a run (None when it has no rows).

    Used to derive the COMMON window start: the restore does not re-record
    the t0 tick itself, so restored arms' coverage starts at t0 + dt while
    the base keeps its t0 anchor row; comparing from the LATEST first row
    keeps every arm on identical row coverage."""
    return get_scan(run, layout).first_time


def _coverage_s(run: RunPaths, t0_s: float, layout: Layout) -> float:
    """Recorded post-t0 length: (last row time + dt) - t0; a row covers
    [t, t+dt), so 542 s of 0.5 s rows end at row time 541.5."""
    scan = get_scan(run, layout)
    if scan.last_time is None:
        return 0.0
    return max(0.0, scan.last_time + layout.physics_dt_s - t0_s)


def _onset_from_base(root: Root, layout: Layout) -> Optional[float]:
    return _onset_s(get_scan(root.base, layout).ticks, layout)


# ------------------------------------------------------------------ model
class RecordedBranchWorldModel:
    """predict() over the RECORDED branch outcomes of one root.

    Answers only for plans whose per-CAV map equals a recorded constant-plan
    arm at this root (the native sibling supplies native_*); M = 1, K = 1;
    contract v1.1 fields are always reported. Any other plan gets
    ``cell_valid=False`` (fail closed)."""

    def __init__(
        self,
        root: Root,
        *,
        layout: Optional[Layout] = None,
        horizon_s: Optional[float] = None,
        delay_provider: Optional[DelayProvider] = None,
        delay_provider_name: Optional[str] = None,
        p4_search_paths: Sequence[str] = P4_SEARCH_PATHS,
        clock: Optional[Callable[[], float]] = None,
        window_start_s: Optional[float] = None,
        window_s: Optional[float] = None,
        outcomes: Optional[Mapping[str, BranchOutcome]] = None,
        onset_s: Optional[float] = None,
    ):
        self.root = root
        self.layout = layout or Layout()
        self.t0_s = root.t0_s
        # PL-02a idempotence: an offline recorded replay has no meaningful
        # wall clock.  Measuring one by DEFAULT leaked a nondeterministic
        # ``meta["wall_clock_s"]`` into the deterministic RESULTS.json payload
        # (Windows ``time.monotonic`` ~15.6 ms resolution normally read 0.0,
        # but under load the two reads straddled a tick and the idempotence
        # test failed intermittently).  The clock is used only when a caller
        # explicitly injects one (diagnostics); otherwise the recorded model
        # reports the fixed deterministic value 0.0.
        self.clock = clock
        if outcomes is not None:
            # PL-02a-C1 cache-only path: outcomes were loaded from (or
            # computed into) the run cache; NO physics file is read here.
            if window_start_s is None or window_s is None:
                raise ValueError("outcomes injection requires explicit window_start_s and window_s")
            self.window_start_s = float(window_start_s)
            self.window_s = float(window_s)
            self.coverage_s = self.window_s
            self.onset_s = onset_s
            self.outcomes = dict(outcomes)
            self.delay_provider_name = (
                delay_provider_name
                or (
                    self.outcomes["base"].delay_provider
                    if "base" in self.outcomes
                    else next(iter(self.outcomes.values())).delay_provider
                )
                if self.outcomes
                else (delay_provider_name or "custom")
            )
        else:
            # PL-02a PILOT: the common comparison start.  The restore does not
            # re-record the t0 tick itself, so restored arms' coverage starts
            # at t0 + dt while the base keeps its t0 anchor row; comparing
            # from the LATEST first row over the compared arms keeps identical
            # coverage (default None keeps the old behaviour: start at t0).
            if window_start_s is not None:
                self.window_start_s = float(window_start_s)
            else:
                starts = [
                    first_row_time(run, self.layout)
                    for run in [root.base] + list(root.branches.values())
                ]
                starts = [s for s in starts if s is not None]
                self.window_start_s = max(starts) if starts else self.t0_s
            if delay_provider is None:
                try:
                    delay_provider = p4_delay_provider(p4_search_paths)
                    self.delay_provider_name = "p4"
                except P4DelayUnavailable as exc:
                    # precise fallback: built-in cross-check provider, reason
                    # recorded on the model for the runner to surface
                    delay_provider = default_delay
                    self.delay_provider_name = "builtin"
                    self.delay_provider_fallback_reason = str(exc)
            else:
                self.delay_provider_name = delay_provider_name or "custom"
            # common window: min post-start coverage over base + native + branches
            arms = [root.base] + list(root.branches.values())
            coverages = {
                run.arm: _coverage_s(run, self.window_start_s, self.layout) for run in arms
            }
            self.coverage_s = min(coverages.values()) if coverages else 0.0
            self.window_s = (
                float(window_s)
                if window_s is not None
                else (float(horizon_s) if horizon_s is not None else self.coverage_s)
            )
            self.onset_s = onset_s if onset_s is not None else _onset_from_base(root, self.layout)
            # entry waits are scheduled in the BASE run's scenario file
            base_departs = _read_departs(root.base.rou_paths)
            self.outcomes: Dict[str, BranchOutcome] = {
                "base": branch_outcome(
                    root.base,
                    t0_s=self.t0_s,
                    window_s=self.window_s,
                    layout=self.layout,
                    delay_provider=delay_provider,
                    departs=base_departs,
                    provider_name=self.delay_provider_name,
                    window_start_s=self.window_start_s,
                    t0_present_ids=root.t0_present_ids,
                )
            }
            for arm, run in root.branches.items():
                self.outcomes[arm] = branch_outcome(
                    run,
                    t0_s=self.t0_s,
                    window_s=self.window_s,
                    layout=self.layout,
                    delay_provider=delay_provider,
                    departs=base_departs,
                    provider_name=self.delay_provider_name,
                    window_start_s=self.window_start_s,
                    t0_present_ids=root.t0_present_ids,
                )
        self.plans: Dict[Tuple[Tuple[str, str], ...], str] = {}
        for arm, run in root.branches.items():
            if arm in self.layout.integrity_arm_names:
                continue  # e.g. "verify": integrity check, NOT a candidate
            outcome = self.outcomes[arm]
            if outcome.is_constant and outcome.realised_plan:
                key = tuple(sorted(outcome.realised_plan.items()))
                self.plans.setdefault(key, arm)
        # native candidate: every CAV native (CAV universe from the native
        # run's own outcome -- cav_actions keys + cav_ids lists over ALL rows)
        native_outcome = self.outcomes.get("native")
        if native_outcome is not None and native_outcome.cav_ids_all:
            key = tuple(sorted((cav, "native") for cav in native_outcome.cav_ids_all))
            self.plans.setdefault(key, "native")
        # damping-base candidate: every CAV on the base prefix controller
        base_outcome = self.outcomes.get("base")
        if base_outcome is not None and base_outcome.cav_ids_all:
            controller = root.base.world_meta.get("controller_name", "damping")
            key = tuple(sorted((cav, controller) for cav in base_outcome.cav_ids_all))
            self.plans.setdefault(key, "base")

    # ------------------------------------------------------------------ api
    def predict_many(
        self,
        state_samples: Sequence[Any],
        plans: Sequence[Any],
        controller_state: Optional[Mapping[str, Any]] = None,
        *,
        horizon_s: Optional[float] = None,
        trajectory_s: Optional[float] = None,
    ) -> "list[Prediction]":
        started = self.clock() if self.clock is not None else None
        predictions = []
        for plan in plans:
            predictions.append(self._prediction_for(plan, state_samples))
        _ = horizon_s, trajectory_s  # recorded data has one fixed window
        wall_clock = (self.clock() - started) if self.clock is not None else 0.0
        for prediction in predictions:
            prediction.meta["wall_clock_s"] = wall_clock
        return predictions

    def predict(self, state_samples, plan, controller_state=None, **kw):
        return self.predict_many(state_samples, [plan], controller_state, **kw)[0]

    # -------------------------------------------------------------- plumbing
    def _prediction_for(self, plan, state_samples) -> Prediction:
        key = tuple(sorted((cav, str(action)) for cav, action in plan.per_cav.items()))
        arm = self.plans.get(key)
        outcome = self.outcomes.get(arm) if arm else None
        native = self.outcomes.get("native")
        known = outcome is not None
        cell_valid = known and outcome.cell_valid and (native is not None and native.cell_valid)
        nan = float("nan")
        min_ttc = (
            outcome.min_ttc_s if outcome is not None and outcome.min_ttc_s is not None else nan
        )
        native_ttc = (
            native.min_ttc_s if native is not None and native.min_ttc_s is not None else nan
        )
        delay = outcome.delay_s if known else nan
        native_delay = native.delay_s if native is not None else nan
        meta = {
            "contract_version": CONTRACT_VERSION_V1_1,
            "model_id": f"recorded:{self.root.label}",
            "weights_sha256": [f"recorded:{self.root.label}:{arm or 'unknown'}"],
            "sample_ids": [s.meta["sample_id"] for s in state_samples] or ["recorded"],
            "member_ids": ["m0"],
            "part_a_horizon_s": self.window_s,
            "safety_window_s": self.window_s,  # full-horizon window (declared)
            "horizon_s": self.window_s,
            "native_kept_walls": [],
            "native_unlocked_available": False,
            "device": "recorded",
            "wall_clock_s": 0.0,
            "recorded_arm": arm,
        }
        return Prediction(
            delay_vs_native_s=np.array([[delay - native_delay]]),
            plan_delay_s=np.array([[delay]]),
            native_delay_s=np.array([[native_delay]]),
            delay_part_a_s=np.array([[delay]]),  # full-horizon window: A=all
            delay_part_b_s=np.array([[0.0]]),
            collision=np.array([[outcome.collision if known else False]]),
            min_ttc_s=np.array([[min_ttc]]),
            ttc_lt3_exposure_s=np.array([[outcome.ttc_lt3_exposure_s if known else 0.0]]),
            native_ttc_lt3_exposure_s=np.array(
                [[native.ttc_lt3_exposure_s if native is not None else 0.0]]
            ),
            # T-A: the primary 1.5 s exposure (v1.1 predictions carry the
            # window fields only -- no future block)
            ttc_lt1p5_exposure_s=np.array(
                [[outcome.ttc_lt1p5_exposure_s if known else 0.0]]
            ),
            native_ttc_lt1p5_exposure_s=np.array(
                [[native.ttc_lt1p5_exposure_s if native is not None else 0.0]]
            ),
            hard_brake_s=np.array([[outcome.hard_brake_s if known else 0.0]]),
            native_hard_brake_s=np.array([[native.hard_brake_s if native is not None else 0.0]]),
            native_collision=np.array([[native.collision if native is not None else False]]),
            native_min_ttc_s=np.array([[native_ttc]]),
            trajectories_tracked={},
            native_trajectories_tracked={},
            trajectories_hidden=None,
            walls={},
            cell_valid=np.array([[cell_valid]]),
            native_cell_valid=np.array([[native.cell_valid if native is not None else False]]),
            has_closing_pair=np.array([[outcome.has_closing_pair if known else False]]),
            native_has_closing_pair=np.array(
                [[native.has_closing_pair if native is not None else False]]
            ),
            meta=meta,
        )
