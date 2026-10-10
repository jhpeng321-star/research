"""PL-04 closed-loop planner runner (output/planner_v1/PL04_CLOSED_LOOP_DESIGN.md).

Runs the planner IN the loop against a master world: the planner re-decides at
every decision point (``decision_period_s``, default 30, range 10-60) and its
decisions are executed by the per-CAV dispatcher and the wall group. This
module is the RUNNER only -- every piece it reuses lives where it was built:
decision logic ``Planner.decide`` (loop.py, ``mode="closed_loop"``), plan ->
commands ``dispatch.to_dispatcher_v3``, walls ``walls.WallRegistry``,
predictor ``predictor: perfect_sim | learned | mock``, saved-state guards
``perfect_sim_state`` (precision 17 + RNG, PL-10c scenario guard, PL-10d fresh
process per branch).

SUMO-facing surfaces (process launch, traci stepping, saveState/loadState)
sit behind injectable interfaces so the WHOLE runner is exercised with fakes:

- :class:`MasterHarness` -- the master world seam. The fake
  (:class:`FakeMasterHarness`) is a deterministic scripted world; the real
  (:class:`RealMasterHarness`) owns the SUMO launch (own free port, scenario
  COPY, CAV speedMode 27, ``--save-state.precision 17 --save-state.rng true``)
  and the MixedAssignmentSession + CommandOwnership + SumoLocalSensor wiring.
  PL-04d closes the real loop: ``step_physics`` executes the current
  dispatcher plan on every 1 s control tick through the SAME shared control
  write as the lookahead branches (:class:`SessionControlWrite`), the issued
  command rows drain into ``dispatcher_rows.jsonl``, ``observe`` builds the
  K=1 sample from the RSU sensor frames only (real frames/coverage/
  merge_observed; ground truth never enters the sample), and the run's
  ``finally`` hands every owned CAV back through ``owner.release_all``.
  Everything is exercised with faked Popen/traci/sensor/ownership; the first
  real-SUMO execution is the <=80 sim-s smoke window.
- The lookahead predictor is an ordinary ``WorldModel``; perfect-sim lookahead
  runs ONE FRESH process per (plan, draw) branch via the existing
  ``BranchProcessFactory`` (PL-10d), ``continuation="native"`` only. The
  MASTER process never calls loadState except on resume.

Guards implemented here: gate refusal -> logged ``gate_refusal`` veto + ALL
non-locked CAVs native (assembly locks kept; HOLD walls follow the
release/abort rules); the owner-only gate override loaded ONLY from
``--gate-override-file`` JSON (``owner:`` approved_by, expiry, per-decision +
receipt path/sha256, no CLI flag or default creates one);
``predictor_class`` oracle/learned labelling in every decisions.jsonl row and
the receipt with a mixed-class-refusing aggregation helper; the HARD
``--max-sim-s`` cap over master + ALL lookahead with an up-front
``estimate_cost`` projection refusal and a clean ``budget_exhausted`` stop;
HOLDOUT/TEST refusal without an explicit owner flag; write-once run dir;
checkpoints at every decision point (guarded SUMO state + python state + log
byte offsets) with resume = fresh process + log truncation; the
``pl04_closed_loop_receipt/1`` written atomically ONLY on completion.
PL-04f: ``--initial-plan damping|pi`` PRE-SETs a non-native initial plan on
the roster at t0 and labels the run ``run_purpose='control_path_proof'``
(control-path evidence only, refused by the aggregation helpers) -- never
planner behaviour.

Zero simulation at import time: no SUMO/traci/libsumo import anywhere; the
real harness imports traci lazily inside its methods.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .candidate_replay import CANDIDATES_FILE
from .config import PlannerConfig
from .dispatch import (
    dispatcher_plan_to_mixed_assignment_v3_real,
    plan_id_for,
    to_dispatcher_v3,
)
from .harness_state import capture_harness_state, previous_lane_ids, restore_harness_state
from .interface import ACTIONS, PlanCandidate, WallSpec, with_actions, with_wall_release
from .loop import (  # noqa: F401 (re-export)
    NO_DECISION_UNITS,
    DecisionReason,
    Planner,
    select_decision_units,
)
from .mock import MockWorldModel
from .paths import research_dense_root
from .perfect_sim import (
    NATIVE_PLAN_LABEL,
    BranchProcessFactory,
    MetricsExtractor,
    P4DelayEvaluator,
    PerfectSimWorldModel,
    SimpleArrivalDraws,
    StateSnapshotter,
    TraciBranchRunner,
    TraciStateSnapshotter,
    estimate_cost,
)
from .perfect_sim_state import (
    MANIFEST_FILE,
    REQUIRED_SAVE_STATE_CLI,
    SCHEMA,
    SUMO_STATE_FILE,
    StateSnapshotError,
    check_scenario_exactness,
    parse_sumo_version,
    require_exact_state_file,
    sha256_file,
    sha256_text,
)
from .safety import FutureSafetyRefusal, safety_check
from .walls import WallRegistry
__all__ = [
    "BELOW_NORMAL_PRIORITY_CLASS",
    "BaseRunRefusedError",
    "BranchControlWrite",
    "BudgetRefusedError",
    "BuildFailedError",
    "BuildContext",
    "CAV_SPEED_MODE",
    "CHECKPOINT_SCHEMA",
    "ClosedLoopConfig",
    "ClosedLoopError",
    "ClosedLoopGateOverride",
    "ClosedLoopRunner",
    "ControlPathProofAggregationRefusedError",
    "DECISION_ROW_SCHEMA",
    "FakeMasterHarness",
    "INITIAL_PLAN_ACTIONS",
    "LoadedGateOverride",
    "LearnedObservationUnavailableError",
    "MasterBoundSnapshotter",
    "MergeMapUnavailableError",
    "NothingToResumeError",
    "OverrideRefusedError",
    "PathGuardError",
    "RECEIPT_SCHEMA",
    "REAL_HARNESS_NOTE",
    "RealBranchProcessFactory",
    "RealMasterHarness",
    "RecordingControlWrite",
    "RosterSourceError",
    "RUN_PURPOSE_CONTROL_PATH_PROOF",
    "RUN_PURPOSE_PLANNER",
    "RUN_PURPOSE_REFERENCE",
    "RunnerComponents",
    "RunDirExistsError",
    "SessionControlWrite",
    "SplitRefusedError",
    "WallsOnRefusedError",
    "aggregate_decisions",
    "classify_veto",
    "config_from_args",
    "copy_scenario_into_run_dir",
    "initial_plan_for_roster",
    "fixed_policy_for_roster",
    "load_gate_override_file",
    "main",
    "mixed_assignment_plan",
    "refuse_control_path_proof",
    "resolve_base_run",
    "roster_from_base_run",
    "select_decision_units",
]

RECEIPT_SCHEMA = "pl04_closed_loop_receipt/1"
CHECKPOINT_SCHEMA = "pl04_checkpoint/1"
DECISION_ROW_SCHEMA = "pl04_decision/1"
CAV_SPEED_MODE = 27  # design section Loop: CAV speedMode 27 on the master

#: predictor -> predictor_class (design Safety gates, Oracle labelling):
#: perfect_sim reads the full truth (oracle); the learned predictor does not.
PREDICTOR_CLASS = {"perfect_sim": "oracle", "learned": "learned", "mock": "mock"}
PREDICTORS = tuple(PREDICTOR_CLASS)
GATE_OVERRIDE_SCOPES = ("future_ttc", "relative_safety", "both")
OWNER_PREFIX = "owner:"

#: PL-04f: the run purpose carried by the receipt and EVERY decisions.jsonl
#: row. 'control_path_proof' marks a run whose non-native initial plan was
#: PRE-SET (``--initial-plan damping|pi``): its rows are CONTROL-PATH
#: evidence only, never planner-behaviour evidence, and the aggregation
#: helpers refuse them (they are never pooled with planner results).
RUN_PURPOSE_PLANNER = "planner"
RUN_PURPOSE_CONTROL_PATH_PROOF = "control_path_proof"
#: T-A (--fixed-policy): a REFERENCE run on the same runner/harness/base-run
#: restore and full roster with NO decision points and NO lookahead
#: branches -- every roster CAV holds one fixed policy from t0. Never
#: planner behaviour; the PL-EXP1b comparator identity
#: (run_purpose='reference' + fixed_policy='<arm>') that the analyzer
#: guards on.
RUN_PURPOSE_REFERENCE = "reference"

#: PL-04f ``--initial-plan`` choices -> the per-CAV action of the pre-set t0
#: plan (None = native: the current behaviour, the planner decides all).
INITIAL_PLAN_ACTIONS = {"native": None, "damping": "damping", "pi": "pi-external-u"}

#: T-A ``--fixed-policy`` choices -> the per-CAV action every roster CAV
#: holds for the whole reference episode (native included: the lenient plan
#: default). The receipt records the CLI token ('native'|'damping'|'pi') as
#: ``fixed_policy`` -- the analyzer's comparator identity.
FIXED_POLICY_CHOICES = ("native", "damping", "pi")
FIXED_POLICY_ACTIONS = {
    "native": "native",
    "damping": "damping",
    "pi": "pi-external-u",
}

#: Dispatcher-side physics/control cadence (design: 0.5 s physics, 1 s control).
PHYSICS_DT_S = 0.5
CONTROL_DT_S = 1.0

LOG_FILES = ("physics_steps.jsonl", "dispatcher_rows.jsonl", "decisions.jsonl", "walls.jsonl")

#: Windows low-priority launch (design Guards): the master and every
#: lookahead branch process are started BELOW_NORMAL.
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000

#: PL-04c: there is NO default launch seed. The master and every lookahead
#: branch process launch with the run's REQUIRED ``--seed`` (the run config
#: seed, recorded in the receipt); ``--components real`` without an
#: explicit ``--seed`` is refused (a hard-coded seed silently decoupled the
#: real run from the world's seed).

#: Launch attempts per branch process (own free port each; a failed init
#: kills OUR Popen child and retries on a fresh port, at most this many).
BRANCH_LAUNCH_ATTEMPTS = 3

REAL_HARNESS_NOTE = (
    "The real master path (launch, shared control write, RSU observation, "
    "guarded save/restore) runs ONLY against faked Popen/traci/sensor/"
    "ownership surfaces in this worktree -- zero SUMO is ever launched; the "
    "<=80 sim-s closed-loop smoke joins the PL-10 confirmation-smoke window "
    "and is its first real-SUMO execution."
)


# ------------------------------------------------------------------- errors
class ClosedLoopError(RuntimeError):
    """Base class: every runner refusal carries a precise, logged message."""


class SplitRefusedError(ClosedLoopError):
    """A HOLDOUT/TEST world was requested without the explicit owner flag."""


class PathGuardError(ClosedLoopError):
    """A configured path is outside the sanctioned E: workspace roots."""


class OverrideRefusedError(ClosedLoopError):
    """A gate-override file is missing, malformed, non-owner or expired."""


class BudgetRefusedError(ClosedLoopError):
    """The up-front cost projection exceeds the --max-sim-s cap."""


class RunDirExistsError(ClosedLoopError):
    """The run dir exists and the command did not pass --resume (write-once)."""


class BaseRunResumeRefusedError(ClosedLoopError):
    """``--resume`` of a run that was started with ``--base-run`` (T-B #11).

    A base-run master lands on its t0 with ONE guarded loadState and carries
    restored harness state (ownership / loop state / session / PI) that the
    checkpoint format does not reproduce bit-exactly across a process
    boundary for such a start. Such runs are RERUN into a NEW run dir
    (write-once run dirs; never resumed)."""


class NothingToResumeError(ClosedLoopError):
    """--resume was passed but the run dir carries no usable checkpoint."""


class BuildFailedError(ClosedLoopError):
    """The components builder cannot construct the requested runner parts."""


class HarnessNotStartedError(ClosedLoopError):
    """A master-harness surface was used before ``start()`` (PL-04i).

    The runner deliberately builds every component BEFORE the harness
    starts (a build failure must not spend a SUMO launch), so the master
    traci connection does not exist at build time. Every start()-only
    attribute is initialized to None in the constructor, and anything that
    truly needs the live master api raises this NAMED error -- never an
    ``AttributeError`` that could masquerade as a wiring bug."""


class LearnedObservationUnavailableError(ClosedLoopError):
    """The learned predictor's sample hybrid cannot be built on the real
    path: no kit/p2C hybrid-state builder is wired into the harness, so the
    K=1 sample would have to be invented. Fail closed at sample build time
    (RealMasterHarness.observe); never fall back to an opaque marker for the
    learned predictor (the oracle marker is only for perfect_sim, whose
    branches read the truth through their own snapshot)."""


class MergeMapUnavailableError(ClosedLoopError):
    """The RSU merge-region lane map (lane_id -> lane-end distance to the
    merge, metres) could not be resolved for a REAL run (PL-04j): the live
    connection's ``observation_map`` failed or came back EMPTY. Running on
    would be the silent PL-04 defect -- no frame could coverage-reach the
    merge region, ``merge_observed`` would resolve False (contract 8.15,
    fail closed) and the planner would abstain at every decision while the
    smoke reported INCONCLUSIVE -- so start() refuses instead, loudly."""


class BaseRunRefusedError(ClosedLoopError):
    """A --base-run directory is missing, malformed, or fails its saved-state
    guards (PL-04e2): no saved_state, a manifest that is not
    ``wm41_saved_state/1`` with precision-17 + RNG options, a state-file
    digest drift, a scenario-digest drift against the base run's own files,
    or an empty CAV roster."""


class RosterSourceError(BaseRunRefusedError):
    """The --base-run CAV ROSTER SOURCE is missing, empty or inconsistent
    (PL-04m): ``run/physics_steps.jsonl`` is absent or carries no rows, a
    row without a usable ``cav_ids`` list, an EMPTY ``cav_ids`` list, or
    rows that DISAGREE about the designated CAV list. The campaign's full
    designated CAV set is a CONSTANT of the world -- a disagreement means
    the directory is not a readable campaign base run, never a smaller
    roster (fail closed; never shrink to the agreeable subset)."""


class WallsOnRefusedError(ClosedLoopError):
    """``--walls on`` on a REAL closed-loop run before PL-04e2-walls: the
    WM-59 wall-member helpers are not promoted yet, so wall execution has no
    sanctioned real path here. Named refusal -- never a silent fallback."""


class ControlPathProofAggregationRefusedError(ClosedLoopError):
    """An aggregation was offered control-path-proof rows/receipts (PL-04f):
    a run whose non-native initial plan was PRE-SET (``--initial-plan
    damping|pi``) proves the control path ONLY -- its rows are never
    planner-behaviour evidence, so they are refused (never pooled with
    planner results, never summarized as planner decisions)."""


# ------------------------------------------------------------ gate override
@dataclass(frozen=True)
class ClosedLoopGateOverride:
    """Owner-approved gate override (design Safety gates, explicit override).

    Only ever produced by :func:`load_gate_override_file` -- there is no CLI
    flag, programmatic default or session-created override file (owner-only
    rule, main discussion 2026-09-28)."""

    reason: str
    approved_by: str
    scope: str  # future_ttc | relative_safety | both
    expires_utc: str  # REQUIRED: every override is time-bounded (owner rule)

    @property
    def covers_future_ttc(self) -> bool:
        return self.scope in ("future_ttc", "both")

    @property
    def covers_relative_safety(self) -> bool:
        return self.scope in ("relative_safety", "both")


@dataclass(frozen=True)
class LoadedGateOverride:
    """The override plus the identity of the file it was loaded from."""

    override: ClosedLoopGateOverride
    path: str
    sha256: str


def _parse_expires_utc(text: str) -> datetime:
    value = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value


def load_gate_override_file(path, *, now_utc: Optional[datetime] = None) -> LoadedGateOverride:
    """The ONLY source of a gate override: an owner-approved JSON file.

    Refusals (:class:`OverrideRefusedError`, never silent): unparseable or
    non-mapping JSON; a missing/empty ``reason``; a missing or non-``owner:``
    ``approved_by``; an unknown ``scope``; a missing, empty, unparseable or
    already-expired ``expires_utc`` (EVERY override is time-bounded: a
    missing ``expires_utc`` is refused, never "never expires"). The file's
    path and sha256 travel with the override so every affected decision
    record and the receipt can carry them."""

    resolved = Path(path)
    now = now_utc if now_utc is not None else datetime.now(timezone.utc)

    def refuse(why: str) -> None:
        raise OverrideRefusedError(
            f"gate override file {str(resolved)!r} refused: {why} (owner-only "
            "rule: an override may only be LOADED from an owner-approved file)"
        )

    if not resolved.is_file():
        refuse("no such file")
    raw = resolved.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        refuse(f"not valid JSON ({exc})")
    if not isinstance(data, Mapping):
        refuse("the JSON must be an object")
    reason = data.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        refuse("a non-empty 'reason' is required")
    approved_by = data.get("approved_by")
    if approved_by is None:
        refuse("'approved_by' is missing")
    if not isinstance(approved_by, str) or not approved_by.startswith(OWNER_PREFIX):
        refuse(f"'approved_by' must start with {OWNER_PREFIX!r} (got {approved_by!r})")
    scope = data.get("scope")
    if scope not in GATE_OVERRIDE_SCOPES:
        refuse(f"'scope' must be one of {list(GATE_OVERRIDE_SCOPES)} (got {scope!r})")
    expires_raw = data.get("expires_utc")
    if expires_raw is None:
        refuse("'expires_utc' is required (every override is time-bounded)")
    if not isinstance(expires_raw, str):
        refuse("'expires_utc' must be an ISO-8601 timestamp string")
    if not expires_raw.strip():
        refuse("'expires_utc' must be a non-empty ISO-8601 timestamp")
    try:
        expires = _parse_expires_utc(expires_raw)
    except ValueError as exc:
        refuse(f"'expires_utc' is not ISO-8601 ({exc})")
    if expires <= now:
        refuse(f"'expires_utc' {expires_raw} has already expired")
    expires_utc = expires_raw
    return LoadedGateOverride(
        override=ClosedLoopGateOverride(
            reason=reason,
            approved_by=approved_by,
            scope=scope,
            expires_utc=expires_utc,
        ),
        path=str(resolved),
        sha256=sha,
    )


# ---------------------------------------------------------- base-run (PL-04e2)
#: The copied scenario's net-file rewrite target: the preserved network in the
#: research-dense assets mount (read-only; see paths.py). The rewrite is
#: value-agnostic (the un-migrated copies carry a retired drive literal).
_ASSETS_NET = (
    research_dense_root() / "assets" / "research" / "highway_merge_shepherd_long" / "osm.net.xml.gz"
)
_NET_FILE_VALUE_RE = re.compile(r'(<net-file value=")[^"]*(")')


@dataclass(frozen=True)
class BaseRunState:
    """A validated campaign-format base run (PL-04e2 ``--base-run``).

    Produced ONLY by :func:`resolve_base_run` (every field is evidence-checked
    against the directory); the master's start consumes ``as_saved_dict()`` --
    the shape the guarded snapshotter restore expects."""

    base_run: Path
    manifest: dict
    state_dir: Path
    state_file: Path
    scenario_dir: Path
    t0_s: float
    seed: int
    state_sha256: str
    # T-B: sha256 of the STATE_MANIFEST.json bytes (the restore source the
    # receipt records; '' only for hand-built test states)
    manifest_sha256: str = ""

    def as_saved_dict(self) -> dict:
        return {
            "saved_state_dir": str(self.state_dir),
            "manifest": dict(self.manifest),
            "t0_s": float(self.t0_s),
            "manifest_sha256": self.manifest_sha256,
        }


def resolve_base_run(base_run) -> BaseRunState:
    """Validate a campaign-format base run for the ``--base-run`` start.

    Refusals (:class:`BaseRunRefusedError`, never silent): no
    ``<dir>/saved_state`` (manifest or state file missing); a manifest that
    is not ``wm41_saved_state/1``; missing ``branch_point_t0_s``/``seed``;
    state options that do not prove a precision-17 RNG-carrying snapshot;
    a state file that fails the exactness guard or its recorded sha256; a
    missing ``run/scenario/scenario.sumocfg``; a manifest scenario digest
    that does not match the base run's own scenario file. READ-ONLY:
    nothing under ``<dir>`` is ever written."""
    base = Path(base_run)

    def refuse(why: str) -> None:
        raise BaseRunRefusedError(
            f"--base-run {str(base)!r} refused: {why} (the master can only "
            "start from a guarded campaign saved_state)"
        )

    state_dir = base / "saved_state"
    manifest_file = state_dir / MANIFEST_FILE
    state_file = state_dir / SUMO_STATE_FILE
    if not manifest_file.is_file() or not state_file.is_file():
        refuse(
            f"{state_dir} must carry {MANIFEST_FILE} + {SUMO_STATE_FILE} "
            "(no saved_state -> no base start)"
        )
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        refuse(f"{MANIFEST_FILE} is unreadable ({exc})")
    if not isinstance(manifest, Mapping) or manifest.get("schema") != SCHEMA:
        refuse(f"manifest schema is {manifest.get('schema')!r}, expected {SCHEMA!r}")
    try:
        t0_s = float(manifest["branch_point_t0_s"])
        seed = int(manifest["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        refuse(f"manifest lacks a usable branch_point_t0_s/seed ({exc})")
    options = manifest.get("state_options") or {}
    if not options.get("rng_state_in_file"):
        refuse("manifest records rng_state_in_file=False (state cannot restore bit-exactly)")
    if int(options.get("save_state_precision_decimals", 0)) < 17:
        refuse("manifest records save_state_precision_decimals < 17 (state is position-rounded)")
    recorded_cli = tuple(options.get("required_cli", ()))
    if recorded_cli and recorded_cli != REQUIRED_SAVE_STATE_CLI:
        refuse(f"manifest required_cli {list(recorded_cli)} != the exactness contract")
    try:
        require_exact_state_file(state_file)  # the REAL exactness guard on the bytes
    except StateSnapshotError as exc:  # narrowed alias below
        refuse(f"state file fails its exactness guard ({exc})")
    recorded_sha = manifest.get("sumo_state_sha256")
    if recorded_sha and recorded_sha != sha256_file(state_file):
        refuse("saved state.xml fails its manifest sha256")
    scenario_dir = base / "run" / "scenario"
    if not (scenario_dir / "scenario.sumocfg").is_file():
        refuse(f"no run/scenario/scenario.sumocfg under {str(base)!r}")
    for rel, digest in sorted((manifest.get("scenario_digests") or {}).items()):
        path = base / "run" / str(rel)
        if not path.is_file() or sha256_file(path) != digest:
            refuse(f"scenario digest drift for {rel!r}: the saved state does not match the files")
    return BaseRunState(
        base_run=base,
        manifest=dict(manifest),
        state_dir=state_dir,
        state_file=state_file,
        scenario_dir=scenario_dir,
        t0_s=t0_s,
        seed=seed,
        state_sha256=sha256_file(state_file),
        manifest_sha256=sha256_file(manifest_file),
    )


def roster_from_base_run(base_run, t0_s: float) -> "tuple[tuple[str, ...], dict]":
    """The designated CAV roster of a base run (PL-04m: the FULL set).

    The AUTHORITATIVE designation source is the campaign's designated-CAV
    list carried by EVERY physics row: ``<base_run>/run/physics_steps.jsonl``
    ``cav_ids`` -- CONSTANT from the first row (t=0.5) on, and INCLUDING
    CAVs inserted after ``t0_s`` and even after the run ends. The roster is
    that full constant list; it is NOT filtered to ``inserted_s <= t0``
    (the pre-PL-04m rule left every post-t0 CAV undesignated -- speedMode 31
    with no ownership -- and the planner abstained once the t0 CAVs left).

    READ-ONLY streaming: the first row fixes the list, every later row must
    carry the SAME list. The list is never re-sampled and never derived
    from insertion or type heuristics (``insertions.jsonl`` ``is_cav`` is
    unpopulated in this campaign format); ``insertions.jsonl`` is read
    READ-ONLY only for the receipt counts ``n_inserted_le_t0`` /
    ``n_inserted_after_t0_or_never``. A missing, empty or inconsistent
    source raises the named :class:`RosterSourceError` (a
    :class:`BaseRunRefusedError`) -- never a shrunk or invented roster."""
    base = Path(base_run)
    physics = base / "run" / "physics_steps.jsonl"
    insertions = base / "run" / "insertions.jsonl"
    if not physics.is_file():
        raise RosterSourceError(
            f"--base-run {str(base)!r} refused: run/physics_steps.jsonl is "
            "required for the CAV roster (every physics row carries the "
            "campaign's full designated cav_ids list)"
        )
    roster: Optional[tuple] = None
    first_time_s: Any = None
    rows_read = 0
    with physics.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RosterSourceError(
                    f"--base-run {str(base)!r} refused: physics row "
                    f"{rows_read + 1} is not valid JSON ({exc})"
                ) from exc
            rows_read += 1
            cav_ids = row.get("cav_ids")
            if not isinstance(cav_ids, list) or not cav_ids:
                raise RosterSourceError(
                    f"--base-run {str(base)!r} refused: physics row "
                    f"{rows_read} carries no usable cav_ids list ({cav_ids!r})"
                )
            ids = tuple(str(c) for c in cav_ids)
            if roster is None:
                roster = ids
                first_time_s = row.get("time_s", row.get("t_s"))
            elif ids != roster:
                raise RosterSourceError(
                    f"--base-run {str(base)!r} refused: physics row "
                    f"{rows_read} (t={row.get('time_s', row.get('t_s'))!r}) "
                    f"disagrees about the designated CAV list: {list(ids)!r} "
                    f"!= the first row's {list(roster)!r}"
                )
    if roster is None:
        raise RosterSourceError(
            f"--base-run {str(base)!r} refused: run/physics_steps.jsonl "
            "carries no physics rows (no cav_ids roster source)"
        )
    if not insertions.is_file():
        raise RosterSourceError(
            f"--base-run {str(base)!r} refused: run/insertions.jsonl is "
            "required for the roster receipt counts (n_inserted_le_t0 / "
            "n_inserted_after_t0_or_never); it is never a designation source"
        )
    inserted: dict = {}
    insertions_read = 0
    for line in insertions.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            inserted[str(row["vehicle_id"])] = float(row["inserted_s"])
            insertions_read += 1
    n_le_t0 = 0
    for cav in roster:
        at = inserted.get(cav)
        if at is not None and at <= float(t0_s) + 1e-9:
            n_le_t0 += 1
    evidence = {
        "source": str(base / "run"),
        "roster_source": "base_run_physics_cav_ids",
        "rule": "the campaign's FULL designated CAV list, read from the "
        "cav_ids field of every physics row (constant from the first row; "
        "includes CAVs inserted after t0); no insertion/type heuristic",
        "roster_source_file": str(physics),
        "physics_rows_read": rows_read,
        "roster_first_row_time_s": first_time_s,
        "t0_s_recorded_not_a_filter": float(t0_s),
        "roster_size": len(roster),
        "n_inserted_le_t0": n_le_t0,
        "n_inserted_after_t0_or_never": len(roster) - n_le_t0,
        "insertions_read": insertions_read,
        "roster_ids": list(roster),
    }
    return roster, evidence


def copy_scenario_into_run_dir(scenario_dir, run_dir) -> dict:
    """Copy a campaign-format scenario into the run dir (PL-04e2).

    The scenario cfg + every sibling file are COPIED to
    ``<run_dir>/scenario/``; the copy's ``<net-file>`` value is rewritten to
    the E: research assets network (value-agnostic; no legacy drive literal
    lives here); the PL-10c scenario-exactness guard is run ON THE COPY and
    must pass (BuildFailedError with the offenses otherwise). The SOURCE
    directory is never written."""
    scenario_dir = Path(scenario_dir)
    target = Path(run_dir) / "scenario"
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in sorted(scenario_dir.iterdir()):
        if path.is_file():
            shutil.copy2(path, target / path.name)
            copied.append(path.name)
    cfg = target / "scenario.sumocfg"
    text = cfg.read_text(encoding="utf-8")
    net_rewrite = {"rewritten": False, "net_file": _ASSETS_NET.as_posix()}
    if _ASSETS_NET.as_posix() not in text:
        text = _NET_FILE_VALUE_RE.sub(
            lambda m: m.group(1) + _ASSETS_NET.as_posix() + m.group(2), text
        )
        cfg.write_text(text, encoding="utf-8")
        net_rewrite["rewritten"] = True
    net_rewrite["net_exists"] = _ASSETS_NET.is_file()
    guard = check_scenario_exactness([cfg], explicit_arrivals_installed=False)
    if not guard.get("exact"):
        raise BuildFailedError(
            f"the copied scenario fails the PL-10c exactness guard: "
            f"{guard.get('offenses', [])[:3]}"
        )
    return {
        "scenario_dir_source": str(scenario_dir),
        "copied_cfg": str(cfg),
        "copied_files": copied,
        "net_file": net_rewrite,
        "scenario_exactness_guard": guard,
    }


def _scenario_departs(cfg_path) -> dict:
    """T-A #1: the scheduled departures of the run's scenario route files
    (the p4 §8.14 entry-wait schedule): ``recorded._read_departs`` over
    every ``*.rou.xml`` sibling of the scenario cfg. Fail closed when the
    scenario dir carries no route file -- a silently empty schedule would
    drop the insertion-backlog term from the branch delay."""
    from .recorded import _read_departs  # noqa: PLC0415 (lazy; analyzer module)

    cfg_dir = Path(cfg_path).parent
    rou_paths = sorted(cfg_dir.glob("*.rou.xml"))
    if not rou_paths:
        raise BuildFailedError(
            f"scenario dir {str(cfg_dir)!r} carries no *.rou.xml route "
            "file: the p4 branch delay evaluator cannot resolve the "
            "scheduled departures (fail closed)"
        )
    return _read_departs([Path(p) for p in rou_paths])


# -------------------------------------------------------------------- config
def _forbidden_split_token(text: str) -> Optional[str]:
    """'HOLDOUT'/'TEST' as a whole token of the uppercased text, else None."""
    tokens = re.split(r"[^A-Za-z0-9]+", str(text).upper())
    for forbidden in ("HOLDOUT", "TEST"):
        if forbidden in tokens:
            return forbidden
    return None


@dataclass(frozen=True)
class ClosedLoopConfig:
    """One closed-loop run's configuration (receipt config sha covers this)."""

    out_root: Path
    run_id: str
    world: str
    seed: int = 0
    split: str = "train"
    horizon_s: float = 120.0
    warmup_s: float = 0.0
    decision_period_s: float = 30.0
    predictor: str = "mock"
    lookahead_horizon_s: float = 60.0
    lookahead_commit_s: Optional[float] = None
    k_samples: int = 1
    max_sim_s: float = 3600.0
    max_wall_s: Optional[float] = None
    cav_speed_mode: int = CAV_SPEED_MODE
    process_start_s: float = 0.0
    allow_holdout: bool = False
    gate_override: Optional[LoadedGateOverride] = None
    # PL-04e2: start from a validated campaign base run's saved_state (the
    # start, NOT a resume), the decision-unit cap, and the walls switch.
    base_run: Optional[Path] = None
    base_t0_s: Optional[float] = None
    base_state_sha256: Optional[str] = None
    max_decision_units: Optional[int] = None
    walls: str = "on"  # resolved: "on"|"off" (real runs: off is the only value)
    # PL-04f control-path proof: the action PRE-SET on every roster CAV at
    # t0 (None = native: the current behaviour, the planner decides all).
    initial_plan_action: Optional[str] = None
    # T-A --fixed-policy: the CLI token ('native'|'damping'|'pi') of a
    # REFERENCE run (run_purpose='reference'): no decision points, no
    # lookahead branches; every roster CAV holds the policy from t0 and the
    # ownership control write runs EVERY control tick in all three modes
    # (CommandOwnership sets speedMode 27 on every live CAV, native
    # included). Mutually exclusive with --initial-plan.
    fixed_policy: Optional[str] = None
    # PER-CANDIDATE logging (candidates.jsonl, one row per priced candidate
    # per decision). PURE logging: it never changes a decision, is excluded
    # from the config sha (to_canonical_json) and defaults ON.
    log_candidates: bool = True

    def __post_init__(self):
        if not isinstance(self.log_candidates, bool):
            raise ValueError("log_candidates must be a bool")
        if self.predictor not in PREDICTORS:
            raise ValueError(f"predictor must be one of {list(PREDICTORS)}")
        if not 10.0 <= float(self.decision_period_s) <= 60.0:
            raise ValueError("decision_period_s must be within the allowed range [10, 60]")
        if float(self.horizon_s) <= 0:
            raise ValueError("horizon_s must be positive")
        if float(self.warmup_s) < 0:
            raise ValueError("warmup_s must be >= 0")
        if float(self.warmup_s) >= float(self.horizon_s):
            raise ValueError("warmup_s must be shorter than horizon_s")
        if int(self.k_samples) < 1:
            raise ValueError("k_samples must be >= 1")
        if float(self.max_sim_s) <= 0:
            raise ValueError("max_sim_s must be positive")
        if self.max_wall_s is not None and float(self.max_wall_s) <= 0:
            raise ValueError("max_wall_s must be positive or None")
        if float(self.lookahead_horizon_s) <= 0:
            raise ValueError("lookahead_horizon_s must be positive")
        if float(self.process_start_s) < 0:
            raise ValueError("process_start_s must be >= 0")
        if not isinstance(self.allow_holdout, bool):
            raise ValueError("allow_holdout must be a bool")
        if self.gate_override is not None and not isinstance(self.gate_override, LoadedGateOverride):
            raise ValueError("gate_override must be a LoadedGateOverride (never built by hand)")
        if self.walls not in ("on", "off"):
            raise ValueError("walls must be 'on' or 'off'")
        if self.initial_plan_action is not None and self.initial_plan_action not in (
            "damping",
            "pi-external-u",
        ):
            raise ValueError(
                "initial_plan_action must be 'damping' or 'pi-external-u' "
                "(None = native: the planner decides everything)"
            )
        if self.fixed_policy is not None:
            if self.fixed_policy not in FIXED_POLICY_ACTIONS:
                raise ValueError(
                    f"fixed_policy must be one of {list(FIXED_POLICY_CHOICES)}, "
                    f"got {self.fixed_policy!r}"
                )
            if self.initial_plan_action is not None:
                raise ValueError(
                    "--fixed-policy is mutually exclusive with --initial-plan: "
                    "a REFERENCE run (run_purpose='reference') pre-sets its "
                    "own fixed policy on the whole roster and has no "
                    "control-path-proof initial plan"
                )
        if self.max_decision_units is not None and (
            isinstance(self.max_decision_units, bool) or int(self.max_decision_units) < 1
        ):
            raise ValueError("max_decision_units must be an int >= 1 or None")
        if (self.base_run is None) != (self.base_t0_s is None):
            raise ValueError("base_run and base_t0_s must be set together")
        if self.base_run is not None:
            if not isinstance(self.base_run, Path):
                raise ValueError("base_run must be a resolved Path (config_from_args resolves it)")
            if self.base_state_sha256 is None:
                raise ValueError(
                    "a base_run start carries the base state sha256 (resolve_base_run)"
                )
        if not self.allow_holdout:
            for label in (self.world, self.split):
                token = _forbidden_split_token(label)
                if token:
                    raise SplitRefusedError(
                        f"{label} {str(label)!r} names the {token} split: HOLDOUT/TEST "
                        "worlds are refused without the explicit owner flag "
                        "(--owner-allow-holdout)"
                    )
        elif any(_forbidden_split_token(label) for label in (self.world, self.split)):
            # PL-12: the owner flag permits HOLDOUT in principle, but the
            # reconstructor's exclusion guard still refuses the exposed
            # worlds (per world, incl. every re-run) -- fail closed, BEFORE
            # any run directory is created or any outcome file is opened.
            from traffic_shepherd.planner import holdout as holdout_guard

            holdout_guard.assert_holdout_eligible(
                [self.world], holdout_requested=True
            )

    @property
    def run_dir(self) -> Path:
        return Path(self.out_root) / self.run_id

    @property
    def predictor_class(self) -> str:
        return PREDICTOR_CLASS[self.predictor]

    @property
    def run_purpose(self) -> str:
        """PL-04f: 'control_path_proof' iff a non-native initial plan was
        pre-set (``--initial-plan damping|pi``); T-A: 'reference' iff
        ``--fixed-policy`` is set; else 'planner'."""
        if self.fixed_policy is not None:
            return RUN_PURPOSE_REFERENCE
        return (
            RUN_PURPOSE_PLANNER
            if self.initial_plan_action is None
            else RUN_PURPOSE_CONTROL_PATH_PROOF
        )

    def to_canonical_json(self) -> str:
        payload = dataclasses.asdict(self)
        # the out root is a location, not run semantics: the config sha must
        # not change when the same run moves between roots (e.g. resume drills)
        del payload["out_root"]
        # logging switch, not run semantics: on/off runs share one config sha
        del payload["log_candidates"]
        payload["gate_override"] = (
            None
            if self.gate_override is None
            else {
                "path": self.gate_override.path,
                "sha256": self.gate_override.sha256,
                "reason": self.gate_override.override.reason,
                "approved_by": self.gate_override.override.approved_by,
                "scope": self.gate_override.override.scope,
                "expires_utc": self.gate_override.override.expires_utc,
            }
        )
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


# ------------------------------------------------------------------ helpers
def _json_safe(obj):
    """JSON-safe copy: non-finite floats become None (rows never carry NaN)."""
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else None
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, bool, int)) or obj is None:
        return obj
    return str(obj)


def _dumps(payload: Mapping) -> str:
    return json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"))


def _write_json_atomic(path: Path, payload: Mapping) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(_dumps(payload) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _plan_to_json(plan: Optional[PlanCandidate]) -> Optional[dict]:
    if plan is None:
        return None
    return {
        "per_cav": dict(plan.per_cav),
        "walls": [
            {
                "wall_id": w.wall_id,
                "members": list(w.members),
                "speed_mps": float(w.speed_mps),
                "phase": w.phase,
                "release_at_s": None if w.release_at_s is None else float(w.release_at_s),
                "hold_s": None if w.hold_s is None else float(w.hold_s),
            }
            for w in plan.walls
        ],
    }


def _plan_from_json(payload: Optional[Mapping]) -> Optional[PlanCandidate]:
    if payload is None:
        return None
    walls = tuple(
        WallSpec(
            wall_id=w["wall_id"],
            members=tuple(w["members"]),
            speed_mps=float(w["speed_mps"]),
            phase=w["phase"],
            release_at_s=None if w["release_at_s"] is None else float(w["release_at_s"]),
            hold_s=None if w["hold_s"] is None else float(w["hold_s"]),
        )
        for w in payload["walls"]
    )
    return PlanCandidate(per_cav=dict(payload["per_cav"]), walls=walls)


def _registry_to_json(registry: WallRegistry) -> list:
    return [
        {
            "wall_id": r.wall_id,
            "members": list(r.members),
            "speed_mps": float(r.speed_mps),
            "phase": r.phase,
            "hold_s": None if r.hold_s is None else float(r.hold_s),
            "release_at_s": None if r.release_at_s is None else float(r.release_at_s),
            "dispatched_at_s": None if r.dispatched_at_s is None else float(r.dispatched_at_s),
            "assembled_at_s": None if r.assembled_at_s is None else float(r.assembled_at_s),
        }
        for r in registry.all()
    ]


def _restore_registry(registry: WallRegistry, records: Sequence[Mapping]) -> None:
    for record in records:
        spec = WallSpec(
            wall_id=record["wall_id"],
            members=tuple(record["members"]),
            speed_mps=float(record["speed_mps"]),
            phase=record["phase"],
            release_at_s=None if record["release_at_s"] is None else float(record["release_at_s"]),
            hold_s=None if record["hold_s"] is None else float(record["hold_s"]),
        )
        restored = registry.register_spec(
            spec, dispatched_at_s=record["dispatched_at_s"]
        )
        restored.phase = record["phase"]
        restored.assembled_at_s = record["assembled_at_s"]


# --------------------------------------------------------------- aggregation
def _row_run_purpose(row: Mapping) -> str:
    """The row's/receipt's run_purpose (PL-04f; rows written before PL-04f
    carry no key and count as 'planner')."""
    return str(row.get("run_purpose") or RUN_PURPOSE_PLANNER)


def refuse_control_path_proof(payload) -> None:
    """The NAMED aggregation guard for rows or a receipt (PL-04f).

    Every analysis/pooling entry point (``aggregate_decisions`` today, any
    future PL-EXP1 analysis) calls this before touching closed-loop rows or
    a ``RUN_RECEIPT.json``: a ``run_purpose='control_path_proof'`` payload
    -- a run whose non-native initial plan was PRE-SET -- is control-path
    evidence only and raises
    :class:`ControlPathProofAggregationRefusedError` here, so it can never
    be pooled with, or reported as, planner behaviour."""
    mappings = [payload] if isinstance(payload, Mapping) else list(payload)
    for item in mappings:
        if _row_run_purpose(item) == RUN_PURPOSE_CONTROL_PATH_PROOF:
            raise ControlPathProofAggregationRefusedError(
                "run_purpose='control_path_proof' is refused: this run's "
                "non-native initial plan was PRE-SET (--initial-plan "
                "damping|pi), so its rows/receipt are control-path evidence "
                "only, NEVER planner-behaviour evidence -- they are never "
                "pooled with or summarized as planner results"
            )


def aggregate_decisions(
    rows: Sequence[Mapping],
    *,
    accept_run_purpose: str = RUN_PURPOSE_PLANNER,
) -> dict:
    """Campaign-compatible aggregation of decisions.jsonl rows.

    REFUSES to mix ``predictor_class`` values (oracle vs learned runs are
    never aggregated together -- design Safety gates, Oracle labelling).
    PL-04f: REFUSES ``run_purpose='control_path_proof'`` rows (the named
    :class:`ControlPathProofAggregationRefusedError`) and any MIX of run
    purposes, so a control-path proof can never be pooled with planner
    results; a run's own receipt aggregation passes its own purpose via
    ``accept_run_purpose`` (one run = one purpose -- still no pooling)."""
    purposes = sorted({_row_run_purpose(row) for row in rows})
    if len(purposes) > 1:
        raise ControlPathProofAggregationRefusedError(
            f"mixed run_purpose values in one aggregation: {purposes}; "
            "control-path-proof rows are never pooled with planner rows"
        )
    if purposes == [RUN_PURPOSE_CONTROL_PATH_PROOF] and (
        accept_run_purpose != RUN_PURPOSE_CONTROL_PATH_PROOF
    ):
        raise ControlPathProofAggregationRefusedError(
            "run_purpose='control_path_proof' rows are refused: this run's "
            "non-native initial plan was PRE-SET (--initial-plan damping|pi), "
            "so its rows are control-path evidence only, NEVER "
            "planner-behaviour evidence (a run's own receipt aggregation "
            "passes accept_run_purpose='control_path_proof')"
        )
    classes = sorted({str(r.get("predictor_class")) for r in rows})
    if len(classes) > 1:
        raise ValueError(
            f"mixed predictor_class values in one aggregation: {classes}; "
            "oracle and learned runs are never mixed"
        )
    vetoes: dict = {}
    abstains = 0
    abstain_reasons: dict = {}
    abs_ttc_classes: dict = {}
    refusals = 0
    override_rows = 0
    lookahead = 0.0
    for row in rows:
        for entry in row.get("abs_ttc_vetoes", ()) or ():
            klass = str((entry or {}).get("pair_class") or "unknown_pair")
            abs_ttc_classes[klass] = abs_ttc_classes.get(klass, 0) + 1
        if row.get("abstain"):
            reason = str(row.get("abstain_reason") or "unspecified")
            abstain_reasons[reason] = abstain_reasons.get(reason, 0) + 1
        for veto in row.get("vetoes", ()):
            rule = str(veto.get("rule", "unknown"))
            vetoes[rule] = vetoes.get(rule, 0) + 1
            if rule == "gate_refusal":
                refusals += 1
        if row.get("abstain"):
            abstains += 1
        if (row.get("gate") or {}).get("override"):
            override_rows += 1
        lookahead += float(row.get("lookahead_sim_s") or 0.0)
    return {
        "decisions": len(rows),
        "predictor_class": classes[0] if classes else None,
        "vetoes_by_rule": dict(sorted(vetoes.items())),
        "abstains": abstains,
        "abstain_reasons": dict(sorted(abstain_reasons.items())),
        "abs_ttc_veto_counts": dict(sorted(abs_ttc_classes.items())),
        "gate_refusals": refusals,
        "override_decisions": override_rows,
        "lookahead_sim_s": lookahead,
    }


def classify_veto(reason: DecisionReason) -> str:
    """Rule name of one veto reason (design Logging: 'every veto with its
    rule'). The detail text names the specific absolute / relative / window /
    future-collision / wall-bar rule; the classifier maps the vocabulary the
    decision loop emits onto those names."""
    detail = (reason.detail or "").lower()
    if "no safe action" in detail:
        return "no_safe_action"
    if "window-only relative" in detail:
        return "window"
    if "assembly bar" in detail or "wall bar" in detail:
        return "wall_bar"
    if "p_future_collision" in detail or "future collision" in detail:
        return "future_collision"
    if "relative" in detail:
        return "relative"
    if "simultaneous wall cap" in detail or "max_walls" in detail:
        return "wall_cap"
    if "feasibility" in detail or "group_cap" in detail or "frame" in detail:
        return "wall_feasibility"
    if "absolute" in detail or "collision" in detail or "min ttc" in detail or "unsafe" in detail:
        return "absolute"
    if reason.step == "wall":
        return "wall_bar"
    return "safety"


# --------------------------------------------------------------------- budget
def project_run_cost(
    config: ClosedLoopConfig, *, n_cavs: int, planner_config: PlannerConfig
) -> dict:
    """Up-front HARD-cap projection (design Budget): ``estimate_cost`` over
    every decision point, including ``process_start_s``; master seconds =
    the full horizon. Refuses (:class:`BudgetRefusedError`) when the
    projected simulated seconds exceed ``config.max_sim_s``. The declared
    upper-bound lookahead shape is one full-cap evaluation group and one
    wall candidate; a run under this projection stays under the cap.
    T-A: a ``--fixed-policy`` REFERENCE run has NO decision points and NO
    lookahead branches -- its projection is the master horizon ONLY."""

    n_decisions = 0
    t = float(config.warmup_s)
    horizon = float(config.horizon_s)
    if config.fixed_policy is None:
        while t < horizon - 1e-9:
            n_decisions += 1
            t += float(config.decision_period_s)
    master_sim_s = horizon
    lookahead_sim_s = 0.0
    branches = 0
    process_start_seconds = 0.0
    # PL-04e2: the individual stage's decision units -- the cap (when set)
    # bounds how many roster CAVs the planner may move, so the projection's
    # N is the capped count, and walls off prices no wall candidate.
    n_units = max(1, int(n_cavs))
    if planner_config.max_decision_units is not None:
        n_units = min(n_units, int(planner_config.max_decision_units))
    if config.predictor == "perfect_sim" and n_decisions:
        group = max(1, min(int(n_units), int(planner_config.group_cap)))
        walls_candidates = 0 if planner_config.max_walls == 0 else 1
        estimate = estimate_cost(
            N=int(n_units),
            K=int(config.k_samples),
            W=walls_candidates,
            n_speeds=len(planner_config.wall_speeds_mps),
            n_releases=len(planner_config.wall_release_options_s),
            H=float(planner_config.horizon_s),
            r=1.0,
            workers=1,
            group_sizes=[group],
            decision_points=n_decisions,
            process_start_s=float(config.process_start_s),
        )
        lookahead_sim_s = estimate.sim_seconds
        branches = estimate.branches
        process_start_seconds = estimate.process_start_seconds
    projected = master_sim_s + lookahead_sim_s
    projection = {
        "decision_points": n_decisions,
        "decision_units": int(n_units),
        "master_sim_s": master_sim_s,
        "lookahead_sim_s": lookahead_sim_s,
        "projected_sim_s": projected,
        "cap_s": float(config.max_sim_s),
        "branches": branches,
        "process_start_seconds": process_start_seconds,
    }
    if projected > float(config.max_sim_s):
        raise BudgetRefusedError(
            f"cost projection {projected:g} sim-s (master {master_sim_s:g} + "
            f"lookahead {lookahead_sim_s:g}, {n_decisions} decision points) exceeds "
            f"the --max-sim-s cap {config.max_sim_s:g}: refusing to start"
        )
    return projection


class SimLedger:
    """Spent simulated seconds: master physics + every lookahead branch."""

    def __init__(self, cap_s: float):
        self.cap_s = float(cap_s)
        self.master_s = 0.0
        self.lookahead_s = 0.0

    @property
    def total(self) -> float:
        return self.master_s + self.lookahead_s

    def add_master(self, seconds: float) -> None:
        self.master_s += float(seconds)

    def add_lookahead(self, seconds: float) -> None:
        self.lookahead_s += float(seconds)

    def over_cap(self) -> bool:
        return self.total >= self.cap_s - 1e-9

    def to_json(self) -> dict:
        return {"master_s": self.master_s, "lookahead_s": self.lookahead_s, "total_s": self.total}


# ------------------------------------------------------------ metered model
class _MeteredWorldModel:
    """WorldModel wrapper that prices each predict_many call in sim-seconds.

    ``sim_seconds_fn(n_plans, k, horizon_s)`` prices one lookahead call; the
    default perfect-sim pricing is (1 + n_plans) * K * horizon (one branch
    per plan plus the shared native reference, PL-10d fresh process each);
    analytic/learned predictors price 0 (no simulation). Also captures the
    first prediction meta of the run for the predictor id/sha records."""

    def __init__(self, inner, sim_seconds_fn: Callable[[int, int, float], float]):
        self._inner = inner
        self._sim_seconds_fn = sim_seconds_fn
        self.decision_sim_s = 0.0
        self.decision_plans = 0
        self.decision_calls = 0
        self._in_decision = False
        self.first_meta: Optional[dict] = None
        self.calls: list = []

    def begin_decision(self) -> None:
        self.decision_sim_s = 0.0
        self.decision_plans = 0
        self.decision_calls = 0
        self._in_decision = True

    def end_decision(self) -> float:
        self._in_decision = False
        return self.decision_sim_s

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def predict_many(self, state_samples, plans, controller_state=None, **kw):
        outs = self._inner.predict_many(state_samples, plans, controller_state=controller_state, **kw)
        sim_s = float(
            self._sim_seconds_fn(len(plans), len(state_samples), float(kw.get("horizon_s", 600.0)))
        )
        if self._in_decision:
            self.decision_sim_s += sim_s
            self.decision_plans += len(plans)
            self.decision_calls += 1
        if self.first_meta is None and outs:
            meta = getattr(outs[0], "meta", None)
            if meta is not None:
                self.first_meta = dict(meta)
        self.calls.append({"n_plans": len(plans), "n_samples": len(state_samples), "sim_s": sim_s})
        return outs


def _perfect_sim_pricing(n_plans: int, k: int, horizon_s: float) -> float:
    return float(1 + n_plans) * int(k) * float(horizon_s)


def _zero_pricing(n_plans: int, k: int, horizon_s: float) -> float:
    del n_plans, k, horizon_s
    return 0.0


# ----------------------------------------------------------- master harness
class FakeMasterHarness:
    """Deterministic scripted master world (the fake behind every test).

    Physics: 0.5 s steps; each CAV moves at its commanded speed (native 20,
    damping 19, pi-external-u 18, wall members at the wall speed). State is
    fully captured by (time, positions, assignments, walls), so guarded
    save/load round-trips are exact and resume runs are bit-identical. The
    SUMO-state surface (``master_api.simulation.saveState``) writes a
    full-precision RNG-carrying state file, so the REAL exactness guard
    (:func:`require_exact_state_file`) passes on it."""

    ACTION_SPEEDS = {"native": 20.0, "damping": 19.0, "pi-external-u": 18.0}
    STATE_SCHEMA = "pl04_fake_master_state/1"

    def __init__(
        self,
        cav_ids: Sequence[str],
        *,
        seed: int = 0,
        dt_s: float = PHYSICS_DT_S,
        k_samples: int = 1,
        init_s_m: float = 100.0,
        merge_observed_fraction: float = 1.0,
        lanes: Optional[Mapping[str, float]] = None,
        distances_to_merge_m: Optional[Mapping[str, float]] = None,
        emit_control_rows: bool = False,
    ):
        self.cav_ids = tuple(str(c) for c in cav_ids)
        if not self.cav_ids:
            raise BuildFailedError("FakeMasterHarness needs a non-empty CAV roster")
        self.seed = int(seed)
        self.dt_s = float(dt_s)
        self.k_samples = int(k_samples)
        self.time = 0.0
        self.pos = {c: float(init_s_m) + 10.0 * i for i, c in enumerate(self.cav_ids)}
        self.lane_of = dict(lanes) if lanes else {c: float(i % 2) for i, c in enumerate(self.cav_ids)}
        self.assignments = {c: "native" for c in self.cav_ids}
        # PL-04e2: the RSU frame distances the decision-unit cap ranks
        # (empty = no distance information -> the cap holds everyone)
        self.distances_to_merge_m = dict(distances_to_merge_m or {})
        self.walls: dict = {}  # wall_id -> {members, speed_mps, phase, applied_at_s}
        self.merge_observed_fraction = float(merge_observed_fraction)
        self.load_state_calls: list = []
        self.save_state_calls: list = []
        self.release_control_calls: list = []
        self.started = False
        self.closed = False
        self.simulation = _FakeMasterSim(self)
        # T-A: opt-in per-control-tick command rows (the fake mirror of the
        # REAL master control write: native rows are the ownership
        # 'selected_native' record, damping/pi rows are commanded; every
        # live roster CAV every control tick). Default False keeps every
        # historical fake behaviour byte-identical.
        self.emit_control_rows = bool(emit_control_rows)
        self._control_rows_tick: Optional[float] = None

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True

    @property
    def master_api(self):
        """The traci-like master surface (snapshotter saveState/getTime)."""
        return self

    # ------------------------------------------------------------- stepping
    def now_s(self) -> float:
        return self.time

    def _speed_of(self, cav: str) -> float:
        for wall in self.walls.values():
            if cav in wall["members"]:
                return float(wall["speed_mps"])
        return self.ACTION_SPEEDS[self.assignments[cav]]

    def step_physics(self) -> dict:
        vehicles = {}
        for cav in self.cav_ids:
            speed = self._speed_of(cav)
            self.pos[cav] = self.pos[cav] + speed * self.dt_s
            vehicles[cav] = {
                "s_m": self.pos[cav],
                "v_mps": speed,
                "lane_index": self.lane_of[cav],
                "accel_mps2": 0.0,
            }
        self.time = self.time + self.dt_s
        return {
            "kind": "physics",
            "t_s": self.time,
            "dt_s": self.dt_s,
            "cav_speed_mode": CAV_SPEED_MODE,
            "vehicles": vehicles,
        }

    # ------------------------------------------------------------ observation
    def observe(self) -> Tuple[list, dict]:
        member_of = {}
        for wall_id, wall in self.walls.items():
            for member in wall["members"]:
                member_of[member] = wall_id
        controller_state = {}
        for cav in self.cav_ids:
            wall_id = member_of.get(cav)
            if wall_id is not None:
                wall = self.walls[wall_id]
                controller_state[cav] = {
                    "action": "wall",
                    "wall_id": wall_id,
                    "wall_phase": wall["phase"],
                    "time_in_phase_s": self.time - wall["applied_at_s"],
                    "wall_speed_mps": wall["speed_mps"],
                    "locked": True,
                }
            else:
                controller_state[cav] = {
                    "action": self.assignments[cav],
                    "wall_id": None,
                    "wall_phase": None,
                    "time_in_phase_s": 0.0,
                    "wall_speed_mps": None,
                    "locked": False,
                }
        n_observed = int(round(self.merge_observed_fraction * self.k_samples))
        samples = []
        for i in range(self.k_samples):
            samples.append(
                SimpleNamespace(
                    hybrid={"tracked": {c: self.pos[c] for c in self.cav_ids}},
                    arrivals={"sample_id": f"k{i}", "departures": ()},
                    meta={
                        "sample_id": f"k{i}",
                        "time_s": self.time,
                        "merge_observed": i < n_observed,
                        "coverage": 1.0,
                    },
                    frames={c: self.time for c in self.cav_ids},
                    frames_distance_to_merge_m=dict(self.distances_to_merge_m),
                )
            )
        return samples, controller_state

    # -------------------------------------------------------------- execution
    def drain_control_rows(self) -> list:
        """Per-tick executed command rows (the shared control write). The
        fake executes assignments at apply_plan time, so it never has
        pending control rows -- UNLESS ``emit_control_rows`` is set (T-A):
        then every 1 s control tick issues one row per live roster CAV with
        the real ownership semantics (native -> command 'native' with
        fallback 'selected_native'; damping/pi -> command 'damping' with
        the assigned action telling a PI command apart)."""
        if not self.emit_control_rows:
            return []
        now = self.time
        if abs(now / CONTROL_DT_S - round(now / CONTROL_DT_S)) > 1e-6:
            return []  # a physics tick between control ticks: no write
        if self._control_rows_tick == now:
            return []  # this control tick was already drained
        self._control_rows_tick = now
        rows = []
        for cav in self.cav_ids:
            action = self.assignments.get(cav, "native")
            rows.append(
                {
                    "kind": "control",
                    "t_s": float(now),
                    "plan_id": None,
                    "cav": cav,
                    "command": "native" if action == "native" else "damping",
                    "assigned_action": action,
                    "target_speed_mps": None,
                    "fallback": "selected_native" if action == "native" else None,
                    "released_to_native": False,
                    "owner": "planner",
                    "source": "master_control_tick",
                }
            )
        return rows

    def release_control(self) -> None:
        """End-of-run ownership handback seam: the runner calls this in the
        run's ``finally`` (normal end, interrupt or error). The fake
        records the call."""
        self.release_control_calls.append(self.time)

    def apply_plan(self, dispatcher_plan, t_s: float, *, note: str = "") -> list:
        plan_walls = {w.wall_id: w for w in dispatcher_plan.walls}
        for wall_id in list(self.walls):  # omission after hold = release
            if wall_id not in plan_walls:
                del self.walls[wall_id]
        for wall_id, wall in plan_walls.items():
            if wall_id in self.walls:
                self.walls[wall_id]["phase"] = wall.phase
            else:
                self.walls[wall_id] = {
                    "members": tuple(wall.members),
                    "speed_mps": float(wall.wall_speed_mps),
                    "phase": wall.phase,
                    "applied_at_s": float(t_s),
                }
        member_of = {m: wid for wid, wall in self.walls.items() for m in wall["members"]}
        rows = []
        for cav in self.cav_ids:
            action = dispatcher_plan.assignments.get(cav)
            fallback = None
            if action is None:
                fallback = "missing_assignment"
                action = "native"
            if cav not in self.assignments:
                fallback = "not_in_roster"
            self.assignments[cav] = action
            wall_id = member_of.get(cav)
            rows.append(
                {
                    "kind": "command",
                    "t_s": float(t_s),
                    "plan_id": dispatcher_plan.plan_id,
                    "cav": cav,
                    "command": action,
                    "wall_id": wall_id,
                    "wall_speed_mps": None if wall_id is None else float(
                        self.walls[wall_id]["speed_mps"]
                    ),
                    "fallback": fallback,
                    "owner": "planner",
                    "commit_horizon_s": dispatcher_plan.commit_horizon_s,
                    "note": note,
                }
            )
        return rows

    # ------------------------------------------------------ guarded state io
    def _state_xml(self) -> str:
        vehicles = "".join(
            f'<vehicle id="{c}" pos="{self.pos[c]:.17f}" speed="{self._speed_of(c):.17f}"/>'
            for c in self.cav_ids
        )
        return f'<state><rngState seed="{self.seed}"/>{vehicles}</state>'

    def save_state(self, state_dir) -> dict:
        state_dir = Path(state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        state_file = state_dir / "state.xml"
        world_file = state_dir / "world.json"
        state_file.write_text(self._state_xml(), encoding="utf-8")
        world = {
            "time": self.time,
            "pos": dict(self.pos),
            "lane_of": dict(self.lane_of),
            "assignments": dict(self.assignments),
            "walls": {
                wid: {
                    "members": list(w["members"]),
                    "speed_mps": w["speed_mps"],
                    "phase": w["phase"],
                    "applied_at_s": w["applied_at_s"],
                }
                for wid, w in self.walls.items()
            },
        }
        world_file.write_text(_dumps(world), encoding="utf-8")
        require_exact_state_file(state_file)  # the REAL exactness guard
        manifest = {
            "schema": self.STATE_SCHEMA,
            "branch_point_t0_s": self.time,
            "sumo_state_file": "state.xml",
            "sumo_state_sha256": sha256_file(state_file),
            "world_state_file": "world.json",
            "world_state_sha256": sha256_file(world_file),
            "state_options": {
                "save_state_precision_decimals": 17,
                "rng_state_in_file": True,
                "required_cli": ["--save-state.precision", "17", "--save-state.rng", "true"],
            },
        }
        (state_dir / "STATE_MANIFEST.json").write_text(_dumps(manifest), encoding="utf-8")
        self.save_state_calls.append(str(state_dir))
        return {
            "saved_state_dir": str(state_dir),
            "saved_state_id": sha256_text(_dumps(manifest) + manifest["sumo_state_sha256"]),
            "manifest": manifest,
            "t0_s": self.time,
        }

    def load_state(self, saved: Mapping) -> None:
        state_dir = Path(saved["saved_state_dir"])
        manifest = dict(saved["manifest"])
        if manifest.get("schema") != self.STATE_SCHEMA:
            raise ValueError(f"unexpected master state schema {manifest.get('schema')!r}")
        state_file = state_dir / "state.xml"
        world_file = state_dir / "world.json"
        require_exact_state_file(state_file)
        if sha256_file(state_file) != manifest.get("sumo_state_sha256"):
            raise ValueError("master state.xml fails its digest")
        if sha256_file(world_file) != manifest.get("world_state_sha256"):
            raise ValueError("master world.json fails its digest")
        world = json.loads(world_file.read_text(encoding="utf-8"))
        self.time = float(world["time"])
        self.pos = {c: float(v) for c, v in world["pos"].items()}
        self.lane_of = {c: float(v) for c, v in world["lane_of"].items()}
        self.assignments = {c: str(v) for c, v in world["assignments"].items()}
        self.walls = {
            wid: {
                "members": tuple(w["members"]),
                "speed_mps": float(w["speed_mps"]),
                "phase": w["phase"],
                "applied_at_s": float(w["applied_at_s"]),
            }
            for wid, w in world["walls"].items()
        }
        # the master's traci-surface ledger: a resume loadState happens ONCE
        self.simulation.loaded.append(str(state_dir))
        self.load_state_calls.append(str(state_dir))


class _FakeMasterSim:
    """traci.simulation surface of the fake master (saveState + getTime)."""

    def __init__(self, harness: FakeMasterHarness):
        self._harness = harness
        self.saved: list = []
        self.loaded: list = []

    def saveState(self, path) -> None:
        Path(path).write_text(self._harness._state_xml(), encoding="utf-8")
        self.saved.append(str(path))

    def loadState(self, path) -> None:
        self.loaded.append(str(path))

    def getTime(self) -> float:
        return self._harness.time


class RealMasterHarness:
    """The REAL SUMO master (faked-surface tests only; first real execution
    is the coordinated <=80 sim-s smoke).

    Owns the launch described by the design Loop section: one master SUMO
    process on a free port, a scenario COPY of a campaign-format world (the
    PL-10c guard must pass), CAV speedMode 27 and
    ``--save-state.precision 17 --save-state.rng true``. PL-04d closes the
    loop for real:

    - ``step_physics`` executes the CURRENT dispatcher plan on every 1 s
      control tick through :class:`SessionControlWrite` -- the SAME shared
      write the lookahead branches use (sensor frames for live roster CAVs
      -> ``CommandOwnership`` (cav_speed_mode 27) -> ``session.step``);
      physics ticks in between write nothing. The issued command rows drain
      into ``dispatcher_rows.jsonl`` (``drain_control_rows``, consumed by
      the runner after every physics step).
    - ``observe`` builds the K=1 sample from the RSU sensor frames ONLY
      (real frames/coverage/merge_observed per contract 8.15; ground truth
      beyond sensor range never enters the sample).
    - ``release_control`` (the runner calls it in the run's ``finally`` --
      normal end, interrupt or error) hands every still-owned CAV back to
      SUMO through ``owner.release_all``.

    State save/restore goes through the REAL guarded ``TraciStateSnapshotter``
    (loadState only on resume). Only this harness's own Popen children are
    ever signalled (never by image name or pattern)."""

    def __init__(
        self,
        *,
        sumo_binary,
        scenario_cfg,
        scenario_routes,
        roster,
        adapter_config,
        seed,
        predictor: str = "perfect_sim",
        hybrid_builder: Optional[Callable] = None,
        merge_lane_distances: Optional[Mapping[str, float]] = None,
        base_state: Optional[Mapping] = None,
    ):
        self.sumo_binary = str(sumo_binary)
        self.scenario_cfg = Path(scenario_cfg)
        self.scenario_routes = [Path(p) for p in scenario_routes]
        self.roster = tuple(str(c) for c in roster)
        self.adapter_config = adapter_config
        # PL-04c: the run's REQUIRED seed (no default) -- the master and
        # every lookahead branch launch with the SAME --seed value
        self.seed = int(seed)
        # PL-04d: which predictor consumes the observe() samples. The
        # learned predictor REQUIRES the injected kit/p2C hybrid-state
        # builder; there is none in this worktree, so the default wiring
        # (perfect_sim) is the only buildable one and observe fails closed
        # with LearnedObservationUnavailableError otherwise.
        if predictor not in PREDICTORS:
            raise ValueError(f"predictor must be one of {list(PREDICTORS)}")
        self.predictor = str(predictor)
        self._hybrid_builder = hybrid_builder
        # RSU merge-region input (lane_id -> lane-end distance to the merge,
        # metres) forwarded to the shared sensor. PL-04j: an EXPLICITLY
        # injected NON-empty map is honoured as-is ("injected"); anything
        # else (no injection, an empty injection) is resolved at start()
        # from the LIVE master connection via ``scenarios.observation_map``
        # ("observation_map") -- BEFORE this fix the real path always ran
        # with an EMPTY map, so no frame could coverage-reach the merge
        # region, merge_observed resolved False (contract 8.15 fail closed)
        # and the planner abstained at every decision. start() raises the
        # NAMED MergeMapUnavailableError when the resolved map is empty --
        # never a silent empty-map run.
        self._merge_lane_distances = (
            dict(merge_lane_distances) if merge_lane_distances else None
        )
        # PL-04e2 --base-run: a validated base state (resolve_base_run ->
        # BaseRunState.as_saved_dict()); start() lands the FRESH master
        # process on its t0 with ONE guarded loadState -- the START of the
        # run, never a resume.
        if base_state is not None and not (
            isinstance(base_state, Mapping) and "saved_state_dir" in base_state
        ):
            raise ValueError(
                "base_state must be a BaseRunState.as_saved_dict() mapping "
                "(resolve_base_run validates the directory)"
            )
        self.base_state = dict(base_state) if base_state is not None else None
        self.proc = None
        self.port = None
        self.snapshotter = None
        self.session = None
        self.owner = None
        self.launch_record: Optional[dict] = None
        self._control_write: Optional[SessionControlWrite] = None
        self._pending_control_rows: list = []
        self._prev_lane_ids: dict = {}
        # T-B receipt evidence
        self.harness_state_restored = False
        self.restore_source_manifest_sha256: Optional[str] = None
        self.restore_evidence: Optional[dict] = None
        self._started = False
        # PL-04i: every start()-set attribute exists (None) from
        # construction on -- the runner builds all components BEFORE
        # start(), so a pre-start read sees a declared None and the
        # master_api property raises the NAMED HarnessNotStartedError
        # instead of an AttributeError.
        self._traci = None

    def _launch(self):
        import socket
        import subprocess

        import traci  # noqa: PLC0415  (lazy: this harness IS the SUMO touchpoint)

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        cmd = [
            self.sumo_binary,
            "-c",
            str(self.scenario_cfg),
            "--remote-port",
            str(self.port),
            "--seed",
            str(self.seed),
            "--step-length",
            str(PHYSICS_DT_S),
            "--save-state.precision",
            "17",
            "--save-state.rng",
            "true",
            "--no-warnings",
            "true",
        ]
        self.proc = subprocess.Popen(  # noqa: S603  (fixed argv, no shell)
            cmd,
            creationflags=BELOW_NORMAL_PRIORITY_CLASS,  # design Guards: low priority
        )
        self.launch_record = {
            "kind": "master",
            "label": "pl04_master",
            "port": self.port,
            "pid": self.proc.pid,
            "cmd": list(cmd),
        }
        # PL-04l: the master holds its OWN traci Connection object -- never
        # the traci module. Real traci.init(label=...) defaults to
        # doSwitch=True, which SWITCHES the module-level current connection:
        # after a lookahead branch's init the module routes to that branch,
        # and after the branch closes the master's module-level calls die
        # with FatalTraCIError('Connection already closed') (the cl_smoke_2
        # defect, second predict_many of the first decision). doSwitch=False
        # + getConnection keeps the module untouched, and every master
        # consumer below receives THIS connection object only.
        traci.init(
            port=self.port,
            host="127.0.0.1",
            numRetries=4,
            proc=self.proc,
            label="pl04",
            doSwitch=False,
        )
        return traci.getConnection("pl04")

    def _resolve_merge_map(self, api) -> "tuple[dict, str]":
        """The RSU merge-region lane map for THIS connection (PL-04j).

        An explicitly injected map wins ("injected"); otherwise the map is
        computed ON THE LIVE CONNECTION via ``scenarios.observation_map``
        ("observation_map") -- the same construction the PL-10 re-smoke and
        run_confirm use for their sensors. An empty or unresolvable map
        raises the NAMED :class:`MergeMapUnavailableError`: a real run must
        never silently start with an empty map, because contract 8.15's
        fail-closed merge_observed would then abstain the planner at every
        decision (the silent PL-04 defect this closes)."""
        if self._merge_lane_distances:
            return dict(self._merge_lane_distances), "injected"
        from traffic_shepherd.scenarios import observation_map  # noqa: PLC0415

        try:
            resolved = dict(observation_map(api, self.adapter_config))
        except Exception as exc:  # pragma: no cover - defensive: wrapped loud
            raise MergeMapUnavailableError(
                "RealMasterHarness: the RSU merge-region lane map could not "
                f"be resolved on the live connection (observation_map: {exc}"
                f"; {type(exc).__name__}); refusing to start -- an empty map "
                "would leave merge_observed False and the planner would "
                "silently abstain at every decision"
            ) from exc
        if not resolved:
            raise MergeMapUnavailableError(
                "RealMasterHarness: observation_map resolved an EMPTY "
                "merge-region lane map on the live connection; refusing to "
                "start -- no frame could coverage-reach the merge region, "
                "merge_observed would resolve False (contract 8.15, fail "
                "closed) and the planner would silently abstain at every "
                "decision"
            )
        return resolved, "observation_map"

    def start(self) -> None:
        traci = self._launch()
        self._traci = traci
        # PL-04j: resolve the RSU merge-region map on the LIVE master
        # connection RIGHT AFTER connect (explicitly injected maps win) and
        # record source + size on the launch record (the receipt's
        # sumo_launches master entry). An empty resolution refuses the run
        # here -- before any branch could launch.
        merge_map, merge_map_source = self._resolve_merge_map(traci)
        self._merge_lane_distances = merge_map
        self.launch_record["merge_lane_map_source"] = merge_map_source
        self.launch_record["merge_lane_map_size"] = len(merge_map)
        self.snapshotter = TraciStateSnapshotterForHarness(
            traci,
            self.adapter_config,
            roster=self.roster,
            scenario_routes=[str(p) for p in self.scenario_routes],
            seed=self.seed,
            harness_state_factory=self.capture_harness_state,
            physics_row_factory=self.snapshot_physics_row,
        )
        controller = self.snapshotter.damping_controller()
        # PL-04k: the ONE shared construction (registered PI session bound)
        self.session = build_mixed_session(
            self.adapter_config, self.roster, controller=controller
        )
        # PL-04d: the master control write IS the shared per-connection
        # write (PL-04c's class): sensor frames -> CommandOwnership
        # (cav_speed_mode 27) -> session.step. The owner it creates is THE
        # master's sole longitudinal writer. PL-04j: the sensor carries the
        # RESOLVED merge-region map -- without it no frame can
        # coverage-reach the merge region and the planner abstains.
        self._control_write = SessionControlWrite(
            traci,
            self.session,
            roster=self.roster,
            adapter_config=self.adapter_config,
            lane_end_distance_to_merge=self._merge_lane_distances,
        )
        self.owner = self._control_write.owner
        if self.base_state is not None:
            # PL-04e2 --base-run start: ONE guarded loadState lands this
            # fresh process on the base run's t0 (validated by the real
            # snapshotter restore: schema, exactness options, digests, SUMO
            # version drift, and the restored clock). This is the START of
            # the run -- the receipt records start_mode="base_state".
            self.snapshotter.restore(
                self.base_state, plan=None, draw_id="base_state", api=traci
            )
            # T-B (#1): restore the harness's cross-tick python state
            # EXACTLY like the campaign's run_suffix (ownership, loop state,
            # previous-row lane seeding). The FIRST control tick is then
            # run AT t0, BEFORE the first physics step (step_physics runs
            # its tick before every step on the control grid), so a CAV the
            # base run had damping-owned is never left uncommanded.
            self._restore_base_harness_state()
        for cav in self.roster:
            if cav in set(traci.vehicle.getIDList()):
                traci.vehicle.setSpeedMode(cav, CAV_SPEED_MODE)
        self._started = True

    def _restore_base_harness_state(self) -> None:
        manifest = self.base_state.get("manifest") or {}
        state = manifest.get("harness_state")
        if not isinstance(state, Mapping) or "ownership" not in state:
            raise BaseRunRefusedError(
                "the base saved-state manifest carries no harness_state.ownership; "
                "refusing to start (an uncommanded first control interval would "
                "silently perturb the run)"
            )
        evidence = restore_harness_state(self._control_write, self.session, state)
        self._prev_lane_ids = previous_lane_ids(manifest.get("physics_row_at_t0"))
        self.harness_state_restored = True
        self.restore_source_manifest_sha256 = (
            str(self.base_state.get("manifest_sha256") or "") or None
        )
        self.restore_evidence = evidence

    # ---------------------------------------------- T-B carried-state capture
    def capture_harness_state(self, controller_state=None, kept_wall_ids=()) -> dict:
        """The master's carried control-loop state (snapshot harness_state).

        Captured at a decision point: BEFORE the control tick of that time,
        i.e. exactly the state a branch restored at t0 needs before ITS
        first tick (which runs at t0, plan applied first)."""
        state = capture_harness_state(self._control_write, self.session, self.roster)
        state["controller_state"] = json.loads(
            json.dumps(dict(controller_state or {}), default=repr, sort_keys=True)
        )
        state["kept_wall_ids"] = list(kept_wall_ids)
        return state

    def snapshot_physics_row(self, t0_s: float) -> dict:
        """Slim previous-row (lane ids) for the snapshot manifest."""
        return {
            "time_s": float(t0_s),
            "vehicles": {v: {"lane_id": lane} for v, lane in self._prev_lane_ids.items()},
        }

    def close(self) -> None:
        if self.proc is not None:
            try:
                # PL-04i: _traci is None when the launch itself failed
                # (Popen ran, traci.init never connected) -- retire the
                # child without the traci close, never an AttributeError
                if self._traci is not None:
                    self._traci.close()
            finally:
                self.proc.wait(timeout=30)  # own child only, never by name
                self.proc = None

    def now_s(self) -> float:
        return float(self._traci.simulation.getTime())

    @property
    def first_control_tick_s(self) -> Optional[float]:
        """Sim time of the master's first control tick (T-B: at t0, before
        the first physics step, on a base-run start)."""
        write = self._control_write
        return None if write is None else write.first_control_tick_s

    def step_physics(self) -> dict:
        traci = self._traci
        # T-B (#5): the control tick runs BEFORE the physics step (the
        # campaign order: tick at t, then the 0.5 s substeps). A plan
        # chosen at decision time t (applied to the session just before
        # this call) therefore takes effect at tick t; a lookahead branch
        # restored at t0 applies the plan in the SAME order (plan, tick,
        # step).
        self._run_control_tick(float(traci.simulation.getTime()))
        traci.simulationStep()
        now = float(traci.simulation.getTime())
        vehicles = {}
        roster = set(self.roster)
        prev_lane = self._prev_lane_ids
        lane_now = {}
        for raw in traci.vehicle.getIDList():
            vid = str(raw)
            lane_id = str(traci.vehicle.getLaneID(vid))
            lane_now[vid] = lane_id
            lane_pos = float(traci.vehicle.getLanePosition(vid))
            speed = float(traci.vehicle.getSpeed(vid))
            accel = float(traci.vehicle.getAcceleration(vid))
            vehicles[vid] = {
                # PL-04k: the campaign per-vehicle fields (physics_steps.jsonl)
                "road_id": str(traci.vehicle.getRoadID(vid)),
                "lane_id": lane_id,
                "lane_position_m": lane_pos,
                "speed_mps": speed,
                "acceleration_mps2": accel,
                "type_id": str(traci.vehicle.getTypeID(vid)),
                "speed_mode": int(traci.vehicle.getSpeedMode(vid)),
                "lane_changed": bool(vid in prev_lane and prev_lane[vid] != lane_id),
                # the runner's original keys, kept (same values, no conflict)
                "s_m": lane_pos,
                "v_mps": speed,
                "lane_index": float(traci.vehicle.getLaneIndex(vid)),
                "accel_mps2": accel,
            }
        self._prev_lane_ids = lane_now
        sim = traci.simulation
        departed = [str(v) for v in sim.getDepartedIDList()]
        arrived = [str(v) for v in sim.getArrivedIDList()]
        live = sorted(vehicles)
        cav_live = sorted(v for v in live if v in roster)
        committed = dict(getattr(self.session, "_committed", {}) or {})
        return {
            "kind": "physics",
            "t_s": now,
            "dt_s": PHYSICS_DT_S,
            "cav_speed_mode": CAV_SPEED_MODE,
            # PL-04k: the campaign row-level fields
            "time_s": now,
            "live_ids": live,
            "departed_ids": departed,
            "arrived_ids": arrived,
            "colliding_ids": [str(v) for v in sim.getCollidingVehiclesIDList()],
            "cav_ids": sorted(roster),
            "speed_modes": {v: vehicles[v]["speed_mode"] for v in cav_live},
            "cav_actions": {
                v: {
                    "owner": committed.get(v, "native"),
                    "action": committed.get(v, "native"),
                    "commanded_speed_mps": None,
                    "controller_state_digest": None,
                    "wall_id": None,
                    "wall_membership": None,
                    "wall_phase": None,
                }
                for v in cav_live
            },
            "boundary_crossings": {
                "mainline_entry": sum(1 for v in departed if v.startswith("mainline.")),
                "mainline_exit": sum(1 for v in arrived if v.startswith("mainline.")),
                "ramp_entry": sum(1 for v in departed if v.startswith("ramp.")),
            },
            "vehicles": vehicles,
            "starting_teleports": [str(v) for v in sim.getStartingTeleportIDList()],
            "ending_teleports": [str(v) for v in sim.getEndingTeleportIDList()],
        }

    def _run_control_tick(self, now: float) -> None:
        """Design Loop step 2, for real: on the 1 s control grid the shared
        control write executes the session's CURRENT committed plan (sensor
        capture for live roster CAVs -> ``session.step(owner, frames,
        now_s, live_ids)``); the issued command records are kept as
        dispatcher rows for the runner to drain. Physics ticks between
        control ticks write nothing."""
        write = self._control_write
        if write is None or not self._started:
            return
        records = write.tick(now)
        if records is None:
            return
        plan_id = getattr(self.session, "_plan_id", None)
        for record in records:
            self._pending_control_rows.append(
                {
                    "kind": "control",
                    "t_s": float(now),
                    "plan_id": plan_id,
                    "cav": str(record.vehicle_id),
                    "command": getattr(record, "owner", None),
                    "assigned_action": dict(
                        getattr(self.session, "_committed", {}) or {}
                    ).get(str(record.vehicle_id)),
                    "target_speed_mps": getattr(record, "issued_target_speed_mps", None),
                    "fallback": getattr(record, "fallback_reason", None),
                    "released_to_native": bool(getattr(record, "released_to_native", False)),
                    "owner": "planner",
                    "source": "master_control_tick",
                }
            )

    def drain_control_rows(self) -> list:
        """Executed per-tick command rows since the last drain (the runner
        emits them into dispatcher_rows.jsonl after every physics step)."""
        rows = self._pending_control_rows
        self._pending_control_rows = []
        return rows

    def release_control(self) -> None:
        """``owner.release_all`` -- the runner calls this in the run's
        ``finally`` (normal end, interrupt or error) BEFORE the process is
        closed, so no CAV stays speed-commanded after the run."""
        if self._control_write is not None:
            self._control_write.release()

    def observe(self) -> Tuple[list, dict]:
        """The REAL RSU observation path (K=1, design Components).

        Built from the SumoLocalSensor frames ONLY -- CAV self state +
        locally detected neighbours; the traci ground-truth vehicle list is
        consulted for liveness of the ROSTER captures, never read INTO the
        sample (vehicles beyond sensor range are absent, never invented):

        - ``sample.frames``: {vehicle: last-frame time} from the captured
          frames (walls.py's freshness filter reads this); a roster CAV
          that is not live or whose capture raised ValueError has NO frame;
        - ``meta.coverage``: the roster share with a frame at this tick;
        - ``meta.merge_observed``: contract 8.15, coverage-based and
          occupancy-independent -- True iff a captured frame's sensor
          coverage reaches the merge region (``distance_to_merge_m <=
          front_coverage_m``); no merge-region input in any frame resolves
          None -> False (fail closed) as does no frame reaching it; the
          reason travels in ``meta.merge_observed_reason``;
        - ``meta.frame_age_s`` / ``meta.frame_missing``: per-CAV receipt
          age and missingness.

        ``sample.hybrid`` follows the predictor: the learned predictor is
        built by the injected kit/p2C hybrid-state builder -- without one
        this raises :class:`LearnedObservationUnavailableError` at sample
        build time; the perfect_sim oracle reads truth through its own
        branch snapshot, so its ``hybrid`` is a documented opaque marker
        while frames/coverage/meta stay the real sensor path."""
        if self.predictor == "learned" and self._hybrid_builder is None:
            raise LearnedObservationUnavailableError(
                "RealMasterHarness.observe: predictor='learned' needs the "
                "kit/p2C hybrid-state builder wired in (hybrid_builder); "
                "none is available in this worktree, so the K=1 sample "
                "cannot be built honestly -- refusing (fail closed)"
            )
        if not self._started or self._control_write is None:
            raise ClosedLoopError("RealMasterHarness.observe called before start()")
        now = self.now_s()
        live = {str(v) for v in self._traci.vehicle.getIDList()}
        frames: "dict[str, Any]" = {}
        for cav in sorted(self.roster):
            if cav not in live:
                continue  # not yet inserted / departed: no frame (missing)
            try:
                frames[cav] = self._control_write.sensor.capture(cav, now_s=now)
            except ValueError:
                frames[cav] = None  # explicit unavailability, never invented
        good = {cav: f for cav, f in frames.items() if f is not None}
        frames_map = {cav: float(f.timestamp_s) for cav, f in good.items()}
        # PL-04e2: the per-CAV distance to the merge from the SAME frames
        # (the decision-unit cap ranks decision units by it; a frame
        # without the field contributes nothing, never an invented value)
        distances_map = {
            cav: float(f.distance_to_merge_m)
            for cav, f in good.items()
            if getattr(f, "distance_to_merge_m", None) is not None
        }
        coverage = (len(frames_map) / len(self.roster)) if self.roster else 0.0
        frame_age_s = {cav: max(0.0, now - ts) for cav, ts in frames_map.items()}
        frame_missing = [cav for cav in sorted(self.roster) if cav not in frames_map]

        merge_reached = []
        any_merge_input = False
        for cav, frame in good.items():
            distance = getattr(frame, "distance_to_merge_m", None)
            if distance is None:
                continue
            any_merge_input = True
            merge_reached.append(
                float(distance) <= float(getattr(frame, "front_coverage_m", 0.0)) + 1e-9
            )
        if not any_merge_input:
            merge_observed = False
            merge_reason = "no RSU merge-region inputs in any sensor frame"
        elif any(merge_reached):
            merge_observed = True
            merge_reason = None
        else:
            merge_observed = False
            merge_reason = "no live CAV sensor frame reaches the merge region"

        if self.predictor == "learned":
            hybrid = self._hybrid_builder(
                frames=dict(good), now_s=now, roster=self.roster,
                coverage=coverage, merge_observed=merge_observed,
            )
        else:
            hybrid = {
                "opaque": (
                    "perfect_sim oracle: each lookahead branch restores the "
                    "full true state from its own guarded snapshot; this "
                    "master-side hybrid is a marker only, while frames/"
                    "coverage/meta are the real RSU sensor path"
                ),
            }

        sample = SimpleNamespace(
            hybrid=hybrid,
            arrivals={"sample_id": "k0", "departures": ()},
            meta={
                "sample_id": "k0",
                "time_s": float(now),
                "merge_observed": bool(merge_observed),
                "coverage": float(coverage),
                "merge_observed_reason": merge_reason,
                "frame_age_s": frame_age_s,
                "frame_missing": frame_missing,
            },
            frames=frames_map,
            frames_distance_to_merge_m=distances_map,
        )
        # controller_state = the dispatcher session's committed reality (the
        # planner validates locks/plans against exactly this): members of an
        # active /3 wall session report their wall (the raw dispatcher phase
        # vocabulary -- walls.py:TELEMETRY_WALL_PHASES maps it on ingestion),
        # everyone else the committed mode (native is the /3 default
        # everywhere); wall MEMBER SELECTION stays planner-side (PL-04e2).
        committed = dict(getattr(self.session, "_committed", {}) or {})
        active_walls = dict(getattr(self.session, "_walls", {}) or {})
        wall_of = {
            member: (wall_id, wall)
            for wall_id, wall in active_walls.items()
            for member in wall.members
        }
        controller_state = {}
        for cav in self.roster:
            wall_entry = wall_of.get(cav)
            if wall_entry is not None:
                wall_id, wall = wall_entry
                in_phase_from = (
                    wall.assembled_at_s if wall.phase == "hold" else wall.assembly_since
                )
                controller_state[cav] = {
                    "action": "wall",
                    "wall_id": wall_id,
                    "wall_phase": wall.phase,
                    "time_in_phase_s": (
                        0.0 if in_phase_from is None else max(0.0, now - in_phase_from)
                    ),
                    "wall_speed_mps": float(wall.wall_speed_mps),
                    "locked": True,
                }
                continue
            mode = str(committed.get(cav, "native"))
            controller_state[cav] = {
                "action": mode if mode in ACTIONS else "native",
                "wall_id": None,
                "wall_phase": None,
                "time_in_phase_s": 0.0,
                "wall_speed_mps": None,
                "locked": False,
            }
        return [sample], controller_state

    def apply_plan(self, dispatcher_plan, t_s: float, *, note: str = "") -> list:
        plan3 = mixed_assignment_plan(dispatcher_plan)
        self.session.apply_plan(plan3, now_s=float(t_s))
        return [
            {
                "kind": "command",
                "t_s": float(t_s),
                "plan_id": plan3.plan_id,
                "cav": cav,
                "command": plan3.assignments.get(cav, "native"),
                "fallback": None if cav in plan3.assignments else "missing_assignment",
                "owner": "planner",
                "commit_horizon_s": plan3.commit_horizon_s,
                "note": note,
            }
            for cav in sorted(self.roster)
        ]

    def save_state(self, state_dir) -> dict:
        return self.snapshotter.save(
            state_dir, t0_s=self.now_s(), controller_state={}, kept_wall_ids=[]
        )

    def load_state(self, saved: Mapping) -> None:
        # A resumed run is a FRESH process: launch + wire the control path
        # first, then the ONE guarded loadState (PL-10d: one load per
        # process; a straight-through master never receives one).
        if not self._started:
            self.start()
        connection = self._traci
        self.snapshotter.restore(saved, plan=None, draw_id="resume", api=connection)
        state = (saved.get("manifest") or {}).get("harness_state")
        if isinstance(state, Mapping) and state.get("session") is not None:
            # a planner-written checkpoint: carry the master's python state
            restore_harness_state(self._control_write, self.session, state)
            self._prev_lane_ids = previous_lane_ids(
                (saved.get("manifest") or {}).get("physics_row_at_t0")
            )

    @property
    def master_api(self):
        """The master's live traci surface (set by ``start()``).

        PL-04i: the lookahead resolves this LAZILY at first use (the
        MasterBoundSnapshotter binds on its first save), so nothing may
        call this during the build; a call before ``start()`` raises the
        NAMED :class:`HarnessNotStartedError` -- never an AttributeError."""
        if self._traci is None:
            raise HarnessNotStartedError(
                "RealMasterHarness.master_api: harness not started (the "
                "master traci connection exists only after start(); the "
                "runner builds every component BEFORE start, so resolve "
                "the master api lazily at first use)"
            )
        return self._traci


def _circles_controller(adapter_config):
    """The frozen damping controller (CirclesController over the adopted
    Circles adapter config): the ONE construction site shared by the
    master harness and every lookahead branch, so both run the same
    controller type/config (PL-04c; no duplicated wiring)."""
    from traffic_shepherd.damping import CirclesController  # noqa: PLC0415

    return CirclesController(adapter_config)


class PiSessionMissingError(ClosedLoopError):
    """A plan assigning ``pi-external-u`` reached a dispatcher session that
    has NO PI session bound (PL-04k). Named refusal, raised before the
    plan touches the session -- the real runner always binds the registered
    PI session via :func:`build_mixed_session`; only an explicit
    ``with_pi=False`` build can hit this."""


REGISTERED_PI_U_MPS = 31.29  # the campaign's registered PI setpoint


def _make_planner_session_class():
    from traffic_shepherd.mixed_assignment import MixedAssignmentSession  # noqa: PLC0415

    class PlannerMixedAssignmentSession(MixedAssignmentSession):
        def apply_plan(self, plan, *, now_s):
            if "pi-external-u" in plan.assignments.values() and self.pi_session is None:
                raise PiSessionMissingError(
                    "plan assigns pi-external-u but this dispatcher session has "
                    "no PI session bound (build_mixed_session(with_pi=False))"
                )
            return super().apply_plan(plan, now_s=now_s)

    return PlannerMixedAssignmentSession


def build_mixed_session(adapter_config, cav_ids, *, controller=None, with_pi=True):
    """The ONE dispatcher-session construction shared by the master harness
    and every lookahead branch (PL-04k), wired as the campaign does
    (run_chunk: create_controller('damping') + create_pi_session(sorted
    roster, desired_speed_mps=31.29) + MixedAssignmentSession). ``controller``
    lets the master pass the snapshotter's damping instance;
    ``with_pi=False`` builds the PL-04j session without a PI session (any
    pi-external-u plan then raises :class:`PiSessionMissingError`)."""
    from traffic_shepherd.controller_factory import create_pi_session  # noqa: PLC0415

    roster = sorted(str(c) for c in cav_ids)
    if controller is None:
        controller = _circles_controller(adapter_config)
    pi_session = (
        create_pi_session(adapter_config, roster, desired_speed_mps=REGISTERED_PI_U_MPS)
        if with_pi
        else None
    )
    return _make_planner_session_class()(
        circles_controller=controller, pi_session=pi_session, cav_ids=frozenset(roster)
    )


def _planner_config_from_actions(actions):
    """``None`` -> the full vocabulary (PlannerConfig default); a comma
    string or tuple restricts the proposed actions explicitly."""
    if actions is None:
        return PlannerConfig()
    if isinstance(actions, str):
        actions = tuple(a.strip() for a in actions.split(",") if a.strip())
    return PlannerConfig(actions=tuple(actions))


class TraciStateSnapshotterForHarness:
    """Thin adapter over the REAL TraciStateSnapshotter for the master run."""

    def __init__(self, traci_api, adapter_config, *, roster, scenario_routes, seed,
                 harness_state_factory=None, physics_row_factory=None):
        from traffic_shepherd.planner.perfect_sim import TraciStateSnapshotter  # noqa: PLC0415

        self._inner = TraciStateSnapshotter(
            traci_api,
            "state_root",  # replaced per save (the runner-provided checkpoint dir)
            seed=int(seed),
            scenario_digests={},
            wm34_output_root="",
            harness_state_factory=harness_state_factory,
            physics_row_factory=physics_row_factory,
        )
        self._traci_api = traci_api
        self._roster = tuple(roster)
        self._scenario_routes = tuple(str(p) for p in scenario_routes)
        self._adapter_config = adapter_config
        self._seq = 0

    def damping_controller(self):
        return _circles_controller(self._adapter_config)

    def save(self, state_dir, *, t0_s, controller_state, kept_wall_ids):
        self._seq += 1
        # a distinct attempt dir per save: a resumed process re-saves the same
        # checkpoint time without colliding with the interrupted attempt
        self._inner.state_root = Path(state_dir) / f"attempt_{self._seq:06d}"
        return self._inner.save(
            t0_s=t0_s, controller_state=controller_state, kept_wall_ids=kept_wall_ids
        )

    def restore(self, saved, *, plan, draw_id, api):
        self._inner.restore(saved, plan=plan, draw_id=draw_id, api=api)


class MasterBoundSnapshotter(StateSnapshotter):
    """The real ``TraciStateSnapshotter`` bound to the LIVE MASTER connection.

    The runner builds the world model BEFORE the master SUMO process starts,
    so the real snapshotter -- and its ``sumo_version`` from the master
    Connection's ``getVersion()`` -- is created lazily at the FIRST ``save``
    from the master harness's live api. ``restore`` always targets the BRANCH'S
    OWN fresh process (the required ``api`` argument, PL-10d), so the master
    connection only ever issues ``saveState`` -- NEVER ``loadState``
    (design Loop step 3; resume loads go through the harness, not here)."""

    def __init__(self, master_api_provider, state_root, *, seed,
                 harness_state_factory=None, physics_row_factory=None):
        self._master_api_provider = master_api_provider
        # T-B (#2): callables used at every save (the master harness's
        # carried-state capture); None keeps the legacy bare
        # controller_state manifest (fakes)
        self._harness_state_factory = harness_state_factory
        self._physics_row_factory = physics_row_factory
        self.state_root = Path(state_root)
        # PL-04c: the run's REQUIRED seed (no default)
        self.seed = int(seed)
        self._inner: Optional[TraciStateSnapshotter] = None

    def _bound(self) -> TraciStateSnapshotter:
        if self._inner is None:
            # PL-04i: the master api is resolved BEFORE anything else, so
            # a first use before start() raises the NAMED HarnessNotStartedError
            master_api = self._master_api_provider()
            # PL-04l: the version comes from that SAME master Connection
            # object -- never the traci module (module-level calls follow
            # the CURRENT label and can land on a closed lookahead branch)
            self._inner = TraciStateSnapshotter(
                master_api,
                self.state_root,
                seed=self.seed,
                scenario_digests={},
                harness_state_factory=self._harness_state_factory,
                physics_row_factory=self._physics_row_factory,
                sumo_version=parse_sumo_version(master_api.getVersion()),
                # standalone worktree: the vendored thin adapter owns the
                # guarded save/restore (declared so in every manifest)
                wm34_output_root="no-such-wm34-root (vendored adapter)",
            )
        return self._inner

    def save(self, *, t0_s: float, controller_state, kept_wall_ids) -> dict:
        return self._bound().save(
            t0_s=t0_s, controller_state=controller_state, kept_wall_ids=kept_wall_ids
        )

    def restore(self, manifest, *, plan, draw_id, api) -> None:
        return self._bound().restore(manifest, plan=plan, draw_id=draw_id, api=api)


class RealBranchProcessFactory(BranchProcessFactory):
    """ONE FRESH SUMO process per lookahead branch (PL-10d, the real path).

    Launches the SAME scenario cfg as the master on its OWN free port, with
    the master's seed/step length and the exactness flags
    (``--save-state.precision 17 --save-state.rng true``), BELOW_NORMAL
    priority, at most :data:`BRANCH_LAUNCH_ATTEMPTS` attempts (a failed
    traci init kills OUR Popen child and retries on a fresh port).
    ``close_process`` closes the branch's traci connection and waits ONLY on
    this factory's own Popen child -- nothing is ever signalled by image
    name or pattern. PL-04l: branches are launched with
    ``init(..., doSwitch=False)`` and driven ONLY through their own
    Connection object -- a branch never touches the module-level current
    connection the master's module could route through."""

    def __init__(
        self,
        *,
        sumo_binary: str,
        scenario_cfg,
        seed,
        step_length_s: float = PHYSICS_DT_S,
        tag: str = "pl04_branch",
    ):
        self.sumo_binary = str(sumo_binary)
        self.scenario_cfg = Path(scenario_cfg)
        # PL-04c: the run's REQUIRED seed (no default) -- the SAME value
        # the master launches with (launch parity is the exactness contract)
        self.seed = int(seed)
        self.step_length_s = float(step_length_s)
        self.tag = str(tag)
        self._seq = 0
        self.launched: list = []  # one {"label", "port", "pid", "cmd"} per start
        self._children: "dict[int, tuple[str, Any]]" = {}  # id(conn) -> (label, proc)

    def _free_port(self) -> int:
        import socket  # noqa: PLC0415

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    def _command(self, port: int) -> list:
        """The branch launch: the master's flags verbatim (master/branch
        parity is the exactness contract; own --remote-port per process)."""
        return [
            self.sumo_binary,
            "-c",
            str(self.scenario_cfg),
            "--remote-port",
            str(port),
            "--seed",
            str(self.seed),
            "--step-length",
            str(self.step_length_s),
            "--save-state.precision",
            "17",
            "--save-state.rng",
            "true",
            "--no-warnings",
            "true",
        ]

    def start_process(self, *, label=None):
        import subprocess  # noqa: PLC0415
        import traci  # noqa: PLC0415  (lazy: this factory IS a SUMO touchpoint)

        last: Optional[Exception] = None
        for _attempt in range(BRANCH_LAUNCH_ATTEMPTS):
            self._seq += 1
            name = f"{self.tag}{self._seq}_{label or 'branch'}"
            port = self._free_port()
            cmd = self._command(port)
            proc = subprocess.Popen(  # noqa: S603  (fixed argv, no shell)
                cmd, creationflags=BELOW_NORMAL_PRIORITY_CLASS
            )
            try:
                # PL-04l: doSwitch=False -- a branch init must NEVER steal
                # the traci module's current connection; the branch is used
                # exclusively through the Connection object returned below
                traci.init(
                    port=port,
                    host="127.0.0.1",
                    numRetries=4,
                    proc=proc,
                    label=name,
                    doSwitch=False,
                )
                connection = traci.getConnection(name)
            except Exception as exc:  # noqa: BLE001 - retry on a fresh port
                last = exc
                try:
                    proc.kill()  # our OWN Popen child, never by image name
                    proc.wait(timeout=10)
                except Exception:  # noqa: BLE001
                    pass
                continue
            self.launched.append(
                {"kind": "branch", "label": name, "port": port, "pid": proc.pid, "cmd": list(cmd)}
            )
            self._children[id(connection)] = (name, proc)
            return connection
        raise BuildFailedError(
            f"could not launch a branch SUMO process after "
            f"{BRANCH_LAUNCH_ATTEMPTS} attempts: {last}"
        )

    def close_process(self, connection) -> None:
        """Retire the branch: traci close on ITS OWN labeled connection,
        then wait on this factory's own Popen child (never a kill by image
        name or pattern)."""
        try:
            label, proc = self._children.pop(id(connection))
        except KeyError:
            raise ClosedLoopError(
                "close_process: the connection was not started by this factory"
            ) from None
        del label  # the Connection object closes its own labeled connection
        connection.close(wait=False)  # traci close; we reap with a timeout
        proc.wait(timeout=30)  # own child only


# ------------------------------------------------- shared control write
class SessionControlWrite:
    """The REAL frozen control write of ONE dispatcher session on ONE traci
    connection -- the single implementation shared by the master harness
    (every master control tick, PL-04d) and each lookahead branch
    (:class:`BranchControlWrite`, the PL-04c ``tick_hook_factory`` protocol).

    Built on a connection plus the dispatcher session it drives: a
    ``SumoLocalSensor`` + ``CommandOwnership`` pair over the SAME
    damping-controller instance the session holds (``session.circles_
    controller`` -- ``step`` refuses any other; one construction lineage
    per connection via the shared ``_circles_controller`` site). Each
    CONTROL tick (1 s grid; physics ticks between write nothing): capture a
    frame for every LIVE roster CAV (a ``ValueError`` capture -> a ``None``
    frame -- explicit unavailability, never an invented frame), then
    ``session.step(owner, frames, now_s, live_ids)``. The CAV speed mode
    matches the master (``CAV_SPEED_MODE`` = 27, configured by the owner on
    first observation). ``release`` -- the caller runs it in a ``finally``
    at run or branch end OR error -- hands every still-owned CAV back to
    SUMO through ``owner.release_all``."""

    def __init__(self, connection, session, *, roster, adapter_config,
                 control_dt_s: float = CONTROL_DT_S, owner=None,
                 lane_end_distance_to_merge: Optional[Mapping[str, float]] = None):
        from traffic_shepherd.ownership import CommandOwnership  # noqa: PLC0415
        from traffic_shepherd.sensing import SumoLocalSensor  # noqa: PLC0415

        self.connection = connection
        self.session = session
        self.roster = frozenset(str(c) for c in roster)
        self.control_dt_s = float(control_dt_s)
        self.sensor = SumoLocalSensor(
            connection, adapter_config,
            lane_end_distance_to_merge=lane_end_distance_to_merge,
        )
        self.owner = owner if owner is not None else CommandOwnership(
            connection,
            self.roster,
            session.circles_controller,
            cav_speed_mode=CAV_SPEED_MODE,
        )
        self.control_ticks = 0
        self.command_rows = 0
        # T-B (#17): the cross-tick loop state the campaign runners carry
        # (run_suffix / workflow.evaluate): previous_live + retired CAVs,
        # so id-reuse (a CAV id that left and re-entered) reaches the
        # session as disappeared/reappeared ids. Restored from a base-run
        # manifest or a planner snapshot (harness_state.restore_*).
        self.loop_state = {"previous_live": [], "retired_cavs": []}
        self.first_control_tick_s: Optional[float] = None

    def tick(self, now):
        """One possibly-control tick: write ONLY on the control grid.
        Returns the issued command records, or None between control ticks."""
        now = float(now)
        if abs(now / self.control_dt_s - round(now / self.control_dt_s)) > 1e-6:
            return None  # a physics tick between control ticks: no write
        live = {str(v) for v in self.connection.vehicle.getIDList()}
        frames = {}
        for cav in sorted(self.roster & live):
            try:
                frames[cav] = self.sensor.capture(cav, now_s=now)
            except ValueError:
                frames[cav] = None  # explicit unavailability, never invented
        # T-B (#17): run_suffix's lifecycle bookkeeping, verbatim semantics
        previous_live = set(self.loop_state.get("previous_live", ()))
        retired_prev = set(self.loop_state.get("retired_cavs", ()))
        disappeared = (previous_live - live) & self.roster
        reappeared = retired_prev & live
        arrived = {str(v) for v in self.connection.simulation.getArrivedIDList()}
        retired = retired_prev | disappeared | (arrived & self.roster)
        retired -= retired & live
        self.loop_state = {
            "previous_live": sorted(live),
            "retired_cavs": sorted(retired),
        }
        records = self.session.step(
            self.owner,
            frames,
            now_s=now,
            live_ids=live,
            disappeared_ids=tuple(sorted(disappeared)),
            reappeared_ids=tuple(sorted(reappeared)),
        )
        if self.first_control_tick_s is None:
            self.first_control_tick_s = now
        self.control_ticks += 1
        self.command_rows += len(records)
        return records

    def __call__(self, session, now) -> None:
        """The TraciBranchRunner tick-hook protocol (PL-04c); ``session`` is
        the session this write was built for."""
        self.tick(now)

    def release(self) -> None:
        """End-of-run/branch ownership handback (caller's ``finally``)."""
        self.owner.release_all(live_ids={str(v) for v in self.connection.vehicle.getIDList()})


#: The per-branch name of the SAME class (the PL-04c protocol name: the
#: ``tick_hook_factory`` product, carrying ``release()``).
BranchControlWrite = SessionControlWrite


class BranchControlRowLog:
    """Per-branch issued control rows of the lookahead (PL-04e2).

    The REAL DONE-criterion (b) evidence: every non-native lookahead branch
    must issue control rows on its own fresh process. The writer appends one
    JSON row per issued record to ``<run_dir>/branch_control_rows.jsonl``
    and summarizes the counts (receipt + the real closed-loop smoke)."""

    def __init__(self, path):
        self._path = Path(path)
        self._count = 0
        self._by_branch: "dict[str, int]" = {}

    def record(self, branch: str, records: Sequence, assigned: Optional[Mapping] = None) -> None:
        """``assigned`` (PL-04k): the session's committed per-CAV action at
        the tick. ``command`` is the record OWNER, which is ``damping`` for
        both damping and PI issues (both write through the damping-owned
        path), so ``assigned_action`` is what tells a PI command apart."""
        assigned = dict(assigned or {})
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as stream:
            for record in records:
                row = {
                    "kind": "branch_control",
                    "branch": str(branch),
                    "t_s": float(getattr(record, "t_s", 0.0)),
                    "cav": str(getattr(record, "vehicle_id", "")),
                    "command": str(getattr(record, "owner", "")),
                    "assigned_action": assigned.get(str(getattr(record, "vehicle_id", ""))),
                    "issued_target_speed_mps": getattr(
                        record, "issued_target_speed_mps", None
                    ),
                    "fallback": getattr(record, "fallback_reason", None),
                    "released_to_native": bool(
                        getattr(record, "released_to_native", False)
                    ),
                }
                stream.write(_dumps(row) + "\n")
                self._count += 1
                self._by_branch[str(branch)] = self._by_branch.get(str(branch), 0) + 1

    def summary(self) -> dict:
        return {
            "file": str(self._path),
            "count": self._count,
            "by_branch": dict(sorted(self._by_branch.items())),
        }


class RecordingControlWrite(SessionControlWrite):
    """The per-branch control write that MIRRORS every issued record into a
    :class:`BranchControlRowLog` (PL-04e2 criterion (b) evidence). The write
    itself is the unchanged shared :class:`SessionControlWrite`."""

    def __init__(self, *args, sink: Optional[BranchControlRowLog] = None,
                 branch: str = "branch", **kwargs):
        super().__init__(*args, **kwargs)
        self._sink = sink
        self._branch = str(branch)

    def tick(self, now):
        records = super().tick(now)
        if records and self._sink is not None:
            self._sink.record(
                self._branch, records, assigned=getattr(self.session, "_committed", None)
            )
        return records


def mixed_assignment_plan(dispatcher_plan, *, plan_id=None):
    """A v3 ``DispatcherPlanV3`` -> the INSTALLED mixed_assignment/3
    ``AssignmentPlan`` -- the ONE plan conversion the real control path
    accepts (master ``apply_plan`` and the lookahead branch plan factory
    both materialize their plans through it; no duplicated conversion).

    Built on :func:`traffic_shepherd.planner.dispatch.dispatcher_plan_to_
    mixed_assignment_v3_real` (the same version guard and real-object
    construction as ``to_mixed_assignment_v3_real``): walls pass through to
    real ``WallAssignment`` entries and are EXECUTED by the /3 wall-group
    path (:class:`traffic_shepherd.wall_group.WallGroupSession`) shared with
    the WM-59 wall pilot; ``strict_roster=True`` and
    ``commit_horizon_s = 0`` per the WM-40 planner contract (the planner
    re-decides, dwell 0). The installed module's VERSION is checked at plan
    build time: anything but ``mixed_assignment/3`` raises the named
    :class:`~traffic_shepherd.planner.dispatch.WallAssignmentError` -- there
    is no /2 fallback. ``plan_id`` overrides the v3 plan id (the branch
    factory keeps its own branch-plan ids)."""
    plan3 = dispatcher_plan_to_mixed_assignment_v3_real(dispatcher_plan)
    if plan_id is not None and str(plan_id) != plan3.plan_id:
        return dataclasses.replace(plan3, plan_id=str(plan_id))
    return plan3


# --------------------------------------------------------------- components
@dataclass(frozen=True)
class BuildContext:
    """What a components factory gets to build its parts for one process."""

    run_dir: Path
    resume: bool
    process_seq: int  # 1 for a fresh run; +1 per resume cycle


@dataclass
class RunnerComponents:
    """The injectable runner parts (fakes in tests; real at the smoke)."""

    make_harness: Callable[[BuildContext], Any]
    make_world_model: Callable[[BuildContext], Any]
    roster: Sequence[str] = ()
    planner_config: Optional[PlannerConfig] = None
    initial_plan: Optional[PlanCandidate] = None
    # T-A --fixed-policy: the fixed policy EVERY roster CAV holds for the
    # whole reference episode (built by main() from
    # :func:`fixed_policy_for_roster`); a reference run applies it at t0
    # and never reaches a decision point.
    fixed_plan: Optional[PlanCandidate] = None
    sim_seconds_fn: Optional[Callable[[int, int, float], float]] = None
    # PL-04e2 receipt evidence (callables: the parts are built lazily)
    launch_records: Optional[Callable[[], Sequence[Mapping]]] = None
    branch_control_summary: Optional[Callable[[], Mapping]] = None
    # PL-04m: the roster's source evidence (roster_from_base_run's dict on
    # the real path; None on the mock/legacy path) -- the receipt's
    # roster_source / roster_size / n_inserted_* fields read this.
    roster_evidence: Optional[Mapping] = None


def initial_plan_for_roster(roster, action: str) -> PlanCandidate:
    """The PL-04f CONTROL-PATH PROOF initial plan: ``action`` (damping or
    pi-external-u) on EVERY roster CAV, applied by the runner at t0 through
    ``harness.apply_plan(..., note='initial_plan')`` -- before the planner's
    first decision. This mode shows that the DONE criteria (a)-(c) can be
    met by the control path itself when the planner would choose native; it
    is NEVER planner-behaviour evidence (run_purpose='control_path_proof')."""
    if action not in ("damping", "pi-external-u"):
        raise ValueError(f"initial plan action must be damping/pi-external-u, got {action!r}")
    ids = tuple(str(c) for c in roster)
    if not ids:
        raise BuildFailedError(
            "initial_plan_for_roster: the roster is empty (a control-path "
            "proof needs at least one roster CAV to command)"
        )
    return PlanCandidate(per_cav={cav: action for cav in ids})


def fixed_policy_for_roster(roster, policy: str) -> PlanCandidate:
    """The T-A REFERENCE fixed policy: ``policy`` ('native' | 'damping' |
    'pi') on EVERY roster CAV, applied by the runner at t0 through
    ``harness.apply_plan(..., note='fixed_policy')``; the ownership control
    write then executes it EVERY control tick (native included: the
    dispatcher default), so CommandOwnership sets speedMode 27 on every
    live CAV in all three modes. Never planner behaviour
    (run_purpose='reference')."""
    action = FIXED_POLICY_ACTIONS.get(str(policy))
    if action is None:
        raise ValueError(f"fixed policy must be one of {list(FIXED_POLICY_CHOICES)}, got {policy!r}")
    ids = tuple(str(c) for c in roster)
    if not ids:
        raise BuildFailedError(
            "fixed_policy_for_roster: the roster is empty (a reference run "
            "needs at least one roster CAV)"
        )
    return PlanCandidate(per_cav={cav: action for cav in ids})


# ----------------------------------------------------------------- receipt
def _predictor_identity(meter: _MeteredWorldModel) -> Tuple[Optional[str], list]:
    meta = meter.first_meta or {}
    predictor_id = meta.get("model_id")
    sha = meta.get("weights_sha256")
    if isinstance(sha, str):
        sha = [sha]
    return (None if predictor_id is None else str(predictor_id)), list(sha or ())


def _mk_cells(prediction, name: str):
    """One [M, K] prediction array as JSON-safe nested lists (None when the
    predictor did not report the field; NaN/inf cells become None)."""
    arr = getattr(prediction, name, None)
    if arr is None:
        return None
    return _json_safe(np.asarray(arr).tolist())


def _part_a_v21_row(prediction) -> Optional[dict]:
    """The contract v2.1 (Part A v2 general) block of one decisions.jsonl
    row, from the CHOSEN PLAN's prediction and its shared native reference.

    LOGGING-ONLY (PL-11): the four optional per-cell fields and the four
    meta fields appear when the predictor reports them and are null when it
    does not (perfect_sim and the mock emit none of the four cells fields,
    and stay valid). The block carries the plan-vs-native delay breakdown
    the scope rule governs: ``plan.delay_part_a_s`` is comparable across
    predictions only under equal ``part_a_scope`` (rules.compare_part_a),
    while the TOTALS (plan_delay_s / native_delay_s) are always comparable.
    None when the decision carries no chosen prediction (e.g. a gate
    refusal veto row)."""
    if prediction is None:
        return None
    meta = prediction.meta or {}
    return {
        "part_a_scope": meta.get("part_a_scope"),
        "part_a_model": meta.get("part_a_model"),
        "param_set_id": meta.get("param_set_id"),
        "recon_schema_version": meta.get("recon_schema_version"),
        "plan": {
            "n_tracked": _mk_cells(prediction, "n_tracked"),
            "n_exited_tracked": _mk_cells(prediction, "n_exited_tracked"),
            "n_free_road_assumed": _mk_cells(prediction, "n_free_road_assumed"),
            "untracked_arrivals_expected": _mk_cells(
                prediction, "untracked_arrivals_expected"
            ),
            "delay_part_a_s": _mk_cells(prediction, "delay_part_a_s"),
            "delay_part_b_s": _mk_cells(prediction, "delay_part_b_s"),
            "delay_total_s": _mk_cells(prediction, "plan_delay_s"),
        },
        "native": {"delay_total_s": _mk_cells(prediction, "native_delay_s")},
    }


def _part_a_v21_receipt_meta(meta: Optional[Mapping]) -> dict:
    """The contract v2.1 meta block of RUN_RECEIPT (from the run's first
    prediction meta; every key null when the predictor does not report
    it)."""
    meta = meta or {}
    return {
        "part_a_scope": meta.get("part_a_scope"),
        "part_a_model": meta.get("part_a_model"),
        "param_set_id": meta.get("param_set_id"),
        "recon_schema_version": meta.get("recon_schema_version"),
    }


def _ttc_exposure_row(prediction, planner_cfg) -> Optional[dict]:
    """The T-A TTC-exposure block of one decisions.jsonl row: BOTH
    exposures (the PRIMARY ttc_lt1p5 and the frozen ttc_lt3; plan + native,
    window + future twins when reported) of the CHOSEN PLAN's prediction,
    the configured threshold, and the 3 s SHADOW relative verdict
    (report-only) computed on the frozen 3 s fields. None when the decision
    carries no chosen prediction (e.g. a gate-refusal veto row)."""
    if prediction is None:
        return None
    shadow = safety_check(prediction, planner_cfg).shadow_relative_3s
    return {
        "relative_ttc_threshold_s": float(planner_cfg.relative_ttc_threshold_s),
        "plan": {
            "ttc_lt1p5_exposure_s": _mk_cells(prediction, "ttc_lt1p5_exposure_s"),
            "native_ttc_lt1p5_exposure_s": _mk_cells(
                prediction, "native_ttc_lt1p5_exposure_s"
            ),
            "ttc_lt3_exposure_s": _mk_cells(prediction, "ttc_lt3_exposure_s"),
            "native_ttc_lt3_exposure_s": _mk_cells(prediction, "native_ttc_lt3_exposure_s"),
        },
        "future": {
            "ttc_lt1p5_exposure_s": _mk_cells(prediction, "future_ttc_lt1p5_exposure_s"),
            "native_ttc_lt1p5_exposure_s": _mk_cells(
                prediction, "native_future_ttc_lt1p5_exposure_s"
            ),
            "ttc_lt3_exposure_s": _mk_cells(prediction, "future_ttc_lt3_exposure_s"),
            "native_ttc_lt3_exposure_s": _mk_cells(
                prediction, "native_future_ttc_lt3_exposure_s"
            ),
        },
        "shadow_relative_3s": [
            {
                "metric": v.metric,
                "mean_increase_s": v.mean_increase_s,
                "fraction_within_tol": v.fraction_within_tol,
                "ok": bool(v.ok),
            }
            for v in shadow
        ],
    }


def _ttc_exposure_summary(rows: Sequence[Mapping]) -> dict:
    """Receipt-level T-A exposure summary over the run's decision rows."""
    logged = [row.get("ttc_exposure") for row in rows if row.get("ttc_exposure")]
    return {
        "decision_rows": len(rows),
        "rows_with_exposures": len(logged),
        "thresholds": sorted(
            {row["relative_ttc_threshold_s"] for row in logged if "relative_ttc_threshold_s" in row}
        ),
        "shadow_relative_3s_fail": sum(
            1
            for row in logged
            for v in row.get("shadow_relative_3s") or []
            if not v.get("ok")
        ),
    }


def _planner_code_sha() -> str:
    return sha256_text(Path(__file__).read_text(encoding="utf-8"))


# ------------------------------------------------------------------- runner
class ClosedLoopRunner:
    """The PL-04 loop (design sections Loop / Safety gates / Logging /
    Resumability / Budget / Guards). One instance = one master process."""

    def __init__(self, config: ClosedLoopConfig, components: RunnerComponents):
        self.config = config
        self.components = components
        self.meter: Optional[_MeteredWorldModel] = None
        self.receipt: Optional[dict] = None
        self._logs: dict = {}
        self._decision_rows: list = []
        self._fallbacks = 0
        self._resumed_from: list = []
        self._decision_index = 0
        self._candidate_rows = 0
        self._harness = None  # T-B: receipt evidence source (restore fields)

    # ------------------------------------------------------------------ api
    def run(self, *, resume: bool = False) -> dict:
        cfg = self.config
        if resume and cfg.base_run is not None:
            raise BaseRunResumeRefusedError(
                f"--resume refused: run {cfg.run_id!r} was started with "
                f"--base-run {str(cfg.base_run)!r}; base-run runs are never "
                "resumed -- rerun them into a NEW run dir (write-once run dirs)"
            )
        projection = project_run_cost(
            cfg, n_cavs=len(self.components_roster()), planner_config=self._planner_config()
        )
        run_dir = cfg.run_dir
        checkpoint = None
        if run_dir.exists():
            if not resume:
                raise RunDirExistsError(
                    f"run dir {str(run_dir)!r} already exists (write-once run dirs); "
                    "pass --resume to continue it"
                )
            checkpoint = self._latest_checkpoint(run_dir)
            if checkpoint is None:
                raise NothingToResumeError(f"no checkpoint under {str(run_dir)!r} to resume")
        elif resume:
            raise NothingToResumeError(f"run dir {str(run_dir)!r} does not exist; nothing to resume")
        else:
            run_dir.mkdir(parents=True)

        process_seq = self._next_process_seq(run_dir)
        context = BuildContext(run_dir=run_dir, resume=resume, process_seq=process_seq)
        # PL-04i, the build, then start order (deliberate): ALL components
        # are constructed BEFORE harness.start(), so a build failure refuses
        # the run without spending a SUMO launch. Every master-surface access
        # therefore resolves LAZILY at first use -- the MasterBoundSnapshotter
        # binds on its first save and the TraciBranchRunner carries only a
        # lazy provider for its symmetry-only api; a pre-start dereference
        # raises the NAMED HarnessNotStartedError, never an AttributeError.
        harness = self.components.make_harness(context)
        self._harness = harness
        model = self.components.make_world_model(context)
        if cfg.predictor == "perfect_sim" and getattr(model, "screen_horizon_s", None) is not None:
            raise BuildFailedError(
                "perfect-sim lookahead screening is out of scope for PL-04 v1 "
                "(continuation='native' whole-horizon branches only)"
            )
        sim_fn = self.components.sim_seconds_fn or (
            _perfect_sim_pricing if cfg.predictor == "perfect_sim" else _zero_pricing
        )
        self.meter = _MeteredWorldModel(model, sim_fn)
        planner_cfg = self._planner_config()
        resume_plan = None
        if checkpoint is not None:
            resume_plan = _plan_from_json((checkpoint.get("python_state") or {}).get("current_plan"))
        elif self.components.initial_plan is not None:
            resume_plan = self.components.initial_plan
        fixed_plan = getattr(self.components, "fixed_plan", None)
        if cfg.fixed_policy is not None and (
            getattr(self.components, "initial_plan", None) is not None
        ):
            raise BuildFailedError(
                "a --fixed-policy reference run carries no initial plan: "
                "fixed_policy and initial_plan are mutually exclusive"
            )
        planner = Planner(
            self.meter,
            planner_cfg,
            current_plan=resume_plan,
            log_candidates=bool(cfg.log_candidates),
        )

        (run_dir / "checkpoints").mkdir(exist_ok=True)
        self._open_logs(run_dir)
        ledger = SimLedger(cfg.max_sim_s)
        started_monotonic = time.monotonic()
        exit_status = None
        # PL-04e2 --base-run: the horizon counts from the base state's t0
        # (a fresh state at t0 is the START; absolute sim times shift by it).
        start_s = float(cfg.base_t0_s or 0.0)
        horizon_end_s = start_s + float(cfg.horizon_s)
        warmup_end_s = start_s + float(cfg.warmup_s)
        try:
            if checkpoint is not None:
                self._restore_from_checkpoint(checkpoint, harness, planner, ledger)
            else:
                harness.start()
                if fixed_plan is not None:
                    # T-A reference run: the fixed policy on every roster CAV
                    # at t0; the master's ownership control write executes it
                    # on EVERY control tick from here on (native included).
                    plan_v3 = to_dispatcher_v3(
                        SimpleNamespace(plan=fixed_plan), planner_cfg, t0_s=0.0
                    )
                    rows = harness.apply_plan(plan_v3, harness.now_s(), note="fixed_policy")
                    self._fallbacks += sum(1 for r in rows if r.get("fallback"))
                    self._emit("dispatcher_rows.jsonl", rows)
                elif self.components.initial_plan is not None:
                    plan_v3 = to_dispatcher_v3(
                        SimpleNamespace(plan=self.components.initial_plan), planner_cfg, t0_s=0.0
                    )
                    rows = harness.apply_plan(plan_v3, harness.now_s(), note="initial_plan")
                    self._fallbacks += sum(1 for r in rows if r.get("fallback"))
                    self._emit("dispatcher_rows.jsonl", rows)
                while harness.now_s() < warmup_end_s - 1e-9:
                    self._step_and_log(harness, ledger)

            # T-A: a reference run NEVER reaches a decision point (no
            # lookahead branches, no lookahead sim-seconds) -- the loop
            # below just steps the master to the horizon.
            next_decision_s: Optional[float] = (
                None if cfg.fixed_policy is not None else warmup_end_s
            )
            if checkpoint is not None:
                next_decision_s = checkpoint["python_state"].get("next_decision_s")
            while True:
                now = harness.now_s()
                if now >= horizon_end_s - 1e-9:
                    exit_status = "complete"
                    break
                if ledger.over_cap() or self._wall_budget_exceeded(started_monotonic):
                    exit_status = "budget_exhausted"
                    break
                if (
                    next_decision_s is not None
                    and now >= float(next_decision_s) - 1e-9
                    and self._lookahead_fits(planner_cfg, ledger)
                ):
                    self._decision_point(harness, planner, planner_cfg, ledger, float(now))
                    next_decision_s = float(next_decision_s) + cfg.decision_period_s
                    if next_decision_s >= horizon_end_s - 1e-9:
                        next_decision_s = None
                    continue
                if next_decision_s is not None and not self._lookahead_fits(planner_cfg, ledger):
                    exit_status = "budget_exhausted"
                    break
                self._step_and_log(harness, ledger)
        finally:
            self._close_logs()
            try:
                # PL-04d: hand every still-owned CAV back to SUMO before the
                # process is retired -- on normal end, interrupt OR error.
                harness.release_control()
            finally:
                harness.close()

        receipt = self._build_receipt(exit_status, ledger, projection)
        _write_json_atomic(run_dir / "RUN_RECEIPT.json", receipt)
        self.receipt = receipt
        return receipt

    # ------------------------------------------------------------- plumbing
    def components_roster(self) -> Sequence[str]:
        roster = tuple(self.components.roster or ())
        if not roster:
            raise BuildFailedError(
                "RunnerComponents.roster is empty: the budget projection needs "
                "the CAV roster"
            )
        return roster

    def _planner_config(self) -> PlannerConfig:
        override = self.config.gate_override
        base = self.components.planner_config or PlannerConfig()
        commit = self.config.lookahead_commit_s
        if commit is None:
            commit = min(self.config.decision_period_s, self.config.lookahead_horizon_s)
        # PL-04e2: --walls off pins max_walls=0 (no wall proposals); the
        # decision-unit cap travels into the planner config unchanged.
        max_walls = base.max_walls
        if self.config.walls == "off":
            max_walls = 0
        return dataclasses.replace(
            base,
            horizon_s=float(self.config.lookahead_horizon_s),
            trajectory_s=None,
            commit_s=float(commit),
            decide_every_s=float(self.config.decision_period_s),
            continuation="native",  # PL-04 v1: lookahead branches are native-only
            mode="controlled_experiment" if override else "closed_loop",
            allow_unvalidated_future_ttc=bool(override and override.override.covers_future_ttc),
            allow_unvalidated_relative_safety=bool(
                override and override.override.covers_relative_safety
            ),
            max_walls=max_walls,
            max_decision_units=self.config.max_decision_units,
        )

    def _next_process_seq(self, run_dir: Path) -> int:
        root = run_dir / "lookahead_state"
        if not root.is_dir():
            return 1
        existing = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("p")]
        return len(existing) + 1

    def _wall_budget_exceeded(self, started_monotonic: float) -> bool:
        return self.config.max_wall_s is not None and (
            time.monotonic() - started_monotonic > float(self.config.max_wall_s)
        )

    def _lookahead_fits(self, planner_cfg: PlannerConfig, ledger: SimLedger) -> bool:
        """Conservative pre-decision cap check: at minimum the gate-probe
        call (current plan + native) must fit under the remaining budget."""
        if self.config.predictor != "perfect_sim":
            return True
        minimum = _perfect_sim_pricing(1, self.config.k_samples, planner_cfg.horizon_s)
        return ledger.total + minimum < ledger.cap_s - 1e-9

    def _step_and_log(self, harness, ledger: SimLedger) -> None:
        row = harness.step_physics()
        ledger.add_master(float(row.get("dt_s", PHYSICS_DT_S)))
        self._emit("physics_steps.jsonl", [row])
        # PL-04d: the master's per-control-tick executed command rows (the
        # shared control write) land in dispatcher_rows.jsonl next to the
        # apply_plan rows.
        control_rows = harness.drain_control_rows()
        if control_rows:
            self._fallbacks += sum(1 for r in control_rows if r.get("fallback"))
            self._emit("dispatcher_rows.jsonl", control_rows)

    # ------------------------------------------------------------ decisions
    def _decision_point(self, harness, planner: Planner, planner_cfg, ledger: SimLedger, t: float):
        index = self._decision_index
        self._decision_index += 1
        samples, controller_state = harness.observe()
        registry_before = _registry_to_json(planner.wall_registry)
        checkpoint_id = f"ckpt_{index:06d}"
        state_dir = self.config.run_dir / "checkpoints" / checkpoint_id / "master_state"
        saved = harness.save_state(state_dir)
        checkpoint = {
            "schema": CHECKPOINT_SCHEMA,
            "checkpoint_id": checkpoint_id,
            "t_s": t,
            "python_state": {
                "decision_index": index,
                "next_decision_s": t,
                "last_decide_s": planner._last_decide_s,
                "current_plan": _plan_to_json(planner._current_plan),
                "wall_registry": registry_before,
                "log_offsets": self._log_offsets(),
                # candidates.jsonl offset kept OUT of log_offsets (that map is
                # the fixed LOG_FILES set); None when the log is off
                "candidate_log_offset": (
                    self._logs[CANDIDATES_FILE].tell() if CANDIDATES_FILE in self._logs else None
                ),
                "budget": ledger.to_json(),
            },
            "master_state": {
                "saved_state_dir": saved["saved_state_dir"],
                "saved_state_id": saved["saved_state_id"],
                "manifest": saved["manifest"],
                "t0_s": saved["t0_s"],
            },
        }
        _write_json_atomic(self.config.run_dir / "checkpoints" / f"{checkpoint_id}.json", checkpoint)

        self.meter.begin_decision()
        try:
            decision = planner.decide(samples, t, controller_state=controller_state)
        except FutureSafetyRefusal as refusal:
            self._gate_refusal(
                harness,
                planner,
                planner_cfg,
                ledger,
                samples,
                controller_state,
                t,
                refusal,
                registry_before=registry_before,
            )
            return
        lookahead_s = self.meter.end_decision()
        ledger.add_lookahead(lookahead_s)
        self._log_decision(
            harness, planner, planner_cfg, ledger, t, samples, decision, lookahead_s, refusal=None
        )
        plan_v3 = to_dispatcher_v3(decision, planner_cfg, t0_s=t)
        rows = harness.apply_plan(plan_v3, t)
        self._fallbacks += sum(1 for r in rows if r.get("fallback"))
        self._emit("dispatcher_rows.jsonl", rows)
        self._emit_wall_events(
            _registry_to_json(planner.wall_registry), t, baseline=registry_before
        )

    def _gate_refusal(
        self,
        harness,
        planner,
        planner_cfg,
        ledger,
        samples,
        controller_state,
        t,
        refusal,
        *,
        registry_before,
    ):
        """Design Safety gates: the decision point becomes a logged
        ``gate_refusal`` veto; ALL non-locked CAVs revert to native (the
        reference and default); locked walls in assembly keep their lock and
        HOLD walls follow the release rule. The runner never silently keeps
        an unvalidated plan."""
        lookahead_s = self.meter.end_decision()
        ledger.add_lookahead(lookahead_s)
        fallback = self._fallback_plan(planner, controller_state)
        planner._current_plan = fallback  # the next decide must start from it
        plan_v3 = to_dispatcher_v3(SimpleNamespace(plan=fallback), planner_cfg, t0_s=t)
        rows = harness.apply_plan(plan_v3, t, note="gate_refusal")
        self._fallbacks += sum(1 for r in rows if r.get("fallback"))
        self._emit("dispatcher_rows.jsonl", rows)
        decision = SimpleNamespace(
            plan=fallback,
            reasons=(
                DecisionReason(
                    "gate",
                    "*",
                    "gate_refusal",
                    "veto",
                    safety_ok=False,
                    detail=str(refusal),
                ),
            ),
            groups=(),
            abstain=False,
            cadence_skipped=False,
            switched=True,
        )
        self._log_decision(
            harness,
            planner,
            planner_cfg,
            ledger,
            t,
            samples,
            decision,
            lookahead_s,
            refusal=str(refusal),
            veto_reasons=decision.reasons,
        )
        self._emit_wall_events(
            _registry_to_json(planner.wall_registry), t, baseline=registry_before
        )

    def _fallback_plan(self, planner: Planner, controller_state: Mapping) -> PlanCandidate:
        base = planner._current_plan
        if base is None:
            if any(entry.get("locked") for entry in controller_state.values()):
                raise ClosedLoopError(
                    "gate refusal with locked wall members requires the runner's "
                    "current plan (components.initial_plan or a checkpoint)"
                )
            per_cav = {}
            for cav, entry in controller_state.items():
                action = entry.get("action")
                per_cav[cav] = action if action in ACTIONS else "native"
            base = PlanCandidate(per_cav=per_cav)
        plan = base
        for wall in [w for w in plan.walls if w.phase == "hold"]:
            plan = with_wall_release(plan, wall.wall_id)
            planner.wall_registry.mark_release(wall.wall_id)
        return with_actions(plan, {cav: "native" for cav in plan.per_cav})

    # ---------------------------------------------------------- log writers
    def _log_decision(
        self,
        harness,
        planner,
        planner_cfg,
        ledger,
        t,
        samples,
        decision,
        lookahead_s,
        *,
        refusal,
        veto_reasons=None,
    ):
        del harness
        override = self.config.gate_override
        reasons = veto_reasons if veto_reasons is not None else decision.reasons
        vetoes = [
            {
                "rule": "gate_refusal" if refusal else classify_veto(r),
                "step": r.step,
                "subject": r.subject,
                "candidate": r.candidate,
                "detail": r.detail,
            }
            for r in reasons
            if r.action == "veto"
        ]
        switch_checks = [
            {
                "subject": r.subject,
                "candidate": r.candidate,
                "median_saving_s": r.median_saving_s,
                "margin_s": r.margin_s,
                "agreement": r.agreement,
            }
            for r in reasons
            if r.margin_s is not None or r.agreement is not None
        ]
        meta = self.meter.first_meta or {}
        weights = meta.get("weights_sha256")
        if isinstance(weights, str):
            weights = [weights]
        row = {
            "schema": DECISION_ROW_SCHEMA,
            "run_purpose": self.config.run_purpose,
            "t_s": t,
            "decision_index": self._decision_index - 1,
            "sample_ids": [
                str((s.meta or {}).get("sample_id", i)) for i, s in enumerate(samples)
            ],
            "coverage": [float((s.meta or {}).get("coverage", 0.0)) for s in samples],
            "merge_observed": [
                bool((s.meta or {}).get("merge_observed", False)) for s in samples
            ],
            "predictor": self.config.predictor,
            "predictor_class": self.config.predictor_class,
            "predictor_id": meta.get("model_id"),
            "predictor_sha256": list(weights or ()),
            "n_candidates_evaluated": self.meter.decision_plans,
            "n_predict_calls": self.meter.decision_calls,
            "groups": [list(g) for g in getattr(decision, "groups", ())],
            "plan_id": plan_id_for(decision.plan),
            "actions": dict(decision.plan.per_cav),
            "walls": _plan_to_json(decision.plan)["walls"],
            "switched": bool(getattr(decision, "switched", False)),
            "abstain": bool(getattr(decision, "abstain", False)),
            # T-B (#10): why the decision abstained (None when it did not);
            # 'no_decision_units' = zero eligible decision units
            "abstain_reason": (
                None
                if not getattr(decision, "abstain", False)
                else (
                    NO_DECISION_UNITS
                    if any(
                        r.step == "abstain" and r.candidate == NO_DECISION_UNITS
                        for r in reasons
                    )
                    else "merge_observed"
                )
            ),
            # T-B (#7): every ABSOLUTE min-TTC floor breach of an evaluated
            # candidate: follower/leader ids, TTC, pair_class
            # ('cav_involved' | 'hdv_hdv' | 'unknown_pair'); logging only
            "abs_ttc_vetoes": [
                dict(entry) for entry in (getattr(decision, "ttc_vetoes", ()) or ())
            ],
            "vetoes": vetoes,
            "switch_checks": switch_checks,
            "gate": {
                "mode": "closed_loop",
                "planner_gate_mode": planner_cfg.mode,
                "refusal": refusal,
                "override": override is not None,
                "override_reason": None if override is None else override.override.reason,
                "override_file": None if override is None else override.path,
                "override_sha256": None if override is None else override.sha256,
            },
            "lookahead_sim_s": lookahead_s,
            "budget_sim_s": ledger.to_json(),
            # PL-11 (contract v2.1, Part A v2 general): the optional per-cell
            # + meta fields of the CHOSEN PLAN's prediction and its native
            # reference -- present when the predictor reports them, null when
            # it does not, None when no chosen prediction exists. No veto or
            # abstain depends on them.
            "part_a_v21": _part_a_v21_row(getattr(decision, "chosen_prediction", None)),
            # T-A: BOTH TTC exposures (1.5 primary / 3 secondary) of the
            # chosen plan + native, and the report-only 3 s shadow relative
            # verdict.
            "ttc_exposure": _ttc_exposure_row(
                getattr(decision, "chosen_prediction", None), planner_cfg
            ),
        }
        self._decision_rows.append(row)
        self._emit("decisions.jsonl", [row])
        self._emit_candidates(t, planner)

    def _emit_candidates(self, t, planner) -> None:
        """candidates.jsonl: one row per priced candidate of THIS decision
        (planner.last_candidates), tagged with decision_index and t_s."""
        if CANDIDATES_FILE not in self._logs:
            return
        index = self._decision_index - 1
        rows = [
            {**cand, "decision_index": index, "t_s": t}
            for cand in getattr(planner, "last_candidates", ())
        ]
        if rows:
            self._emit(CANDIDATES_FILE, rows)
            self._candidate_rows += len(rows)

    def _emit_wall_events(self, registry_after, t, *, baseline):
        """Registry diff -> walls.jsonl lifecycle rows (register / phase
        change / dissolve)."""
        before = {r["wall_id"]: r for r in (baseline or [])}
        after = {r["wall_id"]: r for r in registry_after}
        events = []
        for wall_id, record in after.items():
            prior = before.get(wall_id)
            if prior is None:
                events.append({"event": "registered", "phase": record["phase"], **record})
            elif prior["phase"] != record["phase"]:
                events.append({"event": record["phase"], "phase": record["phase"], **record})
        for wall_id, prior in before.items():
            if wall_id not in after:
                events.append({"event": "dissolved", "phase": None, **prior})
        rows = [{"t_s": t, **event} for event in events]
        if rows:
            self._emit("walls.jsonl", rows)

    def _emit(self, name: str, rows: Sequence[Mapping]) -> None:
        handle = self._logs[name]
        for row in rows:
            handle.write((_dumps(row) + "\n").encode("utf-8"))
        handle.flush()

    def _open_logs(self, run_dir: Path) -> None:
        names = LOG_FILES + ((CANDIDATES_FILE,) if self.config.log_candidates else ())
        for name in names:
            self._logs[name] = (run_dir / name).open("ab")

    def _close_logs(self) -> None:
        for handle in self._logs.values():
            if not handle.closed:
                handle.close()
        self._logs = {}

    def _log_offsets(self) -> dict:
        return {
            name: handle.tell() for name, handle in self._logs.items() if name != CANDIDATES_FILE
        }

    # ---------------------------------------------------------- checkpoints
    def _latest_checkpoint(self, run_dir: Path) -> Optional[dict]:
        checkpoints = sorted((run_dir / "checkpoints").glob("ckpt_*.json"))
        if not checkpoints:
            return None
        return json.loads(checkpoints[-1].read_text(encoding="utf-8"))

    def _restore_from_checkpoint(self, checkpoint, harness, planner: Planner, ledger: SimLedger):
        state = checkpoint["python_state"]
        harness.load_state(checkpoint["master_state"])
        planner._last_decide_s = state.get("last_decide_s")
        _restore_registry(planner.wall_registry, state.get("wall_registry", []))
        ledger.master_s = float(state["budget"]["master_s"])
        ledger.lookahead_s = float(state["budget"]["lookahead_s"])
        self._decision_index = int(state["decision_index"])
        self._resumed_from = [checkpoint["checkpoint_id"]]
        # Truncate the logs to the recorded offsets through the already-open
        # append handles (a partial tail from the interrupted process goes).
        offsets = dict(state["log_offsets"])
        if state.get("candidate_log_offset") is not None:
            offsets[CANDIDATES_FILE] = state["candidate_log_offset"]
        for name, offset in offsets.items():
            handle = self._logs.get(name)
            if handle is None or handle.closed:
                continue
            handle.flush()
            if handle.tell() > int(offset):
                handle.truncate(int(offset))
                handle.seek(0, os.SEEK_END)
        # The receipt of the resumed run must equal the straight-through one:
        # reload the decisions and dispatcher rows written before the cutoff.
        run_dir = self.config.run_dir
        for line in (run_dir / "decisions.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                self._decision_rows.append(json.loads(line))
        for line in (run_dir / "dispatcher_rows.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line).get("fallback"):
                self._fallbacks += 1
        cand_path = run_dir / CANDIDATES_FILE
        if CANDIDATES_FILE in self._logs and cand_path.exists():
            self._candidate_rows = sum(
                1 for line in cand_path.read_text(encoding="utf-8").splitlines() if line.strip()
            )

    # -------------------------------------------------------------- receipt
    def _build_receipt(self, exit_status: str, ledger: SimLedger, projection: dict) -> dict:
        cfg = self.config
        override = cfg.gate_override
        predictor_id, predictor_sha = _predictor_identity(self.meter)
        # PL-04f: the run's own aggregation passes its own purpose -- one
        # run is one purpose, and a control-path-proof receipt still
        # aggregates its own rows (the PUBLIC entry point refuses them).
        aggregate = aggregate_decisions(
            self._decision_rows, accept_run_purpose=cfg.run_purpose
        )
        launch_records = getattr(self.components, "launch_records", None)
        branch_control = getattr(self.components, "branch_control_summary", None)
        roster_evidence = getattr(self.components, "roster_evidence", None)
        return {
            "schema": RECEIPT_SCHEMA,
            "run_id": cfg.run_id,
            "world": cfg.world,
            "split": cfg.split,
            "seed": cfg.seed,
            "run_purpose": cfg.run_purpose,
            "initial_plan": (
                "native" if cfg.initial_plan_action is None else cfg.initial_plan_action
            ),
            # T-A: the reference-run identity (None on planner runs); the
            # analyzer's comparator guard reads run_purpose='reference' +
            # this token ('native'|'damping'|'pi').
            "fixed_policy": cfg.fixed_policy,
            "predictor": cfg.predictor,
            "predictor_class": cfg.predictor_class,
            "predictor_id": predictor_id,
            "predictor_sha256": predictor_sha,
            "cav_speed_mode": cfg.cav_speed_mode,
            "config_sha256": sha256_text(cfg.to_canonical_json()),
            "planner_code_sha256": _planner_code_sha(),
            "horizon_s": cfg.horizon_s,
            "warmup_s": cfg.warmup_s,
            "decision_period_s": cfg.decision_period_s,
            "lookahead_horizon_s": cfg.lookahead_horizon_s,
            # PL-04e2: the run options and the base-run start record
            "walls": cfg.walls,
            "max_decision_units": cfg.max_decision_units,
            "cav_roster": list(self.components_roster()),
            # PL-04m: the roster's truthful source receipt (the base run's
            # constant physics cav_ids; None on the mock/legacy path where
            # no base-run roster evidence exists)
            "roster_source": (
                None
                if roster_evidence is None
                else roster_evidence.get("roster_source")
            ),
            "roster_size": (
                None if roster_evidence is None else roster_evidence.get("roster_size")
            ),
            "n_inserted_le_t0": (
                None if roster_evidence is None
                else roster_evidence.get("n_inserted_le_t0")
            ),
            "n_inserted_after_t0_or_never": (
                None if roster_evidence is None
                else roster_evidence.get("n_inserted_after_t0_or_never")
            ),
            "start_mode": "base_state" if cfg.base_run is not None else "fresh",
            # T-B (#11): restore-parity evidence of the master harness
            "harness_state_restored": bool(
                getattr(self._harness, "harness_state_restored", False)
            ),
            "first_control_tick_s": getattr(self._harness, "first_control_tick_s", None),
            "restore_source_manifest_sha256": getattr(
                self._harness, "restore_source_manifest_sha256", None
            ),
            "base_run": None if cfg.base_run is None else str(cfg.base_run),
            "base_t0_s": cfg.base_t0_s,
            "base_state_sha256": cfg.base_state_sha256,
            "sumo_launches": (
                None if not callable(launch_records) else list(launch_records())
            ),
            "branch_control_rows": (
                None if not callable(branch_control) else dict(branch_control())
            ),
            "sim_s": {
                "master_s": ledger.master_s,
                "lookahead_s": ledger.lookahead_s,
                "total_s": ledger.total,
                "cap_s": cfg.max_sim_s,
            },
            "budget_projection": projection,
            "decisions": aggregate["decisions"],
            # per-candidate log: file + row count (logging only)
            "candidate_log": {
                "enabled": bool(cfg.log_candidates),
                "file": CANDIDATES_FILE if cfg.log_candidates else None,
                "rows": self._candidate_rows if cfg.log_candidates else 0,
            },
            "vetoes_by_rule": aggregate["vetoes_by_rule"],
            "abstains": aggregate["abstains"],
            "abstain_reasons": aggregate["abstain_reasons"],
            # T-B (#7): counts of absolute min-TTC vetoes per pair class
            "abs_ttc_veto_counts": aggregate["abs_ttc_veto_counts"],
            "gate_refusals": aggregate["gate_refusals"],
            "overrides": {
                "count": aggregate["override_decisions"],
                "approved_by": None if override is None else override.override.approved_by,
                "reason": None if override is None else override.override.reason,
                "scope": None if override is None else override.override.scope,
                "file": None if override is None else override.path,
                "sha256": None if override is None else override.sha256,
            },
            "fallbacks": self._fallbacks,
            # PL-11 (contract v2.1): the run's predictor meta block
            # (part_a_scope / part_a_model / param_set_id /
            # recon_schema_version; null per key when the predictor does
            # not report it, e.g. a mock run).
            "part_a_v21_meta": _part_a_v21_receipt_meta(
                getattr(self.meter, "first_meta", None)
            ),
            # T-A: the run-level TTC exposure summary (both thresholds'
            # logging coverage + the 3 s shadow verdict tally, report-only)
            "ttc_exposure_summary": _ttc_exposure_summary(self._decision_rows),
            "exit_status": exit_status,
            "resumed_from": list(self._resumed_from),
        }


# ---------------------------------------------------------------------- CLI
def config_from_args(args) -> ClosedLoopConfig:
    """CLI -> config; enforces the E:-paths-only guard on the out root, the
    PL-04c seed rule (``--components real`` requires an explicit ``--seed``;
    the mock path keeps the historical 0 default), and the PL-04e2 run
    options: ``--cav-ids`` is mock-only, real runs REQUIRE ``--base-run``
    (validated here: saved-state guards, roster), ``--scenario`` is a
    base-run conflict, ``--walls on`` is refused on real runs until
    PL-04e2-walls, and ``--max-decision-units`` must be >= 1."""
    out_root = Path(args.out).resolve()
    if out_root.drive.upper() != "E:":
        raise PathGuardError(
            f"--out {str(out_root)!r} is not on the E: workspace root (E: paths only)"
        )
    override = None
    if args.gate_override_file is not None:
        override = load_gate_override_file(args.gate_override_file)
    components = getattr(args, "components", "real")
    if getattr(args, "seed", None) is None:
        if components == "real":
            raise BuildFailedError(
                "--components real requires an explicit --seed <int> (the "
                "master and every lookahead branch launch with the SAME run "
                "seed, recorded in the receipt; there is no default)"
            )
        seed = 0  # mock default (unchanged behaviour)
    else:
        seed = int(args.seed)
    cav_ids_flag = getattr(args, "cav_ids", None)
    base: Optional[BaseRunState] = None
    if components == "real":
        if cav_ids_flag:
            raise BuildFailedError(
                "--cav-ids is mock-only: a real run takes its CAV roster "
                "from --base-run (the constant cav_ids list of the base "
                "run's physics steps, PL-04m)"
            )
        base_flag = getattr(args, "base_run", None)
        if not base_flag:
            raise BuildFailedError(
                "--components real requires --base-run <campaign base dir> "
                "(PL-04e2: the master STARTS from the base saved_state at "
                "its t0 and the CAV roster is the base run's full "
                "designated cav_ids list, PL-04m; there is no --cav-ids "
                "path for real runs)"
            )
        if getattr(args, "scenario", None):
            raise BuildFailedError(
                "--scenario is not used together with --base-run: the "
                "scenario cfg + routes are COPIED from "
                "<base-run>/run/scenario into the run dir (nothing is "
                "written under the base run)"
            )
        base = resolve_base_run(base_flag)
        if seed != base.seed:
            raise BaseRunRefusedError(
                f"--seed {seed} does not match the base state's seed "
                f"{base.seed} (the master and every branch launch with the "
                "world's seed)"
            )
    elif getattr(args, "base_run", None):
        raise BuildFailedError(
            "--base-run is a real-run option: mock runs keep the scripted "
            "fake world and --cav-ids"
        )
    walls_flag = getattr(args, "walls", None)
    walls = walls_flag if walls_flag is not None else (
        "on" if components == "mock" else "off"
    )
    if components == "real" and walls == "on":
        raise WallsOnRefusedError(
            "--walls on is refused on real runs until PL-04e2-walls: the "
            "WM-59 wall-member helpers are not promoted to the shared wall "
            "execution path yet (run --walls off)"
        )
    max_units = getattr(args, "max_decision_units", None)
    if max_units is not None and int(max_units) < 1:
        raise BuildFailedError("--max-decision-units must be >= 1")
    initial_plan_action = INITIAL_PLAN_ACTIONS[
        getattr(args, "initial_plan", None) or "native"
    ]
    fixed_policy = getattr(args, "fixed_policy", None)
    if fixed_policy is not None and initial_plan_action is not None:
        raise BuildFailedError(
            "--fixed-policy is mutually exclusive with --initial-plan: a "
            "reference run (run_purpose='reference') has no control-path-"
            "proof initial plan"
        )
    return ClosedLoopConfig(
        out_root=out_root,
        run_id=args.run_id,
        world=args.world,
        seed=seed,
        split=args.split,
        horizon_s=float(args.horizon_s),
        warmup_s=float(args.warmup_s),
        decision_period_s=float(args.decision_period_s),
        predictor=args.predictor,
        lookahead_horizon_s=float(args.lookahead_horizon_s),
        lookahead_commit_s=None if args.lookahead_commit_s is None else float(args.lookahead_commit_s),
        k_samples=int(args.k_samples),
        max_sim_s=float(args.max_sim_s),
        max_wall_s=None if args.max_wall_s is None else float(args.max_wall_s),
        process_start_s=float(args.process_start_s),
        allow_holdout=bool(args.owner_allow_holdout),
        gate_override=override,
        base_run=base.base_run if base is not None else None,
        base_t0_s=base.t0_s if base is not None else None,
        base_state_sha256=base.state_sha256 if base is not None else None,
        max_decision_units=None if max_units is None else int(max_units),
        walls=walls,
        initial_plan_action=initial_plan_action,
        fixed_policy=None if fixed_policy is None else str(fixed_policy),
        log_candidates=str(getattr(args, "log_candidates", "on") or "on") == "on",
    )


VALIDATED_MOCK_META = {
    "future_min_ttc_validated": True,
    "future_min_ttc_validation_id": "mock-validated",
    "relative_safety_validated": True,
    "relative_safety_validation_id": "mock-validated",
}


def build_mock_components(args) -> RunnerComponents:
    """Deterministic fake components (never launches SUMO): the scripted
    master world plus a contract-v2 mock predictor whose validation meta is
    the validated-producer default (tests flip it to exercise refusals).
    ``--cav-ids`` is the mock roster (default ``c0,c1``; mock-only)."""
    cav_ids = tuple(
        s for s in str(getattr(args, "cav_ids", None) or "c0,c1").split(",") if s
    )

    def make_harness(context: BuildContext) -> FakeMasterHarness:
        del context
        # mock default: seed 0 when --seed was not given (real refuses)
        return FakeMasterHarness(
            cav_ids, seed=int(getattr(args, "seed", None) or 0), k_samples=int(args.k_samples)
        )

    def make_world_model(context: BuildContext) -> MockWorldModel:
        del context
        model = MockWorldModel(
            cav_ids,
            part_a_horizon_s=float(args.lookahead_horizon_s),
            contract_version="wd01_predict/2",
            model_id="mock/closed-loop",
        )
        model.meta_script.update(VALIDATED_MOCK_META)
        return model

    return RunnerComponents(
        make_harness=make_harness, make_world_model=make_world_model, roster=cav_ids
    )


def _default_sumo_binary() -> str:
    """--components real SUMO binary: ``$SUMO_BINARY``, else ``$SUMO_HOME``
    ``bin/sumo.exe``, else bare ``sumo`` (PATH)."""
    explicit = os.environ.get("SUMO_BINARY")
    if explicit:
        return explicit
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        return str(Path(sumo_home) / "bin" / "sumo.exe")
    return "sumo"


def _lookahead_commit_s(args) -> float:
    """The commit window the lookahead model must cover (the same value
    ``ClosedLoopRunner._planner_config`` will pass to ``predict_many``)."""
    if getattr(args, "lookahead_commit_s", None) is not None:
        return float(args.lookahead_commit_s)
    return min(float(args.decision_period_s), float(args.lookahead_horizon_s))


def build_real_components(args):  # pragma: no cover - joins the smoke window
    """The REAL components (SUMO master + perfect-sim lookahead).

    PL-04e2 ``--base-run`` path (the CLI's only real-run contract): the CAV
    roster is the base run's FULL designated CAV set -- the constant
    ``cav_ids`` list carried by every row of the base run's
    ``run/physics_steps.jsonl`` (PL-04m; includes CAVs inserted after t0),
    with the source evidence carried on the components for the receipt --
    the scenario cfg + routes are COPIED into the run
    dir (net-file rewritten to the E: assets network, PL-10c guard on the
    copy; nothing under the base run is written), and the master harness
    receives the validated base state -- ONE guarded loadState lands the
    fresh master process on the base t0 (the START, recorded as
    ``start_mode="base_state"`` with the base state sha256). The lookahead
    branches launch from the COPIED cfg. The low-level legacy wiring path
    (``--scenario`` + ``--cav-ids``-shaped args) is retained ONLY for the
    faked-surface wiring tests; the CLI refuses it (config_from_args).

    Wiring (design Loop step 3, perfect_sim arm): a PerfectSimWorldModel
    over a :class:`MasterBoundSnapshotter` (the REAL guarded snapshotter,
    bound to the MASTER traci connection on first save -- saveState only,
    restores go to each branch's own fresh process) and a REAL
    :class:`TraciBranchRunner` carrying a REAL
    :class:`RealBranchProcessFactory` (ONE FRESH SUMO process per (plan,
    draw) branch from the SAME scenario cfg, own free port, the run's
    REQUIRED ``--seed`` and step length, precision 17 + rng, BELOW_NORMAL
    priority, <= 3 launch attempts, close = traci close + wait on its own
    Popen child). PL-04c: every branch that assigns a non-native action or
    wall gets the REAL per-branch control write (``tick_hook_factory`` ->
    :class:`RecordingControlWrite` over :class:`SessionControlWrite`:
    sensor frames -> CommandOwnership -> ``session.step`` on the branch's
    own fresh connection, master controller type/config, speedMode 27,
    ``release_all`` at branch end in a finally); the runner FAILS CLOSED
    without a control write, and every issued branch record is mirrored
    into ``<run_dir>/branch_control_rows.jsonl`` (PL-04e2 criterion (b)
    evidence). ``continuation="native"`` only; the lookahead state root
    lives under ``<run_dir>/lookahead_state/p<process_seq>``. NOT executed
    in the unit tests: the <=80 sim-s smoke runs it first
    (:data:`REAL_HARNESS_NOTE`)."""
    if args.predictor != "perfect_sim":
        raise BuildFailedError("--components real supports predictor=perfect_sim (v1)")
    if getattr(args, "seed", None) is None:
        raise BuildFailedError(
            "--components real requires an explicit --seed <int> (PL-04c: the "
            "master and every lookahead branch launch with the SAME run seed; "
            "there is no default)"
        )
    seed = int(args.seed)
    base_flag = getattr(args, "base_run", None)
    base: Optional[BaseRunState] = None
    if base_flag is not None:
        base = resolve_base_run(base_flag)
        if seed != base.seed:
            raise BaseRunRefusedError(
                f"--seed {seed} does not match the base state's seed {base.seed}"
            )
        cav_ids, roster_evidence = roster_from_base_run(base.base_run, base.t0_s)
        scenario_cfg = None  # resolved per build: the COPY inside the run dir
    else:
        if not getattr(args, "scenario", None):
            raise BuildFailedError(
                "--components real requires --base-run (PL-04e2) or a "
                "legacy --scenario <scenario.sumocfg> (wiring tests only)"
            )
        cav_ids = tuple(s for s in str(args.cav_ids).split(",") if s)
        scenario_cfg = Path(args.scenario)
    sumo_binary = _default_sumo_binary()
    commit_s = _lookahead_commit_s(args)
    made: dict = {}  # make_harness -> make_world_model hand-off (run order)
    if base is not None:
        # PL-04m: the roster's source evidence travels to the receipt
        made["roster_evidence"] = dict(roster_evidence)

    def adapter_config():
        """The adopted Circles adapter config, loaded ONCE per build from
        the research assets mount (read-only; no SUMO)."""
        if "adapter_config" not in made:
            from traffic_shepherd.config import load_config  # noqa: PLC0415

            made["adapter_config"] = load_config(
                research_dense_root() / "baselines" / "circles" / "adapter.json"
            )
        return made["adapter_config"]

    def make_harness(context: BuildContext) -> RealMasterHarness:
        if base is not None:
            # the scenario copy lives INSIDE the run dir (never under the
            # base run); the PL-10c guard must pass on the copy
            made["scenario_info"] = copy_scenario_into_run_dir(
                base.scenario_dir, context.run_dir
            )
            made["scenario_cfg"] = Path(made["scenario_info"]["copied_cfg"])
        else:
            made["scenario_cfg"] = scenario_cfg
        harness = RealMasterHarness(
            sumo_binary=sumo_binary,
            scenario_cfg=made["scenario_cfg"],
            # the cfg itself: the PL-10c guard expands its <input> routes
            scenario_routes=[made["scenario_cfg"]],
            roster=cav_ids,
            adapter_config=adapter_config(),
            seed=seed,
            # PL-04d: the harness observes for THIS predictor; the learned
            # predictor would need a hybrid_builder (none exists here, and
            # the predictor gate above already refuses it for v1)
            predictor=args.predictor,
            # PL-04e2: the validated base state (start, not a resume)
            base_state=base.as_saved_dict() if base is not None else None,
        )
        made["harness"] = harness
        return harness

    def branch_dispatcher_session(controller_state, kept_wall_ids):
        """The REAL dispatcher session of one branch (the /3 module wired
        exactly as ``RealMasterHarness.start`` wires the master's: the same
        damping controller type/config via the shared construction site)."""
        del controller_state, kept_wall_ids
        return build_mixed_session(adapter_config(), cav_ids)

    def branch_plan_factory(plan, plan_id, controller_state, kept_wall_ids):
        """Planner plan -> the INSTALLED dispatcher's AssignmentPlan.

        The ONE shared conversion (:func:`mixed_assignment_plan` -- the same
        one the master's ``apply_plan`` runs): v3 validation through
        ``to_dispatcher_v3``, then the real /3 AssignmentPlan
        (``strict_roster=True``, ``commit_horizon_s = 0``; walls stage as
        ``WallAssignment`` entries and EXECUTE through the /3 wall-group
        path). The branch runner's native plan bypasses the planner-plan
        conversion and stages all-native directly (a native label carries
        no per_cav map to convert)."""
        del controller_state, kept_wall_ids
        from traffic_shepherd.mixed_assignment import AssignmentPlan  # noqa: PLC0415

        if getattr(plan, "label", None) == NATIVE_PLAN_LABEL:
            return AssignmentPlan(
                assignments={cav: "native" for cav in cav_ids},
                plan_id=plan_id,
                commit_horizon_s=0.0,
            )
        return mixed_assignment_plan(
            to_dispatcher_v3(SimpleNamespace(plan=plan), t0_s=0.0), plan_id=plan_id
        )

    def branch_tick_hook_factory(connection, session, plan):
        """The REAL per-branch control write, created on the branch's OWN
        fresh connection (PL-04c): a :class:`RecordingControlWrite` -- the
        shared ``SessionControlWrite`` (sensor frames -> CommandOwnership ->
        ``session.step`` over the SAME controller instance the branch
        dispatcher session holds) that MIRRORS every issued record into the
        run's ``branch_control_rows.jsonl`` (PL-04e2 criterion (b)).
        PL-04j sensor parity: the branch write carries the merge-region map
        resolved on ITS OWN connection via ``observation_map`` (the same
        resolution the master's start() runs), so branch frames carry the
        same merge-region inputs the master's do."""
        made["branch_seq"] = made.get("branch_seq", 0) + 1
        label = f"{made['branch_seq']}:{getattr(plan, 'label', 'plan')}"
        from traffic_shepherd.scenarios import observation_map  # noqa: PLC0415

        try:
            merge_map = dict(observation_map(connection, adapter_config()))
        except Exception as exc:  # pragma: no cover - defensive: wrapped loud
            raise MergeMapUnavailableError(
                f"branch control write {label!r}: the RSU merge-region lane "
                f"map could not be resolved on the branch connection "
                f"(observation_map: {exc}; {type(exc).__name__})"
            ) from exc
        if not merge_map:
            raise MergeMapUnavailableError(
                f"branch control write {label!r}: observation_map resolved "
                "an EMPTY merge-region lane map on the branch connection "
                "(no sensor parity with the master)"
            )
        return RecordingControlWrite(
            connection,
            session,
            roster=cav_ids,
            adapter_config=adapter_config(),
            lane_end_distance_to_merge=merge_map,
            sink=made.get("branch_control_log"),
            branch=label,
        )

    def make_world_model(context: BuildContext) -> PerfectSimWorldModel:
        harness = made.get("harness")
        if harness is None:
            raise BuildFailedError(
                "make_world_model runs after make_harness (the ClosedLoopRunner "
                "build order); no master harness to bind the lookahead to"
            )
        cfg_path = made.get("scenario_cfg") or scenario_cfg
        made["branch_control_log"] = BranchControlRowLog(
            context.run_dir / "branch_control_rows.jsonl"
        )
        state_root = context.run_dir / "lookahead_state" / f"p{context.process_seq}"
        snapshotter = MasterBoundSnapshotter(
            lambda: harness.master_api,
            state_root,
            seed=seed,
            # T-B (#2): every snapshot carries the master's python control
            # state so the branches restore it (harness_state.py)
            harness_state_factory=lambda controller_state, kept: harness.capture_harness_state(
                controller_state, kept
            ),
            physics_row_factory=lambda t0: harness.snapshot_physics_row(t0),
        )
        factory = RealBranchProcessFactory(
            sumo_binary=sumo_binary,
            scenario_cfg=cfg_path,
            seed=seed,
            step_length_s=PHYSICS_DT_S,
        )
        made.setdefault("branch_factories", []).append(factory)
        runner = TraciBranchRunner(
            # wiring symmetry only: branches run on their OWN fresh
            # connection. PL-04i: a LAZY provider, not the api itself --
            # make_world_model runs BEFORE harness.start() (the runner's
            # build, then start order), so the master connection is
            # resolved at first use, never dereferenced at build time.
            lambda: harness.master_api,
            snapshotter=snapshotter,
            cav_ids=cav_ids,
            dt_s=PHYSICS_DT_S,
            control_dt_s=CONTROL_DT_S,
            dispatcher_factory=branch_dispatcher_session,
            plan_factory=branch_plan_factory,
            # PL-04c: the REAL per-branch control write on the branch's own
            # fresh connection -- without it a damping/wall branch would
            # silently simulate as native (the runner now refuses that)
            tick_hook_factory=branch_tick_hook_factory,
            process_factory=factory,
        )
        # T-A #1: the lookahead delay IS the evaluation delay -- the p4
        # delay over ALL corridor vehicles (entry waits incl. the insertion
        # backlog) on the branch's campaign-schema physics rows. Fail
        # closed when the WD-01 library is unavailable: no silently legacy
        # (CAV-presence) lookahead objective.
        from .recorded import P4DelayUnavailable  # noqa: PLC0415 (lazy)

        try:
            delay_evaluator = P4DelayEvaluator(departs=_scenario_departs(cfg_path))
        except P4DelayUnavailable as exc:
            raise BuildFailedError(
                f"the p4 branch delay evaluator is unavailable: {exc}"
            ) from exc
        return PerfectSimWorldModel(
            snapshotter,
            SimpleArrivalDraws(),
            runner,
            part_a_horizon_s=commit_s,
            contract_version="wd01_predict/2",
            model_id="perfect_sim/pl04-closed-loop",
            scenario_route_files=[cfg_path],  # the PL-10c guard must pass
            metrics_extractor=MetricsExtractor(delay_evaluator=delay_evaluator),
        )

    def launch_records() -> list:
        rows = []
        harness = made.get("harness")
        master = getattr(harness, "launch_record", None)
        if master:
            rows.append(dict(master))
        for factory in made.get("branch_factories", ()):
            rows.extend(dict(record) for record in getattr(factory, "launched", ()))
        return rows

    def branch_control_summary() -> dict:
        log = made.get("branch_control_log")
        if log is None:
            return {"file": None, "count": 0, "by_branch": {}}
        return log.summary()

    return RunnerComponents(
        make_harness=make_harness,
        make_world_model=make_world_model,
        roster=cav_ids,
        # PL-04k: the real sessions bind the registered PI session (one
        # shared construction, build_mixed_session), so the full action
        # vocabulary is executable and the PL-04j native/damping pin is
        # gone. ``args.actions`` (comma list) keeps the restriction as an
        # explicit option.
        planner_config=_planner_config_from_actions(getattr(args, "actions", None)),
        launch_records=launch_records,
        branch_control_summary=branch_control_summary,
        # PL-04m: the roster's source evidence for the receipt (None on the
        # legacy --scenario wiring path)
        roster_evidence=made.get("roster_evidence"),
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m traffic_shepherd.planner.closed_loop",
        description="PL-04 closed-loop planner runner (SUMO-free with "
        "--components mock; the real components join the coordinated smoke).",
    )
    parser.add_argument("--out", required=True, help="run root (E: paths only)")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--world", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="run seed (REQUIRED for --components real: the master and "
        "every lookahead branch launch with it; no default)",
    )
    parser.add_argument("--horizon-s", type=float, required=True)
    parser.add_argument("--warmup-s", type=float, default=0.0)
    parser.add_argument("--decision-period-s", type=float, default=30.0)
    parser.add_argument("--predictor", choices=list(PREDICTORS), required=True)
    parser.add_argument("--lookahead-horizon-s", type=float, default=60.0)
    parser.add_argument("--lookahead-commit-s", type=float, default=None)
    parser.add_argument("--k-samples", type=int, default=1)
    parser.add_argument(
        "--cav-ids",
        default=None,
        help="MOCK runs only (default c0,c1); real runs take the roster "
        "from --base-run and refuse --cav-ids",
    )
    parser.add_argument(
        "--base-run",
        default=None,
        help="PL-04e2: campaign-format base run dir (<world>__base); the "
        "REAL master STARTS from <dir>/saved_state at its t0 (one guarded "
        "loadState, start_mode=base_state) and the CAV roster is the base "
        "run's FULL designated cav_ids list (the constant cav_ids of "
        "run/physics_steps.jsonl, PL-04m; required for --components real)",
    )
    parser.add_argument(
        "--max-decision-units",
        type=int,
        default=None,
        help="PL-04e2: the individual stage only considers the N roster "
        "CAVs nearest upstream of the merge that are present in the "
        "current RSU frames (the others keep their current action); "
        "default: uncapped",
    )
    parser.add_argument(
        "--walls",
        choices=["on", "off"],
        default=None,
        help="PL-04e2: off -> max_walls=0 (no wall proposals). Default: on "
        "for mock runs, off for real runs; real runs with on are refused "
        "until PL-04e2-walls",
    )
    parser.add_argument(
        "--initial-plan",
        choices=list(INITIAL_PLAN_ACTIONS),
        default="native",
        help="PL-04f control-path proof: PRE-SET the initial plan to this "
        "action on every roster CAV at t0 (damping/pi), labelling the run "
        "run_purpose='control_path_proof' -- control-path evidence only, "
        "never planner behaviour. Default native = the planner decides "
        "everything (run_purpose='planner')",
    )
    parser.add_argument(
        "--fixed-policy",
        choices=list(FIXED_POLICY_CHOICES),
        default=None,
        help="T-A REFERENCE run (run_purpose='reference'): every roster CAV "
        "holds this policy from t0 (native/damping/pi); NO decision points "
        "and NO lookahead branches on the SAME runner/harness/base-run "
        "restore; the ownership control write runs EVERY control tick in "
        "all three modes (speedMode 27 on every live CAV). Mutually "
        "exclusive with --initial-plan; the budget projection is the "
        "master horizon only",
    )
    parser.add_argument("--max-sim-s", type=float, required=True,
                        help="HARD cap over master + ALL lookahead sim-seconds")
    parser.add_argument("--max-wall-s", type=float, default=None)
    parser.add_argument("--process-start-s", type=float, default=0.0)
    parser.add_argument("--gate-override-file", default=None,
                        help="the ONLY source of a gate override (owner-approved JSON)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--owner-allow-holdout", action="store_true",
                        help="explicit owner flag: allow HOLDOUT/TEST worlds")
    parser.add_argument("--components", choices=["real", "mock"], default="real")
    parser.add_argument("--components-spec", default=None,
                        help="module:factory building RunnerComponents from the args")
    parser.add_argument("--scenario", default=None, help="real harness scenario cfg")
    parser.add_argument(
        "--log-candidates",
        choices=["on", "off"],
        default="on",
        help="write candidates.jsonl (one row per priced candidate per "
        "decision: delays, rule-C safety quantities, veto/switch outcome, "
        "chosen flag). Pure logging -- decisions are identical on/off. "
        "Default: on",
    )
    parser.add_argument(
        "--actions",
        default=None,
        help="PL-04k: restrict the planner's proposed actions (comma list "
        "of native,damping,pi-external-u). Default: the full vocabulary",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = make_parser().parse_args(list(argv) if argv is not None else None)
    try:
        config = config_from_args(args)
        if args.components_spec:
            module_name, _, factory_name = str(args.components_spec).partition(":")
            module = __import__(module_name, fromlist=[factory_name])
            components = getattr(module, factory_name)(args)
        elif args.components == "mock":
            components = build_mock_components(args)
        else:
            components = build_real_components(args)
        if config.initial_plan_action is not None:
            # PL-04f control-path proof: the PRE-SET non-native initial plan
            # on the run's roster (real runs: the --base-run roster), applied
            # at t0 by the runner BEFORE the planner's first decision. The
            # run is labelled run_purpose='control_path_proof' -- control-
            # path evidence only, never planner behaviour.
            components.initial_plan = initial_plan_for_roster(
                components.roster, config.initial_plan_action
            )
        if config.fixed_policy is not None:
            # T-A reference run: the fixed policy on every roster CAV from
            # t0; run_purpose='reference', no decision points, no lookahead.
            components.fixed_plan = fixed_policy_for_roster(
                components.roster, config.fixed_policy
            )
        runner = ClosedLoopRunner(config, components)
        receipt = runner.run(resume=bool(args.resume))
    except ClosedLoopError as exc:
        print(f"closed_loop: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "receipt": str(config.run_dir / "RUN_RECEIPT.json"),
                "exit_status": receipt["exit_status"],
                "decisions": receipt["decisions"],
                "sim_s": receipt["sim_s"]["total_s"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
