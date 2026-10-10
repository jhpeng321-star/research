"""Perfect-simulator world model (PL01-C skeleton, PL-10 contract-v2).

Implements the ``wd01_predict`` API -- dual-version: v1, v1.1 and (since
PL-10) the contract-v2 commit-window semantics of the WD-01 worktree's
PREDICT_CONTRACT_v2.md (read-only; sections 1-8
incl. G-1, G-2, native reference N1, the future_* heads and
future_has_closing_pair) -- by branching the true SUMO state at the decision
time t0 and simulating every candidate plan --
output/planner_v1/PERFECT_SIM_DESIGN.md, sections Mechanism / Arrival draws
(option A) / Outputs mapping / Cost model / Skeleton.
DEVELOPMENT ORACLE ONLY: it reads the full true state, which violates the RSU
runtime-input constraint, so it is never a deployable controller input; its
results bound what the planner could achieve with a perfect world model and
say nothing about a learned model.

Contract v2 (PL-10): a plan is the action for the COMMIT WINDOW ``commit_s``
starting at t0; ``meta.safety_window_s == commit_s``; ``delay_part_a_s``
covers [t0, t0+commit_s] and part B the remainder under ``continuation`` --
"native" (all CAVs native after the commit window; fully implemented) or
"planner" (closed-loop pi_k, PL-04, NOT built: raises NotImplementedError).
The future_* heads are REALISED branch values (p is 0/1, q05 the realised
minimum over (t0+commit_s, t0+horizon_s]); the validity heads fail closed on
any run error, early stop, screening or teleport; the simulator is ground
truth, so ``future_min_ttc_validated`` / ``relative_safety_validated`` are
True with validation id ``'perfect_sim'`` (G-1/G-2 pass in planner live
mode). v1/v1.1 output modes remain available (constructor
``contract_version=...``); their semantics are unchanged.

This file imports NO SUMO/traci/libsumo module at module import time (zero-
simulation rule; only the production collaborators below own real traci
calls, all of them injected).

Collaborators (all injected; tests use fakes):
- ``StateSnapshotter``  saves at t0 ONCE per predict_many call: the SUMO
  state file + manifest and the python-side harness state. File names and
  the harness-component inventory reuse the WM-41 saved-state protocol
  (the wm34 worktree's campaign_pilot_v1/saved_state.py, read-only;
  vendored thin adapter in ``perfect_sim_state.py``):
  ``state.xml`` / ``STATE_MANIFEST.json`` and (ownership, circles_controller,
  pi_session, loop_state, request_ledger, random_state); SUMO >= 1.15.0
  carries the RNG state inside the state file (pinned 1.27.1 there). The
  EXACTNESS GUARD is mandatory: a snapshot without ``--save-state.precision
  17`` / ``--save-state.rng`` is refused on save and on restore
  (:mod:`traffic_shepherd.planner.perfect_sim_state`).
- ``ArrivalDraws``      option A: explicit post-t0 departure lists; draw k is
  installed for EVERY plan's k-th branch and for native, so common random
  numbers hold by construction. A kit ``ArrivalDraw`` {demand_model_id, seed}
  payload declares the K=1 restored-RNG mode instead (no explicit install;
  the snapshot's SUMO RNG drives the arrivals) -- declared, not CRN across
  K (K must be 1).
- ``BranchRunner``      runs one plan x one draw in a FRESH SUMO process
  (PL-10d): the runner carries a :class:`BranchProcessFactory`
  (``start_process()`` -> the branch's own traci connection;
  ``close_process(connection)`` after the branch, exceptions included);
  loadState, install the draw, restore the harness snapshot, apply the plan
  through the mixed-assignment/3 dispatcher (strict_roster,
  commit_horizon_s=0; walls via the WM-40 wall-group path), run
  ``horizon_s`` (v2: plan for ``commit_s``, then the continuation) and
  report one :class:`BranchRecord`. A connection that already restored once
  is SPENT: a second restore raises :class:`ProcessReuseError` BEFORE any
  loadState (re-smoke evidence: state carries over across loadState).
- ``MetricsExtractor``  BranchRecord -> contract outputs: delay parts,
  safety-window metrics (``traffic_shepherd.metrics`` closing_ttc /
  ttc_exposure / braking_effort semantics), tracked trajectories
  (T = trajectory_s + 1, NaN from the exit step on), wall events, and
  (v2) the realised future-window safety block with closing-pair
  bookkeeping.

Early stop: a branch that records a collision inside the safety window stops
immediately; its delay cells are NaN and flagged in meta (the plan is vetoed
either way, so its Part B is never simulated).

Screening: with ``screen_horizon_s`` set, every branch first runs at the
short horizon, plans are ranked by the median paired delay_vs_native over K,
and only the top ``rescore_top_n`` plans plus ``rescore_always`` (default
index 0 -- the caller's current plan under the planner's candidate ordering)
plus the native reference run again at the full horizon. Screened-out plans
keep their screen-horizon safety and trajectories and report NaN delay with
meta flags (``screened_out``, ``evaluated_horizon_s``).

:func:`estimate_cost` reproduces the design's cost model: per decision
P = 2N + sum(2|g| + 2) + W*speeds*releases + 1 candidate plans, B = K*(1+P)
branches; CPU hours = (sim-seconds * r + B * process_start_s) / 3600, wall
hours = CPU / workers (``process_start_s``: one fresh SUMO launch per
branch, PL-10d; default 0 = TBD, measured later).
"""

import dataclasses
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from .harness_state import restore_harness_state
from .interface import (
    CONTRACT_VERSION,
    CONTRACT_VERSION_V1_1,
    CONTRACT_VERSION_V2,
    CONTINUATIONS,
    Prediction,
    WallPrediction,
)
from .perfect_sim_state import (
    HARNESS_COMPONENTS,
    PINNED_SUMO_VERSION,
    REQUIRED_SAVE_STATE_CLI,
    SCHEMA,
    STATE_DIR_NAME,
    SUMO_RNG_STATE_MIN_VERSION,
    SUMO_STATE_FILE,
    WM34_OUTPUT_ROOT,
    ProcessReuseError,
    ScenarioNotExactError,
    StateSnapshotError,
    check_scenario_exactness,
    optional_real_saved_state,
    parse_sumo_version,
    require_sumo_rng_state,
    require_exact_state_file,
    sha256_file,
    sha256_text,
    validate_manifest_for_restore,
    validate_state_file_for_exactness,
    write_manifest,
)
from traffic_shepherd.metrics import braking_effort, closing_ttc

__all__ = [
    "ArrivalDraws",
    "BranchProcessFactory",
    "BranchRecord",
    "BranchRunner",
    "CostEstimate",
    "MetricsExtractor",
    "MissingControlWriteError",
    "NATIVE_PLAN",
    "NATIVE_PLAN_LABEL",
    "P4DelayEvaluator",
    "PerfectSimWorldModel",
    "ProcessReuseError",
    "ScenarioNotExactError",
    "SimpleArrivalDraws",
    "StateSnapshotter",
    "StepRecord",
    "SUMO_RNG_STATE_MIN_VERSION",
    "check_scenario_exactness",
    "parse_sumo_version",
    "require_sumo_rng_state",
    "validate_state_file_for_exactness",
    "TraciBranchRunner",
    "TraciStateSnapshotter",
    "VehicleStep",
    "WallEventRecord",
    "estimate_cost",
]

NATIVE_PLAN_LABEL = "native"


class MissingControlWriteError(RuntimeError):
    """A branch needs control writes but the runner has no tick_hook.

    PL-04c fail-closed rule: a plan branch that assigns any non-native
    action or any wall MUST be built with a control write (``tick_hook``,
    or ``tick_hook_factory`` for the per-branch real write). Without one
    the branch would silently simulate as native -- a wrong lookahead
    value, not a crash -- so :meth:`TraciBranchRunner.run` refuses BEFORE
    the restore (no loadState is spent on a branch that cannot be
    simulated honestly). A native-only plan (every per_cav entry "native",
    no walls, no kept walls) may run without one."""


class _NativePlan:
    """Sentinel for the N1 native reference branch (shared, never a plan)."""

    label = NATIVE_PLAN_LABEL

    def __repr__(self):
        return "NATIVE_PLAN"


NATIVE_PLAN = _NativePlan()

#: WM-41 saved-state protocol names are RE-EXPORTED from
#: :mod:`traffic_shepherd.planner.perfect_sim_state` (the vendored thin
#: adapter of the wm34 saved_state.py): SUMO_STATE_FILE,
#: MANIFEST_FILE, HARNESS_COMPONENTS, SUMO_RNG_STATE_MIN_VERSION, SCHEMA,
#: STATE_DIR_NAME, REQUIRED_SAVE_STATE_CLI. Nothing is duplicated here.

#: Safety-window thresholds, mirroring traffic_shepherd.metrics semantics.
TTC_EXPOSURE_THRESHOLD_S = 3.0
#: T-A (PL-EXP1b addendum 2): the PRIMARY relative-TTC exposure threshold
#: the extractor reports alongside the frozen 3 s one. Same pair
#: arithmetic, different threshold; planner-side selection is config-driven
#: (PlannerConfig.relative_ttc_threshold_s).
TTC_EXPOSURE_PRIMARY_THRESHOLD_S = 1.5
HARD_BRAKE_MPS2 = -3.0

#: Contract-v2 value-model / validation ids (the simulator IS ground truth,
#: contract v2 sections 4, 6 and 8): the planner's live-mode G-1/G-2 gate
#: accepts a v2 prediction whose meta carries these.
VALUE_MODEL_ID = "perfect_sim"
FUTURE_MIN_TTC_VALIDATION_ID = "perfect_sim"
RELATIVE_SAFETY_VALIDATION_ID = "perfect_sim"

_PLANNER_CONTINUATION_MESSAGE = (
    "PerfectSimWorldModel: continuation='planner' requires a closed-loop "
    "pi_k runner inside each branch (the planner's own policy re-deciding "
    "after the commit window) -- PL-04, not built. The perfect simulator "
    "implements continuation='native' only; do not fake planner "
    "continuation (PREDICT_CONTRACT_v2.md sections 1 and 6)."
)


# ----------------------------------------------------------------- records
@dataclass(frozen=True)
class VehicleStep:
    """One vehicle's state at one reported step of a branch.

    T-A: ``road_id`` / ``sumo_lane_id`` (the SUMO lane id string) carry the
    campaign physics-row identity when the connection reports them (the
    real traci surface does; a bare test fake may omit both), so the
    extractor can rebuild campaign-schema physics rows and evaluate the
    SAME p4 delay the evaluation uses. ``leader_gap_m`` is the PHYSICAL
    bumper-to-bumper gap (the raw 200 m ``getLeader`` gap PLUS the
    follower's ``getMinGap``), matching ``metrics.ttc_exposure``."""

    s_m: float
    lane_id: float
    speed_mps: float
    accel_mps2: float
    entry_wait_s: float = 0.0  # seconds spent in an entry queue this step
    on_section: bool = True
    leader_gap_m: Optional[float] = None
    leader_speed_mps: Optional[float] = None
    road_id: Optional[str] = None
    sumo_lane_id: Optional[str] = None
    # T-B (#7): the leader's id (pair logging of the absolute TTC veto)
    leader_id: Optional[str] = None


@dataclass(frozen=True)
class StepRecord:
    """One reported step: 1 s for trajectories; ``dt_s`` is its interval."""

    time_s: float
    dt_s: float
    vehicles: Mapping[str, VehicleStep]
    colliding_ids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class WallEventRecord:
    """Wall-session events for one wall in one branch (design step 3)."""

    assembled_at_s: Optional[float] = None
    safety_abort: bool = False
    abort_reason: int = 0  # 0 none / 1 safety / 2 assembly_timeout


@dataclass(frozen=True)
class BranchRecord:
    """What one plan x one draw branch reports (BranchRunner output).

    PL-10 validity inputs: ``run_error`` flags a branch that could not
    complete (fail closed); ``teleport_times`` are the branch's teleport
    instants (a teleport anywhere inside a window invalidates that window's
    cell -- contract v2 section 3, fail closed)."""

    plan_label: str
    draw_id: str
    t0_s: float
    horizon_s: float
    dt_s: float
    free_flow_s: Mapping[str, float]  # billed vehicles: t0-present + arrivals
    steps: Tuple[StepRecord, ...]
    collision_times: Tuple[float, ...] = ()
    wall_events: Mapping[str, WallEventRecord] = field(default_factory=dict)
    tracked: Tuple[str, ...] = ()  # trajectory roster: CAVs + observed at t0
    stopped_early: bool = False
    stopped_at_s: Optional[float] = None
    run_error: bool = False
    teleport_times: Tuple[float, ...] = ()
    # T-A: the designated CAV universe of the branch (the master roster) and
    # the vehicles LIVE at the restore (t0) -- the p4 delay evaluator's
    # t0_present_ids (zero entry wait, contract §8.14). Empty on legacy
    # records (the legacy extractor path needs neither).
    cav_ids: Tuple[str, ...] = ()
    t0_present_ids: Tuple[str, ...] = ()
    # p4-delay evaluator rows: EVERY physics step (dt_s = the physics step,
    # 0.5 s) in the campaign schema, exactly the cadence of the master's
    # physics rows and of the recorded-run evaluation. ``steps`` stays the
    # 1 s control-cadence telemetry (trajectories / TTC / safety windows).
    # Empty on legacy records (the evaluator then falls back to ``steps``,
    # correct only for fixtures whose ``steps`` are already every physics step).
    physics_steps: Tuple[StepRecord, ...] = ()


# ------------------------------------------------------------ collaborators
class StateSnapshotter:
    """Save/restore seam for the t0 SUMO + harness state (skeleton seam).

    ``save`` runs ONCE per predict_many call and returns an opaque manifest
    dict (state dir, SUMO version, network digest, harness components).
    Production subclasses call ``traci.simulation.saveState`` and deepcopy
    the harness session objects per PERFECT_SIM_DESIGN.md step 2; tests
    inject a fake. ``restore`` re-loads both halves for one branch IN THE
    BRANCH'S OWN FRESH PROCESS (``api``; PL-10d -- no restore without an
    explicit destination process), and refuses a second restore on a
    connection that already restored once (:class:`ProcessReuseError`,
    raised before any loadState).
    """

    def save(self, *, t0_s: float, controller_state, kept_wall_ids) -> dict:
        raise NotImplementedError(
            "StateSnapshotter.save requires a production or fake "
            "implementation (zero simulation: nothing is launched here)"
        )

    def restore(self, manifest, *, plan, draw_id, api) -> None:
        raise NotImplementedError(
            "StateSnapshotter.restore requires a production or fake "
            "implementation (the BranchRunner drives it per branch, into "
            "the branch's own fresh process -- PL-10d)"
        )


class BranchProcessFactory:
    """Process-launcher seam: ONE fresh SUMO process per restored branch.

    ``start_process(label=...)`` launches and connects one SUMO process and
    returns its traci connection; ``close_process(connection)`` retires it.
    PerfectSimWorldModel.predict_many drives one start/run/close cycle per
    (plan, draw) branch, native reference included, with the close in a
    finally clause so a failing branch still retires its process. Production
    implementations live at the sanctioned SUMO touchpoint (the re-smoke /
    campaign launcher, which owns process management and only ever signals
    its own children); tests inject fakes that record loadState calls per
    connection. NO SUMO/traci import here: the module stays zero-simulation
    at import time."""

    def start_process(self, *, label: Optional[str] = None):
        raise NotImplementedError(
            "BranchProcessFactory.start_process requires a production or "
            "fake implementation (zero simulation: nothing is launched here)"
        )

    def close_process(self, connection) -> None:
        raise NotImplementedError(
            "BranchProcessFactory.close_process requires a production or "
            "fake implementation (zero simulation: nothing is launched here)"
        )


_FRESH_PROCESS_MESSAGE = (
    "BranchRunner: no process factory injected -- PL-10d requires a FRESH "
    "SUMO process per restored branch (a process that already ran one "
    "branch, native or controlled, carries state across loadState and is "
    "refused: ProcessReuseError). Inject "
    "BranchRunner(process_factory=...) with start_process()/close_process()."
)


class ArrivalDraws:
    """Option-A seam: explicit post-t0 departure lists, one per draw k."""

    def draw_id(self, arrivals) -> str:
        raise NotImplementedError("ArrivalDraws.draw_id requires an implementation (CRN labels)")

    def departures(self, arrivals) -> Sequence:
        """The explicit post-t0 departure list installed after loadState."""
        raise NotImplementedError("ArrivalDraws.departures requires an implementation")


class SimpleArrivalDraws(ArrivalDraws):
    """Option-A draws (CRN by construction) with one declared exception.

    - Mapping payload with ``sample_id`` (contract section 3.1 option A):
      the payload IS the explicit departure list; the draw id is the
      sample's stable ``sample_id``.
    - Mapping payload ``{demand_model_id, seed}`` (the kit's ArrivalDraw):
      the K=1 RESTORED-RNG mode -- no explicit list is installed, the
      snapshot's SUMO RNG drives the post-t0 arrivals. CRN holds trivially
      (K = 1); this is the declared fallback for draw-style payloads, which
      need campaign-side support to expand (design option A note).
    - Any other sequence: explicit departure rows; the draw id is a content
      digest of the rows, so every plan's k-th branch sees the identical id.
    - ``None`` (contract v2.1 section 2: arrivals is OPTIONAL): no payload
      at all -- the draw id is the stable ``"arrivals:none"`` and no
      departure is installed (an empty-but-typed payload keeps its content
      digest). Never refused.

    Every plan's k-th branch and native see the identical payload, so
    common random numbers hold for each k."""

    # restored-RNG marker: an EMPTY explicit list means "install nothing;
    # the snapshot RNG drives arrivals" (runners must not draw their own)
    RESTORED_RNG_MARKER = "restored-rng"
    #: stable draw id for an absent (None) arrivals payload (v2.1 section 2)
    NO_ARRIVALS_ID = "arrivals:none"

    def draw_id(self, arrivals) -> str:
        if arrivals is None:
            return self.NO_ARRIVALS_ID
        if isinstance(arrivals, Mapping):
            if "demand_model_id" in arrivals or "seed" in arrivals:
                return self._restored_rng_id(
                    arrivals.get("demand_model_id", "draw"), arrivals.get("seed", 0)
                )
            return str(arrivals.get("sample_id", "draw"))
        demand_id = getattr(arrivals, "demand_model_id", None)  # kit ArrivalDraw
        if demand_id is not None:
            return self._restored_rng_id(demand_id, getattr(arrivals, "seed", 0))
        sample_id = getattr(arrivals, "sample_id", None)
        if sample_id is not None:
            return str(sample_id)
        return "arrivals:" + self._digest(arrivals)

    def departures(self, arrivals) -> Sequence:
        if arrivals is None:
            return ()  # OPTIONAL input (v2.1 section 2): nothing to install
        if isinstance(arrivals, Mapping):
            if "departures" in arrivals:
                return tuple(arrivals["departures"])
            if "demand_model_id" in arrivals or "seed" in arrivals:
                return ()  # restored-RNG mode: nothing explicit to install
            raise ValueError(
                "arrivals mapping must carry 'departures' (option A) or "
                "be an ArrivalDraw {demand_model_id, seed} (K=1 restored-RNG)"
            )
        if getattr(arrivals, "demand_model_id", None) is not None:
            return ()  # kit ArrivalDraw: restored-RNG mode
        if isinstance(arrivals, (str, bytes)):
            raise ValueError("arrivals payload must be a departure list")
        return arrivals if isinstance(arrivals, Sequence) else tuple(arrivals)

    @classmethod
    def _restored_rng_id(cls, demand_model_id, seed) -> str:
        return f"{cls.RESTORED_RNG_MARKER}:{demand_model_id}:{seed}"

    @staticmethod
    def _digest(rows) -> str:
        def canonical(row):
            if dataclasses.is_dataclass(row) and not isinstance(row, type):
                return dataclasses.asdict(row)
            if isinstance(row, Mapping):
                return dict(row)
            return row

        payload = json.dumps(
            [canonical(r) for r in rows], sort_keys=True, default=repr, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class BranchRunner:
    """One plan x one draw branch (skeleton seam; owns real traci in prod).

    ``run`` restores the saved state from ``manifest`` IN THE BRANCH'S OWN
    CONNECTION (``connection`` is a REQUIRED keyword -- there is no
    in-process fallback), installs the explicit draw, restores the harness
    snapshot, applies ``plan`` (dwell 0; walls via the WM-40 wall-group
    path), runs ``horizon_s`` and returns a :class:`BranchRecord`.
    ``stop_on_window_collision`` implements early stop: a collision inside
    ``safety_window_s`` ends the branch and sets ``BranchRecord.stopped_early``.

    PL-10 (contract v2): ``commit_s`` / ``continuation`` carry the commit-
    window semantics -- apply ``plan`` for ``commit_s`` starting at t0, then
    continue under ``continuation`` ("native": every CAV back to native;
    "planner": closed-loop pi_k, PL-04). None (v1/v1.1 calls) keeps the v1
    semantics: the plan is constant from t0 for the whole horizon.

    PL-10d (fresh process per branch, the ONLY path): the runner carries a
    :class:`BranchProcessFactory`; ``start_process()`` hands out one fresh
    SUMO connection per branch and ``close_process(connection)`` retires it
    after the branch (PerfectSimWorldModel.predict_many wraps every branch
    in try/finally, so exceptions close the process too). The snapshotter's
    per-connection ledger refuses a second restore on any connection
    (:class:`ProcessReuseError`) BEFORE a second loadState can be issued.
    """

    def __init__(self, process_factory: Optional[BranchProcessFactory] = None):
        self.process_factory = process_factory

    def start_process(self, *, label: Optional[str] = None):
        """ONE fresh SUMO process/connection for one branch (PL-10d)."""
        factory = getattr(self, "process_factory", None)
        if factory is None:
            raise NotImplementedError(_FRESH_PROCESS_MESSAGE)
        return factory.start_process(label=label)

    def close_process(self, connection) -> None:
        """Retire the branch's process (called in a finally clause)."""
        factory = getattr(self, "process_factory", None)
        if factory is None:
            raise NotImplementedError(_FRESH_PROCESS_MESSAGE)
        factory.close_process(connection)

    def run(
        self,
        plan,
        *,
        connection,
        manifest,
        draw_id,
        departures,
        horizon_s,
        safety_window_s,
        controller_state,
        kept_wall_ids,
        stop_on_window_collision=True,
        commit_s=None,
        continuation=None,
    ) -> BranchRecord:
        raise NotImplementedError(
            "BranchRunner.run requires a production or fake implementation "
            "(zero simulation: nothing is launched here); the branch runs "
            "in the fresh process handed out by start_process() -- PL-10d"
        )


class TraciStateSnapshotter(StateSnapshotter):
    """Production save/restore over the REAL WM-41 saved-state protocol.

    All traci access goes through the injected ``api`` (the real traci
    module at launch, a fake in tests) -- nothing is imported or launched at
    module import time. ``save`` writes ``<state_root>/decision_NNNNNN/``
    via the real ``campaign_pilot_v1.saved_state.save_branch_state`` when the
    wm34 tree is importable (read-only import), else via the vendored thin
    adapter (:mod:`traffic_shepherd.planner.perfect_sim_state`) -- both run
    the exactness guard, so a low-precision or RNG-less snapshot is REFUSED
    loudly (PERFECT_SIM_DESIGN.md 'Snapshot exactness requirements'):
    the saving SUMO must run with ``--save-state.precision 17`` and
    ``--save-state.rng``. ``restore`` re-validates the manifest (schema,
    branch time, version drift, digest, scenario digests, state options)
    before ``api.simulation.loadState`` and refuses anything that cannot
    restore bit-exactly. PL-10d: ``restore`` targets the BRANCH'S OWN
    process (required ``api`` -- the fresh connection
    :meth:`BranchRunner.start_process` handed out) and keeps a per-connection
    ledger: a connection that restored once is SPENT, and any further
    restore on it raises :class:`ProcessReuseError` BEFORE loadState is
    invoked (the re-smoke measured state carry-over across loadState; one
    loadState per process, ever)."""

    def __init__(
        self,
        api,
        state_root,
        *,
        seed: int = 0,
        scenario_digests: Optional[Mapping[str, str]] = None,
        sumo_version=PINNED_SUMO_VERSION,
        harness_state_factory=None,
        physics_row_factory=None,
        wm34_output_root: str = WM34_OUTPUT_ROOT,
    ):
        self.api = api
        self.state_root = Path(state_root)
        self.seed = int(seed)
        self.scenario_digests = dict(scenario_digests or {})
        self.sumo_version = parse_sumo_version(self._resolve_sumo_version(api, sumo_version))
        self.harness_state_factory = harness_state_factory
        self.physics_row_factory = physics_row_factory
        self._real = optional_real_saved_state(wm34_output_root)
        self._decision = 0
        # PL-10d per-process restore ledger: (connection, first branch id)
        # pairs, matched by identity. A connection listed here is SPENT.
        self._spent_processes: "list[tuple[object, str]]" = []

    @staticmethod
    def _resolve_sumo_version(api, sumo_version):
        """The CONNECTED interpreter's version wins over a caller constant:
        when the caller left the default, ask the injected api (the live
        traci module exposes getVersion(); fakes without it keep the pinned
        default). PL-10c nit: never record an unqueried version where the
        live traci object is available."""
        if sumo_version != PINNED_SUMO_VERSION:
            return sumo_version
        get_version = getattr(api, "getVersion", None)
        if callable(get_version):
            return get_version()
        return PINNED_SUMO_VERSION

    # ------------------------------------------------------------------ api
    def save(self, *, t0_s: float, controller_state, kept_wall_ids) -> dict:
        require_sumo_rng_state(self.sumo_version)
        self._decision += 1
        world_dir = self.state_root / f"decision_{self._decision:06d}"
        world_dir.mkdir(parents=True, exist_ok=False)
        harness_state = (
            dict(self.harness_state_factory(controller_state, kept_wall_ids))
            if self.harness_state_factory is not None
            else {
                "controller_state": json.loads(
                    json.dumps(dict(controller_state or {}), default=repr, sort_keys=True)
                ),
                "kept_wall_ids": list(kept_wall_ids),
            }
        )
        physics_row = (
            self.physics_row_factory(t0_s)
            if self.physics_row_factory is not None
            else {"time_s": float(t0_s)}
        )
        if self._real is not None:
            saved = self._real.save_branch_state(
                self.api,
                world_dir,
                t_b=float(t0_s),
                seed=self.seed,
                harness_state=harness_state,
                scenario_digests=self.scenario_digests,
                sumo_version=tuple(self.sumo_version),
                t_b_physics_row=physics_row,
            )
            manifest = dict(saved["manifest"])
            manifest["implementation"] = "campaign_pilot_v1.saved_state"
            saved_state_id = saved["saved_state_id"]
        else:
            state_dir = world_dir / STATE_DIR_NAME
            state_dir.mkdir(parents=True)
            state_file = state_dir / SUMO_STATE_FILE
            self.api.simulation.saveState(str(state_file))
            if not state_file.is_file() or state_file.stat().st_size == 0:
                raise StateSnapshotError(
                    "traci.simulation.saveState produced no state file; "
                    "refusing to register the branch point"
                )
            check = require_exact_state_file(state_file)  # the WM-44 v8 guard
            manifest = {
                "schema": SCHEMA,
                "seed": self.seed,
                "branch_point_t0_s": float(t0_s),
                "state_options": {
                    "save_state_precision_decimals": check["max_pos_decimals"],
                    "rng_state_in_file": check["has_rng"],
                    "required_cli": list(REQUIRED_SAVE_STATE_CLI),
                },
                "physics_row_at_t0": physics_row,
                "sumo_version": list(self.sumo_version),
                "sumo_state_file": SUMO_STATE_FILE,
                "sumo_state_sha256": sha256_file(state_file),
                "harness_state": harness_state,
                "harness_components": list(HARNESS_COMPONENTS),
                "scenario_digests": dict(self.scenario_digests),
                "implementation": "vendored_adapter(perfect_sim_state)",
            }
            manifest["saved_state_id"] = sha256_text(
                write_manifest(state_dir, manifest).read_text(encoding="utf-8")
                + manifest["sumo_state_sha256"]
            )
            saved_state_id = manifest["saved_state_id"]
        return {
            "saved_state_dir": str(world_dir / STATE_DIR_NAME),
            "saved_state_id": saved_state_id,
            "manifest": manifest,
            "t0_s": float(t0_s),
        }

    def restore(self, manifest, *, plan, draw_id, api) -> None:
        """Restore into the BRANCH'S OWN process (required ``api``; PL-10d).

        The per-connection ledger is checked FIRST: a connection that
        already restored once raises :class:`ProcessReuseError` before any
        validation output and before loadState is invoked. The connection is
        marked spent BEFORE ``loadState`` is issued, so even a failed load
        spends the process (one loadState per process, ever)."""
        connection = api
        branch_id = f"{getattr(plan, 'label', 'plan')}@{draw_id}"
        self._require_fresh_process(connection)
        info = dict(manifest)
        state_dir = Path(info["saved_state_dir"])
        if self._real is not None:
            self._mark_spent(connection, branch_id)
            self._real.load_branch_state(
                connection,
                state_dir.parent,
                expected_t_b=float(info["t0_s"]),
                sumo_version=tuple(self.sumo_version),
                scenario_digests=self.scenario_digests,
            )
            return
        validate_manifest_for_restore(
            info["manifest"],
            expected_t0_s=float(info["t0_s"]),
            sumo_version=tuple(self.sumo_version),
            scenario_digests=self.scenario_digests,
            state_path=state_dir / SUMO_STATE_FILE,
        )
        self._mark_spent(connection, branch_id)
        connection.simulation.loadState(str(state_dir / SUMO_STATE_FILE))
        restored = float(connection.simulation.getTime())
        if abs(restored - float(info["t0_s"])) > 1e-6:
            raise StateSnapshotError(f"loadState landed at t={restored}, expected {info['t0_s']}")

    # ------------------------------------------------------- PL-10d ledger
    def _require_fresh_process(self, connection) -> None:
        """Refuse any connection that already restored+ran one branch.

        Identity-matched (``is``), so distinct fake/real connection objects
        never collide; the refusal precedes every loadState by construction
        (the ledger is appended only inside :meth:`restore`, before the
        load calls)."""
        for spent_connection, spent_branch in self._spent_processes:
            if spent_connection is connection:
                raise ProcessReuseError(
                    "process reuse refused (PL-10d): this SUMO process "
                    f"already restored+ran branch {spent_branch!r}; a "
                    "process runs EXACTLY ONE restored branch (native "
                    "included -- state carries over across loadState). "
                    "Start a fresh process for the next branch "
                    "(BranchRunner.start_process())."
                )

    def _mark_spent(self, connection, branch_id: str) -> None:
        self._spent_processes.append((connection, branch_id))


class TraciBranchRunner(BranchRunner):
    """Production branch driver: restore, install the draw, dispatch, run.

    Owns the real traci calls of one branch (injected ``api`` and
    ``snapshotter``; zero simulation at import). The plan is applied through
    the mixed_assignment/3 dispatcher (``AssignmentPlan(assignments=...,
    plan_id=..., commit_horizon_s=0.0, walls=(WallAssignment...),
    strict_roster=True)`` -- walls via the WM-40 wall-group path) at the
    first control tick. With ``commit_s``/``continuation`` set (contract v2)
    an all-native coloring is applied at t0 + commit_s under
    ``continuation='native'`` (wall omission releases the wall members;
    'planner' is refused -- PL-04).

    The frozen control write itself (sensor frames -> ownership ->
    ``MixedAssignmentSession.step``) is the injected ``tick_hook`` -- PL-02
    binds the frozen factories there exactly as the real ``suffix_runner``
    does; this runner owns restore/draw/dispatch/telemetry and records
    :class:`StepRecord` telemetry at the 1 s control cadence (the
    trajectory grid; physics advances at ``dt_s``), teleports
    (``getStartingTeleportIDList``), collisions and wall events
    (``session.wall_records``) into the :class:`BranchRecord` for the
    fail-closed validity heads.

    PL-04c fail-closed control write: a branch whose plan assigns ANY
    non-native action or wall (the N1 kept walls included) refuses to run
    without a control write -- ``run`` raises
    :class:`MissingControlWriteError` BEFORE the restore, because with no
    ``tick_hook`` the branch would silently simulate as native. A
    native-only plan may run without one. The write is injected as EITHER
    ``tick_hook`` (a plain ``callable(session, now_s)``, shared by every
    branch) OR ``tick_hook_factory`` (a
    ``callable(connection=..., session=..., plan=...)`` invoked ONCE per
    branch, on the branch's own fresh connection, returning the per-branch
    ``callable(session, now_s)``; both is refused). A per-branch hook may
    expose ``release()`` -- the end-of-branch ownership handback -- which
    ``run`` calls in a ``finally`` (branch end OR error, e.g. the real
    ``owner.release_all``; PL-04c).

    PL-10d: EVERY traci call of the branch goes through the branch's own
    fresh connection (required ``run(connection=...)`` -- handed out by
    :meth:`BranchRunner.start_process`); there is no in-process fallback.
    The constructor ``api`` (the master connection the snapshotter saves
    through) is retained for wiring symmetry but is NOT used to run
    branches: it may be a live connection, a LAZY zero-argument provider
    resolved at first use (the closed-loop build passes
    ``lambda: harness.master_api`` -- PL-04i's build-then-start order
    means the master connection does not exist at build time), or None.
    Nothing here ever calls it."""

    def __init__(
        self,
        api=None,
        *,
        snapshotter=None,
        cav_ids,
        tracked_ids=(),
        free_flow_of=None,
        dt_s: float = 0.5,
        control_dt_s: float = 1.0,
        dispatcher_factory=None,
        plan_factory=None,
        tick_hook=None,
        tick_hook_factory=None,
        process_factory: Optional[BranchProcessFactory] = None,
    ):
        super().__init__(process_factory=process_factory)
        if tick_hook is not None and tick_hook_factory is not None:
            raise ValueError(
                "TraciBranchRunner takes ONE control write: tick_hook (a "
                "plain callable(session, now_s) shared by every branch) or "
                "tick_hook_factory (called per branch with connection=/"
                "session=/plan=), not both"
            )
        # wiring symmetry only: a connection, a lazy provider, or None --
        # stored as-is, never called (branches run on their OWN connection)
        self.api = api
        self.snapshotter = snapshotter
        self.cav_ids = tuple(str(c) for c in cav_ids)
        self.tracked_ids = tuple(str(t) for t in (tracked_ids or self.cav_ids))
        self.free_flow_of = free_flow_of or (lambda vid: 0.0)
        self.dt_s = float(dt_s)
        self.control_dt_s = float(control_dt_s)
        self._dispatcher_factory = dispatcher_factory
        # T-B evidence: branches that restored the master's carried state
        self.carried_state_restores = 0
        self._plan_factory = plan_factory
        self._tick_hook = tick_hook
        self._tick_hook_factory = tick_hook_factory

    # ------------------------------------------------------------ dispatch
    @staticmethod
    def _plan_needs_control_write(plan, kept_wall_ids) -> bool:
        """Whether this branch assigns ANY non-native action or any wall.

        The planner re-decides, so a PlanCandidate with any non-native
        ``per_cav`` entry -- or any wall -- needs the frozen control write
        over its whole commit window. The N1 native reference is
        all-native EXCEPT the kept walls it continues (their locked
        members stage as wall-v2 at the staged wall speed -- control
        writes too)."""
        if getattr(plan, "label", None) == NATIVE_PLAN_LABEL:
            return bool(kept_wall_ids)
        per_cav = getattr(plan, "per_cav", None) or {}
        if any(str(action) != "native" for action in per_cav.values()):
            return True
        return bool(getattr(plan, "walls", None))
    def _wall_assignment(self, wall):
        from traffic_shepherd.mixed_assignment import WallAssignment  # noqa: PLC0415

        return WallAssignment(
            wall_id=wall.wall_id,
            members=tuple(wall.members),
            wall_speed_mps=float(wall.speed_mps),
            phase=str(wall.phase),
            hold_s=None if wall.hold_s is None else float(wall.hold_s),
            release_at_s=None if wall.release_at_s is None else float(wall.release_at_s),
        )

    def _build_plan(self, plan, plan_id: str, controller_state, kept_wall_ids):
        """Planner PlanCandidate (or NATIVE_PLAN) -> AssignmentPlan.

        N1 for the native reference: walls the controller reports in the
        ASSEMBLE phase at t0 continue (their locked members go wall-v2 at
        the staged wall speed; hold/release walls are dissolved -- everyone
        native, no wall staged)."""
        if self._plan_factory is not None:
            return self._plan_factory(plan, plan_id, controller_state, kept_wall_ids)
        from traffic_shepherd.mixed_assignment import (  # noqa: PLC0415
            AssignmentPlan,
        )

        if getattr(plan, "label", None) == NATIVE_PLAN_LABEL:
            assignments = {cav: "native" for cav in self.cav_ids}
            walls = []
            for wall_id in kept_wall_ids or ():
                entries = [
                    (cav, entry)
                    for cav, entry in sorted((controller_state or {}).items())
                    if entry.get("wall_id") == wall_id
                ]
                members = tuple(cav for cav, entry in entries)
                if len(members) < 2:
                    raise StateSnapshotError(
                        f"kept wall {wall_id!r} has {len(members)} member(s) in "
                        "the controller state; a wall spans several lanes -- "
                        "the N1 reference cannot be staged"
                    )
                speeds = [entry.get("wall_speed_mps") for _, entry in entries]
                if any(s is None for s in speeds):
                    raise StateSnapshotError(
                        f"kept wall {wall_id!r} is missing wall_speed_mps in "
                        "the controller state; the N1 reference cannot be "
                        "staged (fail loudly, never guess a wall speed)"
                    )
                walls.append(self._wall_assignment(_NativeWall(wall_id, members, float(speeds[0]))))
            return AssignmentPlan(
                assignments=assignments,
                plan_id=plan_id,
                commit_horizon_s=0.0,
                walls=tuple(walls),
                strict_roster=True,
            )
        return AssignmentPlan(
            assignments=dict(plan.per_cav),
            plan_id=plan_id,
            commit_horizon_s=0.0,  # the planner re-decides; dwell 0
            walls=tuple(self._wall_assignment(w) for w in plan.walls),
            strict_roster=True,
        )

    # ------------------------------------------------------------------ api
    def run(
        self,
        plan,
        *,
        connection,
        manifest,
        draw_id,
        departures,
        horizon_s,
        safety_window_s,
        controller_state,
        kept_wall_ids,
        stop_on_window_collision=True,
        commit_s=None,
        continuation=None,
    ) -> BranchRecord:
        from traffic_shepherd.mixed_assignment import (  # noqa: PLC0415
            MixedAssignmentSession,
        )

        if continuation == "planner":
            raise NotImplementedError(_PLANNER_CONTINUATION_MESSAGE)
        if connection is None:
            raise NotImplementedError(
                "TraciBranchRunner.run requires the branch's own fresh "
                "SUMO connection (BranchRunner.start_process(); PL-10d "
                "fresh process per branch) -- no in-process fallback exists"
            )
        api = connection
        is_native = getattr(plan, "label", None) == NATIVE_PLAN_LABEL
        # T-A (#16): the NATIVE reference branch is NEVER early-stopped --
        # a truncated native window breaks the paired delay comparison and
        # makes the PL-06 native-collision option A inoperative. Plan
        # branches keep the requested behaviour (they are vetoed on an
        # in-window collision either way; their early stop saves the
        # remainder of a window that is already vetoed).
        if is_native and stop_on_window_collision:
            stop_on_window_collision = False
        state_id = str(manifest.get("saved_state_id", ""))[:12]
        plan_id = f"{'native' if is_native else 'plan'}@{state_id}:{draw_id}"
        # 0) PL-04c fail-closed: a branch with any non-native action or
        #    wall is refused WITHOUT its control write BEFORE the restore
        #    (a silent native simulation would be a wrong lookahead value;
        #    no loadState is spent on a branch that cannot run honestly).
        needs_control_write = self._plan_needs_control_write(plan, kept_wall_ids)
        if (
            needs_control_write
            and self._tick_hook is None
            and self._tick_hook_factory is None
        ):
            raise MissingControlWriteError(
                f"branch {plan_id!r} assigns non-native actions or walls, but "
                "this TraciBranchRunner was built without a control write "
                "(tick_hook): the branch would silently simulate as native. "
                "Build the runner with a tick_hook, or a tick_hook_factory "
                "for the per-branch real write on the branch's own fresh "
                "connection (PL-04c)."
            )
        t0 = float(manifest["t0_s"])
        horizon = float(horizon_s)
        window_end = t0 + (float(commit_s) if commit_s is not None else horizon)
        # 1) restore the exact snapshot INTO THIS PROCESS (the snapshotter
        #    owns the guard and the per-process reuse ledger)
        self.snapshotter.restore(manifest, plan=plan, draw_id=draw_id, api=api)
        # T-A (#1): the vehicles LIVE at t0 -- the p4 delay evaluator's
        # t0_present_ids (inserted at/before t0 -> zero entry wait, §8.14).
        # Captured BEFORE any option-A departure is installed.
        t0_present = tuple(sorted(str(v) for v in api.vehicle.getIDList()))
        # 2) install the explicit post-t0 draw, if option A provided one
        for row in departures or ():
            self._add_departure(row, api)
        # 3) the dispatcher session (WM-40 wall groups; per-CAV plans)
        session = (
            self._dispatcher_factory(controller_state, kept_wall_ids)
            if self._dispatcher_factory is not None
            else MixedAssignmentSession(
                circles_controller=self._damping_controller(),
                pi_session=None,
                cav_ids=set(self.cav_ids),
            )
        )
        # 4) the per-branch control write: the factory runs ONCE per branch
        #    on THIS branch's own fresh connection (PL-04c); a shared plain
        #    tick_hook is used as-is. The hook may carry release() for the
        #    end-of-branch ownership handback (called in the finally below).
        hook = self._tick_hook
        # T-B (#4): the ownership control write runs in EVERY branch when a
        # factory exists -- the all-native branches included -- so
        # CommandOwnership configures speedMode 27 on every live CAV and
        # releases any inherited override (a native branch must not keep
        # the master's damping commands alive by not writing at all).
        if hook is None and self._tick_hook_factory is not None:
            hook = self._tick_hook_factory(connection=api, session=session, plan=plan)
            if not callable(hook):
                raise MissingControlWriteError(
                    "tick_hook_factory must return the per-branch control "
                    f"write (a callable(session, now_s)); got "
                    f"{type(hook).__name__}"
                )
        # T-B (#1/#2): the master's carried control-loop state at the
        # snapshot (ownership _owned/_last/_time, loop state, dispatcher
        # committed assignments/clocks, PI recurrence) is restored into the
        # fresh session and its ownership BEFORE the plan is applied, so the
        # branch's tick at t0 starts from exactly the state the master's
        # tick at t0 does (PERFECT_SIM_DESIGN step 2b/c).
        carried = ((manifest.get("manifest") or {}).get("harness_state")) or {}
        if (
            hook is not None
            and isinstance(carried, Mapping)
            and "ownership" in carried
            and hasattr(hook, "owner")
            and hasattr(hook, "loop_state")
        ):
            restore_harness_state(hook, session, carried)
            self.carried_state_restores += 1
        wall_plan = self._build_plan(plan, plan_id, controller_state, kept_wall_ids)
        native_plan = self._build_plan(NATIVE_PLAN, plan_id + ":native", controller_state, ())
        steps: list = []
        physics: list = []  # p4-evaluator rows: EVERY physics step (0.5 s)
        collisions: list = []
        teleports: list = []
        wall_events: dict = {}
        stopped_early = False
        stopped_at = None
        now = float(api.simulation.getTime())
        end = t0 + horizon
        released = not (commit_s is not None and continuation == "native")
        plan_applied = False
        n_drained = 0
        try:
            while now < end - 1e-9:
                if not plan_applied:
                    session.apply_plan(wall_plan, now_s=now)
                    plan_applied = True
                if not released and now >= window_end - 1e-9:
                    # continuation='native': all CAVs native after the commit
                    # window (wall omission releases staged walls)
                    session.apply_plan(native_plan, now_s=now)
                    released = True
                if hook is not None:
                    hook(session, now)
                api.simulationStep()
                now = float(api.simulation.getTime())
                teleports.extend(str(v) for v in api.simulation.getStartingTeleportIDList())
                colliding = [str(v) for v in api.simulation.getCollidingVehiclesIDList()]
                if colliding:
                    collisions.append(now)
                if abs(now / self.control_dt_s - round(now / self.control_dt_s)) < 1e-6:
                    # telemetry at the 1 s control cadence (the trajectory grid)
                    full = self._vehicle_step(now, api, colliding)
                    steps.append(full)
                    physics.append(full)
                else:
                    # the in-between physics step: the p4 delay bills presence
                    # at EVERY physics row (the evaluation's definition)
                    physics.append(self._light_step(now, api, colliding))
                n_drained = self._drain_walls(session, wall_events, n_drained)
                if (
                    stop_on_window_collision
                    and collisions
                    and collisions[-1] <= t0 + float(safety_window_s) + 1e-9
                ):
                    stopped_early = True
                    stopped_at = collisions[-1]
                    break
        finally:
            release = getattr(hook, "release", None)
            if callable(release):
                # end-of-branch ownership handback (BranchControlWrite:
                # owner.release_all) -- a branch that raises still hands
                # its CAVs back before the process is retired
                release()
        return BranchRecord(
            plan_label=plan_id,
            draw_id=draw_id,
            t0_s=t0,
            horizon_s=horizon,
            dt_s=self.dt_s,
            free_flow_s={vid: float(self.free_flow_of(vid)) for vid in self.tracked_ids},
            steps=tuple(steps),
            physics_steps=tuple(physics),
            collision_times=tuple(collisions),
            wall_events=wall_events,
            tracked=self.tracked_ids,
            stopped_early=stopped_early,
            stopped_at_s=stopped_at,
            teleport_times=tuple(teleports),
            cav_ids=tuple(self.cav_ids),
            t0_present_ids=t0_present,
        )

    # -------------------------------------------------------------- plumbing
    def _damping_controller(self):
        from traffic_shepherd.config import load_config  # noqa: PLC0415
        from traffic_shepherd.controller_factory import create_controller  # noqa: PLC0415

        controller, _identity = create_controller("damping", load_config())
        return controller

    def _add_departure(self, row, api) -> None:
        """One explicit option-A departure row -> traci vehicle add.

        Accepted shapes: (t_s, route, speed_mps, is_cav, vehicle_type) tuples
        (the contract option-A payload) or a mapping with those keys plus an
        optional ``vehicle_id``."""
        if isinstance(row, Mapping):
            rid = str(row.get("vehicle_id", f"arr_{row.get('t_s', '')}"))
            route = str(row.get("route", row.get("edge", "")))
            speed = float(row.get("speed_mps", 0.0) or 0.0)
            type_id = str(row.get("vehicle_type", "passenger"))
        else:
            values = (list(row) + [None] * 5)[:5]
            t_s, route, speed, _is_cav, type_id = values
            rid = f"arr_{t_s}"
            route = str(route)
            speed = float(speed or 0.0)
            type_id = str(type_id)
        api.vehicle.add(rid, route, typeID=type_id, departSpeed=speed, departLane="free")

    def _vehicle_step(self, now: float, api, colliding: Sequence[str] = ()) -> StepRecord:
        vehicles: dict = {}
        # Campaign physics-row surfaces: the REAL traci api provides all of
        # them; a bare test fake may omit the row-identity / minGap methods
        # (recorded as absent -> None / +0.0 gap, the pre-T-A telemetry).
        get_road = getattr(api.vehicle, "getRoadID", None)
        get_lane_id = getattr(api.vehicle, "getLaneID", None)
        get_min_gap = getattr(api.vehicle, "getMinGap", None)
        for raw_id in api.vehicle.getIDList():
            vid = str(raw_id)
            leader = api.vehicle.getLeader(vid, 200.0)
            has_leader = bool(leader) and bool(leader[0])
            # T-A (#6): the PHYSICAL bumper-to-bumper gap -- the raw 200 m
            # query gap PLUS the follower's minGap, exactly what
            # ``metrics.ttc_exposure`` adds before its closing_ttc call.
            min_gap = float(get_min_gap(vid)) if get_min_gap is not None else 0.0
            vehicles[vid] = VehicleStep(
                s_m=float(api.vehicle.getLanePosition(vid)),
                lane_id=float(api.vehicle.getLaneIndex(vid)),
                speed_mps=float(api.vehicle.getSpeed(vid)),
                accel_mps2=float(api.vehicle.getAcceleration(vid)),
                on_section=True,
                leader_gap_m=(float(leader[1]) + min_gap) if has_leader else None,
                leader_speed_mps=(
                    float(api.vehicle.getSpeed(str(leader[0]))) if has_leader else None
                ),
                leader_id=str(leader[0]) if has_leader else None,
                road_id=str(get_road(vid)) if get_road is not None else None,
                sumo_lane_id=str(get_lane_id(vid)) if get_lane_id is not None else None,
            )
        return StepRecord(time_s=now, dt_s=self.dt_s, vehicles=vehicles,
                          colliding_ids=tuple(str(v) for v in colliding))

    def _light_step(self, now: float, api, colliding: Sequence[str] = ()) -> StepRecord:
        """A physics-cadence row carrying ONLY the campaign fields the p4
        delay reads (road/lane identity, position, speed, acceleration) --
        no leader queries (those stay on the 1 s telemetry rows)."""
        get_road = getattr(api.vehicle, "getRoadID", None)
        get_lane_id = getattr(api.vehicle, "getLaneID", None)
        vehicles: dict = {}
        for raw_id in api.vehicle.getIDList():
            vid = str(raw_id)
            vehicles[vid] = VehicleStep(
                s_m=float(api.vehicle.getLanePosition(vid)),
                lane_id=float(api.vehicle.getLaneIndex(vid)),
                speed_mps=float(api.vehicle.getSpeed(vid)),
                accel_mps2=float(api.vehicle.getAcceleration(vid)),
                on_section=True,
                road_id=str(get_road(vid)) if get_road is not None else None,
                sumo_lane_id=str(get_lane_id(vid)) if get_lane_id is not None else None,
            )
        return StepRecord(time_s=now, dt_s=self.dt_s, vehicles=vehicles,
                          colliding_ids=tuple(str(v) for v in colliding))

    @staticmethod
    def _drain_walls(session, current: dict, n_consumed: int) -> int:
        """New ``session.wall_records`` events -> WallEventRecord updates.

        Wall-group event dicts carry ``event`` in {assembly_start, assembled,
        released, safety_abort} with ``timestamp_s`` (mixed_assignment/3
        WallGroupSession contract); only the assembled/safety_abort pair
        maps onto the contract's wall outputs (abort code 1 = safety).
        Returns the advanced consumed count."""
        records = getattr(session, "wall_records", ()) or ()
        while n_consumed < len(records):
            event = records[n_consumed]
            n_consumed += 1
            wall_id = str(event.get("wall_id"))
            if event.get("event") == "assembled":
                current[wall_id] = WallEventRecord(
                    assembled_at_s=float(event.get("timestamp_s", 0.0))
                )
            elif event.get("event") == "safety_abort":
                prior = current.get(wall_id, WallEventRecord())
                current[wall_id] = WallEventRecord(
                    assembled_at_s=prior.assembled_at_s, safety_abort=True, abort_reason=1
                )
        return n_consumed


class _NativeWall:
    """Minimal WallSpec-shaped stand-in for the N1 kept-wall staging."""

    def __init__(self, wall_id: str, members, speed_mps: float):
        self.wall_id = wall_id
        self.members = tuple(members)
        self.speed_mps = float(speed_mps)
        self.phase = "assemble"
        self.hold_s = 3600.0  # open hold: the continuation release ends it
        self.release_at_s = None


class MetricsExtractor:
    """BranchRecord -> contract outputs (design Outputs mapping).

    Delay (vehicle-seconds): T-A (#1) the branch's delay is the p4
    EVALUATION delay whenever the record carries campaign physics rows and
    the extractor was built with a ``delay_evaluator`` (the real wiring
    builds :class:`P4DelayEvaluator`): ALL corridor vehicles (CAVs and
    HDVs alike), advance-based free flow, and the §8.14 entry waits incl.
    the insertion backlog -- the same function over the same rows and
    window the evaluation calls. Records WITHOUT road identity (bare test
    fixtures) keep the legacy time-on-section arithmetic (CAV free-flow
    credit, zero waits). ``delay_part_a_s`` covers [t0, t0 + part_a] (the
    committed window), part B the remainder (contract v2: part_a IS the
    commit window) -- both from the same evaluator.
    Safety covers the safety window (``safety_window_s``, default = part_a;
    v2: commit_s): collision presence, minimum closing TTC, TTC < 3 s
    exposure (``metrics.closing_ttc`` semantics over the PHYSICAL gap the
    runner recorded; T-A adds the PRIMARY TTC < 1.5 s exposure with the
    identical pair arithmetic -- the frozen 3 s field keeps its meaning),
    hard braking via ``metrics.braking_effort`` (accel < -3 m/s^2
    vehicle-seconds), plus realised ``has_closing_pair`` bookkeeping (any
    follower-leader pair closing in the window; v1.1/v2 outputs: min TTC is
    NaN exactly where no closing pair exists). With ``future_horizon_s``
    set, the realised future block over (t0 + safety_window,
    t0 + future_horizon_s] is reported the same way (contract v2 section 3:
    exposure, hard braking, 0/1 collision, min TTC) at BOTH thresholds.
    Trajectories: (T, 4) per tracked vehicle at 1 s steps,
    T = trajectory_s + 1 incl. t0, NaN before insertion and from the exit
    step on; columns (t_s - t0, s_m, lane_id, speed_mps).
    """

    def __init__(self, *, delay_evaluator=None):
        # T-A (#1): callable(rows, t0_s, window_s) -> delay vehicle-seconds
        # over the branch's campaign-schema physics rows (P4DelayEvaluator).
        # None = the legacy time-on-section arithmetic (unchanged).
        self.delay_evaluator = delay_evaluator

    def extract(
        self,
        record: BranchRecord,
        *,
        part_a_horizon_s: float,
        trajectory_s: float,
        safety_window_s: Optional[float] = None,
        future_horizon_s: Optional[float] = None,
    ) -> dict:
        window_s = part_a_horizon_s if safety_window_s is None else float(safety_window_s)
        if self.delay_evaluator is not None:
            rows = self._physics_rows(record)
            present = frozenset(str(v) for v in (record.t0_present_ids or ()))
            delay = float(self.delay_evaluator(rows, record.t0_s, record.horizon_s, present))
            delay_a = float(self.delay_evaluator(rows, record.t0_s, part_a_horizon_s, present))
        else:
            on_section, waits = self._transit(record, record.horizon_s)
            free_flow = record.free_flow_s
            delay = sum(on_section[v] - float(free_flow.get(v, 0.0)) for v in on_section) + sum(
                waits.values()
            )
            on_a, waits_a = self._transit(record, part_a_horizon_s)
            delay_a = sum(
                on_a[v] - float(free_flow.get(v, 0.0)) for v in on_a
            ) + sum(waits_a.values())
        window = self._steps_in(record, 0.0, window_s)
        collision = self._collision_in(record, 0.0, window_s)
        min_ttc, exposure, exposure_lt1p5, hard_brake, has_pair = self._safety(window)
        out = {
            "min_ttc_pair": self._min_ttc_pair(window, record),
            "delay_s": delay,
            "delay_part_a_s": delay_a,
            "delay_part_b_s": delay - delay_a,
            "collision": collision,
            "min_ttc_s": min_ttc,
            "ttc_lt3_exposure_s": exposure,
            "ttc_lt1p5_exposure_s": exposure_lt1p5,
            "hard_brake_s": hard_brake,
            "has_closing_pair": has_pair,
            "trajectories": self._trajectories(record, trajectory_s),
        }
        if future_horizon_s is not None:
            future = self._steps_in(record, window_s, float(future_horizon_s))
            f_min, f_exposure, f_exposure_lt1p5, f_brake, f_pair = self._safety(future)
            out.update(
                {
                    "future_collision": self._collision_in(
                        record, window_s, float(future_horizon_s)
                    ),
                    "future_min_ttc_s": f_min,
                    "future_ttc_lt3_exposure_s": f_exposure,
                    "future_ttc_lt1p5_exposure_s": f_exposure_lt1p5,
                    "future_hard_brake_s": f_brake,
                    "future_has_closing_pair": f_pair,
                }
            )
        return out

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _physics_rows(record: BranchRecord) -> list:
        """The branch's steps as campaign-schema physics rows (T-A #1).

        The SAME schema the master writes (recorded.Layout defaults:
        time_s / vehicles / road_id / lane_id / lane_position_m / speed_mps
        / acceleration_mps2 / colliding_ids / cav_ids), so the p4 delay
        evaluator consumes the branch exactly like a recorded run. Raises
        when the steps carry no road identity (a legacy fixture cannot be
        evaluated under the p4 definition -- never silently fall back)."""
        rows = []
        # every-physics-step rows when the runner recorded them (the p4
        # evaluation's cadence); legacy records fall back to the 1 s steps
        for step in (record.physics_steps or record.steps):
            vehicles = {}
            for vid, vs in step.vehicles.items():
                if vs.road_id is None or vs.sumo_lane_id is None:
                    raise ValueError(
                        "MetricsExtractor: a p4 delay evaluator is configured "
                        "but the branch record carries no campaign road "
                        "identity (road_id/sumo_lane_id) -- the p4 delay "
                        "cannot be computed honestly (fail closed)"
                    )
                vehicles[vid] = {
                    "road_id": vs.road_id,
                    "lane_id": vs.sumo_lane_id,
                    "lane_position_m": vs.s_m,
                    "speed_mps": vs.speed_mps,
                    "acceleration_mps2": vs.accel_mps2,
                }
            rows.append(
                {
                    "time_s": step.time_s,
                    "vehicles": vehicles,
                    "colliding_ids": list(step.colliding_ids),
                    "cav_ids": list(record.cav_ids),
                }
            )
        return rows

    @staticmethod
    def _steps_in(record: BranchRecord, from_off_s: float, to_off_s: float) -> list:
        """Reported steps with t0-offset in [from, to] when from == 0 (the
        closed safety window [t0, t0 + w]), or in (from, to] otherwise (the
        future window excludes t0 itself -- contract v2 section 3)."""
        if from_off_s <= 1e-9:
            lo = record.t0_s - 1e-9
        else:
            lo = record.t0_s + from_off_s + 1e-9  # exclusive at the start
        hi = record.t0_s + to_off_s + 1e-9
        return [s for s in record.steps if lo <= s.time_s <= hi]

    @staticmethod
    def _collision_in(record: BranchRecord, from_off_s: float, to_off_s: float) -> bool:
        if from_off_s <= 1e-9:
            lo = record.t0_s - 1e-9
        else:
            lo = record.t0_s + from_off_s + 1e-9
        hi = record.t0_s + to_off_s + 1e-9
        return any(lo <= c <= hi for c in record.collision_times)

    @staticmethod
    def _transit(record: BranchRecord, limit_s: float):
        on_section: dict = {}
        waits: dict = {}
        horizon_end = record.t0_s + limit_s + 1e-9
        for step in record.steps:
            if step.time_s > horizon_end:
                continue
            for vid, vs in step.vehicles.items():
                if vs.on_section:
                    on_section[vid] = on_section.get(vid, 0.0) + step.dt_s
                waits[vid] = waits.get(vid, 0.0) + vs.entry_wait_s
        return on_section, waits

    def _safety(self, window_steps):
        """(min TTC | None, exposure@3, exposure@1.5, brake seconds,
        has_closing_pair).

        A closing pair exists exactly when a finite TTC was realisable from
        the window telemetry (closing speed > 1e-6 m/s); min TTC is None
        exactly then -- NaN min TTC therefore means "no closing pair", never
        "unknown safety" (contract v1.1 section 9 / v2 section 3). T-A: the
        pair TTC is computed ONCE per vehicle-step (over the PHYSICAL gap
        the runner recorded) and accumulated at BOTH thresholds -- the
        frozen 3 s exposure and the PRIMARY 1.5 s exposure (identical pair
        arithmetic, different threshold)."""
        minimum = None
        exposure = 0.0
        exposure_lt1p5 = 0.0
        brake_seconds = 0.0
        for step in window_steps:
            accels = []
            for vs in step.vehicles.values():
                accels.append(vs.accel_mps2)
                if vs.leader_gap_m is None or vs.leader_speed_mps is None:
                    continue
                if not (
                    np.isfinite(vs.leader_gap_m)
                    and np.isfinite(vs.leader_speed_mps)
                    and np.isfinite(vs.speed_mps)
                ):
                    continue  # malformed telemetry: skip, never guess
                ttc = closing_ttc(
                    float(vs.leader_gap_m), float(vs.speed_mps), float(vs.leader_speed_mps)
                )
                if ttc is None:
                    continue
                minimum = ttc if minimum is None else min(minimum, ttc)
                if ttc < TTC_EXPOSURE_THRESHOLD_S:
                    exposure += step.dt_s
                if ttc < TTC_EXPOSURE_PRIMARY_THRESHOLD_S:
                    exposure_lt1p5 += step.dt_s
            finite = [a for a in accels if np.isfinite(a)]
            if finite:
                effort = braking_effort(finite, step.dt_s)
                brake_seconds += float(effort["braking_vehicle_seconds"])
        return minimum, exposure, exposure_lt1p5, brake_seconds, minimum is not None

    @staticmethod
    def _min_ttc_pair(window_steps, record: BranchRecord):
        """T-B (#7): the follower/leader pair realising the window's minimum
        closing TTC (the pair an absolute min-TTC veto is triggered by), or
        None when no closing pair exists. Same pair arithmetic as
        :meth:`_safety` (physical gap, closing speed > 1e-6); the class is
        ``cav_involved`` when either vehicle is in the record's CAV roster,
        else ``hdv_hdv``; a leader id the telemetry did not carry gives
        class ``unknown_leader``. Ties keep the earliest step, then the
        smallest follower id (deterministic)."""
        cavs = {str(c) for c in (record.cav_ids or ())}
        best = None
        for step in window_steps:
            for vid in sorted(step.vehicles):
                vs = step.vehicles[vid]
                if vs.leader_gap_m is None or vs.leader_speed_mps is None:
                    continue
                if not (
                    np.isfinite(vs.leader_gap_m)
                    and np.isfinite(vs.leader_speed_mps)
                    and np.isfinite(vs.speed_mps)
                ):
                    continue
                ttc = closing_ttc(
                    float(vs.leader_gap_m), float(vs.speed_mps), float(vs.leader_speed_mps)
                )
                if ttc is None:
                    continue
                if best is None or ttc < best["ttc_s"]:
                    leader = vs.leader_id
                    if leader is None:
                        klass = "unknown_leader"
                    elif str(vid) in cavs or str(leader) in cavs:
                        klass = "cav_involved"
                    else:
                        klass = "hdv_hdv"
                    best = {
                        "follower_id": str(vid),
                        "leader_id": None if leader is None else str(leader),
                        "ttc_s": float(ttc),
                        "time_s": float(step.time_s),
                        "pair_class": klass,
                    }
        return best

    @staticmethod
    def _trajectories(record: BranchRecord, trajectory_s: float):
        n_steps = int(round(float(trajectory_s))) + 1  # includes t0
        by_row = {}
        for step in record.steps:
            row = int(round(step.time_s - record.t0_s))
            if 0 <= row < n_steps:
                by_row[row] = step
        out = {}
        for vid in record.tracked:
            arr = np.full((n_steps, 4), np.nan)
            for row, step in by_row.items():
                vs = step.vehicles.get(vid)
                if vs is None or not vs.on_section:
                    continue  # NaN before insertion and from the exit on
                arr[row] = (step.time_s - record.t0_s, vs.s_m, vs.lane_id, vs.speed_mps)
            out[vid] = arr
        return out


# ------------------------------------------------------------------- model
class P4DelayEvaluator:
    """The branch delay evaluator: the recorded-run p4 delay (T-A #1).

    Wraps :func:`traffic_shepherd.planner.recorded.p4_delay` -- the SAME
    function, geometry (WM-30 over the frozen assets network) and §8.14
    entry-wait accounting the PL-EXP1b evaluation calls on recorded runs --
    over a branch's campaign-schema physics rows. Constructed with the
    scenario's scheduled departures (``recorded._read_departs`` over the
    scenario route files); ``__call__`` receives (rows, t0_s, window_s,
    t0_present_ids) from :meth:`MetricsExtractor.extract` -- the branch's
    own t0-present ids (vehicles live at the restore, zero entry wait,
    §8.14) and returns the delay vehicle-seconds over [t0, t0 + window_s].
    The p4 library import is lazy (zero
    simulation / zero heavy import at module load) and raises the recorded
    module's :class:`~traffic_shepherd.planner.recorded.P4DelayUnavailable`
    when the WD-01 library is absent -- fail closed, never a silent legacy
    delay."""

    def __init__(
        self,
        *,
        departs: Mapping[str, float],
        layout=None,
        core=None,
        geom=None,
        lane_chains=None,
        search_paths=None,
        strict: bool = True,
    ):
        from . import recorded as _recorded  # noqa: PLC0415 (lazy; see class doc)

        self._recorded = _recorded
        self.departs = {str(vid): float(d) for vid, d in dict(departs).items()}
        self.layout = layout if layout is not None else _recorded.Layout()
        self.strict = bool(strict)
        if core is not None and geom is not None:
            self.core = core
            self.geom = geom
            self.lane_chains = lane_chains
        else:
            paths = search_paths if search_paths is not None else _recorded.P4_SEARCH_PATHS
            core_loaded, geometry_loaded = _recorded.load_p4_delay(paths)
            self.core = core_loaded
            self.geom = (
                geom if geom is not None
                else geometry_loaded.wm30_geometry(net_path=_recorded._p4_net_path())
            )
            self.lane_chains = (
                lane_chains
                if lane_chains is not None
                else getattr(geometry_loaded, "WM30_CHAINS", None)
            )

    def __call__(self, rows, t0_s: float, window_s: float, t0_present_ids=frozenset()) -> float:
        delay, _waits, _n_vehicles, _diag = self._recorded.p4_delay(
            rows,
            t0_s,
            window_s,
            self.layout,
            self.departs,
            self.core,
            self.geom,
            strict=self.strict,
            t0_present_ids=frozenset(str(v) for v in (t0_present_ids or ())),
            lane_chains=self.lane_chains,
        )
        return float(delay)


class PerfectSimWorldModel:
    """``wd01_predict/1`` oracle: branch the true state, simulate each plan.

    M = 1 always; K = len(state_samples) arrival draws with CRN across plans
    and native (contract section 1). All collaborators are injected; this
    class itself never imports or launches SUMO.
    """

    def __init__(
        self,
        snapshotter: StateSnapshotter,
        arrival_draws: ArrivalDraws,
        branch_runner: BranchRunner,
        *,
        metrics_extractor: Optional[MetricsExtractor] = None,
        part_a_horizon_s: float = 60.0,
        contract_version: str = CONTRACT_VERSION,
        model_id: str = "perfect_sim/1",
        sumo_version: str = "1.27.1",
        network_sha256: str = "unknown",
        screen_horizon_s: Optional[float] = None,
        rescore_top_n: int = 5,
        rescore_always: Sequence[int] = (0,),
        stop_on_window_collision: bool = True,
        scenario_route_files: Optional[Sequence] = None,
        clock=time.monotonic,
    ):
        if part_a_horizon_s <= 0:
            raise ValueError("part_a_horizon_s must be positive")
        if contract_version not in (
            CONTRACT_VERSION,
            CONTRACT_VERSION_V1_1,
            CONTRACT_VERSION_V2,
        ):
            raise ValueError(
                f"unsupported contract_version {contract_version!r}; expected "
                f"'{CONTRACT_VERSION}', '{CONTRACT_VERSION_V1_1}' or "
                f"'{CONTRACT_VERSION_V2}'"
            )
        self.snapshotter = snapshotter
        self.arrival_draws = arrival_draws
        self.branch_runner = branch_runner
        self.metrics = metrics_extractor if metrics_extractor is not None else MetricsExtractor()
        self.part_a_horizon_s = float(part_a_horizon_s)
        self.contract_version = contract_version
        self.model_id = model_id
        self.sumo_version = str(sumo_version)
        self.network_sha256 = str(network_sha256)
        self.screen_horizon_s = None if screen_horizon_s is None else float(screen_horizon_s)
        self.rescore_top_n = int(rescore_top_n)
        self.rescore_always = tuple(int(i) for i in rescore_always)
        self.stop_on_window_collision = bool(stop_on_window_collision)
        if scenario_route_files is None:
            self.scenario_route_files = None
        else:
            entries = (
                [scenario_route_files]
                if isinstance(scenario_route_files, (str, Path))
                else list(scenario_route_files)
            )
            if not entries:
                raise ValueError("scenario_route_files must be non-empty when given")
            self.scenario_route_files = tuple(str(p) for p in entries)
        self._clock = clock

    # ------------------------------------------------------------------ api
    def predict_many(
        self,
        state_samples: Sequence,
        plans: Sequence,
        controller_state: Optional[Mapping[str, Mapping]] = None,
        *,
        horizon_s: float = 600.0,
        commit_s: float = 30.0,
        continuation: str = "planner",
        trajectory_s: Optional[float] = None,
    ) -> "list[Prediction]":
        """Branch the true state; simulate every plan (contract v1/1.1/v2).

        v2 (``contract_version='wd01_predict/2'``): the plan is the action
        for the commit window [t0, t0+commit_s]; afterwards plan and native
        continue under ``continuation`` -- 'native' (every CAV native;
        fully implemented) or 'planner' (closed-loop pi_k: PL-04, not built
        -- raises NotImplementedError). The safety window IS the commit
        window (``meta.safety_window_s == commit_s``); the future_* heads
        are the realised branch values over (t0+commit_s, t0+horizon_s].
        v1/v1.1 ignore commit_s/continuation (a v1 plan is constant from
        t0; the safety window is part_a_horizon_s)."""
        started = self._clock()
        samples = list(state_samples)
        plans = list(plans)
        if not samples:
            raise ValueError("state_samples must be non-empty")
        if continuation not in CONTINUATIONS:
            raise ValueError(f"continuation must be one of {CONTINUATIONS}, got {continuation!r}")
        if getattr(self.branch_runner, "process_factory", None) is None:
            # PL-10d before anything else: without a launcher there is no
            # legal way to run a branch (fresh process per branch is the
            # only path), so refuse before the snapshot and any restore.
            raise NotImplementedError(
                "PerfectSimWorldModel: the branch runner has no process "
                "factory -- PL-10d requires a FRESH SUMO process per "
                "restored branch. Inject BranchRunner(process_factory=...) "
                "(start_process() -> connection, close_process(connection) "
                "after the branch); refusing to restore into a reused "
                "process."
            )
        t0 = self._validate_samples(samples)
        is_v2 = self.contract_version == CONTRACT_VERSION_V2
        if is_v2:
            if continuation == "planner":
                # contract v2 section 6: pi_k closed-loop is PL-04; never faked
                raise NotImplementedError(_PLANNER_CONTINUATION_MESSAGE)
            commit = float(commit_s)
            if commit <= 0:
                raise ValueError("commit_s must be positive")
            if commit > self.part_a_horizon_s + 1e-9:
                raise ValueError(
                    f"commit_s {commit} must not exceed part_a_horizon_s "
                    f"{self.part_a_horizon_s} (contract v2 section 4)"
                )
            window = commit
            used_traj = commit if trajectory_s is None else float(trajectory_s)
        else:
            commit = None
            window = self.part_a_horizon_s
            used_traj = self.part_a_horizon_s if trajectory_s is None else float(trajectory_s)
        if used_traj > self.part_a_horizon_s + 1e-9:
            raise ValueError(f"trajectory_s {used_traj} > part_a_horizon_s {self.part_a_horizon_s}")
        horizon = float(horizon_s)
        if horizon <= 0:
            raise ValueError("horizon_s must be positive")
        if horizon < window - 1e-9:
            raise ValueError(f"horizon_s {horizon} must cover the safety window ({window})")
        screening = self.screen_horizon_s is not None and bool(plans)
        if screening:
            self._validate_screening(horizon, window)
        kept = sorted(self._kept_wall_ids(controller_state))
        self._require_scenario_exactness(samples)
        manifest = self.snapshotter.save(
            t0_s=t0, controller_state=controller_state, kept_wall_ids=kept
        )
        runner_commit = commit if is_v2 else None
        runner_continuation = continuation if is_v2 else None

        def run_stage(stage_horizon: float, indices) -> dict:
            records = {}
            branches = [(NATIVE_PLAN_LABEL, NATIVE_PLAN)] + [
                (f"plan{i}", plans[i]) for i in indices
            ]
            for label, plan in branches:
                for k, sample in enumerate(samples):
                    # T-A (#16): the native reference branch NEVER
                    # early-stops on an in-window collision -- its truncated
                    # metrics would break the paired comparison and make the
                    # PL-06 native-collision option A inoperative. Plan
                    # branches keep the model-level flag (an in-window
                    # collision vetoes them either way).
                    records[(label, k)] = self._run_branch_in_fresh_process(
                        plan,
                        process_label=f"{label}:k{k}",
                        manifest=manifest,
                        draw_id=self.arrival_draws.draw_id(sample.arrivals),
                        departures=self.arrival_draws.departures(sample.arrivals),
                        horizon_s=stage_horizon,
                        safety_window_s=window,
                        controller_state=controller_state,
                        kept_wall_ids=kept,
                        stop_on_window_collision=(
                            self.stop_on_window_collision
                            and getattr(plan, "label", None) != NATIVE_PLAN_LABEL
                        ),
                        commit_s=runner_commit,
                        continuation=runner_continuation,
                    )
            return records

        records = run_stage(self.screen_horizon_s if screening else horizon, range(len(plans)))
        rescored: "set[int]" = set(range(len(plans)))
        evaluated = {
            i: (horizon if not screening else min(self.screen_horizon_s, horizon))
            for i in range(len(plans))
        }
        if screening:
            rescored = self._screen_selection(records, len(plans), len(samples))
            full = run_stage(horizon, sorted(rescored))
            rerun = [NATIVE_PLAN_LABEL] + [f"plan{i}" for i in sorted(rescored)]
            for label in rerun:  # top plans AND the native reference
                for k in range(len(samples)):
                    records[(label, k)] = full[(label, k)]
            for i in sorted(rescored):
                evaluated[i] = horizon

        extract_kw = dict(part_a_horizon_s=window, trajectory_s=used_traj)
        if is_v2:
            extract_kw["future_horizon_s"] = horizon
        native = [
            self.metrics.extract(records[(NATIVE_PLAN_LABEL, k)], **extract_kw)
            for k in range(len(samples))
        ]
        k_count = len(samples)
        native_arrays = self._native_arrays(native, records, k_count, window, horizon, is_v2)
        wall_clock = self._clock() - started
        return [
            self._prediction(
                i,
                records,
                native_arrays,
                samples,
                used_traj,
                horizon,
                window,
                kept,
                evaluated[i],
                rescored=(screening and i not in rescored),
                wall_clock_s=wall_clock,
                commit=commit,
                continuation=continuation if is_v2 else None,
                plan=plans[i],
            )
            for i in range(len(plans))
        ]

    # -------------------------------------------------------------- plumbing
    def _run_branch_in_fresh_process(self, plan, *, process_label: str, **run_kw) -> BranchRecord:
        """PL-10d: ONE fresh SUMO process per (plan, draw) branch.

        ``start_process()`` hands out the branch's own connection and
        ``close_process(connection)`` runs in a finally clause, so a branch
        that raises still retires its process before the exception
        propagates (fail closed -- the prediction itself fails)."""
        connection = self.branch_runner.start_process(label=process_label)
        try:
            return self.branch_runner.run(plan, connection=connection, **run_kw)
        finally:
            self.branch_runner.close_process(connection)

    def _require_scenario_exactness(self, samples) -> None:
        """PL-10c guard, enforced BEFORE the snapshot and any branch: the
        scenario must carry no post-t0 randomness unless EVERY sample
        installs a non-empty explicit option-A departure list (the one
        exception; the K=1 restored-RNG mode is NOT an exception). With no
        scenario configured, a sample without explicit departures is itself
        a refusal -- exactness must never be silently assumed."""
        explicit_installed = all(tuple(self.arrival_draws.departures(s.arrivals)) for s in samples)
        if self.scenario_route_files is None:
            if explicit_installed:
                return
            raise ScenarioNotExactError(
                "PerfectSimWorldModel: no scenario_route_files configured and "
                "not every sample installs explicit arrival draws; the K=1 "
                "restored-RNG mode is NOT an exception, so post-t0 departure "
                "determinism cannot be proven -- refusing (declare the "
                "scenario or install option-A draws)"
            )
        check_scenario_exactness(
            self.scenario_route_files,
            explicit_arrivals_installed=explicit_installed,
        )

    @staticmethod
    def _sample_meta(sample) -> Mapping:
        """The sample's meta as a mapping (kit ``SampleMeta`` dataclasses are
        converted; a mapping passes through)."""
        meta = getattr(sample, "meta", None)
        if dataclasses.is_dataclass(meta) and not isinstance(meta, type):
            return dataclasses.asdict(meta)  # kit SampleMeta
        if not isinstance(meta, Mapping):
            raise ValueError("each StateSample needs a meta mapping")
        return meta

    def _validate_samples(self, samples) -> float:
        times = set()
        for sample in samples:
            meta = self._sample_meta(sample)
            missing = [f for f in ("sample_id", "time_s") if f not in meta]
            if missing:
                raise ValueError(f"sample meta missing fields: {missing}")
            times.add(float(meta["time_s"]))
        if len(times) > 1:
            raise ValueError("all K samples must share the same time_s (t0)")
        return times.pop()

    def _validate_screening(self, horizon: float, window: float) -> None:
        if not window - 1e-9 <= self.screen_horizon_s < horizon:
            raise ValueError(
                f"screen_horizon_s {self.screen_horizon_s} must cover the "
                f"safety window (>= {window}) and be shorter "
                f"than horizon_s {horizon}"
            )
        if self.rescore_top_n < 1:
            raise ValueError("rescore_top_n must be >= 1")

    # ------------------------------------------------------- v1.1/v2 heads
    def _window_valid(self, record: BranchRecord, window: float, *, screened_out: bool) -> bool:
        """Window-cell validity (fail closed): a run error, an early stop, a
        screened-out plan (no full evaluation exists) or a teleport inside
        [t0, t0+window] invalidates the window cell (contract v2 section 3)."""
        if record.stopped_early or record.run_error or screened_out:
            return False
        return not any(
            record.t0_s - 1e-9 <= tp <= record.t0_s + window + 1e-9 for tp in record.teleport_times
        )

    def _future_valid(
        self, record: BranchRecord, window: float, horizon: float, *, screened_out: bool
    ) -> bool:
        """Future-cell validity (fail closed): the branch must have run to
        the horizon without error/early stop/screening and teleported
        nowhere inside (t0+window, t0+horizon]."""
        if record.stopped_early or record.run_error or screened_out:
            return False
        if record.horizon_s < horizon - 1e-9:
            return False
        return not any(
            record.t0_s + window + 1e-9 < tp <= record.t0_s + horizon + 1e-9
            for tp in record.teleport_times
        )

    def _native_arrays(
        self, native: list, records: dict, k_count: int, window: float, horizon: float, is_v2: bool
    ) -> dict:
        """Shared native reference arrays (computed ONCE per call)."""
        diagnostics = self.contract_version in (CONTRACT_VERSION_V1_1, CONTRACT_VERSION_V2)
        native_stopped = [records[(NATIVE_PLAN_LABEL, k)].stopped_early for k in range(k_count)]
        arrays = {
            "delay_s": np.array(
                [[np.nan if native_stopped[k] else native[k]["delay_s"] for k in range(k_count)]]
            ),
            "collision": np.array([[m["collision"] for m in native]]),
            "min_ttc_s": np.array(
                [[np.nan if m["min_ttc_s"] is None else m["min_ttc_s"] for m in native]]
            ),
            "min_ttc_pairs": [m.get("min_ttc_pair") for m in native],
            "ttc_lt3_exposure_s": np.array([[m["ttc_lt3_exposure_s"] for m in native]]),
            "ttc_lt1p5_exposure_s": np.array([[m["ttc_lt1p5_exposure_s"] for m in native]]),
            "hard_brake_s": np.array([[m["hard_brake_s"] for m in native]]),
            "trajectories": _native_trajectories(native),
        }
        if diagnostics:
            arrays["cell_valid"] = np.array(
                [
                    [
                        self._window_valid(
                            records[(NATIVE_PLAN_LABEL, k)], window, screened_out=False
                        )
                        for k in range(k_count)
                    ]
                ],
                dtype=bool,
            )
            arrays["has_closing_pair"] = np.array(
                [[bool(m["has_closing_pair"]) for m in native]], dtype=bool
            )
        if is_v2:
            future_valid = np.array(
                [
                    [
                        self._future_valid(
                            records[(NATIVE_PLAN_LABEL, k)], window, horizon, screened_out=False
                        )
                        for k in range(k_count)
                    ]
                ],
                dtype=bool,
            )
            arrays.update(
                self._realised_future_block(
                    native,
                    future_valid,
                    exposure_keys=(
                        "future_ttc_lt3_exposure_s",
                        "future_ttc_lt1p5_exposure_s",
                    ),
                    brake_key="future_hard_brake_s",
                )
            )
            arrays["future_cell_valid"] = future_valid
        return arrays

    @staticmethod
    def _realised_future_block(
        extracted: list, valid: np.ndarray, *, exposure_keys, brake_key: str
    ) -> dict:
        """Realised future heads for one set of per-k extractions.

        In a VALID future cell every float is finite except
        future_min_ttc_q05_s, which is NaN exactly where no closing pair
        exists (contract v2 section 3); an INVALID cell carries NaN
        everywhere with the pair flag False (nothing was realised there).
        ``exposure_keys`` names one array built per exposure field (T-A:
        the frozen 3 s field AND the primary 1.5 s twin)."""
        k_count = len(extracted)
        exposure = {key: np.full((1, k_count), np.nan) for key in exposure_keys}
        brake = np.full((1, k_count), np.nan)
        p_collision = np.full((1, k_count), np.nan)
        q05 = np.full((1, k_count), np.nan)
        pair = np.zeros((1, k_count), dtype=bool)
        for k in range(k_count):
            if not valid[0, k]:
                continue
            m = extracted[k]
            for key in exposure_keys:
                exposure[key][0, k] = float(m[key])
            brake[0, k] = float(m[brake_key])
            p_collision[0, k] = 1.0 if m["future_collision"] else 0.0
            q05[0, k] = np.nan if m["future_min_ttc_s"] is None else float(m["future_min_ttc_s"])
            pair[0, k] = bool(m["future_has_closing_pair"])
        out = {key: exposure[key] for key in exposure_keys}
        out.update(
            {
                brake_key: brake,
                "p_future_collision": p_collision,
                "future_min_ttc_q05_s": q05,
                "future_has_closing_pair": pair,
            }
        )
        return out

    @staticmethod
    def _kept_wall_ids(controller_state) -> "list[str]":
        """N1: ASSEMBLE walls continue (locked); HOLD/RELEASE walls dissolve."""
        kept = set()
        for cav, entry in sorted((controller_state or {}).items()):
            if (
                entry.get("locked")
                and entry.get("wall_phase") == "assemble"
                and entry.get("wall_id")
            ):
                kept.add(str(entry["wall_id"]))
        return sorted(kept)

    def _screen_selection(self, records: dict, n_plans: int, k_count: int) -> "set[int]":
        """Rank plans by median screen-horizon delay_vs_native over K cells.

        Returns the rescore set: top ``rescore_top_n`` plans by that ranking
        plus ``rescore_always`` indices (the caller's current plan), all
        clamped to the plan range. Early-stopped cells never rank (the plan
        is vetoed regardless of its screen delay).
        """
        ranked = sorted((self._screen_median(records, i, k_count), i) for i in range(n_plans))
        chosen = {i for _, i in ranked[: self.rescore_top_n]}
        chosen.update(i for i in self.rescore_always if 0 <= i < n_plans)
        return chosen

    def _screen_median(self, records: dict, plan_index: int, k_count: int) -> float:
        cells = []
        for k in range(k_count):
            plan_rec = records[(f"plan{plan_index}", k)]
            native_rec = records[(NATIVE_PLAN_LABEL, k)]
            if plan_rec.stopped_early or native_rec.stopped_early:
                continue
            plan = self.metrics.extract(
                plan_rec, part_a_horizon_s=self.part_a_horizon_s, trajectory_s=0.0
            )["delay_s"]
            native = self.metrics.extract(
                native_rec, part_a_horizon_s=self.part_a_horizon_s, trajectory_s=0.0
            )["delay_s"]
            cells.append(plan - native)
        if not cells:
            return float("inf")  # unsafe everywhere: never worth rescoring
        return float(np.median(cells))

    def _prediction(
        self,
        index: int,
        records: dict,
        native_arrays: dict,
        samples,
        used_traj: float,
        horizon: float,
        window: float,
        kept,
        evaluated: float,
        *,
        rescored: bool,
        wall_clock_s: float,
        commit: Optional[float],
        continuation: Optional[str],
        plan=None,
    ) -> Prediction:
        label = f"plan{index}"
        k_count = len(samples)
        diagnostics = self.contract_version in (CONTRACT_VERSION_V1_1, CONTRACT_VERSION_V2)
        is_v2 = self.contract_version == CONTRACT_VERSION_V2
        plan_delay = np.empty((1, k_count))
        dvn = np.empty((1, k_count))
        part_a = np.empty((1, k_count))
        part_b = np.empty((1, k_count))
        collision = np.zeros((1, k_count), dtype=bool)
        min_ttc = np.empty((1, k_count))
        exposure = np.empty((1, k_count))
        exposure_lt1p5 = np.empty((1, k_count))
        hard_brake = np.empty((1, k_count))
        early_cells: "list[int]" = []
        min_ttc_pairs: list = []
        trajectories: dict = {}
        cell_valid = np.ones((1, k_count), dtype=bool) if diagnostics else None
        has_pair = np.zeros((1, k_count), dtype=bool) if diagnostics else None
        future_valid = np.zeros((1, k_count), dtype=bool) if is_v2 else None
        future_block = (
            {
                "future_ttc_lt3_exposure_s": np.full((1, k_count), np.nan),
                "future_ttc_lt1p5_exposure_s": np.full((1, k_count), np.nan),
                "future_hard_brake_s": np.full((1, k_count), np.nan),
                "p_future_collision": np.full((1, k_count), np.nan),
                "future_min_ttc_q05_s": np.full((1, k_count), np.nan),
                "future_has_closing_pair": np.zeros((1, k_count), dtype=bool),
            }
            if is_v2
            else {}
        )
        for k in range(k_count):
            record = records[(label, k)]
            extracted = self.metrics.extract(
                record,
                part_a_horizon_s=window,
                trajectory_s=used_traj,
                **({"future_horizon_s": horizon} if is_v2 else {}),
            )
            for vid, arr in extracted["trajectories"].items():
                block = trajectories.setdefault(vid, np.full((1, k_count) + arr.shape, np.nan))
                block[0, k] = arr
            # delay is NaN for an early-stopped branch (unsafe in the safety
            # window; vetoed either way), for a run-error branch and for a
            # screened-out plan (only the screen-horizon stage ran; the
            # full-horizon delay does not exist). Safety and trajectories
            # stay valid for early-stopped and screened-out branches.
            delay_valid = not record.stopped_early and not rescored and not record.run_error
            if not delay_valid:
                early_cells.extend([k] if record.stopped_early else [])
                plan_delay[0, k] = np.nan
                dvn[0, k] = np.nan
                part_a[0, k] = np.nan
                part_b[0, k] = np.nan
            else:
                plan_delay[0, k] = extracted["delay_s"]
                part_a[0, k] = extracted["delay_part_a_s"]
                part_b[0, k] = extracted["delay_part_b_s"]
                native_delay = native_arrays["delay_s"][0, k]
                dvn[0, k] = (
                    np.nan if np.isnan(native_delay) else extracted["delay_s"] - native_delay
                )
            collision[0, k] = extracted["collision"]
            min_ttc[0, k] = np.nan if extracted["min_ttc_s"] is None else extracted["min_ttc_s"]
            min_ttc_pairs.append(extracted.get("min_ttc_pair"))
            exposure[0, k] = extracted["ttc_lt3_exposure_s"]
            exposure_lt1p5[0, k] = extracted["ttc_lt1p5_exposure_s"]
            hard_brake[0, k] = extracted["hard_brake_s"]
            if diagnostics:
                # v1.1/v2: NaN min TTC means "no closing pair", never a
                # failure -- the realised pair flag carries the distinction
                cell_valid[0, k] = self._window_valid(record, window, screened_out=rescored)
                has_pair[0, k] = bool(extracted["has_closing_pair"])
            if is_v2:
                future_valid[0, k] = self._future_valid(
                    record, window, horizon, screened_out=rescored
                )
                if future_valid[0, k]:
                    future_block["future_ttc_lt3_exposure_s"][0, k] = extracted[
                        "future_ttc_lt3_exposure_s"
                    ]
                    future_block["future_ttc_lt1p5_exposure_s"][0, k] = extracted[
                        "future_ttc_lt1p5_exposure_s"
                    ]
                    future_block["future_hard_brake_s"][0, k] = extracted["future_hard_brake_s"]
                    future_block["p_future_collision"][0, k] = (
                        1.0 if extracted["future_collision"] else 0.0
                    )
                    future_block["future_min_ttc_q05_s"][0, k] = (
                        np.nan
                        if extracted["future_min_ttc_s"] is None
                        else extracted["future_min_ttc_s"]
                    )
                    future_block["future_has_closing_pair"][0, k] = bool(
                        extracted["future_has_closing_pair"]
                    )
        walls = (
            self._wall_arrays_v2(label, records, k_count, plan, future_valid)
            if is_v2
            else self._wall_arrays(label, records, k_count)
        )
        meta = {
            "contract_version": self.contract_version,
            "model_id": self.model_id,
            "weights_sha256": [f"simulator:sumo-{self.sumo_version}+{self.network_sha256}"],
            "sample_ids": [self._sample_meta(s)["sample_id"] for s in samples],
            "member_ids": ["m0"],
            "part_a_horizon_s": self.part_a_horizon_s,
            "safety_window_s": float(window),
            "horizon_s": float(horizon),
            "native_kept_walls": list(kept),
            "native_unlocked_available": False,
            "device": "sumo-traci",
            "wall_clock_s": wall_clock_s,
            "early_stop_cells": early_cells,
            "screened_out": bool(rescored),
            "evaluated_horizon_s": float(evaluated),
            # contract v2.1 section 3: the perfect simulator bills every
            # vehicle of the corridor (t0-present + arrivals) in BOTH parts,
            # so its delay_part_a_s is over all vehicles, not the tracked
            # subset -- it declares 'all_vehicles'.
            "part_a_scope": "all_vehicles",
            # T-B (#7): per-cell triggering pair of the window's min closing
            # TTC (None where no closing pair) for the plan and its native
            # reference -- LOGGING ONLY, no rule reads it
            "min_ttc_pairs": list(min_ttc_pairs),
            "native_min_ttc_pairs": list(native_arrays.get("min_ttc_pairs") or []),
        }
        if is_v2:
            # contract v2 section 4 meta; the simulator is ground truth, so
            # both validation flags are True with their own ids (sections 4,
            # 6 and 8: G-1/G-2 accept this prediction in planner live mode)
            meta.update(
                {
                    "commit_s": float(commit),
                    "continuation": str(continuation),
                    "continuation_policy_id": None,  # continuation == "native"
                    "value_model_id": VALUE_MODEL_ID,
                    "value_model_sha256": [
                        f"simulator:sumo-{self.sumo_version}+{self.network_sha256}"
                    ],
                    "future_min_ttc_validated": True,
                    "future_min_ttc_validation_id": FUTURE_MIN_TTC_VALIDATION_ID,
                    "relative_safety_validated": True,
                    "relative_safety_validation_id": RELATIVE_SAFETY_VALIDATION_ID,
                }
            )
        optional: dict = {}
        # T-A: the PRIMARY 1.5 s exposure and its native twin on EVERY
        # contract version (optional additions; the frozen 3 s fields are
        # unchanged)
        optional.update(
            {
                "ttc_lt1p5_exposure_s": exposure_lt1p5,
                "native_ttc_lt1p5_exposure_s": native_arrays["ttc_lt1p5_exposure_s"],
            }
        )
        if diagnostics:
            optional.update(
                {
                    "cell_valid": cell_valid,
                    "native_cell_valid": native_arrays["cell_valid"],
                    "has_closing_pair": has_pair,
                    "native_has_closing_pair": native_arrays["has_closing_pair"],
                }
            )
        if is_v2:
            optional.update(
                {
                    "future_ttc_lt3_exposure_s": future_block["future_ttc_lt3_exposure_s"],
                    "native_future_ttc_lt3_exposure_s": native_arrays["future_ttc_lt3_exposure_s"],
                    # T-A: the 1.5 s future twins
                    "future_ttc_lt1p5_exposure_s": future_block["future_ttc_lt1p5_exposure_s"],
                    "native_future_ttc_lt1p5_exposure_s": native_arrays[
                        "future_ttc_lt1p5_exposure_s"
                    ],
                    "future_hard_brake_s": future_block["future_hard_brake_s"],
                    "native_future_hard_brake_s": native_arrays["future_hard_brake_s"],
                    "p_future_collision": future_block["p_future_collision"],
                    "native_p_future_collision": native_arrays["p_future_collision"],
                    "future_min_ttc_q05_s": future_block["future_min_ttc_q05_s"],
                    "native_future_min_ttc_q05_s": native_arrays["future_min_ttc_q05_s"],
                    "future_cell_valid": future_valid,
                    "native_future_cell_valid": native_arrays["future_cell_valid"],
                    "future_has_closing_pair": future_block["future_has_closing_pair"],
                    "native_future_has_closing_pair": native_arrays["future_has_closing_pair"],
                }
            )
        return Prediction(
            delay_vs_native_s=dvn,
            plan_delay_s=plan_delay,
            native_delay_s=native_arrays["delay_s"],
            delay_part_a_s=part_a,
            delay_part_b_s=part_b,
            collision=collision,
            min_ttc_s=min_ttc,
            ttc_lt3_exposure_s=exposure,
            native_ttc_lt3_exposure_s=native_arrays["ttc_lt3_exposure_s"],
            hard_brake_s=hard_brake,
            native_hard_brake_s=native_arrays["hard_brake_s"],
            native_collision=native_arrays["collision"],
            native_min_ttc_s=native_arrays["min_ttc_s"],
            trajectories_tracked=trajectories,
            native_trajectories_tracked=native_arrays["trajectories"],
            walls=walls,
            meta=meta,
            **optional,
        )

    def _wall_arrays(self, label: str, records: dict, k_count: int) -> "dict[str, WallPrediction]":
        blocks: "dict[str, dict]" = {}
        for k in range(k_count):
            for wall_id, event in records[(label, k)].wall_events.items():
                block = blocks.setdefault(
                    wall_id,
                    {
                        "assembled": np.zeros((1, k_count), dtype=bool),
                        "assembled_at_s": np.full((1, k_count), np.nan),
                        "safety_abort": np.zeros((1, k_count), dtype=bool),
                        "abort_reason": np.zeros((1, k_count), dtype=np.int8),
                    },
                )
                if event.assembled_at_s is not None:
                    block["assembled"][0, k] = True
                    block["assembled_at_s"][0, k] = event.assembled_at_s
                block["safety_abort"][0, k] = event.safety_abort
                block["abort_reason"][0, k] = int(event.abort_reason)
        return {wall_id: WallPrediction(**block) for wall_id, block in blocks.items()}

    def _wall_arrays_v2(
        self,
        label: str,
        records: dict,
        k_count: int,
        plan,
        future_valid: Optional[np.ndarray],
    ) -> "dict[str, WallPrediction]":
        """v2 wall blocks: the in-window v1 fields plus the realised future
        heads (contract v2 section 3). Every wall of the plan gets a block
        (a wall that never assembled reports p_assembly_completes == 0 with
        a NaN expected time); there are no native_ versions."""
        blocks: "dict[str, dict]" = {}
        wall_ids = [w.wall_id for w in getattr(plan, "walls", ()) if plan is not None]
        for k in range(k_count):
            for wall_id, event in records[(label, k)].wall_events.items():
                if wall_id not in wall_ids:
                    wall_ids.append(wall_id)
        for wall_id in wall_ids:
            block = {
                "assembled": np.zeros((1, k_count), dtype=bool),
                "assembled_at_s": np.full((1, k_count), np.nan),
                "safety_abort": np.zeros((1, k_count), dtype=bool),
                "abort_reason": np.zeros((1, k_count), dtype=np.int8),
                "p_assembly_completes": np.zeros((1, k_count), dtype=float),
                "expected_assembly_time_s": np.full((1, k_count), np.nan),
                "p_assembly_abort": np.zeros((1, k_count), dtype=float),
            }
            for k in range(k_count):
                event = records[(label, k)].wall_events.get(wall_id)
                if event is None:
                    continue
                if event.assembled_at_s is not None:
                    block["assembled"][0, k] = True
                    block["assembled_at_s"][0, k] = event.assembled_at_s
                    block["p_assembly_completes"][0, k] = 1.0
                    block["expected_assembly_time_s"][0, k] = event.assembled_at_s
                block["safety_abort"][0, k] = event.safety_abort
                block["abort_reason"][0, k] = int(event.abort_reason)
                block["p_assembly_abort"][0, k] = 1.0 if int(event.abort_reason) != 0 else 0.0
            blocks[wall_id] = WallPrediction(**block)
        return blocks


def _native_trajectories(native: list) -> dict:
    out: dict = {}
    for k, extracted in enumerate(native):
        for vid, arr in extracted["trajectories"].items():
            block = out.setdefault(vid, np.full((1, len(native)) + arr.shape, np.nan))
            block[0, k] = arr
    return out


# -------------------------------------------------------------- cost model
@dataclass(frozen=True)
class CostEstimate:
    """Branch and CPU accounting for one decision point (design Cost model)."""

    plans_p: int  # candidate plans per decision (native shared, not counted)
    branches: int
    sim_seconds: float
    cpu_hours: float
    wall_hours: float
    process_start_seconds: float = 0.0  # branches * process_start_s (PL-10d)


def estimate_cost(
    N: int,
    K: int,
    W: int,
    n_speeds: int,
    n_releases: int,
    H: float,
    r: float,
    workers: int,
    group_sizes: Sequence[int] = (),
    screen: Optional[tuple] = None,
    decision_points: int = 1,
    early_stop_fraction: float = 0.0,
    safety_window_s: Optional[float] = None,
    process_start_s: float = 0.0,
) -> CostEstimate:
    """Design-doc cost arithmetic for one decision point.

    P = 2N (individual stage; native shared) + sum(2|g| + 2) per group
    + W*speeds*releases + 1 combined-plan check. Branches B = K*(1+P), each
    of horizon H: sim_seconds = B*H. With ``screen=(screen_h, full_h,
    top_n)`` every branch first runs at screen_h and only top_n plans + the
    current plan + native rerun at full_h: sim_seconds = K*(1+P)*screen_h
    + K*(top_n + 2)*full_h. ``decision_points`` multiplies the whole
    decision (branches and sim-seconds) for cadence sweeps. With
    ``early_stop_fraction=f`` (> 0 requires ``safety_window_s``), the
    expected share f of unsafe branches stops at the safety window instead
    of running the full stage horizon: a stage of B branches at horizon h
    costs B*(h*(1-f) + min(h, safety_window_s)*f).

    PL-10d (fresh process per branch): ``process_start_s`` is the per-branch
    launch overhead in wall seconds -- every branch pays ONE fresh SUMO
    process start (launch + connect + network load), so the decision adds
    ``branches * process_start_s`` CPU-seconds on top of ``sim_seconds * r``.
    The default 0.0 is a documented TBD: no measurement exists yet; price it
    from the WM-41 pilot THROUGHPUT.json or the deferred confirmation smoke
    before relying on any total.

    CPU hours = (sim_seconds * r + branches * process_start_s) / 3600;
    wall hours = CPU hours / workers.
    """
    if N < 0 or K < 1 or W < 0 or n_speeds < 1 or n_releases < 1:
        raise ValueError("invalid cost-model inputs")
    if H <= 0 or r < 0 or workers < 1:
        raise ValueError("H must be > 0, r >= 0, workers >= 1")
    if any(int(g) < 1 for g in group_sizes):
        raise ValueError("group sizes must be >= 1")
    if decision_points < 1:
        raise ValueError("decision_points must be >= 1")
    if not 0.0 <= early_stop_fraction < 1.0:
        raise ValueError("early_stop_fraction must be in [0, 1)")
    if early_stop_fraction > 0.0 and (safety_window_s is None or float(safety_window_s) <= 0):
        raise ValueError("early_stop_fraction > 0 needs safety_window_s > 0")
    if process_start_s < 0.0:
        raise ValueError("process_start_s must be >= 0 (default 0.0 = TBD, measured later)")

    def stage_seconds(n_branches: int, horizon: float) -> float:
        if early_stop_fraction <= 0.0:
            return n_branches * horizon
        stop_h = min(horizon, float(safety_window_s))
        return n_branches * (horizon * (1.0 - early_stop_fraction) + stop_h * early_stop_fraction)

    plans_p = 2 * N + sum(2 * int(g) + 2 for g in group_sizes) + W * n_speeds * n_releases + 1
    if screen is None:
        branches = K * (1 + plans_p)
        sim_seconds = stage_seconds(branches, float(H))
    else:
        screen_h, full_h, top_n = (float(screen[0]), float(screen[1]), int(screen[2]))
        if not 0 < screen_h < full_h:
            raise ValueError("screen needs 0 < screen_h < full_h")
        if top_n < 1:
            raise ValueError("screen top_n must be >= 1")
        screen_branches = K * (1 + plans_p)
        rescore_branches = K * (top_n + 2)  # top_n + current + native
        branches = screen_branches + rescore_branches
        sim_seconds = stage_seconds(screen_branches, screen_h) + stage_seconds(
            rescore_branches, full_h
        )
    branches *= decision_points
    sim_seconds *= decision_points
    process_start_seconds = branches * float(process_start_s)
    cpu_hours = (sim_seconds * float(r) + process_start_seconds) / 3600.0
    return CostEstimate(
        plans_p=plans_p,
        branches=branches,
        sim_seconds=sim_seconds,
        cpu_hours=cpu_hours,
        wall_hours=cpu_hours / float(workers),
        process_start_seconds=process_start_seconds,
    )
