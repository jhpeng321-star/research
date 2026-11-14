"""WM-59 WALL-1: wall arms in the restored-branch driver (zero SUMO here).

One wall branch = restore the world's mode-27 base t0 saved state (ONE
fresh SUMO process, EXACTLY ONE ``loadState``), then run [t0, t0 + 600]
through the frozen WM-40 dispatcher (``MixedAssignmentSession`` +
``CommandOwnership``, both read-only) with:

- the ONLINE request rule (``campaign_walls_v1.wall_request``, law
  v2.2 'merge-release'): the arm's target time (early t0 / late t0+60),
  the feasible pair evaluated from CAV report frames only, members =
  the pair nearest upstream of the merge (one per lane), wall speed =
  factor x the members' mean reported speed (floor 2.5 m/s), the v2.2
  proposer gate D_min (the lead member's distance to the MERGE START
  must fit assembly + a minimal meaningful hold; refusals are logged
  with the ``dmin_unreachable`` reason), and ``not_feasible`` if no
  pair appears within 120 s after the target;
- members locked from assembly to handback (the dispatcher's wall
  lifecycle: ``WallGroupSession`` via ``apply_plan``/``step``); the
  registered hold draw H is counted from assembly completion and the
  dispatcher auto-releases at its expiry - and law v2.2 CAPS the hold
  at the MERGE START: the wall session fires ``merge_release`` (a
  release kind, never an abort) the moment the LEAD member's edge-aware
  cumulative coordinate ``s_road`` reaches ``MERGE_RELEASE_POINT_S``
  (= 2644.2 m, the merge start; owner-confirmed 2026-10-01), handing
  the members back to NATIVE; safety aborts (any phase) also hand
  members back to native; the driver logs the realized hold cap
  (``hold_cap`` row + branch_record fields);
- non-members NATIVE for the whole branch (the owner's pairing decision:
  the world's existing native sibling is the reference, never re-run);
- per branch ``wall_events.jsonl`` (request, dispatcher lifecycle events
  incl. ``merge_release``, per-tick member lane rows, hold cap, handback,
  not_feasible) and ``branch_record.json`` wall fields with
  outcome in {assembled_released, merge_released, aborted_in_assembly,
  aborted_in_hold, not_feasible};
- the chunk-2 data contract files unchanged: ``physics_steps.jsonl``
  (0.5 s rows), ``dispatcher_rows.jsonl`` (WM-42 DISPATCHER_ROW_FIELDS at
  0.5 s - the wall fields the dispatcher emits are PRESERVED here, unlike
  the per-CAV branch builder), ``insertions.jsonl``, ``outcome_record.json``.

A wall still active at the horizon end (neither released nor aborted) is
a contract anomaly and FAILS the branch loudly - it is never recorded as
a release.  The v2.1 A4 road-end release (``released_road_end``) is
SUPERSEDED: v2.2 always hands back at the merge start, so a member never
runs out of road while held (the wallrt eee755f road-end repair is
subsumed by the edge-aware ``s_road`` coordinate).

WM-72 parity (2026-09-30, job J4): the wall branch carries the same
driver guarantees as ``campaign_stage1_v3.branch_driver``: every LIVE
roster CAV is released with ``setSpeed(cav, -1)`` exactly once at t0,
before the first control tick (logged as ``t0_release`` rows); departures
and teleports are read after EACH 0.5 s physics substep (each row carries
its own substep's teleports; insertions are stamped with their substep
time); dispatcher rows are keyed to the CURRENT control tick; the branch
length comes from the plan (never a hard-coded default).
"""
from __future__ import annotations

from pathlib import Path

from campaign_pilot_v1.branch_types import DISPATCHER_ROW_FIELDS, schedule_digest
from campaign_pilot_v1.logging_schema import BOUNDARY_EDGES_FALLBACK
from campaign_pilot_v1.suffix_runner import build_physics_row
from campaign_stage1_v3 import branch_driver
from campaign_stage1_v3.common import append_jsonl, utc_now, write_json
from campaign_stage1_v3.speed_mode27 import (
    CAV_SPEED_MODE,
    SpeedMode27Guard,
    SpeedMode27GuardError,
)
from campaign_walls_v1.common import (
    MERGE_RELEASE_POINT_S,
    WALLS_WORKERS,
    WallsPilotError,
    arm_speed_factor,
    arm_timing,
    wall_id_for,
)
from campaign_walls_v1.wall_request import (
    build_wall_plan,
    choose_pair,
    dmin_refusal,
    edge_of_lane,
    hold_cap_s,
    s_road,
    target_and_deadline,
)

SCHEMA_BRANCH_RECORD = "wm59_wall_branch_record/1"
SCHEMA_OUTCOME_RECORD = "wm44_branch_outcome/1"  # unchanged chunk-2 contract

OUTCOME_ASSEMBLED_RELEASED = "assembled_released"
OUTCOME_MERGE_RELEASED = "merge_released"
OUTCOME_ABORTED_IN_ASSEMBLY = "aborted_in_assembly"
OUTCOME_ABORTED_IN_HOLD = "aborted_in_hold"
OUTCOME_NOT_FEASIBLE = "not_feasible"
#: SUPERSEDED by law v2.2 (kept only so archived J4 branch records still
#: replay; the v2.2 driver never issues a road-end release and never
#: records this outcome - it is deliberately OUTSIDE ``OUTCOMES``).
OUTCOME_RELEASED_ROAD_END = "released_road_end"
REASON_MEMBER_LEFT_NETWORK = "member_left_network"
OUTCOMES = frozenset({
    OUTCOME_ASSEMBLED_RELEASED,
    OUTCOME_MERGE_RELEASED,
    OUTCOME_ABORTED_IN_ASSEMBLY,
    OUTCOME_ABORTED_IN_HOLD,
    OUTCOME_NOT_FEASIBLE,
})

WALL_ACTION = "wall-v2"
PER_CAV_ACTIONS = frozenset({"native", "damping", "pi-external-u"})


class WallDriverError(WallsPilotError):
    """A wall-branch contract violation; the branch fails loudly."""


def install_load_state_guard(api) -> dict:
    """Refuse a SECOND ``loadState`` on this process's api (declared
    invariant PL-10d / v3: one fresh SUMO process per branch, exactly one
    loadState).  Installs a counting wrapper around ``api.simulation.loadState``;
    the first call passes through, any further call raises.  Returns the
    counter dict (recorded in the run receipt as ``load_state_calls``).
    """
    simulation = api.simulation
    original = simulation.loadState
    count = {"calls": 0}

    def _guarded(path):
        count["calls"] += 1
        if count["calls"] > 1:
            raise WallDriverError(
                "SECOND loadState refused: one fresh SUMO process per "
                f"branch, exactly one loadState (path={path!r})")
        return original(path)

    simulation.loadState = _guarded
    return count


def validate_wall_dispatcher_row(row: dict) -> None:
    """WM-42 field contract for WALL branches: exactly
    ``DISPATCHER_ROW_FIELDS``; wall-v2 rows are locked with a wall id and
    an assembly/hold phase; every other action carries NO wall fields
    (a non-member is never locked, never commanded by the wall)."""
    missing = [f for f in DISPATCHER_ROW_FIELDS if f not in row]
    if missing:
        raise WallDriverError(f"dispatcher row missing fields: {missing}")
    extra = set(row) - set(DISPATCHER_ROW_FIELDS)
    if extra:
        raise WallDriverError(f"dispatcher row carries unknown fields: {sorted(extra)}")
    action = row["action"]
    if action == WALL_ACTION:
        if not row["wall_id"]:
            raise WallDriverError("wall-v2 row without a wall id")
        if row["locked"] is not True:
            raise WallDriverError(f"wall-v2 row for {row['vehicle_id']} not locked")
        if row["wall_phase"] not in ("assembly", "hold"):
            raise WallDriverError(
                f"wall-v2 row with phase {row['wall_phase']!r}")
    elif action in PER_CAV_ACTIONS:
        for name in ("wall_id", "wall_phase", "time_in_phase_s", "wall_speed_mps"):
            if row[name] is not None:
                raise WallDriverError(
                    f"{name} must be null on a {action} row (non-members "
                    "are never locked)")
        if row["locked"] is not False:
            raise WallDriverError(f"locked must be False on a {action} row")
    else:
        raise WallDriverError(f"unknown dispatcher action {action!r}")
    state = row["controller_state"]
    if not isinstance(state, dict):
        raise WallDriverError("controller_state must be a dict")


def wall_dispatcher_rows_from_telemetry(telemetry_rows, timestamp_s: float,
                                        plan_id=None) -> list[dict]:
    """WM-42 rows for one 0.5 s physics tick from the dispatcher's latest
    per-tick telemetry, PRESERVING the wall fields the dispatcher emits
    (the per-CAV branch builder zeroes them; wall branches must show
    members locked during assembly/hold).  Same 0.5 s re-stamp contract
    as ``campaign_pilot_v1.branch_types.dispatcher_rows_from_telemetry``.
    """
    rows = []
    for row in sorted(telemetry_rows, key=lambda r: str(r.get("vehicle_id"))):
        out = {
            "timestamp_s": round(float(timestamp_s), 6),
            "plan_id": row.get("plan_id", plan_id),
            "vehicle_id": str(row["vehicle_id"]),
            "action": row["action"],
            "wall_id": row.get("wall_id"),
            "wall_phase": row.get("wall_phase"),
            "time_in_phase_s": row.get("time_in_phase_s"),
            "wall_speed_mps": row.get("wall_speed_mps"),
            "locked": bool(row.get("locked", False)),
            "controller_state": dict(row.get("controller_state") or {}),
        }
        validate_wall_dispatcher_row(out)
        rows.append(out)
    return rows


def member_lane_snapshot(frames, members, *, last_lanes: dict) -> dict:
    """One tick's per-member lane observation for the wall events (J4
    prep, 2026-09-30, WM-43b): each member's report frame gives its
    lane id (edge + lane index); ``lane_changed`` compares against the
    PREVIOUS logged tick (False on the first row, None when either side
    is unknown).  A member whose frame is missing this tick is logged
    with all-None lane fields - the missingness stays visible.  Reads
    ONLY the report frames the driver already captured (never truth)."""
    snapshot: dict[str, dict] = {}
    for member in members:
        frame = frames.get(member)
        edge = (edge_of_lane(getattr(frame, "lane_id", None))
                if frame is not None else None)
        lane_index = (getattr(frame, "lane_index", None)
                      if frame is not None else None)
        lane_index = int(lane_index) if lane_index is not None else None
        previous = last_lanes.get(member)
        current = ((edge, lane_index)
                   if edge is not None and lane_index is not None else None)
        if previous is None or current is None:
            lane_changed = None if current is None else False
        else:
            lane_changed = current != previous
        snapshot[member] = {
            "edge": edge,
            "lane_index": lane_index,
            "lane_id": (str(getattr(frame, "lane_id"))
                        if frame is not None
                        and getattr(frame, "lane_id", None) is not None
                        else None),
            "lane_changed": lane_changed,
        }
        if current is not None:
            last_lanes[member] = current
    return snapshot


class WallEpisodeTracker:
    """Wall lifecycle state from the dispatcher's ``wall_event`` rows.

    Phase at an abort: ``assembly`` before the ``assembled`` event,
    ``hold`` after - the classification the outcome enum records."""

    def __init__(self, wall_id: str):
        self.wall_id = wall_id
        self.members: tuple | None = None
        self.assembly_started_at: float | None = None
        self.assembled_at: float | None = None
        self.finished_at: float | None = None
        self.finished_how: str | None = None  # "released" | "safety_abort"
        self.abort_reason: str | None = None

    def observe(self, event: dict) -> None:
        if event.get("wall_id") != self.wall_id:
            raise WallDriverError(
                f"unexpected wall event for {event.get('wall_id')!r} "
                f"(this branch drives {self.wall_id!r})")
        kind = event.get("event")
        if kind == "assembly_start":
            self.assembly_started_at = float(event["timestamp_s"])
        elif kind == "assembled":
            self.assembled_at = float(event["timestamp_s"])
            members = event.get("members")
            if members:
                self.members = tuple(members)
        elif kind in ("released", "merge_release", "safety_abort"):
            # v2.2: ``merge_release`` is a RELEASE kind (native handback at
            # the merge start), never an abort - it finishes the wall just
            # like a planner release does.
            if self.finished_at is not None:
                raise WallDriverError(
                    f"wall {self.wall_id} finished twice "
                    f"({self.finished_how} then {kind})")
            self.finished_at = float(event["timestamp_s"])
            self.finished_how = kind
            if kind == "safety_abort":
                self.abort_reason = str(event.get("reason"))

    @property
    def active(self) -> bool:
        return self.assembly_started_at is not None and self.finished_at is None

    @property
    def phase_at_finish(self) -> str:
        return "hold" if self.assembled_at is not None else "assembly"

    def outcome(self) -> str:
        if self.finished_how == "released":
            return OUTCOME_ASSEMBLED_RELEASED
        if self.finished_how == "merge_release":
            return OUTCOME_MERGE_RELEASED
        if self.finished_how == "safety_abort":
            return (OUTCOME_ABORTED_IN_HOLD if self.phase_at_finish == "hold"
                    else OUTCOME_ABORTED_IN_ASSEMBLY)
        raise WallDriverError("no wall outcome before release or abort")


class _WallRequest:
    """The arm's ONLINE request state machine (waiting -> requested |
    not_feasible), evaluated from report frames only."""

    def __init__(self, *, run_name: str, arm: str, t0_s: float, hold_s: float,
                 cav_ids, plan_factory=None, min_span_m: float | None = None):
        self.run_name = run_name
        self.arm = arm
        self.factor = arm_speed_factor(arm)
        self.timing = arm_timing(arm)
        self.target_s, self.deadline_s = target_and_deadline(t0_s, self.timing)
        self.hold_s = float(hold_s)
        self.cav_ids = frozenset(cav_ids)
        self.plan_factory = plan_factory
        self.min_span_m = (None if min_span_m is None
                           else float(min_span_m))  # SMOKE-ONLY wide window
        self.wall_id = wall_id_for(run_name)
        self.state = "waiting"
        self.members: tuple | None = None
        self.lanes: list | None = None
        self.span_m: float | None = None
        self.wall_speed_mps: float | None = None
        self.requested_at: float | None = None
        self.plan_id: str | None = None
        self.pair: dict | None = None
        # v2.2 merge-release bookkeeping: the lead member's LAST observed
        # s_road while in HOLD (the frames a hold tick releases on are the
        # same ones captured here, so on a ``merge_release`` finish this is
        # the release s_road), the lead member's s_road AT ASSEMBLY (the
        # frames the ``assembled`` event fired on - the a-priori hold-cap
        # logging input), the last D_min refusal seen while waiting, and
        # the realized hold cap (written once at the finish).
        self.last_hold_lead_s_road_m: float | None = None
        self.assembled_lead_s_road_m: float | None = None
        self.last_dmin_refusal: dict | None = None
        self.hold_cap_logged = False

    def note_assembled_frames(self, frames) -> None:
        """At the ASSEMBLY tick (called from the driver's emit path when
        the dispatcher's ``assembled`` event passes through, with the
        SAME report frames that tick acted on): record the lead member's
        s_road so the ``hold_cap`` row's ``a_priori_effective_hold_s``
        is the a-priori cap FROM THE ASSEMBLY POSITION,
        ``min(H, (MERGE_RELEASE_POINT_S - s_lead_at_assembly) / v_wall)``
        - never a release-time number (v2.2 logging contract, smoke R2
        fix; report frames only, no control-path effect)."""
        if self.assembled_lead_s_road_m is not None:
            return
        roads = [s_road(frames.get(member))
                 for member in (self.members or ())]
        roads = [float(s) for s in roads if s is not None]
        if roads:
            self.assembled_lead_s_road_m = max(roads)

    def observe_hold(self, frames, assembled: bool, finished: bool) -> None:
        """Per-tick v2.2 bookkeeping while the wall is in HOLD: record the
        LEAD member's s_road (max over members, report frames only)."""
        if self.state != "requested" or not assembled or finished:
            return
        roads = [s_road(frames.get(member))
                 for member in (self.members or ())]
        roads = [float(s) for s in roads if s is not None]
        if roads:
            self.last_hold_lead_s_road_m = max(roads)

    def note_dmin_refusal(self, frames) -> None:
        """While WAITING: remember the current D_min refusal (if any) so a
        request-window ``not_feasible`` names the D_min reason when the
        proposal was refused on the merge-start gate."""
        if self.state != "waiting" or self.factor is None:
            return
        refusal = dmin_refusal(frames, self.factor)
        if refusal is not None:
            self.last_dmin_refusal = refusal

    def hold_cap_row(self, tracker) -> dict | None:
        """The realized v2.2 hold cap (logged ONCE, right after the wall's
        terminal dispatcher event): the registered draw vs the effective
        hold, and whether the merge start capped it.  The a-priori
        effective hold is computed AT ASSEMBLY from the assembly lead
        position (``hold_cap_s`` of the ASSEMBLY s_road), never from the
        release position."""
        if (self.state != "requested" or self.hold_cap_logged
                or tracker.finished_at is None
                or tracker.assembled_at is None):
            return None
        self.hold_cap_logged = True
        effective = round(float(tracker.finished_at)
                          - float(tracker.assembled_at), 6)
        capped = tracker.finished_how == "merge_release"
        a_priori_cap = (None if self.assembled_lead_s_road_m is None
                        else hold_cap_s(self.hold_s,
                                        self.assembled_lead_s_road_m,
                                        self.wall_speed_mps))
        return {
            "kind": "hold_cap",
            "wall_id": self.wall_id,
            "timestamp_s": round(float(tracker.finished_at), 6),
            "requested_hold_s": float(self.hold_s),
            "effective_hold_s": effective,
            "capped_at_merge": bool(capped),
            "cap_point_s": MERGE_RELEASE_POINT_S,
            "a_priori_effective_hold_s": a_priori_cap,
            "lead_s_road_at_assembly_m": self.assembled_lead_s_road_m,
            "lead_s_road_at_release_m": self.last_hold_lead_s_road_m,
            "wall_speed_mps": self.wall_speed_mps,
            "source": "cav_report_frames",
        }

    def maybe_request(self, session, frames, now_s: float, sink) -> None:
        """At/after the target time: request on the first feasible tick;
        record ``not_feasible`` at the first tick past the deadline (with
        the v2.2 D_min refusal reason when the proposal was refused on
        the merge-start gate rather than for want of a pair)."""
        if self.state != "waiting" or now_s < self.target_s - 1e-9:
            return
        pair = choose_pair(frames, self.factor, min_span_m=self.min_span_m)
        if pair is not None:
            plan, wall_id, speed = build_wall_plan(
                run_name=self.run_name, cav_ids=sorted(self.cav_ids),
                pair=pair, factor=self.factor, hold_s=self.hold_s,
                now_s=now_s, plan_factory=self.plan_factory)
            session.apply_plan(plan, now_s=now_s)
            self.state = "requested"
            self.pair = pair
            self.members = tuple(pair["members"])
            self.lanes = list(pair["lanes"])
            self.span_m = float(pair["span_m"])
            self.wall_speed_mps = float(speed)
            self.requested_at = float(now_s)
            self.plan_id = plan.plan_id
            mean_speed = (sum(pair["member_speeds"].values())
                          / len(pair["member_speeds"]))
            sink({
                "kind": "wall_request",
                "wall_id": wall_id,
                "timestamp_s": round(float(now_s), 6),
                "timing": self.timing,
                "target_time_s": round(self.target_s, 6),
                "deadline_s": round(self.deadline_s, 6),
                "edge": pair["edge"],
                "members": list(self.members),
                "lanes": list(self.lanes),
                "span_m": self.span_m,
                "min_span_m": self.min_span_m,
                "road_left_m": pair["road_left_m"],
                "lead_s_road_m": pair["lead_s_road_m"],
                "lead_distance_to_merge_m": pair["lead_distance_to_merge_m"],
                "dmin_m": pair["dmin_m"],
                "speed_factor": self.factor,
                "mean_member_speed_mps": round(float(mean_speed), 6),
                "wall_speed_mps": float(speed),
                "hold_s": self.hold_s,
                "source": "cav_report_frames",
            })
            return
        self.note_dmin_refusal(frames)
        if now_s > self.deadline_s + 1e-9:
            self.state = "not_feasible"
            row = {
                "kind": "not_feasible",
                "wall_id": self.wall_id,
                "timestamp_s": round(float(now_s), 6),
                "timing": self.timing,
                "target_time_s": round(self.target_s, 6),
                "deadline_s": round(self.deadline_s, 6),
                "reason": "no feasible pair within the 120 s request window",
            }
            refusal = self.last_dmin_refusal
            if refusal is not None:
                # v2.2: span-feasible pairs existed but the proposal was
                # refused - assembly + a minimal meaningful hold cannot
                # complete before the merge start.
                row["reason"] = refusal["reason"]
                row["dmin"] = {k: refusal[k] for k in (
                    "lead_distance_to_merge_m", "dmin_m",
                    "wall_speed_mps", "span_m")}
            sink(row)


def drive_wall_branch(api, world: dict, arm: dict, run_dir: Path, restore: dict,
                      *, session, ownership, sensor, plan_factory=None,
                      route_departures: dict | None = None,
                      horizon_ticks: float | None = None,
                      hold_override_s: float | None = None,
                      wide_pair_min_span_m: float | None = None,
                      post_restore_hook=None,
                      branch_length_s_override: float | None = None) -> dict:
    """Run one restored WALL branch; returns the summary for the receipt.

    Same loop discipline as the v3 per-CAV branch driver (1 s control
    ticks on the SUMO grid, 2 x 0.5 s physics substeps, teleport and
    insertion lists) with the wall request replacing the per-CAV switch
    schedule.  The api/dispatcher/ownership/sensor are injected; ZERO
    SUMO here.  ``horizon_ticks`` is the registered smoke cap only.
    ``wide_pair_min_span_m`` is a SMOKE-ONLY request window (the
    wide-pair smoke mode's declared [40, 75] m span): refused on real
    branches exactly like the smoke-only hold override.

    Per-tick MEMBER LANE LOGGING (J4 prep, 2026-09-30, WM-43b): for
    every control tick from the request to the wall's finish (the
    finish tick included - its row is the evidence), the incremental
    ``wall_events.jsonl`` carries a ``wall_member_lanes`` row with each
    member's reported edge, lane index, lane id and a ``lane_changed``
    flag vs the previous tick; a ``lane_coverage_violation`` safety
    abort is annotated with the member(s) whose lane differs from the
    requested lane assignment, so the mover is identifiable.
    """
    if session is None or ownership is None:
        raise WallDriverError(
            "the wall branch drive requires the injected dispatcher "
            "session and ownership (the frozen stack is wired at launch); "
            "tests inject fakes")
    if wide_pair_min_span_m is not None and horizon_ticks is None:
        # SMOKE ONLY (the wide-pair smoke passes it; real branches never do)
        raise WallDriverError(
            "wide_pair_min_span_m is a smoke-only option (needs "
            "horizon_ticks)")
    manifest = restore["manifest"]
    cav_ids = set(manifest["harness_state"]["roster"]["cav_ids"])
    if not cav_ids:
        raise WallDriverError("restored roster is empty")
    run_name = str(arm["run"])
    hold_s = float((arm.get("wall") or {}).get("hold_s", 0.0) or 0.0)
    if hold_override_s is not None:
        # SMOKE ONLY (the smoke passes it; real branches never do): the
        # drawn H is replaced so the release fits the capped horizon
        if horizon_ticks is None:
            raise WallDriverError(
                "hold_override_s is a smoke-only option (needs horizon_ticks)")
        hold_s = float(hold_override_s)
    if hold_s <= 0:
        raise WallDriverError(f"arm {arm['arm']!r} carries no recorded hold_s")
    t0_s = float(arm["restores_state"]["at_s"])
    if horizon_ticks is not None:
        horizon_s = t0_s + float(horizon_ticks)
    elif branch_length_s_override is not None:
        # INC1 hook patch (A1.2): optional branch horizon override
        # (the INC1 300 s arm).  Absent, the pure v2.2 plan horizon
        # path is unchanged.
        horizon_s = t0_s + float(branch_length_s_override)
    else:
        # WM-72 #12: the branch length comes FROM THE PLAN (arm
        # duration_s, else the world's branch_length_s); none refuses.
        try:
            plan_length_s = branch_driver._branch_length_s(world, arm, None)
        except branch_driver.BranchDriverError as exc:
            raise WallDriverError(str(exc)) from exc
        horizon_s = t0_s + plan_length_s
    if horizon_s <= t0_s:
        raise WallDriverError("nonpositive branch horizon")

    insertions_path = Path(run_dir) / "insertions.jsonl"
    insertions_path.touch()  # the inflow file always exists, even if empty
    insertions: list = []
    teleports_seen: set = set()
    previous_row = restore["manifest"].get("physics_row_at_t0") or None
    boundary_edges = dict(BOUNDARY_EDGES_FALLBACK)
    boundary_edges.update({
        k: str(v)
        for k, v in restore["manifest"]["harness_state"]["roster"]
        .get("boundary_edges", {})
        .items()
    })
    assignment_rows: list = []   # THIS control tick's dispatcher telemetry
    wall_events: list[dict] = []  # driver rows + dispatcher passthrough
    events_path = Path(run_dir) / "wall_events.jsonl"
    events_path.touch()    # the event log exists even if the branch fails

    def log_event(row: dict) -> None:
        """Append INCREMENTALLY (flushed per event) so a failing branch
        still leaves its event log."""
        wall_events.append(row)
        append_jsonl(events_path, row)
    tracker = WallEpisodeTracker(wall_id_for(run_name))
    request = _WallRequest(
        run_name=run_name, arm=str(arm["arm"]), t0_s=t0_s, hold_s=hold_s,
        cav_ids=cav_ids, plan_factory=plan_factory,
        min_span_m=wide_pair_min_span_m)

    arrived_members: set = set()   # members seen in physics arrived_ids
    # per-tick member lane logging (WM-43b): last observed (edge, lane)
    # per member, the pending snapshot of THIS tick (captured before the
    # dispatcher step, consumed by the abort annotation and emitted
    # after it) and the number of logged rows
    member_last_lanes: dict = {}
    lane_state: dict = {"pending": None, "last_snapshot": None}
    member_lane_rows = 0
    # the CURRENT control tick's report frames (set right before the
    # dispatcher step): the ``assembled`` event reads them to record the
    # lead member's ASSEMBLY s_road for the hold-cap logging (v2.2)
    frames_holder: dict = {"frames": None}

    def _emit(kind: str, row: dict) -> None:
        if kind == "wall_event":
            event = {"kind": "wall_event", **row}
            if event.get("event") == "assembled":
                # v2.2 logging contract: capture the lead member's s_road
                # AT ASSEMBLY (this tick's frames - the same reports the
                # dispatcher declared assembly on) for the hold-cap row.
                request.note_assembled_frames(frames_holder["frames"])
            if (event.get("event") == "safety_abort"
                    and event.get("reason") == "member_frame_missing"
                    and arrived_members
                    and arrived_members & set(request.members or ())):
                # A4: the member LEFT THE NETWORK (physics arrived_ids), it
                # did not merely lose its frame
                event["original_reason"] = event["reason"]
                event["reason"] = REASON_MEMBER_LEFT_NETWORK
                event["arrived_members"] = sorted(
                    arrived_members & set(request.members or ()))
            if (event.get("event") == "safety_abort"
                    and event.get("reason") == "lane_coverage_violation"):
                # WM-43b: name the member(s) whose lane differs from the
                # REQUESTED lane assignment, from this tick's pending
                # snapshot (the abort fires inside the same step) or the
                # last logged one - the mover is identifiable.
                source = (lane_state["pending"]
                          or lane_state["last_snapshot"] or {})
                snapshot = source.get("lanes")
                if snapshot:
                    assignment = {
                        member: (request.pair["edge"], int(lane))
                        for member, lane in zip(request.members or (),
                                                request.lanes or [])}
                    event["lane_changed_members"] = sorted(
                        member for member, observed in snapshot.items()
                        if observed.get("edge") is not None
                        and observed.get("lane_index") is not None
                        and assignment.get(member) is not None
                        and (observed["edge"], observed["lane_index"])
                        != assignment[member])
                else:
                    event["lane_changed_members"] = []
                event["lane_snapshot_s"] = (
                    lane_state["pending"] or lane_state["last_snapshot"] or {}
                ).get("timestamp_s")
            tracker.observe(event)
            log_event(event)
        elif kind == "mixed_assignment":
            assignment_rows.append(dict(row))
        # other kinds (PI evidence etc.) flow through the command path only

    bind = getattr(session, "bind_emit", None)
    if callable(bind):
        bind(_emit)
    pi_session = getattr(session, "pi_session", None)
    pi_retired: set = set()
    ownership_mode = getattr(ownership, "cav_speed_mode", None)
    guard = SpeedMode27Guard(api, cav_ids) if ownership_mode == CAV_SPEED_MODE else None
    unavailable: list = []
    rows_written = 0
    dispatcher_rows_written = 0
    # WM-72 part-2 #4: the restored state may carry the base leg's LAST
    # speed command; RELEASE every live roster CAV exactly once at t0,
    # before the first control tick, and log each release.  (Wall members
    # are commanded by the dispatcher from the first tick on; the release
    # only clears what the saved state carried.)
    t0_release: list = []
    for cav in sorted(cav_ids & set(api.vehicle.getIDList())):
        api.vehicle.setSpeed(cav, -1)
        t0_release.append(cav)
        append_jsonl(
            Path(run_dir) / "suffix_commands.jsonl",
            {
                "kind": "t0_release",
                "timestamp_s": t0_s,
                "vehicle_id": cav,
                "owner": "native",
                "issued_target_speed_mps": None,
                "fallback_reason": "t0_release",
            },
        )
    # INC1 hook patch (A1.2): optional post-restore callback, invoked
    # EXACTLY once right after the state restore and before the first
    # control tick (INC1 uses it to insert the stopped obstacle).  Its
    # return value is logged in the branch record.  Absent, the pure
    # v2.2 driver behaviour is unchanged.
    post_restore_result = None
    if post_restore_hook is not None:
        post_restore_result = post_restore_hook(api, t0_s)
    for tick in range(int(round(t0_s)), int(round(horizon_s))):
        now = float(api.simulation.getTime())
        if abs(now - tick * 1.0) > 1e-6:
            raise WallDriverError(f"time grid mismatch at t={now}")
        live = list(api.vehicle.getIDList())
        frames = branch_driver._capture_frames(
            sensor, live, cav_ids, now, run_dir, unavailable)
        assignment_rows.clear()
        request.maybe_request(
            session, frames, now, log_event)
        # v2.2: hold cap bookkeeping (the session's ``merge_release`` is
        # the enforcing handback; the driver only observes and logs it)
        request.observe_hold(frames, tracker.assembled_at is not None,
                             tracker.finished_at is not None)
        # Per-tick member lane logging (WM-43b): capture the members'
        # reported lanes BEFORE the dispatcher step, so the snapshot is
        # available to a lane_coverage_violation abort fired inside the
        # same step.  Emitted after the step when the wall is (or just
        # was) active - the finish tick's row is the abort evidence.
        lane_state["pending"] = None
        if request.state == "requested":
            phase = "hold" if tracker.assembled_at is not None else "assembly"
            lane_state["pending"] = {
                "timestamp_s": round(float(now), 6),
                "phase": phase,
                "lanes": member_lane_snapshot(
                    frames, request.members, last_lanes=member_last_lanes),
            }
        # COMMAND ISSUANCE CONTRACT (WM-44, carried): the frozen
        # CommandOwnership and the dispatcher issue every speed command
        # THEMSELVES inside session.step; the driver only logs records.
        frames_holder["frames"] = frames
        records = session.step(ownership, frames, now_s=now, live_ids=live)
        if guard is not None:
            try:
                guard.check(now)
            except SpeedMode27GuardError as exc:
                raise WallDriverError(str(exc)) from exc
        for record in records:
            append_jsonl(
                Path(run_dir) / "suffix_commands.jsonl",
                {
                    "kind": "command",
                    "timestamp_s": now,
                    "vehicle_id": record.vehicle_id,
                    "owner": record.owner,
                    "issued_target_speed_mps": record.issued_target_speed_mps,
                    "fallback_reason": record.fallback_reason,
                },
            )
        if lane_state["pending"] is not None and (
                tracker.finished_at is None
                or abs(tracker.finished_at - now) <= 1e-9):
            # the finish tick included (its row is the abort evidence);
            # ticks after the handback are never logged
            row = {"kind": "wall_member_lanes",
                   "wall_id": request.wall_id,
                   **lane_state["pending"]}
            lane_state["last_snapshot"] = row
            lane_state["pending"] = None
            member_lane_rows += 1
            log_event(row)
        if pi_session is not None:
            pi_session.flush_evidence(
                lambda payload: append_jsonl(
                    Path(run_dir) / "suffix_commands.jsonl", payload),
                finalize=False,
            )
            pi_session.commit_evidence()
        pi_live = set(live)
        for _substep in range(2):  # 0.5 s physics cadence (WD-01)
            api.simulationStep()
            after = float(api.simulation.getTime())
            if pi_session is not None:
                live_after = set(api.vehicle.getIDList())
                disappeared_after = (pi_live - live_after) & cav_ids
                arrived_after = set(api.simulation.getArrivedIDList()) & cav_ids
                pi_retired.update(disappeared_after | arrived_after)
                reappeared_after = pi_retired & live_after
                pi_retired.difference_update(reappeared_after)
                pi_session.observe_lifecycle(
                    live_ids=live_after,
                    disappeared_ids=disappeared_after,
                    id_reuse_ids=reappeared_after,
                )
                pi_live = live_after
            row = build_physics_row(
                api,
                after,
                previous_row=previous_row,
                cav_ids=cav_ids,
                cav_actions={},
                boundary_edges=boundary_edges,
                live_by_tick=None,
            )
            # WM-72 part-2 #1: teleports sampled per 0.5 s substep, so each
            # row carries its OWN substep's teleports (incl. the last one)
            starting_tp = sorted(
                str(v) for v in api.simulation.getStartingTeleportIDList())
            ending_tp = sorted(
                str(v) for v in api.simulation.getEndingTeleportIDList())
            teleports_seen.update(starting_tp)
            row["starting_teleports"] = starting_tp
            row["ending_teleports"] = ending_tp
            append_jsonl(Path(run_dir) / "physics_steps.jsonl", row)
            arrived_members.update(str(v) for v in row.get("arrived_ids") or []
                                   if str(v) in set(request.members or ()))
            rows_written += 1
            previous_row = row
            for dispatch_row in wall_dispatcher_rows_from_telemetry(
                    assignment_rows, timestamp_s=after,
                    plan_id=request.plan_id):
                append_jsonl(Path(run_dir) / "dispatcher_rows.jsonl", dispatch_row)
                dispatcher_rows_written += 1
            # WM-44 inflow list: realised insertions, stamped with THEIR
            # substep time (from the row's drained departed ids)
            for vehicle_id in sorted(str(v) for v in row.get("departed_ids") or []):
                route_info = (route_departures or {}).get(vehicle_id, {})
                speed = None
                try:
                    speed = float(api.vehicle.getSpeed(vehicle_id))
                except Exception:  # noqa: BLE001 - vehicle may leave instantly
                    speed = None
                insertion = {
                    "vehicle_id": vehicle_id,
                    "scheduled_depart_s": route_info.get("depart_s"),
                    "inserted_s": after,
                    "entrance": branch_driver.entrance_of(
                        route_info.get("route", "")),
                    "depart_speed_mps": speed,
                    "is_cav": vehicle_id in cav_ids,
                    "type_id": route_info.get("type_id"),
                }
                insertions.append(insertion)
                append_jsonl(Path(run_dir) / "insertions.jsonl", insertion)
    # ---- post-loop wall contract (loud, never misrecorded) ----
    if request.state == "waiting":
        raise WallDriverError(
            f"wall request never resolved by the horizon: target "
            f"{request.target_s:g}s deadline {request.deadline_s:g}s "
            f"horizon {horizon_s:g}s (a shorter-than-window horizon is a "
            "smoke-configuration error)")
    if tracker.active:
        raise WallDriverError(
            f"wall {tracker.wall_id} still active at the horizon end "
            f"(phase {tracker.phase_at_finish}, neither released nor "
            "aborted): the branch fails loudly instead of recording a "
            "fictitious outcome")
    if request.state == "not_feasible":
        outcome = OUTCOME_NOT_FEASIBLE
    else:
        outcome = tracker.outcome()
    if outcome not in OUTCOMES:
        raise WallDriverError(f"unclassified wall outcome {outcome!r}")
    # the realized v2.2 hold cap (logged once, after the terminal
    # dispatcher event and before the handback row)
    hold_cap = request.hold_cap_row(tracker)
    if hold_cap is not None:
        log_event(hold_cap)
    # explicit handback row after the terminal dispatcher event
    if tracker.finished_at is not None:
        log_event({
            "kind": "wall_handback",
            "wall_id": tracker.wall_id,
            "timestamp_s": round(float(tracker.finished_at), 6),
            "how": tracker.finished_how,
            "release_kind": tracker.finished_how,
            "phase": tracker.phase_at_finish,
            "reason": tracker.abort_reason,
            "capped_at_merge": bool(hold_cap and hold_cap["capped_at_merge"]),
            "members": list(tracker.members or request.members or ()),
            "handback": "native",
        })

    wall_spec = {
        "schema": "wm59_wall_spec/1",
        "arm": arm["arm"],
        "wall_id": request.wall_id,
        "members": list(request.members) if request.members else None,
        "speed_factor": request.factor,
        "wall_speed_mps": request.wall_speed_mps,
        "timing": request.timing,
        "hold_s": hold_s,
        "target_time_s": round(request.target_s, 6),
        "request_s": request.requested_at,
    }
    achieved_hold_s = (
        None if tracker.assembled_at is None or tracker.finished_at is None
        else round(float(tracker.finished_at) - float(tracker.assembled_at), 6))
    branch_record = {
        "schema": SCHEMA_BRANCH_RECORD,
        "root_id": world["world"],
        "branch_id": run_name,
        "branch_type": arm["arm"],
        "t0_s": t0_s,
        "seed": world["seed"],
        "native_sibling": arm.get("native_sibling_run"),
        "saved_state_id": restore["saved_state_id"],
        "branch_seed": arm.get("branch_seed"),
        "teleports": len(teleports_seen),
        "cav_speed_mode": ownership_mode,
        "observation_unavailable_rows": len(unavailable),
        # WM-72 part-2 #4: the t0 release evidence
        "t0_release_count": len(t0_release),
        # ---- wall fields (owner spec) ----
        "wall_id": request.wall_id,
        "members": list(request.members) if request.members else None,
        "speed_factor": request.factor,
        "wall_speed_mps": request.wall_speed_mps,
        "timing": request.timing,
        "hold_s": hold_s,
        "requested_hold_s": hold_s,
        "achieved_hold_s": achieved_hold_s,
        "merge_release_s": (tracker.finished_at
                            if tracker.finished_how == "merge_release"
                            else None),
        "hold_capped_at_merge": bool(hold_cap and hold_cap["capped_at_merge"]),
        "lead_s_road_at_release_m": request.last_hold_lead_s_road_m,
        "request_s": request.requested_at,
        "assembled_at_s": tracker.assembled_at,
        "released_at_s": (tracker.finished_at
                          if tracker.finished_how in ("released",
                                                      "merge_release")
                          else None),
        "aborted_at_s": (tracker.finished_at
                         if tracker.finished_how == "safety_abort" else None),
        "abort_reason": tracker.abort_reason,
        "outcome": outcome,
        "member_lane_rows": member_lane_rows,
        "wide_pair_min_span_m": wide_pair_min_span_m,
        "wall_spec_sha256": schedule_digest(wall_spec),
        "created_utc": utc_now(),
    }
    if post_restore_hook is not None:
        # INC1 hook patch (A1.2): log the post-restore callback's
        # return value in the branch record.
        branch_record["post_restore_hook"] = post_restore_result
    write_json(Path(run_dir) / "branch_record.json", branch_record)
    outcome_record = {
        "schema": SCHEMA_OUTCOME_RECORD,  # unchanged chunk-2 contract
        "branch_id": run_name,
        "teleports": len(teleports_seen),
        "created_utc": utc_now(),
    }
    write_json(Path(run_dir) / "outcome_record.json", outcome_record)
    return {
        "rows_written": rows_written,
        "dispatcher_rows_written": dispatcher_rows_written,
        "wall_outcome": outcome,
        "requested_hold_s": hold_s,
        "hold_override_s": hold_override_s,
        "achieved_hold_s": achieved_hold_s,
        "merge_release_s": branch_record["merge_release_s"],
        "hold_capped_at_merge": branch_record["hold_capped_at_merge"],
        "wall_id": request.wall_id,
        "members": list(request.members) if request.members else None,
        "wall_speed_mps": request.wall_speed_mps,
        "request_s": request.requested_at,
        "assembled_at_s": tracker.assembled_at,
        "released_at_s": branch_record["released_at_s"],
        "aborted_at_s": branch_record["aborted_at_s"],
        "abort_reason": tracker.abort_reason,
        "member_lane_rows": member_lane_rows,
        "wide_pair_min_span_m": wide_pair_min_span_m,
        "cav_speed_mode": ownership_mode,
        "observation_unavailable_rows": len(unavailable),
        "t0_release_count": len(t0_release),
        "speed_mode_guard_checked": guard.checked if guard else 0,
        "speed_mode_guard_violations": len(guard.violations) if guard else 0,
        "load_state_calls": 1,  # the child's guard enforces exactly one
        "horizon_s": horizon_s,
        "t0_s": t0_s,
        "workers_declared": WALLS_WORKERS,
    }
