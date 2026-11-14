"""WM-59 WALL-1: wall arms in the restored-branch driver (zero SUMO here).

One wall branch = restore the world's mode-27 base t0 saved state (ONE
fresh SUMO process, EXACTLY ONE ``loadState``), then run [t0, t0 + 600]
through the frozen WM-40 dispatcher (``MixedAssignmentSession`` +
``CommandOwnership``, both read-only) with:

- the ONLINE request rule (``campaign_walls_v1.wall_request``): the arm's
  target time (early t0 / late t0+60), the feasible pair evaluated from
  CAV report frames only, members = the pair nearest upstream of the
  merge (one per lane), wall speed = factor x the members' mean reported
  speed (floor 2.5 m/s), ``not_feasible`` if no pair appears within
  120 s after the target;
- members locked from assembly to handback (the dispatcher's wall
  lifecycle: ``WallGroupSession`` via ``apply_plan``/``step``); hold H
  (recorded, from the branch seed) counted from assembly completion, then
  the planner release hands members back to NATIVE; safety aborts (any
  phase) also hand members back to native;
- non-members NATIVE for the whole branch (the owner's pairing decision:
  the world's existing native sibling is the reference, never re-run);
- per branch ``wall_events.jsonl`` (request, dispatcher lifecycle events,
  handback, not_feasible) and ``branch_record.json`` wall fields with
  outcome in {assembled_released, aborted_in_assembly, aborted_in_hold,
  not_feasible};
- the chunk-2 data contract files unchanged: ``physics_steps.jsonl``
  (0.5 s rows), ``dispatcher_rows.jsonl`` (WM-42 DISPATCHER_ROW_FIELDS at
  0.5 s - the wall fields the dispatcher emits are PRESERVED here, unlike
  the per-CAV branch builder), ``insertions.jsonl``, ``outcome_record.json``.

A wall still active at the horizon end (neither released nor aborted) is
a contract anomaly and FAILS the branch loudly - it is never recorded as
a release.
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
    BRANCH_LENGTH_S,
    WALLS_WORKERS,
    WallsPilotError,
    arm_speed_factor,
    arm_timing,
    wall_id_for,
)
from campaign_walls_v1.wall_request import (
    build_wall_plan,
    choose_pair,
    target_and_deadline,
)

SCHEMA_BRANCH_RECORD = "wm59_wall_branch_record/1"
SCHEMA_OUTCOME_RECORD = "wm44_branch_outcome/1"  # unchanged chunk-2 contract

OUTCOME_ASSEMBLED_RELEASED = "assembled_released"
OUTCOME_ABORTED_IN_ASSEMBLY = "aborted_in_assembly"
OUTCOME_ABORTED_IN_HOLD = "aborted_in_hold"
OUTCOME_NOT_FEASIBLE = "not_feasible"
OUTCOMES = frozenset({
    OUTCOME_ASSEMBLED_RELEASED,
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
        elif kind in ("released", "safety_abort"):
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
        if self.finished_how == "safety_abort":
            return (OUTCOME_ABORTED_IN_HOLD if self.phase_at_finish == "hold"
                    else OUTCOME_ABORTED_IN_ASSEMBLY)
        raise WallDriverError("no wall outcome before release or abort")


class _WallRequest:
    """The arm's ONLINE request state machine (waiting -> requested |
    not_feasible), evaluated from report frames only."""

    def __init__(self, *, run_name: str, arm: str, t0_s: float, hold_s: float,
                 cav_ids, plan_factory=None):
        self.run_name = run_name
        self.arm = arm
        self.factor = arm_speed_factor(arm)
        self.timing = arm_timing(arm)
        self.target_s, self.deadline_s = target_and_deadline(t0_s, self.timing)
        self.hold_s = float(hold_s)
        self.cav_ids = frozenset(cav_ids)
        self.plan_factory = plan_factory
        self.wall_id = wall_id_for(run_name)
        self.state = "waiting"
        self.members: tuple | None = None
        self.lanes: list | None = None
        self.span_m: float | None = None
        self.wall_speed_mps: float | None = None
        self.requested_at: float | None = None
        self.plan_id: str | None = None

    def maybe_request(self, session, frames, now_s: float, sink) -> None:
        """At/after the target time: request on the first feasible tick;
        record ``not_feasible`` at the first tick past the deadline."""
        if self.state != "waiting" or now_s < self.target_s - 1e-9:
            return
        pair = choose_pair(frames)
        if pair is not None:
            plan, wall_id, speed = build_wall_plan(
                run_name=self.run_name, cav_ids=sorted(self.cav_ids),
                pair=pair, factor=self.factor, hold_s=self.hold_s,
                now_s=now_s, plan_factory=self.plan_factory)
            session.apply_plan(plan, now_s=now_s)
            self.state = "requested"
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
                "speed_factor": self.factor,
                "mean_member_speed_mps": round(float(mean_speed), 6),
                "wall_speed_mps": float(speed),
                "hold_s": self.hold_s,
                "source": "cav_report_frames",
            })
            return
        if now_s > self.deadline_s + 1e-9:
            self.state = "not_feasible"
            sink({
                "kind": "not_feasible",
                "wall_id": self.wall_id,
                "timestamp_s": round(float(now_s), 6),
                "timing": self.timing,
                "target_time_s": round(self.target_s, 6),
                "deadline_s": round(self.deadline_s, 6),
                "reason": "no feasible pair within the 120 s request window",
            })


def drive_wall_branch(api, world: dict, arm: dict, run_dir: Path, restore: dict,
                      *, session, ownership, sensor, plan_factory=None,
                      route_departures: dict | None = None,
                      horizon_ticks: float | None = None) -> dict:
    """Run one restored WALL branch; returns the summary for the receipt.

    Same loop discipline as the v3 per-CAV branch driver (1 s control
    ticks on the SUMO grid, 2 x 0.5 s physics substeps, teleport and
    insertion lists) with the wall request replacing the per-CAV switch
    schedule.  The api/dispatcher/ownership/sensor are injected; ZERO
    SUMO here.  ``horizon_ticks`` is the registered smoke cap only.
    """
    if session is None or ownership is None:
        raise WallDriverError(
            "the wall branch drive requires the injected dispatcher "
            "session and ownership (the frozen stack is wired at launch); "
            "tests inject fakes")
    manifest = restore["manifest"]
    cav_ids = set(manifest["harness_state"]["roster"]["cav_ids"])
    if not cav_ids:
        raise WallDriverError("restored roster is empty")
    run_name = str(arm["run"])
    hold_s = float((arm.get("wall") or {}).get("hold_s", 0.0) or 0.0)
    if hold_s <= 0:
        raise WallDriverError(f"arm {arm['arm']!r} carries no recorded hold_s")
    t0_s = float(arm["restores_state"]["at_s"])
    if horizon_ticks is not None:
        horizon_s = t0_s + float(horizon_ticks)
    else:
        horizon_s = t0_s + float(world.get(
            "branch_length_s", arm.get("branch_length_s", BRANCH_LENGTH_S)))
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
    tracker = WallEpisodeTracker(wall_id_for(run_name))
    request = _WallRequest(
        run_name=run_name, arm=str(arm["arm"]), t0_s=t0_s, hold_s=hold_s,
        cav_ids=cav_ids, plan_factory=plan_factory)

    def _emit(kind: str, row: dict) -> None:
        if kind == "wall_event":
            event = {"kind": "wall_event", **row}
            tracker.observe(event)
            wall_events.append(event)
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
    for tick in range(int(round(t0_s)), int(round(horizon_s))):
        now = float(api.simulation.getTime())
        if abs(now - tick * 1.0) > 1e-6:
            raise WallDriverError(f"time grid mismatch at t={now}")
        live = list(api.vehicle.getIDList())
        frames = branch_driver._capture_frames(
            sensor, live, cav_ids, now, run_dir, unavailable)
        assignment_rows.clear()
        request.maybe_request(
            session, frames, now,
            lambda row: wall_events.append(row))
        # COMMAND ISSUANCE CONTRACT (WM-44, carried): the frozen
        # CommandOwnership and the dispatcher issue every speed command
        # THEMSELVES inside session.step; the driver only logs records.
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
        if pi_session is not None:
            pi_session.flush_evidence(
                lambda payload: append_jsonl(
                    Path(run_dir) / "suffix_commands.jsonl", payload),
                finalize=False,
            )
            pi_session.commit_evidence()
        pi_live = set(live)
        departed_now = sorted(str(v) for v in api.simulation.getDepartedIDList())
        starting_tp = sorted(str(v) for v in api.simulation.getStartingTeleportIDList())
        ending_tp = sorted(str(v) for v in api.simulation.getEndingTeleportIDList())
        teleports_seen.update(starting_tp)
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
            row["starting_teleports"] = starting_tp
            row["ending_teleports"] = ending_tp
            append_jsonl(Path(run_dir) / "physics_steps.jsonl", row)
            rows_written += 1
            previous_row = row
            for dispatch_row in wall_dispatcher_rows_from_telemetry(
                    assignment_rows, timestamp_s=after,
                    plan_id=request.plan_id):
                append_jsonl(Path(run_dir) / "dispatcher_rows.jsonl", dispatch_row)
                dispatcher_rows_written += 1
        for vehicle_id in departed_now:
            route_info = (route_departures or {}).get(vehicle_id, {})
            speed = None
            try:
                speed = float(api.vehicle.getSpeed(vehicle_id))
            except Exception:  # noqa: BLE001 - vehicle may leave instantly
                speed = None
            insertion = {
                "vehicle_id": vehicle_id,
                "scheduled_depart_s": route_info.get("depart_s"),
                "inserted_s": now,
                "entrance": branch_driver.entrance_of(route_info.get("route", "")),
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
    # explicit handback row after the terminal dispatcher event
    if tracker.finished_at is not None:
        wall_events.append({
            "kind": "wall_handback",
            "wall_id": tracker.wall_id,
            "timestamp_s": round(float(tracker.finished_at), 6),
            "how": tracker.finished_how,
            "phase": tracker.phase_at_finish,
            "reason": tracker.abort_reason,
            "members": list(tracker.members or request.members or ()),
            "handback": "native",
        })
    for row in wall_events:
        append_jsonl(Path(run_dir) / "wall_events.jsonl", row)

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
        # ---- wall fields (owner spec) ----
        "wall_id": request.wall_id,
        "members": list(request.members) if request.members else None,
        "speed_factor": request.factor,
        "wall_speed_mps": request.wall_speed_mps,
        "timing": request.timing,
        "hold_s": hold_s,
        "request_s": request.requested_at,
        "assembled_at_s": tracker.assembled_at,
        "released_at_s": (tracker.finished_at
                          if tracker.finished_how == "released" else None),
        "aborted_at_s": (tracker.finished_at
                         if tracker.finished_how == "safety_abort" else None),
        "abort_reason": tracker.abort_reason,
        "outcome": outcome,
        "wall_spec_sha256": schedule_digest(wall_spec),
        "created_utc": utc_now(),
    }
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
        "wall_id": request.wall_id,
        "members": list(request.members) if request.members else None,
        "wall_speed_mps": request.wall_speed_mps,
        "request_s": request.requested_at,
        "assembled_at_s": tracker.assembled_at,
        "released_at_s": branch_record["released_at_s"],
        "aborted_at_s": branch_record["aborted_at_s"],
        "abort_reason": tracker.abort_reason,
        "cav_speed_mode": ownership_mode,
        "observation_unavailable_rows": len(unavailable),
        "speed_mode_guard_checked": guard.checked if guard else 0,
        "speed_mode_guard_violations": len(guard.violations) if guard else 0,
        "load_state_calls": 1,  # the child's guard enforces exactly one
        "horizon_s": horizon_s,
        "workers_declared": WALLS_WORKERS,
    }
