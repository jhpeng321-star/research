"""WM-43 mechanics control loop: dispatcher wall groups on the real ownership path.

Mirrors the frozen ``traffic_shepherd.workflow.evaluate`` loop discipline
(SUMO start, 1 s control / 0.5 s physics grid asserts, per-substep safety
metrics, response receipts, release_all + clean shutdown) but drives
``MixedAssignmentSession`` (mixed_assignment/3) instead of a fleet session.
Zero retries: any exception propagates to the child's failed receipt.

Evidence rows written to telemetry.jsonl (checks parse these, never memory
state): ``command``, ``response``, ``mixed_assignment``, ``wall_event``
(dispatcher lifecycle), ``wall_proximity`` (per-tick rear-end surveillance
geometry for non-members), ``scripted_actuator`` (A3 fault injections),
``observation_unavailable``.

The A3 scripted actuator is a fault INJECTOR, not a control write: it pins
the declared front member's speed with ``traci.vehicle.setSpeed`` AFTER the
single ownership write of that tick, only while the wall is assembled and
not yet finished, and every injection is logged.  All control commands
still flow through ``CommandOwnership`` alone.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from traffic_shepherd import wall_group
from traffic_shepherd.config import AdapterConfig, load_config
from traffic_shepherd.controller_factory import create_controller
from traffic_shepherd.metrics import braking_effort, ttc_exposure
from traffic_shepherd.mixed_assignment import (AssignmentPlan, MixedAssignmentSession,
                                               WallAssignment)
from traffic_shepherd.ownership import CommandOwnership
from traffic_shepherd.scenarios import observation_map
from traffic_shepherd.sensing import SumoLocalSensor
from traffic_shepherd.simulation import close_sumo, start_sumo

from wm43_plan import ARM_SPECS, HORIZON_TICKS

SUMMARY_SCHEMA = "wm43_run_summary/1"
TELEMETRY_NAME = "telemetry.jsonl"
INTERVALS_NAME = "intervals.jsonl"


def initial_plan(world: dict) -> AssignmentPlan:
    """The arm's single planner plan, applied at t=0 (immediate placement).

    Wall members are assigned ``wall-v2``; the A2 follower is ``damping``;
    every other roster CAV is named ``native`` (the plan is strict_roster).
    """
    spec = ARM_SPECS[world["arm"]]
    assignments = {member: "wall-v2" for member in spec["members"]}
    assignments.update(spec["assignment_extra"])
    for cav in spec["extra_cavs"]:
        assignments.setdefault(cav, "native")
    return AssignmentPlan(
        assignments=assignments,
        plan_id=f"{world['run']}-p0",
        commit_horizon_s=0.0,
        walls=(WallAssignment(wall_id=spec["wall_id"], members=tuple(spec["members"]),
                              wall_speed_mps=world["wall_speed_mps"], phase="assemble",
                              release_at_s=spec["release_at_s"]),),
        strict_roster=True,
    )


def run_mechanics(world: dict, run_dir: Path, *, api=None, controller=None,
                  sumo_binary: str = "sumo", config: AdapterConfig | None = None,
                  horizon_ticks: int | None = None) -> dict:
    """One mechanics run; the scenario must already exist under run_dir/scenario.

    ``api``/``controller`` are injectable for mocked-TraCI tests; production
    uses the real traci module and the frozen CIRCLES damping controller.
    Returns the run summary dict (the child turns it into receipts).
    """
    from wm43_scenario import build_scenario  # deferred: keeps the import surface small
    from wm43_validate import validate_scenario

    config = config if config is not None else load_config()
    # horizon: the registered 600-tick run, or a pre-launch smoke cap (the
    # standing smoke uses 70 ticks - just past the assembly critical point)
    ticks_planned = HORIZON_TICKS if horizon_ticks is None else int(horizon_ticks)
    if not 1 <= ticks_planned <= HORIZON_TICKS:
        raise ValueError(f"horizon_ticks must be in [1, {HORIZON_TICKS}]")
    started = time.monotonic()
    scene, roster, scenario_manifest = build_scenario(run_dir, world, config)
    # static gate BEFORE any SUMO start: rejects the WM-43 r1 failure class
    # (vehicles referencing an undefined vType) and every other registered
    # dangling-reference/path/rule violation, with zero simulation.
    problems = validate_scenario(Path(scene).parent)
    if problems:
        raise RuntimeError("scenario validation refused the generated inputs: "
                           + " | ".join(problems))
    cav_ids = {v.vehicle_id for v in roster if v.is_cav}
    spec = ARM_SPECS[world["arm"]]
    members = tuple(spec["members"])
    injection = spec["injection"]
    wall_speed = float(world["wall_speed_mps"])

    if api is None:
        import traci as api  # noqa: PLC0415 - launch-time only
    if controller is None:
        controller, _ = create_controller("damping", config)

    launch_command: list = []
    start_sumo(sumo_binary, scene, seed=int(world["seed"]), config=config, api=api,
               command_record=launch_command)
    ownership = CommandOwnership(api, cav_ids, controller)
    session = MixedAssignmentSession(circles_controller=controller, pi_session=None,
                                     cav_ids=cav_ids)
    sensor = SumoLocalSensor(api, config, observation_map(api, config))
    wall_state = {"started": False, "assembled": False, "done": False, "active": False}
    command_counts: dict[str, int] = {}
    fallback_counts: dict[str, int] = {}
    totals = {"braking_vehicle_seconds": 0.0, "acceleration_energy": 0.0,
              "ttc_below_3_vehicle_seconds": 0.0, "collision_involvements": 0,
              "starting_teleports": 0}
    minimum_ttc = None
    previous_live: set = set()
    retired_cavs: set = set()
    collision_previous: set = set()
    ticks_executed = 0
    injection_count = 0
    wall_events: list[dict] = []
    sumo_version: list | None = None

    try:
        with (run_dir / TELEMETRY_NAME).open("x", encoding="utf-8") as telemetry, \
                (run_dir / INTERVALS_NAME).open("x", encoding="utf-8") as intervals:
            def emit(kind: str, payload: dict) -> None:
                if kind == "wall_event":
                    event = payload.get("event")
                    if event == "assembly_start":
                        wall_state["started"] = True
                        wall_state["active"] = True
                    elif event == "assembled":
                        wall_state["assembled"] = True
                    elif event in ("released", "safety_abort"):
                        wall_state["active"] = False
                        wall_state["done"] = True
                    wall_events.append(dict(payload))
                telemetry.write(json.dumps(dict(kind=kind, **payload),
                                           allow_nan=False) + "\n")

            session.bind_emit(emit)
            # A3 fix: the plan is submitted on the FIRST TICK WHERE THE FULL
            # ROSTER IS LIVE, not blindly at t=0.  Evidence (r2 smoke,
            # a1_seed920244): at simulation time 0.0 - before the first
            # simulationStep() - SUMO has not yet inserted the depart=0
            # vehicles, so getIDList() was empty; the staged wall's
            # member_frame_missing emergency abort fired at t=0
            # (telemetry: wall_event safety_abort reason=member_frame_missing
            # timestamp_s=0.0) and the wall was handed back to native before
            # assembly_start could ever fire.  Staging is planner-side
            # behavior: deferring it until every member is observable keeps
            # the dispatcher untouched and the abort semantics intact.
            plan_applied = False

            for tick in range(ticks_planned):
                now = float(api.simulation.getTime())
                if abs(now - tick * config.control_dt_s) > 1e-6:
                    raise RuntimeError("SUMO/control time grid mismatch")
                live = set(api.vehicle.getIDList())
                if not plan_applied:
                    if cav_ids <= live:
                        session.apply_plan(initial_plan(world), now_s=now)
                        plan_applied = True
                        emit("plan_submitted", dict(timestamp_s=now,
                                                    plan_id=f"{world['run']}-p0",
                                                    live_roster=sorted(cav_ids)))
                    else:
                        emit("plan_deferred", dict(
                            timestamp_s=now,
                            reason="roster_not_fully_live",
                            live=sorted(live & cav_ids),
                            missing=sorted(cav_ids - live)))
                disappeared = (previous_live - live) & cav_ids
                arrivals = set(api.simulation.getArrivedIDList()) & cav_ids
                retired_cavs.update(disappeared | arrivals)
                reappeared = retired_cavs & live
                retired_cavs.difference_update(reappeared)
                frames = {}
                for cav in sorted(live & cav_ids):
                    try:
                        frames[cav] = sensor.capture(cav, now_s=now)
                    except ValueError as exc:
                        emit("observation_unavailable", dict(vehicle_id=cav,
                                                             timestamp_s=now,
                                                             reason=str(exc)))
                records = session.step(ownership, frames, now_s=now, live_ids=live,
                                       disappeared_ids=disappeared,
                                       reappeared_ids=reappeared)
                ticks_executed = tick + 1
                # wall-v2.1: injection phase "assembly" (A5) starts as soon as
                # the wall session has started assembling; the default
                # (A3) waits for the assembled event
                inject_gate = (wall_state["started"]
                               if injection is not None
                               and injection.get("phase") == "assembly"
                               else wall_state["assembled"])
                if (injection is not None and inject_gate
                        and not wall_state["done"] and now >= injection["start_s"]):
                    api.vehicle.setSpeed(injection["member"], float(injection["target_mps"]))
                    injection_count += 1
                    emit("scripted_actuator",
                         dict(timestamp_s=now, vehicle_id=injection["member"],
                              method="setSpeed", value_mps=float(injection["target_mps"]),
                              reason=("wm43_a5_assembly_timeout_fault"
                                     if injection.get("phase") == "assembly"
                                     else "wm43_a3_induced_span_fault")))
                if wall_state["active"]:
                    _emit_wall_frames(emit, now, frames, members, spec["wall_id"])
                    _emit_wall_proximity(emit, now, frames, members, spec, wall_speed)
                commanded = set()
                for record in records:
                    row = record.as_dict()
                    emit("command", row)
                    commanded.add(record.vehicle_id)
                    command_counts[record.owner] = command_counts.get(record.owner, 0) + 1
                    if record.fallback_reason:
                        fallback_counts[record.fallback_reason] = (
                            fallback_counts.get(record.fallback_reason, 0) + 1)
                interval = {"braking_vehicle_seconds": 0.0, "acceleration_energy": 0.0,
                            "ttc_below_3_vehicle_seconds": 0.0,
                            "collision_involvements": 0, "starting_teleports": 0}
                after = now
                for substep in range(round(config.control_dt_s / config.physics_dt_s)):
                    api.simulationStep()
                    after = float(api.simulation.getTime())
                    if abs(after - (now + (substep + 1) * config.physics_dt_s)) > 1e-6:
                        raise RuntimeError("SUMO physics time grid mismatch")
                    live_after = set(api.vehicle.getIDList())
                    accelerations = [float(api.vehicle.getAcceleration(v))
                                     for v in sorted(live_after)]
                    effort = braking_effort(accelerations, config.physics_dt_s)
                    interval["braking_vehicle_seconds"] += effort["braking_vehicle_seconds"]
                    interval["acceleration_energy"] += effort["acceleration_energy"]
                    exposure, closing_minimum = ttc_exposure(api, sorted(live_after),
                                                             config.physics_dt_s)
                    interval["ttc_below_3_vehicle_seconds"] += exposure
                    if closing_minimum is not None:
                        minimum_ttc = (closing_minimum if minimum_ttc is None
                                       else min(minimum_ttc, closing_minimum))
                    colliding = set(api.simulation.getCollidingVehiclesIDList())
                    interval["collision_involvements"] += len(colliding - collision_previous)
                    collision_previous = colliding
                    interval["starting_teleports"] += api.simulation.getStartingTeleportNumber()
                    live_after_set = live_after
                    for cav in sorted(commanded & live_after_set):
                        response = ownership.record_response(
                            cav, timestamp_s=after,
                            speed_mps=float(api.vehicle.getSpeed(cav)),
                            acceleration_mps2=float(api.vehicle.getAcceleration(cav)))
                        emit("response", response.as_dict())
                for key in interval:
                    totals[key] += interval[key]
                intervals.write(json.dumps(dict(start_s=now, end_s=after, **interval),
                                           allow_nan=False) + "\n")
                telemetry.flush()
                intervals.flush()
                previous_live = live

        ownership.release_all(live_ids=set(api.vehicle.getIDList()))
        ownership = None
        sumo_version = list(api.getVersion())
        close_sumo(api)
        api = None
    finally:
        if api is not None:
            try:
                if ownership is not None:
                    ownership.release_all(live_ids=set(api.vehicle.getIDList()))
            finally:
                close_sumo(api)

    return dict(
        schema=SUMMARY_SCHEMA, run=world["run"], arm=world["arm"], seed=int(world["seed"]),
        ticks_executed=ticks_executed, horizon_ticks=ticks_planned, totals=totals,
        minimum_closing_ttc_s=minimum_ttc, command_counts=command_counts,
        fallback_counts=fallback_counts, wall_events=wall_events,
        scripted_injections=injection_count, roster=sorted(cav_ids),
        scenario_manifest=scenario_manifest, launch_command=launch_command,
        sumo_version=sumo_version, elapsed_wall_s=time.monotonic() - started,
    )


def _emit_wall_frames(emit, now_s: float, frames: dict, members: tuple,
                      wall_id: str) -> None:
    """wall-v2.1 evidence: the member frame content the session decided on
    (speed and the downstream coordinate s_m = -distance_to_merge_m), so the
    checks can re-derive the closure law offline.  Pure telemetry."""
    rows = {}
    for member in members:
        frame = frames.get(member)
        if frame is None:
            return
        s_m = wall_group.downstream_position_m(frame)
        if s_m is None:
            return
        rows[member] = dict(speed_mps=float(frame.speed_mps), s_m=float(s_m))
    emit("wall_frames", dict(timestamp_s=now_s, wall_id=wall_id, members=rows))


def _emit_wall_proximity(emit, now_s: float, frames: dict, members: tuple,
                         spec: dict, wall_speed_mps: float) -> None:
    """Mirror the session's rear-end surveillance geometry as evidence rows.

    The session aborts when a non-member is inside ``[rear - REAR_GAP_M,
    rear]`` AND faster than the wall speed plus ``REAR_SPEED_EXCESS_MPS``;
    the row records the same comparison so the checks can prove the region
    was never entered at an aborting speed (no false abort, arm A2).
    """
    member_frames = [frames.get(member) for member in members]
    if any(frame is None for frame in member_frames):
        return
    rear = min(float(frame.position_m) for frame in member_frames)
    for cav in spec["extra_cavs"]:
        frame = frames.get(cav)
        if frame is None:
            continue
        gap = rear - float(frame.position_m)
        excess = float(frame.speed_mps) - wall_speed_mps
        in_region = 0.0 <= gap <= wall_group.REAR_GAP_M
        emit("wall_proximity", dict(
            timestamp_s=now_s, wall_id=spec["wall_id"], non_member=cav,
            rear_gap_m=gap, speed_mps=float(frame.speed_mps),
            speed_excess_mps=excess, in_rear_region=in_region,
            would_abort=bool(in_region
                             and excess > wall_group.REAR_SPEED_EXCESS_MPS)))
