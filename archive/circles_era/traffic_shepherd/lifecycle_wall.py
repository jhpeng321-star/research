"""Finite acquired-report wall commitment with an explicit coasting handback.

The planning model integrates only the proposed members' own commands. Other
vehicles are acquired constant-speed constraints, never simulator rollouts.
Its conditional forecast is not a safety invariant or a benefit predictor.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from math import isfinite
from statistics import mean

from .formation import make_proposal, assess_formation, _local_clearance_problem
from .gentle_wall import GentleProfile, resolve_command
from .paired_experiments import PolicyOutput, digest
from .report_dissolution import downstream_envelope

VERSION = "finite_lifecycle_wall/2"
TERMINAL_CONTRACT = "wall_command_zero_measured_acceleration_within_one_ramp_step_then_identical_damping"


@dataclass(frozen=True)
class LifecycleProfile(GentleProfile):
    version: str = VERSION

    def __post_init__(self):
        expected = asdict(GentleProfile())
        expected["version"] = VERSION
        if (asdict(self) != expected or type(self.readiness_samples) is not int
                or any(type(v) is bool for v in asdict(self).values())):
            raise ValueError("one frozen finite lifecycle profile required")


def coordination_requests(rows, proposal, *, exiting=False):
    """One common law for the declared plan and actual acquired-state control."""
    p = proposal.profile
    centre = mean(f.distance_to_merge_m for f in rows)
    result = {}
    for f in rows:
        offset = max(-p.maximum_coordination_offset_mps, min(p.maximum_coordination_offset_mps,
            p.position_gain_per_s * (f.distance_to_merge_m-centre)))
        desired = max(0., min(f.allowed_speed_mps, proposal.target_speed_mps+offset))
        result[f.vehicle_id] = (0. if exiting else p.speed_gain_per_s*(desired-f.speed_mps), desired)
    return result


def coast_terminal(rows, previous, profile):
    """Zero is a WALL coast endpoint, not a numerical native/IDM reference."""
    return all(f.vehicle_id in previous and abs(previous[f.vehicle_id]) <= 1e-8
        and abs(f.acceleration_mps2) <= profile.acceleration_change_limit_mps3+1e-8 for f in rows)


def _settled(rows, proposal):
    p = proposal.profile
    distances = [f.distance_to_merge_m for f in rows]
    speeds = [f.speed_mps for f in rows]
    return (max(distances)-min(distances) <= p.formed_span_m
        and max(speeds)-min(speeds) <= p.formed_speed_spread_mps
        and all(abs(v-proposal.target_speed_mps) <= p.target_speed_tolerance_mps for v in speeds))


def _fixed_window_plan(proposal, reports, *, now_s, previous_commands, readiness_count=0,
                       ready_at_s=None, exit_started_at_s=None, exit_now=False,
                       hold_not_before_s=None):
    """Integrate the finite remaining candidate maneuver, in seconds and metres.

    Readiness_count includes this tick's real observation. Future counts are
    model predictions, never substituted for measured runtime readiness. The
    model uses constant acceleration over each 1 s command, samples .5 s prefix
    clearances, and assumes commanded response. SUMO can respond differently;
    every next actual tick replans from acquired acceleration and position.
    No future damping acceleration is requested or forecast.
    """
    p = proposal.profile
    reports = tuple(reports)
    assessment = assess_formation(proposal, reports, now_s=now_s)
    base = dict(version=VERSION, terminal_contract=TERMINAL_CONTRACT,
        units=dict(time="s", distance="m", acceleration="m/s^2", ramp="m/s^3"),
        conditional_command_response_model=True, constant_reported_downstream_speed=True,
        unobserved_traffic_certified=False, safety_guarantee=False,
        rear_protection_horizon_s=p.control_dt_s, rear_future_response_certified=False,
        completed_hold_charged_again=False, exit_now=exit_now)
    if not assessment.approved:
        return dict(base, feasible=False, failure_kind="information_or_local_geometry",
                    reason=assessment.reason, remaining_total_s=None)
    frames = {r.frame.vehicle_id: r.frame for r in reports}
    rows = [frames[v] for v in proposal.member_ids]
    if any(f.position_m is None for f in rows):
        return dict(base, feasible=False, failure_kind="information", reason="missing_member_position",
                    remaining_total_s=None)
    downstream = downstream_envelope(proposal, reports, now_s=now_s)
    constraints = [{k: v for k, v in r.items() if k != "speed_envelope_mps"}
                   for r in downstream["constraints"]]
    # Local anonymous detections supplement CAV self reports. No vehicle type is
    # inferred, and absent detection beyond coverage is explicitly unknown.
    for f in rows:
        if f.front is not None:
            age = now_s-f.timestamp_s
            closing = -f.front.relative_speed_mps
            constraints.append(dict(member_id=f.vehicle_id, downstream_reporter_id=None,
                observation_age_s=age, downstream_speed_mps=max(0., f.speed_mps+f.front.relative_speed_mps),
                available_distance_m=f.front.gap_m-max(0., closing)*age-p.minimum_front_gap_m,
                source="anonymous_local_front"))
    prev = dict(previous_commands)
    origin = {f.vehicle_id: f.distance_to_merge_m for f in rows}
    travel = {f.vehicle_id: 0. for f in rows}
    map_margin = min(f.distance_to_merge_m-p.minimum_runway_m for f in rows)
    traffic_margin = None
    count = readiness_count
    ready = ready_at_s
    exit_at = exit_started_at_s
    if exit_now and exit_at is None:
        exit_at = now_s
    assembly_ticks = hold_ticks = exit_ticks = 0
    samples = []
    reason, kind = "finite_maneuver_and_coast_terminal_feasible", None
    for tick in range(int(p.assembly_deadline_s+p.hold_s+p.taper_s)+2):
        time = now_s+tick
        if ready is None and exit_at is None:
            if tick:
                count = count+1 if _settled(rows, proposal) else 0
            if count >= p.readiness_samples and (hold_not_before_s is None or time >= hold_not_before_s):
                ready = time
            elif time-proposal.start_s >= p.assembly_deadline_s:
                reason, kind = "predicted_readiness_after_assembly_deadline", "assembly"
                break
        if exit_at is None and ready is not None and time >= ready+p.hold_s:
            exit_at = time
        phase = "taper" if exit_at is not None else "hold" if ready is not None else "assembly"
        if phase == "hold":
            distances = [f.distance_to_merge_m for f in rows]
            speeds = [f.speed_mps for f in rows]
            if (max(distances)-min(distances) > p.formed_span_m
                    or max(speeds)-min(speeds) > p.formed_speed_spread_mps):
                reason, kind = "predicted_loss_of_compact_hold", "formation"
                break
        if phase == "taper" and coast_terminal(rows, prev, p):
            break
        if phase == "taper" and time-exit_at >= p.taper_s:
            reason, kind = "coast_terminal_not_reachable_before_exit_deadline", "exit"
            break
        requests = coordination_requests(rows, proposal, exiting=phase == "taper")
        next_rows = []
        midpoint_distances, midpoint_speeds = [], []
        for f in rows:
            acceleration, _ = resolve_command(f, requests[f.vehicle_id][0], p,
                                               previous=prev.get(f.vehicle_id))
            if acceleration is None:
                reason, kind = "predicted_ramp_or_speed_bounds_incompatible", "response"
                break
            for dt in (.5, 1.):
                displacement = f.speed_mps*dt+.5*acceleration*dt*dt
                total = travel[f.vehicle_id]+displacement
                # Account for receipt age as additional already travelled road.
                age_reserve = frames[f.vehicle_id].speed_mps*(now_s-frames[f.vehicle_id].timestamp_s)
                map_margin = min(map_margin, origin[f.vehicle_id]-age_reserve-total-p.minimum_runway_m)
                for c in constraints:
                    if c["member_id"] == f.vehicle_id:
                        margin = c["available_distance_m"]-total+c["downstream_speed_mps"]*(tick+dt)
                        traffic_margin = margin if traffic_margin is None else min(traffic_margin, margin)
            displacement = f.speed_mps+.5*acceleration
            midpoint_distances.append(f.distance_to_merge_m-.5*f.speed_mps-.125*acceleration)
            midpoint_speeds.append(f.speed_mps+.5*acceleration)
            travel[f.vehicle_id] += displacement
            prev[f.vehicle_id] = acceleration
            next_rows.append(replace(f, timestamp_s=time+1, speed_mps=max(0., f.speed_mps+acceleration),
                acceleration_mps2=acceleration, distance_to_merge_m=f.distance_to_merge_m-displacement,
                position_m=f.position_m+displacement))
        if kind is not None:
            break
        if phase == "hold" and (
                max(midpoint_distances)-min(midpoint_distances) > p.formed_span_m
                or max(midpoint_speeds)-min(midpoint_speeds) > p.formed_speed_spread_mps
                or max(f.distance_to_merge_m for f in next_rows)-min(
                    f.distance_to_merge_m for f in next_rows) > p.formed_span_m
                or max(f.speed_mps for f in next_rows)-min(f.speed_mps for f in next_rows)
                    > p.formed_speed_spread_mps):
            reason, kind = "predicted_loss_of_compact_hold", "formation"
            break
        rows = next_rows
        assembly_ticks += phase == "assembly"
        hold_ticks += phase == "hold"
        exit_ticks += phase == "taper"
        samples.append(dict(elapsed_s=tick+1, phase=phase,
                            commands_mps2=dict(prev), distance_travelled_m=dict(travel)))
    else:
        reason, kind = "finite_planner_iteration_bound", "implementation"
    # Keep independent margins even when both fail: map extension cannot remove
    # a traffic constraint. Duration/accounting diagnostics remain available.
    map_ok = map_margin >= 0
    traffic_ok = traffic_margin is None or traffic_margin >= 0
    if kind is None and not map_ok:
        reason, kind = "remaining_maneuver_exceeds_map_boundary", "map_boundary"
    if kind is None and not traffic_ok:
        reason, kind = "reported_traffic_intersects_remaining_maneuver", "traffic"
    return dict(base, feasible=kind is None, reason=reason, failure_kind=kind,
        predicted_hold_start_s=ready,
        remaining_assembly_s=assembly_ticks, remaining_hold_s=hold_ticks,
        remaining_exit_s=exit_ticks, remaining_total_s=len(samples),
        minimum_map_margin_m=map_margin, minimum_traffic_margin_m=traffic_margin,
        map_boundary_feasible=map_ok, reported_traffic_feasible=traffic_ok,
        downstream_constraint_count=len(constraints), downstream_reports_available=downstream["available"],
        fresh_downstream_reporter_ids=sorted({c["downstream_reporter_id"] for c in constraints
                                            if c["downstream_reporter_id"] is not None}),
        stale_report_count=sum(now_s-r.frame.timestamp_s > p.max_report_age_s for r in reports),
        member_front_coverage_m={f.vehicle_id:f.front_coverage_m for f in rows},
        member_rear_coverage_m={f.vehicle_id:f.rear_coverage_m for f in rows},
        constraints=constraints, planned_samples=samples)


def remaining_plan(proposal, reports, *, now_s, previous_commands, readiness_count=0,
                   ready_at_s=None, exit_started_at_s=None, exit_now=False):
    """Earliest maintainable HOLD window under one unchanged command law.

    Assembly and HOLD share the same physical law, so postponing a premature
    HOLD label adds no alternative control parameter. A failed compactness
    forecast advances the candidate start by one existing control tick. Each
    candidate still requires consecutive readiness and a complete fixed HOLD
    plus coast exit before it qualifies. Deadline, map, traffic, response and
    information failures are not relaxed. This is bounded deterministic phase
    scheduling, not a fitted duration/target/braking search or a SUMO rollout.

    Once HOLD has actually started its clock is immutable. The caller alone
    maintains measured readiness; candidate future samples cannot increment it.
    """
    common = dict(now_s=now_s, previous_commands=previous_commands,
        readiness_count=readiness_count, ready_at_s=ready_at_s,
        exit_started_at_s=exit_started_at_s, exit_now=exit_now)
    reports = tuple(reports)
    attempted = []
    earliest = now_s
    limit = int(proposal.profile.assembly_deadline_s/proposal.profile.control_dt_s)+1
    for _ in range(limit):
        plan = _fixed_window_plan(proposal, reports, **common, hold_not_before_s=earliest)
        start = plan.get("predicted_hold_start_s")
        attempted.append(dict(candidate_hold_start_s=start, feasible=plan["feasible"],
                              reason=plan["reason"]))
        adjustable = ready_at_s is None and exit_started_at_s is None and not exit_now
        if (not adjustable or plan["feasible"] or plan["failure_kind"] != "formation"
                or start is None):
            return dict(plan, maintainable_readiness_version=2,
                readiness_candidates=attempted, candidate_count=len(attempted),
                instantaneous_readiness_satisfied=readiness_count >= proposal.profile.readiness_samples,
                maintainable_hold_ready_now=bool(plan["feasible"] and adjustable and start == now_s))
        earliest = start+proposal.profile.control_dt_s
    # Impossible for an ordinary <=50s assembly interval, but never substitute
    # the last invalid forecast if invalid timing inputs exhaust the bound.
    return dict(plan, feasible=False, failure_kind="assembly",
        reason="no_maintainable_hold_window_before_assembly_deadline",
        maintainable_readiness_version=2, readiness_candidates=attempted,
        candidate_count=len(attempted), maintainable_hold_ready_now=False,
        instantaneous_readiness_satisfied=readiness_count >= proposal.profile.readiness_samples)


class LifecycleWallPolicy:
    """Registered diagnostic treatment; no manually claimed benefit approval."""
    def __init__(self, job, registration):
        if job["arm"] != "wall":
            raise ValueError("finite lifecycle is an explicit wall treatment")
        self.job, self.registration = deepcopy(job), registration
        self.profile = LifecycleProfile(**self.job["policy_spec"]["profile"])
        self.proposal = self.terminal = self.last_time = None
        self.terminal_at = None
        self.previous_commands = {}
        self.readiness = 0
        self.ready_at = self.taper_at = None
        self.early_exit_reason = None

    def __call__(self, context):
        now, reports = context.now_s, tuple(context.reports)
        if now < self.job["decision_time_s"]:
            return PolicyOutput()
        if self.last_time is not None and now != self.last_time+1:
            raise ValueError("finite lifecycle requires exact one-second clock")
        self.last_time = now
        spec = self.job["policy_spec"]
        if self.proposal is None and self.terminal is None:
            if now != self.job["decision_time_s"]:
                raise ValueError("no late or replacement opportunity")
            try:
                self.proposal = make_proposal(reports, now_s=now,
                    required_lanes=tuple(spec["required_lanes"]), member_ids=tuple(spec["member_ids"]),
                    profile=self.profile)
            except ValueError as exc:
                self.terminal = "refused", "proposal_unavailable:"+str(exc)
        if self.proposal is None:
            return PolicyOutput({}, *self.terminal, tuple(spec["member_ids"]), digest(spec),
                digest([r.payload() for r in reports]), dict(lifecycle_version=VERSION))
        proposal, p = self.proposal, self.profile
        assessment = assess_formation(proposal, reports, now_s=now)
        frames = {r.frame.vehicle_id:r.frame for r in reports}
        details = dict(lifecycle_version=VERSION, assessment=asdict(assessment),
            span_m=assessment.initial_or_current_span_m, speed_spread_mps=assessment.speed_spread_mps,
            target_speed_mps=proposal.target_speed_mps, readiness_samples_required=p.readiness_samples,
            formed_span_m=p.formed_span_m, formed_speed_spread_mps=p.formed_speed_spread_mps,
            target_speed_tolerance_mps=p.target_speed_tolerance_mps,
            absolute_wall_min_mps2=p.acceleration_min_mps2, command_ramp_mps3=p.acceleration_change_limit_mps3,
            experimental_authorization="registered_experimental_assignment_not_trained_benefit_gate")

        def emit(phase, reason, commands=None):
            if phase in {"refused", "abort", "release"}:
                if self.terminal is None:
                    self.terminal_at = now
                self.terminal = phase, reason
            values = commands or {}
            details.update(readiness_count=self.readiness, assembly_completed_at_s=self.ready_at,
                instantaneous_readiness_satisfied=self.readiness >= p.readiness_samples,
                hold_commitment_started=self.ready_at is not None,
                exit_started_at_s=self.taper_at, early_exit_reason=self.early_exit_reason,
                nominal_maximum_hold_s=p.hold_s,
                actual_hold_elapsed_s=0. if self.ready_at is None else max(0.,
                    (self.taper_at if self.taper_at is not None else
                     self.terminal_at if self.terminal_at is not None else now)-self.ready_at),
                terminal_at_s=self.terminal_at,
                terminal_contract=TERMINAL_CONTRACT, override_released=not bool(values),
                continued_wall_ownership=bool(values))
            if values:
                self.previous_commands = dict(values)
            return PolicyOutput(values, phase, reason, proposal.member_ids,
                proposal.fingerprint, assessment.evidence_fingerprint, details)

        if self.terminal:
            return emit(*self.terminal)
        failure_phase = "refused" if now == proposal.start_s else "abort"
        if not assessment.approved:
            return emit(failure_phase, "lifecycle_safety_or_information:"+assessment.reason)
        members = set(proposal.member_ids)
        if not members <= set(context.damping_commands) | set(context.damping_native_ids) or any(
            not isfinite(context.damping_commands[v]) for v in members if v in context.damping_commands):
            return emit(failure_phase, "lifecycle_missing_current_local_damping_reference")
        rows = [frames[v] for v in proposal.member_ids]
        compact = (assessment.initial_or_current_span_m <= p.formed_span_m
                   and assessment.speed_spread_mps <= p.formed_speed_spread_mps)
        details["compact"] = compact
        if self.ready_at is None and self.taper_at is None:
            self.readiness = self.readiness+1 if _settled(rows, proposal) else 0
            if (self.readiness < p.readiness_samples
                    and now-proposal.start_s >= p.assembly_deadline_s):
                return emit("abort", "lifecycle_assembly_deadline_without_measured_readiness")
        if self.ready_at is not None and self.taper_at is None:
            if not compact:
                return emit("abort", "lifecycle_lost_compact_hold")
            if now >= self.ready_at+p.hold_s:
                self.taper_at = now
        if self.taper_at is not None and coast_terminal(rows, self.previous_commands, p):
            return emit("release", "lifecycle_coast_terminal_return_to_identical_damping")
        if self.taper_at is not None and now-self.taper_at >= p.taper_s:
            return emit("abort", "lifecycle_exit_deadline_without_measured_coast_terminal")
        plan = remaining_plan(proposal, reports, now_s=now, previous_commands=self.previous_commands,
            readiness_count=self.readiness, ready_at_s=self.ready_at, exit_started_at_s=self.taper_at)
        details["lifecycle_plan"] = dict(commitment=plan)
        if (self.ready_at is None and self.taper_at is None
                and plan.get("maintainable_hold_ready_now")):
            # Both tests refer to this acquired tick. Predicted future samples
            # remain diagnostics and never become runtime readiness credit.
            if self.readiness < p.readiness_samples:
                raise AssertionError("maintainable HOLD cannot bypass measured readiness")
            self.ready_at = now
        if not plan["feasible"]:
            if now == proposal.start_s:
                return emit("refused", "lifecycle_entry:"+plan["reason"])
            exit_plan = remaining_plan(proposal, reports, now_s=now, previous_commands=self.previous_commands,
                readiness_count=self.readiness, ready_at_s=self.ready_at,
                exit_started_at_s=self.taper_at, exit_now=True)
            details["lifecycle_plan"]["exit_now"] = exit_plan
            if not exit_plan["feasible"]:
                return emit("abort", "lifecycle_no_feasible_owned_exit:"+exit_plan["reason"])
            if self.taper_at is None:
                self.taper_at, self.early_exit_reason = now, plan["reason"]
            if coast_terminal(rows, self.previous_commands, p):
                return emit("release", "lifecycle_early_coast_terminal_return_to_identical_damping")
        phase = "taper" if self.taper_at is not None else "hold" if self.ready_at is not None else "assembly"
        requests = coordination_requests(rows, proposal, exiting=phase == "taper")
        commands, resolutions = {}, {}
        for f in rows:
            acceleration, evidence = resolve_command(f, requests[f.vehicle_id][0], p,
                previous=self.previous_commands.get(f.vehicle_id))
            resolutions[f.vehicle_id] = dict(evidence, desired_speed_mps=requests[f.vehicle_id][1])
            details["command_resolution"] = resolutions
            if acceleration is None:
                return emit(failure_phase, "lifecycle_measured_response_incompatible")
            problem = _local_clearance_problem(f, p, now, acceleration=acceleration)
            if problem:
                return emit(failure_phase, "lifecycle_protective_clearance:"+problem[1])
            commands[f.vehicle_id] = acceleration
        return emit(phase, "lifecycle_finite_remaining_commitment_and_coast_exit", commands)


def policy_factory(job, registration):
    return LifecycleWallPolicy(job, registration)
