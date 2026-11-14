"""Versioned gentle wall; acquired reports only, no simulator command writer.

The final resolver is shared by assembly, HOLD and report-aware dissolution.
It bounds ABSOLUTE wall acceleration, not a residual on CIRCLES or IDM.
"""
from dataclasses import asdict, dataclass
from math import isfinite
from statistics import mean

from .formation import FormationProfile, make_proposal, assess_formation, _local_clearance_problem
from .paired_experiments import PolicyOutput, digest
from .report_dissolution import downstream_envelope

VERSION = "gentle_report_wall/1"
IMPLEMENTATION_REVISION = "gentle_report_wall/1.0.1"


@dataclass(frozen=True)
class GentleProfile(FormationProfile):
    version: str = VERSION
    acceleration_min_mps2: float = -.3
    acceleration_change_limit_mps3: float = .15
    hold_s: float = 20.

    def __post_init__(self):
        # Validate the unchanged geometric/lifecycle contract without changing
        # the historical profile, its source, or the frozen damping adapter.
        old = asdict(self)
        old.pop("acceleration_change_limit_mps3")
        old["acceleration_min_mps2"] = -3.
        FormationProfile(**old)
        if (self.version != VERSION or self.acceleration_min_mps2 != -.3
                or self.acceleration_change_limit_mps3 != .15):
            raise ValueError("one frozen gentle wall version, absolute bound and ramp required")


def resolve_command(frame, requested, profile, *, previous=None):
    """Intersect absolute, slew, acquired acceleration and speed-target bounds.

    Empty intersection means handback, never a stronger or unsafe wall command.
    Measured acceleration is a self observation, not an invented native action.
    """
    p = profile
    measured = frame.acceleration_mps2
    if not all(isfinite(v) for v in (requested, measured)):
        raise ValueError("finite requested and acquired acceleration required")
    step = p.acceleration_change_limit_mps3 * p.control_dt_s
    prior = measured if previous is None else previous
    lo = max(p.acceleration_min_mps2, prior-step, measured-step, -frame.speed_mps/p.control_dt_s)
    hi = min(p.acceleration_max_mps2, prior+step, measured+step,
             (frame.allowed_speed_mps-frame.speed_mps)/p.control_dt_s)
    evidence = dict(requested_acceleration_mps2=requested, previous_wall_command_mps2=previous,
        measured_entry_or_current_acceleration_mps2=measured, admissible_min_mps2=lo,
        admissible_max_mps2=hi, maximum_command_change_mps2=step,
        entry_reference="acquired_measured_acceleration" if previous is None else "previous_wall_command_and_acquired_acceleration")
    if lo > hi + 1e-10:
        return None, dict(evidence, reason="gentle_smooth_speed_bounds_incompatible")
    value = max(lo, min(hi, requested))
    return float(value), dict(evidence, resolved_wall_acceleration_mps2=value)


def ramped_braking_distance(speed, target, acceleration, profile):
    """Kinematic optimistic stopping distance with this exact discrete ramp.

    Constant target/downstream speed; no hidden rollout. Integrate each 1 s
    constant-acceleration segment, including a fractional final segment.
    This is an estimate, not a future safety guarantee.
    """
    if speed <= target:
        return 0., 0.
    distance, elapsed, v, a = 0., 0., speed, acceleration
    for _ in range(10000):
        a = max(profile.acceleration_min_mps2, a-profile.acceleration_change_limit_mps3)
        dt = min(1., (v-target)/(-a)) if a < 0 else 1.
        distance += v*dt + .5*a*dt*dt
        v += a*dt
        elapsed += dt
        if v <= target+1e-10:
            return distance, elapsed
    raise ValueError("unbounded kinematic estimate")


def gentle_envelope(proposal, reports, *, now_s, previous_commands):
    # Reuse EXACT acquired reporter selection/age/length/reserve logic. Replace
    # the v1 constant-1.5 speed envelope by this ramped relative-distance check.
    evidence = downstream_envelope(proposal, reports, now_s=now_s)
    frames = {r.frame.vehicle_id: r.frame for r in reports}
    constraints = []
    for row in evidence["constraints"]:
        row = {k:v for k,v in row.items() if k != "speed_envelope_mps"}
        f = frames[row["member_id"]]
        prior = max(f.acceleration_mps2, previous_commands.get(f.vehicle_id, f.acceleration_mps2))
        distance, time = ramped_braking_distance(f.speed_mps, row["downstream_speed_mps"], prior, proposal.profile)
        relative = distance-row["downstream_speed_mps"]*time
        constraints.append(dict(row, required_ramped_relative_distance_m=relative,
            ramped_braking_time_s=time, gentle_stopping_feasible=relative <= row["available_distance_m"]))
    # Drop the legacy speed number so no downstream branch can use its 1.5 cap.
    return dict(available=evidence["available"], constraints=constraints,
        gentle_stopping_feasible=all(r["gentle_stopping_feasible"] for r in constraints),
        comfortable_deceleration_mps2=-proposal.profile.acceleration_min_mps2,
        acceleration_change_limit_mps3=proposal.profile.acceleration_change_limit_mps3,
        constant_downstream_speed_assumption=True, safety_guarantee=False)


def gentle_front_clearance(frame, acceleration, profile, *, now_s):
    """Check this actual command then ramped closure to acquired front speed.

    A constant-speed detected leader is a conditional model, not ground truth
    about future braking. SUMO's original protection remains available.
    """
    if frame.front is None:
        return None
    leader_speed = max(0., frame.speed_mps+frame.front.relative_speed_mps)
    age = now_s-frame.timestamp_s
    closing = frame.speed_mps-leader_speed
    gap = frame.front.gap_m-max(0., closing)*age
    dt = profile.control_dt_s
    next_speed = max(0.,frame.speed_mps+acceleration*dt)
    distance,time = ramped_braking_distance(next_speed,leader_speed,acceleration,profile)
    next_closure = closing*dt+.5*acceleration*dt**2
    required = max(0.,next_closure+distance-leader_speed*time)
    available = gap-profile.minimum_front_gap_m
    return dict(required_relative_distance_m=required,available_relative_distance_m=available,
        acquired_leader_speed_mps=leader_speed, gentle_feasible=required <= available,
        constant_leader_speed_assumption=True,safety_guarantee=False)


class GentleWallPolicy:
    def __init__(self, job, registration):
        if job["arm"] != "wall":
            raise ValueError("gentle policy is an explicit registered treatment only")
        self.job, self.registration = job, registration
        self.proposal = None
        self.previous_commands = {}
        self.terminal = None
        self.last_time = None
        self.readiness = 0
        self.ready_at = self.taper_at = None
        self.initial_estimates = None
        self.report_bridge_active = False

    def __call__(self, context):
        now = context.now_s
        if now < self.job["decision_time_s"]:
            return PolicyOutput()
        if self.last_time is not None and now != self.last_time+1:
            raise ValueError("gentle wall requires the exact one-second clock")
        self.last_time = now
        reports = tuple(context.reports)
        spec = self.job["policy_spec"]
        if self.proposal is None and self.terminal is None:
            if now != self.job["decision_time_s"]:
                raise ValueError("no late or replacement opportunity")
            try:
                self.proposal = make_proposal(reports, now_s=now,
                    required_lanes=tuple(spec["required_lanes"]), member_ids=tuple(spec["member_ids"]),
                    profile=GentleProfile(**spec["profile"]))
            except ValueError as exc:
                self.terminal = ("refused", "proposal_unavailable:"+str(exc))
        if self.proposal is None:
            return PolicyOutput({}, *self.terminal, tuple(spec["member_ids"]), digest(spec),
                digest([r.payload() for r in reports]), dict(proposal_materialized=False))
        proposal, p = self.proposal, self.proposal.profile
        assessment = assess_formation(proposal, reports, now_s=now)
        frames = {r.frame.vehicle_id: r.frame for r in reports}
        compact = bool(assessment.approved and assessment.initial_or_current_span_m <= p.formed_span_m
                       and assessment.speed_spread_mps <= p.formed_speed_spread_mps)
        details = dict(version=VERSION, readiness_count=self.readiness, span_m=assessment.initial_or_current_span_m,
            speed_spread_mps=assessment.speed_spread_mps, target_speed_mps=proposal.target_speed_mps,
            compact=compact, assembly_completed_at_s=self.ready_at, readiness_samples_required=p.readiness_samples,
            formed_span_m=p.formed_span_m, formed_speed_spread_mps=p.formed_speed_spread_mps,
            target_speed_tolerance_mps=p.target_speed_tolerance_mps, assessment=asdict(assessment),
            absolute_wall_min_mps2=p.acceleration_min_mps2, command_ramp_mps3=p.acceleration_change_limit_mps3,
            implementation_revision=IMPLEMENTATION_REVISION,
            experimental_authorization="registered_experimental_assignment_not_trained_benefit_gate")

        def emit(phase, reason, commands=None):
            if phase in {"refused", "abort", "release"}:
                self.terminal = phase, reason
            values = commands or {}
            details.update(readiness_count=self.readiness, assembly_completed_at_s=self.ready_at,
                initial_ramp_estimates=self.initial_estimates,
                override_released=not bool(values), continued_wall_ownership=bool(values))
            if values:
                self.previous_commands = dict(values)
            return PolicyOutput(values, phase, reason, proposal.member_ids, proposal.fingerprint,
                                assessment.evidence_fingerprint, details)

        if self.terminal:
            return emit(*self.terminal)
        failure_phase = "refused" if now == proposal.start_s else "abort"
        if not assessment.approved:
            return emit(failure_phase, "gentle_safety_or_information:"+assessment.reason)
        members = set(proposal.member_ids)
        if not members <= set(context.damping_commands) | set(context.damping_native_ids):
            return emit(failure_phase, "gentle_missing_current_local_damping_reference")
        rows = [frames[v] for v in proposal.member_ids]
        if self.initial_estimates is None:
            self.initial_estimates = {f.vehicle_id: dict(zip(("minimum_speed_adjustment_distance_m", "minimum_speed_adjustment_time_s"),
                ramped_braking_distance(f.speed_mps, proposal.target_speed_mps, f.acceleration_mps2, p))) for f in rows}
        if self.taper_at is not None and now >= self.taper_at+p.taper_s:
            return emit("release", "gentle_transition_deadline_return_to_identical_damping")
        settled = compact and all(abs(f.speed_mps-proposal.target_speed_mps) <= p.target_speed_tolerance_mps for f in rows)
        if self.ready_at is None:
            self.readiness = self.readiness+1 if settled else 0
            if self.readiness >= p.readiness_samples:
                self.ready_at, self.taper_at = now, now+p.hold_s
            elif now-proposal.start_s >= p.assembly_deadline_s:
                return emit("abort", "gentle_assembly_deadline_without_consecutive_readiness")
        phase = "assembly" if self.ready_at is None else "hold" if now < self.taper_at else "taper"
        if phase == "hold" and not compact:
            return emit("abort", "gentle_lost_compact_hold_or_speed_coherence")
        centre = mean(f.distance_to_merge_m for f in rows)
        target = proposal.target_speed_mps
        native = members & set(context.damping_native_ids)
        release_targets = None
        if phase == "taper":
            envelope = gentle_envelope(proposal, reports, now_s=now, previous_commands=self.previous_commands)
            details["dissolution"] = envelope
            if self.report_bridge_active and not envelope['available']:
                return emit("abort", "gentle_report_bridge_evidence_lost_return_to_identical_damping")
            if envelope["available"] and not envelope["gentle_stopping_feasible"]:
                return emit("abort", "gentle_reported_protective_braking_infeasible_return_to_identical_damping")
            if not native:
                if any(context.damping_commands[v] < p.acceleration_min_mps2 for v in members):
                    return emit("release", "gentle_protective_numeric_handover_return_to_identical_damping")
                release_targets = {v: context.damping_commands[v] for v in members}
            elif envelope["available"] and any(r["downstream_speed_mps"] < target for r in envelope["constraints"]):
                # Same acquired-report release concept, gentle and ramp feasible.
                target = min(target, min(r["downstream_speed_mps"] for r in envelope["constraints"]))
                self.report_bridge_active = True
                details["dissolution"]["mode"] = "report_bridge_gentle_deceleration"
            else:
                # Explicit short WALL coasting transition, never an IDM reference.
                release_targets = {v: 0. for v in members}
                details["dissolution"]["mode"] = "wall_coast_transition_before_native_handover"
            if release_targets is not None and self.previous_commands and all(
                    abs(self.previous_commands[v]-release_targets[v]) < 1e-8
                    and abs(frames[v].acceleration_mps2-release_targets[v]) <= p.acceleration_change_limit_mps3+1e-8
                    for v in members):
                return emit("release", "gentle_transition_aligned_return_to_identical_damping")
        commands, resolutions = {}, {}
        for f in rows:
            offset = max(-p.maximum_coordination_offset_mps, min(p.maximum_coordination_offset_mps,
                p.position_gain_per_s*(f.distance_to_merge_m-centre)))
            desired = max(0., min(f.allowed_speed_mps, target+offset))
            requested = release_targets[f.vehicle_id] if release_targets is not None else p.speed_gain_per_s*(desired-f.speed_mps)
            acceleration, evidence = resolve_command(f, requested, p, previous=self.previous_commands.get(f.vehicle_id))
            resolutions[f.vehicle_id] = dict(evidence, desired_speed_mps=desired)
            details["command_resolution"] = resolutions
            if acceleration is None:
                return emit(failure_phase, "gentle_smooth_entry_or_response_incompatible_return_to_identical_damping")
            problem = _local_clearance_problem(f, p, now, acceleration=acceleration)
            if problem:
                return emit(failure_phase, "gentle_protective_clearance_abort:"+problem[1])
            front = gentle_front_clearance(f, acceleration, p, now_s=now)
            resolutions[f.vehicle_id]['ramped_front_clearance'] = front
            if front is not None and not front['gentle_feasible']:
                return emit(failure_phase, "gentle_ramped_front_braking_infeasible_return_to_identical_damping")
            commands[f.vehicle_id] = acceleration
        return emit(phase, "gentle_report_only_relative_position_speed_and_release", commands)


def policy_factory(job, registration):
    return GentleWallPolicy(job, registration)
