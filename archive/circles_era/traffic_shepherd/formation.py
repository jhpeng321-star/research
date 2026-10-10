"""Report-only, longitudinal single-wall formation and conditional safety checks.

The controller assembles dispersed CAVs by correcting relative position AND
speed. It emits absolute acceleration requests for CommandOwnership, and never
talks to a simulator. Conditional eligibility is not physical validation: local
reports cannot certify the future behaviour of an unobserved vehicle.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from math import isfinite, log
from statistics import mean
from typing import Iterable, Mapping

from traffic_shepherd.reporting import CAVReport
from traffic_shepherd.wall import WallMember


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _clip(value, lo, hi):
    return min(hi, max(lo, value))


@dataclass(frozen=True)
class FormationProfile:
    """One development treatment; parameters are frozen into every proposal.

    Position gain [1/s] maps centre-relative position [m] to speed offset [m/s].
    Speed gain [1/s] maps desired-minus-measured speed [m/s] to acceleration.
    The 3 m/s offset is a coordination velocity, not an acceleration projection.
    50 m defines a formed wall; it is never an initial admission span cutoff.
    """
    version: str = "report_longitudinal_formation/1"
    control_dt_s: float = 1.0
    assembly_deadline_s: float = 50.0
    hold_s: float = 60.0
    taper_s: float = 10.0
    position_gain_per_s: float = 0.12
    speed_gain_per_s: float = 0.6
    maximum_coordination_offset_mps: float = 3.0
    acceleration_min_mps2: float = -3.0
    acceleration_max_mps2: float = 1.5
    formed_span_m: float = 50.0
    formed_speed_spread_mps: float = 1.0
    target_speed_tolerance_mps: float = 1.0
    readiness_samples: int = 3
    target_speed_reduction_mps: float = 2.0
    maximum_target_speed_mps: float = 24.0
    max_report_age_s: float = 1.0
    minimum_front_gap_m: float = 5.0
    minimum_rear_gap_m: float = 5.0
    minimum_front_ttc_s: float = 3.0
    minimum_rear_ttc_s: float = 3.0
    minimum_runway_m: float = 50.0

    def __post_init__(self):
        values = asdict(self)
        if not self.version or not all(isfinite(v) for k, v in values.items() if k != "version"):
            raise ValueError("named profile and finite numeric parameters required")
        if self.control_dt_s != 1 or self.acceleration_min_mps2 != -3 or self.acceleration_max_mps2 != 1.5:
            raise ValueError("formation preserves adopted 1 s clock and [-3,1.5] absolute envelope")
        if type(self.readiness_samples) is not int or self.readiness_samples < 2:
            raise ValueError("readiness requires at least two consecutive measured samples")
        if min(self.assembly_deadline_s, self.hold_s, self.taper_s,
               self.position_gain_per_s, self.speed_gain_per_s,
               self.maximum_coordination_offset_mps, self.formed_span_m,
               self.maximum_target_speed_mps) <= 0:
            raise ValueError("positive formation durations, gains, span and speed bounds required")
        nonnegative = (self.formed_speed_spread_mps, self.target_speed_tolerance_mps,
                       self.target_speed_reduction_mps, self.max_report_age_s,
                       self.minimum_front_gap_m, self.minimum_rear_gap_m,
                       self.minimum_front_ttc_s, self.minimum_rear_ttc_s, self.minimum_runway_m)
        if min(nonnegative) < 0:
            raise ValueError("negative observation, readiness or safety threshold")
        if self.max_report_age_s > self.control_dt_s:
            raise ValueError("formation evidence may be at most one control interval old")
        for duration in (self.assembly_deadline_s, self.hold_s, self.taper_s):
            if duration % self.control_dt_s:
                raise ValueError("phase times must align with the adopted control clock")


@dataclass(frozen=True)
class FormationProposal:
    start_s: float
    required_lanes: tuple[str, ...]
    members: tuple[WallMember, ...]
    target_speed_mps: float
    profile: FormationProfile

    def __post_init__(self):
        if not isfinite(self.start_s) or self.start_s < 0 or self.start_s % self.profile.control_dt_s:
            raise ValueError("proposal start must lie on the control clock")
        if not isinstance(self.required_lanes, tuple) or not isinstance(self.members, tuple):
            raise ValueError("required lanes and membership must be immutable tuples")
        if not self.required_lanes or any(not x for x in self.required_lanes) or len(set(self.required_lanes)) != len(self.required_lanes):
            raise ValueError("explicit distinct map lanes required")
        if any(not isinstance(m, WallMember) for m in self.members):
            raise TypeError("typed CAV wall members required")
        if len(set(m.vehicle_id for m in self.members)) != len(self.members) or len(set(m.lane_id for m in self.members)) != len(self.members):
            raise ValueError("one distinct CAV per proposed lane")
        if any(m.lane_id not in self.required_lanes for m in self.members):
            raise ValueError("member lane outside required map lanes")
        if not isfinite(self.target_speed_mps) or not 0 <= self.target_speed_mps <= self.profile.maximum_target_speed_mps:
            raise ValueError("invalid frozen wall target speed")

    @property
    def fingerprint(self):
        return _digest(asdict(self))

    @property
    def member_ids(self):
        return tuple(m.vehicle_id for m in self.members)


def _packets(reports, now_s):
    if not isfinite(now_s) or now_s < 0:
        raise ValueError("invalid report evaluation clock")
    packets = list(reports)
    if any(not isinstance(report, CAVReport) for report in packets):
        raise TypeError("only received CAVReport packets are admitted")
    if len(set(r.frame.vehicle_id for r in packets)) != len(packets):
        raise ValueError("one latest received report per CAV is required")
    return {r.frame.vehicle_id: r for r in packets}


def make_proposal(reports: Iterable[CAVReport], *, now_s: float,
                  required_lanes: tuple[str, ...], member_ids: tuple[str, ...],
                  profile: FormationProfile | None = None) -> FormationProposal:
    """Freeze a mean-reported-speed-minus-2 target, without a congestion filter.

    Required lane IDs are supplied from the road map. Ordering member_ids in the
    same lane order avoids silently inferring required lanes from missing reports.
    Feasibility must still be checked; this function never authorizes activation.
    """
    p = profile or FormationProfile()
    by_id = _packets(reports, now_s)
    if len(member_ids) != len(required_lanes) or len(set(member_ids)) != len(member_ids):
        raise ValueError("one distinct member must be proposed for every map lane")
    if any(v not in by_id for v in member_ids):
        raise ValueError("cannot derive a target from missing member reports")
    frames = [by_id[v].frame for v in member_ids]
    target = min(p.maximum_target_speed_mps, min(f.allowed_speed_mps for f in frames),
                 max(0., mean(f.speed_mps for f in frames) - p.target_speed_reduction_mps))
    return FormationProposal(now_s, tuple(required_lanes),
        tuple(WallMember(v, lane) for v, lane in zip(member_ids, required_lanes)), float(target), p)


@dataclass(frozen=True)
class FormationAssessment:
    proposal_fingerprint: str
    evidence_fingerprint: str
    now_s: float
    status: str
    reason: str
    initial_or_current_span_m: float | None
    speed_spread_mps: float | None
    nearest_boundary_m: float | None
    estimated_minimum_closing_time_s: float | None
    estimated_assembly_s: float | None
    estimated_delivery_runway_m: float | None
    estimated_runway_slack_m: float | None
    worst_duration_runway_m: float | None
    # False: these local tests do not prove successful future formation.
    physically_validated: bool = False

    @property
    def approved(self):
        return self.status == "feasible"


def _local_clearance_problem(frame, profile, now_s, *, acceleration=None):
    """Current clearance plus one-command constant-neighbour-speed check.

    This is a refusal/abort check, not the removed tight damping projection. It
    never rewrites the CIRCLES command. SUMO speedMode31 remains in force.
    Acceleration of a neighbour after sensing is unknown, so this is explicitly
    a conditional guard rather than an invariant safety proof.
    """
    age = now_s - frame.timestamp_s
    for label, neighbor, coverage, gap_min, ttc_min in (
        ("front", frame.front, frame.front_coverage_m, profile.minimum_front_gap_m, profile.minimum_front_ttc_s),
        ("rear", frame.rear, frame.rear_coverage_m, profile.minimum_rear_gap_m, profile.minimum_rear_ttc_s),
    ):
        if coverage < gap_min or coverage <= 0:
            return "insufficient_information", f"{label}_coverage_unknown"
        if neighbor is None:
            continue
        sign = -1 if label == "front" else 1
        closing = sign * neighbor.relative_speed_mps
        age_gap = neighbor.gap_m - max(0., closing) * age
        if age_gap < gap_min or (closing > 0 and age_gap / closing < ttc_min):
            return "unsafe_geometry", f"{label}_clearance_or_ttc"
        if acceleration is not None:
            dt = profile.control_dt_s
            # Front closes under +ego acceleration; rear closes under -ego acceleration.
            effective = acceleration if label == "front" else -acceleration
            projected_gap = age_gap - closing * dt - .5 * effective * dt**2
            projected_closing = closing + effective * dt
            if projected_gap < gap_min or (projected_closing > 0 and projected_gap / projected_closing < ttc_min):
                return "unsafe_geometry", f"{label}_next_interval_clearance_or_ttc"
    return None


def assess_formation(proposal: FormationProposal, reports: Iterable[CAVReport], *, now_s: float) -> FormationAssessment:
    """Report-conditioned admission; initial span is measured, never thresholded.

    A finite one-tick runway check protects the declared map release boundary.
    Worst-duration runway is reported, not interpreted as a guaranteed required
    distance: assembly can finish early. Insufficient actual runway aborts and
    counts as failed delivery; eligibility alone is not a feasibility label.
    """
    by_id = _packets(reports, now_s)
    p = proposal.profile
    packets = [by_id[m.vehicle_id] for m in proposal.members if m.vehicle_id in by_id]
    digest = _digest(dict(now_s=now_s, reports=[r.payload() for r in packets]))
    frames = [r.frame for r in packets]
    distances = [f.distance_to_merge_m for f in frames if f.distance_to_merge_m is not None]
    speeds = [f.speed_mps for f in frames]
    span = max(distances) - min(distances) if len(distances) == len(proposal.members) and distances else None
    spread = max(speeds) - min(speeds) if len(speeds) == len(proposal.members) and speeds else None
    nearest = min(distances) if distances else None
    min_closing = max(0., span - p.formed_span_m) / (2 * p.maximum_coordination_offset_mps) if span is not None else None
    worst_runway = p.minimum_runway_m + (proposal.target_speed_mps + p.maximum_coordination_offset_mps) * (p.assembly_deadline_s + p.hold_s + p.taper_s)
    estimated_assembly = estimated_runway = estimated_slack = None
    if span is not None and speeds:
        # Screening estimate from this proportional law, not a simulated future or
        # physical guarantee. The saturated two-member closure rate is 2*offset;
        # the unsaturated rate is approximately k_position*span. A response-lag
        # allowance and three-sample readiness follow. It is logged even when it
        # predicts failure; the outcome label comes from the actual trajectory.
        saturated_span = 2 * p.maximum_coordination_offset_mps / p.position_gain_per_s
        readiness_span = max(1e-6, min(p.formed_span_m,
            p.formed_speed_spread_mps / p.position_gain_per_s,
            2 * p.target_speed_tolerance_mps / p.position_gain_per_s))
        saturated_s = max(0., span - saturated_span) / (2 * p.maximum_coordination_offset_mps)
        proportional_s = max(0., log(max(readiness_span, min(span, saturated_span)) / readiness_span)) / p.position_gain_per_s
        speed_settle_s = max(abs(v - proposal.target_speed_mps) for v in speeds) / min(-p.acceleration_min_mps2, p.acceleration_max_mps2)
        estimated_assembly = max(saturated_s + proportional_s, speed_settle_s) + 2 / p.speed_gain_per_s + (p.readiness_samples - 1) * p.control_dt_s
        estimated_runway = p.minimum_runway_m + proposal.target_speed_mps * (estimated_assembly + p.hold_s + p.taper_s)
        estimated_runway += max(0., max(speeds) - proposal.target_speed_mps) * estimated_assembly
        estimated_slack = nearest - estimated_runway

    def result(status, reason):
        return FormationAssessment(proposal.fingerprint, digest, now_s, status, reason,
                                   span, spread, nearest, min_closing, estimated_assembly,
                                   estimated_runway, estimated_slack, worst_runway)

    if {m.lane_id for m in proposal.members} != set(proposal.required_lanes):
        return result("insufficient_members", "one_cav_required_per_map_lane")
    if len(packets) != len(proposal.members):
        return result("insufficient_information", "missing_member_report")
    for member, packet in zip(proposal.members, packets):
        frame = packet.frame
        if packet.sent_at_s > now_s or not 0 <= now_s - frame.timestamp_s <= p.max_report_age_s:
            return result("insufficient_information", "stale_or_future_member_report")
        if frame.lane_id is None or frame.distance_to_merge_m is None:
            return result("insufficient_information", "missing_map_localization")
        if frame.lane_id != member.lane_id:
            return result("unsafe_geometry", "member_lane_loss")
        if proposal.target_speed_mps > frame.allowed_speed_mps:
            return result("unsafe_geometry", "target_above_allowed_speed")
        one_tick = frame.speed_mps * (now_s - frame.timestamp_s + p.control_dt_s) + .5 * p.acceleration_max_mps2 * p.control_dt_s**2
        if frame.distance_to_merge_m < p.minimum_runway_m + one_tick:
            return result("insufficient_runway", "release_boundary_before_next_control_tick")
        problem = _local_clearance_problem(frame, p, now_s)
        if problem:
            return result(*problem)
    return result("feasible", "conditional_report_geometry_only_delivery_not_yet_measured")


@dataclass(frozen=True)
class FormationAuthorization:
    proposal_fingerprint: str
    evidence_fingerprint: str
    evidence_time_s: float
    source: str
    registration_id: str
    arm: str
    qualification_sha256: str | None = None
    feature_row_sha256: str | None = None
    report_history_sha256: str | None = None
    feasibility_artifact_sha256: str | None = None
    benefit_artifact_sha256: str | None = None


_QUALIFIED_AUTHORITY = object()


class _QualifiedFormationAuthorization(FormationAuthorization):
    """In-process authorization capability minted only after artifact evaluation."""
    def __init__(self, *args, _authority=None, **kwargs):
        if _authority is not _QUALIFIED_AUTHORITY:
            raise ValueError("qualified authorization must be minted by both verified gates")
        super().__init__(*args, **kwargs)


def authorize_experiment(proposal: FormationProposal, assessment: FormationAssessment, *,
                         registration_id: str, arm: str = "wall") -> FormationAuthorization:
    """A registered treatment assignment, never a manually invented benefit gate."""
    if not registration_id.strip() or arm not in {"wall", "assembly_only"}:
        raise ValueError("explicit experimental registration and treatment arm required")
    if assessment.proposal_fingerprint != proposal.fingerprint or assessment.now_s != proposal.start_s:
        raise ValueError("authorization must bind this proposal and exact decision-time evidence")
    if not assessment.approved:
        raise ValueError("experimental assignment cannot bypass observed feasibility refusal")
    return FormationAuthorization(proposal.fingerprint, assessment.evidence_fingerprint,
                                  assessment.now_s, "registered_experimental_assignment", registration_id, arm)


def authorize_qualified_gates(proposal, assessment, *, packet, qualification):
    """Separate policy authorization: recompute both frozen artifact decisions.

There is no scalar manual-benefit argument. Loading a qualified bundle requires
physical, report/label, split/calibration and frozen-policy evidence. Current
proposal, member packets, history and feature row must agree exactly. Existing
experimental assignment remains a distinct provenance path.
"""
    from .escalation_policy import QualifiedGateBundle
    if not isinstance(qualification, QualifiedGateBundle):
        raise TypeError("a verified qualified-gate bundle is required")
    decisions = qualification.decisions(proposal, assessment, packet)
    if not assessment.approved or not all(d.approved and not d.abstained for d in decisions):
        raise ValueError("both qualified gates must approve this exact current proposal")
    common = (proposal.fingerprint, packet["evidence_sha256"], packet["row_sha256"],
              assessment.evidence_fingerprint, proposal.start_s, qualification.sha256)
    if any((d.proposal_sha256, d.evidence_sha256, d.feature_row_sha256,
            d.assessment_evidence_sha256, d.decision_time_s, d.qualification_sha256) != common for d in decisions):
        raise ValueError("gate approvals do not bind the same proposal and evidence")
    return _QualifiedFormationAuthorization(proposal.fingerprint, assessment.evidence_fingerprint, proposal.start_s,
        "qualified_gate_artifacts", qualification.sha256, "wall", qualification.sha256,
        packet["row_sha256"], packet["evidence_sha256"], decisions[0].artifact_sha256, decisions[1].artifact_sha256,
        _authority=_QUALIFIED_AUTHORITY)


@dataclass(frozen=True)
class FormationStep:
    phase: str
    acceleration_items: tuple[tuple[str, float], ...]
    reason: str
    member_ids: tuple[str, ...]
    proposal_fingerprint: str
    evidence_fingerprint: str
    readiness_count: int
    span_m: float | None
    speed_spread_mps: float | None
    target_speed_mps: float
    compact: bool
    assembly_completed_at_s: float | None
    desired_speed_items: tuple[tuple[str, float], ...] = ()

    @property
    def commands(self):
        """Empty map returns to the identical damping/native-fallback owner."""
        return dict(self.acceleration_items)

    def as_dict(self):
        return asdict(self)


class FormationController:
    """One fixed cohort; no lane changes, substitutions, simulator or global input.

    The current centre moves with the cohort; a member farther upstream gets a
    positive relative-speed correction. At readiness the clock starts HOLD.
    Safety/member/report loss aborts the whole wall atomically. HOLD must remain
    compact, and taper blends toward the identical damping instance's outputs.
    """
    def __init__(self, proposal: FormationProposal, authorization: FormationAuthorization):
        if authorization.proposal_fingerprint != proposal.fingerprint or authorization.evidence_time_s != proposal.start_s:
            raise ValueError("authorization does not bind this formation proposal")
        if authorization.source == "registered_experimental_assignment":
            if authorization.arm not in {"wall", "assembly_only"}:
                raise ValueError("registered experimental arm required")
        elif authorization.source == "qualified_gate_artifacts":
            if not isinstance(authorization, _QualifiedFormationAuthorization):
                raise ValueError("qualified authorization must come from the verified gate factory")
            proofs = (authorization.qualification_sha256, authorization.feature_row_sha256,
                      authorization.report_history_sha256, authorization.feasibility_artifact_sha256,
                      authorization.benefit_artifact_sha256)
            if authorization.arm != "wall" or authorization.registration_id != authorization.qualification_sha256:
                raise ValueError("qualified authorization must name its exact frozen qualification")
            if any(not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v) for v in proofs):
                raise ValueError("qualified authorization requires complete artifact and evidence fingerprints")
        else:
            raise ValueError("explicit registered experiment or qualified-gate authorization required")
        self.proposal = proposal
        self.authorization = authorization
        self._last_time = None
        self._terminal = None
        self._readiness = 0
        self._ready_at = None
        self._taper_at = None

    def step(self, reports: Iterable[CAVReport], *, now_s: float,
             damping_commands: Mapping[str, float], damping_native_ids=()) -> FormationStep:
        p = self.proposal.profile
        if not isfinite(now_s) or (self._last_time is None and now_s != self.proposal.start_s):
            raise ValueError("formation must begin at its registered start")
        if self._last_time is not None and now_s != self._last_time + p.control_dt_s:
            raise ValueError("formation clock must advance exactly one control interval")
        first = self._last_time is None
        damping_native = frozenset(damping_native_ids)
        if not damping_native <= set(self.proposal.member_ids) or damping_native & set(damping_commands):
            raise ValueError("native damping references must name members and be disjoint from numeric references")
        self._last_time = now_s
        packets = list(reports)
        assessment = assess_formation(self.proposal, packets, now_s=now_s)
        frames = {r.frame.vehicle_id: r.frame for r in packets}
        compact = bool(assessment.approved and assessment.initial_or_current_span_m <= p.formed_span_m
                       and assessment.speed_spread_mps <= p.formed_speed_spread_mps)

        def result(phase, reason, commands=(), desired=()):
            if phase in {"release", "abort"}:
                self._terminal = (phase, reason)
            return FormationStep(phase, tuple(commands), reason, self.proposal.member_ids,
                self.proposal.fingerprint, assessment.evidence_fingerprint, self._readiness,
                assessment.initial_or_current_span_m, assessment.speed_spread_mps,
                self.proposal.target_speed_mps, compact, self._ready_at, tuple(desired))

        if self._terminal:
            return result(*self._terminal)
        if first and assessment.evidence_fingerprint != self.authorization.evidence_fingerprint:
            return result("abort", "activation_evidence_changed_requires_fresh_authorization")
        if not assessment.approved:
            return result("abort", f"{assessment.status}:{assessment.reason}")
        # A cached RSU packet is not proof that a CAV can execute this tick.
        # These references come from the CAV's current local damping decision,
        # including an explicit native fallback, never a simulator-global census.
        # Lose any current local reference: atomically relinquish the whole wall
        # before CommandOwnership validates or dispatches the cohort mask.
        known = set(damping_commands) | damping_native
        if not set(self.proposal.member_ids) <= known or any(
            not isfinite(damping_commands[v]) for v in self.proposal.member_ids if v in damping_commands
        ):
            return result("abort", "missing_current_local_damping_reference")
        if self._taper_at is not None and now_s >= self._taper_at + p.taper_s:
            return result("release", "declared_taper_complete_return_to_identical_damping")

        rows = [frames[v] for v in self.proposal.member_ids]
        settled = compact and all(abs(f.speed_mps - self.proposal.target_speed_mps) <= p.target_speed_tolerance_mps for f in rows)
        if self._ready_at is None:
            self._readiness = self._readiness + 1 if settled else 0
            if self._readiness >= p.readiness_samples:
                self._ready_at = now_s
                self._taper_at = now_s if self.authorization.arm == "assembly_only" else now_s + p.hold_s
            elif now_s - self.proposal.start_s >= p.assembly_deadline_s:
                return result("abort", "assembly_deadline_without_consecutive_readiness")
        phase = "assembly" if self._ready_at is None else "hold" if now_s < self._taper_at else "taper"
        if phase == "hold" and not compact:
            return result("abort", "lost_compact_hold_or_speed_coherence")
        if phase == "taper" and damping_native:
            # Native IDM has no acquirable numerical acceleration reference.
            # Hand back the whole cohort instead of inventing a zero command or
            # a privileged counterfactual acceleration. Each member then uses
            # the identical damping owner's actual native/numeric decision.
            known = set(damping_commands) | damping_native
            if not set(self.proposal.member_ids) <= known:
                return result("abort", "missing_identical_damping_reference_for_taper")
            return result("release", "identical_damping_requests_native_explicit_handback")
        centre = mean(f.distance_to_merge_m for f in rows)
        commands, desired = [], []
        for frame in rows:
            offset = _clip(p.position_gain_per_s * (frame.distance_to_merge_m - centre),
                           -p.maximum_coordination_offset_mps, p.maximum_coordination_offset_mps)
            target = _clip(self.proposal.target_speed_mps + offset, 0., frame.allowed_speed_mps)
            acceleration = _clip(p.speed_gain_per_s * (target - frame.speed_mps),
                                 p.acceleration_min_mps2, p.acceleration_max_mps2)
            if phase == "taper":
                damping = damping_commands.get(frame.vehicle_id)
                if damping is None or not isfinite(damping):
                    return result("abort", "missing_identical_damping_reference_for_taper")
                fraction = (now_s - self._taper_at + p.control_dt_s) / p.taper_s
                acceleration = _clip((1 - fraction) * acceleration + fraction * damping,
                                     p.acceleration_min_mps2, p.acceleration_max_mps2)
            problem = _local_clearance_problem(frame, p, now_s, acceleration=acceleration)
            if problem:
                return result("abort", f"{problem[0]}:{problem[1]}")
            commands.append((frame.vehicle_id, float(acceleration)))
            desired.append((frame.vehicle_id, float(target)))
        return result(phase, "report_only_relative_position_and_speed_coordination", commands, desired)
