"""One experimental, report-only bridge across native handback's sensing gap.

This keeps the frozen CIRCLES controller and original formation law intact.
Continued transition commands are explicitly wall-owned, never called native.
"""
from dataclasses import asdict
from math import sqrt, isfinite
from statistics import mean

from .formation import FormationProfile, make_proposal, assess_formation, _local_clearance_problem
from .paired_experiments import PolicyOutput, fixed_formation_policy, digest


VERSION = "rsu_reported_dissolution/1"
COMFORT_DECEL_MPS2 = 1.5
PASSENGER_LENGTH_ALLOWANCE_M = 5.


def downstream_envelope(proposal, reports, *, now_s):
    """Same-lane acquired CAV self states; a conditional stopping-speed envelope."""
    p = proposal.profile
    packets = list(reports)
    if any(r.sent_at_s > now_s or r.frame.timestamp_s > now_s for r in packets):
        raise ValueError("future report cannot determine dissolution")
    if len({r.frame.vehicle_id for r in packets}) != len(packets):
        raise ValueError("one latest report per CAV required")
    fresh = {r.frame.vehicle_id: r.frame for r in packets if 0 <= now_s-r.frame.timestamp_s <= p.max_report_age_s}
    members = [fresh.get(v) for v in proposal.member_ids]
    if any(f is None or f.position_m is None for f in members):
        return dict(available=False, target_speed_mps=None, constraints=[], reason="missing_member_localization")
    constraints = []
    for ego in members:
        for other in fresh.values():
            if (other.vehicle_id in proposal.member_ids or other.lane_id != ego.lane_id
                    or other.position_m is None or other.position_m <= ego.position_m):
                continue
            age = max(now_s-ego.timestamp_s, now_s-other.timestamp_s)
            separation = other.position_m-ego.position_m
            reserve = PASSENGER_LENGTH_ALLOWANCE_M+p.minimum_front_gap_m+ego.speed_mps*(p.control_dt_s+age)
            distance = max(0., separation-reserve)
            envelope = sqrt(other.speed_mps**2+2*COMFORT_DECEL_MPS2*distance)
            constraints.append(dict(member_id=ego.vehicle_id, downstream_reporter_id=other.vehicle_id,
                separation_m=separation, observation_age_s=age, downstream_speed_mps=other.speed_mps,
                available_distance_m=distance, speed_envelope_mps=envelope))
    return dict(available=bool(constraints),
        target_speed_mps=min((r["speed_envelope_mps"] for r in constraints), default=None),
        constraints=constraints, reason="acquired_same_lane_cav_self_reports",
        comfortable_deceleration_mps2=COMFORT_DECEL_MPS2,
        passenger_length_allowance_m=PASSENGER_LENGTH_ALLOWANCE_M,
        constant_downstream_speed_assumption=True, safety_guarantee=False)


class ReportDissolutionPolicy:
    def __init__(self, job, registration):
        self.job, self.base = job, fixed_formation_policy(job, registration)
        self.proposal, self.started_at, self.terminal = None, None, None

    def __call__(self, context):
        baseline = self.base(context)
        if self.job["arm"] != "wall":
            return baseline
        if context.now_s == self.job["decision_time_s"] and baseline.phase != "refused":
            spec = self.job["policy_spec"]
            self.proposal = make_proposal(context.reports, now_s=context.now_s,
                required_lanes=tuple(spec["required_lanes"]), member_ids=tuple(spec["member_ids"]),
                profile=FormationProfile(**spec["profile"]))
            if self.proposal.fingerprint != baseline.proposal_sha256:
                raise ValueError("dissolution must bind the identical concrete formation proposal")
        if self.proposal is None:
            return baseline
        proposal, now, p = self.proposal, context.now_s, self.proposal.profile

        def emit(phase, reason, commands=None, evidence=None):
            if phase in {"release", "abort"}:
                self.terminal = (phase, reason)
            return PolicyOutput(commands or {}, phase, reason, proposal.member_ids,
                proposal.fingerprint, digest(dict(now_s=now, reports=[r.payload() for r in context.reports])),
                dict(baseline.details, dissolution=dict(version=VERSION, started_at_s=self.started_at,
                     continued_wall_ownership=bool(commands), override_released=not bool(commands),
                     envelope=evidence, hypothesis_not_qualified_gate=True)))

        if self.terminal is not None:
            return emit(*self.terminal)
        if self.started_at is None:
            if baseline.reason != "identical_damping_requests_native_explicit_handback":
                return baseline
            evidence = downstream_envelope(proposal, context.reports, now_s=now)
            if not evidence["available"] or evidence["target_speed_mps"] >= proposal.target_speed_mps:
                # No relevant reported constraint: preserve the original release.
                self.terminal = (baseline.phase, baseline.reason)
                return baseline
            self.started_at = now
        if now >= self.started_at+p.taper_s:
            return emit("release", "report_bridge_timeout_return_to_identical_damping")
        members = set(proposal.member_ids)
        known = set(context.damping_commands) | set(context.damping_native_ids)
        if not members <= known:
            return emit("abort", "report_bridge_missing_current_local_reference")
        if members <= set(context.damping_commands):
            return emit("release", "report_bridge_local_numeric_feedback_available")
        assessment = assess_formation(proposal, context.reports, now_s=now)
        if not assessment.approved:
            return emit("abort", "report_bridge_"+assessment.reason)
        evidence = downstream_envelope(proposal, context.reports, now_s=now)
        if not evidence["available"]:
            return emit("release", "report_bridge_evidence_unavailable", evidence=evidence)
        target = min(proposal.target_speed_mps, evidence["target_speed_mps"])
        frames = {r.frame.vehicle_id: r.frame for r in context.reports}
        centre = mean(frames[v].distance_to_merge_m for v in members)
        commands = {}
        for vid in sorted(members):
            f = frames[vid]
            offset = max(-p.maximum_coordination_offset_mps, min(p.maximum_coordination_offset_mps,
                p.position_gain_per_s*(f.distance_to_merge_m-centre)))
            desired = max(0., min(f.allowed_speed_mps, target+offset))
            acceleration = max(p.acceleration_min_mps2, min(p.acceleration_max_mps2,
                p.speed_gain_per_s*(desired-f.speed_mps)))
            if not isfinite(acceleration):
                raise ValueError("nonfinite report-derived wall transition")
            problem = _local_clearance_problem(f, p, now, acceleration=acceleration)
            if problem:
                return emit("abort", "report_bridge_"+problem[1], evidence=evidence)
            commands[vid] = acceleration
        evidence.update(common_transition_target_mps=target, assessment=asdict(assessment))
        return emit("taper", "report_bridge_to_local_feedback", commands, evidence)


def policy_factory(job, registration):
    if job["policy_spec"]["profile"]["version"] != VERSION:
        raise ValueError("report dissolution requires its versioned proposal")
    return ReportDissolutionPolicy(job, registration)
