"""One report-only recovery law; conditional planning is not a safety guarantee."""
from dataclasses import dataclass, replace
from statistics import mean

from .formation import _local_clearance_problem, assess_formation
from .gentle_wall import gentle_front_clearance, resolve_command as gentle_resolve
from .lifecycle_wall import coast_terminal
from .report_dissolution import downstream_envelope

VERSION = "formation_recovery_release/2"


def positive_speed_rise(acceleration, ramp, dt):
    """m/s gained while tapering this command to zero at the allowed slew."""
    change, a = 0., max(0., acceleration)
    for _ in range(100):
        if a <= 1e-12:
            return change
        change += a*dt
        a = max(0., a-ramp*dt)
    raise ValueError("positive acceleration taper exceeded finite envelope")


def resolve_command(frame, requested, profile, *, previous=None):
    """Recovery-only anticipation of the speed limit, preserving gentle bounds.

    One-step speed clipping can strand positive acceleration at the limit,
    making next tick's slew and speed constraints incompatible. Reserve the
    full speed rise needed to taper positive acceleration to zero. This does
    not alter assembly/HOLD, native IDM or the frozen damping controller.
    """
    command, evidence = gentle_resolve(frame, requested, profile, previous=previous)
    if command is None:
        return command, evidence
    headroom = max(0., frame.allowed_speed_mps-frame.speed_mps)
    ramp, dt = profile.acceleration_change_limit_mps3, profile.control_dt_s
    rise = lambda a: positive_speed_rise(a, ramp, dt)  # noqa: E731
    if rise(command) <= headroom+1e-10:
        return command, dict(evidence, positive_taper_speed_rise_mps=rise(command), speed_headroom_mps=headroom)
    lo, hi = evidence['admissible_min_mps2'], command
    if rise(lo) > headroom+1e-10:
        return None, dict(evidence, reason='positive_taper_speed_headroom_incompatible')
    for _ in range(50):
        mid = (lo+hi)/2
        if rise(mid) <= headroom:
            lo = mid
        else:
            hi = mid
    return lo, dict(evidence, resolved_wall_acceleration_mps2=lo,
        positive_taper_speed_rise_mps=rise(lo), speed_headroom_mps=headroom,
        anticipatory_positive_acceleration_taper=True)


def eligible_packets(context):
    """Filter BOTH clocks before deterministic latest-per-ID selection."""
    latest = {}
    for receipt, report in context.history:
        f = report.frame
        if not (f.timestamp_s <= report.sent_at_s <= receipt <= context.now_s
                and 0 <= context.now_s - f.timestamp_s <= 1
                and 0 <= context.now_s - receipt <= 1):
            continue
        key = (f.timestamp_s, receipt, report.sent_at_s)
        old = latest.get(f.vehicle_id)
        if old is None or key > old[0]:
            latest[f.vehicle_id] = (key, report)
        elif key == old[0] and old[1] != report:
            raise ValueError("conflicting duplicate acquired packet")
    return tuple(latest[v][1] for v in sorted(latest))


def estimate_downstream_reference(proposal, context):
    reports = eligible_packets(context)
    frames = {r.frame.vehicle_id: r.frame for r in reports}
    members = [frames.get(m.vehicle_id) for m in proposal.members]
    if any(f is None or f.distance_to_merge_m is None for f in members):
        return dict(target_speed_mps=None, per_lane={}, details={}, source_count=0)
    leading = min(f.distance_to_merge_m for f in members)
    result, details = {}, {}
    for member, mf in zip(proposal.members, members):
        candidates = sorted((f for f in frames.values()
            if f.vehicle_id not in proposal.member_ids and f.lane_id == member.lane_id
            and f.distance_to_merge_m is not None
            and 20 <= leading - f.distance_to_merge_m <= 300),
            key=lambda f: (-f.distance_to_merge_m, f.vehicle_id))[:3]
        if candidates:
            result[member.lane_id] = mean(f.speed_mps for f in candidates)
            details[member.lane_id] = dict(mode="self", source_ids=[f.vehicle_id for f in candidates])
        elif mf.front is not None and 20 <= mf.front.gap_m <= mf.front_coverage_m:
            result[member.lane_id] = max(0., mf.speed_mps + mf.front.relative_speed_mps)
            details[member.lane_id] = dict(mode="local_front_fallback", source_ids=[mf.vehicle_id])
        else:
            details[member.lane_id] = dict(mode="missing", source_ids=[])
    target = min(*result.values(), *(f.allowed_speed_mps for f in members)) if len(result) == len(members) else None
    return dict(target_speed_mps=target, per_lane=result, details=details,
                source_count=sum(len(d["source_ids"]) for d in details.values()))


def compact(rows, p):
    return (max(f.distance_to_merge_m for f in rows)-min(f.distance_to_merge_m for f in rows) <= p.formed_span_m
            and max(f.speed_mps for f in rows)-min(f.speed_mps for f in rows) <= p.formed_speed_spread_mps)


def recovery_requests(rows, target, reference, p, *, coast=False):
    # Once the unslewed reference is reached, explicitly request zero to permit
    # a finite measured coast endpoint, rather than asymptotic tiny commands.
    settle = abs(target-reference) <= 1e-8 and all(abs(f.speed_mps-reference) <= .5 for f in rows)
    centre = mean(f.distance_to_merge_m for f in rows)
    values = {}
    for f in rows:
        offset = max(-p.maximum_coordination_offset_mps, min(p.maximum_coordination_offset_mps,
            p.position_gain_per_s*(f.distance_to_merge_m-centre)))
        desired = max(0., min(f.allowed_speed_mps, target+offset))
        values[f.vehicle_id] = 0. if coast or settle else p.speed_gain_per_s*(desired-f.speed_mps)
    return values


def recovery_plan(proposal, reports, *, now_s, previous, target, reference, remaining_s):
    """Full remaining recovery then <=10s coast; constant observed front speeds.

    Project only member kinematics, sample each half-second, retain road and
    acquired-front constraints. Future rear response remains unknown; actual
    one-tick rear clearance is separately enforced on every issued command.
    """
    p = proposal.profile
    frames = {r.frame.vehicle_id: r.frame for r in reports}
    rows = [frames[v] for v in proposal.member_ids]
    constraints = [{k: v for k, v in c.items() if k != "speed_envelope_mps"}
        for c in downstream_envelope(proposal, reports, now_s=now_s)["constraints"]]
    for f in rows:
        if f.front is not None:
            constraints.append(dict(member_id=f.vehicle_id,
                downstream_speed_mps=max(0., f.speed_mps+f.front.relative_speed_mps),
                available_distance_m=f.front.gap_m-max(0., -f.front.relative_speed_mps)*(now_s-f.timestamp_s)-p.minimum_front_gap_m))
    travel = {f.vehicle_id: 0. for f in rows}
    previous = dict(previous)
    map_margin, traffic_margin, samples = float("inf"), None, []
    reason = None
    terminal = False
    for tick in range(int(remaining_s)+11):
        coast = tick >= remaining_s
        if coast and coast_terminal(rows, previous, p):
            terminal = True
            break
        if tick == remaining_s+10:
            reason = "coast_deadline"
            break
        if tick:
            target = min(reference, target+.5)
        requests = recovery_requests(rows, target, reference, p, coast=coast)
        commands, next_rows = {}, []
        for f in rows:
            a, _ = resolve_command(f, requests[f.vehicle_id], p, previous=previous.get(f.vehicle_id))
            if a is None:
                reason = "predicted_command_incompatible"
                break
            commands[f.vehicle_id] = a
            for dt in (.5, 1.):
                total = travel[f.vehicle_id]+f.speed_mps*dt+.5*a*dt*dt
                original = frames[f.vehicle_id]
                margin = original.distance_to_merge_m-original.speed_mps*(now_s-original.timestamp_s)-total-p.minimum_runway_m
                map_margin = min(map_margin, margin)
                for c in constraints:
                    if c["member_id"] == f.vehicle_id:
                        margin = c["available_distance_m"]-total+c["downstream_speed_mps"]*(tick+dt)
                        traffic_margin = margin if traffic_margin is None else min(traffic_margin, margin)
            displacement = f.speed_mps+.5*a
            travel[f.vehicle_id] += displacement
            next_rows.append(replace(f, timestamp_s=now_s+tick+1, speed_mps=max(0., f.speed_mps+a),
                acceleration_mps2=a, distance_to_merge_m=f.distance_to_merge_m-displacement,
                position_m=f.position_m+displacement))
        if reason:
            break
        midpoint = [replace(f, speed_mps=f.speed_mps+.5*commands[f.vehicle_id],
            distance_to_merge_m=f.distance_to_merge_m-.5*f.speed_mps-.125*commands[f.vehicle_id]) for f in rows]
        if not coast and (not compact(midpoint, p) or not compact(next_rows, p)):
            reason = "predicted_formation_loss"
            break
        rows, previous = next_rows, commands
        samples.append(dict(elapsed_s=tick+1, phase="coast" if coast else "recovery", commands=commands))
    if reason is None and (map_margin < 0 or (traffic_margin is not None and traffic_margin < 0)):
        reason = "remaining_map_or_front_constraint"
    return dict(feasible=reason is None and terminal, reason=reason or "full_remaining_recovery_and_coast_feasible",
        remaining_recovery_s=remaining_s, coast_cap_s=10., planned_samples=samples,
        minimum_map_margin_m=None if map_margin == float("inf") else map_margin,
        minimum_front_margin_m=traffic_margin, conditional_command_response=True,
        constant_observed_front_speed=True, future_rear_response_certified=False, safety_guarantee=False)


@dataclass
class RecoveryController:
    proposal: object
    previous_commands: dict
    start_s: float

    def __post_init__(self):
        self.previous_commands = dict(self.previous_commands)
        self.target = None
        self.last_time = None
        self.ready = 0
        self.terminal_result = None

    def step(self, context):
        if self.terminal_result is not None:
            return self.terminal_result
        now, p = float(context.now_s), self.proposal.profile
        if (self.last_time is None and now != self.start_s) or (self.last_time is not None and now != self.last_time+1):
            return self._fallback("clock_gap")
        self.last_time = now
        if now-self.start_s >= 20:
            return self._fallback("recovery_deadline")
        reports = eligible_packets(context)
        assessment = assess_formation(self.proposal, reports, now_s=now)
        if not assessment.approved:
            return self._fallback("acquired_formation:"+assessment.reason)
        frames = {r.frame.vehicle_id: r.frame for r in reports}
        rows = [frames[v] for v in self.proposal.member_ids]
        if not compact(rows, p) or any(f.position_m is None for f in rows):
            return self._fallback("formation_or_position_missing")
        if not set(self.proposal.member_ids) <= set(context.damping_commands)|set(context.damping_native_ids):
            return self._fallback("current_damping_reference_missing")
        ref = estimate_downstream_reference(self.proposal, context)
        if ref["target_speed_mps"] is None:
            return self._fallback("references_missing", dict(references=ref))
        reference = ref["target_speed_mps"]
        if self.target is None:
            self.target = mean(f.speed_mps for f in rows)
        self.target = min(reference, self.target+.5)
        details = dict(target_speed_mps=self.target, references=ref,
            span_m=assessment.initial_or_current_span_m, speed_spread_mps=assessment.speed_spread_mps)
        plan = recovery_plan(self.proposal, reports, now_s=now, previous=self.previous_commands,
            target=self.target, reference=reference, remaining_s=20-(now-self.start_s))
        details["plan"] = plan
        if not plan["feasible"]:
            return self._fallback("recovery_plan:"+plan["reason"], details)
        ready = (abs(self.target-reference) <= 1e-8 and all(abs(f.speed_mps-reference) <= .5 for f in rows)
                 and coast_terminal(rows, self.previous_commands, p))
        self.ready = self.ready+1 if ready else 0
        details["measured_ready_ticks"] = self.ready
        if self.ready >= 3:
            self.terminal_result = dict(status="release", commands={}, reason="measured_three_tick_handoff", details=details)
            return self.terminal_result
        requests = recovery_requests(rows, self.target, reference, p)
        commands, resolutions = {}, {}
        for f in rows:
            a, evidence = resolve_command(f, requests[f.vehicle_id], p, previous=self.previous_commands.get(f.vehicle_id))
            if a is None or _local_clearance_problem(f, p, now, acceleration=a):
                return self._fallback("unsafe_command", details)
            front = gentle_front_clearance(f, a, p, now_s=now)
            if front is not None and not front["gentle_feasible"]:
                return self._fallback("unsafe_front_clearance", details)
            commands[f.vehicle_id], resolutions[f.vehicle_id] = a, evidence
        details["command_resolution"] = resolutions
        self.previous_commands = dict(commands)
        return dict(status="active", commands=commands, reason="recovery_active", details=details)

    def _fallback(self, reason, details=None):
        self.terminal_result = dict(status="fallback", commands={}, reason=reason,
            details=dict(details or {}, fallback_coast_s=10.))
        return self.terminal_result
