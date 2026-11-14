"""Pure, diagnostic-only single-wall gates and lifecycle; no default wall policy.

Regression intent retained from historical test_multiwall_coordinator.py and
test_damping_plus_wall.py: immutable membership, two gates, bounded lifecycle,
and return to the same damping. Historical thresholds/controllers are not reused.
Report geometry is not a fleet census or proof of safe lane occupancy. No learned
benefit predictor, accepted wall design, or physical efficacy is implemented here.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from math import isfinite
from typing import Iterable, Mapping

from .reporting import CAVReport


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(asdict(value), sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class WallMember:
    vehicle_id: str
    lane_id: str

    def __post_init__(self):
        if not self.vehicle_id or not self.lane_id:
            raise ValueError("member and expected map lane must be named")


@dataclass(frozen=True)
class WallParameters:
    control_dt_s: float
    assembly_s: float
    hold_s: float
    taper_s: float
    target_speed_mps: float
    speed_gain_per_s: float
    acceleration_min_mps2: float
    acceleration_max_mps2: float
    assembly_speed_tolerance_mps: float
    maximum_member_span_m: float
    max_report_age_s: float
    minimum_front_gap_m: float
    minimum_rear_gap_m: float
    minimum_front_ttc_s: float
    minimum_rear_ttc_s: float

    def __post_init__(self):
        values = asdict(self)
        if not all(isfinite(v) for v in values.values()):
            raise ValueError("wall parameters must be finite")
        if min(self.control_dt_s, self.assembly_s, self.hold_s, self.taper_s,
               self.speed_gain_per_s, self.maximum_member_span_m) <= 0:
            raise ValueError("wall durations, gain and span must be positive")
        if min(self.target_speed_mps, self.assembly_speed_tolerance_mps,
               self.max_report_age_s, self.minimum_front_gap_m, self.minimum_rear_gap_m,
               self.minimum_front_ttc_s, self.minimum_rear_ttc_s) < 0:
            raise ValueError("negative wall thresholds")
        if not self.acceleration_min_mps2 < 0 < self.acceleration_max_mps2:
            raise ValueError("wall acceleration envelope must straddle zero")
        for duration in (self.assembly_s, self.hold_s, self.taper_s):
            if abs(duration / self.control_dt_s - round(duration / self.control_dt_s)) > 1e-8:
                raise ValueError("phase durations must align with the control clock")


@dataclass(frozen=True)
class WallProposal:
    start_s: float
    required_lanes: tuple[str, ...]
    members: tuple[WallMember, ...]
    parameters: WallParameters

    def __post_init__(self):
        if not isinstance(self.required_lanes, tuple) or not isinstance(self.members, tuple):
            raise ValueError("proposal members and map lanes must be immutable tuples")
        if not isfinite(self.start_s) or self.start_s < 0:
            raise ValueError("invalid wall start")
        if not self.required_lanes or any(not lane for lane in self.required_lanes) or len(set(self.required_lanes)) != len(self.required_lanes):
            raise ValueError("explicit distinct expected map lanes are required")
        if len({m.vehicle_id for m in self.members}) != len(self.members) or len({m.lane_id for m in self.members}) != len(self.members):
            raise ValueError("one distinct CAV per wall lane; duplicate membership forbidden")
        if any(m.lane_id not in self.required_lanes for m in self.members):
            raise ValueError("member assigned outside expected map lanes")
        if abs(self.start_s / self.parameters.control_dt_s - round(self.start_s / self.parameters.control_dt_s)) > 1e-8:
            raise ValueError("start must align with the control clock")

    @property
    def fingerprint(self):
        return _digest(self)


@dataclass(frozen=True)
class MemberEvidence:
    vehicle_id: str
    lane_id: str | None
    timestamp_s: float
    distance_to_merge_m: float | None
    speed_mps: float
    allowed_speed_mps: float
    front_gap_m: float | None
    front_relative_speed_mps: float | None
    front_coverage_m: float
    rear_gap_m: float | None
    rear_relative_speed_mps: float | None
    rear_coverage_m: float


@dataclass(frozen=True)
class CAVWallEvidence:
    """Allowlisted self states/local detections only, with no global health inputs."""
    now_s: float
    members: tuple[MemberEvidence, ...]

    @property
    def fingerprint(self):
        return _digest(self)


@dataclass(frozen=True)
class FeasibilityApproval:
    proposal_fingerprint: str
    evidence: CAVWallEvidence
    status: str
    reason: str

    @property
    def approved(self):
        return self.status == "feasible"


def assess_feasibility(proposal: WallProposal, reports: Iterable[CAVReport], *, now_s: float) -> FeasibilityApproval:
    """Conservative reported geometry check; call only with already received packets.

    Missing reporters imply insufficient information, never an empty expected lane.
    Report construction/receipt and final CAV command masks remain caller contracts.
    """
    if not isfinite(now_s):
        raise ValueError("invalid evidence clock")
    packets = list(reports)
    if any(not isinstance(r, CAVReport) for r in packets):
        raise TypeError("wall evidence admits only CAVReport packets")
    by_id = {}
    for report in packets:
        identity = report.frame.vehicle_id
        if identity in by_id:
            raise ValueError("supply one latest received report per CAV")
        by_id[identity] = report
    evidence_rows = []
    for member in proposal.members:
        report = by_id.get(member.vehicle_id)
        if report is None:
            continue
        f = report.frame
        evidence_rows.append(MemberEvidence(f.vehicle_id, f.lane_id, f.timestamp_s,
            f.distance_to_merge_m, f.speed_mps, f.allowed_speed_mps,
            f.front.gap_m if f.front else None, f.front.relative_speed_mps if f.front else None,
            f.front_coverage_m, f.rear.gap_m if f.rear else None,
            f.rear.relative_speed_mps if f.rear else None, f.rear_coverage_m))
    evidence = CAVWallEvidence(now_s, tuple(evidence_rows))

    def result(status, reason):
        return FeasibilityApproval(proposal.fingerprint, evidence, status, reason)

    if {m.lane_id for m in proposal.members} != set(proposal.required_lanes):
        return result("insufficient_members", "proposal does not assign a CAV to every expected map lane")
    if len(evidence_rows) != len(proposal.members):
        return result("insufficient_information", "missing member report; lane occupancy is unknown")
    p = proposal.parameters
    for member, row in zip(proposal.members, evidence_rows):
        packet = by_id[member.vehicle_id]
        if packet.sent_at_s > now_s or not 0 <= now_s - row.timestamp_s <= p.max_report_age_s:
            return result("insufficient_information", "future or stale member report")
        if row.lane_id is None or row.distance_to_merge_m is None:
            return result("insufficient_information", "member map location is unavailable")
        if row.lane_id != member.lane_id:
            return result("unsafe_geometry", "reported member left its proposed lane")
        if p.target_speed_mps > row.allowed_speed_mps:
            return result("unsafe_geometry", "wall target exceeds member allowed speed")
        for label, gap, relative, coverage, minimum_gap, minimum_ttc in (
            ("front", row.front_gap_m, row.front_relative_speed_mps, row.front_coverage_m, p.minimum_front_gap_m, p.minimum_front_ttc_s),
            ("rear", row.rear_gap_m, row.rear_relative_speed_mps, row.rear_coverage_m, p.minimum_rear_gap_m, p.minimum_rear_ttc_s),
        ):
            if coverage <= 0 or coverage < minimum_gap:
                return result("insufficient_information", f"{label} coverage cannot support the specified clearance")
            if gap is not None:
                closing = -relative if label == "front" else relative
                if gap < minimum_gap or (closing > 0 and gap / closing < minimum_ttc):
                    return result("unsafe_geometry", f"reported {label} clearance or TTC violates the proposed bound")
    distances = [r.distance_to_merge_m for r in evidence_rows]
    if max(distances) - min(distances) > p.maximum_member_span_m:
        return result("unsafe_geometry", "reported members exceed the proposed longitudinal span")
    return result("feasible", "reported geometry only; unobserved traffic and physical feasibility remain unvalidated")


@dataclass(frozen=True)
class DiagnosticBenefitApproval:
    proposal_fingerprint: str
    evidence_fingerprint: str
    approved: bool
    report_only_score: float
    explicit_threshold: float
    diagnostic_source: str
    # This is intentionally not an accepted predictor result.
    scientifically_validated: bool = False


def diagnostic_benefit_approval(proposal: WallProposal, feasibility: FeasibilityApproval, *,
                                report_only_score: float, explicit_threshold: float,
                                diagnostic_source: str) -> DiagnosticBenefitApproval:
    """Record a manually supplied report-only diagnostic judgment; never fit a gate.

    The source must describe how only feasibility.evidence supplied the judgment.
    This narrow interface cannot attest the truth of a caller's provenance claim.
    """
    if feasibility.proposal_fingerprint != proposal.fingerprint:
        raise ValueError("benefit judgment refers to a different proposed wall")
    if not diagnostic_source.strip() or not all(isfinite(x) for x in (report_only_score, explicit_threshold)):
        raise ValueError("explicit finite diagnostic score, threshold and provenance are required")
    return DiagnosticBenefitApproval(proposal.fingerprint, feasibility.evidence.fingerprint,
        report_only_score >= explicit_threshold, report_only_score, explicit_threshold, diagnostic_source)


@dataclass(frozen=True)
class DiagnosticWallAuthorization:
    proposal_fingerprint: str
    evidence_fingerprint: str
    evidence_time_s: float
    diagnostic_only: bool = True


def authorize_wall(proposal: WallProposal, feasibility: FeasibilityApproval,
                   benefit: DiagnosticBenefitApproval) -> DiagnosticWallAuthorization:
    if not (proposal.fingerprint == feasibility.proposal_fingerprint == benefit.proposal_fingerprint):
        raise ValueError("both approvals must reference the identical immutable wall proposal")
    if feasibility.evidence.fingerprint != benefit.evidence_fingerprint:
        raise ValueError("both approvals must use the same CAV evidence snapshot")
    if not feasibility.approved or not benefit.approved or benefit.scientifically_validated:
        raise ValueError("both diagnostic gates must approve; scientific acceptance is not implemented")
    if feasibility.evidence.now_s != proposal.start_s:
        raise ValueError("diagnostic authorization requires evidence evaluated exactly at the proposed start; prestart planning is not implemented")
    return DiagnosticWallAuthorization(proposal.fingerprint, feasibility.evidence.fingerprint, feasibility.evidence.now_s)


@dataclass(frozen=True)
class WallStep:
    phase: str
    acceleration_items: tuple[tuple[str, float], ...]
    reason: str
    proposal_fingerprint: str

    @property
    def commands(self):
        """Absolute accelerations for CommandOwnership; empty map returns to damping."""
        return dict(self.acceleration_items)


class WallLifecycle:
    """Minimal speed-feedback mechanics with no default activation or parameters.

    Assembly is only speed settling within already reported compact geometry;
    lane changing and formation planning are not implemented. Abort/release is
    terminal and emits an empty wall mask. Ownership then uses identical damping
    (or its explicit native fallback). This is not a release-safety guarantee.
    Taper blends toward damping but retains the proposed wall/speed envelope
    until release; that envelope never constrains damping after wall release.
    """
    def __init__(self, proposal: WallProposal, authorization: DiagnosticWallAuthorization):
        if authorization.proposal_fingerprint != proposal.fingerprint or not authorization.diagnostic_only:
            raise ValueError("lifecycle requires this proposal's diagnostic authorization")
        self.proposal = proposal
        self._authorization_evidence_fingerprint = authorization.evidence_fingerprint
        self._last_time = None
        self._terminal = None
        self._assembly_confirmed = False

    def step(self, reports: Iterable[CAVReport], *, now_s: float,
             damping_commands: Mapping[str, float]) -> WallStep:
        p = self.proposal.parameters
        first_step = self._last_time is None
        if not isfinite(now_s) or (self._last_time is not None and now_s != self._last_time + p.control_dt_s):
            raise ValueError("wall clock must advance exactly one configured control step")
        if self._last_time is None and now_s != self.proposal.start_s:
            raise ValueError("wall lifecycle must begin at its proposed start")
        self._last_time = now_s

        def result(phase, reason, commands=()):
            if phase in {"release", "abort"}:
                self._terminal = (phase, reason)
            return WallStep(phase, tuple(commands), reason, self.proposal.fingerprint)

        if self._terminal:
            return result(*self._terminal)
        elapsed = now_s - self.proposal.start_s
        if elapsed >= p.assembly_s + p.hold_s + p.taper_s:
            return result("release", "declared wall duration completed; return ownership to damping")
        feasibility = assess_feasibility(self.proposal, reports, now_s=now_s)
        if first_step and feasibility.evidence.fingerprint != self._authorization_evidence_fingerprint:
            return result("abort", "authorization evidence changed; reauthorization required before activation")
        if not feasibility.approved:
            return result("abort", f"{feasibility.status}: {feasibility.reason}")
        rows = feasibility.evidence.members
        if elapsed >= p.assembly_s and not self._assembly_confirmed:
            if any(abs(r.speed_mps - p.target_speed_mps) > p.assembly_speed_tolerance_mps for r in rows):
                return result("abort", "assembly speed settling deadline missed")
            self._assembly_confirmed = True
        phase = "assembly" if elapsed < p.assembly_s else "hold" if elapsed < p.assembly_s + p.hold_s else "taper"
        commands = []
        for r in rows:
            lo = max(p.acceleration_min_mps2, -r.speed_mps / p.control_dt_s)
            hi = min(p.acceleration_max_mps2, (r.allowed_speed_mps - r.speed_mps) / p.control_dt_s)
            if lo > hi:
                return result("abort", "infeasible member speed bounds")
            a = min(hi, max(lo, p.speed_gain_per_s * (p.target_speed_mps - r.speed_mps)))
            if phase == "taper":
                damping = damping_commands.get(r.vehicle_id)
                if damping is None or not isfinite(damping):
                    return result("abort", "missing valid identical-damping reference during taper")
                fraction = (elapsed - p.assembly_s - p.hold_s) / p.taper_s
                a = (1 - fraction) * a + fraction * damping
                # Retain the explicit treatment envelope while a wall still owns
                # this member. Release emits no wall command, so identical damping
                # regains its own full adopted envelope on the following tick.
                a = min(hi, max(lo, a))
            commands.append((r.vehicle_id, float(a)))
        return result(phase, "diagnostic mechanics only", commands)
