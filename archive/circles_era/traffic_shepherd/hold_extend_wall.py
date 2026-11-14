"""Registered one-increment HOLD extension on the delivered raise-26 wall.

Versioned continuation commitment added to the existing target-adjust history
(24 -> 26 m/s at measured HOLD+5). At the registered extension tick (measured
HOLD start + 20 s = the original scheduled taper tick), before the original
taper command is issued, the policy evaluates one +10 s extension from
acquired reports:

- feasible  -> both the proposal and the scheduling profile move to the
  registered extension profile (identical limits, hold_s 30); the HOLD clock,
  readiness and target are untouched; remaining-maneuver/exit accounting
  recomputes per tick from acquired reports; the extension applies exactly
  once;
- infeasible-> the extension is declined and the original H20 schedule
  (taper at readiness + 20 s) proceeds unchanged - an acquired-state physical
  outcome, never a replaced value or a defect.

Original behavior is byte-equivalent before the extension branch: the class
inherits the delivered target-adjust policy and adds no behavior until that
tick. The original H20 release behavior is retained whenever the extension is
not selected. Protective exits and identical-CIRCLES handback are unchanged;
HDVs receive no commands.

Receipts: the target adjustment is recorded in ``self.receipt`` (parent);
the extension is recorded separately in ``self.ext_receipt`` so the two
branch records can never overwrite each other.
"""
from dataclasses import asdict, dataclass, replace

from .gentle_wall import GentleProfile
from .lifecycle_wall import remaining_plan
from .paired_experiments import digest
from .target_adjust_wall import HoldTargetAdjustPolicy

VERSION = "hold_extend/2"
EXTENSION_S = 10.0
BRANCH_AFTER_HOLD_S = 20.0


@dataclass(frozen=True)
class ExtensionProfile(GentleProfile):
    """The continuation limits with hold_s 30 (one registered extension).

    Also carries the continuation target bound (40 m/s) because this profile
    only ever governs the already-raised 26 m/s continuation proposal; the
    24 m/s entry bound is untouched.
    """

    version: str = VERSION
    maximum_target_speed_mps: float = 40.0
    hold_s: float = 20.0 + EXTENSION_S

    def __post_init__(self):
        expected = asdict(GentleProfile())
        expected["version"] = VERSION
        expected["maximum_target_speed_mps"] = 40.0
        expected["hold_s"] = 20.0 + EXTENSION_S
        if (asdict(self) != expected or type(self.readiness_samples) is not int
                or any(type(v) is bool for v in asdict(self).values())):
            raise ValueError("one frozen hold-extension profile required")


def extension_profile():
    return ExtensionProfile()


def validate_extension_spec(spec, *, adjust_expected_hold_start_s=None):
    """Pure validation before any allocation.

    Modes: "absolute" (corridor; branch tick and measured HOLD start are
    registered and verified at runtime) and "hold_relative" (isolated
    fixture; the branch is the 20th second after measured readiness).
    """
    if float(spec["extension_s"]) != EXTENSION_S:
        raise ValueError("extension outside the registered single increment")
    mode = spec["branch_mode"]
    if mode == "absolute":
        ready = float(spec["expected_hold_start_s"])
        branch = float(spec["branch_time_s"])
        if branch != ready + BRANCH_AFTER_HOLD_S:
            raise ValueError("extension branch must lie at measured HOLD start + 20 s")
        if adjust_expected_hold_start_s is not None and ready != float(adjust_expected_hold_start_s):
            raise ValueError("extension history differs from the target-adjust history")
        return dict(branch_mode="absolute", expected_hold_start_s=ready,
                    branch_time_s=branch, extension_s=EXTENSION_S)
    if mode == "hold_relative":
        if float(spec["branch_after_hold_start_s"]) != BRANCH_AFTER_HOLD_S:
            raise ValueError("relative extension branch must lie at HOLD start + 20 s")
        return dict(branch_mode="hold_relative", expected_hold_start_s=None,
                    branch_time_s=None, extension_s=EXTENSION_S)
    raise ValueError("extension branch mode must be absolute or hold_relative")


class HoldExtendPolicy(HoldTargetAdjustPolicy):
    """Target-adjust policy plus one registered extension at the scheduled end."""

    def __init__(self, job, registration):
        super().__init__(job, registration)
        spec = self.job["policy_spec"]["hold_extension"]
        self.ext = validate_extension_spec(
            spec, adjust_expected_hold_start_s=self.expected_hold_start_s)
        # Separate receipt: the parent's self.receipt records the target
        # adjustment and must never be overwritten by the extension.
        self.ext_receipt = dict(
            version=VERSION, requested=False, verified=False, applied=False,
            declined_reason=None, branch_mode=self.ext["branch_mode"],
            expected_hold_start_s=self.ext["expected_hold_start_s"],
            registered_branch_time_s=self.ext["branch_time_s"],
            extension_s=self.ext["extension_s"], actual_hold_start_s=None,
            request_time_s=None, original_schedule_end_s=None,
            extended_schedule_end_s=None, feasibility=None,
            early_exit_reason=None, terminal_at_s=None, terminal_phase=None)

    def _request_extension(self, now, reports):
        if self.ready_at is None or self.taper_at is not None:
            raise ValueError("extension requires an active measured HOLD")
        if (self.ext["branch_mode"] == "absolute"
                and self.ready_at != self.ext["expected_hold_start_s"]):
            raise ValueError("measured HOLD start differs from the registered extension")
        original_hold_s = self.proposal.profile.hold_s

        def assessment(profile):
            """Full remaining-maneuver evaluation of one candidate commitment."""
            candidate = replace(self.proposal, profile=profile)
            plan = remaining_plan(candidate, reports, now_s=now,
                previous_commands=self.previous_commands,
                readiness_count=self.readiness, ready_at_s=self.ready_at)
            return candidate, dict(
                assessed_hold_s=profile.hold_s,
                feasible=bool(plan.get("feasible")),
                reason=plan.get("reason"),
                remaining_hold_s=plan.get("remaining_hold_s"),
                remaining_exit_s=plan.get("remaining_exit_s"),
                remaining_total_s=plan.get("remaining_total_s"),
                minimum_map_margin_m=plan.get("minimum_map_margin_m"),
                minimum_traffic_margin_m=plan.get("minimum_traffic_margin_m"))

        # Assess the ORIGINAL H20 commitment as it stands, then build the
        # PROPOSED EXTENDED commitment (hold_s 30, same reports/commands/
        # readiness) and assess THAT. Only a feasible extended assessment
        # commits. The v1 admission evaluated the original plan's final
        # tick (remaining hold 0 + exit 1) instead - vacuous for the
        # extension decision.
        _, original_assessment = assessment(self.proposal.profile)
        ext_profile = extension_profile()
        ext_proposal, extension_assessment = assessment(ext_profile)
        selected = "extend" if extension_assessment["feasible"] else "decline"
        self.ext_receipt.update(requested=True, verified=True,
            actual_hold_start_s=self.ready_at, request_time_s=now,
            original_schedule_end_s=self.ready_at + original_hold_s,
            assessments=dict(
                original_plan=original_assessment,
                extension_plan=extension_assessment,
                selected=selected,
                bound_to=dict(proposal_fingerprint=self.proposal.fingerprint,
                    evidence_digest=digest([r.payload() for r in reports]),
                    now_s=now, ready_at_s=self.ready_at,
                    elapsed_hold_s=now - self.ready_at,
                    target_mps=self.proposal.target_speed_mps,
                    command_limits_mps2=dict(
                        min=self.profile.acceleration_min_mps2,
                        max=self.profile.acceleration_max_mps2,
                        change_per_tick=self.profile.acceleration_change_limit_mps3))))
        if selected != "extend":
            self.ext_receipt["declined_reason"] = (
                "proposed_extension_infeasible:" + str(extension_assessment["reason"]))
            return  # no state mutation: original H20 schedule proceeds
        # Commit only after the extended assessment passed.
        self.proposal = ext_proposal
        self.profile = ext_profile
        self.ext_receipt.update(applied=True, selected="extend",
            extended_schedule_end_s=self.ready_at + 20.0 + self.ext["extension_s"])

    def __call__(self, context):
        now = context.now_s
        # Runtime trigger (both modes): the 20th second of the measured HOLD.
        # Absolute mode additionally pins the tick to the registered history.
        if not self.ext_receipt["requested"] and self.last_output_phase == "hold" \
                and self.ready_at is not None and self.taper_at is None \
                and now == self.ready_at + BRANCH_AFTER_HOLD_S:
            if self.ext["branch_mode"] == "absolute":
                if now != self.ext["branch_time_s"] \
                        or self.ready_at != self.ext["expected_hold_start_s"]:
                    raise ValueError("measured HOLD history differs from the "
                                     "registered extension")
            self._request_extension(now, context.reports)
        if (not self.ext_receipt["requested"] and self.ext["branch_mode"] == "absolute"
                and self.last_output_phase is not None
                and self.last_output_phase != "damping"
                and self.ext["branch_time_s"] is not None
                and now >= self.ext["branch_time_s"]):
            raise ValueError("registered extension branch was missed; no late replacement")
        if self.ext_receipt["applied"]:
            if self.profile.hold_s != 20.0 + self.ext["extension_s"]:
                raise ValueError("extension profile lost after the registered branch")
        output = super().__call__(context)
        self.last_output_phase = output.phase
        if self.ext_receipt["requested"]:
            if self.ext_receipt["early_exit_reason"] is None:
                self.ext_receipt["early_exit_reason"] = output.details.get("early_exit_reason")
            if output.phase in {"release", "abort"} and self.ext_receipt["terminal_at_s"] is None:
                self.ext_receipt.update(terminal_at_s=now, terminal_phase=output.phase)
            compact = dict(self.ext_receipt)
            if now != self.ext_receipt["request_time_s"]:
                compact.pop("expected_hold_start_s", None)
                compact.pop("feasibility", None)
            else:
                # One-time observation packet at the new decision: only
                # reports actually received by this tick, plus the evaluated
                # commitment. Sent time is the report's own transmission
                # stamp; the RSU receipt time rides the telemetry row.
                compact["observation_packet"] = dict(
                    decision_time_s=now,
                    current_target_mps=self.proposal.target_speed_mps,
                    proposal_target_mps=self.proposal.target_speed_mps,
                    elapsed_hold_s=now - (self.ready_at or now),
                    reports=[
                        dict(vehicle_id=r.frame.vehicle_id,
                             acquired_s=r.frame.timestamp_s, sent_s=r.sent_at_s,
                             age_s=round(r.sent_at_s - r.frame.timestamp_s, 3),
                             lane_id=r.frame.lane_id,
                             distance_to_merge_m=r.frame.distance_to_merge_m,
                             speed_mps=r.frame.speed_mps,
                             acceleration_mps2=r.frame.acceleration_mps2,
                             front=None if r.frame.front is None else dict(
                                 gap_m=r.frame.front.gap_m,
                                 relative_speed_mps=r.frame.front.relative_speed_mps),
                             rear=None if r.frame.rear is None else dict(
                                 gap_m=r.frame.rear.gap_m,
                                 relative_speed_mps=r.frame.rear.relative_speed_mps),
                             front_coverage_m=r.frame.front_coverage_m,
                             rear_coverage_m=r.frame.rear_coverage_m)
                        for r in context.reports],
                    feasibility=self.ext_receipt["feasibility"])
            return replace(output, details=dict(output.details,
                                                hold_extension_branch=compact))
        return output


def policy_factory(job, registration):
    return HoldExtendPolicy(job, registration)


def branch_receipt(policy):
    if not isinstance(policy, HoldExtendPolicy):
        raise TypeError("hold-extend policy instance required for its receipt")
    return dict(policy.ext_receipt)
