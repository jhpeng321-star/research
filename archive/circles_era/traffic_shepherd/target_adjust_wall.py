"""Registered mid-coordination target adjustment on the unchanged lifecycle wall.

Version 2 repairs the v1 configuration failure: the +2 m/s arm built its
adjusted proposal under the unchanged entry profile, whose frozen bound
(targets <= 24 m/s) rejected it after allocation. The two target contracts are
now explicit:

- Entry/assembly target: frozen by the job's registered profile
  (LifecycleProfile, bound 24 m/s). Entry behavior is untouched.
- Continuation target after the registered HOLD+5 branch: the entry target
  plus one registered adjustment ({-2, +2} m/s), validated against an explicit
  continuation profile (identical limits, bound 40 m/s) and the members'
  entry-reported allowed speeds BEFORE any simulator allocation. A target
  outside the registered set is an implementation failure that raises before
  allocation; an acquired-state feasibility refusal or protective early exit
  after allocation remains a physical policy outcome.

Everything else is unchanged: the same finite lifecycle law schedules taper at
measured readiness + 20 s, the resolver enforces the -0.3 m/s^2 absolute bound
and 0.15 m/s^3 command slew by clipping, the exit and identical-CIRCLES
handback law is untouched, the adjustment is applied exactly once, and the
HOLD clock and scheduled end are never reset. This is a continuation action;
the entry proposal, cohort and assembly remain those of the source wall.
"""
from dataclasses import asdict, dataclass, replace

from .gentle_wall import GentleProfile
from .lifecycle_wall import LifecycleWallPolicy

VERSION = "hold_target_adjust/2"
CONTINUATION_MAXIMUM_TARGET_MPS = 40.0
REGISTERED_ADJUSTMENTS_MPS = (-2.0, 2.0)


@dataclass(frozen=True)
class ContinuationTargetProfile(GentleProfile):
    """The frozen wall limits with the continuation-only target bound.

    Identical to LifecycleProfile except maximum_target_speed_mps. It
    authorizes nothing by itself and is used only inside the adjusted proposal
    created at the registered branch; entry proposals never use it.
    """

    version: str = VERSION
    maximum_target_speed_mps: float = CONTINUATION_MAXIMUM_TARGET_MPS

    def __post_init__(self):
        expected = asdict(GentleProfile())
        expected["version"] = VERSION
        expected["maximum_target_speed_mps"] = CONTINUATION_MAXIMUM_TARGET_MPS
        if (asdict(self) != expected or type(self.readiness_samples) is not int
                or any(type(v) is bool for v in asdict(self).values())):
            raise ValueError("one frozen continuation target profile required")


def continuation_target(original_target_mps, adjust_mps, member_allowed_speeds_mps=()):
    """Pure continuation-target validation; raises before any allocation.

    member_allowed_speeds_mps may be any iterable of floats (dict inputs are
    treated as vehicle-id -> allowed-speed mappings).
    """
    if adjust_mps not in REGISTERED_ADJUSTMENTS_MPS:
        raise ValueError("adjustment outside the registered continuation set")
    if isinstance(member_allowed_speeds_mps, dict):
        member_allowed_speeds_mps = member_allowed_speeds_mps.values()
    original = float(original_target_mps)
    adjusted = original + adjust_mps
    if original < 0 or not 0. <= adjusted <= CONTINUATION_MAXIMUM_TARGET_MPS:
        raise ValueError("continuation target outside the continuation profile bound")
    allowed = [float(v) for v in (member_allowed_speeds_mps or ())]
    if allowed and adjusted > min(allowed):
        raise ValueError("continuation target above a member allowed speed")
    return adjusted, ContinuationTargetProfile()


class HoldTargetAdjustPolicy(LifecycleWallPolicy):
    """LifecycleWallPolicy plus one registered target adjustment at the branch."""

    def __init__(self, job, registration):
        super().__init__(job, registration)
        spec = self.job["policy_spec"]["target_adjust_branch"]
        self.adjust_mps = float(spec["adjust_mps"])
        self.branch_mode = spec["branch_mode"]
        self.expected_hold_start_s = spec.get("expected_hold_start_s")
        self.branch_time_s = spec.get("branch_time_s")
        if self.adjust_mps not in REGISTERED_ADJUSTMENTS_MPS:
            raise ValueError("adjustment outside the registered continuation set")
        if self.branch_mode == "absolute":
            if (self.branch_time_s is None or self.expected_hold_start_s is None
                    or float(self.branch_time_s) != float(self.expected_hold_start_s)+5.):
                raise ValueError("absolute branch must lie five seconds after measured HOLD start")
        elif self.branch_mode != "hold_relative":
            raise ValueError("branch mode must be absolute or hold_relative")
        self._hold_ticks = 0
        self.last_output_phase = None
        self.entry_allowed_speeds = {}
        self.receipt = dict(version=VERSION, requested=False, verified=False,
            branch_mode=self.branch_mode, adjust_mps=self.adjust_mps,
            expected_hold_start_s=self.expected_hold_start_s,
            registered_branch_time_s=self.branch_time_s, original_target_speed_mps=None,
            adjusted_target_speed_mps=None, actual_hold_start_s=None,
            request_time_s=None, early_exit_reason=None,
            terminal_at_s=None, terminal_phase=None)

    def _request(self, now):
        if self.ready_at is None or self.taper_at is not None:
            raise ValueError("target adjustment requires an active measured HOLD")
        adjusted, profile = continuation_target(
            self.proposal.target_speed_mps, self.adjust_mps, self.entry_allowed_speeds)
        self.proposal = replace(self.proposal, target_speed_mps=adjusted, profile=profile)
        self.receipt.update(requested=True, verified=True,
            actual_hold_start_s=self.ready_at, request_time_s=now,
            original_target_speed_mps=self.proposal.target_speed_mps-self.adjust_mps,
            adjusted_target_speed_mps=self.proposal.target_speed_mps)

    def __call__(self, context):
        now = context.now_s
        if not self.receipt["requested"]:
            if self.branch_mode == "absolute":
                if now == self.branch_time_s:
                    if self.last_output_phase != "hold":
                        raise ValueError("registered target-adjust branch missed the measured HOLD")
                    if self.ready_at != self.expected_hold_start_s:
                        raise ValueError("measured HOLD start differs from the registered source")
                    self._request(now)
            elif self.last_output_phase == "hold":
                self._hold_ticks += 1
                if self._hold_ticks == 5:
                    self._request(now)
        if not self.receipt["requested"] and self.branch_mode == "absolute" \
                and self.last_output_phase is not None \
                and self.last_output_phase != "damping" and now >= self.branch_time_s:
            raise ValueError("registered target-adjust branch was missed; no late replacement")
        if self.receipt["requested"] and self.last_output_phase == "hold" \
                and self.proposal is not None \
                and self.proposal.target_speed_mps != self.receipt["adjusted_target_speed_mps"]:
            raise ValueError("continuation target changed after the registered branch")
        output = super().__call__(context)
        if self.proposal is not None and not self.entry_allowed_speeds:
            self.entry_allowed_speeds = {
                r.frame.vehicle_id: r.frame.allowed_speed_mps for r in context.reports
                if r.frame.vehicle_id in set(self.job["policy_spec"]["member_ids"])}
        self.last_output_phase = output.phase
        if self.receipt["requested"]:
            if self.receipt["early_exit_reason"] is None:
                self.receipt["early_exit_reason"] = output.details.get("early_exit_reason")
            if output.phase in {"release", "abort"} and self.receipt["terminal_at_s"] is None:
                self.receipt.update(terminal_at_s=now, terminal_phase=output.phase)
            compact = dict(self.receipt)
            if now != self.receipt["request_time_s"]:
                # Full acquired context rides once at the branch; later telemetry
                # keeps the compact state, as in the duration branch contract.
                compact.pop("expected_hold_start_s", None)
            return replace(output, details=dict(output.details, target_adjust_branch=compact))
        return output


def policy_factory(job, registration):
    return HoldTargetAdjustPolicy(job, registration)


def branch_receipt(policy):
    if not isinstance(policy, HoldTargetAdjustPolicy):
        raise TypeError("target-adjust policy instance required for its receipt")
    return dict(policy.receipt)
