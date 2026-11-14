"""One registered alternative target rule for the existing finite lifecycle wall.

The rule is the formation's own design law without the 24 m/s cap:
target = reported member mean speed - 2 m/s, still limited by member allowed
speeds. Everything else in the frozen gentle lifecycle contract is unchanged:
-0.3 m/s^2 absolute wall braking, 0.15 m/s^3 ramp, 1 s control / 0.5 s physics,
readiness samples, compact HOLD, coast exit and identical-damping handback.
The 40 m/s registration bound cannot bind in this corridor (member allowed
speeds and ambient speeds are far below it), so the commanded target is
exactly the uncapped report-based rule. This module adds no controller
behavior; it changes one frozen profile parameter through a subclass. The
lifecycle law version in telemetry remains finite_lifecycle_wall/2 because
that law is unchanged; this profile version records the target rule.
"""
from dataclasses import asdict, dataclass
from copy import deepcopy
from statistics import fmean

from .gentle_wall import GentleProfile
from .lifecycle_wall import LifecycleWallPolicy

VERSION = "target_speed_wall/1"
REGISTERED_MAXIMUM_TARGET_MPS = 40.0
TARGET_RULE = "reported_member_mean_speed_mps_minus_2_bounded_by_member_allowed_speed"


@dataclass(frozen=True)
class TargetSpeedProfile(GentleProfile):
    version: str = VERSION
    maximum_target_speed_mps: float = REGISTERED_MAXIMUM_TARGET_MPS

    def __post_init__(self):
        expected = asdict(GentleProfile())
        expected["version"] = VERSION
        expected["maximum_target_speed_mps"] = REGISTERED_MAXIMUM_TARGET_MPS
        if (asdict(self) != expected or type(self.readiness_samples) is not int
                or any(type(v) is bool for v in asdict(self).values())):
            raise ValueError("one frozen uncapped-target lifecycle profile required")


class TargetSpeedWallPolicy(LifecycleWallPolicy):
    """The unchanged finite lifecycle policy with the registered target rule.

    The parent constructor freezes a LifecycleProfile, so the identical
    initialization is repeated here with this registered profile.
    """

    def __init__(self, job, registration):
        if job["arm"] != "wall":
            raise ValueError("target speed wall is an explicit wall treatment")
        self.job, self.registration = deepcopy(job), registration
        self.profile = TargetSpeedProfile(**self.job["policy_spec"]["profile"])
        self.proposal = self.terminal = self.last_time = None
        self.terminal_at = None
        self.previous_commands = {}
        self.readiness = 0
        self.ready_at = self.taper_at = None
        self.early_exit_reason = None


def policy_factory(job, registration):
    return TargetSpeedWallPolicy(job, registration)


def expected_uncapped_target(member_speeds, allowed_speeds):
    """The registered target law, for offline verification of issued proposals."""
    return min(REGISTERED_MAXIMUM_TARGET_MPS, min(allowed_speeds),
               max(0., fmean(member_speeds) - 2.))


def predecision_identity_keys():
    return ("decision_time_s", "predecision_state_sha256", "predecision_report_evidence_sha256",
            "predecision_report_prefix_sha256", "predecision_command_prefix_sha256")
