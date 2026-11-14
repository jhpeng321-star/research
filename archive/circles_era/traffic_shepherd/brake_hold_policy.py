"""Two-factor research policy over the unchanged executed gentle-wall code.

Only absolute minimum acceleration and HOLD duration vary. This is an explicit
experimental assignment, not a qualified feasibility or incremental-benefit gate.
The subclass substitutes proposal construction because GentleWallPolicy fixes
its profile constructor. All subsequent physical branches remain inherited,
including its existing timing of the downstream-envelope check at taper.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace

from .formation import make_proposal
from .gentle_wall import GentleProfile, GentleWallPolicy


VERSION = "brake_hold_factorial/1"
BRAKING_MINIMA_MPS2 = (-.3, -3.)
HOLD_DURATIONS_S = (0., 20., 40.)


@dataclass(frozen=True)
class BrakeHoldProfile(GentleProfile):
    """The six registered factor combinations; every other field is frozen.

    HOLD=0 means enter the existing taper branch on the third readiness sample.
    It does not remove assembly, readiness, taper or protective handback.
    The -3 factor still has the identical 0.15 m/s³ command ramp.
    """
    version: str = VERSION

    def __post_init__(self):
        if (type(self.acceleration_min_mps2) not in (int, float)
                or self.acceleration_min_mps2 not in BRAKING_MINIMA_MPS2
                or type(self.hold_s) not in (int, float)
                or self.hold_s not in HOLD_DURATIONS_S):
            raise ValueError("only braking minima -0.3/-3 and HOLD 0/20/40 are registered factors")
        # The canonical gentle profile validates the remaining geometry, safety,
        # ramp and clock. H0 is the sole deliberate duration extension: it must
        # not be sent through FormationProfile's positive-HOLD validator.
        expected = asdict(GentleProfile())
        expected.update(version=VERSION, acceleration_min_mps2=self.acceleration_min_mps2,
                        hold_s=self.hold_s)
        values = asdict(self)
        if (values != expected or type(self.readiness_samples) is not int
                or any(type(v) is bool for v in values.values())):
            raise ValueError("all nonfactor fields must equal the frozen gentle-wall profile")

    def factors(self):
        return dict(acceleration_min_mps2=self.acceleration_min_mps2, hold_s=self.hold_s,
                    acceleration_change_limit_mps3=self.acceleration_change_limit_mps3)


class BrakeHoldPolicy(GentleWallPolicy):
    """Use the original physical policy with an explicitly selected profile.

    No global monkeypatch, copied controller, simulator access or damping change.
    The extra provenance keys are namespaced to avoid PolicyOutput/evaluator
    fields such as assignment, phase and wall_commands.
    """

    def __init__(self, job, registration):
        super().__init__(deepcopy(job), registration)
        self.factor_profile = BrakeHoldProfile(**self.job["policy_spec"]["profile"])

    def __call__(self, context):
        now = context.now_s
        if (now >= self.job["decision_time_s"]
                and self.proposal is None and self.terminal is None):
            # Preserve the base policy's clock and unavailable-proposal behavior
            # while replacing only its hard-coded GentleProfile constructor.
            if self.last_time is not None and now != self.last_time + 1:
                raise ValueError("gentle wall requires the exact one-second clock")
            if now != self.job["decision_time_s"]:
                raise ValueError("no late or replacement opportunity")
            spec = self.job["policy_spec"]
            try:
                self.proposal = make_proposal(tuple(context.reports), now_s=now,
                    required_lanes=tuple(spec["required_lanes"]), member_ids=tuple(spec["member_ids"]),
                    profile=self.factor_profile)
            except ValueError as exc:
                self.terminal = ("refused", "proposal_unavailable:" + str(exc))
        output = super().__call__(context)
        details = dict(output.details)
        details.update(brake_hold_study_version=VERSION,
                       brake_hold_factors=self.factor_profile.factors())
        return replace(output, details=details)


def policy_factory(job, registration):
    return BrakeHoldPolicy(job, registration)
