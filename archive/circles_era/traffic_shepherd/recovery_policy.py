"""OE-28: replace only the acquired H5 exit with bounded formation recovery."""
from dataclasses import replace

from .duration_study import DurationBranchPolicy, DurationAssignmentError, _inner
from .paired_experiments import PolicyOutput, digest
from .recovery_release import RecoveryController

VERSION = "recovery_release/2"


class RecoveryReleasePolicy(DurationBranchPolicy):
    def __init__(self, job, registration):
        super().__init__(job, registration)
        self.recovery = None
        self.recovery_receipt = None
        self.coasting = False
        self.finished = None

    def __call__(self, context):
        now = context.now_s
        if now < self.spec["branch_time_s"]:
            # Exact original H5 policy, including its prebranch telemetry.
            return super().__call__(context)
        inner = _inner(self.base)
        if self.recovery is None:
            if now != self.spec["branch_time_s"]:
                raise DurationAssignmentError("Missed registered recovery branch")
            proof = self._verify_branch(context, inner)
            self.recovery = RecoveryController(inner.proposal, inner.previous_commands, now)
            self.recovery_receipt = dict(version=VERSION, requested_at_s=now,
                actual_hold_start_s=inner.ready_at, delivered_original_hold_s=5.,
                original_branch_proof=proof, recovery_started_at_s=None,
                fallback_at_s=None, fallback_reason=None, terminal_at_s=None)
        if self.finished is not None:
            return self._decorate(self.finished, now)
        if self.coasting:
            result = self.base(context)
            return self._capture_terminal(result, now)
        decision = self.recovery.step(context)
        self.recovery_receipt["latest"] = decision
        if decision["status"] == "fallback":
            # Recovery has not written this tick. Re-enter the unchanged coast
            # once, from the ACTUAL previous recovery command and original state.
            self.coasting = True
            inner.taper_at = now
            inner.early_exit_reason = "recovery_fallback:" + decision["reason"]
            self.recovery_receipt.update(fallback_at_s=now, fallback_reason=decision["reason"])
            return self._capture_terminal(self.base(context), now)
        if inner.last_time != now - 1:
            raise ValueError("Recovery wrapper requires consecutive control ticks")
        inner.last_time = now
        if decision["status"] == "release":
            inner.terminal = ("release", decision["reason"])
            inner.terminal_at = now
            result = PolicyOutput({}, "release", decision["reason"], inner.proposal.member_ids,
                                  inner.proposal.fingerprint, digest([r.payload() for r in context.reports]))
            return self._capture_terminal(result, now)
        if self.recovery_receipt["recovery_started_at_s"] is None:
            self.recovery_receipt["recovery_started_at_s"] = now
        inner.previous_commands = dict(decision["commands"])
        result = PolicyOutput(decision["commands"], "taper", decision["reason"],
            inner.proposal.member_ids, inner.proposal.fingerprint,
            digest([r.payload() for r in context.reports]))
        return self._decorate(result, now)

    def _capture_terminal(self, result, now):
        if result.phase in {"release", "abort", "refused"}:
            self.finished = result
            self.recovery_receipt["terminal_at_s"] = now
        return self._decorate(result, now)

    def _decorate(self, result, now):
        receipt = dict(self.recovery_receipt)
        if now != self.spec["branch_time_s"]:
            receipt.pop("original_branch_proof", None)
        end = receipt["terminal_at_s"] if receipt["terminal_at_s"] is not None else now
        receipt["elapsed_postbranch_s"] = end - receipt["requested_at_s"]
        return replace(result, details=dict(result.details, recovery_release=receipt,
            actual_hold_elapsed_s=5., assembly_completed_at_s=receipt["actual_hold_start_s"],
            override_released=not bool(result.wall_commands)))


def policy_factory(job, registration):
    return RecoveryReleasePolicy(job, registration)
