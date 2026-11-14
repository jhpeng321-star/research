"""Versioned diagnostic two-activation wall probe (no benefit gate)."""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from math import isfinite

from .cohort_study import LANES, ReportSelectedWall, acquired_evidence
from .formation import make_proposal
from .lifecycle_wall import LifecycleProfile, LifecycleWallPolicy, remaining_plan
from .paired_experiments import PolicyInput, digest

VERSION = "multi_wall_probe/1"
EXTEND_S = 20.0
BRANCH_AFTER_HOLD_S = 20.0


@dataclass(frozen=True)
class ProbeExtensionProfile(LifecycleProfile):
    version: str = VERSION
    hold_s: float = 40.0

    def __post_init__(self):
        expected = asdict(LifecycleProfile())
        expected.update(version=VERSION, hold_s=40.0)
        if asdict(self) != expected or type(self.readiness_samples) is not int:
            raise ValueError("one frozen multi-wall 40-second continuation profile required")


def received_reports(context):
    """Current evidence must have an actual, nonfuture receipt in history."""
    receipts = {digest(r.payload()): t for t, r in context.history
                if t <= context.now_s and 0 <= context.now_s-t <= 1}
    return tuple(r for r in context.reports
                 if 0 <= context.now_s-r.frame.timestamp_s <= 1
                 and digest(r.payload()) in receipts)


def select_upstream(reports, *, now_s, excluded_ids, minimum_distance_m):
    """Select one disjoint, upstream pair from fresh acquired reports."""
    reports = tuple(sorted(reports, key=lambda r: r.frame.vehicle_id))
    if (not reports or len({r.frame.vehicle_id for r in reports}) != len(reports)
            or any(r.sent_at_s > now_s or r.frame.timestamp_s > now_s for r in reports)):
        return dict(selected=None, reason="second_selection_requires_fresh_current_reports",
                    candidate_count=0, inventory=[])
    candidates_reports = tuple(r for r in reports if 0 <= now_s-r.frame.timestamp_s <= 1.0)
    profile = LifecycleProfile()
    candidates = []
    for lane in LANES:
        candidates.append([r for r in candidates_reports if r.frame.lane_id == lane
                           and r.frame.vehicle_id not in excluded_ids
                           and r.frame.distance_to_merge_m is not None
                           and r.frame.distance_to_merge_m >= 200.
                           and r.frame.distance_to_merge_m >= minimum_distance_m])
    inventory = []
    for left in candidates[0]:
        for right in candidates[1]:
            chosen = (left, right)
            try:
                proposal = make_proposal(reports, now_s=now_s, required_lanes=LANES,
                                         member_ids=tuple(r.frame.vehicle_id for r in chosen),
                                         profile=profile)
                plan = remaining_plan(proposal, reports, now_s=now_s,
                                      previous_commands={}, readiness_count=0)
                inventory.append(dict(member_ids=list(proposal.member_ids),
                                      proposal_sha256=proposal.fingerprint,
                                      feasible=bool(plan["feasible"]), reason=plan["reason"],
                                      predicted_assembly_s=plan.get("remaining_assembly_s"),
                                      plan=plan))
            except (ValueError, KeyError) as exc:
                inventory.append(dict(member_ids=[r.frame.vehicle_id for r in chosen],
                                      feasible=False, reason=str(exc)))
    feasible = [r for r in inventory if r["feasible"]]
    selected = min(feasible, key=lambda r: (r.get("predicted_assembly_s"), tuple(r["member_ids"]))) if feasible else None
    return dict(selected=selected, reason=None if selected else "no_finite_feasible_distinct_upstream_pair",
                candidate_count=len(inventory), inventory=inventory)


class MultiWallProbePolicy:
    """One first H20 activation plus exactly one diagnostic continuation/replacement."""
    def __init__(self, job, registration, *, mode, first_factory=ReportSelectedWall):
        if mode not in {"same_cohort_extend20", "distinct_upstream_h20"}:
            raise ValueError("unsupported multi-wall probe mode")
        self.job, self.registration, self.mode = deepcopy(job), registration, mode
        self.first = first_factory(self.job, registration)
        self.second = None
        self.first_terminal = None
        self.second_attempted = False
        self.first_engaged = False
        self.receipt = dict(version=VERSION, mode=mode, activation_index=0,
                            second_activation_attempted=False, branch=None,
                            selection=None, first_terminal=None)

    def _inner(self):
        return self.first.delegate if isinstance(self.first, ReportSelectedWall) else self.first

    def _extension(self, context):
        delegate = self._inner()
        if (self.receipt['branch'] is not None or delegate is None
                or delegate.terminal is not None or delegate.ready_at is None
                or delegate.taper_at is not None):
            return
        if context.now_s != delegate.ready_at + BRANCH_AFTER_HOLD_S:
            return
        original = delegate.proposal
        ext_values = asdict(delegate.profile)
        ext_values.update(version=VERSION, hold_s=40.0)
        ext_profile = ProbeExtensionProfile(**ext_values)
        ext_proposal = replace(original, profile=ext_profile)
        plan_old = remaining_plan(original, context.reports, now_s=context.now_s,
                                  previous_commands=delegate.previous_commands,
                                  readiness_count=delegate.readiness, ready_at_s=delegate.ready_at)
        plan_new = remaining_plan(ext_proposal, context.reports, now_s=context.now_s,
                                  previous_commands=delegate.previous_commands,
                                  readiness_count=delegate.readiness, ready_at_s=delegate.ready_at)
        bound = digest([r.payload() for r in context.reports])
        branch = dict(branch_time_s=context.now_s, reports_sha256=bound,
                      original_proposal_sha256=original.fingerprint,
                      proposed_proposal_sha256=ext_proposal.fingerprint,
                      original_profile=asdict(delegate.profile), proposed_profile=asdict(ext_profile),
                      original_plan=plan_old, proposed_plan=plan_new,
                      selected="extend" if plan_new.get("feasible") else "decline")
        self.receipt["branch"] = branch
        if plan_new.get("feasible"):
            delegate.proposal, delegate.profile = ext_proposal, ext_profile

    def _start_second(self, context):
        self.second_attempted = True
        self.receipt["second_activation_attempted"] = True
        first_ids = set(self.job["policy_spec"]["member_ids"])
        fresh = received_reports(context)
        frames = {r.frame.vehicle_id: r.frame for r in fresh}
        first_d = [frames[v].distance_to_merge_m for v in first_ids if v in frames]
        if (len(first_d) != len(first_ids) or any(d is None or not isfinite(d) or d < 0 for d in first_d)):
            self.receipt["selection"] = dict(selected=None,
                reason="first_member_current_report_unavailable", selection_time_s=context.now_s,
                evidence_sha256=digest(acquired_evidence(context)))
            return
        selection = select_upstream(fresh, now_s=context.now_s,
                                    excluded_ids=first_ids, minimum_distance_m=max(first_d))
        self.receipt["selection"] = dict(selection,
            reports_sha256=digest([r.payload() for r in context.reports]),
            eligible_reports_sha256=digest([r.payload() for r in fresh]),
            evidence_sha256=digest(acquired_evidence(context)),
            minimum_distance_m=max(first_d),
            selection_time_s=context.now_s)
        chosen = selection.get("selected")
        if chosen is None:
            return
        job = deepcopy(self.job)
        job["id"] = job["id"] + "_second"
        job["decision_time_s"] = context.now_s
        job["policy_spec"]["member_ids"] = chosen["member_ids"]
        job['policy_spec'].pop('cohort_selection', None)
        job["policy_spec"]["probe_activation"] = 1
        self.second = LifecycleWallPolicy(job, self.registration)

    def __call__(self, context: PolicyInput):
        if not isinstance(context, PolicyInput):
            raise TypeError("acquired PolicyInput only")
        if self.second is not None:
            out = self.second(context)
            return replace(out, details=dict(out.details, activation_index=1,
                                              activation_mode=self.mode,
                                              multi_wall_probe=deepcopy(self.receipt)))
        # The first terminal row is emitted by the original delegate unchanged
        # except for additive probe metadata.
        if self.mode == "same_cohort_extend20":
            self._extension(context)
        out = self.first(context)
        if out.wall_commands:
            self.first_engaged = True
        details = dict(out.details, activation_index=0, activation_mode=self.mode,
                       multi_wall_probe=self.receipt)
        if out.phase in {"release", "abort", "refused"} and self.first_terminal is None:
            self.first_terminal = dict(time_s=context.now_s, phase=out.phase, reason=out.reason)
            self.receipt["first_terminal"] = self.first_terminal
            if self.mode == "same_cohort_extend20" and self.receipt["branch"] is None:
                self.receipt["branch"] = dict(branch_time_s=context.now_s,
                    selected="unavailable_before_scheduled_branch",
                    reason="first_activation_terminal_before_extension_branch",
                    reports_sha256=digest([r.payload() for r in context.reports]))
        # An executed wall may hand back normally or protectively. A NEW,
        # disjoint proposal must independently pass full admission. An initial
        # refusal never counts as an executed first wall.
        if (self.mode == "distinct_upstream_h20" and out.phase in {"release", "abort"}
                and getattr(self, 'first_engaged', False)
                and self.first_terminal and not self.second_attempted
                and context.now_s == self.first_terminal["time_s"] + 1):
            self._start_second(context)
            if self.second is not None:
                second = self.second(context)
                expected = self.receipt["selection"]["selected"]["proposal_sha256"]
                if self.second.proposal is None or self.second.proposal.fingerprint != expected:
                    self.receipt["selection"]["proposal_binding"] = "mismatch"
                    self.second = None
                    return replace(out, details=dict(details, multi_wall_probe=deepcopy(self.receipt)))
                self.receipt["selection"]["proposal_binding"] = "verified"
                return replace(second, details=dict(second.details,
                    activation_index=1, activation_mode=self.mode,
                    multi_wall_probe=deepcopy(self.receipt)))
        return replace(out, details=deepcopy(details))


def policy_factory(job, registration, *, mode):
    return MultiWallProbePolicy(job, registration, mode=mode)
