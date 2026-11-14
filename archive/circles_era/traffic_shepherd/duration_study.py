"""Versioned acquired HOLD+5 release contrast over the unchanged V2 controller.

The H20 proposal and every prebranch control decision remain unchanged. Only
after validating the registered acquired branch does this wrapper set the
existing exit-start clock, before calling the underlying controller once.
There is no phantom HOLD command, native acceleration estimate or new exit law.
"""
from copy import deepcopy
from dataclasses import replace
from math import isfinite

from traffic_shepherd.cohort_study import ReportSelectedWall, acquired_evidence
from traffic_shepherd.formation import assess_formation, _local_clearance_problem
from traffic_shepherd.lifecycle_wall import LifecycleProfile, LifecycleWallPolicy, remaining_plan, VERSION as WALL_VERSION
from traffic_shepherd.paired_experiments import PolicyInput, digest

VERSION = 'registered_hold_offset_release/1'
HOLD_OFFSET_S = 5.
HASH_FIELDS = ('branch_evidence_sha256','previous_wall_commands_sha256',
               'proposal_sha256','source_manifest_sha256')
BRANCH_FIELDS = frozenset(('version','hold_offset_s','expected_hold_start_s','branch_time_s',*HASH_FIELDS))


class DurationAssignmentError(ValueError):
    """Invalid registered branch: preserve evidence and STOP, no later retry."""


def validate_branch_spec(spec):
    if set(spec) != BRANCH_FIELDS or spec['version'] != VERSION or spec['hold_offset_s'] != HOLD_OFFSET_S:
        raise ValueError('One exact registered HOLD+5 contrast required')
    times = (spec['hold_offset_s'],spec['expected_hold_start_s'],spec['branch_time_s'])
    if (any(type(t) not in (int,float) or not isfinite(t) or t < 0 or t % 1 for t in times)
            or spec['branch_time_s'] != spec['expected_hold_start_s']+HOLD_OFFSET_S):
        raise ValueError('Branch must be exactly actual committed HOLD plus five seconds')
    for key in HASH_FIELDS:
        value = spec[key]
        if not isinstance(value,str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Branch hashes must bind exact immutable evidence: '+key)
    return deepcopy(spec)


def make_branch_spec(*,context,expected_hold_start_s,previous_wall_commands,
                     proposal_sha256,source_manifest_sha256):
    return validate_branch_spec(dict(version=VERSION,hold_offset_s=HOLD_OFFSET_S,
        expected_hold_start_s=expected_hold_start_s,branch_time_s=context.now_s,
        branch_evidence_sha256=digest(acquired_evidence(context)),
        previous_wall_commands_sha256=digest(dict(previous_wall_commands)),
        proposal_sha256=proposal_sha256,source_manifest_sha256=source_manifest_sha256))


def continuation_factory(job,registration):
    """The original complete continuation, not an unprotected forced H20 arm."""
    original = deepcopy(job)
    original['policy_spec'].pop('duration_branch',None)
    if 'cohort_selection' in original['policy_spec']:
        return ReportSelectedWall(original,registration)
    return LifecycleWallPolicy(original,registration)


def _inner(base):
    if isinstance(base,LifecycleWallPolicy):
        return base
    if isinstance(base,ReportSelectedWall):
        return base.delegate
    raise TypeError('Only the unchanged original or report-selected V2 controller')


class DurationBranchPolicy:
    def __init__(self,job,registration):
        self.job = deepcopy(job)
        self.spec = validate_branch_spec(self.job['policy_spec']['duration_branch'])
        profile = LifecycleProfile(**self.job['policy_spec']['profile'])
        if WALL_VERSION != 'finite_lifecycle_wall/2' or profile.hold_s != 20.:
            raise ValueError('Duration contrast preserves the exact original H20 proposal')
        if not (self.job['decision_time_s'] <= self.spec['expected_hold_start_s']
                <= self.job['decision_time_s']+profile.assembly_deadline_s):
            raise ValueError('Committed HOLD must lie within the original assembly deadline')
        if 'duration_s' in job and self.spec['branch_time_s'] >= job['duration_s']:
            raise ValueError('Registered duration branch must precede the common horizon')
        self.base = continuation_factory(job,registration)
        self.last_output_phase = None
        self.receipt = dict(version=VERSION,hold_offset_s=HOLD_OFFSET_S,
            registered_hold_start_s=self.spec['expected_hold_start_s'],
            registered_branch_time_s=self.spec['branch_time_s'],
            source_manifest_sha256=self.spec['source_manifest_sha256'],
            original_profile_hold_s=20.,deliberate_release=True,feasibility_triggered=False,
            requested=False,verified=False,request_time_s=None,actual_hold_start_s=None,
            exit_started_at_s=None,terminal_at_s=None,terminal_phase=None)

    def _verify_branch(self,context,inner):
        now = context.now_s
        if (inner is None or inner.proposal is None or inner.terminal is not None
                or inner.taper_at is not None or self.last_output_phase != 'hold'
                or inner.ready_at != self.spec['expected_hold_start_s']
                or now != inner.ready_at+HOLD_OFFSET_S or inner.last_time != now-1):
            raise DurationAssignmentError('Registered branch lacks the same active committed HOLD')
        proposal,p = inner.proposal,inner.profile
        if (proposal.fingerprint != self.spec['proposal_sha256']
                or digest(acquired_evidence(context)) != self.spec['branch_evidence_sha256']
                or digest(inner.previous_commands) != self.spec['previous_wall_commands_sha256']):
            raise DurationAssignmentError('Registered branch proposal/report/prior-command evidence changed')
        assessment = assess_formation(proposal,context.reports,now_s=now)
        if (not assessment.approved or assessment.initial_or_current_span_m > p.formed_span_m
                or assessment.speed_spread_mps > p.formed_speed_spread_mps):
            raise DurationAssignmentError('Current acquired branch geometry is not a valid compact HOLD')
        members = set(proposal.member_ids)
        known = set(context.damping_commands)|set(context.damping_native_ids)
        if not members <= known or any(not isfinite(context.damping_commands[v])
                for v in members if v in context.damping_commands):
            raise DurationAssignmentError('Current identical local damping/native references unavailable')
        common = dict(now_s=now,previous_commands=inner.previous_commands,
            readiness_count=inner.readiness,ready_at_s=inner.ready_at,exit_started_at_s=None)
        continuation = remaining_plan(proposal,context.reports,**common)
        exit_plan = remaining_plan(proposal,context.reports,**common,exit_now=True)
        if not continuation['feasible'] or not exit_plan['feasible']:
            raise DurationAssignmentError('Registered continue/release contrast is not feasible on acquired reports')
        # The finite model does not extrapolate follower behavior. Check the
        # exact first coast command through the existing local one-step guard,
        # rather than allowing branch eligibility to bypass that protection.
        if exit_plan['planned_samples']:
            first = exit_plan['planned_samples'][0]['commands_mps2']
            frames = {r.frame.vehicle_id:r.frame for r in context.reports}
            if any(_local_clearance_problem(frames[v],p,now,acceleration=first[v]) for v in members):
                raise DurationAssignmentError('Registered first coast command fails existing local clearance')
        return dict(continuation_plan=continuation,exit_now_plan=exit_plan,
            continuation_feasible=True,exit_now_feasible=True,
            proposal_sha256=proposal.fingerprint,
            branch_evidence_sha256=digest(acquired_evidence(context)),
            previous_wall_commands=dict(inner.previous_commands),
            previous_wall_commands_sha256=digest(inner.previous_commands))

    def __call__(self,context):
        if not isinstance(context,PolicyInput):
            raise TypeError('Only acquired PolicyInput may determine a duration branch')
        now = context.now_s
        branch = now == self.spec['branch_time_s'] and not self.receipt['requested']
        if now >= self.spec['branch_time_s'] and not self.receipt['requested']:
            if not branch:
                raise DurationAssignmentError('Registered duration branch was missed; no late replacement')
            inner = _inner(self.base)
            proof = self._verify_branch(context,inner)
            # This is the sole intervention: invoke the existing exit path on
            # THIS tick using unchanged prior-tick command state. Do not call
            # the controller first and then replace its resulting HOLD command.
            inner.taper_at = now
            self.receipt.update(proof,requested=True,verified=True,request_time_s=now,
                actual_hold_start_s=inner.ready_at,exit_started_at_s=now)
        output = self.base(context)
        self.last_output_phase = output.phase
        inner = _inner(self.base)
        if self.receipt['requested']:
            if inner.ready_at != self.receipt['actual_hold_start_s'] or inner.taper_at != self.receipt['request_time_s']:
                raise DurationAssignmentError('Duration switch changed original readiness or requested exit clock')
            if output.phase in {'release','abort'} and self.receipt['terminal_at_s'] is None:
                self.receipt.update(terminal_at_s=now,terminal_phase=output.phase)
        # Full acquired plans occur once at the branch; later telemetry keeps
        # a compact receipt. All original controller/core diagnostics survive.
        details = dict(self.receipt)
        if not branch:
            for key in ('continuation_plan','exit_now_plan','previous_wall_commands'):
                details.pop(key,None)
        return replace(output,details=dict(output.details,duration_branch=details))


def policy_factory(job,registration):
    return DurationBranchPolicy(job,registration)


def branch_receipt(policy):
    if not isinstance(policy,DurationBranchPolicy):
        raise TypeError('Duration policy instance required for its branch receipt')
    return deepcopy(policy.receipt)


def validate_branch_receipt(receipt,spec, *, require_terminal=False):
    """Assignment verification, separate from measured exit/safety efficacy.

    A later protective abort is an executed release-attempt outcome, not a
    successful controlled exit. The independent lifecycle scorer decides that.
    """
    spec = validate_branch_spec(spec)
    expected = dict(version=VERSION,hold_offset_s=HOLD_OFFSET_S,requested=True,verified=True,
        registered_hold_start_s=spec['expected_hold_start_s'],actual_hold_start_s=spec['expected_hold_start_s'],
        registered_branch_time_s=spec['branch_time_s'],request_time_s=spec['branch_time_s'],
        exit_started_at_s=spec['branch_time_s'],source_manifest_sha256=spec['source_manifest_sha256'],
        original_profile_hold_s=20.,deliberate_release=True,feasibility_triggered=False,
        continuation_feasible=True,exit_now_feasible=True,
        branch_evidence_sha256=spec['branch_evidence_sha256'],
        previous_wall_commands_sha256=spec['previous_wall_commands_sha256'],proposal_sha256=spec['proposal_sha256'])
    if any(receipt.get(k) != value for k,value in expected.items()):
        raise DurationAssignmentError('Duration receipt does not prove the registered acquired release request')
    if require_terminal and (receipt.get('terminal_phase') not in {'release','abort'}
            or not isinstance(receipt.get('terminal_at_s'),(int,float))
            or receipt['terminal_at_s'] < spec['branch_time_s']):
        raise DurationAssignmentError('Duration receipt lacks an executed terminal outcome')
    return True
