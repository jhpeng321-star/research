from copy import deepcopy
from dataclasses import asdict,replace

import pytest

from traffic_shepherd.cohort_study import LANES,bind_selected_job,acquired_evidence
from traffic_shepherd.lifecycle_wall import LifecycleProfile,LifecycleWallPolicy
from traffic_shepherd.paired_experiments import PolicyInput,digest
from traffic_shepherd.reporting import CAVReport
from traffic_shepherd.sensing import SensorFrame

from traffic_shepherd import duration_study as duration


def packets(now=5.,speed=23.345754604179337,acceleration=.31855186227641497):
    return tuple(CAVReport(SensorFrame(v,now,speed,acceleration,24.,i,
        distance_to_merge_m=2400.+10*i,front_coverage_m=100.,rear_coverage_m=100.,
        lane_id=lane,position_m=1100.-10*i),now,'wall') for i,(v,lane) in enumerate(zip(('a','b'),LANES)))


def context(rows):
    now = rows[0].frame.timestamp_s
    return PolicyInput(now,rows,tuple((now,r) for r in rows),{},frozenset(r.frame.vehicle_id for r in rows))


def job(start=5):
    return dict(id='source',arm='wall',decision_time_s=start,policy_spec=dict(
        required_lanes=list(LANES),member_ids=['a','b'],profile=asdict(LifecycleProfile())))


def advance(rows,output):
    answer = []
    for report in rows:
        f = report.frame
        a = output.wall_commands[f.vehicle_id]
        dx = f.speed_mps+.5*a
        answer.append(replace(report,sent_at_s=f.timestamp_s+1,frame=replace(f,
            timestamp_s=f.timestamp_s+1,speed_mps=f.speed_mps+a,acceleration_mps2=a,
            position_m=f.position_m+dx,distance_to_merge_m=f.distance_to_merge_m-dx)))
    return tuple(answer)


def source_trace():
    source_job = job()
    source,frames = LifecycleWallPolicy(source_job,{}),packets()
    contexts,outputs = {},{}
    branch_spec = None
    for now in range(5,70):
        ctx = context(frames)
        if source.ready_at is not None and now == source.ready_at+5:
            branch_spec = duration.make_branch_spec(context=ctx,expected_hold_start_s=source.ready_at,
                previous_wall_commands=source.previous_commands,proposal_sha256=source.proposal.fingerprint,
                source_manifest_sha256='a'*64)
        contexts[now] = ctx
        output = source(ctx)
        outputs[now] = output
        if not output.wall_commands:
            break
        frames = advance(frames,output)
    assert branch_spec is not None
    branch_job = deepcopy(source_job)
    branch_job['policy_spec']['duration_branch'] = branch_spec
    return branch_job,contexts,outputs


def armed():
    branch_job,contexts,outputs = source_trace()
    policy = duration.policy_factory(branch_job,{})
    at = int(branch_job['policy_spec']['duration_branch']['branch_time_s'])
    for now in range(5,at):
        policy(contexts[now])
    return policy,contexts[at],outputs[at]


def test_exact_prefix_preserves_original_h20_proposal_and_all_core_outputs():
    branch_job,contexts,outputs = source_trace()
    policy = duration.policy_factory(branch_job,{})
    branch = branch_job['policy_spec']['duration_branch']['branch_time_s']
    for now in range(5,int(branch)):
        output = policy(contexts[now]).payload()
        output.pop('duration_branch')
        assert output == outputs[now].payload()
    inner = duration._inner(policy.base)
    assert inner.profile.hold_s == 20
    assert inner.ready_at == branch-5
    assert inner.taper_at is None
    assert inner.proposal.fingerprint == branch_job['policy_spec']['duration_branch']['proposal_sha256']


def test_branch_uses_actual_prior_commands_without_phantom_hold_resolution():
    policy,ctx,continued = armed()
    inner = duration._inner(policy.base)
    prior = dict(inner.previous_commands)
    assert any(abs(a)>1e-8 for a in prior.values())
    expected_exit = duration.remaining_plan(inner.proposal,ctx.reports,now_s=ctx.now_s,
        previous_commands=prior,readiness_count=inner.readiness,ready_at_s=inner.ready_at,exit_now=True)
    output = policy(ctx)
    assert output.phase == 'taper'
    assert dict(output.wall_commands) == expected_exit['planned_samples'][0]['commands_mps2']
    assert dict(output.wall_commands) != dict(continued.wall_commands)
    assert set(output.wall_commands.values()) == {0.}
    receipt = output.details['duration_branch']
    assert receipt['previous_wall_commands'] == prior
    assert receipt['previous_wall_commands_sha256'] == digest(prior)
    assert receipt['request_time_s'] == inner.ready_at+5
    assert output.details['early_exit_reason'] is None
    assert receipt['deliberate_release'] and not receipt['feasibility_triggered']
    assert receipt['continuation_feasible'] and receipt['exit_now_feasible']


def test_measured_coast_then_handback_no_reengagement_and_no_clock_reset():
    policy,ctx,_ = armed()
    inner = duration._inner(policy.base)
    ready,readiness,proposal = inner.ready_at,inner.readiness,inner.proposal
    output = policy(ctx)
    next_frames = advance(ctx.reports,output)
    final = policy(context(next_frames))
    assert final.phase == 'release' and not final.wall_commands
    assert final.details['actual_hold_elapsed_s'] == 5
    assert inner.ready_at == ready and inner.readiness == readiness and inner.proposal == proposal
    receipt = duration.branch_receipt(policy)
    assert receipt['terminal_at_s'] == ctx.now_s+1 and receipt['terminal_phase'] == 'release'
    assert duration.validate_branch_receipt(receipt,policy.spec,require_terminal=True)
    later = tuple(replace(r,sent_at_s=r.sent_at_s+1,frame=replace(r.frame,timestamp_s=r.frame.timestamp_s+1)) for r in next_frames)
    repeated = policy(context(later))
    assert repeated.phase == 'release' and not repeated.wall_commands
    assert duration.branch_receipt(policy)['terminal_at_s'] == ctx.now_s+1


def test_already_zero_measured_coast_can_release_without_invented_extra_command():
    policy,ctx,_ = armed()
    inner = duration._inner(policy.base)
    inner.previous_commands = {'a':0.,'b':0.}
    rows = tuple(replace(r,frame=replace(r.frame,acceleration_mps2=0.)) for r in ctx.reports)
    coast_ctx = context(rows)
    policy.spec = duration.make_branch_spec(context=coast_ctx,expected_hold_start_s=inner.ready_at,
        previous_wall_commands=inner.previous_commands,proposal_sha256=inner.proposal.fingerprint,
        source_manifest_sha256='a'*64)
    output = policy(coast_ctx)
    assert output.phase == 'release' and not output.wall_commands
    assert duration.branch_receipt(policy)['exit_now_plan']['remaining_total_s'] == 0


@pytest.mark.parametrize('field',duration.HASH_FIELDS[:-1])
def test_assignment_mismatch_fails_before_any_inner_mutation(field):
    policy,ctx,_ = armed()
    inner = duration._inner(policy.base)
    before = deepcopy(inner.__dict__)
    policy.spec[field] = '0'*64
    with pytest.raises(duration.DurationAssignmentError,match='evidence changed'):
        policy(ctx)
    assert inner.__dict__ == before
    assert not duration.branch_receipt(policy)['requested']


@pytest.mark.parametrize('which',['continuation','exit'])
def test_infeasible_branch_never_mutates_state_or_selects_later_offset(monkeypatch,which):
    policy,ctx,_ = armed()
    inner = duration._inner(policy.base)
    before = deepcopy(inner.__dict__)
    original = duration.remaining_plan
    def altered(*args,**kwargs):
        plan = original(*args,**kwargs)
        if bool(kwargs.get('exit_now')) == (which=='exit'):
            return dict(plan,feasible=False)
        return plan
    monkeypatch.setattr(duration,'remaining_plan',altered)
    with pytest.raises(duration.DurationAssignmentError,match='not feasible'):
        policy(ctx)
    assert inner.__dict__ == before and not duration.branch_receipt(policy)['requested']


def test_missing_current_member_report_is_ineligible_even_with_matching_hash():
    policy,ctx,_ = armed()
    inner = duration._inner(policy.base)
    before = deepcopy(inner.__dict__)
    missing = context(ctx.reports[:1])
    policy.spec['branch_evidence_sha256'] = digest(acquired_evidence(missing))
    with pytest.raises(duration.DurationAssignmentError,match='geometry'):
        policy(missing)
    assert inner.__dict__ == before


def test_missed_branch_never_calls_late_release():
    policy,ctx,_ = armed()
    inner = duration._inner(policy.base)
    before = deepcopy(inner.__dict__)
    late = tuple(replace(r,sent_at_s=r.sent_at_s+1,frame=replace(r.frame,timestamp_s=r.frame.timestamp_s+1)) for r in ctx.reports)
    with pytest.raises(duration.DurationAssignmentError,match='missed'):
        policy(context(late))
    assert inner.__dict__ == before


def test_no_genuine_hold_at_branch_is_invalid_assignment():
    policy,ctx,_ = armed()
    inner = duration._inner(policy.base)
    inner.ready_at = None
    before = deepcopy(inner.__dict__)
    with pytest.raises(duration.DurationAssignmentError,match='active committed HOLD'):
        policy(ctx)
    assert inner.__dict__ == before


def test_unexecuted_receipt_cannot_be_accepted_as_duration_evidence():
    policy,_,_ = armed()
    with pytest.raises(duration.DurationAssignmentError,match='does not prove'):
        duration.validate_branch_receipt(duration.branch_receipt(policy),policy.spec)


def test_profile_h5_or_different_offset_is_rejected_before_any_execution():
    branch_job,_,_ = source_trace()
    changed = deepcopy(branch_job)
    changed['policy_spec']['profile']['hold_s'] = 5
    with pytest.raises(ValueError):
        duration.policy_factory(changed,{})
    changed = deepcopy(branch_job)
    changed['policy_spec']['duration_branch']['hold_offset_s'] = 6.
    with pytest.raises(ValueError):
        duration.policy_factory(changed,{})


def test_report_selected_c_base_is_retained_without_bypassing_selector():
    rows = packets(600.)
    ctx = context(rows)
    evidence = acquired_evidence(ctx)
    original = job(start=600)
    original.update(seed=4108,loading=2700,penetration=.2,duration_s=2400)
    bound,choice = bind_selected_job(original,dict(evidence=evidence,decision_time_s=600.,
        predecision_report_evidence_sha256=digest(evidence)))
    # Only entry delegation is tested here; the actual committed clock is
    # established by the separately verified source C trajectory before launch.
    bound['policy_spec']['duration_branch'] = dict(version=duration.VERSION,hold_offset_s=5.,
        expected_hold_start_s=620.,branch_time_s=625.,branch_evidence_sha256='a'*64,
        previous_wall_commands_sha256='a'*64,proposal_sha256=choice['selected']['proposal_sha256'],
        source_manifest_sha256='a'*64)
    out = duration.policy_factory(bound,{})(ctx)
    assert out.details['cohort_selection_verified']
    assert out.details['cohort_selection_sha256'] == choice['selection_sha256']
    assert out.phase == 'assembly' and out.wall_commands
