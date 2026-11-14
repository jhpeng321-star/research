from copy import deepcopy
from dataclasses import asdict,replace
import json
from pathlib import Path

import pytest

from traffic_shepherd.lifecycle_wall import LifecycleProfile
from traffic_shepherd.paired_experiments import PolicyInput,digest
from traffic_shepherd.reporting import CAVReport
from traffic_shepherd.sensing import SensorFrame
from traffic_shepherd.config import ROOT
from traffic_shepherd import cohort_study as study

def reports(now=600.):
    return tuple(CAVReport(SensorFrame(v,now,24.,0.,30.,i,
        distance_to_merge_m=2300.,front_coverage_m=100.,rear_coverage_m=100.,
        lane_id=lane,position_m=300.),now,'native')
        for i,(v,lane) in enumerate(zip(('a','b'),study.LANES)))


def context(rows=None,now=600.):
    rows = reports(now) if rows is None else rows
    return PolicyInput(now,rows,tuple((now,r) for r in rows),{},
                       frozenset(r.frame.vehicle_id for r in rows))


def binding():
    ctx = context()
    evidence = study.acquired_evidence(ctx)
    snapshot = dict(evidence=evidence,decision_time_s=600.,predecision_report_evidence_sha256=digest(evidence))
    job = dict(id='base4108',seed=4108,loading=2700,penetration=.2,duration_s=2400,
        decision_time_s=600,arm='wall',policy_spec=dict(required_lanes=list(study.LANES),
            member_ids=['a','b'],profile=asdict(LifecycleProfile())))
    job,selection = study.bind_selected_job(job,snapshot)
    return job,selection,ctx


def test_runtime_rederives_from_reports_and_delegates_exact_law():
    job,selection,ctx = binding()
    policy = study.policy_factory(job,{})
    out = policy(ctx)
    assert out.phase == 'assembly' and out.wall_commands
    assert out.details['cohort_selection_verified']
    assert out.details['cohort_selection_sha256'] == selection['selection_sha256']
    baseline = study.LifecycleWallPolicy(job,{})(ctx)
    assert dict(out.wall_commands) == dict(baseline.wall_commands)
    assert out.proposal_sha256 == baseline.proposal_sha256


@pytest.mark.parametrize('field',['evidence_sha256','selection_sha256','proposal_sha256'])
def test_binding_mismatch_refuses_atomically_and_cannot_reengage(field):
    job,_,ctx = binding()
    job['policy_spec']['cohort_selection'][field] = '0'*64
    policy = study.policy_factory(job,{})
    out = policy(ctx)
    assert out.phase == 'refused' and not out.wall_commands
    out = policy(context(now=601.))
    assert out.phase == 'refused' and not out.wall_commands


def test_no_seed_feature_and_no_override_before_decision():
    job,_,ctx = binding()
    changed = deepcopy(job)
    changed['seed'],changed['loading'],changed['penetration'] = 99999,9999,.01
    a = study.policy_factory(job,{})
    b = study.policy_factory(changed,{})
    assert not a(context(now=599.)).wall_commands
    assert dict(a(ctx).wall_commands) == dict(b(ctx).wall_commands)


def test_runtime_incomplete_reports_refuses_instead_of_forced_registered_cohort():
    job,_,_ = binding()
    out = study.policy_factory(job,{})(context(rows=reports()[:1]))
    assert out.phase == 'refused'
    assert out.reason == 'cohort_no_finite_feasible_pair'


def test_complete_universe_and_deterministic_tie_break(monkeypatch):
    rows = reports()
    extra = replace(rows[0],frame=replace(rows[0].frame,vehicle_id='aa'))
    monkeypatch.setattr(study,'remaining_plan',lambda *a,**kw:dict(
        feasible=True,reason='synthetic_feasible',remaining_assembly_s=8))
    result = study.select_cohort((extra,rows[1],rows[0]),now_s=600.)
    assert result['candidate_count'] == 2
    assert result['selected']['member_ids'] == ['a','b']
    reverse = study.select_cohort(tuple(reversed((extra,rows[1],rows[0]))),now_s=600.)
    assert reverse == result


def test_saved_case_choices_and_exact_universe_match_predeclared_audit():
    expected = json.loads((ROOT/'output/lifecycle_repair_v2/PART_C_PREREQUISITE.json').read_text())
    for seed in (4107,4108):
        case = expected['cases'][str(seed)]
        snapshot = json.loads(Path(case['snapshot_path']).read_text())['600']
        choice = study.snapshot_selection(snapshot)
        assert choice['candidate_count'] == case['whole_approach_pairs']
        assert choice['selected']['member_ids'] == case['selected_whole_approach']['member_ids']


def test_reuse_cannot_be_approved_by_matching_members_only():
    job,choice,_ = binding()
    with pytest.raises(ValueError):
        study.validate_reuse_proof(dict(complete=True),selection=choice,
            source_manifest_sha256='x',source_job=job)


def test_budget_cannot_expand_to_two_new_jobs():
    job,_,_ = binding()
    other = deepcopy(job)
    other.update(seed=4107,id='cohort4107')
    bindings = {job['id']:dict(mode='simulate'),other['id']:dict(mode='verified_reuse')}
    assert study.allocation_contract([job,other],bindings)['maximum_simulated_seconds'] == 2400
    bindings[other['id']]['mode'] = 'simulate'
    with pytest.raises(ValueError):
        study.allocation_contract([job,other],bindings)


def test_physical_prefix_mismatch_is_not_hidden_by_acquired_evidence_match():
    left = {k:'same' for k in study.PREFIX_KEYS}
    right = dict(left,predecision_state_sha256='changed')
    with pytest.raises(ValueError):
        study.matched_prefix(left,right)
