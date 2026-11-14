from copy import deepcopy
import json
import sys

import pytest


from traffic_shepherd import cohort_study as selection
from traffic_shepherd import cohort_campaign as campaign


def no_benefit(seed):
    return dict(seed=seed,benefit=dict(label='harmful'),
                lifecycle=dict(registered_maneuver_delivered=False))


def test_completed_qualified_b_prevents_unnecessary_c():
    results = [no_benefit(4107),no_benefit(4108)]
    campaign._qualified_part_b(results)
    results[0]['benefit']['label'] = 'beneficial'
    results[0]['lifecycle']['registered_maneuver_delivered'] = True
    with pytest.raises(ValueError,match='already has qualifying'):
        campaign._qualified_part_b(results)


def test_missing_b_seed_does_not_open_gate():
    with pytest.raises(ValueError,match='both exact'):
        campaign._qualified_part_b([no_benefit(4107)])


def test_incomplete_b_stops_before_creating_registration(tmp_path,monkeypatch):
    # Install only an inert replay import so prerequisite sequencing is tested.
    from types import ModuleType
    evidence = ModuleType('traffic_shepherd.cohort_evidence')
    evidence.replay_reuse = lambda *a:pytest.fail('Replay cannot precede completed B')
    monkeypatch.setitem(sys.modules,'traffic_shepherd.cohort_evidence',evidence)
    monkeypatch.setattr(campaign.paired,'current_identity',lambda:{})
    def incomplete(*args):
        raise ValueError('Complete current-V2 Part B physical verdict required')
    monkeypatch.setattr(campaign,'verified_part_b',incomplete)
    output = tmp_path/'new'
    with pytest.raises(ValueError,match='physical verdict'):
        campaign.register(output,tmp_path/'b.json')
    assert not output.exists()


def prepared_run(tmp_path,monkeypatch,verified=True,raises=False):
    job = dict(id='new4108',policy_spec=dict(profile={}))
    data = dict(jobs=[job],bindings={'new4108':dict(mode='simulate')},registration_sha256='registered')
    monkeypatch.setattr(campaign,'read_registration',lambda *a,**kw:(tmp_path,data))
    monkeypatch.setattr(campaign,'policy_factory',lambda *args:object())
    monkeypatch.setattr(campaign,'LifecycleProfile',lambda **kwargs:object())
    monkeypatch.setattr(campaign,'command_contract',lambda *args:dict(passed=True))
    calls = []
    def evaluate(**kwargs):
        calls.append(kwargs)
        if raises:
            raise RuntimeError('synthetic execution failure')
        path = kwargs['output']
        path.mkdir(parents=True)
        (path/'policy.jsonl').write_text(json.dumps(dict(time_s=600.,cohort_selection_verified=verified))+'\n')
        return dict(status='complete')
    monkeypatch.setattr(campaign.paired,'evaluate_arm',evaluate)
    return job,data,calls


def test_reused_job_cannot_be_simulated(tmp_path,monkeypatch):
    _,data,calls = prepared_run(tmp_path,monkeypatch)
    data['bindings']['new4108']['mode'] = 'verified_reuse'
    with pytest.raises(ValueError,match='reused'):
        campaign.run_one('registration','new4108')
    assert not calls and not (tmp_path/'allocations').exists()


def test_exact_one_exclusive_allocation_prevents_retry(tmp_path,monkeypatch):
    _,_,calls = prepared_run(tmp_path,monkeypatch)
    assert campaign.run_one('registration','new4108')['status'] == 'complete'
    allocation = json.loads((tmp_path/'allocations/new4108.json').read_text())
    assert allocation['allocated_simulated_seconds'] == 2400 and allocation['retries'] == 0
    with pytest.raises(FileExistsError):
        campaign.run_one('registration','new4108')
    assert len(calls) == 1


def test_runtime_selection_mismatch_preserves_charge_and_stops(tmp_path,monkeypatch):
    _,_,calls = prepared_run(tmp_path,monkeypatch,verified=False)
    with pytest.raises(ValueError,match='selection mismatch'):
        campaign.run_one('registration','new4108')
    assert len(calls) == 1
    assert (tmp_path/'STOP.json').exists()
    assert (tmp_path/'allocations/new4108.json').exists()
    with pytest.raises(ValueError,match='stopped'):
        campaign.run_one('registration','new4108')
    assert len(calls) == 1


def test_execution_failure_preserves_negative_and_never_retries(tmp_path,monkeypatch):
    _,_,calls = prepared_run(tmp_path,monkeypatch,raises=True)
    with pytest.raises(RuntimeError,match='synthetic'):
        campaign.run_one('registration','new4108')
    assert (tmp_path/'STOP.json').exists()
    assert (tmp_path/'allocations/new4108.json').exists()
    assert len(calls) == 1


def test_unexpected_allocation_is_not_ignored(tmp_path,monkeypatch):
    _,_,calls = prepared_run(tmp_path,monkeypatch)
    (tmp_path/'allocations').mkdir()
    (tmp_path/'allocations/other.json').write_text('{}')
    with pytest.raises(ValueError,match='Unexpected allocation'):
        campaign.run_one('registration','new4108')
    assert not calls


def test_reuse_proof_requires_actual_zero_based_decision_clock():
    selected = dict(selection_sha256='selection',selected=dict(member_ids=['a','b']))
    job = dict(id='base',policy_spec=dict(member_ids=['a','b']))
    proof = dict(source_manifest_sha256='manifest',selection_sha256='selection',source_job_id='base',
        decision_count=2400,first_decision_s=0.,last_decision_s=2399.,all_phases_and_members_equal=True,
        all_wall_commands_equal=True,all_actual_actuator_commands_and_native_release_flags_equal=True,
        no_simulator_rollout=True,complete=True)
    assert selection.validate_reuse_proof(proof,selection=selected,source_manifest_sha256='manifest',source_job=job)
    wrong = deepcopy(proof)
    wrong.update(first_decision_s=1.,last_decision_s=2400.)
    with pytest.raises(ValueError):
        selection.validate_reuse_proof(wrong,selection=selected,source_manifest_sha256='manifest',source_job=job)


def test_stopped_complete_manifest_cannot_mint_efficacy_analysis(tmp_path,monkeypatch):
    from types import ModuleType
    evidence = ModuleType('traffic_shepherd.cohort_evidence')
    evidence.compare_case = lambda *a:pytest.fail('Stopped study must not compare physical efficacy')
    monkeypatch.setitem(sys.modules,'traffic_shepherd.cohort_evidence',evidence)
    job = dict(id='new4108')
    physical = tmp_path/'jobs/new4108'
    physical.mkdir(parents=True)
    # A completed simulator manifest can coexist with failed selection/command
    # integrity discovered just after evaluate_arm returned. Completion alone
    # must not turn that retained negative artifact into an efficacy result.
    (physical/'manifest.json').write_text(json.dumps(dict(status='complete')))
    (tmp_path/'STOP.json').write_text(json.dumps(dict(reason='acquired selection mismatch')))
    data = dict(jobs=[job],bindings={'new4108':dict(mode='simulate',path=str(physical))})
    monkeypatch.setattr(campaign,'read_registration',lambda *a,**kw:(tmp_path,data))
    with pytest.raises(ValueError,match='Stopped Part C'):
        campaign.analyze('registration')
    assert not (tmp_path/'analysis.json').exists()
    assert (physical/'manifest.json').exists()
