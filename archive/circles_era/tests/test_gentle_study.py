import json

from traffic_shepherd.gentle_study import command_contract


def test_actual_issued_override_cannot_bypass_gentle_resolver(tmp_path):
    policy = dict(time_s=600.,wall_commands={'a':-.15},command_resolution={'a':dict(
        measured_entry_or_current_acceleration_mps2=0.,resolved_wall_acceleration_mps2=-.15)})
    (tmp_path/'policy.jsonl').write_text(json.dumps(policy)+'\n')
    command = dict(kind='command',timestamp_s=600.,vehicle_id='a',owner='wall',issued_acceleration_mps2=-.15)
    (tmp_path/'telemetry.jsonl').write_text(json.dumps(command)+'\n')
    assert command_contract(tmp_path)['passed']
    command['issued_acceleration_mps2'] = -1.
    (tmp_path/'telemetry.jsonl').write_text(json.dumps(command)+'\n')
    assert not command_contract(tmp_path)['passed']


def test_missing_command_and_excess_slew_are_implementation_failures(tmp_path):
    rows = [dict(time_s=t,wall_commands={'a':a},command_resolution={'a':dict(
        measured_entry_or_current_acceleration_mps2=a,resolved_wall_acceleration_mps2=a)})
        for t,a in ((600.,-.15),(601.,.5))]
    (tmp_path/'policy.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (tmp_path/'telemetry.jsonl').write_text('')
    result = command_contract(tmp_path)
    assert not result['passed']
    assert {r['kind'] for r in result['failures']} == {'resolver_contract','missing_issued_command'}


def test_failed_allocations_never_become_zero_effect_or_harmful_labels(tmp_path,monkeypatch):
    from traffic_shepherd import gentle_study
    path = tmp_path/'jobs'/'failed'
    path.mkdir(parents=True)
    (path/'manifest.json').write_text(json.dumps(dict(status='failed',error='logger defect',
        interaction_diagnostics={'final_time_s':600.})))
    data = dict(jobs=[dict(id='failed',duration_s=2400)],identity={'active_source_sha256':{'original':'hash'}})
    monkeypatch.setattr(gentle_study,'read_registration',lambda *a,**k:(tmp_path,data))
    result = gentle_study.analyze('unused')[0]
    assert result['effect'] is None and result['benefit'] is None
    assert not result['complete_horizon'] and not result['qualified_gate']
    assert not json.loads((tmp_path/'analysis.json').read_text())['complete_physical_comparison']
