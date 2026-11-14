"""Exporter provenance/label regressions; all fixtures are synthetic, no SUMO."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from traffic_shepherd.formation import FormationProfile, make_proposal
from traffic_shepherd.reporting import CAVReport
from traffic_shepherd.sensing import NeighborMeasurement, SensorFrame


HERE = Path(__file__).parent
if (HERE/'escalation_dataset.py').exists():
    spec = importlib.util.spec_from_file_location('traffic_shepherd._dataset_prior_events', HERE/'escalation_dataset.py')
    dataset = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = dataset
    spec.loader.exec_module(dataset)
else:
    from traffic_shepherd import escalation_dataset as dataset


def fixture(*, unsafe_registered=False, alternative=False):
    reports = []
    groups = [('registered', 1800., 2. if unsafe_registered else 50.)]
    if alternative:
        groups.append(('alternative', 1880., 50.))
    for group, base, gap in groups:
        for lane in range(2):
            reports.append(CAVReport(SensorFrame(f'{group}{lane}', 10., 20., 0., 30., lane,
                distance_to_merge_m=base-lane, front=NeighborMeasurement(gap, 0.),
                front_coverage_m=100., rear_coverage_m=100., lane_id=f'L{lane}'), 10., 'damping'))
    profile = FormationProfile(hold_s=20.)
    proposal = make_proposal(reports, now_s=10., required_lanes=('L0', 'L1'),
                            member_ids=('registered0', 'registered1'), profile=profile)
    job = dict(id='wall', case_id='case', seed=4107, loading=2000, penetration=.2, role='exposed',
        arm='wall', duration_s=2400, decision_time_s=10, reference_job_id='damping',
        policy_spec=dict(required_lanes=['L0', 'L1'], member_ids=list(proposal.member_ids), profile=asdict(profile)))
    snapshot = dict(decision_time_s=10., proposal_sha256=proposal.fingerprint, evidence=dict(
        reports=[r.payload() for r in reports], history=[dict(received_at_s=10., **r.payload()) for r in reports]))
    return job, snapshot


def effect(*, harmful=False, complete=True):
    return dict(control='damping', treatment='wall', improvements=dict(
        braking_seconds_per_request=-.2 if harmful else 0., effort_per_request=0.,
        system_seconds_per_request=0., ttc_seconds_per_request=0., source_wait_seconds_per_request=0.),
        excess_collisions=0, excess_teleports=0, excess_unfinished=0,
        control_unfinished_requested=0 if complete else 1, treatment_unfinished_requested=0 if complete else 1,
        control_collision_involvements=0, treatment_collision_involvements=0,
        control_starting_teleports=0, treatment_starting_teleports=0,
        complete_requested_service_control=complete, complete_requested_service_treatment=complete)


def save_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def arm_files(root, job, data, snapshot, *, phase='refused', owner='damping'):
    path = root/'jobs'/job['id']
    path.mkdir(parents=True)
    manifest = dict(status='complete', registration_sha256=data['registration_sha256'], **data['identity'],
        job_id=job['id'], case_id=job['case_id'], seed=job['seed'], mainline_vph=job['loading'],
        penetration=job['penetration'], role=job['role'], controller=job['arm'], duration_s=job['duration_s'],
        decision_time_s=job['decision_time_s'], policy_spec=job['policy_spec'],
        reference_job_id=job.get('reference_job_id'), snapshot_times_s=job.get('snapshot_times_s', [10]))
    (path/'intervals.jsonl').write_text(''.join(json.dumps(dict(start_s=float(t), end_s=float(t+1),
        collision_involvements=0, starting_teleports=0))+'\n' for t in range(job['duration_s'])), encoding='utf-8')
    manifest['intervals_sha256'] = dataset._sha(path/'intervals.jsonl')
    save_json(path/'manifest.json', manifest)
    save_json(path/'summary.json', {})
    save_json(path/'decision_snapshots.json', {'10': snapshot})
    (path/'telemetry.jsonl').write_text(json.dumps(dict(kind='command', timestamp_s=10., vehicle_id='registered0',
        owner=owner, wall_request_mps2=-.5 if owner=='wall' else None,
        issued_acceleration_mps2=-.5))+'\n', encoding='utf-8')
    (path/'policy.jsonl').write_text(json.dumps(dict(time_s=10., phase=phase))+'\n', encoding='utf-8')
    (path/'formation.jsonl').write_text('', encoding='utf-8')
    return path


def setup_export(tmp_path, monkeypatch, *, phase='refused', owner='damping', harmful=False, complete=True):
    root = tmp_path/'runs'
    wall, snapshot = fixture()
    damping = {**wall, 'id': 'damping', 'arm': 'damping'}
    damping.pop('reference_job_id')
    data = dict(registration_sha256='registered', identity=dict(config_sha256='config', active_source_sha256={'x': 'y'}),
        jobs=[damping, wall], communication=dict(history_s=20.), criteria=dict(
            primary_minimum_improvement={'braking_seconds_per_request': .02},
            maximum_cost_increase={key: .1 for key in effect()['improvements']}))
    control = arm_files(root, damping, data, snapshot)
    treatment = arm_files(root, wall, data, snapshot, phase=phase, owner=owner)
    monkeypatch.setattr(dataset, 'read_registration', lambda *a, **k: (root, data))
    monkeypatch.setattr(dataset, 'verify_pair', lambda *a: dict(mechanics_only=False,
        effect=effect(harmful=harmful, complete=complete), censored_service=not complete))
    monkeypatch.setattr(dataset, 'score_formation_files', lambda *a, **k: dict(delivered=False, terminal_phase=phase))
    return root, data, wall, snapshot, control, treatment


def test_admission_label_belongs_to_registered_cohort_not_safe_alternative():
    job, snapshot = fixture(unsafe_registered=True, alternative=True)
    _, _, counts = dataset.reconstruct_features(job, snapshot)
    assert counts['status'] == 'unsafe_geometry' and not counts['local_admission']
    assert counts['selection_status'] == 'feasible' and counts['selected_alternative_local_admission']
    assert not counts['registered_selection_reproduced']


@pytest.mark.parametrize('key,value', [('registration_sha256', 'another-registration'), ('seed', 4108),
    ('config_sha256', 'different-model-wrapper'), ('job_id', 'other-job'), ('policy_spec', {})])
def test_mutually_valid_pair_cannot_be_relabelled_under_another_job(tmp_path, monkeypatch, key, value):
    _, _, _, _, _, treatment = setup_export(tmp_path, monkeypatch)
    manifest = json.loads((treatment/'manifest.json').read_text())
    manifest[key] = value
    save_json(treatment/'manifest.json', manifest)
    called = []
    monkeypatch.setattr(dataset, 'verify_pair', lambda *a: called.append(True))
    with pytest.raises(ValueError):
        dataset.export_dataset('synthetic-registration', tmp_path/'data')
    assert not called


def test_refusal_retains_itt_outcome_without_fabricating_wall_negative(tmp_path, monkeypatch):
    setup_export(tmp_path, monkeypatch)
    target = tmp_path/'data'
    manifest = dataset.export_dataset('fixture', target)
    row = json.loads((target/'rows.jsonl').read_text())
    assert row['offline_labels']['physical']['label'] == 'nonbeneficial'
    assert row['offline_labels']['physical_label_scope'] == 'intent_to_treat_registered_assignment'
    assert row['opportunity_accounting']['refused_without_attempt']
    assert not row['offline_labels']['wall_benefit_target']['observed']
    assert row['offline_labels']['wall_benefit_target']['label'] is None
    assert not row['offline_labels']['wall_benefit_target']['benefit_training_eligible']
    assert manifest['row_count'] == 1 and manifest['observed_wall_effect_rows'] == 0


def test_failed_but_actuated_assembly_remains_a_complete_service_harmful_example(tmp_path, monkeypatch):
    setup_export(tmp_path, monkeypatch, phase='abort', owner='wall', harmful=True)
    output = tmp_path/'data'
    dataset.export_dataset('fixture', output)
    row = json.loads((output/'rows.jsonl').read_text())
    assert not row['opportunity_accounting']['formation_delivered']
    assert row['opportunity_accounting']['actuation_attempted']
    assert row['offline_labels']['wall_benefit_target']['label'] == 'harmful'
    assert row['offline_labels']['wall_benefit_target']['benefit_training_eligible']


def test_unfinished_service_keeps_observed_harm_but_not_uncensored_training_target(tmp_path, monkeypatch):
    setup_export(tmp_path, monkeypatch, phase='abort', owner='wall', harmful=True, complete=False)
    output = tmp_path/'data'
    dataset.export_dataset('fixture', output)
    row = json.loads((output/'rows.jsonl').read_text())
    assert row['offline_labels']['physical']['label'] == 'harmful'
    assert row['offline_labels']['wall_benefit_target']['observed']
    assert not row['offline_labels']['wall_benefit_target']['benefit_training_eligible']


def test_inventory_distinguishes_failed_allocated_and_unallocated_arms(tmp_path, monkeypatch):
    root, _, wall, _, _, treatment = setup_export(tmp_path, monkeypatch)
    (root/'allocations').mkdir()
    save_json(root/'allocations'/'wall.json', {'allocated': 2400})
    manifest = json.loads((treatment/'manifest.json').read_text())
    manifest.update(status='failed', error='synthetic process failure')
    save_json(treatment/'manifest.json', manifest)
    state = dataset._inventory_state(root, wall)
    assert state['status'] == 'failed' and state['error'] == 'synthetic process failure' and state['allocated']
    absent = dict(wall, id='not-launched')
    assert dataset._inventory_state(root, absent)['status'] == 'unallocated'
    save_json(root/'allocations'/'not-launched.json', {})
    assert dataset._inventory_state(root, absent)['status'] == 'allocated_not_started'
    exported = dataset.export_dataset('fixture', tmp_path/'data')
    inventory = exported['unavailable_pairs'][0]
    assert inventory['treatment_state']['status'] == 'failed'
    assert inventory['reference_state']['status'] == 'complete'
    assert inventory['offline_metadata']['seed'] == 4107 and exported['row_count'] == 0


def test_missing_first_feature_row_does_not_hide_later_schema(tmp_path, monkeypatch):
    root, data, wall, snapshot, _, treatment = setup_export(tmp_path, monkeypatch)
    absent = json.loads(json.dumps(snapshot))
    absent['evidence']['reports'] = []
    absent['evidence']['history'] = []
    save_json(treatment/'decision_snapshots.json', {'10': absent})
    second = dict(wall, id='wall2', case_id='case2')
    data['jobs'].append(second)
    arm_files(root, second, data, snapshot)
    manifest = dataset.export_dataset('fixture', tmp_path/'data')
    assert manifest['row_count'] == 2 and manifest['feature_eligible_rows'] == 1
    assert 'target_speed_mps' in manifest['feature_names']


def test_manifest_binds_exact_rows_and_rejects_changed_row_bytes(tmp_path, monkeypatch):
    setup_export(tmp_path, monkeypatch)
    output = tmp_path/'data'
    original = dataset.export_dataset('fixture', output)
    assert dataset.verify_dataset(output) == original
    rows = output/'rows.jsonl'
    row = json.loads(rows.read_text())
    row['offline_metadata']['seed'] = 1234
    row['row_sha256'] = dataset.digest({k:v for k,v in row.items() if k != 'row_sha256'})
    rows.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError, match='row bytes'):
        dataset.verify_dataset(output)


def replace_events(path, events, *, update_hash=True):
    rows = [json.loads(line) for line in (path/'intervals.jsonl').read_text().splitlines()]
    for start, key, value in events:
        rows[start][key] = value
    (path/'intervals.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
    if update_hash:
        manifest = json.loads((path/'manifest.json').read_text())
        manifest['intervals_sha256'] = dataset._sha(path/'intervals.jsonl')
        save_json(path/'manifest.json', manifest)


def collision_effect(*, before=0, control_after=0, treatment_after=0, improvements=False):
    result = effect()
    result.update(control_collision_involvements=before+control_after,
        treatment_collision_involvements=before+treatment_after,
        excess_collisions=treatment_after-control_after)
    if improvements:
        result['improvements'] = {key: 1. for key in result['improvements']}
    return result


def set_pair(monkeypatch, value):
    monkeypatch.setattr(dataset, 'verify_pair', lambda *a: dict(mechanics_only=False,
        effect=value, censored_service=False))


def test_prior_collision_does_not_become_a_whole_horizon_benefit_training_negative(tmp_path, monkeypatch):
    _, _, _, _, control, treatment = setup_export(tmp_path, monkeypatch, phase='hold', owner='wall')
    replace_events(control, [(7, 'collision_involvements', 2), (20, 'collision_involvements', 2)])
    replace_events(treatment, [(7, 'collision_involvements', 2)])
    original_effect = collision_effect(before=2, control_after=2, improvements=True)
    set_pair(monkeypatch, original_effect)
    output = tmp_path/'data'
    manifest = dataset.export_dataset('fixture', output)
    row = json.loads((output/'rows.jsonl').read_text())
    assert row['offline_labels']['physical']['label'] == 'harmful'
    assert row['offline_labels']['effect'] == original_effect
    target = row['offline_labels']['wall_benefit_target']
    assert target['observed'] and target['label'] == 'harmful'
    assert not target['benefit_training_eligible']
    assert target['training_ineligibility_reasons'] == ['registered_zero_event_target_failed_before_action']
    provenance = row['offline_labels']['predecision_event_eligibility']
    assert provenance['control']['before_action']['collision_involvements'] == 2
    assert provenance['treatment']['before_action']['collision_involvements'] == 2
    assert provenance['control']['after_action']['collision_involvements'] == 2
    assert provenance['treatment']['after_action']['collision_involvements'] == 0
    assert provenance['matched_predecision_intervals']
    assert manifest['row_count'] == 1 and manifest['predecision_safety_ineligible_rows'] == 1
    assert manifest['benefit_training_eligible_rows'] == 0
    assert dataset.verify_dataset(output) == manifest


@pytest.mark.parametrize('event_start,prior', [(9, True), (10, False)])
def test_event_boundary_is_the_action_time_not_interval_start_label(tmp_path, monkeypatch, event_start, prior):
    _, _, _, _, control, treatment = setup_export(tmp_path, monkeypatch, phase='hold', owner='wall')
    replace_events(treatment, [(event_start, 'collision_involvements', 2)])
    if prior:
        replace_events(control, [(event_start, 'collision_involvements', 2)])
    set_pair(monkeypatch, collision_effect(before=2 if prior else 0, treatment_after=0 if prior else 2))
    output = tmp_path/'data'
    dataset.export_dataset('fixture', output)
    row = json.loads((output/'rows.jsonl').read_text())
    target = row['offline_labels']['wall_benefit_target']
    assert row['offline_labels']['physical']['label'] == 'harmful'
    assert target['benefit_training_eligible'] is not prior
    provenance = row['offline_labels']['predecision_event_eligibility']
    assert provenance['treatment']['event_intervals'][0]['period'] == ('before_action' if prior else 'after_action')


def test_prior_teleport_also_precludes_the_registered_zero_event_target(tmp_path, monkeypatch):
    _, _, _, _, control, treatment = setup_export(tmp_path, monkeypatch, phase='hold', owner='wall')
    for path in (control, treatment):
        replace_events(path, [(4, 'starting_teleports', 1)])
    value = effect()
    value.update(control_starting_teleports=1, treatment_starting_teleports=1)
    set_pair(monkeypatch, value)
    output = tmp_path/'data'
    dataset.export_dataset('fixture', output)
    row = json.loads((output/'rows.jsonl').read_text())
    assert row['offline_labels']['physical']['label'] == 'harmful'
    assert not row['offline_labels']['wall_benefit_target']['benefit_training_eligible']


def test_prior_exclusion_preserves_additional_postdecision_safety_harm(tmp_path, monkeypatch):
    _, _, _, _, control, treatment = setup_export(tmp_path, monkeypatch, phase='hold', owner='wall')
    replace_events(control, [(7, 'collision_involvements', 2)])
    replace_events(treatment, [(7, 'collision_involvements', 2), (12, 'collision_involvements', 2)])
    set_pair(monkeypatch, collision_effect(before=2, treatment_after=2))
    output = tmp_path/'data'
    dataset.export_dataset('fixture', output)
    row = json.loads((output/'rows.jsonl').read_text())
    assert row['offline_labels']['physical']['label'] == 'harmful'
    assert row['offline_labels']['effect']['excess_collisions'] == 2
    assert row['offline_labels']['predecision_event_eligibility']['treatment']['after_action']['collision_involvements'] == 2
    assert not row['offline_labels']['wall_benefit_target']['benefit_training_eligible']


def test_changed_interval_bytes_cannot_supply_predecision_event_evidence(tmp_path, monkeypatch):
    _, _, _, _, _, treatment = setup_export(tmp_path, monkeypatch, phase='hold', owner='wall')
    replace_events(treatment, [(7, 'collision_involvements', 2)], update_hash=False)
    with pytest.raises(ValueError, match='interval bytes'):
        dataset.export_dataset('fixture', tmp_path/'data')


def test_matching_prior_counts_at_different_times_do_not_pass_prefix_identity(tmp_path, monkeypatch):
    _, _, _, _, control, treatment = setup_export(tmp_path, monkeypatch, phase='hold', owner='wall')
    replace_events(control, [(7, 'collision_involvements', 2)])
    replace_events(treatment, [(8, 'collision_involvements', 2)])
    set_pair(monkeypatch, collision_effect(before=2))
    with pytest.raises(ValueError, match='different measured predecision'):
        dataset.export_dataset('fixture', tmp_path/'data')


def test_event_interval_totals_must_match_the_original_physical_effect(tmp_path, monkeypatch):
    _, _, _, _, _, treatment = setup_export(tmp_path, monkeypatch, phase='hold', owner='wall')
    replace_events(treatment, [(12, 'collision_involvements', 2)])
    with pytest.raises(ValueError, match='safety totals disagree'):
        dataset.export_dataset('fixture', tmp_path/'data')
