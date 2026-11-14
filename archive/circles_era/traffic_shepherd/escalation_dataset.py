"""Verified offline development rows: CAV features separated from physical labels.

This exporter never fits a model or promotes exposed seeds to a prospective set.
Failed delivery and unchanged/refused opportunities remain in the inventory.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .escalation_evidence import BenefitCriteria, benefit_label
from .escalation_observations import report_from_payload
from .formation import FormationProfile, make_proposal, assess_formation
from .formation_evidence import score_formation_files
from .paired_experiments import read_registration, verify_pair, digest
from .proposal_features import feature_packet, select_proposal


def snapshot_history(snapshot):
    history = []
    for payload in snapshot['evidence']['history']:
        row = dict(payload)
        receipt = row.pop('received_at_s')
        history.append((receipt, report_from_payload(row)))
    return tuple(history)


def reconstruct_features(job, snapshot, *, history_s=20.):
    """Proposal and observations must reproduce the recorded predecision binding."""
    if snapshot['decision_time_s'] != job['decision_time_s']:
        raise ValueError('decision time mismatch')
    now = float(snapshot['decision_time_s'])
    reports = [report_from_payload(row) for row in snapshot['evidence']['reports']]
    spec = job['policy_spec']
    profile = FormationProfile(**spec['profile'])
    try:
        proposal = make_proposal(reports, now_s=now, required_lanes=tuple(spec['required_lanes']),
                                member_ids=tuple(spec['member_ids']), profile=profile)
    except ValueError:
        # Unavailable specific members are real no-proposal opportunities. They
        # are retained in the inventory, with no fabricated feature vector.
        return None, None, dict(status='registered_members_unavailable', feature_eligible=False)
    if proposal.fingerprint != snapshot['proposal_sha256']:
        raise ValueError('predecision proposal cannot be reconstructed exactly')
    history = snapshot_history(snapshot)
    feature = feature_packet(proposal, history, now_s=now, history_s=history_s)
    actual_assessment = assess_formation(proposal, reports, now_s=now)
    selected, selected_assessment, counts = select_proposal(history, now_s=now,
        required_lanes=tuple(spec['required_lanes']), profile=profile)
    counts.update(selection_status=counts['status'], status=actual_assessment.status,
        registered_assessment=asdict(actual_assessment),
        feature_eligible=True, registered_selection_reproduced=bool(
        selected and selected.fingerprint == proposal.fingerprint),
        local_admission=actual_assessment.approved,
        selected_alternative_local_admission=bool(selected_assessment and selected_assessment.approved))
    return proposal, feature, counts


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _job_metadata(job, registration):
    return dict(seed=job['seed'], loading=job['loading'], penetration=job['penetration'],
        decision_time_s=job['decision_time_s'], job_id=job['id'], case_id=job['case_id'],
        source_registration_sha256=registration['registration_sha256'], split='exposed_development_only')


def _inventory_state(root, job):
    path = root/'jobs'/job['id']
    allocated = (root/'allocations'/f"{job['id']}.json").is_file()
    result = dict(job_id=job['id'], allocated=allocated, summary_present=(path/'summary.json').is_file())
    if not path.exists():
        return dict(result, status='allocated_not_started' if allocated else 'unallocated')
    try:
        manifest = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        return dict(result, status='invalid_or_partial_manifest', error=str(exc))
    status = manifest.get('status')
    if status not in {'preparing', 'running', 'failed', 'complete'}:
        return dict(result, status='invalid_manifest_status', recorded_status=status)
    if status == 'complete' and not result['summary_present']:
        return dict(result, status='complete_manifest_missing_summary')
    return dict(result, status=status, error=manifest.get('error'))


def _registered_manifest(path, job, registration):
    """A mutually valid pair must also belong to this exact exported allocation."""
    manifest = json.loads((Path(path)/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('registration_sha256') != registration['registration_sha256']:
        raise ValueError('arm manifest belongs to a different registration')
    if any(manifest.get(key) != value for key,value in registration['identity'].items()):
        raise ValueError('arm identity differs from its frozen registration')
    fields = dict(job_id='id', case_id='case_id', seed='seed', mainline_vph='loading', penetration='penetration',
        role='role', controller='arm', duration_s='duration_s', decision_time_s='decision_time_s', policy_spec='policy_spec')
    if any(manifest.get(key) != job[field] for key,field in fields.items()):
        raise ValueError('arm metadata differs from its registered job')
    if manifest.get('reference_job_id') != job.get('reference_job_id') or manifest.get('snapshot_times_s') != sorted(
            job.get('snapshot_times_s', [job['decision_time_s']])):
        raise ValueError('arm reference or decision snapshots differ from the job')
    return manifest


def _attempt_state(telemetry_path, policy_path, *, decision_time_s, delivery):
    """Refusal policy value is distinct from an observed attempted-wall effect."""
    commanded = set()
    wall_decisions, requested_wall_decisions = 0, 0
    with Path(telemetry_path).open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('kind') == 'command' and row['timestamp_s'] >= decision_time_s:
                requested_wall_decisions += int(row.get('wall_request_mps2') is not None)
                if row.get('owner') == 'wall' and row.get('issued_acceleration_mps2') is not None:
                    wall_decisions += 1
                    commanded.add(row['vehicle_id'])
    phases = set()
    with Path(policy_path).open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['time_s'] >= decision_time_s:
                phases.add(row['phase'])
    return dict(experimental_assignment=True,
        controller_attempted=bool(phases & {'assembly', 'hold', 'taper', 'abort'}),
        actuation_attempted=wall_decisions > 0, commanded_wall_decisions=wall_decisions,
        requested_wall_decisions=requested_wall_decisions,
        commanded_member_ids=sorted(commanded), refused_without_attempt='refused' in phases and wall_decisions == 0,
        wall_effect_observed=wall_decisions > 0,
        formation_delivered=bool(delivery and delivery['delivered']))


def _predecision_event_eligibility(control, treatment, *, manifests, decision_time_s, effect):
    """Separate an already-failed absolute target from future action evidence.

    These verified full-horizon intervals remain offline labels/provenance. A
    safety event observed by the decision cannot be undone by any later action,
    so the registered whole-episode zero-event benefit target is already lost.
    This does not relabel the physical outcome or infer collision causation.
    """
    keys = ('collision_involvements', 'starting_teleports')
    if type(decision_time_s) is not int or decision_time_s < 0:
        raise ValueError('event partition needs an integer control decision time')
    arms = {}
    for name, path, manifest in zip(('control', 'treatment'), (control, treatment), manifests):
        raw = (Path(path)/'intervals.jsonl').read_bytes()
        source_hash = hashlib.sha256(raw).hexdigest()
        if source_hash != manifest['intervals_sha256']:
            raise ValueError('event interval bytes differ from the verified arm manifest')
        before, after = dict.fromkeys(keys, 0), dict.fromkeys(keys, 0)
        prefix, events, previous, count = hashlib.sha256(), [], 0., 0
        for line in raw.splitlines(keepends=True):
            row = json.loads(line)
            start, end = row['start_s'], row['end_s']
            if start != previous or end != start+1.:
                raise ValueError('missing or misaligned event accounting interval')
            previous = end
            if start < decision_time_s < end:
                raise ValueError('event accounting interval straddles the decision')
            if any(type(row.get(key)) is not int or row[key] < 0 for key in keys):
                raise ValueError('explicit nonnegative integer safety event counts required')
            is_before = end <= decision_time_s
            totals = before if is_before else after
            for key in keys:
                totals[key] += row[key]
            if is_before:
                prefix.update(line)
                count += 1
            if any(row[key] for key in keys):
                events.append(dict(start_s=start, end_s=end,
                    period='before_action' if is_before else 'after_action',
                    **{key: row[key] for key in keys}))
        if previous != manifest['duration_s'] or not decision_time_s < previous:
            raise ValueError('event partition requires the complete registered horizon')
        if any(before[key]+after[key] != effect.get(name+'_'+key) for key in keys):
            raise ValueError('interval safety totals disagree with the paired physical effect')
        arms[name] = dict(intervals_sha256=source_hash,
            predecision_intervals_sha256=prefix.hexdigest(), predecision_interval_count=count,
            before_action=before, after_action=after, event_intervals=events)
    if (arms['control']['predecision_intervals_sha256'] != arms['treatment']['predecision_intervals_sha256']
            or arms['control']['before_action'] != arms['treatment']['before_action']):
        raise ValueError('matched arms have different measured predecision outcome intervals')
    prior_event = any(arms['treatment']['before_action'].values())
    return dict(schema='paired_predecision_event_eligibility/1', decision_time_s=decision_time_s,
        boundary_rule='interval_end_at_or_before_decision_is_before_action',
        registered_absolute_safety_eligible_at_decision=not prior_event,
        reason='prior_safety_event_already_fails_whole_horizon_zero_event_target' if prior_event else None,
        matched_predecision_intervals=True, **arms,
        interpretation='Offline attribution and training eligibility only; registered physical label unchanged.')


def export_dataset(registration, output):
    root, data = read_registration(registration, require_current=False)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows, unavailable = [], []
    for job in data['jobs']:
        if job['arm'] != 'wall':
            continue
        treatment = root/'jobs'/job['id']
        references = [j for j in data['jobs'] if j['arm'] == 'damping' and
            (j['id'] == job['reference_job_id'] if 'reference_job_id' in job else j['case_id'] == job['case_id'])]
        if len(references) != 1:
            raise ValueError('each development row needs one registered reference')
        control = root/'jobs'/references[0]['id']
        treatment_state, control_state = _inventory_state(root, job), _inventory_state(root, references[0])
        if treatment_state['status'] != 'complete' or control_state['status'] != 'complete':
            unavailable.append(dict(job_id=job['id'], reason='pair_not_complete',
                treatment_state=treatment_state, reference_state=control_state,
                offline_metadata=_job_metadata(job, data), retained=True))
            continue
        treatment_manifest = _registered_manifest(treatment, job, data)
        control_manifest = _registered_manifest(control, references[0], data)
        pair = verify_pair(control,treatment)
        if pair['mechanics_only'] or pair['effect'] is None:
            raise ValueError('short mechanics cannot enter physical benefit data')
        prior_events = _predecision_event_eligibility(control, treatment,
            manifests=(control_manifest, treatment_manifest),
            decision_time_s=job['decision_time_s'], effect=pair['effect'])
        snapshots = json.loads((treatment/'decision_snapshots.json').read_text(encoding='utf-8'))
        snapshot = snapshots[str(job['decision_time_s'])]
        proposal, feature, counts = reconstruct_features(job,snapshot, history_s=data['communication']['history_s'])
        delivery = score_formation_files(treatment/'formation.jsonl',treatment/'telemetry.jsonl',
            proposal=asdict(proposal)) if proposal else None
        physical = benefit_label(pair['effect'],BenefitCriteria(**data['criteria']))
        attempt = _attempt_state(treatment/'telemetry.jsonl', treatment/'policy.jsonl',
            decision_time_s=job['decision_time_s'], delivery=delivery)
        complete_service = (pair['effect'].get('complete_requested_service_control') is True and
                            pair['effect'].get('complete_requested_service_treatment') is True)
        training_exclusions = []
        if not feature:
            training_exclusions.append('acquired_feature_vector_unavailable')
        if not attempt['wall_effect_observed']:
            training_exclusions.append('wall_actuation_not_observed')
        if not complete_service:
            training_exclusions.append('incomplete_requested_service')
        if physical['label'] not in {'beneficial', 'nonbeneficial', 'harmful'}:
            training_exclusions.append('unresolved_physical_label')
        if not prior_events['registered_absolute_safety_eligible_at_decision']:
            training_exclusions.append('registered_zero_event_target_failed_before_action')
        wall_target = dict(observed=attempt['wall_effect_observed'],
            label=physical['label'] if attempt['wall_effect_observed'] else None,
            benefit_training_eligible=not training_exclusions,
            training_ineligibility_reasons=training_exclusions,
            registered_absolute_safety_eligible_at_decision=prior_events['registered_absolute_safety_eligible_at_decision'],
            scope='actual_attempt_including_failed_assembly' if attempt['wall_effect_observed'] else
                'unobserved_wall_effect_refusal_or_no_actuation',
            complete_requested_service_in_both_arms=complete_service)
        row = dict(schema='paired_cav_development_row/1',features=feature['values'] if feature else None,
            feature_binding=({k:v for k,v in feature.items() if k not in {'values','proposal'}} if feature else None),
            proposal=asdict(proposal) if proposal else None,
            opportunity_accounting=dict(counts, **attempt),
            offline_metadata=_job_metadata(job, data),
            offline_labels=dict(physical=physical, physical_label_scope='intent_to_treat_registered_assignment',
                                wall_benefit_target=wall_target, predecision_event_eligibility=prior_events,
                                formation=delivery,effect=pair['effect']),
            evidence=dict(pair=pair,predecision_snapshot_sha256=digest(snapshot)),
            retained_regardless_of_formation_or_effect=True)
        row['row_sha256']=digest(row)
        rows.append(row)
    schemas = {tuple(sorted(row['features'])) for row in rows if row['features'] is not None}
    if len(schemas) > 1:
        raise ValueError('nonnull feature vectors disagree on their acquired schema')
    with (output/'rows.jsonl').open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
    seeds=sorted({r['offline_metadata']['seed'] for r in rows})
    manifest=dict(schema='cav_development_dataset/1',source_registration_sha256=data['registration_sha256'],
        exporter_source_sha256=_sha(__file__),
        row_count=len(rows), rows_sha256=_sha(output/'rows.jsonl'), unavailable_pairs=unavailable,
        registered_wall_opportunities=sum(j['arm']=='wall' for j in data['jobs']),
        feature_eligible_rows=sum(row['features'] is not None for row in rows),
        observed_wall_effect_rows=sum(row['opportunity_accounting']['wall_effect_observed'] for row in rows),
        benefit_training_eligible_rows=sum(row['offline_labels']['wall_benefit_target']['benefit_training_eligible'] for row in rows),
        predecision_safety_ineligible_rows=sum(not row['offline_labels']['predecision_event_eligibility'][
            'registered_absolute_safety_eligible_at_decision'] for row in rows),
        benefit_training_eligibility_policy='observed_complete_service_prior_safety_clean/2',
        split_audit=dict(exposed_development_seeds=seeds,calibration_seeds=[],prospective_seeds=[],
            entire_episode_grouping=True,existing_sealed_blocks_used=False,
            warning='No independent calibration or prospective claim is possible from these exposed development rows.'),
        feature_names=list(next(iter(schemas))) if schemas else [],
        metadata_and_labels_are_not_online_inputs=True,predictor_scale=False,
        interpretation='Measured development evidence only; no fitted or calibrated gate is produced.')
    manifest['manifest_sha256'] = digest(manifest)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n', encoding='utf-8')
    return manifest


def verify_dataset(output):
    output = Path(output)
    manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    if manifest['manifest_sha256'] != digest({k:v for k,v in manifest.items() if k != 'manifest_sha256'}):
        raise ValueError('dataset manifest changed')
    if _sha(output/'rows.jsonl') != manifest['rows_sha256']:
        raise ValueError('training row bytes differ from the frozen dataset manifest')
    rows = [json.loads(line) for line in (output/'rows.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(rows) != manifest['row_count']:
        raise ValueError('row count differs from dataset manifest')
    if len(rows)+len(manifest['unavailable_pairs']) != manifest['registered_wall_opportunities']:
        raise ValueError('registered opportunity inventory is incomplete')
    for row in rows:
        if row['row_sha256'] != digest({k:v for k,v in row.items() if k != 'row_sha256'}):
            raise ValueError('dataset row binding changed')
        if row['features'] is not None and sorted(row['features']) != manifest['feature_names']:
            raise ValueError('feature schema differs from manifest')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(json.dumps(export_dataset(args.registration,args.output),indent=2))


if __name__ == '__main__':
    main()
