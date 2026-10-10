"""Exactly two gentle wall treatment allocations; immutable verified controls."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

from .config import ROOT
from . import paired_experiments as paired
from .dissolution_study import reference_proof, REFERENCES, ORIGINAL
from .gentle_wall import VERSION, GentleProfile, policy_factory
from .formation import make_proposal
from .formation_evidence import score_formation_files
from .escalation_observations import report_from_payload
from .escalation_evidence import BenefitCriteria, benefit_label, paired_effect
from .workflow import sha, write_json


def logging_repair_mechanics_proof(mechanics_registration, failed_registration):
    """Explicit replacement registration only; never launches or retries a job.

    Reuse mechanics only for the exact metadata-only repair preserved in the
    stopped-block audit. Different physical policy code requires new mechanics.
    """
    from .gentle_mechanics import read_registration as read_mechanics, _verify_case, CASE_IDS
    mr, md = read_mechanics(mechanics_registration, require_current=False)
    fr, fd = read_registration(failed_registration, require_current=False)
    audit_path = ROOT/'output/gentle_wall_v1/stopped_block_audit.json'
    audit = json.loads(audit_path.read_text())
    current = paired.current_identity()['active_source_sha256']
    changed = {name for name,value in md['identity']['active_source_sha256'].items() if current.get(name) != value}
    if not changed <= {'gentle_wall.py','gentle_study.py','escalation_study.py'}:
        raise ValueError('logging repair cannot bridge changed mechanics or baseline code')
    if sha(ROOT/'traffic_shepherd/gentle_wall.py') != audit['repaired_source_sha256']:
        raise ValueError('policy differs from the audited metadata-only repair')
    original = (mr/'source_snapshot/gentle_wall.py').read_text()
    restored = (ROOT/'traffic_shepherd/gentle_wall.py').read_text().replace(
        'IMPLEMENTATION_REVISION = "gentle_report_wall/1.0.1"\n','').replace(
        '            implementation_revision=IMPLEMENTATION_REVISION,\n'
        '            experimental_authorization="registered_experimental_assignment_not_trained_benefit_gate")',
        '            assignment="registered_experimental_assignment_not_trained_benefit_gate")')
    if restored != original or {j['job_id'] for j in audit['jobs']} != {j['id'] for j in fd['jobs']}:
        raise ValueError('repair or failed-case scope differs from the preserved execution')
    for row in audit['jobs']:
        if row['wall_commands_issued'] != 0 or not all(row['predecision_snapshot_checks'].values()):
            raise ValueError('this replacement bridge is only for the pre-dispatch logger failure')
        for name,checksum in row['preservation_sha256'].items():
            if sha(fr/'jobs'/row['job_id']/name) != checksum:
                raise ValueError('failed allocation evidence changed')
    return dict(passed=True,cases=[_verify_case(mr,md,case) for case in CASE_IDS],
        original_mechanics_registration_sha256=md['registration_sha256'],
        replaces_failed_registration_sha256=fd['registration_sha256'],
        metadata_only_source_bridge=True,audit_sha256=sha(audit_path),
        new_mechanics_simulations=0,requires_explicit_replacement_registration=True)


def register(output, mechanics_registration, *, replaces_failed_registration=None):
    from .gentle_mechanics import verify_completed
    mechanics = (verify_completed(mechanics_registration) if replaces_failed_registration is None
                 else logging_repair_mechanics_proof(mechanics_registration,replaces_failed_registration))
    identity = paired.current_identity()
    original = json.loads(ORIGINAL.read_text())
    jobs, references = [], {}
    for seed in (4107, 4108):
        job = deepcopy(next(j for j in original['jobs'] if j['id'] == f's{seed}_m2700_p20_t600_wall'))
        job['id'] = f's{seed}_m2700_p20_t600_gentle_wall'
        job['policy_spec']['profile'].update(version=VERSION, acceleration_min_mps2=-.3,
                                             acceleration_change_limit_mps3=.15)
        GentleProfile(**job['policy_spec']['profile'])
        paired.validate_job(job)
        _, _, proof = reference_proof(REFERENCES/'jobs'/job['reference_job_id'], identity)
        references[job['id']] = proof
        jobs.append(job)
    data = dict(schema='gentle_wall_study/1', registered_at_utc=datetime.now(timezone.utc).isoformat(),
        identity=identity, jobs=jobs, references=references, communication=original['communication'],
        assignment='registered_experimental_assignment_not_trained_benefit_gate', interaction_diagnostics=True,
        criteria=original['criteria'], policy_version=VERSION, maximum_attempts=2,
        maximum_simulated_seconds=4800, automatic_retries=0, automatic_successors=False,
        mechanics_evidence=mechanics, protocol_path='docs/GENTLE_WALL_CHECK.md',
        protocol_sha256=sha(ROOT/'docs/GENTLE_WALL_CHECK.md'))
    data['registration_sha256'] = paired.digest(data)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'registration.json', data)
    return dict(registration=str(output/'registration.json'), jobs=[j['id'] for j in jobs],
                maximum_simulated_seconds=4800, reused_damping_arms=2, simulations_started=0)


def read_registration(path, *, require_current=True):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if data.get('schema') != 'gentle_wall_study/1' or data.get('registration_sha256') != paired.digest(
            {k:v for k,v in data.items() if k != 'registration_sha256'}):
        raise ValueError('gentle registration changed')
    if (data['maximum_attempts'] != 2 or len(data['jobs']) != 2 or data['maximum_simulated_seconds'] != 4800
            or data['automatic_retries'] != 0 or data['automatic_successors']):
        raise ValueError('gentle budget changed')
    if require_current and data['identity'] != paired.current_identity():
        raise ValueError('source changed after gentle registration')
    if require_current and sha(ROOT/data['protocol_path']) != data['protocol_sha256']:
        raise ValueError('protocol changed after gentle registration')
    for job in data['jobs']:
        paired.validate_job(job)
        GentleProfile(**job['policy_spec']['profile'])
        if job['duration_s'] != 2400 or job['seed'] not in (4107,4108) or job['decision_time_s'] != 600:
            raise ValueError('gentle case changed')
    return path.parent, data


def command_contract(path):
    """Check actual owner-issued commands against every final resolver record."""
    path = Path(path)
    requested, previous, failures, count = {}, {}, [], 0
    with (path/'policy.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            for vid, acceleration in row['wall_commands'].items():
                proof = row['command_resolution'][vid]
                key = (row['time_s'],vid)
                requested[key] = acceleration
                if (not -.3-1e-10 <= acceleration <= 1.5+1e-10
                        or abs(acceleration-proof['measured_entry_or_current_acceleration_mps2']) > .15+1e-8
                        or (vid in previous and abs(acceleration-previous[vid]) > .15+1e-8)
                        or abs(acceleration-proof['resolved_wall_acceleration_mps2']) > 1e-10):
                    failures.append(dict(time_s=row['time_s'],vehicle_id=vid,kind='resolver_contract'))
                previous[vid] = acceleration
    matched = set()
    with (path/'telemetry.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('kind') != 'command':
                continue
            key = (row['timestamp_s'],row['vehicle_id'])
            if row['owner'] == 'wall' or key in requested:
                count += 1
                matched.add(key)
                if (key not in requested or row['owner'] != 'wall'
                        or row['issued_acceleration_mps2'] is None
                        or abs(row['issued_acceleration_mps2']-requested[key]) > 1e-8):
                    failures.append(dict(time_s=key[0],vehicle_id=key[1],kind='issued_differs_from_resolved'))
    if matched != set(requested):
        failures.append(dict(kind='missing_issued_command'))
    return dict(passed=not failures, failures=failures, issued_wall_command_count=count,
                actual_measured_acceleration_not_universally_bounded=True)


def run_one(registration, job_id):
    root, data = read_registration(registration)
    jobs = [j for j in data['jobs'] if j['id'] == job_id]
    if len(jobs) != 1 or (root/'STOP.json').exists():
        raise ValueError('unknown job or stopped study')
    proof = data['references'][job_id]
    if sha(Path(proof['path'])/'manifest.json') != proof['manifest_sha256']:
        raise ValueError('immutable reference changed')
    (root/'allocations').mkdir(exist_ok=True)
    with (root/'allocations'/f'{job_id}.json').open('x') as stream:
        json.dump(dict(job_id=job_id,allocated_simulated_seconds=2400,retries=0,
            registration_sha256=data['registration_sha256']),stream,indent=2)
    try:
        result = paired.evaluate_arm(job=jobs[0], registration=data, output=root/'jobs'/job_id,
                                      policy=policy_factory(jobs[0],data))
        checks = command_contract(root/'jobs'/job_id)
        write_json(root/'jobs'/job_id/'gentle_command_contract.json', checks)
        if not checks['passed']:
            raise RuntimeError('actual gentle command contract failed')
        return result
    except Exception as exc:
        write_json(root/'STOP.json',dict(job_id=job_id,reason=str(exc),retries=0))
        raise


def compare_job(root, data, job):
    reference = Path(data['references'][job['id']]['path'])
    cm, cs, proof = reference_proof(reference, data['identity'])
    treatment_path = root/'jobs'/job['id']
    tm, ts = paired.verified_arm(treatment_path)
    if tm['registration_sha256'] != data['registration_sha256'] or any(tm.get(k) != v for k,v in data['identity'].items()):
        raise ValueError('treatment registration identity mismatch')
    left = json.loads((reference/'decision_snapshots.json').read_text())['600']
    right = json.loads((treatment_path/'decision_snapshots.json').read_text())['600']
    for key in ('decision_time_s','predecision_state_sha256','predecision_report_evidence_sha256',
                'predecision_report_prefix_sha256','predecision_command_prefix_sha256'):
        if left[key] != right[key]:
            raise ValueError('reference prefix mismatch '+key)
    if cm['exogenous_future_sha256'] != tm['exogenous_future_sha256']:
        raise ValueError('different exogenous future')
    derived = dict(cm, active_source_sha256=data['identity']['active_source_sha256'],
        proposal_sha256=right['proposal_sha256'],decision_time_s=600.,predecision_state_sha256=left['predecision_state_sha256'])
    effect = paired_effect(derived,cs,tm,ts)
    label = benefit_label(effect,BenefitCriteria(**data['criteria']))
    proposal = make_proposal([report_from_payload(r) for r in right['evidence']['reports']],now_s=600.,
        required_lanes=tuple(job['policy_spec']['required_lanes']),member_ids=tuple(job['policy_spec']['member_ids']),
        profile=GentleProfile(**job['policy_spec']['profile']))
    delivery = score_formation_files(treatment_path/'formation.jsonl',treatment_path/'telemetry.jsonl',proposal=proposal)
    comparators = {}
    for name,path in (
        ('original_wall', REFERENCES/'jobs'/f"s{job['seed']}_m2700_p20_t600_wall"),
        ('report_dissolution',ROOT/'runs/active/report_dissolution_v1/jobs'/f"s{job['seed']}_m2700_p20_t600_report_dissolution")):
        om, os = paired.verified_arm(path)
        view = dict(om,active_source_sha256=data['identity']['active_source_sha256'],proposal_sha256=right['proposal_sha256'])
        old_effect = paired_effect(derived,cs,view,os)
        comparators[name] = dict(manifest_sha256=sha(path/'manifest.json'),effect_against_damping=old_effect,
            original_criteria_label=benefit_label(old_effect,BenefitCriteria(**data['criteria'])),
            gentle_minus_comparator_cost_per_request={k:effect['costs_treatment'][k]-old_effect['costs_treatment'][k]
                for k in effect['costs_treatment']})
    contract = command_contract(treatment_path)
    return dict(job_id=job['id'],effect=effect,benefit=label,reference_reuse_proof=proof,
        predecision_prefix_exact=True,derived_reference_view=derived,original_manifests_modified=False,
        gentle_command_contract=contract,legacy_fixed_taper_delivery_score=delivery,
        compact_hold_delivered=delivery['compact_hold_delivered'],
        registered_maneuver_delivered=bool(delivery['compact_hold_delivered'] and delivery['release_ownership_verified']
            and delivery['terminal_phase']=='release' and contract['passed']
            and set(delivery['failure_reasons']) <= {'taper_incomplete_without_explicit_native_handback'}),
        comparators=comparators,qualified_gate=False)


def analyze(registration):
    root,data = read_registration(registration,require_current=False)
    results = []
    for job in data['jobs']:
        path = root/'jobs'/job['id']/'manifest.json'
        manifest = json.loads(path.read_text())
        if manifest.get('status') == 'failed':
            # A stopped implementation is neither a harmful treatment label nor
            # a zero-effect completed experiment. Preserve the missing outcome.
            results.append(dict(job_id=job['id'],status='implementation_failed',
                reported_error=manifest.get('error'),effect=None,benefit=None,qualified_gate=False,
                complete_horizon=False,allocated_simulated_seconds=job['duration_s'],
                reported_completed_simulated_seconds=manifest.get('interaction_diagnostics',{}).get('final_time_s'),
                manifest_sha256=sha(path),original_manifest_modified=False))
        else:
            results.append(compare_job(root,data,job))
    write_json(root/'analysis.json',dict(schema='gentle_wall_result/1',results=results,
        maximum_simulated_seconds=4800,automatic_successors=False,analysis_source_sha256=sha(__file__),
        execution_source_sha256=data['identity']['active_source_sha256'],
        complete_physical_comparison=all(r.get('complete_horizon',True) for r in results)))
    return results
