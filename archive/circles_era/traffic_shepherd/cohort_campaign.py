"""Bounded Part C orchestration after completed, verified Part B.

No simulator runs at import or registration. Replay and physical comparisons
are supplied by cohort_evidence, with independent artifact verification.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from traffic_shepherd import paired_experiments as paired
from traffic_shepherd import lifecycle_study as previous
from traffic_shepherd.brake_hold_study import command_contract, rows, source_bridge
from traffic_shepherd.cohort_study import (
    SCHEMA, SELECTOR, allocation_contract, bind_selected_job, matched_prefix,
    policy_factory, validate_reuse_proof,
)
from traffic_shepherd.config import ROOT
from traffic_shepherd.dissolution_study import reference_proof
from traffic_shepherd.lifecycle_wall import LifecycleProfile, VERSION
from traffic_shepherd.workflow import sha, write_json

PROTOCOL = 'docs/COHORT_ELIGIBILITY_CHECK.md'
EXPECTED_SELECTIONS = {4107:['mainline.385','mainline.386'],4108:['mainline.394','mainline.395']}


def _verify_mechanics_receipt(receipt):
    from traffic_shepherd import lifecycle_mechanics as mechanics
    root,data = mechanics.read_registration(receipt['registration_path'],require_current=False)
    if ((root/'STOP.json').exists() or not receipt.get('passed')
            or receipt['registration_sha256'] != data['registration_sha256']
            or receipt['registration_file_sha256'] != sha(root/'registration.json')
            or receipt['source_sha256'] != data['identity']['active_source_sha256']):
        raise ValueError('Part B mechanics receipt does not bind a completed passing stage')
    if {p.stem for p in (root/'allocations').glob('*.json')} != set(mechanics.NEW_CASE_IDS):
        raise ValueError('Part B mechanics allocations changed')
    cases = [data['repair_evidence']['reused_normal']]+[
        dict(mechanics._verify_case(root,data,case),mode='new_allocation') for case in mechanics.NEW_CASE_IDS]
    if receipt['cases'] != cases or receipt['physical_evidence_simulated_seconds'] != 240:
        raise ValueError('Part B mechanics physical evidence changed')
    return paired.digest(receipt)


def _qualified_part_b(results):
    if any(r['benefit']['label'] == 'beneficial' and r['lifecycle']['registered_maneuver_delivered']
           for r in results):
        raise ValueError('Part B already has qualifying delivered benefit; conditional C1 is not selected')
    if len(results) != 2 or {r['seed'] for r in results} != {4107,4108}:
        raise ValueError('Part B must cover both exact development states')


def verified_part_b(registration, identity):
    root,data = previous.read_registration(registration,require_current=False)
    if (root/'STOP.json').exists():
        raise ValueError('Stopped Part B cannot authorize Part C')
    analysis_path = root/'analysis.json'
    analysis = json.loads(analysis_path.read_text())
    if not analysis.get('complete') or analysis.get('pending') or data['policy_version'] != VERSION:
        raise ValueError('Complete current-V2 Part B physical verdict required')
    mechanics_sha = _verify_mechanics_receipt(data['mechanics_evidence'])
    # Reconstruct final analysis through the original comparator instead of
    # accepting a hand-written qualifying/nonqualifying result dictionary.
    verified = [previous.compare_job(root,data,job) for job in data['jobs']]
    if paired.digest(verified) != paired.digest(analysis['results']):
        raise ValueError('Saved Part B analysis does not match verified physical artifacts')
    _qualified_part_b(verified)
    for job in data['jobs']:
        manifest,_ = paired.verified_arm(root/'jobs'/job['id'])
        source_bridge(manifest,identity)
    proof = dict(registration_path=str(root/'registration.json'),
        registration_file_sha256=sha(root/'registration.json'),registration_sha256=data['registration_sha256'],
        analysis_path=str(analysis_path),analysis_sha256=sha(analysis_path),
        mechanics_evidence_sha256=mechanics_sha,verified_comparisons_sha256=paired.digest(verified))
    return root,data,proof


def _case_bindings(b_root,b_data,output,identity):
    jobs,selections,snapshots,preceding,references,bindings = [],{},{},{},{},{}
    for b_job in b_data['jobs']:
        b_path = b_root/'jobs'/b_job['id']
        bm,_ = paired.verified_arm(b_path)
        bridge = source_bridge(bm,identity)
        if bm['registration_sha256'] != b_data['registration_sha256'] or bm['duration_s'] != 2400:
            raise ValueError('Part B physical source differs from its registration')
        snapshot = json.loads((b_path/'decision_snapshots.json').read_text())['600']
        c_job,selection = bind_selected_job(b_job,snapshot)
        if selection['selected']['member_ids'] != EXPECTED_SELECTIONS[b_job['seed']]:
            raise ValueError('Acquired choice changed from declared one-reuse/one-new hypothesis')
        paired.validate_job(c_job)
        reference = Path(b_data['references'][b_job['id']]['path'])
        dm,_,dproof = reference_proof(reference,identity)
        if dproof['manifest_sha256'] != b_data['references'][b_job['id']]['manifest_sha256']:
            raise ValueError('Matched damping reference differs from Part B')
        ds = json.loads((reference/'decision_snapshots.json').read_text())['600']
        matched_prefix(ds,snapshot)
        if dm['exogenous_future_sha256'] != bm['exogenous_future_sha256']:
            raise ValueError('Part B and damping requested future differ')
        cid = c_job['id']
        jobs.append(c_job)
        selections[cid],snapshots[cid],references[cid] = selection,snapshot,dproof
        preceding[cid] = dict(path=str(b_path),job=b_job,manifest_sha256=sha(b_path/'manifest.json'),
            summary_sha256=sha(b_path/'summary.json'),snapshot_sha256=sha(b_path/'decision_snapshots.json'),
            registration_path=str(b_root/'registration.json'),registration_sha256=b_data['registration_sha256'],
            source_bridge=bridge)
        reuse = b_job['seed'] == 4107
        bindings[cid] = dict(mode='verified_reuse' if reuse else 'simulate',
            path=str(b_path if reuse else output/'jobs'/cid),source_job_id=b_job['id'],
            source_manifest_sha256=sha(b_path/'manifest.json'))
    return dict(jobs=jobs,selections=selections,bound_snapshots=snapshots,
                preceding_b=preceding,references=references,bindings=bindings)


def register(output,b_registration):
    from traffic_shepherd.cohort_evidence import replay_reuse
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('New Part C registration directory required')
    identity = paired.current_identity()
    b_root,b_data,b_proof = verified_part_b(b_registration,identity)
    cases = _case_bindings(b_root,b_data,output,identity)
    budget = allocation_contract(cases['jobs'],cases['bindings'])
    data = dict(schema=SCHEMA,registered_at_utc=datetime.now(timezone.utc).isoformat(),
        identity=identity,part_b=b_proof,criteria=b_data['criteria'],communication=b_data['communication'],
        selector_version=SELECTOR,policy_version=VERSION,interaction_diagnostics=True,
        assignment='registered_experimental_assignment_not_trained_benefit_gate',
        protocol_path=PROTOCOL,protocol_sha256=sha(ROOT/PROTOCOL),**cases,**budget)
    # The replay policy needs an assignment identifier, but replay proofs bind
    # B artifacts/selection/source and must not bind this provisional digest.
    draft = dict(data,registration_sha256=paired.digest(data))
    for job in data['jobs']:
        binding = data['bindings'][job['id']]
        if binding['mode'] != 'verified_reuse':
            continue
        before = data['preceding_b'][job['id']]
        proof = replay_reuse(Path(before['path']),before['job'],job,b_data,draft)
        validate_reuse_proof(proof,selection=data['selections'][job['id']],
            source_manifest_sha256=before['manifest_sha256'],source_job=before['job'])
        binding['reuse_proof'] = proof
        binding['reuse_proof_sha256'] = paired.digest(proof)
    data['registration_sha256'] = paired.digest(data)
    output.mkdir(parents=True,exist_ok=False)
    snapshot = output/'source_snapshot'
    snapshot.mkdir()
    for name in identity['active_source_sha256']:
        shutil.copy2(ROOT/'traffic_shepherd'/name,snapshot/name)
    write_json(output/'registration.json',data)
    return dict(registration=str(output/'registration.json'),simulations_started=0,
        new_jobs=data['new_job_ids'],reused_jobs=data['reused_job_ids'],maximum_simulated_seconds=2400)


def read_registration(path, *, require_current=True):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if data.get('schema') != SCHEMA or data.get('registration_sha256') != paired.digest(
            {k:v for k,v in data.items() if k != 'registration_sha256'}):
        raise ValueError('Part C registration digest/schema changed')
    expected_budget = allocation_contract(data['jobs'],data['bindings'])
    if any(data.get(k) != v for k,v in expected_budget.items()):
        raise ValueError('Part C exact one-attempt budget changed')
    if (data.get('protocol_path') != PROTOCOL or data.get('selector_version') != SELECTOR
            or data.get('policy_version') != VERSION or data.get('interaction_diagnostics') is not True
            or data.get('assignment') != 'registered_experimental_assignment_not_trained_benefit_gate'):
        raise ValueError('Part C scientific contract changed')
    if require_current and (data['identity'] != paired.current_identity()
            or data['protocol_sha256'] != sha(ROOT/PROTOCOL)):
        raise ValueError('Part C execution source/scenario/protocol changed after registration')
    for name,expected in data['identity']['active_source_sha256'].items():
        if sha(path.parent/'source_snapshot'/name) != expected:
            raise ValueError('Part C frozen source snapshot changed')
    proof = data['part_b']
    if (sha(Path(proof['registration_path'])) != proof['registration_file_sha256']
            or sha(Path(proof['analysis_path'])) != proof['analysis_sha256']):
        raise ValueError('Bound Part B registration or final analysis changed')
    b_root,b_data = previous.read_registration(proof['registration_path'],require_current=False)
    if (b_root/'STOP.json').exists() or b_data['registration_sha256'] != proof['registration_sha256']:
        raise ValueError('Part B prerequisite changed')
    analysis = json.loads(Path(proof['analysis_path']).read_text())
    if not analysis.get('complete') or analysis.get('pending'):
        raise ValueError('Part B verdict is incomplete')
    _qualified_part_b(analysis['results'])
    if (paired.digest(analysis['results']) != proof['verified_comparisons_sha256']
            or _verify_mechanics_receipt(b_data['mechanics_evidence']) != proof['mechanics_evidence_sha256']
            or data['criteria'] != b_data['criteria'] or data['communication'] != b_data['communication']):
        raise ValueError('Original criteria, communication or verified prerequisites changed')
    expected = _case_bindings(b_root,b_data,path.parent,data['identity'])
    for key in ('jobs','selections','bound_snapshots','preceding_b','references'):
        if data[key] != expected[key]:
            raise ValueError('Acquired selection or immutable artifact binding changed: '+key)
    if set(data['bindings']) != set(expected['bindings']):
        raise ValueError('Part C binding scope changed')
    for job in data['jobs']:
        binding = data['bindings'][job['id']]
        expected_binding = expected['bindings'][job['id']]
        if any(binding.get(k) != v for k,v in expected_binding.items()):
            raise ValueError('Bound physical path or mode changed')
        if binding['mode'] == 'verified_reuse':
            allowed = set(expected_binding)|{'reuse_proof','reuse_proof_sha256'}
            if set(binding) != allowed or paired.digest(binding['reuse_proof']) != binding['reuse_proof_sha256']:
                raise ValueError('Reuse proof binding changed')
            before = data['preceding_b'][job['id']]
            validate_reuse_proof(binding['reuse_proof'],selection=data['selections'][job['id']],
                source_manifest_sha256=before['manifest_sha256'],source_job=before['job'])
        elif binding != expected_binding:
            raise ValueError('New episode binding gained unregistered fields')
    return path.parent,data


def run_one(registration,job_id):
    root,data = read_registration(registration)
    job = next((j for j in data['jobs'] if j['id'] == job_id),None)
    if (job is None or (root/'STOP.json').exists()
            or data['bindings'][job_id]['mode'] != 'simulate'):
        raise ValueError('Unknown/reused job or stopped Part C stage')
    allocations = root/'allocations'
    allocations.mkdir(exist_ok=True)
    if any(p.stem != job_id for p in allocations.glob('*.json')):
        raise ValueError('Unexpected allocation consumes or changes the one-attempt budget')
    with (allocations/f'{job_id}.json').open('x') as stream:
        json.dump(dict(job_id=job_id,allocated_simulated_seconds=2400,retries=0,
            registration_sha256=data['registration_sha256']),stream)
    try:
        output = root/'jobs'/job_id
        result = paired.evaluate_arm(job=job,registration=data,output=output,policy=policy_factory(job,data))
        decision = next(r for r in rows(output/'policy.jsonl') if r['time_s'] == 600.)
        if decision.get('cohort_selection_verified') is not True:
            raise ValueError('Runtime acquired selection mismatch invalidates treatment assignment')
        contract = command_contract(output,LifecycleProfile(**job['policy_spec']['profile']))
        write_json(output/'cohort_command_contract.json',contract)
        if not contract['passed']:
            raise ValueError('Part C actual command contract failed')
        return result
    except Exception as exc:
        write_json(root/'STOP.json',dict(job_id=job_id,reason=str(exc),automatic_retries=0))
        raise


def analyze(registration):
    from traffic_shepherd.cohort_evidence import compare_case
    root,data = read_registration(registration,require_current=False)
    if (root/'STOP.json').exists():
        raise ValueError('Stopped Part C cannot produce efficacy analysis')
    results,pending = [],[]
    for job in data['jobs']:
        binding = data['bindings'][job['id']]
        manifest = Path(binding['path'])/'manifest.json'
        if not manifest.exists() or json.loads(manifest.read_text()).get('status') != 'complete':
            pending.append(job['id'])
            continue
        results.append(compare_case(root,data,job))
    completed_new = sum(data['bindings'][r['job_id']]['mode'] == 'simulate' for r in results)
    answer = dict(schema='report_cohort_selection_results/1',complete=len(results)==2 and not pending,
        pending=pending,results=results,independent_seed_count=2,
        new_physical_trajectories=completed_new,maximum_new_physical_trajectories=1,
        maximum_new_simulated_seconds=2400,original_criteria_unchanged=True,
        population_generalization_established=False,automatic_successors=False)
    write_json(root/'analysis.json',answer)
    return dict(analysis_path=str(root/'analysis.json'),complete=answer['complete'],pending=pending,
        results=[dict(job_id=r['job_id'],benefit=r['benefit']) for r in results])
