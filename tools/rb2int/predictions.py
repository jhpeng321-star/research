"""Read only explicitly approved immutable runs; never pool duplicate copies.

Old reserved and new extended tasks may have different orchestration/real-receipt
hashes. Each identity must be individually approved and retain the frozen model
source/recipe hashes. Development bridge parity is a release prerequisite, not a
relaxed task-contract comparison inferred from favorable evaluation outcomes.
"""
from pathlib import Path

import numpy as np

from tools.rbeval.core import ARMS, HS, REGIONS, breakdown, digest, load, safe


def validate_episode(episode, real, *, seed, future):
    if (episode['seed'], episode['future']) != (seed, future):
        raise ValueError('prediction world/future identity differs')
    if episode['initial_sha256'] != real['initial_sha256']:
        raise ValueError('prediction refers to a different recorded start')
    for region in REGIONS:
        bins = episode['bins'][region]
        if len(bins['speed']) != 66 or len(bins['post_counts']) != 60:
            raise ValueError('incomplete region-speed field')
        if bins['speed'][:6] != real['bins'][region]['speed'][:6]:
            raise ValueError('nonidentical recorded prehistory')
        for value, count in zip(bins['speed'][6:], bins['post_counts']):
            if (not isinstance(count, int) or isinstance(count, bool) or count < 0 or
                    ((value is None) != (count == 0)) or
                    (value is not None and not np.isfinite(value))):
                raise ValueError('invalid speed support')
        for h in HS:
            event = breakdown(bins['speed'][:6+h//10])
            stored = bins['events'][str(h)]
            if (event['event'], event['onset_s'], event.get('preexisting', False)) != (
                    stored['event'], stored['onset_s'], stored.get('preexisting', False)):
                raise ValueError('stored event differs from frozen BIC detector')
    for h in HS:
        clock = episode['clocks'][str(h)]
        if not np.isfinite(list(clock.values())).all() or abs(
                clock['team_delay_vs']-clock['road_delay_vs']-clock['source_wait_vs']) > 1e-4:
            raise ValueError('nonfinite or inconsistent source/road delay')


def validate_identity(identity, model):
    if (identity['K'] != 8 or identity['arms'] != list(ARMS) or
            identity['t0'] != 300 or identity['horizon'] != 600 or
            not identity['mix'] or not identity['native_lanes']):
        raise ValueError('registered forecast contract differs')
    if identity['model'] != model['identity_name']:
        raise ValueError('wrong model in approved run')
    for key in ('wm_sha256', 'config_sha256', 'candidate_sha256', 'export_sha256'):
        if identity[key] != model[key]:
            raise ValueError('frozen '+key+' differs')
    if identity['seed'] != 965140 or identity['update'] != 190:
        raise ValueError('forecast RNG/checkpoint clock differs')
    if identity['switches'] != model.get('switches', {}):
        raise ValueError('unregistered explicit model switch')
    for path, expected in model['frozen_model_sources'].items():
        if identity['sources'].get(path) != expected:
            raise ValueError('frozen physical/recipe source missing or changed: '+path)


def read_model(manifest_path, label, real, worlds, *, partial=False):
    manifest = load(manifest_path)
    if manifest['schema'] != 'rb2int-approved-runs/1':
        raise ValueError('explicit approved-run registry required')
    registration = load(manifest['registration_path'])
    if digest(manifest['registration_path']) != manifest['registration_sha256']:
        raise ValueError('run registry registration changed')
    if not set(worlds) <= set(registration['final_comparison_worlds']):
        raise ValueError('unregistered comparison worlds')
    model = manifest['models'][label]
    release = load(manifest['release_path'])
    if (digest(manifest['release_path']) != manifest['release_sha256'] or
            release['status'] != 'READY' or not release['exposure_audit_passed'] or
            'final_comparison' not in release['execution_purposes'] or
            release['registration_sha256'] != manifest['registration_sha256']):
        raise ValueError('frozen exposure-cleared release required')
    binding = release['models'][model['identity_name']]
    if (binding['sources'] != model['frozen_model_sources'] or
            binding['config_sha256'] != model['config_sha256']):
        raise ValueError('approved model differs from its release')
    if not model['development_bridge_passed'] or not model['bridge_receipts']:
        raise ValueError('no development parity receipt for orchestration bridge')
    if 'catalog' in release and release['catalog'][label]['baseline_bridge']['sha256'] not in model['bridge_receipts'].values():
        raise ValueError('model bridge differs from its frozen release proof')
    for path, expected in model['bridge_receipts'].items():
        proof = load(path)
        passed = proof.get('passed') is True or proof.get('receipts', {}).get('tasks/DEV_CANARY_PASS.json', {}).get('passed') is True
        if digest(path) != expected or not passed:
            raise ValueError('development bridge receipt changed')
    lookup = {(r['seed'], r['arm']): r for r in real}
    roots = []
    for run in model['runs']:
        root = safe(run['root'])
        identity_path = root/'IDENTITY.json'
        if digest(identity_path) != run['identity_sha256']:
            raise ValueError('approved run identity changed')
        identity = load(identity_path)
        validate_identity(identity, model)
        if identity['real_receipt_sha256'] != run['real_receipt_sha256']:
            raise ValueError('approved run factual receipt differs')
        roots.append((root, run['identity_sha256']))
    tasks, receipts = {}, []
    for seed in worlds:
        for future in range(8):
            found = [(root/f'w{seed}_k{future}.json', identity_sha)
                     for root, identity_sha in roots if (root/f'w{seed}_k{future}.json').is_file()]
            if len(found) > 1:
                raise ValueError('duplicate forecast task: do not pool device copies')
            if not found:
                continue
            path, identity_sha = found[0]
            task = load(path)
            if (task['schema'] != 'rbeval/1' or task['seed'] != seed or task['future'] != future or
                    task['identity_sha256'] != identity_sha or task['arms'] != list(ARMS) or
                    len(task['episodes']) != 5 or {e['arm'] for e in task['episodes']} != set(ARMS)):
                raise ValueError('forecast task identity differs')
            for episode in task['episodes']:
                validate_episode(episode, lookup[seed, episode['arm']], seed=seed, future=future)
            tasks[seed, future] = task
            receipts.append(dict(task=str(path), sha256=digest(path), identity_sha256=identity_sha))
    missing = [(s, k) for s in worlds for k in range(8) if (s, k) not in tasks]
    if missing and not partial:
        raise ValueError('incomplete registered comparison panel')
    complete = [s for s in worlds if all((s, k) in tasks for k in range(8))]
    pred = {(s, arm): [next(e for e in tasks[s, k]['episodes'] if e['arm'] == arm)
                       for k in range(8)] for s in complete for arm in ARMS}
    return pred, dict(model=label, tasks=len(tasks), complete_worlds=complete,
        requested_worlds=list(worlds), missing_tasks=missing, inputs=receipts,
        approved_run_registry_sha256=digest(manifest_path),
        development_bridge_receipts=model['bridge_receipts'])
