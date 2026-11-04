"""Explicit charged clean1 smoke and final-runtime zero-event safety gate."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import argparse
import multiprocessing
from pathlib import Path
import shutil
import time

from . import launch_production as production, launch_training as training
from . import rerun_cpus as rc, track2_backend as p1, track2_monitor as p3
from .exact_grade import _seal, verify_sealed
from .protocol import digest


def verify_inputs(cfg):
    manifest = training.read(Path(cfg['run_root'])/'STAGING_MANIFEST.json')
    verify_sealed(manifest, 'manifest_sha256')
    for name, entry in manifest['entries'].items():
        if rc.sha(name) != entry['sha256']:
            raise ValueError('required staged reader input changed: '+name)


def smoke(cfg):
    from .launch_preparation import readonly_bank_view
    from .launch_augmentation import run as augment
    from .storage import ledger
    root = Path(cfg['run_root'])
    training.health(root)
    verify_inputs(cfg)
    augmentation = augment(cfg)
    # Augmentation is once-only and has its own prelaunch phase. Smoke charges
    # only the six complete episodes and two matched practice A references.
    training.write(root/'SMOKE_STARTED.json', dict(config_sha256=digest(cfg), unix=time.time()))
    output = root/'smoke'
    output.mkdir()
    for arm in ('B', 'R'):
        readonly_bank_view(root/'replay_bank'/arm, output/'replay_bank'/arm)
    for arm in ('A', 'B', 'R'):
        shutil.copyfile(root/f'{arm}_checkpoint_000.pt', output/f'{arm}_checkpoint_000.pt')
        if arm != 'A':
            shutil.copyfile(root/f'{arm}_track2_000.pt', output/f'{arm}_track2_000.pt')
    shutil.copyfile(root/rc.RULE_FILE, output/rc.RULE_FILE)
    shutil.copyfile(root/'AUTHORITY.md', output/'AUTHORITY.md')
    updates = [training.update(cfg, arm, 1, smoke=True) for arm in ('A', 'B', 'R')]
    monitor = training.monitor_update(cfg, 1, smoke=True)
    for update in updates:
        if update['arm'] == 'A':
            continue
        m, learning = update['metrics'], update['learning']
        if (update['execution']['accepted_moves'] <= 0 or m['decisions'] <= 0
                or m['merge_missing_share'] >= 1 or m['weighted_starts'] <= 0
                or m['augmented_examples'] <= 0 or m['track2_reward']['cav_outcomes'] <= 0
                or learning['bank']['pairs_total'] <= 0 or learning['bank']['optimizer_steps'] <= 0
                or learning['bank']['imitation_failed'] or learning['bank_trigger']['examples'] <= 0):
            raise ValueError('smoke component missing actual usable receipt; fix and requalify')
    dry = [training.diagnostic(cfg, u, dry=True, artifacts=output) for u in (6, 12, 24)]
    if any(r['cells'] != 10 or r['status'] != 'dry_run' or r['branches'] <= 0 for r in dry):
        raise ValueError('complete replicated-test dry grid required')
    charged = sum(r['sim_s'] for r in ledger(root) if r['phase'] == 'track2_smoke')
    if charged > 20000:
        raise ValueError('owner total smoke cap exceeded')
    receipt = _seal(dict(status='SMOKE_PASS', config_sha256=digest(cfg),
        source_manifest_sha256=cfg['source_manifest_sha256'], unix=time.time(),
        smoke_sim_s=charged, augmentation_sha256=augmentation['receipt_sha256'],
        updates=[r['update_sha256'] for r in updates], monitor_sha256=monitor['receipt_sha256'],
        replicated_dry=[r['receipt_sha256'] for r in dry],
        dense_teacher=dict(enabled=False, reason='owner 02:28 redesign'),
        wm_group_credit=dict(enabled=False, reason='owner 02:28 redesign'),
        replacement_reward='native P2 real observable neighbourhood SUMO reward'), 'receipt_sha256')
    training.write(root/'SMOKE_PASS.json', receipt)
    return receipt


def gate_pairs(cfg):
    """Prospective first-three-state gate, full windows and both bank policies."""
    backend = training.MonitorBackend(cfg)
    row = backend.registration['replicates'][0]
    states = backend.resolve_states(row)[:3]
    if len(states) != 3:
        raise ValueError('three pre-registered full-window gate states required')
    native = production.backend_config(cfg)
    common = production.checkpoint_weights(native, 'A', 0)
    pairs = []
    for arm in ('B', 'R'):
        current = production.checkpoint_weights(native, arm, 0)
        job = dict(row, id='clean1_gate_'+arm, arm=arm, update=0,
                   **p3.diagnostic_identity(states))
        for group, specs in p3.isolated_expansion(job, states, current, common):
            pairs.extend(specs)
    cost = sum(s['sim_s'] for s, _ in pairs)
    if cost > rc.GATE_SIM_S:
        raise ValueError('complete full-window gate exceeds its quote; no horizon reduction')
    return pairs


def gate(cfg):
    root, native = Path(cfg['run_root']), production.backend_config(cfg)
    training.health(root)
    smoke_receipt = training.read(root/'SMOKE_PASS.json')
    verify_sealed(smoke_receipt, 'receipt_sha256')
    if smoke_receipt['config_sha256'] != digest(cfg):
        raise ValueError('mandatory smoke does not bind this final runtime')
    verify_inputs(cfg)
    pairs = gate_pairs(cfg)
    jobs = [dict(id='gate_'+s['verification_id'], sim_s=int(s['sim_s'])) for s, _ in pairs]
    training.write(root/'GATE_STARTED.json', dict(config_sha256=digest(cfg),
        unix=time.time(), job_manifest=jobs, specs=[s for s, _ in pairs]))
    p1._reserve(native, jobs, 'gate_bank_verification_ext')
    with ProcessPoolExecutor(max_workers=16, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(production.run_branch, spec, snapshot, native,
            root/'safety_gate'/job['id']) for job, (spec, snapshot) in zip(jobs, pairs)]
        results = [f.result(timeout=3600) for f in futures]
    if any(r.get('status') != 'graded' for r in results):
        training.write(root/'GATE_FAILED.json', dict(status='INCOMPLETE', results=results))
        raise ValueError('incomplete final-runtime safety gate; no automatic retry')
    safety = {k: sum(r['safety'][k] for r in results) for k in p3.SAFETY}
    if safety['collisions'] or safety['teleports']:
        training.write(root/'PAUSE_SAFETY.json', dict(reason='final-runtime gate safety failure', safety=safety))
        raise ValueError('gate requires exactly zero collisions and teleports')
    receipt = _seal(dict(status='PASS', config_sha256=digest(cfg),
        source_manifest_sha256=cfg['source_manifest_sha256'], smoke_sha256=smoke_receipt['receipt_sha256'],
        augmentation_sha256=smoke_receipt['augmentation_sha256'], unix=time.time(), safety=safety,
        charged_sim_s=sum(j['sim_s'] for j in jobs), branches=len(results), results=results,
        rule='0 collisions, 0 teleports; three prospectively fixed probe states, B/R group/solos/base, unchanged full windows'),
        'receipt_sha256')
    training.write(root/'GATE_PASS.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'smoke', 'gate'])
    parser.add_argument('--source-root')
    parser.add_argument('--config')
    parser.add_argument('--resume-preparation', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare':
        from .launch_preparation import prepare
        result = prepare(args.source_root, resume=args.resume_preparation)
        print(digest(result))
    else:
        cfg = training.checked_config(args.config)
        result = smoke(cfg) if args.action == 'smoke' else gate(cfg)
        print(result['receipt_sha256'])


if __name__ == '__main__':
    main()
