"""Clean1 detached arm owners and the single P3 monitoring/diagnostic owner.

All live entry points require the same immutable approved configuration. Smoke
uses separate artifacts/models/plans and the parent experiment's sole ledger.
"""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import time
import traceback

import torch

from . import launch_production as production, rerun_cpus as rc
from . import track2_backend as p1, track2_monitor as p3, track2_reward as p2
from .exact_grade import _seal, verify_sealed
from .protocol import digest


def write(path, value):
    path = Path(rc.guard_path(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    # Detached arms consume each other's receipts. Publish only after complete
    # serialization/fsync, while retaining the original no-overwrite contract.
    temporary = path.with_name('.'+path.name+f'.writer{os.getpid()}.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)  # atomic publication, refuses an existing target
    temporary.unlink()       # the published hard link retains these exact bytes


def read(path):
    return json.loads(Path(rc.guard_path(path)).read_text(encoding='utf-8'))


def checked_config(path):
    cfg = read(path)
    rc.check_pins(cfg)
    production.backend_config(cfg)
    approval = read(Path(cfg['run_root'])/'OWNER_APPROVAL.json')
    if approval.get('approved') is not True or approval.get('config_sha256') != digest(cfg):
        raise ValueError('full production config must be bound by owner approval')
    if cfg['budget']['total']+960 > 5000000 or cfg['smoke']['total_sim_s'] > 20000:
        raise ValueError('cumulative quote/smoke exceeds approved caps')
    manifest = read(Path(cfg['source_root'])/'SOURCE_MANIFEST.json')
    if digest(manifest) != cfg['source_manifest_sha256']:
        raise ValueError('source manifest binding mismatch')
    for name, sha in manifest['files'].items():
        if rc.sha(Path(cfg['source_root'])/name) != sha:
            raise ValueError('immutable production source changed: '+name)
    for name in ('staging_manifest_sha256', 'isolated_registration_sha256'):
        target = 'STAGING_MANIFEST.json' if name.startswith('staging') else 'ISOLATED_REGISTRATION.json'
        if rc.sha(Path(cfg['run_root'])/target) != cfg[name]:
            raise ValueError('approved registration/input manifest changed')
    return cfg


def health(root):
    from .warm_health import assert_running
    if (Path(root)/'FAULT.json').exists() or (Path(root)/'PAUSE_SAFETY.json').exists():
        raise ValueError('durable clean1 fault/safety pause; diagnose before continuation')
    for arm in ('A', 'B', 'R'):
        assert_running(root, arm)


def wait_for(path, root, *, timeout=86400):
    started = time.monotonic()
    while not Path(path).is_file():
        health(root)
        if time.monotonic()-started > timeout:
            raise TimeoutError('waiting for '+str(path))
        time.sleep(2)
    health(root)
    return read(path)


def release(cfg):
    root = Path(cfg['run_root'])
    if 'clean2_budget' in cfg:
        from .launch_clean2 import verify_previous
        verify_previous(cfg)
    if 'clean3_budget' in cfg:
        from .clean3_restart import verify_previous
        verify_previous(cfg)
    for name, status in (('SMOKE_PASS.json', 'SMOKE_PASS'), ('GATE_PASS.json', 'PASS')):
        receipt = read(root/name)
        verify_sealed(receipt, 'receipt_sha256')
        if receipt['status'] != status or receipt['config_sha256'] != digest(cfg):
            raise ValueError('final-runtime '+name+' binding missing/mismatched')
    gate = read(root/'GATE_PASS.json')
    if gate['safety']['collisions'] or gate['safety']['teleports']:
        raise ValueError('gate requires zero collisions and teleports')


def augmentation_receipt(cfg):
    root = Path(cfg['run_root'])
    if cfg['plan_augment']['enabled']:
        return read(root/'AUGMENTATION_READY.json')
    if cfg.get('launch_augmentation_decision') != rc.AUGMENTATION_OFF_DECISION:
        raise ValueError('symmetric owner augmentation-off decision required')
    receipt = read(root/'AUGMENTATION_DISABLED.json')
    verify_sealed(receipt, 'receipt_sha256')
    if receipt['config_sha256'] != digest(cfg) or receipt['admitted'] != dict(B=0, R=0):
        raise ValueError('augmentation-off receipt/config mismatch')
    # All experimental children stay in the archived diagnostic bank. The live
    # bank retains original-path read-only evidence only, for both arms.
    for arm in ('B', 'R'):
        bank = root/'replay_bank'/arm
        if (bank/'PRACTICE_AUGMENTED.json').exists():
            raise ValueError('disabled augmented practice must not enter either arm')
        for folder in ('decisions', 'confirmations_all8'):
            if any(not p.is_symlink() for p in (bank/folder).glob('*.json')):
                raise ValueError('diagnostic augmented bank evidence entered live bank')
    return receipt


def update(cfg, arm, u, *, smoke=False):
    root = Path(cfg['run_root'])
    augmentation = augmentation_receipt(cfg)
    artifacts = root/'smoke' if smoke else root
    native = production.backend_config(cfg, root=artifacts, smoke=smoke)
    native['update'] = u
    from . import g4_runtime as g4
    paired_credit = arm != 'A' and g4.enabled(native, u, smoke=smoke)
    health(root)
    if not smoke and u > 1:
        wait_for(root/f'LEARNING_MONITOR_u{u-1:03}.json', root)
    if arm != 'A':
        wait_for(artifacts/f'A_UPDATE_{u:03}.json', root)
    if arm == 'R' and not paired_credit:
        wait_for(artifacts/f'B_UPDATE_{u:03}.json', root)
    if arm == 'R' and paired_credit:
        # B has frozen its practice plan/count before R starts; neither arm
        # optimizes until both complete ordinary-training inputs are published.
        wait_for(g4.input_path(native, 'B', u), root)
    started = dict(arm=arm, update=u, pid=os.getpid(), unix=time.time(),
                   config_sha256=digest(cfg), smoke=smoke)
    write(artifacts/f'updates/{arm}_{u:03}_STARTED.json', started)
    hooks = p1.build_arm_runtime(native, arm)
    episodes = hooks['collect_update']()
    # Actual TTC events use the same public observable tape reader as prior runs.
    from .replay_bank import ttc_events
    for episode in episodes:
        trace = artifacts/'jobs'/episode['id']/'observable.jsonl.gz'
        episode['ttc_events'] = int(ttc_events(trace, 300, 1200))
    safety = hooks['safety'](episodes)  # unchanged paired A composite BEFORE optimization
    health(root)
    envelope = p2.reward_update(native, arm, u, episodes) if arm != 'A' else None
    if paired_credit:
        synergy = g4.paired_envelope(native, arm, u, episodes,
            wait=lambda path: wait_for(path, root))
        learning = hooks['train_update'](episodes, rewards=envelope, synergy=synergy)
    else:
        learning = hooks['train_update'](episodes, rewards=envelope)
    if arm != 'A':
        bank = production.bank_move_supervision(hooks['runtime'])
        learning.update(bank_move_ce=bank['final_ce'], bank_trigger=bank)
    checkpoint = hooks['save_checkpoint']()
    from .practice_starts import update_report
    metrics = production.update_metrics(arm, episodes, learning, envelope,
        augmented_examples=0 if arm == 'A' else augmentation['admitted'][arm])
    extra = {}
    if 'clean2_budget' in cfg or 'clean3_budget' in cfg:
        from .clean2_receipts import update_evidence
        extra['clean2_fix_receipts'] = update_evidence(arm, episodes, learning, envelope, metrics)
    value = _seal(dict(**started, finished_unix=time.time(), checkpoint=checkpoint,
        checkpoint_sha256=rc.sha(checkpoint), practice=update_report(episodes),
        safety=safety, learning=learning, metrics=metrics,
        execution=dict(accepted_moves=sum(sum(bool(v) for v in s['sample'].get(
            'track2_trigger', {}).get('accepted', {}).values()) for e in episodes for s in e['steps'])),
        physical=[dict(id=e['id'], world=e['world'], totals=e['totals'],
                        ttc_events=e['ttc_events']) for e in episodes], **extra), 'update_sha256')
    write(artifacts/f'{arm}_UPDATE_{u:03}.json', value)
    return value


def run_arm(cfg, arm):
    root = Path(cfg['run_root'])
    release(cfg)
    write(root/f'owners/{arm}.json', dict(pid=os.getpid(), unix=time.time(),
        start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21], config_sha256=digest(cfg)))
    torch.set_num_threads(1)
    for u in range(1, 25):
        # Only completed updates can be adopted. Never redo an attempted update.
        done = root/f'{arm}_UPDATE_{u:03}.json'
        if done.exists():
            receipt = read(done); verify_sealed(receipt, 'update_sha256')
            if receipt['config_sha256'] != digest(cfg):
                raise ValueError('cannot adopt a different final runtime')
            continue
        update(cfg, arm, u)
    write(root/f'{arm}_COMPLETE.json', dict(arm=arm, updates=24, unix=time.time()))


class MonitorBackend:
    def __init__(self, cfg, *, dry=False, root=None):
        self.cfg = deepcopy(cfg)
        self.native = production.backend_config(cfg)
        self.root = Path(root or cfg['run_root'])
        self.registration = read(Path(cfg['run_root'])/'ISOLATED_REGISTRATION.json')
        self.dry, self.pool, self.futures, self.reserved = dry, None, {}, set()

    def resolve_states(self, job):
        row = next(r for r in self.registration['replicates'] if r['world_seed'] == job['world_seed'])
        states = []
        for item in row['states']:
            if (rc.sha(item['snapshot_path']) != item['snapshot_file_sha256'] or
                    rc.sha(item['receipt_path']) != item['receipt_file_sha256']):
                raise ValueError('registered diagnostic state changed')
            states.append((torch.load(item['snapshot_path'], weights_only=False, map_location='cpu'),
                           read(item['receipt_path'])))
        return states

    def weights(self, arm, update):
        common = production.checkpoint_weights(self.native, 'A', 0)
        # A smoke dry-run uses its saved u1 state to expand the future grid;
        # this is explicitly a quote, with no simulated result or forecast.
        native = dict(self.native, run_root=str(self.root))
        current = production.checkpoint_weights(native, arm, 1 if self.dry else update)
        return current, common

    def reserve(self, jobs, phase):
        if self.dry:
            raise ValueError('dry replicated expansion cannot charge')
        health(self.cfg['run_root'])
        p1._reserve(self.native, jobs, phase)
        self.reserved.update(j['id'] for j in jobs)
        return dict(reserved=True, phase=phase, sim_s=sum(j['sim_s'] for j in jobs),
                    job_ids=[j['id'] for j in jobs])

    def prepare_branches(self, prepared):
        if self.dry or self.pool is not None:
            raise ValueError('one owned pool per diagnostic wave')
        self.pool = ProcessPoolExecutor(max_workers=16, mp_context=multiprocessing.get_context('spawn'))
        for job, expanded in prepared:
            for group, pairs in expanded:
                for spec, snapshot in pairs:
                    ident = spec['verification_id']
                    if ident not in self.reserved or ident in self.futures:
                        raise ValueError('branch lacks a unique whole-wave reservation')
                    self.futures[ident] = (spec['branch_sha256'], self.pool.submit(
                        production.run_branch, spec, snapshot, self.native, self.root/'branches'/ident))

    def run_branch(self, spec, snapshot):
        sha, future = self.futures.pop(spec['verification_id'])
        if sha != spec['branch_sha256']:
            raise ValueError('dispatch spec changed after precharge')
        return future.result(timeout=3600)

    def close(self):
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)


def diagnostic(cfg, u, *, dry=False, artifacts=None):
    backend = MonitorBackend(cfg, dry=dry, root=artifacts)
    monitor = production.monitor_config(cfg, root=artifacts)
    monitor.update(isolated_registration=backend.registration,
                   monitor_backend=backend, monitor_dry_run=dry)
    try:
        if u in (2, 4):
            return p3.probe(monitor, u)
        return p3.replicated(monitor, u)
    finally:
        backend.close()


def monitor_update(cfg, u, *, smoke=False):
    root = Path(cfg['run_root'])
    artifacts = root/'smoke' if smoke else root
    rows = []
    for arm in ('A', 'B', 'R'):
        receipt = wait_for(artifacts/f'{arm}_UPDATE_{u:03}.json', root)
        verify_sealed(receipt, 'update_sha256')
        row = p3.monitor_update(production.monitor_config(cfg, root=artifacts), arm, u,
            dict(metrics=receipt['metrics'], sources=dict(update_sha256=receipt['update_sha256'])))
        rows.append(row)
    value = _seal(dict(update=u, unix=time.time(), config_sha256=digest(cfg), arms=rows), 'receipt_sha256')
    if any(row['action'] == 'PAUSE_SAFETY' for row in rows):
        write(root/'PAUSE_SAFETY.json', value)
        raise ValueError('safety failure; pause all arms')
    write(artifacts/f'LEARNING_MONITOR_u{u:03}.json', value)
    return value


def run_monitor(cfg):
    release(cfg)
    root = Path(cfg['run_root'])
    write(root/'owners/monitor.json', dict(pid=os.getpid(), unix=time.time()))
    for u in range(1, 25):
        if not (root/f'LEARNING_MONITOR_u{u:03}.json').exists():
            monitor_update(cfg, u)
        if u in (2, 4, 6, 12, 24):
            diagnostic(cfg, u)
        # Scheduled lessons, u8 fixed and u24 shared paired evaluations are
        # dispatched by the same diagnostic owner through the final adapters.
        from .launch_evaluation import scheduled
        scheduled(cfg, u)
    write(root/'MONITOR_COMPLETE.json', dict(unix=time.time(), updates=24))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['arm', 'monitor'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--arm', choices=['A', 'B', 'R'])
    args = parser.parse_args()
    cfg = checked_config(args.config)
    try:
        if args.action == 'arm':
            run_arm(cfg, args.arm)
        else:
            run_monitor(cfg)
    except BaseException as error:
        root = Path(cfg['run_root'])
        fault = dict(unix=time.time(), pid=os.getpid(), arm=args.arm,
                     error=repr(error), traceback=traceback.format_exc())
        if not (root/'FAULT.json').exists():
            write(root/'FAULT.json', fault)
        if not (root/'STOP.json').exists():
            write(root/'STOP.json', fault)
        raise


if __name__ == '__main__':
    main()
