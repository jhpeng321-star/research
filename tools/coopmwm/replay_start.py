"""MAIN07:35 durable, locked B-REPLAY start watcher and remote supervision.

No SSH connection or local-agent process owns training. All new physics is
precharged; old jobs/claims are never retried. Numerical PPO/search code is reused.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import types

from .protocol import digest, sha
from .storage import read, write

NAME = 'B_REPLAY_START_0835_V4.json'
PREFIX = 'replay0735'


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.'+path.name+f'.{os.getpid()}.{time.time_ns()}')
    with tmp.open('x') as f:
        json.dump(value, f, sort_keys=True);f.flush();os.fsync(f.fileno())
    os.replace(tmp, path)


def owner(pid=None):
    pid = os.getpid() if pid is None else pid
    return dict(pid=pid, start_ticks=Path('/proc', str(pid), 'stat').read_text().split()[21],
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), unix=time.time())


def alive(receipt):
    if receipt['boot_id'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        return False
    p = Path('/proc', str(receipt['pid']))
    if not p.exists():return False
    s = (p/'stat').read_text().split()
    # A reused PID is never adopted or signalled.
    return s[21] == receipt['start_ticks'] and s[2] != 'Z'


def config(root):
    root = Path(root);v = read(root/NAME)
    assert v['amendment_sha256'] in (root/'AUTHORITY.md').read_text()
    s = Path(v['source_root'])
    assert Path(__file__).resolve() == s/'pytools/tools/coopmwm/replay_start.py'
    cpu = read(s/'LS_CPU_RESULT.json')
    assert cpu['pass_'] and cpu['sim_s'] == cpu['skips'] == 0
    for n, h in cpu['source_hashes'].items():assert sha(s/n) == h, n
    return v


def accounting(v):
    from . import accel_storage
    from .accel_protocol import SUPERSEDED_CHARGE
    from tools.coopnative import storage
    b = v['budget']
    assert b['total'] == 14500000 and sum(b['phases'].values()) <= b['total']
    accel_storage._reserve = types.FunctionType(storage.reserve.__code__,
        dict(storage.reserve.__globals__, CAP=b['total']-SUPERSEDED_CHARGE, CAPS=b['phases']))


def prereqs(root, v, verify_gpu=True):
    from .warm_health import assert_running
    root = Path(root);assert_running(root, 'B');assert_running(root, 'R')
    quality = read(root/'replay_bank/B/BANK_QUALITY.json')
    reasons = []
    threshold = v['start_admitted_min']
    assert threshold == 8, 'MAIN08:16 registered start threshold'
    if quality['admitted'] < threshold:reasons.append(f'B admitted bank <{threshold}')
    # MAIN08:35: training consumes the qualified bank, not a GPU query.
    # config() verifies every file against the full Linux release receipt.
    cpu = read(Path(v['source_root'])/'LS_CPU_RESULT.json')
    assert cpu['pass_'] and cpu['sim_s'] == cpu['skips'] == 0
    transfer = read(root/'B_REPLAY_PHASE_TRANSFER_0706.json')
    assert transfer['funding_complete'] and transfer['budget'] == v['budget']
    assert v['budget']['total'] == 14500000
    assert sum(v['budget']['phases'].values()) <= v['budget']['total']
    return dict(unix=time.time(), ready=not reasons, reasons=reasons,
                admitted=quality['admitted'], cooperative=quality['cooperative'],
                start_admitted_min=threshold, replay_release_qualified=True,
                budget_reconciled=True, health_ok=True, gpu_rollout_required=False)


def collection_options(v):
    """Use niche alternation only after its independent collector activation."""
    active = (Path(v['collection_root'])/'LS_NULLFIX_CONFIG.json').exists()
    config = dict(iterations=2, screen_K=2)
    if active:config['niches'] = 1
    return dict(slots=4, fanout_factor=2, owned_pending_only=True,
                niche_alternation=active, search_config=config,
                demand_phase_end_s=v['demand_phase_end_s'])


def binding(root, v, arm):
    from .resource_config import clocks
    b = dict(read(root/'EXACT_OPERATIONAL_BINDING.json'))
    b.update(read(root/'WARM_B_RUNTIME_BINDING_V4.json')['operational'])
    b = clocks(root, b)
    b.update(budget=v['budget'], gpu_deadline=v['gpu_deadline'], b_replay=True,
        replay_beta=1., replay_sources={'B': [v['collection_root']], 'R': []},
        awr=read(root/'AWR_BINDING.json')['config'], resume_update=3, stop_after_update=24,
        runtime_receipt_suffix='_REPLAY0735', workers=12,
        quota_aware_admission=False, parallel_group_queries=False,
        bounded_fanout=False, r_concurrent_lag1_quota=True,
        demand_phase_end_s=v['demand_phase_end_s'], precharged_unattempted=[],
        sumo_verify=dict(root=str(root/'replay_bank'/arm/'sumo_verification'),
            run_root=str(root), phase='sumo_verification', confirm_phase='bank_verification_ext',
            k=8, workers=2, timeout_s=3600, guide_beta=1.))
    return b


def job_namespace(ident, arm):
    # All closed warm_u004 attempts remain permanently closed.
    return PREFIX+'_'+ident


def run_arm(root, arm):
    import torch
    from . import warm_runner as wr
    from .warm_protocol import batch_worlds, SEED
    from .warm_health import assert_running
    root = Path(root);v = config(root);accounting(v)
    torch.set_num_threads(1)
    b = binding(root, v, arm)
    assert sha(root/f'{arm}_checkpoint_003.pt') == v['checkpoints'][arm]
    # The checkpoint loader restores policy, Adam and torch Generator exactly.
    # Old diagnostic counters are irrelevant: replay never executes a plan.
    import numpy as np
    b['resume_progress'] = dict(running=[], beta=read(root/f'{arm}_UPDATE_003.json')['trust'],
        learned=read(root/'B_SEARCH_LEARNING_003.json') if arm == 'B' else None,
        exact_rng_state=np.random.default_rng(SEED).bit_generator.state, exact_count=12)
    wr.verify = lambda r, a: b if a == arm and Path(r) == root else (_ for _ in ()).throw(ValueError('arm/root'))
    wr.budget = lambda: copy.deepcopy(v['budget'])
    original_wave = wr.wave
    last_start = [None]

    def wave(r, cfg, policies, ws, role, ident, phase, deterministic=True, extras=None):
        if role == 'train':
            # Lag-1: R's episodes never wait on searches; at most one update behind B.
            u = int((extras or {})['update'])
            if arm == 'R':
                while not (root/f'B_UPDATE_{u:03}.json').exists():
                    assert_running(root, arm)
                    if time.time() > v['gpu_deadline']:raise ValueError('registered deadline')
                    time.sleep(10)
            if last_start[0] is not None:
                while time.time() < last_start[0]+v['pace_s']:
                    assert_running(root, arm);time.sleep(10)
            last_start[0] = time.time()
            extra = dict(extras or {}, b_replay=True, collector_policy_update=u-1,
                collect=collection_options(v))
            # Capture from the PPO episodes themselves: no extra collection episodes
            # or train-phase spending. All actions remain own-policy/on-policy.
            cfg = dict(cfg, job_id_overrides={})
            return original_wave(r, cfg, policies, ws, role, job_namespace(ident, arm),
                                 phase, deterministic, extra)
        return original_wave(r, cfg, policies, ws, role, job_namespace(ident, arm),
                             phase, deterministic, extras)
    wr.wave = wave
    wr.run_arm(root, arm)


def bridge_once(root, v):
    """Reuse the ONE live LS broker. Never run a second broker on its GPU slots."""
    from .warm_spool import immutable
    root = Path(root);c = Path(v['collection_root']);count = 0
    (c/'requests').mkdir(exist_ok=True)
    (root/'centre_responses').mkdir(exist_ok=True)
    for arm in ('B', 'R'):
        for p in sorted((root/'replay_bank'/arm/'snapshots').glob('*.json')):
            key = p.stem;folder = 'requests' if arm == 'B' else 'centre_requests'
            request = root/folder/p.name
            if not request.exists():continue  # snapshot precedes request publication
            r = read(request)
            assert digest(r) == read(p)['record_sha256']
            # mode='centre' is already part of R's immutable request/digest.
            if arm == 'R':assert r['mode'] == 'centre'
            immutable(c/'requests'/p.name, r)
            if arm == 'R' and (root/'responses'/p.name).exists():
                immutable(root/'centre_responses'/p.name, read(root/'responses'/p.name))
            count += 1
    return count


def grade(root, arm):
    """Async current-policy bank labels; per-list faults alerted, never replayed."""
    import torch
    from . import replay_bank as rb, sumo_verify as sv
    from .warm_health import assert_running
    root = Path(root);v = config(root);accounting(v);torch.set_num_threads(1)
    bank = rb.bank_root(root, arm);bank.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (bank/'LIVE_GRADER.claim').open('a');fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg = binding(root, v, arm);bounds = cfg['accel_bounds']
    decided = {p.stem for n in ('decisions', 'skipped', 'live_failures') for p in (bank/n).glob('*.json')}
    runner = sv.SubprocessBranchRunner(bank/'sumo_verification/branches', workers=2, timeout_s=3600)
    charge = sv.accel_charge(str(root), 'sumo_verification')
    def runner_for(work):return sv.SubprocessBranchRunner(Path(work)/'branches', workers=2, timeout_s=3600)
    while time.time() < v['gpu_deadline']:
        assert_running(root, arm)
        previous = set(decided)
        try:
            rb.verify_pending(root, arm, bounds, runner, charge, 2, decided, limit=1)
            confirm_one(root, arm, cfg['replay_sources'].get(arm, ()), runner_for)
            rb.bank_quality(root, arm, sources=cfg['replay_sources'].get(arm, ()))
        except Exception as error:
            import traceback
            for key in decided-previous or {str(time.time_ns())}:
                write(bank/'live_failures'/(key+'.json'),
                      dict(unix=time.time(), error=repr(error), traceback=traceback.format_exc(),
                           physics_retry=False, alert=True))
            print('ALERT bank grading '+repr(error), flush=True)
        if (root/f'WARM_{arm}_PAUSED.json').exists() and read(root/f'WARM_{arm}_PAUSED.json')['update'] == 24:return
        time.sleep(10)


def confirm_one(root, arm, sources, runner_for):
    """Whole-decision precharge, durable one-use attempt, amended confirmation."""
    from . import replay_bank as rb, sumo_verify as sv, accel_storage
    from .bank_builder import quoted_jobs, precharged_charge
    bank = rb.bank_root(root, arm)
    done = rb.confirmations(root, arm)
    links = {read(p)['verification_id']: read(p) for p in (bank/'decisions').glob('*.json')}
    for r in rb.bank_decisions(root, arm, sources=sources):
        vid = r['verification_id'];attempt = bank/'live_confirmations'/(vid+'.json')
        if r['status'] != 'accepted' or not (r['grade_veh_s'] or 0) > 0 or vid in done or attempt.exists():continue
        snap = rb._decision_snapshot(root, arm, r, links)
        jobs = quoted_jobs(snap, r['chosen_plan'], sv.build_branches, rb.solo_candidate)
        accel_storage.reserve(root, jobs, 'bank_verification_ext')
        write(attempt, dict(unix=time.time(), jobs=jobs, physics_retry=False))
        entries = {e['id']: e for e in accel_storage.ledger(root)}
        original = rb._confirm_charge
        try:
            rb._confirm_charge = lambda rr, phase, kind: precharged_charge(entries, kind)
            rb.confirm(root, arm, {k: x for k, x in r.items() if not k.startswith('_')},
                       snap, runner_for, str(root), 'bank_verification_ext', workers=2,
                       version=r['_policy_version'])
        finally:rb._confirm_charge = original
        return


def spawn(root, v, component, arm=None):
    control = Path(v['control_root']);label = component+('_'+arm if arm else '')
    target = control/(label+'_DISPATCH.json')
    if target.exists():
        d = read(target)
        assert alive(d) or (control/(label+'_COMPLETE.json')).exists(), 'owned child exited; inspect '+label
        return d
    command = [sys.executable, '-B', '-u', '-m', 'tools.coopmwm.replay_start', str(root), component]
    if arm:command += ['--arm', arm]
    with (control/(label+'.log')).open('x') as log:
        p = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True)
    d = dict(owner(p.pid), component=component, arm=arm, argv=command)
    write(target, d);return d


def retire_endpoint(root, arm, expected):
    """Preserve a superseded sub24 certificate without overwriting its bytes."""
    if expected is None:return
    p=Path(root)/f'WARM_{arm}_PAUSED.json'
    dest=p.with_name(f'WARM_{arm}_PAUSED_BEFORE_REPLAY0735.json')
    if dest.exists():
        assert sha(dest)==expected['sha256'] and not p.exists();return
    assert sha(p)==expected['sha256'] and read(p)['update']<24
    p.rename(dest)


def handover_ls(root, v):
    """Retire frozen-u3 admissions at a no-grade boundary; drain saved labels."""
    c=Path(v['collection_root']);dest=c/'SUPERVISOR_DISPATCH_REPLAYDRAIN0735.json'
    if dest.exists():
        assert alive(read(dest)), 'adopt/diagnose existing saved-label drain owner'
        return True
    old=read(c/'SUPERVISOR_DISPATCH_NICHENULL2.json')
    assert alive(old), 'qualified collection owner absent'
    p=Path('/proc',str(old['pid']))
    argv=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
    assert 'ls_collection' in argv and str(c) in argv
    os.kill(old['pid'],signal.SIGSTOP)
    try:
        for _ in range(100):
            if (p/'stat').read_text().split()[2]=='T':break
            time.sleep(.01)
        else:raise RuntimeError('metadata freeze not acknowledged')
        children=(p/'task'/str(old['pid'])/'children').read_text().split()
        active=[k for k in children if 'multiprocessing.resource_tracker' not in
                (Path('/proc',k)/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()]
        labelled={read(q)['verification']['verification_id'] for q in (c/'labels').glob('*.json')}
        claims=c/'label_attempts/SUMOENV1/verification/claims'
        open_claims=[q.stem for q in claims.glob('*.json')
                     if read(q).get('pid')==old['pid'] and q.stem not in labelled]
        if active or open_claims:
            atomic(Path(v['control_root'])/'COLLECTOR_BOUNDARY_WAIT.json',
                   dict(unix=time.time(),active_children=active,open_claims=open_claims))
            return False
        os.kill(old['pid'],signal.SIGTERM)
    finally:
        if alive(old):os.kill(old['pid'],signal.SIGCONT)
    for _ in range(100):
        if not alive(old):break
        time.sleep(.1)
    else:raise RuntimeError('metadata owner did not exit')
    cfg=dict(source_root=v['source_root'],receipt_suffix='_REPLAYDRAIN0735',drain_only=True,
             budget=v['budget'],operational_spend_cap=v['budget']['total'],
             amendment_sha256=v['amendment_sha256'])
    write(c/'LS_REPLAY_DRAIN_CONFIG.json',cfg)
    with (c/'SUPERVISOR_REPLAYDRAIN0735.log').open('x') as log:
        code=('from tools.coopmwm.nullfix_geometry import install;install('+repr(v['source_root']+'/FROZEN_GEOMETRY_SOURCE.py')+','+repr(v['source_root']+'/FROZEN_NETWORK.net.xml.gz')+');'
              'from tools.coopmwm.ls_collection import supervise;supervise('+repr(str(c))+')')
        proc=subprocess.Popen([sys.executable,'-B','-u','-c',code],stdin=subprocess.DEVNULL,
                              stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(dest,dict(owner(proc.pid),old=old,source_root=v['source_root'],drain_only=True))
    return True


def watch(root):
    import fcntl
    root = Path(root);v = config(root);accounting(v);control = Path(v['control_root'])
    lock = (control/'WATCH.claim').open('a');fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    write(control/'WATCH_OWNER.json', owner())
    while time.time() < v['gpu_deadline']:
        try:gate = prereqs(root, v)
        except Exception as error:
            from .warm_transport import SSHConnectionError
            if not isinstance(error, SSHConnectionError):raise
            gate = dict(unix=time.time(), ready=False, reasons=['bounded transport retry exhausted'],
                        error=repr(error), retry_after_s=120)
        atomic(control/'STATUS.json', gate)
        if gate['ready']:break
        time.sleep(120)
    else:raise ValueError('registered deadline before start')
    # GPU rolling release and collector activation proceed independently. Transfer
    # frozen-u3 admissions later; neither operation may delay own-policy PPO.
    # The original builder also sees future own-bank decisions. Transfer its
    # ownership at a graceful confirmation boundary before enabling the new one.
    old = Path(v['collection_root'])/'bank_builder_0426'
    dispatch = read(old/'DISPATCH.json')
    if alive(dispatch):
        p = old/'STOP.json'
        if not p.exists():write(p, dict(unix=time.time(), reason='registered B-REPLAY confirmation handover',
                                       amendment_sha256=v['amendment_sha256']))
        while alive(dispatch):
            if time.time() > v['gpu_deadline']:raise ValueError('builder boundary deadline')
            time.sleep(5)
    for arm in ('B', 'R'):
        assert sha(root/f'{arm}_checkpoint_003.pt') == v['checkpoints'][arm]
        assert not (root/f'{arm}_checkpoint_004.pt').exists(), 'adopt optimizer; do not duplicate'
        from .warm_health import assert_running
        assert_running(root, arm)
        # Probe original arm lock; it must be released by all superseded runs.
        with (root/f'WARM_{arm}.claim').open('a') as f:
            fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)
        retire_endpoint(root, arm, v['old_endpoints'][arm])
    children = [spawn(root, v, 'grade', a) for a in ('B', 'R')]
    children += [spawn(root, v, 'arm', a) for a in ('B', 'R')]
    write(root/'B_REPLAY_STARTED.json', dict(unix=time.time(), gate=gate, children=children,
        checkpoints=v['checkpoints'], first_update=4, final_update=24,
        mode='own-policy PPO + confirmed positive bank AWR; collection never blocks episodes'))
    while time.time() < v['gpu_deadline']:
        bridge_once(root, v)
        if (Path(v['collection_root'])/'LS_NULLFIX_CONFIG.json').exists():
            handover_ls(root, v)
        # Hash-sealed interim episode jobs are a separate, fixed-checkpoint task.
        if (root/'B_UPDATE_008.json').exists():spawn(root, v, 'interim')
        updates = {a: max([3]+[int(p.stem.rsplit('_', 1)[1]) for p in root.glob(a+'_UPDATE_[0-9][0-9][0-9].json')]) for a in ('B', 'R')}
        dead = [d for d in children if not alive(d) and not (control/(d['component']+'_'+d['arm']+'_COMPLETE.json')).exists()]
        atomic(control/'STATUS.json', dict(unix=time.time(), started=True, updates=updates,
            bridge_snapshots=sum(len(list((root/'replay_bank'/a/'snapshots').glob('*.json'))) for a in ('B', 'R')),
            dead_children=dead, alert=bool(dead), outcomes_opened=False))
        if dead:raise RuntimeError('owned replay child exited; preserve jobs and diagnose')
        if all(u >= 24 for u in updates.values()):
            write(control/'COMPLETE.json', dict(unix=time.time(), updates=updates));return
        time.sleep(5)


def interim_specs(panel, arms):
    return [dict(id=f'interim008_{a}_{w["seed"]}', world=w, arm=a, kind='episode',
                 checkpoint_update=8, sim_s=1200, phase='interim_u8_test') for a in arms for w in panel]


def interim(root):
    """Fixed u8 episode-only precompute. Publish identities/hashes, no outcomes."""
    from . import final_runtime as fr
    from .warm_protocol import worlds
    from .accel_storage import reserve, ledger
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    root = Path(root);v = config(root);accounting(v)
    arms = ['A', 'B']+(['R'] if (root/'R_UPDATE_008.json').exists() else [])
    items = interim_specs(worlds('final_test'), arms)
    assert sum(x['sim_s'] for x in items) <= v['interim_quote_sim_s']
    existing = {x['id'] for x in ledger(root)}
    assert not existing.intersection(x['id'] for x in items), 'never retry interim physics'
    write(Path(v['control_root'])/'INTERIM_U8_MANIFEST.json', dict(unix=time.time(), arms=arms,
        specs=items, sim_s=sum(x['sim_s'] for x in items), outcomes_closed=True,
        rule='fixed u8; R only if reached at B-u8 trigger; no branch tests; MAPPO/native reused'))
    b = binding(root, v, 'B')
    # execute_one returns public hash metadata only. Never inspect result values.
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
        for s in items:
            reserve(root, [s], 'interim_u8_test')
            meta = pool.submit(fr.execute_one, root, b, s).result()
            assert meta['outcomes_closed']


def main():
    p = argparse.ArgumentParser();p.add_argument('root', type=Path)
    p.add_argument('component', choices=('watch', 'arm', 'grade', 'interim'));p.add_argument('--arm', choices=('B', 'R'))
    a = p.parse_args();v = config(a.root)
    from .nullfix_geometry import install
    s = Path(v['source_root']);install(str(s/'FROZEN_GEOMETRY_SOURCE.py'), str(s/'FROZEN_NETWORK.net.xml.gz'))
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    os.nice(max(0, 19-os.getpriority(os.PRIO_PROCESS, 0)))
    try:
        if a.component == 'arm':run_arm(a.root, a.arm)
        elif a.component == 'grade':grade(a.root, a.arm)
        elif a.component == 'interim':interim(a.root)
        else:watch(a.root)
    except BaseException as error:
        import traceback
        write(Path(v['control_root'])/(a.component+('_'+a.arm if a.arm else '')+'_FAILED.json'),
              dict(unix=time.time(), error=repr(error), traceback=traceback.format_exc(), physics_retry=False))
        raise
    else:
        write(Path(v['control_root'])/(a.component+('_'+a.arm if a.arm else '')+'_COMPLETE.json'), dict(unix=time.time()))


if __name__ == '__main__':main()
