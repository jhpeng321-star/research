"""MAIN18:38 collection handover. Every operation adopts durable receipts first."""
import argparse
import inspect
import json
import time
from pathlib import Path
from . import ops
from .storage import read, write
from .protocol import REGISTRY, checked, sha

ROOT = '/home/ubuntu/coopmwm20261102_warm1'


def pause_remote(root, amendment):
    import os, signal, hashlib, sys
    from pathlib import Path
    sys.path.insert(0, '/home/ubuntu/coopmwm20261102_verifybeta13/pytools')
    from tools.coopmwm.warm_spool import immutable
    root = Path(root)
    done = root/'LS_COLLECT_PAUSED.json'
    if done.exists():
        return json.loads(done.read_text())
    assert not (root/'B_checkpoint_004.pt').exists()
    assert not (root/'R_checkpoint_004.pt').exists()
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    def identity(pid):
        p = Path('/proc', str(pid))
        try:
            st = (p/'stat').read_text().split()
            return dict(pid=int(pid), start_ticks=st[21], state=st[2],
                        argv=(p/'cmdline').read_bytes().replace(b'\0', b' ').decode(),
                        children=[int(x) for x in (p/'task'/str(pid)/'children').read_text().split()])
        except FileNotFoundError:
            return None
    def send(d, sig):
        cur = identity(d['pid'])
        if cur is None or cur['state'] == 'Z':
            return
        assert cur['start_ticks'] == d['start_ticks']
        assert cur['argv'] == d['argv'], 'PID identity changed'
        os.kill(d['pid'], sig)
    checkpoints = {a: hashlib.sha256((root/f'{a}_checkpoint_003.pt').read_bytes()).hexdigest() for a in ('B','R')}
    owners = []
    for arm in ('B','R'):
        d = json.loads((root/f'DNS_{arm}_DISPATCH_1651.json').read_text())
        assert d['boot_id'] == boot
        cur = identity(d['pid'])
        if cur and cur['state'] != 'Z':
            assert cur['start_ticks'] == d['start_ticks']
            assert cur['argv'].split() == ['/home/ubuntu/miniconda3/bin/python', '-B', '-u', '-m',
                'tools.coopmwm.incident_recovery', str(root), '--component', arm]
            owners.append(cur)
    # Stop parents first so no optimizer/update or additional pool task can start.
    for d in owners:
        send(d, signal.SIGSTOP)
    tree = []
    def descend(d):
        for pid in d['children']:
            child = identity(pid)
            if child is None or child['state'] == 'Z':
                continue
            assert ('multiprocessing' in child['argv'] or
                    ('sumo ' in child['argv'] and str(root/'jobs/warmdns1651_u004_B_') in child['argv']))
            send(child, signal.SIGSTOP)
            tree.append(child)
            descend(child)
    for d in owners:
        descend(d)
    plan = dict(unix=time.time(), boot_id=boot, owners=owners, descendants=tree,
                checkpoints=checkpoints, amendment_sha256=amendment, new_sim_s=0)
    immutable(root/'LS_COLLECT_PAUSE_SIGNAL_PLAN.json', plan)
    # Only this receipt-verified tree is terminated. No GPU/broker/grader signal.
    for d in reversed(tree):
        send(d, signal.SIGTERM)
        send(d, signal.SIGCONT)
    for d in owners:
        send(d, signal.SIGTERM)
        send(d, signal.SIGCONT)
    deadline = time.monotonic()+10
    while time.monotonic() < deadline and any(identity(d['pid']) and identity(d['pid'])['state'] != 'Z' for d in tree+owners):
        time.sleep(.1)
    for d in tree+owners:
        send(d, signal.SIGKILL)
    jobs = []
    for p in sorted((root/'jobs').glob('warmdns1651_u004_B_*')):
        assert not (p/'RESULT.json').exists(), 'inspect newly completed work before closing'
        if (p/'CLAIM.json').exists():
            immutable(p/'LS_COLLECT_CLOSED.json', dict(reason='MAIN18:38 stalled wave closed; never replay',
                unix=time.time(), amendment_sha256=amendment, charges_retained=True))
            if not (p/'CLOSE.json').exists():
                immutable(p/'CLOSE.json', dict(reason='MAIN18:38 collection pause', actual_sim_s=None,
                    charges_retained=True, unix=time.time()))
            jobs.append(p.name)
    assert checkpoints == {a: hashlib.sha256((root/f'{a}_checkpoint_003.pt').read_bytes()).hexdigest() for a in ('B','R')}
    assert not (root/'B_checkpoint_004.pt').exists() and not (root/'R_checkpoint_004.pt').exists()
    result = dict(plan, closed_jobs=jobs, no_u4_update=True, gpu_work_preserved=True,
                  resume_requires_MAIN=True, completed_unix=time.time())
    immutable(done, result)
    return result


def pause_latest_remote(root, amendment):
    """Close the superseded FAST1844 wave, never the original pause receipt."""
    import os, signal, hashlib, sys
    from pathlib import Path
    sys.path.insert(0, '/home/ubuntu/coopmwm20261102_fast1844cpu1/pytools')
    from tools.coopmwm.warm_spool import immutable
    root = Path(root)
    done = root/'LS_COLLECT_PAUSED_1912.json'
    if done.exists():
        return json.loads(done.read_text())
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    def identity(pid):
        p = Path('/proc', str(pid))
        try:
            st = (p/'stat').read_text().split()
            return dict(pid=int(pid), start_ticks=st[21], state=st[2],
                        argv=(p/'cmdline').read_bytes().replace(b'\0', b' ').decode(),
                        children=[int(x) for x in (p/'task'/str(pid)/'children').read_text().split()])
        except FileNotFoundError:
            return None
    def send(d, sig):
        cur = identity(d['pid'])
        if cur is None or cur['state'] == 'Z':
            return
        assert cur['start_ticks'] == d['start_ticks'] and cur['argv'] == d['argv'], 'PID identity changed'
        os.kill(d['pid'], sig)
    for a in ('B', 'R'):
        assert not (root/f'{a}_checkpoint_004.pt').exists(), 'u4 exists: do not overwrite'
    checkpoints = {a: hashlib.sha256((root/f'{a}_checkpoint_003.pt').read_bytes()).hexdigest() for a in ('B','R')}
    expected = json.loads((root/'LS_COLLECT_PAUSED.json').read_text())['checkpoints']
    assert checkpoints == expected
    owners = []
    for a in ('B','R'):
        d = json.loads((root/f'FAST_{a}_DISPATCH_1844.json').read_text())
        assert d['boot_id'] == boot
        cur = identity(d['pid'])
        if cur and cur['state'] != 'Z':
            assert cur['start_ticks'] == d['start_ticks']
            assert cur['argv'].split() == ['/home/ubuntu/miniconda3/bin/python','-B','-u','-m',
                'tools.coopmwm.incident_recovery',str(root),'--component',a]
            owners.append(cur)
    intent=root/'LS_COLLECT_PAUSE_1912_INTENT.json'
    if intent.exists():
        previous=json.loads(intent.read_text())
        assert previous['checkpoints']==checkpoints and previous['amendment_sha256']==amendment
        assert [(d['pid'],d['start_ticks'],d['argv']) for d in previous['owners']]==[(d['pid'],d['start_ticks'],d['argv']) for d in owners]
    else:
        immutable(intent, dict(unix=time.time(), owners=owners,
            boot_id=boot, checkpoints=checkpoints, amendment_sha256=amendment))
    for d in owners:
        send(d, signal.SIGSTOP)
    tree = []
    def descend(d):
        for pid in identity(d['pid'])['children']:
            child = identity(pid)
            if child is None or child['state'] == 'Z':
                continue
            sumo_owned=('sumo ' in child['argv'] and any(prefix in child['argv'] for prefix in (
                str(root/'jobs')+'/', '/dev/shm/coopmwm20261102_warm1/warm_u004_R_',
                '/dev/shm/coopmwm20261102_warm1/warmfast1844_u004_B_')))
            assert 'multiprocessing' in child['argv'] or sumo_owned, child
            send(child, signal.SIGSTOP)
            tree.append(child)
            descend(child)
    for d in owners:
        descend(d)
    immutable(root/'LS_COLLECT_PAUSE_1912_SIGNAL_PLAN.json',dict(unix=time.time(),owners=owners,descendants=tree,
        amendment_sha256=amendment,boot_id=boot))
    for d in list(reversed(tree))+owners:
        send(d, signal.SIGTERM)
        send(d, signal.SIGCONT)
    deadline = time.monotonic()+10
    while time.monotonic()<deadline and any(identity(d['pid']) and identity(d['pid'])['state']!='Z' for d in tree+owners):
        time.sleep(.1)
    for d in tree+owners:
        send(d, signal.SIGKILL)
    jobs=[]
    for pattern in ('warmfast1844_u004_B_*','warm_u004_R_*'):
        for p in sorted((root/'jobs').glob(pattern)):
            if not (p/'CLAIM.json').exists():
                continue
            result_exists=(p/'RESULT.json').exists() or (p/'RESULT.json.gz').exists()
            immutable(p/'LS_COLLECT_CLOSED_1912.json',dict(unix=time.time(),charges_retained=True,
                result_preserved=result_exists,never_replay=True,amendment_sha256=amendment))
            if not (p/'CLOSE.json').exists():
                immutable(p/'CLOSE.json',dict(unix=time.time(),actual_sim_s=None,charges_retained=True,
                    reason='MAIN19:12 forbidden plain-search u4 wave closed; never replay'))
            jobs.append(dict(job=p.name,result_preserved=result_exists))
    assert checkpoints == {a:hashlib.sha256((root/f'{a}_checkpoint_003.pt').read_bytes()).hexdigest() for a in ('B','R')}
    assert not (root/'B_checkpoint_004.pt').exists() and not (root/'R_checkpoint_004.pt').exists()
    v=dict(unix=time.time(),owners=owners,descendants=tree,closed_jobs=jobs,checkpoints=checkpoints,
           no_u4_update=True,charges_retained=True,new_sim_s=0,resume_requires_MAIN=True,
           gpu_work_preserved=True,amendment_sha256=amendment)
    immutable(done,v)
    return v


def pause_latest():
    target=ops.EVIDENCE/'LS_COLLECT_PAUSE_REGISTRATION_1912.json'
    if not target.exists():
        line=('\nS-COOPMWM MAIN19:12 LATEST-ORDER PAUSE: forbidden FAST1844 plain-search u4 '
              'B/R wave closed by receipt/boot/start/argv-verified owned PIDs only. No u4 update; '
              'durable B3/R3 unchanged; all attempted/completed jobs, charges, responses and pools retained, '
              'closed jobs never replayed. LS-COLLECT only, resume requires MAIN addendum. '
              'Latest cap11900000; restricted interim_u8_test900000 not available for collection. No new SUMO.\n')
        write(target,dict(line=line,authority='MAIN19:12',new_sim_s=0,unix=time.time()))
        with checked(REGISTRY).open('a',encoding='utf-8') as f:
            f.write(line+'amendmentSHA'+sha(target)+'\n')
    script=inspect.getsource(pause_latest_remote)
    out=json.loads(ops.ssh.remote('cpuQ','import json,time\n'+script+
        f'\nprint(json.dumps(pause_latest_remote({ROOT!r},{sha(target)!r})))',timeout=45))
    dest=ops.EVIDENCE/'LS_COLLECT_PAUSED_1912.json'
    if not dest.exists():write(dest,out)
    print(json.dumps(dict(paused=True,no_u4_update=out['no_u4_update'],checkpoints=out['checkpoints'],
        closed_jobs=out['closed_jobs'],new_sim_s=0)))


def audit():
    out=ops.ssh.remote('cpuQ',f'''import sys,json,pathlib,time
sys.path.insert(0,'/home/ubuntu/coopmwm20261102_fast1844cpu1/pytools')
from tools.coopmwm.accel_storage import ledger,charged
r=pathlib.Path({ROOT!r});rows=ledger(r);p=json.loads((r/'LS_COLLECT_PAUSED_1912.json').read_text());ids={{v['job'] for v in p['closed_jobs']}};ph={{}}
for c in rows:ph[c['phase']]=ph.get(c['phase'],0)+c['sim_s']
owners=[]
for d in p['owners']+p['descendants']:
 q=pathlib.Path('/proc',str(d['pid']))
 if q.exists():
  s=(q/'stat').read_text().split();owners.append(dict(pid=d['pid'],state=s[2],same_identity=s[21]==d['start_ticks']))
print(json.dumps(dict(unix=time.time(),charged_sim_s=charged(r),phase_charges=ph,closed_wave_charge=sum(c['sim_s'] for c in rows if c['id'] in ids),closed_jobs=len(ids),remaining_processes=owners,u4_absent=all(not (r/(a+'_checkpoint_004.pt')).exists() for a in ('B','R')))))''',timeout=40)
    v=json.loads(out);write(ops.EVIDENCE/f'LS_COLLECT_PAUSE_AUDIT_{int(v["unix"])}.json',v);print(out)


def pause():
    target = ops.EVIDENCE/'LS_COLLECT_PAUSE_REGISTRATION.json'
    if target.exists():
        reg = read(target)
    else:
        line = ('\nS-COOPMWM MAIN2026-11-02 18:38 LS-COLLECT PAUSE REGISTERED: '
                'B/R durable u3; no u4 PPO update. Close receipt-verified stalled warmdns1651_u004_B '
                'wave, never replay jobs; retain every charge, accepted response and full pool. '
                'A48 done. GPU claims continue; no duplicate inference. No new SUMO until '
                'LS_COLLECT_REGISTRATION quote/SHA. Resume only on MAIN addendum.\n')
        with checked(REGISTRY).open('a', encoding='utf-8') as f:
            f.write(line)
        write(target, dict(unix=time.time(), authority='MAIN18:38', line=line, new_sim_s=0))
        reg = read(target)
    amendment = sha(target)
    script = inspect.getsource(pause_remote)
    out = json.loads(ops.ssh.remote('cpuQ',
        'import json,time\n'+script+f'\nprint(json.dumps(pause_remote({ROOT!r},{amendment!r})))', timeout=40))
    dest = ops.EVIDENCE/'LS_COLLECT_PAUSED.json'
    if not dest.exists():
        write(dest, out)
    print(json.dumps(dict(paused=True, checkpoints=out['checkpoints'], closed_jobs=out['closed_jobs'],
                         no_u4_update=out['no_u4_update'], new_sim_s=0)))


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('action', choices=['pause','pause_latest','audit'])
    a = p.parse_args()
    globals()[a.action]()
