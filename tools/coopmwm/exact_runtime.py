"""Detached exact-grade integration: no estimated plan grades, no physics retries."""
import argparse
import json
import os
from pathlib import Path
import time
from . import exact_grade as xg
from .accel_storage import read,write
from .protocol import digest,sha
from .warm_protocol import SEED
from .warm_spool import immutable


def queue(root):
    return xg.GradeQueue(Path(root)/'exact_grades')


def enqueue(root,b,ep,u,arm,beta):
    ckpt=Path(root)/f'{arm}_checkpoint_{u-1:03}.pt'
    meta=dict(run_root=str(root),
        result_path=str(Path(root)/'jobs'/ep['id']/'RESULT.json'),result_sha256=digest(ep),
        update=u,arm=arm,training_seed=SEED,checkpoint_path=str(ckpt),checkpoint_sha256=sha(ckpt),
        checkpoint_update=u-1,noise_seed=xg.train_noise_seed(ep['world']['seed'],u,SEED),
        runtime=b['runtime'],sumo_checkpoint=b['checkpoint'],beta=beta,
        warm_history=b.get('warm_history',False),accel_bounds=b['accel_bounds'])
    if ep.get('sumo_verification_root'):
        from .verification_grades import enqueue_episode
        return enqueue_episode(queue(root),ep,meta)
    return queue(root).enqueue_episode(ep,meta)


def assert_live(root):
    p=Path(root)/'VERIFIED_GRADER_DISPATCH.json'
    if (Path(root)/'DNS_GRADER_DISPATCH_1651.json').exists():p=Path(root)/'DNS_GRADER_DISPATCH_1651.json'
    if not p.exists():p=Path(root)/'MATCH_GRADER_DISPATCH.json'
    if not p.exists():p=Path(root)/'INCIDENT_GRADER_DISPATCH.json'
    if not p.exists():p=Path(root)/'EXACT_GRADER_DISPATCH_V2.json'
    d=read(p if p.exists() else Path(root)/'EXACT_GRADER_DISPATCH.json')
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
    stat=Path('/proc')/str(d['pid'])/'stat'
    assert stat.exists() and stat.read_text().split()[21]==d['start_ticks'],'exact grader owner not alive'
    assert stat.read_text().split()[2]!='Z','exact grader owner is zombie'
    for name in ('GRADER_STOP.json','GRADER_BLOCKED.json'):
        assert not (Path(root)/'exact_grades'/name).exists(),name


def evidence(root,arm,beta):
    assert_live(root)
    q=queue(root)
    # First real successful grade is the one-plan smoke, never duplicated. No
    # trust/imitation/library can consume any grade before its CRN/prefix gates.
    receipts=q.receipts() if (q.root/'SMOKE_PASSED.json').exists() else []
    T=q.frozen_temperature(arm,SEED) if receipts else xg.temperature([],arm,SEED)
    W=xg.imitation_weights(receipts,arm,T,SEED)
    pairs=[p for e in W['weights'].values() for p in xg.imitation_pairs(read(Path(e['result_path'])),e)]
    return receipts,W,pairs,xg.trust_evidence(receipts,beta,arm=arm,training_seed=SEED)


def supervise(root,workers=8):
    """Monitor completed TRAIN receipts and grade in owned isolated subprocesses."""
    import subprocess
    import fcntl
    import signal
    import torch
    torch.set_num_threads(1)
    root=Path(root);q=queue(root);running={};seen=set()
    lock=(q.root/'SUPERVISOR.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    immutable(q.root/'GRADER_CONFIG.json',dict(run_root=str(root),phase='exact_plan_grading',path_map={}))
    while True:
        from .warm_health import assert_running
        assert_running(root,'B')
        for gid,p in list(running.items()):
            if p.poll() is not None:running.pop(gid)
        # Incremental ledger scan; only explicit closed B/R TRAIN result paths.
        from .accel_storage import ledger
        for c in ledger(root):
            ident=c['id']
            if ident in seen or not ident.startswith(('warm_u','warmdns1651_u')):continue
            bits=ident.split('_')
            if len(bits)!=4 or bits[2] not in ('B','R'):continue
            p=root/'jobs'/ident/'RESULT.json'
            if not p.exists():continue
            ep=read(p);u=int(bits[1][1:]);arm=bits[2]
            b=read(root/'EXACT_OPERATIONAL_BINDING.json')
            from .resource_config import clocks
            b=clocks(root,b)
            previous=root/f'{arm}_UPDATE_{u-1:03}.json'
            beta=read(previous)['trust'] if previous.exists() else 1.
            enqueue(root,b,ep,u,arm,beta);seen.add(ident)
        q.recover()
        receipts=q.receipts()
        for r in receipts:
            if r['status'] in ('failed','stopped') or (r['status']=='graded' and not r['noise'].get('crn_verified_t0')):
                immutable(q.root/'GRADER_STOP.json',dict(grade_id=r['grade_id'],status=r['status'],reason=r.get('reason'),no_imputation=True))
                target=root/('WARM_GUIDED_STOP_V5.json' if (root/'DNS_GUIDED_RELEASE_1651.json').exists() else
                    'WARM_GUIDED_STOP_V4.json' if (root/'MATCH_GUIDED_RELEASE.json').exists() else
                    'WARM_GUIDED_STOP_V3.json' if (root/'INCIDENT_GUIDED_RELEASE.json').exists() else
                    'WARM_GUIDED_STOP_V2.json' if (root/'MAIN0508_GUIDED_RELEASE.json').exists() else 'WARM_GUIDED_STOP.json')
                immutable(target,dict(source='exact-grade engineering fault',grade_id=r['grade_id']))
                return
        good=next((r for r in receipts if r['arm']=='B' and r['status']=='graded'),None)
        if good and not (q.root/'SMOKE_PASSED.json').exists():
            assert good['prefix']['pass_'] and good['noise']['crn_verified_t0'] and good['noise']['noise_mode_used']=='crn'
            immutable(q.root/'SMOKE_PASSED.json',dict(xg.summary(good),receipt_sha256=good['receipt_sha256'],one_real_grade_reused=True))
        if (q.root/'GRADER_BLOCKED.json').exists():return
        for gid in q.pending():
            if len(running)>=workers:break
            if gid in running:continue
            p=subprocess.Popen([os.sys.executable,'-B','-u','-m','tools.coopmwm.exact_grade','run-one','--root',str(q.root),'--grade-id',gid])
            running[gid]=p
            stat=Path(f'/proc/{p.pid}/stat').read_text().split()
            with (q.root/'WORKERS.jsonl').open('a') as f:f.write(json.dumps(dict(pid=p.pid,start_ticks=stat[21],grade_id=gid,unix=time.time()))+'\n')
        write(root/f'EXACT_GRADER_STATUS_{time.time_ns()}.json',dict(unix=time.time(),running=len(running),pending=len(q.pending()),receipts=len(receipts),smoke_passed=(q.root/'SMOKE_PASSED.json').exists(),workers=workers))
        time.sleep(10)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();supervise(a.root,a.workers)
