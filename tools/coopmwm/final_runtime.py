"""Outcome-blind frozen-policy precompute and fixed-endpoint final continuation.

Workers write private results; during training the coordinator publishes ONLY job
identities, hashes, resource/charge metadata and exact-prefix gate verdicts. It
does not aggregate or report outcome values before all fixed endpoints complete.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import multiprocessing
import os
from pathlib import Path
import shutil
import time

from .accel_storage import read,write,reserve,ledger,charged
from .protocol import sha,digest
from .warm_protocol import budget,worlds,install_accounting

TIMES=(300,500,700,900)
RAM=Path('/dev/shm/coopmwm20261102_final1')


def specs(arms,branches=False,panel=None):
    panel=worlds('final_test') if panel is None else panel
    return [dict(id=(f'fixedbranch_{a}_{w["seed"]}_t{t}' if branches else f'fixedfinal_{a}_{w["seed"]}'),
                 world=w,arm=a,kind='branch' if branches else 'episode',time_s=t,
                 sim_s=t+300 if branches else 1200,
                 phase='paired_branch' if branches else ('random_guide_test' if a=='R' else 'final_test'))
            for a in arms for w in panel for t in (TIMES if branches else (None,))]


def endpoints_ready(root,arms=('A','B','R')):
    for arm in arms:
        u=budget()['updates'][arm]
        p=root/f'WARM_{arm}_PAUSED.json'
        if not p.exists() or not (root/f'{arm}_checkpoint_{u:03}.pt').is_file():return False
        if read(p)['update']!=u:raise ValueError('fixed endpoint receipt mismatch')
    return True


def public_completion(job,path):
    """Hash only an explicit allowed-artifact list; no scenario traversal."""
    names=('RESULT.json','RESULT.json.gz','observable.jsonl.gz','CLAIM.json','CLOSE.json')
    assert (path/'RESULT.json').is_file() and (path/'CLOSE.json').is_file()
    files={n:dict(sha256=sha(path/n),bytes=(path/n).stat().st_size) for n in names if (path/n).is_file()}
    return dict(id=job['id'],arm=job['arm'],kind=job['kind'],world=job['world'],
                sim_s=job['sim_s'],phase=job['phase'],files=files,unix=time.time(),
                outcomes_closed=True)


def make_job(root,b,spec):
    import torch
    from .acceleration import Policy
    from .accel_learner import load
    a=spec['arm']
    if a=='native':
        from tools.coopnative.policy import Policy as NativePolicy
        weights=NativePolicy().state_dict()
    elif a=='mappo':weights=Policy().state_dict()
    else:
        u=budget()['updates'][a]
        policy,_,_,payload=load(root/f'{a}_checkpoint_{u:03}.pt')
        assert payload['update']==u
        weights=policy.state_dict()
    job=dict(id=spec['id'],path=str(root/'jobs'/spec['id']),world=spec['world'],role='final_test',
             arm=a,weights=weights,runtime=b['runtime'],checkpoint=b['checkpoint'],
             noise_seed=int(digest(spec)[:15],16),deterministic=True,guided=False,
             sim_s=spec['sim_s'],stop=spec['sim_s'],accel_bounds=b['accel_bounds'],
             gpu_deadline=b['gpu_deadline'])
    if a=='native':job.update(native_reference=True,authority=b['authority'])
    return job


def execute_one(root,b,spec):
    """Isolated worker: never return outcomes through the process boundary."""
    os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
    os.sched_setscheduler(0,os.SCHED_IDLE,os.sched_param(0))
    import torch
    from .accel_plant import episode
    from .branch_eval import takeover_controller,check_prefix
    torch.set_num_threads(1)
    torch.manual_seed(20261101)
    job=make_job(root,b,spec)
    if spec['kind'] in ('branch','reuse'):
        # This validator uses only the registered identical-prefix reference; no
        # treatment selection, result summarisation or outcome publication.
        reference=read(root/'jobs'/f'fixedfinal_mappo_{spec["world"]["seed"]}'/'RESULT.json')
        t=spec['time_s']
        job['branch']=dict(time_s=t,takeover_s=t,revert=[],
                          prefixes={str(s):reference['prefixes'][str(s)] for s in range(300,t+1)})
        with takeover_controller():result=episode(job)
        check_prefix(result,reference,t)
        if spec['kind']=='reuse':
            # Check the complete repeated MAPPO continuation, not just its prefix.
            for s in range(300,spec['sim_s']):
                if result['prefixes'][str(s)]!=reference['prefixes'][str(s)]:
                    raise ValueError('MAPPO branch-reference reuse equivalence failed; additional budget needs MAIN')
            if result['final']!=reference['prefixes'][str(spec['sim_s'])]:
                raise ValueError('MAPPO branch-reference final equivalence failed; additional budget needs MAIN')
    else:episode(job)
    meta=public_completion(spec,Path(job['path']))
    if spec['kind'] in ('branch','reuse'):meta['prefix_pass']=True
    write(root/'final_receipts'/(spec['id']+'.json'),meta)
    return meta


def blocked(root):
    return (root/'FINAL_PRECOMPUTE_STOP.json').exists() or (root/'FINAL_PRECOMPUTE_FAILED.json').exists()


def idle_admission(root):
    if blocked(root):raise ValueError('durable final-precompute STOP')
    # SCHED_IDLE children cannot compete with B/R or exact grading. Additionally
    # stop submitting while the grade queue is active or resource guards bind.
    latest=sorted(root.glob('EXACT_GRADER_STATUS_*.json'))
    grade=read(latest[-1]) if latest else None
    if grade is None or grade.get('pending',0) or grade.get('running',0):return False
    mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
    return (mem['MemAvailable']>=8*1024**3 and shutil.disk_usage(RAM).free>=8*1024**3
            and shutil.disk_usage(root).free>=2*1024**3 and os.getloadavg()[0]<28)


def run_stage(root,b,items,workers):
    existing={x['id'] for x in ledger(root)}
    for spec in items:
        # No restarting a partial reservation/attempt, including across crashes.
        assert spec['id'] not in existing,'attempt already reserved; preserve/no physics retry'
        p=root/'jobs'/spec['id'];q=RAM/spec['id']
        assert not p.exists() and not p.is_symlink() and not q.exists()
    active={};pending=list(items)
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        while pending or active:
            while pending and len(active)<workers and idle_admission(root):
                s=pending.pop(0);p=root/'jobs'/s['id'];q=RAM/s['id']
                q.mkdir();p.symlink_to(q,target_is_directory=True)
                reserve(root,[s],s['phase'])
                active[pool.submit(execute_one,root,b,s)]=s
            if active:
                done,_=wait(active,timeout=10,return_when=FIRST_COMPLETED)
                for f in done:
                    s=active.pop(f);meta=f.result()
                    assert meta['id']==s['id'] and meta['outcomes_closed']
            else:time.sleep(10)
            write(root/f'FINAL_PRECOMPUTE_STATUS_{time.time_ns()}.json',dict(unix=time.time(),
                active=[s['id'] for s in active.values()],pending=len(pending),
                completed=len(items)-len(pending)-len(active),charged_sim_s=charged(root),
                outcomes_closed=True,load=os.getloadavg()))


def analyze_complete(root,b,plan):
    """The sole outcome-opening gate. No inference during precompute/training."""
    assert endpoints_ready(root),'all fixed endpoints required before any outcome opening'
    from .branch_eval import analyze,window
    from .accel_runner import panel_report
    write(root/'FINAL_OUTCOMES_OPENED.json',dict(unix=time.time(),fixed_endpoints=budget()['updates'],
        panel_sha256=budget()['final_panel_sha256'],checkpoint_sha256={
            a:sha(root/f'{a}_checkpoint_{u:03}.pt') for a,u in budget()['updates'].items()}))
    eps=[]
    for s in specs(('mappo','native','A','B','R')):
        result=read(root/'jobs'/s['id']/'RESULT.json')
        eps.append({k:result[k] for k in ('id','world','arm','metrics')})
        del result
    panel_report(root,'FINAL_TEST_COMPLETE.json',[e for e in eps if e['arm'] in ('A','B','R')],
                 [e for e in eps if e['arm'] in ('mappo','native')],'final_test')
    records=[]
    for w in worlds('final_test'):
        ref=read(root/'jobs'/f'fixedfinal_mappo_{w["seed"]}'/'RESULT.json')
        for t in TIMES:
            for arm in ('MAPPO','A','B'):
                result=ref if arm=='MAPPO' else read(root/'jobs'/f'fixedbranch_{arm}_{w["seed"]}_t{t}'/'RESULT.json')
                m=window(result,t,300)
                records.append(dict(seed=w['seed'],branch_time_s=t,arm=arm,J=m['team_delay_vs'],window=m))
    write(root/'FINAL_BRANCH_COMPLETE.json',analyze(records,plan))
    write(root/'FINAL_REGISTERED_EVALUATION_COMPLETE.json',dict(unix=time.time(),
        panel_sha256=budget()['final_panel_sha256'],charged_sim_s=charged(root),
        episode_arms=['mappo','native','A','B','R'],branch_arms=['MAPPO','A','B'],
        discovery='separate registered detector-first exact-twin allocation; no automatic budget transfer'))


def run(root,config):
    import fcntl
    lock=(root/'FINAL_PRECOMPUTE.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (root/'FINAL_PRECOMPUTE_STARTED.json').exists(),'one-use precompute already started'
    for name,h in config['source_hashes'].items():assert sha(Path(config['source_root'])/name)==h,name
    assert config['amendment_sha256'] in (root/'AUTHORITY.md').read_text()
    assert config['budget']==budget()
    b=read(root/'WARM_BINDING.json');b=dict(b,budget=budget())
    install_accounting()
    assert RAM.parent.is_dir() and not RAM.exists()
    assert shutil.disk_usage(RAM.parent).free>=16*1024**3
    RAM.mkdir()
    write(root/'FINAL_PRECOMPUTE_STARTED.json',dict(unix=time.time(),config_sha256=digest(config),outcomes_closed=True))
    try:
        while not endpoints_ready(root,('A',)):
            if blocked(root):raise ValueError('precompute STOP while waiting for A48')
            time.sleep(20)
        write(root/'FINAL_PRECOMPUTE_FIXED_A.json',dict(unix=time.time(),update=48,
            checkpoint_sha256=sha(root/'A_checkpoint_048.pt'),outcomes_closed=True))
        # One registered MAPPO episode first, then the 600-s reuse equivalence
        # check before any paired-branch work. Same reference later in the96.
        panel=worlds('final_test');first=specs(('mappo',),panel=panel[:1])
        run_stage(root,b,first,1)
        reuse=dict(id='fixedreuse_mappo_'+str(panel[0]['seed']),world=panel[0],arm='mappo',
                   kind='reuse',time_s=300,sim_s=600,phase='branch_reuse_equivalence')
        run_stage(root,b,[reuse],1)
        write(root/'FINAL_MAPPO_REUSE_PASS.json',dict(pass_=True,sim_s=600,
            receipt_sha256=sha(root/'final_receipts'/(reuse['id']+'.json')),outcomes_closed=True))
        run_stage(root,b,specs(('mappo',),panel=panel[1:])+specs(('native','A')),config['workers'])
        run_stage(root,b,specs(('A',),branches=True),config['workers'])
        write(root/'FINAL_PRECOMPUTE_COMPLETE.json',dict(unix=time.time(),outcomes_closed=True,
            completed_jobs=673,sim_s=691800,charged_sim_s=charged(root)))
        while not endpoints_ready(root):
            if blocked(root):raise ValueError('final STOP while waiting for fixed endpoints')
            time.sleep(30)
        run_stage(root,b,specs(('B','R')),config['workers'])
        run_stage(root,b,specs(('B',),branches=True),config['workers'])
        analyze_complete(root,b,config['branch_plan'])
    except BaseException as error:
        # Deliberately do not stringify worker errors into public receipts:
        # arbitrary backend errors might contain private outcome values.
        write(root/'FINAL_PRECOMPUTE_FAILED.json',dict(unix=time.time(),error_type=type(error).__name__,
            outcomes_closed=not (root/'FINAL_OUTCOMES_OPENED.json').exists(),
            physics_retry=False,action='preserve all charges/results; inspect engineering traceback privately'))
        raise
    finally:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('config',type=Path)
    a=p.parse_args();run(a.root,read(a.config))
