"""Fresh warm namespace: preserve A/Adam/RNG; restart B/R once from MAPPOu0.

The early B/R endpoint is the registered u4 viability pause; fixed full-budget
endpoints remain A/B48,R24. No final-panel execution lives in this supervisor.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import numpy as np
import torch
from .accel_runner import timestamp,wave,routing,panel_report,diagnostic_task
from .accel_learner import Policy,load,save,update,evidence,viability,advantages,UNIT
from .accel_storage import read,write,charged
from .protocol import sha,digest
from .warm_protocol import budget,batch_worlds,worlds,SEED,install_accounting
from .exploration import trust
from . import search_learning


def verify(root,arm):
    b=read(root/'WARM_BINDING.json')
    assert b['budget']==budget()
    assert b['cpu']['pass_'] and not b['cpu']['skips'] and b['cpu']['sim_s']==0
    for name,h in b['hashes'].items():assert sha(root/name)==h,'immutable warm source '+name
    text=(root/'AUTHORITY.md').read_text()
    assert b['scope_amendment_sha256'] in text and b['release_amendment_sha256'] in text
    assert not (root/'STOP.json').exists()
    if arm in ('B','R'):
        release=read(root/'WARM_RELEASE.json')
        assert release['pass_'] and release['N_exec']>=10
        assert release['release_amendment_sha256'] in text
    install_accounting()
    return b


def wait(root,path,deadline):
    while not path.exists():
        if (root/'STOP.json').exists():raise ValueError('durable warm STOP while waiting')
        if time.time()>deadline:raise ValueError('registered warm runtime deadline')
        time.sleep(2)
    return read(path)


def check(root,b,arm,policy,u):
    controls=read(root/'MAPPO_CHECK.json')['records']
    eps=wave(root,b,{arm:policy},worlds('check'),'check',f'warmcheck{u:03}_{arm}','checks')
    records=[{k:e[k] for k in ('id','world','arm','metrics')} for e in eps]
    write(root/f'{arm}_CHECK_RECORDS_{u:03}.json',records)
    panel_report(root,f'{arm}_CHECK_{u:03}.json',records,controls,'check')
    if arm=='B':
        a=wait(root,root/f'A_CHECK_RECORDS_{u:03}.json',b['gpu_deadline'])
        panel_report(root,f'AB_CHECK_{u:03}.json',a+records,controls,'check')


def run_arm(root,arm):
    import fcntl
    import os
    b=verify(root,arm)
    lock=(root/f'WARM_{arm}.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (root/f'WARM_{arm}_STARTED.json').exists(),'one-use warm arm already started'
    os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)));torch.set_num_threads(1)
    torch.manual_seed(SEED)
    if arm=='A':
        policy,opt,rng,payload=load(root/'A_checkpoint_004.pt');assert payload['update']==4
        first,last=5,48
    else:
        policy=Policy();opt=torch.optim.Adam(policy.parameters(),lr=3e-4)
        rng=torch.Generator().manual_seed(SEED);first,last=1,4
        save(root/f'{arm}_checkpoint_000.pt',policy,opt,rng,0,charged(root))
    write(root/f'WARM_{arm}_STARTED.json',dict(timestamp_utc=timestamp(),first_update=first,
        pause_update=last,full_endpoint=budget()['updates'][arm],charged_sim_s=charged(root)))
    running=[];beta=1.;learned=search_learning.empty(arm,SEED,b['accel_bounds'])
    exact_rng=np.random.default_rng(SEED);exact_count=0
    diagnostics=ThreadPoolExecutor(max_workers=1)
    try:
        for u in range(first,last+1):
            if (root/'STOP.json').exists():raise ValueError('durable warm STOP before next wave')
            start=time.monotonic();extra=dict(update=u)
            if arm=='B':extra['search_learning']=learned
            if arm=='R':
                wait(root,root/f'B_RANDOM_QUOTA_{u:03}.json',b['gpu_deadline'])
                extra['random_quota']=str(root/f'B_RANDOM_QUOTA_{u:03}.json')
            eps=wave(root,dict(b,beta=beta),{arm:policy},batch_worlds(u),'train',f'warm_u{u:03}','train',False,extra)
            routes=routing(eps)
            if arm=='B' and routes['executed']<b['N_exec']:
                raise ValueError(f'B plan gate unmet {routes["executed"]}<{b["N_exec"]}; no extra SUMO')
            if arm=='R':
                quota=read(root/f'B_RANDOM_QUOTA_{u:03}.json')
                for regime in ('merge-dense','ordinary'):
                    expected=sorted((t['group_size'],t['duration_s']) for t in quota['tickets'] if t['regime']==regime)
                    actual=sorted((t['group_size'],t['duration_s']) for t in routes['durations'] if t['regime']==regime)
                    assert actual==expected,'R count/regime/group-size/duration gate'
            ev=evidence(eps);running+=ev
            if arm=='B':
                learned=search_learning.update(learned,eps,ev,arm,SEED)
                write(root/f'B_SEARCH_LEARNING_{u:03}.json',learned)
            write(root/f'{arm}_VIABILITY_{u:03}.json',dict(timestamp_utc=timestamp(),update=u,
                arm=arm,viability=viability(ev),cumulative=viability(running),routing=routes))
            # Snapshot the rollout policy BEFORE the optimizer for exact twins.
            state={k:v.detach().clone() for k,v in policy.state_dict().items()}
            metrics=update(policy,opt,eps,rng)
            tr=trust(beta,running) if arm in ('B','R') else dict(beta=0.)
            save(root/f'{arm}_checkpoint_{u:03}.pt',policy,opt,rng,u,charged(root))
            raw=[v*UNIT for e in eps for v,_ in advantages(e['steps'])]
            write(root/f'{arm}_UPDATE_{u:03}.json',dict(timestamp_utc=timestamp(),update=u,arm=arm,
                training_seed=SEED,ppo=metrics,trust=tr,beta_used=beta if arm!='A' else 0.,
                proposal_routing=routes,team_only_noise=dict(advantage_variance_veh_s_squared=float(np.var(raw,ddof=1))),
                wall_s=time.monotonic()-start,eta_final_hours=(budget()['updates'][arm]-u)*(time.monotonic()-start)/3600,
                charged_sim_s=charged(root),reward_source='SUMO source-inclusive team J only'))
            beta=tr
            if arm=='B':
                write(root/f'B_RANDOM_QUOTA_{u:03}.json',dict(sampling=routes['sampling'],
                    tickets=[dict(t,id=i) for i,t in enumerate(routes['durations'])]))
                candidates=[(e,s) for e in eps for s in e['steps'] if s['time_s']<=600 and
                    s['sample'].get('behaviour',{}).get('plan_start')]
                for i in exact_rng.permutation(len(candidates))[:max(0,12-exact_count)]:
                    e,s=candidates[int(i)]
                    diagnostics.submit(diagnostic_task,root,b,state,e,s,exact_count);exact_count+=1
            if u%4==0:check(root,b,arm,policy,u)
            if u%16==0:
                dev=wave(root,b,{arm:policy},worlds('dev'),'dev',f'warmdev{u:03}_{arm}','dev')
                panel_report(root,f'{arm}_DEV_{u:03}.json',dev,read(root/'MAPPO_DEV.json')['records'],'dev')
        diagnostics.shutdown(wait=True)
        write(root/f'WARM_{arm}_PAUSED.json',dict(timestamp_utc=timestamp(),update=last,
            full_endpoint=budget()['updates'][arm],charged_sim_s=charged(root),
            reason='early viability owner review; final panel unopened' if arm!='A' else 'A fixedu48 complete; wait for B48 before final panel'))
    except BaseException as error:
        write(root/f'WARM_{arm}_FAILED.json',dict(timestamp_utc=timestamp(),error=repr(error)))
        if not (root/'STOP.json').exists():write(root/'STOP.json',dict(arm=arm,error=repr(error),timestamp_utc=timestamp()))
        raise
    finally:
        diagnostics.shutdown(wait=True);lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--arm',choices=('A','B','R'),required=True)
    a=p.parse_args();run_arm(a.root,a.arm)
