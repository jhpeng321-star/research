"""Immutable, single-owner budgeted A/B training with a WM credit qualification gate."""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import time
import torch
import numpy as np
from .protocol import (authority,budget,worlds,batch_worlds,sha,digest,UPDATES,CHECK_EVERY)
from .storage import read,write,reserve,ledger,reconciled
from .policy import Policy
from .credit import algebra,calibration,bonus
from tools.coopnative.learner import update,save,load


def verify(root):
    b=read(root/'BINDING.json')
    if b['budget']!=budget() or authority((root/'AUTHORITY.md').read_text())!=b['authority']:
        raise ValueError('registered authority/budget mismatch')
    if not b['cpu']['pass_'] or b['cpu']['skips'] or b['cpu']['sim_s'] or not b['exposure']['pass_']:
        raise ValueError('CPU/exposure qualification absent')
    for name,h in b['hashes'].items():
        if sha(root/name)!=h:raise ValueError('source/asset binding mismatch '+name)
    if (root/'STOP.json').exists():raise ValueError('durable STOP')
    return b


def admission(b):
    from tools.coopnative.runner import admission as prior
    prior(dict(load_max=b['load_max']))


def wave(root,b,policy,ws,arm,role,ident,phase,deterministic=False,extra=None):
    jobs=[];results=[]
    weights={k:v.detach().cpu() for k,v in policy.state_dict().items()}
    for w in ws:
        key=f'{ident}_{arm}_{w["seed"]}'
        path=root/'jobs'/key
        if (path/'RESULT.json').exists():results.append(read(path/'RESULT.json'));continue
        jobs.append(dict(id=key,path=str(path),world=w,role=role,arm=arm,
            weights=weights,runtime=b['runtime'],checkpoint=b['checkpoint'],noise_seed=20261101+w['seed']+sum(map(ord,ident)),
            deterministic=deterministic,sim_s=1200,**(extra or {})))
    if jobs:
        reserve(root,jobs,phase);admission(b)
        from .plant import episode
        with ProcessPoolExecutor(max_workers=b['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
            results.extend(pool.map(episode,jobs))
    return sorted(results,key=lambda r:r['world']['seed'])


def forecast(root,ep,step):
    record=dict(state_id=digest(dict(world=ep['world']['seed'],time=step['time_s'],
                rows=step['rows'],targets=step['targets'])),world_id=ep['world']['seed'],
                time_s=step['time_s'],rows=step['rows'],targets=step['targets'],group=step['group'])
    key=record['state_id'];request=root/'requests'/(key+'.json');result=root/'responses'/(key+'.json')
    if not request.exists():write(request,record)
    while not result.exists():
        if (root/'STOP.json').exists():raise ValueError('STOP during replay')
        time.sleep(2)
    value=read(result)
    if value.get('record_sha256')!=digest(record):raise ValueError('forecast response identity mismatch')
    return value


def select_calibration(episodes):
    rng=np.random.default_rng(20261101);selected=[];missing=[]
    for cell in range(12):
        choices=[(ep,s) for ep in episodes if ep['world']['cell']==cell
                 for s in ep['steps'] if len(s['group'])==2 and s['time_s']<=1110]
        if not choices:missing.append(cell);continue
        ep,s=choices[int(rng.integers(len(choices)))];selected.append((ep,s))
    return selected,missing


def exact_window(root,b,policy,ep,step,index,phase='calibration'):
    t=step['time_s'];group=step['group'];samples={str(s['time_s']):s['sample'] for s in ep['steps']}
    # Reserve every member and all-reversion BEFORE launching any member of a group.
    plans=[('joint',[]),*[(f'revert{i}',[v]) for i,v in enumerate(group)],('all',group)]
    jobs=[];results={}
    for name,revert in plans:
        key=f'{phase}{index:03}_{name}_{ep["world"]["seed"]}';path=root/'jobs'/key
        if (path/'RESULT.json').exists():results[name]=read(path/'RESULT.json');continue
        jobs.append(dict(id=key,path=str(path),world=ep['world'],role='train' if phase=='calibration' else 'dev',
            arm=name,weights=policy.state_dict(),runtime=b['runtime'],checkpoint=b['checkpoint'],
            noise_seed=0,deterministic=True,sim_s=t+90,stop=t+90,samples=samples,
            branch=dict(time_s=t,revert=revert,prefixes=ep['prefixes'])))
    if jobs:
        reserve(root,jobs,phase);admission(b)
        from .plant import episode
        with ProcessPoolExecutor(max_workers=b['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
            for job,r in zip(jobs,pool.map(episode,jobs)):
                name=next(n for n,_ in plans if f'_{n}_' in job['id']);results[name]=r
    # Full-prefix terms cancel; using total J is identical to subtracting common prefix.
    return algebra(results['joint']['totals']['team_delay_vs'],
                   [results[f'revert{i}']['totals']['team_delay_vs'] for i in range(len(group))],
                   results['all']['totals']['team_delay_vs'])


def credit_gate(root,b,policy,episodes):
    if (root/'CREDIT_CHECK.json').exists():return read(root/'CREDIT_CHECK.json')
    selected,missing=select_calibration(episodes)
    manifest=[dict(world=e['world'],time_s=s['time_s'],group=s['group']) for e,s in selected]
    if not (root/'CALIBRATION_WINDOWS.json').exists():write(root/'CALIBRATION_WINDOWS.json',dict(windows=manifest,missing_cells=missing,rng=20261101))
    if missing:
        verdict=dict(pass_=False,reason='no complete RSU group window in registered initial cell',missing_cells=missing)
        write(root/'CREDIT_CHECK.json',verdict);return verdict
    records=[]
    for i,(ep,step) in enumerate(selected):
        path=root/f'CREDIT_WINDOW_{i:02}.json'
        if path.exists():records.append(read(path));continue
        wm=forecast(root,ep,step)
        ex=exact_window(root,b,policy,ep,step,i)
        record=dict(world=ep['world'],time_s=step['time_s'],group=step['group'],
                    wm=wm['credit'],exact=ex,horizon_s=90)
        write(path,record);records.append(record)
    verdict=calibration(records);write(root/'CREDIT_CHECK.json',verdict);return verdict


def compare(records,baseline,left,right,role):
    from tools.coopnative.evaluation import paired
    rs=[dict(r,arm='learned' if r['arm']==left else right) for r in records+baseline
        if r['arm'] in (left,right)]
    return paired(rs,right,role)


def report_panel(root,u,records,controls,role):
    comparisons={}
    available={r['arm'] for r in records}
    for arm in ('A','B'):
        if arm in available:comparisons[arm+'_minus_MAPPO']=compare(records,controls,arm,'mappo',role)
    if {'A','B'}<=available:comparisons['B_minus_A']=compare(records,[], 'B','A',role)
    out=dict(update=u,comparisons=comparisons,charged_sim_s=sum(r['sim_s'] for r in ledger(root)),
             safety={a:{h:{k:sum(r['metrics'][h][k] for r in records if r['arm']==a)
                for k in ('collisions','teleports','ttc_below_3_vs','hard_brake')} for h in ('600','900')}
                for a in available})
    write(root/f'{role.upper()}_{u:03}.json',out)


def discover(root,b,policy,episodes):
    from tools.coopnative.evaluation import behaviour
    candidates=[]
    for ep in episodes:
        info=behaviour(ep,root/'jobs'/ep['id']/'observable.jsonl.gz')
        candidates.append(dict(id=ep['id'],world=ep['world'],behaviour=info))
    write(root/'DISCOVERY_DETECTORS.json',candidates)
    # Chronological complete group windows in detector-bearing worlds; never positive-credit selected.
    index=0;coverage=[]
    for ep,item in zip(episodes,candidates):
        events=item['behaviour']['detectors']['events']
        for event in events:
            times=[v for k,v in event.items() if k in ('start_s','start_time_s','time_s')]
            if not times:
                coverage.append(dict(id=ep['id'],covered=False,reason='detector event has no registered decision clock',event=event));continue
            time_s=float(times[0]);step=next((s for s in ep['steps'] if s['time_s']<=time_s<s['time_s']+10 and len(s['group'])==2 and s['time_s']<=1110),None)
            if step is None:
                coverage.append(dict(id=ep['id'],covered=False,reason='no complete RSU pair/window'));continue
            credit=exact_window(root,b,policy,ep,step,index,'discovery');index+=1
            coverage.append(dict(id=ep['id'],time_s=step['time_s'],group=step['group'],credit=credit,covered=True))
    write(root/'DISCOVERY.json',dict(coverage=coverage,exact_groups=index,horizon_s=90))


def run(root,smoke_only=False):
    b=verify(root)
    import fcntl
    claim=(root/'SUPERVISOR.claim').open('a');fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    torch.set_num_threads(1);torch.manual_seed(20261101)
    policies={};optimizers={};generators={}
    for a in ('A','B'):
        policies[a]=Policy();optimizers[a]=torch.optim.Adam(policies[a].parameters(),lr=3e-4)
        generators[a]=torch.Generator().manual_seed(20261101)
        if not (root/f'{a}_checkpoint_000.pt').exists():save(root/f'{a}_checkpoint_000.pt',policies[a],optimizers[a],generators[a],0,0)
    try:
        reconciled(root)
        if not (root/'SMOKE.json').exists():
            smokes=[]
            for a in ('A','B'):smokes+=wave(root,b,policies[a],worlds('dev')[:1],a,'dev','smoke','mechanics',True)
            base=next(r for r in read(root/'MAPPO_DEV.json')['records'] if r['world']==smokes[0]['world'])
            if any(ep['metrics']!=base['metrics'] for ep in smokes) or smokes[0]['final']!=smokes[1]['final']:
                raise ValueError('MAPPO zero-update full-episode parity failed')
            write(root/'SMOKE.json',dict(pass_=True,arms=['A','B'],episodes=2,sim_s=2400,
                  metrics=smokes[0]['metrics'],MAPPO_baseline_reused=True,new_MAPPO_sim_s=0))
        if smoke_only:return
        controls=wave(root,b,policies['A'],worlds('check'),'mappo','check','control','checks',True)
        active=['A','B'];check0=None
        for u in range(1,UPDATES+1):
            panel=[];dev=[]
            # B initial trajectories qualify the WM before either arm's first optimizer update.
            if u==1:
                initial=wave(root,b,policies['B'],batch_worlds(u),'B','train','u001','train')
                gate=credit_gate(root,b,policies['B'],initial)
                if not gate['pass_']:
                    active=['A'];write(root/'B_STOP.json',dict(reason='registered WM-credit gate failed',credit_check=gate))
            for a in active:
                checkpoint=root/f'{a}_checkpoint_{u:03}.pt'
                if checkpoint.exists():
                    policies[a],optimizers[a],generators[a],_=load(checkpoint)
                else:
                    episodes=initial if u==1 and a=='B' else wave(root,b,policies[a],batch_worlds(u),a,'train',f'u{u:03}','train')
                    training=copy.deepcopy(episodes);credits=[]
                    if a=='B' and u>1:
                        for ep in training:
                            for step in ep['steps']:
                                if len(step['group'])!=2:continue
                                c=forecast(root,ep,step)['credit'];credits.append(c)
                                step['reward']+=bonus(c,u,gate['scale'])
                    metrics=update(policies[a],optimizers[a],training,generators[a])
                    charged=sum(r['sim_s'] for r in ledger(root))
                    save(checkpoint,policies[a],optimizers[a],generators[a],u,charged,ppo=metrics)
                    write(root/f'{a}_UPDATE_{u:03}.json',dict(update=u,arm=a,ppo=metrics,
                        training_sim_s=u*28800,charged_sim_s=charged,wm_windows=len(credits),
                        credit_mean={k:float(np.mean([sum(c[k]) if k=='D_i' else c[k] for c in credits])) if credits else None for k in ('D_i','C_G','D_G')}))
                if u%CHECK_EVERY==0:panel+=wave(root,b,policies[a],worlds('check'),a,'check',f'check{u:03}','checks',True)
                if u in (4,UPDATES):dev+=wave(root,b,policies[a],worlds('dev'),a,'dev',f'dev{u:03}','dev',True)
            if panel and not (root/f'CHECK_{u:03}.json').exists():report_panel(root,u,panel,controls,'check')
            if dev and not (root/f'DEV_{u:03}.json').exists():report_panel(root,u,dev,read(root/'MAPPO_DEV.json')['records'],'dev')
        discover(root,b,policies[active[0]],dev)
        write(root/'COMPLETE.json',dict(updates=UPDATES,active_arms=active,charged_sim_s=reconciled(root),budget=budget()))
    except BaseException as e:
        if not (root/'STOP.json').exists():write(root/'STOP.json',dict(error=repr(e),charged_sim_s=sum(r['sim_s'] for r in ledger(root))))
        raise
    finally:claim.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--smoke-only',action='store_true')
    a=p.parse_args();run(a.root,a.smoke_only)
