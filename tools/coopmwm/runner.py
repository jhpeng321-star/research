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
from .protocol import (authority,budget,worlds,batch_worlds,sha,digest,UPDATES,CHECK_EVERY,HORIZONS,TRAINING_SEEDS)
from .storage import read,write,reserve,ledger,reconciled
from .policy import Policy
from .credit import algebra,multihorizon,multibonus
from tools.coopnative.learner import update,save,load


def verify(root):
    b=read(root/'BINDING.json')
    if b['budget']!=budget() or authority((root/'AUTHORITY.md').read_text())!=b['authority']:
        raise ValueError('registered authority/budget mismatch')
    if not b['cpu']['pass_'] or b['cpu']['skips'] or b['cpu']['sim_s'] or not b['exposure']['pass_']:
        raise ValueError('CPU/exposure qualification absent')
    if not b['exposure'].get('complete_all_hosts',False):
        raise ValueError('full cross-host seed exposure audit incomplete')
    if b.get('final_panel_sha256')!=digest(worlds('final_test')):
        raise ValueError('final panel prospective binding absent')
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
            weights=weights,runtime=b['runtime'],checkpoint=b['checkpoint'],noise_seed=b.get('training_seed',20261101)+w['seed']+sum(map(ord,ident)),
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
                 for s in ep['steps'] if len(s['group'])==2 and s['time_s']<=600]
        if not choices:missing.append(cell);continue
        ep,s=choices[int(rng.integers(len(choices)))];selected.append((ep,s))
    return selected,missing


def exact_window(root,b,policy,ep,step,index,phase='calibration',role=None):
    t=step['time_s'];group=step['group'];samples={str(s['time_s']):s['sample'] for s in ep['steps']}
    # Reserve every member and all-reversion BEFORE launching any member of a group.
    plans=[('joint',[]),*[(f'revert{i}',[v]) for i,v in enumerate(group)],('all',group)]
    jobs=[];results={}
    for name,revert in plans:
        key=f'{phase}{index:03}_{name}_{ep["world"]["seed"]}';path=root/'jobs'/key
        if (path/'RESULT.json').exists():results[name]=read(path/'RESULT.json');continue
        jobs.append(dict(id=key,path=str(path),world=ep['world'],role=role or ('train' if phase=='calibration' else 'dev'),
            arm=name,weights=policy.state_dict(),runtime=b['runtime'],checkpoint=b['checkpoint'],
            noise_seed=0,deterministic=True,sim_s=t+600,stop=t+600,samples=samples,horizons=HORIZONS,
            branch=dict(time_s=t,revert=revert,prefixes=ep['prefixes'])))
    if jobs:
        reserve(root,jobs,phase);admission(b)
        from .plant import episode
        with ProcessPoolExecutor(max_workers=b['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
            for job,r in zip(jobs,pool.map(episode,jobs)):
                name=next(n for n,_ in plans if f'_{n}_' in job['id']);results[name]=r
    # Full-prefix terms cancel; using total J is identical to subtracting common prefix.
    return {str(h):algebra(results['joint']['horizon_totals'][str(h)]['team_delay_vs'],
                   [results[f'revert{i}']['horizon_totals'][str(h)]['team_delay_vs'] for i in range(len(group))],
                   results['all']['horizon_totals'][str(h)]['team_delay_vs']) for h in HORIZONS}


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
                     wm=wm['credits'],exact=ex,noise=wm['noise'],horizons_s=HORIZONS)
        write(path,record);records.append(record)
    verdict=multihorizon(records);write(root/'CREDIT_CHECK.json',verdict);return verdict


def compare(records,baseline,left,right,role):
    import types
    from tools.coopnative.evaluation import paired as original
    paired=types.FunctionType(original.__code__,dict(original.__globals__,worlds=worlds))
    rs=[dict(r,arm='learned' if r['arm']==left else right) for r in records+baseline
        if r['arm'] in (left,right)]
    return paired(rs,right,role)


def report_panel(root,u,records,controls,role,training_seed=0):
    comparisons={}
    available={r['arm'] for r in records}
    for arm in ('A','B'):
        if arm in available:comparisons[arm+'_minus_MAPPO']=compare(records,controls,arm,'mappo',role)
    if {'A','B'}<=available:comparisons['B_minus_A']=compare(records,[], 'B','A',role)
    out=dict(update=u,training_seed=training_seed,comparisons=comparisons,charged_sim_s=sum(r['sim_s'] for r in ledger(root)),
             safety={a:{h:{k:sum(r['metrics'][h][k] for r in records if r['arm']==a)
                for k in ('collisions','teleports','ttc_below_3_vs','hard_brake')} for h in ('600','900')}
                for a in available})
    write(root/f'S{training_seed}_{role.upper()}_{u:03}.json',out)


def discover(root,b,policy,episodes,role='dev'):
    from tools.coopnative.evaluation import behaviour
    candidates=[]
    for ep in episodes:
        info=behaviour(ep,root/'jobs'/ep['id']/'observable.jsonl.gz')
        candidates.append(dict(id=ep['id'],world=ep['world'],behaviour=info))
    write(root/f'{role.upper()}_DISCOVERY_DETECTORS.json',candidates)
    # Chronological complete group windows in detector-bearing worlds; never positive-credit selected.
    index=0;coverage=[]
    for ep,item in zip(episodes,candidates):
        events=item['behaviour']['detectors']['events']
        for event in events:
            times=[v for k,v in event.items() if k in ('start_s','start_time_s','time_s')]
            if not times:
                coverage.append(dict(id=ep['id'],covered=False,reason='detector event has no registered decision clock',event=event));continue
            time_s=float(times[0]);step=next((s for s in ep['steps'] if s['time_s']<=time_s<s['time_s']+10 and len(s['group'])==2 and s['time_s']<=600),None)
            if step is None:
                coverage.append(dict(id=ep['id'],covered=False,reason='no complete RSU pair/window'));continue
            cost=4*(step['time_s']+600)
            if sum(r['sim_s'] for r in ledger(root) if r['phase']=='discovery')+cost>budget()['phases']['discovery']:
                coverage.append(dict(id=ep['id'],time_s=step['time_s'],covered=False,reason='whole-group discovery budget exhausted'));continue
            credit=exact_window(root,b,policy,ep,step,index,'discovery',role);index+=1
            coverage.append(dict(id=ep['id'],time_s=step['time_s'],group=step['group'],credit=credit,covered=True))
    write(root/f'{role.upper()}_DISCOVERY.json',dict(coverage=coverage,exact_groups=index,horizons_s=HORIZONS))


def training_noise(episodes):
    from tools.coopnative.learner import advantages,REWARD_UNIT
    raw=[a*REWARD_UNIT for ep in episodes for a,_ in advantages(ep['steps'])]
    return dict(advantage_variance_veh_s_squared=float(np.var(raw,ddof=1)),windows=len(raw),
                scope='empirical across TRAIN windows; unnormalized GAE, team reward only')


def final_test(root,b,policies):
    """Only reached after every registered training seed; no online publication."""
    if (root/'FINAL_TEST_COMPLETE.json').exists():return
    if not all((root/f'S{i}_TRAINING_COMPLETE.json').exists() for i in range(3)):
        raise ValueError('final test forbidden before all registered training ends')
    required=[root/f'S{i}_{a}_checkpoint_{UPDATES:03}.pt' for i in range(3) for a in ('A','B')]
    if not all(p.exists() for p in required):
        if not (root/'FINAL_TEST_NOT_RUN.json').exists():write(root/'FINAL_TEST_NOT_RUN.json',dict(reason='required registered final checkpoint absent',no_fallback=True))
        return
    if (root/'FINAL_TEST_STARTED.json').exists():raise ValueError('final test one-use claim already exists')
    manifest=dict(panel_sha256=digest(worlds('final_test')),checkpoints={p.name:sha(p) for p in required},
                  primary_training_seed=20261101,final_update=UPDATES,precharged_sim_s=172800)
    write(root/'FINAL_TEST_STARTED.json',manifest)
    records=[]
    zero=Policy()
    # All four complete arms are fixed before the one-use claim, independent of outcomes.
    for a,p in [('mappo',zero),('native',zero),('A',policies['A']),('B',policies['B'])]:
        records+=wave(root,b,p,worlds('final_test'),a,'final_test','final','final_test',True,
                      dict(native_reference=True,authority=b['authority']) if a=='native' else None)
    comparisons={k:compare(records,[],l,r,'final_test') for k,l,r in
                 [('B_minus_MAPPO','B','mappo'),('B_minus_A','B','A'),('A_minus_MAPPO','A','mappo')]}
    cells={str(c):{a:{h:{k:float(np.mean([r['metrics'][h][k] for r in records if r['world']['cell']==c and r['arm']==a]))
          for k in ('team_delay_vs','collisions','ttc_below_3_vs','hard_brake')} for h in ('600','900')}
          for a in ('mappo','native','A','B')} for c in range(12)}
    write(root/'FINAL_TEST_COMPLETE.json',dict(manifest=manifest,comparisons=comparisons,cells=cells,worlds=36))
    discover(root,b,policies['B'],[r for r in records if r['arm']=='B'],'final_test')


def run(root,smoke_only=False):
    b=verify(root)
    import fcntl
    claim=(root/'SUPERVISOR.claim').open('a');fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    torch.set_num_threads(1);torch.manual_seed(20261101)
    primary={}
    try:
        reconciled(root)
        if not (root/'SMOKE.json').exists():
            smokes=[]
            for a in ('A','B'):smokes+=wave(root,b,Policy(),worlds('dev')[:1],a,'dev','smoke','mechanics',True)
            base=next(r for r in read(root/'MAPPO_DEV.json')['records'] if r['world']==smokes[0]['world'])
            if any(ep['metrics']!=base['metrics'] for ep in smokes) or smokes[0]['final']!=smokes[1]['final']:
                raise ValueError('MAPPO zero-update full-episode parity failed')
            write(root/'SMOKE.json',dict(pass_=True,arms=['A','B'],episodes=2,sim_s=2400,
                  metrics=smokes[0]['metrics'],MAPPO_baseline_reused=True,new_MAPPO_sim_s=0))
        if smoke_only:return
        controls=wave(root,b,Policy(),worlds('check'),'mappo','check','control','checks',True)
        active=['A','B'];gate=None
        for si,seed in enumerate(TRAINING_SEEDS):
            b=dict(b,training_seed=seed);policies={};optimizers={};generators={}
            for a in ('A','B'):
                # Same initial parameters AND exploration stream within each A/B pair.
                torch.manual_seed(seed);policies[a]=Policy()
                optimizers[a]=torch.optim.Adam(policies[a].parameters(),lr=3e-4)
                generators[a]=torch.Generator().manual_seed(seed)
                p=root/f'S{si}_{a}_checkpoint_000.pt'
                if not p.exists():save(p,policies[a],optimizers[a],generators[a],0,sum(r['sim_s'] for r in ledger(root)))
            if si==0:
                initial=wave(root,b,policies['B'],batch_worlds(1,si),'B','train','s0_u001','train')
                gate=credit_gate(root,b,policies['B'],initial)
                if not gate['pass_']:
                    active=['A']
                    if not (root/'B_STOP.json').exists():write(root/'B_STOP.json',dict(reason='registered WM-credit gate failed',credit_check=gate))
            for u in range(1,UPDATES+1):
                panel=[];dev=[]
                for a in active:
                    checkpoint=root/f'S{si}_{a}_checkpoint_{u:03}.pt'
                    if checkpoint.exists():policies[a],optimizers[a],generators[a],_=load(checkpoint)
                    else:
                        episodes=initial if si==0 and u==1 and a=='B' else wave(root,b,policies[a],batch_worlds(u,si),a,'train',f's{si}_u{u:03}','train')
                        training=copy.deepcopy(episodes);credits=[]
                        if a=='B' and u>1:
                            for ep in training:
                                for step in ep['steps']:
                                    if len(step['group'])!=2:continue
                                    c=forecast(root,ep,step)['credits'];credits.append(c)
                                    step['reward']+=multibonus(c,u,gate)
                        metrics=update(policies[a],optimizers[a],training,generators[a])
                        charged=sum(r['sim_s'] for r in ledger(root))
                        save(checkpoint,policies[a],optimizers[a],generators[a],u,charged,ppo=metrics)
                        write(root/f'S{si}_{a}_UPDATE_{u:03}.json',dict(update=u,arm=a,training_seed=seed,ppo=metrics,
                            team_only_noise=training_noise(episodes),training_sim_s=u*28800,charged_sim_s=charged,wm_windows=len(credits),
                            credit_mean={h:{k:float(np.mean([sum(c[h][k]) if k=='D_i' else c[h][k] for c in credits])) if credits else None for k in ('D_i','C_G','D_G')} for h in map(str,HORIZONS)}))
                    if u%CHECK_EVERY==0:panel+=wave(root,b,policies[a],worlds('check'),a,'check',f's{si}_check{u:03}','checks',True)
                    if (si==0 and u==4) or u==UPDATES:dev+=wave(root,b,policies[a],worlds('dev'),a,'dev',f's{si}_dev{u:03}','dev',True)
                if panel and not (root/f'S{si}_CHECK_{u:03}.json').exists():report_panel(root,u,panel,controls,'check',si)
                if dev and not (root/f'S{si}_DEV_{u:03}.json').exists():report_panel(root,u,dev,read(root/'MAPPO_DEV.json')['records'],'dev',si)
            if si==0:primary=policies
            write_path=root/f'S{si}_TRAINING_COMPLETE.json'
            if not write_path.exists():write(write_path,dict(training_seed=seed,final_update=UPDATES,active_arms=active))
        # The final test is never used to choose checkpoints or tune any arm.
        final_test(root,b,primary)
        write(root/'COMPLETE.json',dict(updates_per_seed=UPDATES,seeds=TRAINING_SEEDS,active_arms=active,charged_sim_s=reconciled(root),budget=budget()))
    except BaseException as e:
        if not (root/'STOP.json').exists():write(root/'STOP.json',dict(error=repr(e),charged_sim_s=sum(r['sim_s'] for r in ledger(root))))
        raise
    finally:claim.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--smoke-only',action='store_true')
    a=p.parse_args();run(a.root,a.smoke_only)
