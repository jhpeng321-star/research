"""Registered one-use acceleration restart. No old checkpoint/job is overwritten."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import multiprocessing
import os
from pathlib import Path
import time
import types
import numpy as np
import torch
from .accel_protocol import worlds,budget,SEED,UPDATES,guard,digest,sha
from .accel_storage import read,write,reserve,ledger,charged
from .acceleration import Policy
from .accel_learner import save,load,update,evidence,viability,advantages,UNIT
from .exploration import trust
from .plans import SEARCH


def timestamp():return datetime.now(timezone.utc).isoformat()


def verify(root):
    b=read(root/('BINDING_TRAINING.json' if (root/'BINDING_TRAINING.json').exists() else 'BINDING.json'))
    if b['budget']!=budget() or b['search']!=SEARCH:raise ValueError('registered plan/budget mismatch')
    if b['amendment_sha256'] not in (root/'AUTHORITY.md').read_text():raise ValueError('numeric amendment unregistered')
    if not b['cpu']['pass_'] or b['cpu']['skips'] or b['cpu']['sim_s']:raise ValueError('CPU gate')
    if not all(v['pass_'] for v in b['gpu_qualification'].values()):raise ValueError('GPU qualification gate')
    for name,h in b['hashes'].items():
        if sha(root/name)!=h:raise ValueError('immutable source mismatch '+name)
    if (root/'STOP.json').exists():raise ValueError('durable STOP; no automatic retry')
    return b


def wave(root,b,policies,ws,role,ident,phase,deterministic=True,extras=None):
    jobs=[];results=[]
    for w in ws:
        for arm,policy in policies.items():
            key=f'{ident}_{arm}_{w["seed"]}';path=root/'jobs'/key
            if (path/'RESULT.json').exists():results.append(read(path/'RESULT.json'));continue
            if key in {x['id'] for x in ledger(root)}:raise ValueError('attempted job cannot be retried '+key)
            jobs.append(dict(id=key,path=str(path),world=w,role=role,arm=arm,
                weights=policy.state_dict(),runtime=b['runtime'],checkpoint=b['checkpoint'],
                noise_seed=int(digest(dict(seed=SEED,world=w['seed'],wave=ident))[:15],16),
                deterministic=deterministic,sim_s=1200,queue_root=str(root),accel_bounds=b['accel_bounds'],
                gpu_deadline=b['gpu_deadline'],guided=not deterministic and arm in ('B','R'),
                beta=b.get('beta',1.),**(extras or {})))
    if jobs:
        reserve(root,jobs,phase)
        from .runner import admission
        admission(b)
        from .accel_plant import episode
        with ProcessPoolExecutor(max_workers=b['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
            results.extend(pool.map(episode,jobs))
    return sorted(results,key=lambda e:(e['world']['seed'],e['arm']))


def compare(records,left,right,role):
    from tools.coopnative.evaluation import paired
    fn=types.FunctionType(paired.__code__,dict(paired.__globals__,worlds=worlds))
    mapped=[dict(e,arm='learned' if e['arm']==left else right) for e in records if e['arm'] in (left,right)]
    return fn(mapped,right,role)


def panel_report(root,name,records,controls,role):
    available={r['arm'] for r in records};all_records=records+controls
    contrasts={a+'_minus_MAPPO':compare(all_records,a,'mappo',role) for a in ('A','B','R') if a in available}
    if {'A','B'}<=available:contrasts['B_minus_A']=compare(records,'B','A',role)
    if {'B','R'}<=available:contrasts['B_minus_R']=compare(records,'B','R',role)
    metrics=('team_delay_vs','collisions','ttc_below_3_vs','hard_brake')
    cells={str(c):{a:{h:{k:float(np.mean([e['metrics'][h][k] for e in all_records
        if e['world']['cell']==c and e['arm']==a])) for k in metrics} for h in ('600','900')}
        for a in {e['arm'] for e in all_records}} for c in range(12)}
    out=dict(timestamp_utc=timestamp(),comparisons=contrasts,cells=cells,charged_sim_s=charged(root))
    write(root/name,out);return out


def routing(episodes):
    bs=[s['sample'].get('behaviour',{}) for e in episodes for s in e['steps']]
    executed=[b for b in bs if b.get('executed') and b.get('plan_start')]
    def counts(xs):return dict(queries=sum(b.get('route')=='wm' for b in xs),
        accepted=sum(b.get('proposal_accepted',False) for b in xs),
        executed=sum(b.get('executed',False) and b.get('plan_start',False) for b in xs))
    return dict(counts(bs),by_regime={r:counts([b for b in bs if b.get('regime')==r]) for r in ('merge-dense','ordinary')},
        by_bucket={o+'|'+r:counts([b for b in bs if b.get('option')==o and b.get('regime')==r])
            for o in ('O1','O2','O3','O4') for r in ('merge-dense','ordinary')},
        durations=[dict(regime=b['regime'],duration_s=b['duration_s']) for b in executed],
        candidates_searched=sum(b.get('candidates_searched') or 0 for b in bs),
        query_wait_wall_s=sum(b.get('query_wall_s',0.) for b in bs),
        sampling={r:float(np.mean([b.get('sample_probability',0.) for b in bs if b.get('regime')==r]))
            if any(b.get('regime')==r for b in bs) else 0. for r in ('merge-dense','ordinary')})


def exact_plan(root,b,policy,ep,step,index,role='train'):
    from .credit import algebra
    t=step['time_s'];proposal=step['proposal'];plan=proposal['plan'];group=plan['members']
    samples={str(s['time_s']):s['sample'] for s in ep['steps']}
    jobs=[];results={}
    for name,revert in [('joint',[]),('all',group),*[(f'revert{i}',[v]) for i,v in enumerate(group)]]:
        key=f'planexact{index:03}_{name}_{ep["world"]["seed"]}';path=root/'jobs'/key
        jobs.append(dict(id=key,path=str(path),world=ep['world'],role=role,arm=name,
            weights=policy.state_dict(),runtime=b['runtime'],checkpoint=b['checkpoint'],noise_seed=0,
            deterministic=True,sim_s=t+600,stop=t+600,samples=samples,accel_bounds=b['accel_bounds'],gpu_deadline=b['gpu_deadline'],
            branch=dict(time_s=t,revert=revert,group=group,until=t+plan['duration_s'],prefixes=ep['prefixes'])))
    reserve(root,jobs,'discovery')
    from .accel_plant import episode
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        for j,e in zip(jobs,pool.map(episode,jobs)):results[j['arm']]=e
    credits={str(h):algebra(results['joint']['horizon_totals'][str(h)]['team_delay_vs'],
        [results[f'revert{i}']['horizon_totals'][str(h)]['team_delay_vs'] for i in range(len(group))],
        results['all']['horizon_totals'][str(h)]['team_delay_vs']) for h in (60,120,300,600)}
    write(root/f'ACCEL_EXACT_PLAN_{index:03}.json',dict(timestamp_utc=timestamp(),world=ep['world'],
        time_s=t,plan=plan,credits=credits,predicted_saving=proposal['expected_saving']))


def discovery(root,b,policy,episodes,index=12):
    import copy
    from tools.coopfree.analysis import tape
    from tools.coopbehav.detectors import detect
    manifest=[]
    for ep in episodes:
        observed=tape(root/'jobs'/ep['id']/'observable.jsonl.gz',ep['world']['seed'])
        events=detect(observed)['events']
        manifest.append(dict(id=ep['id'],world=ep['world'],events=events))
    write(root/'FINAL_DISCOVERY_DETECTORS.json',manifest)
    coverage=[]
    from .plans import group
    for ep,item in zip(episodes,manifest):
        used=set()
        for event in item['events']:
            times=[float(v) for k,v in event.items() if k in ('start_s','start_time_s','time_s')]
            t=int(times[0]) if times else None
            if t is None or t<300 or t>600 or t in used:continue
            used.add(t);source=next((s for s in ep['steps'] if s['time_s']==t),None)
            members=group(source['rows']) if source else []
            if len(members)!=2:continue
            cost=4*(t+600)
            if sum(x['sim_s'] for x in ledger(root) if x['phase']=='discovery')+cost>budget()['phases']['discovery']:
                coverage.append(dict(id=ep['id'],time_s=t,covered=False,reason='whole group budget exhausted'));continue
            twin=copy.deepcopy(ep)
            for s in twin['steps']:
                if t<=s['time_s']<t+60:
                    s['sample']['absolute_accels']={v:s['requests'].get(v,{}).get('requested_raw',s['commands'][v]) for v in members if v in s['commands']}
            step=next(s for s in twin['steps'] if s['time_s']==t)
            plan=dict(id=f'detected{index}',members=members,duration_s=60,origin='detector-observed-policy',
                shape='recorded1s raw accelerations, no interpolation or family threshold')
            step['proposal']=dict(plan=plan,expected_saving=None)
            exact_plan(root,b,policy,twin,step,index,'final_test');index+=1
            coverage.append(dict(id=ep['id'],time_s=t,covered=True))
    write(root/'FINAL_DISCOVERY_COMPLETE.json',dict(coverage=coverage))


def final(root,b,policies):
    if (root/'FINAL_TEST_STARTED.json').exists():raise ValueError('one-use final panel already claimed')
    for a in ('A','B'):
        if not (root/f'{a}_checkpoint_024.pt').exists():raise ValueError('required fixed final checkpoint absent')
    write(root/'FINAL_TEST_STARTED.json',dict(panel_sha256=digest(worlds('final_test')),
        checkpoints={a:sha(root/f'{a}_checkpoint_024.pt') for a in ('A','B')},timestamp_utc=timestamp()))
    controls=wave(root,b,{'mappo':Policy()},worlds('final_test'),'final_test','final','final_test')
    controls+=wave(root,b,{'native':Policy()},worlds('final_test'),'final_test','final','final_test',
        extras=dict(native_reference=True,authority=b['authority']))
    records=wave(root,b,policies,worlds('final_test'),'final_test','final','final_test')
    # Final outcomes are opened here only, after both fixed primary finals exist.
    panel_report(root,'FINAL_TEST_COMPLETE.json',records,controls,'final_test')
    discovery(root,b,policies['B'],[e for e in records if e['arm']=='B'])


def run(root,smoke_only=False):
    import fcntl
    b=verify(root)
    if b.get('smoke_only') and not smoke_only:raise ValueError('smoke-only binding forbids training')
    lock=(root/'SUPERVISOR.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)));torch.set_num_threads(1)
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    try:
        if not (root/'SMOKE.json').exists():
            smoke=wave(root,b,{a:Policy() for a in ('A','B','R')},worlds('dev')[:1],'dev','smoke','mechanics')
            base=next(e for e in read(root/'MAPPO_DEV.json')['records'] if e['world']==smoke[0]['world'])
            if any(e['metrics']!=base['metrics'] for e in smoke) or len({digest(e['final']) for e in smoke})!=1:
                raise ValueError('one-episode per-arm MAPPO zero-update parity failed')
            write(root/'SMOKE.json',dict(pass_=True,arms=['A','B','R'],sim_s=3600,new_MAPPO_sim_s=0,timestamp_utc=timestamp()))
        if smoke_only:return
        write(root/'TRAINING_STARTED.json',dict(timestamp_utc=timestamp(),search=SEARCH,N_exec=b['N_exec'],
            eta_primary_hours=b['eta_primary_hours'],charged_sim_s=charged(root)))
        primary={};exact_count=0;exact_rng=np.random.default_rng(SEED)
        controls=read(root/'MAPPO_CHECK.json')['records']
        for arms in (('A','B'),('R',)):
            policies={};opts={};rngs={};running={a:[] for a in arms};betas={a:1. for a in arms}
            for a in arms:
                torch.manual_seed(SEED);policies[a]=Policy();opts[a]=torch.optim.Adam(policies[a].parameters(),lr=3e-4)
                rngs[a]=torch.Generator().manual_seed(SEED)
                save(root/f'{a}_checkpoint_000.pt',policies[a],opts[a],rngs[a],0,charged(root))
            for u in range(1,25):
                start=time.monotonic();extra=dict(update=u)
                if arms==('R',):extra['random_quota']=str(root/f'B_RANDOM_QUOTA_{u:03}.json')
                # Each arm's trust differs after update1; keep the same common
                # worker pool while passing arm-specific immutable trust states.
                episodes=[]
                from .protocol import batch_worlds
                if len(arms)==2:
                    # Separate job construction preserves simultaneous A/B use of
                    # admitted capacity; per-job beta is attached by wave below.
                    episodes=wave(root,dict(b,beta=betas['B']),policies,batch_worlds(u),'train',f'u{u:03}','train',False,extra)
                else:episodes=wave(root,dict(b,beta=betas['R']),policies,batch_worlds(u),'train',f'u{u:03}','train',False,extra)
                for a in arms:
                    eps=[e for e in episodes if e['arm']==a];routes=routing(eps)
                    if a in ('B','R'):
                        if a=='B' and routes['executed']<b['N_exec']:raise ValueError(f'B executed plan gate unmet: {routes["executed"]}<{b["N_exec"]}; no extra SUMO')
                        if a=='R':
                            q=read(root/f'B_RANDOM_QUOTA_{u:03}.json')
                            for r in ('merge-dense','ordinary'):
                                expected=sorted(t['duration_s'] for t in q['tickets'] if t['regime']==r)
                                actual=sorted(t['duration_s'] for t in routes['durations'] if t['regime']==r)
                                if actual!=expected:raise ValueError('R count/regime/duration matching gate unmet')
                    ev=evidence(eps);running[a]+=ev
                    vr=viability(ev)
                    # Publish real-SUMO evidence BEFORE any optimizer work.
                    write(root/f'{a}_VIABILITY_{u:03}.json',dict(timestamp_utc=timestamp(),update=u,arm=a,viability=vr,routing=routes))
                    metrics=update(policies[a],opts[a],eps,rngs[a])
                    tr=trust(betas[a],running[a]) if a in ('B','R') else dict(beta=0.)
                    save(root/f'{a}_checkpoint_{u:03}.pt',policies[a],opts[a],rngs[a],u,charged(root))
                    raw=[v*UNIT for e in eps for v,_ in advantages(e['steps'])]
                    write(root/f'{a}_UPDATE_{u:03}.json',dict(timestamp_utc=timestamp(),update=u,arm=a,training_seed=SEED,
                        ppo=metrics,trust=tr,beta_used=betas[a] if a in ('B','R') else 0.,proposal_routing=routes,
                        team_only_noise=dict(advantage_variance_veh_s_squared=float(np.var(raw,ddof=1))),
                        wall_s=time.monotonic()-start,eta_update24_hours=(24-u)*(time.monotonic()-start)/3600,
                        charged_sim_s=charged(root),reward_source='SUMO source-inclusive team J only'))
                    betas[a]=tr
                    if a=='B':
                        write(root/f'B_RANDOM_QUOTA_{u:03}.json',dict(sampling=routes['sampling'],
                            tickets=[dict(t,id=i) for i,t in enumerate(routes['durations'])]))
                        candidates=[(e,s) for e in eps for s in e['steps'] if s['time_s']<=600 and s['sample'].get('behaviour',{}).get('plan_start')]
                        for i in exact_rng.permutation(len(candidates))[:max(0,12-exact_count)]:
                            e,s=candidates[int(i)];exact_plan(root,b,policies[a],e,s,exact_count);exact_count+=1
                if u%4==0:
                    check=wave(root,b,policies,worlds('check'),'check',f'check{u:03}_'+''.join(arms),'checks')
                    panel_report(root,f'{"".join(arms)}_CHECK_{u:03}.json',check,controls,'check')
                if u%8==0:
                    dev=wave(root,b,policies,worlds('dev'),'dev',f'dev{u:03}_'+''.join(arms),'dev')
                    panel_report(root,f'{"".join(arms)}_DEV_{u:03}.json',dev,read(root/'MAPPO_DEV.json')['records'],'dev')
            if arms==('A','B'):primary=policies;final(root,b,policies)
            else:
                if not (root/'FINAL_TEST_COMPLETE.json').exists():raise ValueError('R test before primary panel')
                write(root/'R_FINAL_STARTED.json',dict(checkpoint_sha256=sha(root/'R_checkpoint_024.pt'),panel_sha256=digest(worlds('final_test'))))
                rs=wave(root,b,policies,worlds('final_test'),'final_test','finalR','random_guide_test')
                primary_results=[read(root/'jobs'/f'final_{a}_{w["seed"]}'/'RESULT.json') for a in ('A','B','mappo') for w in worlds('final_test')]
                panel_report(root,'R_FINAL_COMPLETE.json',rs+[r for r in primary_results if r['arm'] in ('A','B')],
                    [r for r in primary_results if r['arm']=='mappo'],'final_test')
        write(root/'COMPLETE.json',dict(timestamp_utc=timestamp(),charged_sim_s=charged(root)))
    except BaseException as e:
        if not (root/'STOP.json').exists():write(root/'STOP.json',dict(error=repr(e),timestamp_utc=timestamp(),charged_sim_s=charged(root)))
        raise
    finally:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--smoke-only',action='store_true');a=p.parse_args();run(a.root,a.smoke_only)
