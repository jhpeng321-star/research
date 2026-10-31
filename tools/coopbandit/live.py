"""Fixed fresh dev test; exact restricted oracle and factual-target event credit."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import os
from pathlib import Path
import time
import traceback
from unittest.mock import patch

from .protocol import canonical, dev_worlds, observable, option, sha, digest
from .storage import read, write, stop
from . import runner

ROOT=None
POLICY=None


def initialize(root):
    global ROOT,POLICY
    ROOT=Path(root)
    runner.init(root)
    import numpy
    import sys
    sys.path.insert(0,'/home/ubuntu/coopbandit20261031_cpu1/deps')
    from .learn import load_policy
    binding=read(ROOT/'LIVE_BINDING.json')
    for name,expected in binding['sources'].items():
        if sha(ROOT/'pytools'/name)!=expected:
            raise ValueError('live scientific source changed '+name)
    if sha(ROOT/'policy.pkl')!=binding['policy_sha256']:
        raise ValueError('full-data policy hash mismatch')
    if binding['training_states']!=300 or binding['partial']:
        raise ValueError('live policy must be fitted on the complete TRAIN panel')
    POLICY=load_policy(ROOT/'policy.pkl')


def episode(job):
    from tools.leverprobe.backend import observations
    from .backend import tick
    from .learn import decide
    world,arm=job
    ident=f'dev_s{world["seed"]}_{arm}'
    previous=runner.completed(ROOT,ident)
    if previous is not None: return ident
    b,path=runner.fresh(ident,world,1200,'live')
    clocks={};decisions=[];audit=dict(requests=0,overrides=0)
    plan=None;initial=None;start=None
    try:
        baseline=None if arm=='MAPPO' else read(ROOT/'jobs'/f'dev_s{world["seed"]}_MAPPO'/'RESULT.json')
        with gzip.open(path/'observable.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as tape:
            while b.env.now<=1200:
                now=int(b.env.now)
                if now%60==0:
                    clocks[str(now)]=dict(totals=dict(b.totals),prefix=b.fingerprint())
                    if now==300 and baseline is not None and canonical(clocks[str(now)])!=canonical(baseline['clocks'][str(now)]):
                        raise ValueError('live paired bitwise trunk mismatch')
                if now==1200: break
                if (now-300)%60==0:
                    rows=observations(b.env);setting=((now-300)//60)%4
                    f=observable(rows,now,setting)
                    alternatives=[option(o,rows,setting) for o in ('O1','O2','O3','O4')]
                    available=[p['available'] for p in alternatives]
                    chosen,pred=('O0',[0.]*4) if arm=='MAPPO' else decide(POLICY,f,available)
                    plan=option(chosen,rows,setting);initial=rows;start=now
                    decisions.append(dict(time_s=now,features=f,plan=plan,initial=rows,predicted_D=pred,
                                          radius=POLICY['radius'],prefix_sha256=digest(b.fingerprint())))
                a=tick(b,plan=plan,initial=initial,start=start,trace=tape)
                for k in audit: audit[k]+=a[k]
        base=clocks['300']['totals']
        metrics={str(h):{k:v-base[k] for k,v in clocks[str(300+h)]['totals'].items()} for h in (600,900)}
        audit['override_share']=audit['overrides']/audit['requests'] if audit['requests'] else 0.
        write(path/'RESULT.json',dict(world=world,arm=arm,clocks=clocks,decisions=decisions,
              metrics=metrics,safety=audit,actual_sim_s=b.actual,trace_sha256=sha(path/'observable.jsonl.gz')))
        return ident
    finally:
        b.close()


def oracle(job):
    from .backend import advance
    from tools.leverprobe.backend import observations
    world,oid=job
    ident=f'dev_s{world["seed"]}_oracle_{oid}'
    if runner.completed(ROOT,ident) is not None: return ident
    baseline=read(ROOT/'jobs'/f'dev_s{world["seed"]}_MAPPO'/'RESULT.json')
    b,path=runner.fresh(ident,world,900,'live')
    try:
        if canonical(b.fingerprint())!=canonical(baseline['clocks']['300']['prefix']):
            raise ValueError('restricted oracle bitwise trunk mismatch')
        rows=observations(b.env);plan=option(oid,rows,0);base=dict(b.totals)
        with gzip.open(path/'observable.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as tape:
            safety=advance(b,900,plan=plan,initial=rows,start=300,trace=tape)
        metrics={k:v-base[k] for k,v in b.totals.items()}
        d=baseline['metrics']['600']['team_delay_vs']-metrics['team_delay_vs']
        write(path/'RESULT.json',dict(world=world,plan=plan,metrics=metrics,D=d,safety=safety,
              actual_sim_s=b.actual,trace_sha256=sha(path/'observable.jsonl.gz')))
        return ident
    finally:
        b.close()


def frozen_requests(actions, rows, event, now, solo):
    """Freeze factual targets, recomputing only their speed-error command in the twin."""
    present={r['vid']:r for r in rows}
    output={}
    during=event['time_s']<=now<event['time_s']+event['plan']['window']
    for a in actions:
        vid=a['vid']
        if vid not in present: continue
        if during and vid in event['plan']['members'] and vid!=solo: continue
        target=a['target_mps']
        output[vid]=dict(target_mps=target,requested_raw=(target-present[vid]['v_mps'])/5.)
    return output,'fixed_factual_targets'


def credit(job):
    from .backend import tick
    world,variant=job
    factual_path=ROOT/'jobs'/f'dev_s{world["seed"]}_LEARNED'
    factual=read(factual_path/'RESULT.json')
    event=next((d for d in factual['decisions'] if d['time_s']<=540 and d['plan']['members']),None)
    if event is None: return dict(world=world['seed'],variant=variant,skipped='no qualifying executed event')
    ident=f'dev_s{world["seed"]}_credit_{variant}'
    if completed_credit(ROOT,ident,event) is not None: return ident
    # Precharge the registered worst-case full prefix; unused reservation is not refunded.
    b,path=runner.fresh(ident,world,1140,'live')
    solo=None if variant=='reference' else event['plan']['members'][int(variant[-1])]
    base=None
    try:
        with gzip.open(factual_path/'observable.jsonl.gz','rt',encoding='utf-8') as original, \
             gzip.open(path/'observable.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as trace:
            while b.env.now<event['time_s']+600:
                row=json.loads(next(original));now=int(b.env.now)
                if row['time_s']!=now: raise ValueError('factual replay clocks differ')
                if now==event['time_s']:
                    if digest(b.fingerprint())!=event['prefix_sha256']:
                        raise ValueError('event credit bitwise trunk mismatch')
                    base=dict(b.totals)
                def replay(plan,rows,targets):
                    return frozen_requests(row['actions'],rows,event,now,solo)
                # Replace only target construction; original executor/safety/physical step are retained.
                with patch('tools.leverprobe.intervention.requests',replay):
                    tick(b,plan=dict(members=['replay'],window=900),initial=[],start=300,trace=trace)
        metrics={k:v-base[k] for k,v in b.totals.items()}
        write(path/'RESULT.json',dict(world=world,event=event,variant=variant,metrics=metrics,
              actual_sim_s=b.actual,trace_sha256=sha(path/'observable.jsonl.gz'),counterfactual='factual future targets frozen'))
        return ident
    finally:
        b.close()


def completed_credit(root, ident, event):
    """Worst-case reservation stays charged when the registered tail ends early."""
    path=Path(root)/'jobs'/ident
    if (path/'RESULT.json').exists() and (path/'CLOSE.json').exists():
        result,closed=read(path/'RESULT.json'),read(path/'CLOSE.json')
        if (closed['charged_sim_s']!=1140 or
            closed['actual_sim_s']!=event['time_s']+600 or
            result['actual_sim_s']!=closed['actual_sim_s'] or
            result['event']!=event):
            raise ValueError('credit reservation/actual/event mismatch '+ident)
        return result
    # Standard incomplete-attempt blocker remains authoritative.
    return runner.completed(Path(root),ident)


def allocation():
    jobs={}
    for w in dev_worlds():
        prefix=f'dev_s{w["seed"]}'
        jobs.update({f'{prefix}_{a}':dict(sim_s=1200,phase='live') for a in ('MAPPO','LEARNED')})
        jobs.update({f'{prefix}_oracle_{o}':dict(sim_s=900,phase='live') for o in ('O1','O2','O3','O4')})
        if w in dev_worlds()[:8]:
            jobs.update({f'{prefix}_credit_{v}':dict(sim_s=1140,phase='live') for v in ('reference','solo0','solo1')})
    assert sum(v['sim_s'] for v in jobs.values())==99360
    return jobs


def run(root,workers):
    root=Path(root)
    import fcntl
    lock=(root/'live_supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(root/f'OWNED_LIVE_{os.getpid()}.json',dict(pid=os.getpid(),unix=time.time()))
    (root/'jobs').mkdir(exist_ok=True)
    try:
        for fn,jobs in ((episode,[(w,'MAPPO') for w in dev_worlds()]),
                        (episode,[(w,'LEARNED') for w in dev_worlds()]),
                        (oracle,[(w,o) for w in dev_worlds() for o in ('O1','O2','O3','O4')]),
                        (credit,[(w,v) for w in dev_worlds()[:8] for v in ('reference','solo0','solo1')])):
            with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(str(root),)) as pool:
                results=list(pool.map(fn,jobs))
            write(root/f'PHASE_{fn.__name__}_{jobs[0][1]}_COMPLETE.json',dict(results=results))
        write(root/'LIVE_COMPLETE.json',dict(worlds=12,unix=time.time()))
    except BaseException:
        stop(root,traceback.format_exc());raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();run(a.root,a.workers)
