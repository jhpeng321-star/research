"""Single owned supervisor, immutable task sources, unique full-prefix jobs."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
import math
import os
from pathlib import Path
import time
import traceback

from .protocol import (TIMES, HIGH, canonical, checked, digest, group, observable,
                       option, sha, worlds, budget, DATA_CAP)
from .storage import admission, charge, read, stop, write

ROOT = None
BINDING = None


def verify(root):
    binding = read(root/'BINDING.json')
    if sha(root/'COOPBANDIT_DESIGN.md') != binding['design_sha256']:
        raise ValueError('design binding')
    for name, expected in binding['sources'].items():
        if sha(root/'pytools'/name) != expected:
            raise ValueError('immutable source binding: '+name)
    approval = read(root/'APPROVAL.json')
    if approval['design_sha256'] != binding['design_sha256'] or not all(
        token in approval['owner_line'] for token in ('OWNER APPROVED','S-COOPBANDIT','1,600,000')):
        raise ValueError('owner approval binding')
    if binding['budget'] != budget():
        raise ValueError('registered budget')
    return binding


def init(root):
    global ROOT, BINDING
    ROOT = Path(root)
    BINDING = verify(ROOT)
    from .backend import setup
    setup(BINDING)


def fresh(ident, world, seconds, phase='data'):
    from .backend import create
    admission(ROOT)
    verify(ROOT)
    charge(ROOT,ident,seconds,phase)
    path = ROOT/'jobs'/ident
    path.mkdir()
    write(path/'START.json',dict(pid=os.getpid(),world=world,charge=seconds,unix=time.time()))
    return create(path,world,seconds,BINDING['checkpoint']), path


def screen(world, repeat=False):
    from .backend import advance, tick
    from tools.leverprobe.backend import observations
    from tools.leverprobe.selection import features
    ident = f's{world["seed"]}_'+('repeat' if repeat else 'screen')
    duration = 900 if repeat else 1140
    begin = time.monotonic()
    b, path = fresh(ident,world,duration)
    checkpoints = {}
    try:
        clocks = sorted(set(TIMES+(900,)+tuple(t+h for t in TIMES for h in (120,300,600))))
        with gzip.open(path/'observable.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as tape:
            while b.env.now <= duration:
                now = int(b.env.now)
                if now in clocks:
                    rows = observations(b.env)
                    j = TIMES.index(now) if now in TIMES else 0
                    checkpoints[str(now)] = dict(prefix=b.fingerprint(),totals=dict(b.totals),
                        features=observable(rows,now,(world['repeat']+j)%4),rows=rows,
                        summary=features(rows))
                if now == duration:
                    break
                tick(b,trace=tape)
        if repeat:
            original = read(ROOT/'jobs'/f's{world["seed"]}_screen'/'RESULT.json')
            for clock, rec in checkpoints.items():
                if canonical(rec) != canonical(original['checkpoints'][clock]):
                    raise ValueError('bitwise complete MAPPO repeat failed '+clock)
        result = dict(world=world,checkpoints=checkpoints,actual_sim_s=b.actual,
                      wall_s=time.monotonic()-begin,trace_sha256=sha(path/'observable.jsonl.gz'),repeat=repeat)
        write(path/'RESULT.json',result)
        return result
    finally:
        b.close()


def select(screens, limits):
    chosen, coverage = [], []
    for cell in range(12):
        candidates = [x for x in screens if x['world']['cell'] == cell]
        candidates.sort(key=lambda x:x['world']['seed'])
        def dense_count(x):
            if cell not in HIGH:
                return 0
            lim = next(l for l in limits if (l['penetration'],l['demand']) ==
                       (x['world']['penetration'],x['world']['demand']))
            return sum(s['summary']['ramp'] and bool(s['summary']['pair']) and
                       all(s['summary'][k] >= lim[k]['0.75'] for k in ('cav_count','density'))
                       for t,s in x['checkpoints'].items() if int(t) in TIMES)
        if cell in HIGH:
            ranked = sorted(candidates,key=lambda x:(-dense_count(x),x['world']['seed']))
            dense = ranked[:5]
            ordinary = [x for x in candidates if x not in dense][:4]
            picked = dense+ordinary
        else:
            picked = candidates[:3]
        for x in picked:
            chosen.append(dict(world=x['world'],dense_states=dense_count(x)))
        coverage.append(dict(cell=cell,selected=len(picked),dense_states=sum(dense_count(x) for x in picked)))
    if len(chosen) != 60:
        raise ValueError('complete registered panel required')
    solo = {x['world']['seed'] for c in HIGH for x in sorted(
            [x for x in chosen if x['world']['cell'] == c],key=lambda x:x['world']['seed'])[:2]}
    return dict(worlds=sorted(chosen,key=lambda x:x['world']['seed']),solo_worlds=sorted(solo),coverage=coverage)


def branch(job):
    from .backend import advance
    from tools.leverprobe.backend import observations
    world, t, option_id, solo_index = job
    setting = (world['repeat']+TIMES.index(t))%4
    baseline = read(ROOT/'jobs'/f's{world["seed"]}_screen'/'RESULT.json')
    state = baseline['checkpoints'][str(t)]
    selected = group(state['rows'])
    # Absent members are explicit no-ops, not unobserved attrition.
    solo = None if solo_index is None else selected[solo_index:solo_index+1]
    plan = option(option_id,state['rows'],setting,members=solo)
    suffix = '' if solo_index is None else f'_solo{solo_index}'
    ident = f's{world["seed"]}_t{t}_{option_id}{suffix}'
    b, path = fresh(ident,world,t+600)
    try:
        advance(b,t)
        prefix = b.fingerprint()
        if canonical(prefix) != canonical(state['prefix']):
            raise ValueError('BITWISE prefix mismatch before intervention')
        if canonical(observations(b.env)) != canonical(state['rows']):
            raise ValueError('observable roster mismatch')
        base = dict(b.totals)
        metrics, safety = {}, dict(requests=0,overrides=0)
        with gzip.open(path/'observable.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as tape:
            for h in (120,300,600):
                audit = advance(b,t+h,plan=plan,initial=state['rows'],start=t,trace=tape)
                for k in safety:
                    safety[k] += audit[k]
                metrics[str(h)] = {k:v-base[k] for k,v in b.totals.items()}
        labels = {str(h):{k:baseline['checkpoints'][str(t+h)]['totals'][k]-state['totals'][k]-v
                         for k,v in metrics[str(h)].items()} for h in (120,300,600)}
        safety['override_share'] = safety['overrides']/safety['requests'] if safety['requests'] else 0.
        result = dict(world=world,t0=t,setting=setting,plan=plan,solo_index=solo_index,
                      prefix_sha256=digest(prefix),features=state['features'],metrics=metrics,D=labels,
                      safety=safety,actual_sim_s=b.actual,trace_sha256=sha(path/'observable.jsonl.gz'))
        write(path/'RESULT.json',result)
        return ident
    finally:
        b.close()


def batch(root, jobs, workers, fn):
    begin = time.monotonic()
    results = []
    with ProcessPoolExecutor(max_workers=workers,initializer=init,initargs=(str(root),)) as pool:
        pending = [pool.submit(fn,j) for j in jobs]
        for future in as_completed(pending):
            try:
                results.append(future.result())
            except BaseException:
                stop(root,traceback.format_exc())
                for f in pending:
                    f.cancel()
                raise
    return results,dict(workers=workers,jobs=len(jobs),wall_s=time.monotonic()-begin,
                        load1=os.getloadavg()[0])


def completed(root, ident):
    """A charged unfinished attempt is a blocker, never an implicit retry."""
    path = root/'jobs'/ident
    if (path/'RESULT.json').exists() and (path/'CLOSE.json').exists():
        result, close = read(path/'RESULT.json'), read(path/'CLOSE.json')
        if result['actual_sim_s'] != close['actual_sim_s'] or close['actual_sim_s'] != close['charged_sim_s']:
            raise ValueError('completed attempt accounting mismatch '+ident)
        return result
    ledger = root/'CHARGES.jsonl'
    if path.exists() or (ledger.exists() and any(json.loads(s)['id']==ident for s in ledger.read_text().splitlines())):
        raise ValueError('unfinished charged attempt; owner disposition required '+ident)
    return None


def run(root, phase='screen', max_workers=8):
    root = checked(root)
    verify(root)
    import fcntl
    lock = (root/'supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(root/f'OWNED_{phase}_{os.getpid()}.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],unix=time.time()))
    (root/'jobs').mkdir(exist_ok=True)
    try:
        init(root)
        candidates = worlds()
        if phase == 'branches':
            jobs = read(root/'BRANCH_JOBS.json')
            jobs = [j for j in jobs if completed(root,job_id(j)) is None]
            _, timing = batch(root,jobs,max_workers,branch)
            write(root/'BRANCH_COMPLETE.json',timing)
            return
        first = completed(root,f's{candidates[0]["seed"]}_screen') or screen(candidates[0])
        repeat = completed(root,f's{candidates[0]["seed"]}_repeat') or screen(candidates[0],True)
        if not (root/'CANARY.json').exists():
            write(root/'CANARY.json',dict(seed=candidates[0]['seed'],bitwise=True,
                 wall_s=first['wall_s']+repeat['wall_s'],actual_sim_s=2040))
        screens, timings = [first], []
        remaining = []
        for world in candidates[1:]:
            record = completed(root,f's{world["seed"]}_screen')
            if record is None:
                remaining.append(world)
            else:
                screens.append(record)
        stages = sorted(set([min(n,max_workers) for n in (2,4,8,16,24)]))
        for workers in stages:
            spare = max(1,int(BINDING.get('resources',{}).get('load_max',24)-os.getloadavg()[0]))
            workers = min(workers,spare)
            n = min(workers,len(remaining)) if workers != stages[-1] else len(remaining)
            if not n:
                continue
            records, timing = batch(root,remaining[:n],workers,screen)
            screens.extend(records)
            timings.append(timing)
            remaining = remaining[n:]
        if remaining:
            records,timing=batch(root,remaining,min(max_workers,max(1,int(24-os.getloadavg()[0]))),screen)
            screens.extend(records)
            timings.append(timing)
        write(root/f'THROUGHPUT_{os.getpid()}.json',timings)
        # Remaining repeats run once per other cell, before ANY intervention.
        for cell in range(1,12):
            world=next(w for w in candidates if w['cell'] == cell)
            if completed(root,f's{world["seed"]}_repeat') is None:
                screen(world,True)
        limits = read(root/'pytools/tools/coopbandit/probe_quantiles.json')['thresholds']
        selection = select(screens,limits)
        if not (root/'SELECTED.json').exists():
            write(root/'SELECTED.json',selection)
        if phase == 'screen':
            write(root/'SCREEN_COMPLETE.json',dict(worlds=len(screens),states=300,unix=time.time()))
            return
        jobs = []
        for entry in selection['worlds']:
            world = entry['world']
            for t in TIMES:
                for oid in ('O1','O2','O3','O4'):
                    jobs.append((world,t,oid,None))
                if world['seed'] in selection['solo_worlds']:
                    jobs.extend((world,t,oid,i) for oid in ('O2','O3') for i in (0,1))
        _, timing = batch(root,jobs,max_workers,branch)
        write(root/'BRANCH_THROUGHPUT.json',timing)
        from .export import export
        export(root)
        write(root/'DATA_COMPLETE.json',dict(states=300,charged_sim_s=DATA_CAP,unix=time.time()))
    except BaseException:
        stop(root,traceback.format_exc())
        raise


def job_id(job):
    world,t,oid,solo=job
    return f's{world["seed"]}_t{t}_{oid}'+('' if solo is None else f'_solo{solo}')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('--phase',choices=('screen','branches'),default='screen')
    p.add_argument('--workers',type=int,default=8)
    args=p.parse_args()
    run(args.root,args.phase,args.workers)
