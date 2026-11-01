"""Fresh-panel paired live test with all-event exact credit and durable precharge."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import gzip
import json
import multiprocessing
import os
from pathlib import Path
import time
import traceback

from tools.coopbandit.protocol import CELLS, canonical, checked, sha
from tools.coopplan.storage import read, write, admission, ledger_lock, stop
from tools.coopplan.predictor import ObservableState
from tools.coopplan.planner import TIMES
from .portable import Student

CAP = 200000
WORLD_CREDIT = 5850
HORIZON = 120


def panel():
    # High share and high demand first, exact two repeats in every cell.
    return [dict(seed=984000+i,cell=CELLS.index((p,d)),repeat=r,penetration=p,demand=d)
            for i,(p,d,r) in enumerate((p,d,r) for p in (.4,.2,.1,.05)
                                      for d in (2700,2400,1900) for r in range(2))]


def budget():
    return dict(paired=24*2400,all_event_credit=24*WORLD_CREDIT,mechanics_extra=1200,
                total=199200,unallocated=800,cap=CAP)


def approval(text):
    lines = [s for s in text.splitlines() if 'S-DISTILL BUDGET APPROVED' in s
             and ('200,000' in s or '200000' in s)
             and not any(w in s.lower() for w in ('awaiting','not approved','pending'))]
    if not lines:
        raise PermissionError('exact MAIN S-DISTILL BUDGET APPROVED/200000 required')
    return lines[-1]


def guard(world):
    if world not in panel() or type(world['seed']) is not int:
        raise ValueError('exact registered fresh DEV world/cell required')
    from traffic_shepherd.seeds import require_development_case
    from traffic_shepherd.c4.seeds import check_case
    require_development_case(world['seed'],world['demand'],'exposed')
    check_case(world['seed'],world['demand'])


def scope():
    from traffic_shepherd import seeds as s
    from traffic_shepherd.c4 import seeds as c
    allowed = {(w['seed'],w['demand']) for w in panel()}
    original_s,original_c = s.require_development_case,c.check_case
    def base(seed,demand,role):
        if type(seed) is int and type(demand) is int and role == 'exposed' and (seed,demand) in allowed:
            return
        return original_s(seed,demand,role)
    def c4(seed,demand):
        if type(seed) is int and type(demand) is int and (seed,demand) in allowed:
            return base(seed,demand,'exposed')
        return original_c(seed,demand)
    s.require_development_case,c.check_case = base,c4


def verify(root):
    b = read(root/'BINDING.json')
    a = read(root/'APPROVAL.json')
    approval(a['line'])
    if b['budget'] != budget() or b['design_sha256'] != sha(root/'DISTILL_LIVE_DESIGN.md'):
        raise ValueError('design/budget mismatch')
    for name,digest in {**b['sources'],**b['artifacts']}.items():
        if sha(root/name) != digest:
            raise ValueError('immutable input/source changed: '+name)
    if b['panel'] != panel() or read(root/'PANEL.json') != panel():
        raise ValueError('fresh panel identity')
    return b


def setup(root):
    from tools.leverprobe import backend as probe
    b = verify(root)
    probe.setup(b['runtime'],b['checkpoint'])
    scope()
    probe.guard = guard
    from tools.coopbandit.metric_compat import install
    install()
    # Reuse only unchanged executor/replay, never the old campaign's setup/ledger.
    return b


def charge(root,ident,seconds,world,kind):
    verify(root)
    with ledger_lock(root):
        p = root/'CHARGES.jsonl'
        ledger = [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
        if any(r['id'] == ident for r in ledger) or (root/'jobs'/ident).exists():
            raise ValueError('duplicate attempt; no automatic retry/refund')
        if world not in panel():
            raise ValueError('unregistered world')
        if sum(r['sim_s'] for r in ledger)+seconds > budget()['total']:
            raise ValueError('global charge cap')
        prior = [r for r in ledger if r['seed'] == world['seed'] and r['kind'] == kind]
        cap = WORLD_CREDIT if kind == 'credit' else 2400 if kind == 'paired' else 1200
        if kind not in ('credit','paired','mechanics') or sum(r['sim_s'] for r in prior)+seconds > cap:
            raise ValueError('disjoint per-world quota')
        if kind == 'mechanics' and world != panel()[0]:
            raise ValueError('one original canary only')
        if kind == 'credit':
            key, suffix = ident.split('_O',1)
            reservation = read(root/'reservations'/f'{key}.json')
            action = reservation['action']
            allowed = ['O0',action,action+'_solo0',action+'_solo1']
            if ('O'+suffix not in allowed or reservation['world'] != world or
                    reservation['cost'] != 4*seconds):
                raise ValueError('complete identity-bound credit reservation required')
        with p.open('a',encoding='utf-8') as f:
            f.write(canonical(dict(id=ident,sim_s=seconds,seed=world['seed'],kind=kind,pid=os.getpid()))+'\n')
            f.flush();os.fsync(f.fileno())
        path = root/'jobs'/ident
        path.mkdir()
        return path


def fresh(root,ident,seconds,world,kind):
    from tools.coopbandit.backend import create
    admission(root)
    b = verify(root)
    path = charge(root,ident,seconds,world,kind)
    return create(path,world,seconds,b['checkpoint']),path


def episode_baseline(root,world,ident,kind='paired',scripted=False):
    from tools.coopplan.plant import tick
    from tools.coopbandit.protocol import option
    b,path = fresh(root,ident,1200,world,kind)
    clocks = {}; warmup = b.fingerprint()
    try:
        with gzip.open(path/'observable.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as tape:
            while b.env.now < 1200:
                t = int(b.env.now)
                if t in (900,1200):
                    clocks[str(t-300)] = dict(b.totals)
                event = dict(time_s=t,executed_action=option('O0',[],0),initial_rows=[]) if scripted else None
                tick(b,event,tape)
        clocks['900'] = dict(b.totals)
        result = dict(world=world,clocks=clocks,warmup=warmup,final_prefix=b.fingerprint(),actual_sim_s=b.actual)
        write(path/'RESULT.json',result)
        return result
    finally:
        b.close()


def credit(root,key,action):
    from tools.coopplan.plant import replay
    from tools.leverprobe.backend import observations
    state = read(root/'states'/f'{key}.json')
    t = state['time_s']; world = state['world']
    b,path = fresh(root,key+'_'+action['id'],t+HORIZON,world,'credit')
    try:
        replay(b,state['history'],t)
        if canonical(b.fingerprint()) != canonical(state['prefix']):
            raise ValueError('BITWISE executed-history prefix mismatch')
        if canonical(observations(b.env)) != canonical(state['rows']):
            raise ValueError('observable prefix mismatch')
        before = dict(b.totals)
        event = dict(time_s=t,executed_action=action,initial_rows=state['rows'])
        with gzip.open(path/'observable.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as tape:
            replay(b,[event],t+HORIZON,tape)
        metrics = {k:v-before[k] for k,v in b.totals.items()}
        write(path/'RESULT.json',dict(metrics=metrics,prefix_pass=True,actual_sim_s=b.actual,action=action))
        return metrics['team_delay_vs']
    finally:
        b.close()


def admit_event(event,spent):
    cost = 4*(event['time_s']+HORIZON)
    proposed = deepcopy(event['executed_action'])
    event['proposed_action'] = proposed
    if proposed['id'] != 'O0' and spent+cost > WORLD_CREDIT:
        event.update(executed_action=dict(id='O0',members=[]),group=[],budget_pruned=True)
        return 0
    event['budget_pruned'] = False
    return cost if proposed['id'] != 'O0' else 0


def episode(root,world):
    from tools.coopplan.plant import tick
    from tools.leverprobe.backend import observations
    setup(root)
    seed = world['seed']
    basepath = root/'jobs'/f's{seed}_mappo'/'RESULT.json'
    # Only the first-world baseline is reused, from the completed mechanics pair.
    base = read(basepath) if world == panel()[0] else episode_baseline(root,world,f's{seed}_mappo')
    b,path = fresh(root,f's{seed}_student',1200,world,'paired')
    history,clocks,credits = [],{},[]
    if canonical(base['warmup']) != canonical(b.fingerprint()):
        b.close();raise ValueError('paired warmup prefix mismatch')
    student = Student(read(root/'POLICY.json'))
    spent = 0; requests = overrides = 0
    try:
        with ProcessPoolExecutor(1,mp_context=multiprocessing.get_context('spawn'),
                                 initializer=setup,initargs=(root,)) as pool:
            with gzip.open(path/'observable.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as tape:
                while b.env.now < 1200:
                    t = int(b.env.now)
                    if t == 900:
                        clocks['600'] = dict(b.totals)
                    if t in TIMES:
                        rows = observations(b.env); key = f's{seed}_t{t}'
                        state = ObservableState(t,tuple(rows),key)
                        event = student.step(state)
                        cost = admit_event(event,spent)
                        if cost:
                            # Durable complete four-branch reservation BEFORE the move.
                            write(root/'reservations'/f'{key}.json',dict(cost=cost,world=world,time_s=t,
                                  action=event['executed_action']['id']))
                            write(root/'states'/f'{key}.json',dict(world=world,time_s=t,rows=rows,
                                  history=deepcopy(history),prefix=b.fingerprint()))
                            chosen = event['executed_action']; actions = [dict(id='O0',members=[]),chosen]
                            for i,member in enumerate(chosen['members']):
                                solo = deepcopy(chosen);solo.update(id=chosen['id']+f'_solo{i}',members=[member])
                                actions.append(solo)
                            values = [pool.submit(credit,root,key,a).result() for a in actions]
                            dg = values[0]-values[1]
                            ds = [values[0]-v for v in values[2:]]
                            credits.append(dict(time_s=t,action=chosen,D_G=dg,member_alone_D=ds,
                                                C_G=dg-sum(ds),horizon_s=HORIZON))
                            spent += cost
                        history.append(event)
                        write(path/f'DECISION_{t}.json',event)
                    audit = tick(b,history[-1] if history else None,tape)
                    requests += audit['requests']; overrides += audit['overrides']
            clocks['900'] = dict(b.totals)
        result = dict(world=world,clocks=clocks,history=history,credits=credits,
                      requests=requests,overrides=overrides,credit_sim_s=spent,
                      actual_sim_s=b.actual,final_prefix=b.fingerprint(),warmup_pass=True)
        if all(e['executed_action']['id'] == 'O0' for e in history):
            if canonical(base['final_prefix']) != canonical(result['final_prefix']):
                raise ValueError('all-O0 student not BITWISE identical to MAPPO')
            result['all_O0_bitwise_pass'] = True
        write(path/'RESULT.json',result)
        return seed
    finally:
        b.close()


def run(root,workers):
    import fcntl
    root = checked(root)
    binding = setup(root)
    if workers < 1 or workers > binding['workers']:
        raise ValueError('worker ceiling')
    if not read(root/'CPU.json')['pass_'] or not read(root/'EXPOSURE.json')['pass_']:
        raise ValueError('CPU/exposure gates')
    with (root/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(root/'DISPATCH_ONCE.json',dict(pid=os.getpid(),unix=time.time()))
        try:
            first = panel()[0]
            base = episode_baseline(root,first,f's{first["seed"]}_mappo')
            repeat = episode_baseline(root,first,'mechanics_all_O0',kind='mechanics',scripted=True)
            if canonical(base['final_prefix']) != canonical(repeat['final_prefix']):
                raise ValueError('mechanics all-O0 BITWISE mismatch')
            write(root/'MECHANICS.json',dict(pass_=True,actual_sim_s=2400,reused_first_baseline=True))
            with ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                try:
                    for f in as_completed([pool.submit(episode,root,w) for w in panel()]):
                        f.result()
                except BaseException:
                    stop(root,traceback.format_exc())
                    raise
            write(root/'COMPLETE.json',dict(worlds=24,unix=time.time()))
        except BaseException:
            stop(root,traceback.format_exc())
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root',type=Path);p.add_argument('--workers',type=int,default=2)
    a = p.parse_args();run(a.root,a.workers)
