"""Two planner arms, reused baselines, all-executed-group exact credit."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
import gzip
import json
import multiprocessing
import os
from pathlib import Path
import time
import traceback

from tools.coopbandit.protocol import canonical, digest, sha
from tools.coopplan.storage import read, write, ledger_lock, stop
from tools.coopplan.predictor import ObservableState
from tools.coopplan.planner import TIMES
from tools.wmplan.design import ARMS, CREDIT, H_CREDIT, PANEL_SHA, panel, budget, admit


def verify(root):
    b = read(root/'BINDING.json')
    a = read(root/'APPROVAL.json')['line']
    if 'S-WMPLAN BUDGET APPROVED' not in a or '200,000' not in a:
        raise PermissionError('owner approval')
    if b['budget'] != budget() or sha(root/'WMPLAN_DESIGN.md') != b['design_sha256']:
        raise ValueError('frozen budget/design')
    if b['panel'] != panel() or digest(panel()) != PANEL_SHA:
        raise ValueError('same24panel')
    for name,expected in {**b['sources'],**b['artifacts'],**b['external_sources']}.items():
        p = Path(name) if name.startswith('/') else root/name
        if sha(p) != expected:
            raise ValueError('immutable source/artifact: '+name)
    if b['model']['selected'] not in ('FULL-v4','FULL-v5'):
        raise ValueError('preregistered model')
    if 'restart' in b:
        r = b['restart']
        if (r['prior_charged_sim_s'],r['prior_actual_sim_s'],r['new_run_upper_sim_s'],
                r['combined_charged_upper_sim_s'],r['combined_actual_upper_sim_s']) != (3600,900,195840,199440,196740):
            raise ValueError('approved restart accounting')
        if (b['model']['selected'] != 'FULL-v4' or
                '15:55' not in read(root/'RESTART_APPROVAL.json')['line']):
            raise ValueError('approved engineering restart only')
    return b


def charge(root, ident, seconds, world, kind):
    binding = verify(root)
    if world not in panel() or kind not in ('paired','credit'):
        raise ValueError('registered world/kind')
    arm = ident.split('_')[1]
    if arm not in ARMS:
        raise ValueError('registered arm')
    with ledger_lock(root):
        p = root/'CHARGES.jsonl'
        ledger = [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
        if any(r['id'] == ident for r in ledger) or (root/'jobs'/ident).exists():
            raise ValueError('duplicate attempt; no retry/refund')
        if sum(r['sim_s'] for r in ledger)+seconds > budget()['total']:
            raise ValueError('global cap')
        if 'restart' in binding and sum(r['sim_s'] for r in ledger)+seconds > binding['restart']['new_run_upper_sim_s']:
            raise ValueError('restart combined charge cap, no prior refund')
        prior = [r for r in ledger if (r['seed'],r['arm'],r['kind']) == (world['seed'],arm,kind)]
        if sum(r['sim_s'] for r in prior)+seconds > (CREDIT if kind == 'credit' else 1200):
            raise ValueError('disjoint world-arm quota')
        if kind == 'credit':
            key,suffix = ident.split('_O',1)
            reservation = read(root/'reservations'/f'{key}.json')
            action = reservation['action']
            if ('O'+suffix not in ('O0',action,action+'_solo0',action+'_solo1') or
                    reservation['world'] != world or reservation['cost'] != 4*seconds):
                raise ValueError('complete identity-bound quartet reservation')
        row = dict(id=ident,sim_s=seconds,seed=world['seed'],arm=arm,kind=kind,pid=os.getpid())
        with p.open('a',encoding='utf-8') as f:
            f.write(canonical(row)+'\n');f.flush();os.fsync(f.fileno())
        path = root/'jobs'/ident
        path.mkdir()
        return path


def configure():
    from tools.distill import live as p
    p.verify = verify
    p.charge = charge
    p.HORIZON = H_CREDIT
    return p


def setup(root):
    return configure().setup(root)


def credit(root,key,action):
    setup(root)
    return configure().credit(root,key,action)


def episode(root,world,arm):
    from tools.coopplan.plant import tick
    from tools.leverprobe.backend import observations
    from tools.wmplan.predict import OnlinePlanner
    binding = setup(root)
    p = configure()
    seed = world['seed']
    baseline = read(Path(binding['baseline_root'])/'jobs'/f's{seed}_mappo'/'RESULT.json')
    b,path = p.fresh(root,f's{seed}_{arm}',1200,world,'paired')
    history,clocks,credits = [],{},[]
    planner = OnlinePlanner(root,seed,arm)
    spent = requests = overrides = 0
    try:
        if canonical(baseline['warmup']) != canonical(b.fingerprint()):
            raise ValueError('reused baseline warmup BITWISE mismatch')
        with ProcessPoolExecutor(1,mp_context=multiprocessing.get_context('spawn')) as pool:
            with gzip.open(path/'observable.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as tape:
                while b.env.now < 1200:
                    t = int(b.env.now)
                    if t == 900:
                        clocks['600'] = dict(b.totals)
                    if t in TIMES:
                        rows = observations(b.env)
                        key = f's{seed}_{arm}_t{t}'
                        event = planner.step(ObservableState(t,tuple(rows),key))
                        # Same event-to-executor ABI as coopplan.runner. The
                        # frozen plant anchors option targets to this roster.
                        event['initial_rows'] = rows
                        cost = admit(event,spent)
                        if cost:
                            chosen = event['executed_action']
                            if len(chosen['members']) != 2:
                                raise ValueError('two-member inherited group')
                            write(root/'reservations'/f'{key}.json',dict(cost=cost,world=world,
                                  time_s=t,action=chosen['id']))
                            write(root/'states'/f'{key}.json',dict(world=world,time_s=t,rows=rows,
                                  history=deepcopy(history),prefix=b.fingerprint()))
                            actions = [dict(id='O0',members=[]),chosen]
                            for i,member in enumerate(chosen['members']):
                                solo = deepcopy(chosen)
                                solo.update(id=chosen['id']+f'_solo{i}',members=[member])
                                actions.append(solo)
                            values = [pool.submit(credit,root,key,a).result() for a in actions]
                            dg = values[0]-values[1]
                            solos = [values[0]-v for v in values[2:]]
                            credits.append(dict(time_s=t,action=chosen,D_G=dg,member_alone_D=solos,
                                                C_G=dg-sum(solos),horizon_s=H_CREDIT))
                            spent += cost
                        history.append(event)
                        write(path/f'DECISION_{t}.json',event)
                    audit = tick(b,history[-1] if history else None,tape)
                    requests += audit['requests'];overrides += audit['overrides']
            clocks['900'] = dict(b.totals)
        result = dict(world=world,arm=arm,clocks=clocks,history=history,credits=credits,
                      requests=requests,overrides=overrides,credit_sim_s=spent,
                      actual_sim_s=b.actual,final_prefix=b.fingerprint(),warmup_pass=True)
        if all(e['executed_action']['id'] == 'O0' for e in history):
            if canonical(baseline['final_prefix']) != canonical(result['final_prefix']):
                raise ValueError('all-O0 planner BITWISE mismatch against reused MAPPO')
            result['all_O0_bitwise_pass'] = True
        write(path/'RESULT.json',result)
        return dict(seed=seed,arm=arm)
    except BaseException:
        # Stop before this failed Future reaches the outer episode supervisor;
        # its pool shutdown can otherwise wait for queued worlds to run.
        stop(root,traceback.format_exc())
        raise
    finally:
        b.close()


def run(root):
    import fcntl
    binding = setup(root)
    if not read(root/'CPU.json')['pass_']:
        raise ValueError('host CPU gate')
    with (root/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(root/'DISPATCH_ONCE.json',dict(pid=os.getpid(),unix=time.time(),budget=budget()))
        try:
            while not all((root/f'GPU_READY_{i}.json').is_file() for i in range(binding['gpu_workers'])):
                if (root/'STOP.json').exists():
                    raise RuntimeError('GPU startup fault before physics')
                time.sleep(1)
            with ProcessPoolExecutor(binding['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
                smoke_world = None
                if 'restart' in binding:
                    smoke_world = panel()[0]
                    result = pool.submit(guarded_episode,root,smoke_world,'expected').result()
                    write(root/'SMOKE_COMPLETE.json',dict(**result,unix=time.time(),
                          result_sha256=sha(root/'jobs'/f's{smoke_world["seed"]}_expected'/'RESULT.json')))
                    while not (root/'SMOKE_APPROVED.json').is_file():
                        if (root/'STOP.json').exists():
                            raise RuntimeError('STOP at smoke gate; no retry')
                        time.sleep(2)
                for arm in ARMS:
                    # Owner14:48: publish the entire primary arm before starting
                    # any secondary episode. No outcome controls this ordering.
                    worlds = [w for w in panel() if not (arm == 'expected' and w == smoke_world)]
                    run_panel(pool,root,worlds,arm,binding['workers'])
                    write(root/f'ARM_COMPLETE_{arm}.json',dict(arm=arm,worlds=24,unix=time.time()))
                    if arm == 'expected':
                        while not (root/'PRIMARY_REPORTED.json').is_file():
                            if (root/'STOP.json').exists():
                                raise RuntimeError('STOP while waiting for primary report')
                            time.sleep(2)
            write(root/'COMPLETE.json',dict(worlds=24,arms=list(ARMS),unix=time.time()))
        except BaseException:
            stop(root,traceback.format_exc())
            raise


def guarded_episode(root,world,arm):
    """Include setup/warmup constructor faults in immediate durable STOP."""
    try:
        if (root/'STOP.json').exists():
            raise RuntimeError('STOP before episode admission')
        return episode(root,world,arm)
    except BaseException:
        stop(root,traceback.format_exc())
        raise


def run_panel(pool,root,worlds,arm,workers):
    """Bound admission to active slots; no queued panel after a failed Future."""
    remaining = iter(worlds)
    active = set()
    def submit():
        if (root/'STOP.json').exists():
            raise RuntimeError('STOP before next episode; no retry')
        world = next(remaining,None)
        if world is not None:
            active.add(pool.submit(guarded_episode,root,world,arm))
    for _ in range(workers):
        submit()
    while active:
        done,_ = wait(active,return_when=FIRST_COMPLETED)
        # Resolve all completed Futures before admitting another episode.
        results = [f.result() for f in done]
        active.difference_update(done)
        for result in results:
            print(json.dumps(dict(event='EPISODE_COMPLETE',**result)),flush=True)
            submit()


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    run(p.parse_args().root)
