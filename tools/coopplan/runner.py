"""One paired world per worker; oracle continuations use an isolated CPU process."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import gzip
import multiprocessing
import os
from pathlib import Path
import time
import traceback

from .predictor import ObservableState, SumoOraclePredictor, Prediction
from .planner import Planner, TIMES, HORIZON, cooperation_credit
from .protocol import worlds, digest, canonical
from .storage import read, write, verify, stop
from . import plant


def learned_proposer(binding):
    if not binding.get('free_checkpoint'):
        return None
    import torch
    from tools.coopfree.policy import SetPointPolicy, encode, decode
    from tools.coopbandit.protocol import group
    model = SetPointPolicy()
    payload = torch.load(binding['free_checkpoint'],map_location='cpu',weights_only=True)
    model.load_state_dict(payload['policy'])
    model.eval()
    def propose(state):
        context = encode(state.rows,state.commands,members=group(state.rows))
        if len(context['members']) != 2:
            return None
        sample = model.sample(context,None,deterministic=True)
        return dict(id='FREE',members=context['members'],window=sample['window'],targets=decode(sample))
    return propose


def baseline(root,world,ident):
    b,path = plant.fresh(root,ident,world,1200)
    try:
        clocks = {}
        with gzip.open(path/'observable.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as tape:
            for until in (900,1200):
                plant.replay(b,[],until,tape)
                clocks[str(until-300)] = dict(b.totals)
        write(path/'RESULT.json',dict(world=world,clocks=clocks,actual_sim_s=b.actual,
                                      final_prefix=b.fingerprint()))
        if world == worlds()[0]:
            reference = read(root/'MECHANICS.json')['baseline_reference']
            if canonical(dict(totals=b.totals,prefix=b.fingerprint())) != canonical(reference):
                raise ValueError('complete 1200s MAPPO mechanics repeat not bitwise identical')
        return clocks
    finally:
        b.close()


def episode(root,world):
    plant.setup(root)
    seed = world['seed']
    baseline(root,world,f's{seed}_mappo')
    b,path = plant.fresh(root,f's{seed}_planner',world,1200)
    history,clocks,credits = [],{},[]
    ctx = multiprocessing.get_context('spawn')
    try:
        with ProcessPoolExecutor(1,mp_context=ctx,initializer=plant.setup,initargs=(root,)) as oracle:
            predictor = SumoOraclePredictor(plant.RerunClient(root,oracle))
            planner = Planner(predictor,learned=learned_proposer(plant.BINDING))
            if plant.BINDING.get('phase') == 'extension':
                from .quota import reserve_actions
                planner.admit = lambda state,actions:reserve_actions(root,state,actions)
            with gzip.open(path/'observable.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as tape:
                while b.env.now < 1200:
                    t = int(b.env.now)
                    if t-300 in (600,900):
                        clocks[str(t-300)] = dict(b.totals)
                    if t in TIMES:
                        from tools.leverprobe.backend import observations
                        rows = observations(b.env)
                        key = f's{seed}_t{t}'
                        snapshot = dict(world=world,time_s=t,rows=rows,history=deepcopy(history),prefix=b.fingerprint())
                        write(root/'states'/f'{key}.json',snapshot)
                        # Current MAPPO command anchors are needed only by an explicitly bound FREE model.
                        commands = plant.preview_commands(b) if planner.learned is not None else {}
                        state = ObservableState(t,tuple(rows),key,commands=commands)
                        event,prediction = planner.step(state)
                        event['initial_rows'] = rows
                        history.append(event)
                        if event['group'] and (not credits or plant.BINDING.get('all_credits')):
                            chosen = event['executed_action']
                            solos = []
                            for i,member in enumerate(chosen['members']):
                                action = deepcopy(chosen)
                                action.update(id=chosen['id']+f'_solo{i}',members=[member])
                                solos.append(action)
                            single = predictor.predict(state,solos,HORIZON,1)
                            combined = Prediction({**prediction.actions,**single.actions},True,'SUMO-ORACLE')
                            cg = cooperation_credit(combined,chosen['id'],[a['id'] for a in solos])
                            credits.append(dict(time_s=t,action=chosen,members=event['group'],C_G=list(cg.values())[0],
                                                horizon_s=HORIZON,semantics='all_executed_groups' if plant.BINDING.get('all_credits') else 'first_executed_group_per_world'))
                        elif plant.BINDING.get('phase') == 'extension' and prediction is not None:
                            from .quota import release_credit
                            release_credit(root,key+'_CG')
                        write(path/f'DECISION_{t}.json',event)
                    plant.tick(b,history[-1] if history else None,tape)
                clocks['900'] = dict(b.totals)
            result = dict(world=world,clocks=clocks,history=history,credits=credits,actual_sim_s=b.actual)
            write(path/'RESULT.json',result)
            return seed
    except BaseException:
        stop(root,traceback.format_exc())
        raise
    finally:
        b.close()


def run(root,workers):
    import fcntl
    root = Path(root)
    binding = verify(root)
    if not read(root/'MECHANICS.json')['pass_']:
        raise ValueError('global pre-registered mechanics gate required')
    if workers > binding['workers'] or workers < 1:
        raise ValueError('host worker cap')
    lock = (root/'supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(root/'DISPATCH_ONCE.json',dict(pid=os.getpid(),unix=time.time()))
    panel = read(root/'ALLOCATION.json')['worlds']
    from .protocol import extension_worlds
    assert all(w in (extension_worlds() if binding.get('phase') == 'extension' else worlds()) for w in panel)
    try:
        if binding.get('phase') == 'extension':
            from .extension import backfill
            backfill(root)
        ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(workers,mp_context=ctx) as pool:
            futures = [pool.submit(episode,root,w) for w in panel]
            for future in as_completed(futures):
                future.result()
        write(root/'COMPLETE.json',dict(worlds=len(panel),unix=time.time()))
    except BaseException:
        stop(root,traceback.format_exc())
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('--workers',type=int,default=1)
    a = p.parse_args()
    run(a.root,a.workers)
