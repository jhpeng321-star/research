"""One registered repaired attempt for failed first-12 diagnostic branches only.

Successful original branches are adopted, never replayed. This separate source
does not change running training, plan search, exact grades, trust or imitation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing
import os
from pathlib import Path
import time
import traceback
import types
import numpy as np
from .accel_storage import read,write,reserve,ledger
from .protocol import digest,sha
from .warm_spool import immutable


def selected(episodes,rng,remaining):
    candidates=[(e,s) for e in episodes for s in e['steps'] if s['time_s']<=600
        and s['sample'].get('behaviour',{}).get('plan_start')]
    return [candidates[int(i)] for i in rng.permutation(len(candidates))[:max(0,remaining)]]


def branches(group):return [('joint',[]),('all',list(group))]+[(f'revert{i}',[v]) for i,v in enumerate(group)]


def valid_adoption(branch,original,t):
    if branch['world']!=original['world']:raise ValueError('diagnostic world mismatch')
    if set(branch['horizon_totals'])!={'60','120','300','600'}:raise ValueError('incomplete diagnostic horizons')
    if branch['actual_sim_s']!=t+600:raise ValueError('incomplete diagnostic episode')
    if any(branch['prefixes'].get(str(s))!=original['prefixes'][str(s)] for s in range(300,t+1)):
        raise ValueError('diagnostic adopted prefix mismatch')
    return True


def jobs(root,b,policy,ep,step,index):
    t=step['time_s'];plan=step['proposal']['plan'];group=plan['members']
    samples={str(s['time_s']):s['sample'] for s in ep['steps']}
    todo=[];adopted={};manifest=[]
    for name,revert in branches(group):
        old=root/'jobs'/f'planexact{index:03}_{name}_{ep["world"]["seed"]}'/'RESULT.json'
        if old.exists():
            original=read(old);valid_adoption(original,ep,t);adopted[name]=original
            manifest.append(dict(branch=name,adopted=True,path=str(old),sha256=sha(old)));continue
        key=f'planexactfix{index:03}_{name}_{ep["world"]["seed"]}';path=root/'jobs'/key
        assert not (path/'CLAIM.json').exists(),'no third attempt or duplicate repair'
        todo.append(dict(id=key,path=str(path),world=ep['world'],role='train',arm=name,
            weights=policy.state_dict(),runtime=b['runtime'],checkpoint=b['checkpoint'],noise_seed=0,
            deterministic=True,sim_s=t+600,stop=t+600,samples=samples,accel_bounds=b['accel_bounds'],gpu_deadline=b['gpu_deadline'],
            branch=dict(time_s=t,revert=revert,group=group,until=t+plan['duration_s'],prefixes=ep['prefixes'],mappo_background=True)))
        manifest.append(dict(branch=name,adopted=False,new_id=key))
    return todo,adopted,manifest


def repair(root,b,ep,step,index,update):
    from .accel_learner import load
    from .credit import algebra
    failure=root/f'ACCEL_EXACT_PLAN_{index:03}_FAILED.json'
    assert failure.exists() and read(failure)['error'].startswith("KeyError("),'only diagnosed departed-roster fault'
    assert not (root/f'ACCEL_EXACT_PLAN_{index:03}.json').exists(),'completed diagnostic never repeated'
    policy,_,_,payload=load(root/f'B_checkpoint_{update-1:03}.pt');assert payload['update']==update-1
    todo,results,manifest=jobs(root,b,policy,ep,step,index)
    cost=sum(j['sim_s'] for j in todo)
    spent=sum(c['sim_s'] for c in ledger(root) if c['phase']=='discovery')
    if spent+cost>b['budget']['phases']['discovery']:
        immutable(root/f'DIAGNOSTIC_REPAIR_{index:03}_BLOCKED.json',dict(spent=spent,whole_missing_group_cost=cost,
            phase_cap=b['budget']['phases']['discovery'],partial_group_spending=False));return False
    immutable(root/f'DIAGNOSTIC_REPAIR_{index:03}_PLAN.json',dict(index=index,update=update,episode=ep['id'],time_s=step['time_s'],
        old_failure_sha256=sha(failure),branches=manifest,new_sim_s=cost,failed_charges_retained=True))
    if todo:
        reserve(root,todo,'discovery')
        from .accel_plant import episode
        with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
            for j,result in zip(todo,pool.map(episode,todo)):
                valid_adoption(result,ep,step['time_s']);results[j['arm']]=result
    credits={str(h):algebra(results['joint']['horizon_totals'][str(h)]['team_delay_vs'],
        [results[f'revert{i}']['horizon_totals'][str(h)]['team_delay_vs'] for i in range(len(step['proposal']['plan']['members']))],
        results['all']['horizon_totals'][str(h)]['team_delay_vs']) for h in (60,120,300,600)}
    t=step['time_s'];proposal=step['proposal'];group=proposal['plan']['members']
    diagnostic=dict(mode='diagnostic',world_id=ep['world']['seed'],time_s=t,rows=step['rows'],group=group,
        commands=step['commands'],policy_residuals={},plan=proposal['plan'])
    if 'history' in proposal:diagnostic['history']=proposal['history']
    diagnostic['state_id']=digest(diagnostic)
    immutable(root/'requests'/(diagnostic['state_id']+'.json'),diagnostic)
    write(root/f'ACCEL_EXACT_PLAN_{index:03}.json',dict(timestamp_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        world=ep['world'],time_s=t,plan=proposal['plan'],credits=credits,predicted_saving=proposal['expected_saving'],wm_request=diagnostic,
        engineering_repair=True,repair_manifest=manifest,original_failed_charges_retained=True))
    return True


def run(root,config):
    import fcntl
    import signal
    import torch
    from . import accel_storage
    from tools.coopnative import storage
    from .accel_protocol import SUPERSEDED_CHARGE
    root=Path(root);config=read(config);source=Path(config['source_root'])
    assert config['amendment_sha256'] in (root/'AUTHORITY.md').read_text()
    assert Path(__file__).resolve()==source/'pytools/tools/coopmwm/diagnostic_repair.py'
    for name,h in config['cpu']['source_hashes'].items():assert sha(source/name)==h,name
    lock=(root/'DIAGNOSTIC_REPAIR.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGHUP,signal.SIG_IGN);torch.set_num_threads(1)
    b=read(root/'EXACT_OPERATIONAL_BINDING.json');b['budget']=read(root/'MAIN0508_REPAIR_BINDING.json')['budget']
    accel_storage._reserve=types.FunctionType(storage.reserve.__code__,dict(storage.reserve.__globals__,
        CAP=b['budget']['total']-SUPERSEDED_CHARGE,CAPS=b['budget']['phases']))
    rng=np.random.default_rng(20261101);count=0;pending=[];update=1
    try:
        while True:
            from .warm_health import assert_running
            assert_running(root,'B')
            if time.time()>b['gpu_deadline']:raise ValueError('original deadline; no extension')
            up=root/f'B_UPDATE_{update:03}.json'
            if count<12 and up.exists():
                from .warm_protocol import batch_worlds
                ws=read(root/'MAIN0508_REPAIR_BINDING.json')['repair_plan']['worlds'] if update==1 else batch_worlds(update)
                eps=[read(root/'jobs'/f'warm_u{update:03}_B_{w["seed"]}'/'RESULT.json') for w in ws]
                for ep,step in selected(eps,rng,12-count):
                    pending.append((count,update,ep,step));count+=1
                update+=1
            remaining=[]
            for index,u,ep,step in pending:
                if (root/f'ACCEL_EXACT_PLAN_{index:03}.json').exists():continue
                failure=root/f'ACCEL_EXACT_PLAN_{index:03}_FAILED.json'
                if not failure.exists():remaining.append((index,u,ep,step));continue
                if not repair(root,b,ep,step,index,u):return
            pending=remaining
            if count==12 and not pending:
                immutable(root/'DIAGNOSTIC_REPAIR_COMPLETE.json',dict(selected=count,registered_sample_unchanged=True));return
            time.sleep(10)
    except BaseException as error:
        immutable(root/'DIAGNOSTIC_REPAIR_FAILED.json',dict(error=repr(error),traceback=traceback.format_exc(),
            diagnostic_only=True,no_additional_attempt=True));raise
    finally:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('config',type=Path)
    a=p.parse_args();run(a.root,a.config)
