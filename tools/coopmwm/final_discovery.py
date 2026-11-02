"""Detector-first final group twins, only after the registered outcome-opening gate."""
import argparse
import math
import os
from pathlib import Path
import time
from .accel_storage import read,write,ledger,charged
from .protocol import sha
from .warm_protocol import budget,worlds,install_accounting

DURATIONS=(30,60,90,120,150,180,240,300)


def group_at(event,rows):
    from .mppi_search import group_components
    mentioned=set(event.get('group',event.get('members',[])))
    if not mentioned:return None
    # The existing observable component rule supplies any additional neighbours;
    # no new behavioural detector or outcome-based group selection is introduced.
    return next((g for g in group_components(rows) if mentioned<=set(g['members'])),None)


def observed_plan(episode,event,index):
    t=math.ceil(float(event.get('start_s',event.get('time_s',-1))))
    if not 300<=t<=600:return None
    source=next((s for s in episode['steps'] if s['time_s']==t),None)
    if source is None:return None
    group=group_at(event,source['rows'])
    if group is None:return None
    duration=next((d for d in DURATIONS if d>=max(1,float(event.get('end_s',t+60))-t)),300)
    plan=dict(id=f'detected{index}',members=group['members'],duration_s=duration,
              origin='detector-observed-final-policy',shape='recorded per-second raw acceleration; no inherited family seed')
    return t,plan


def twin_episode(episode,t,plan):
    import copy
    twin=copy.deepcopy(episode)
    for s in twin['steps']:
        if t<=s['time_s']<t+plan['duration_s']:
            s['sample']['absolute_accels']={v:s['requests'].get(v,{}).get('requested_raw',s['commands'][v])
                for v in plan['members'] if v in s['commands']}
    step=next(s for s in twin['steps'] if s['time_s']==t)
    step['proposal']=dict(plan=plan,expected_saving=None)
    return twin,step


def whole_group_cost(t,n):
    assert 2<=n<=6 and 300<=t<=600
    return (n+2)*(t+600)


def opening_ready(root):
    from .final_runtime import endpoints_ready
    return endpoints_ready(root) and (root/'FINAL_OUTCOMES_OPENED.json').exists() and (root/'FINAL_REGISTERED_EVALUATION_COMPLETE.json').exists()


def run(root,config):
    import fcntl
    lock=(root/'FINAL_DISCOVERY.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (root/'FINAL_DISCOVERY_STARTED.json').exists(),'one-use discovery already started'
    for name,h in config['source_hashes'].items():assert sha(Path(config['source_root'])/name)==h,name
    assert config['amendment_sha256'] in (root/'AUTHORITY.md').read_text()
    os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
    install_accounting()
    try:
        # No detector input/result is opened on the waiting path.
        while not opening_ready(root):
            if (root/'FINAL_DISCOVERY_STOP.json').exists():raise ValueError('durable discovery STOP')
            time.sleep(30)
        from .accel_learner import load
        from .accel_runner import exact_plan
        from .final_runtime import idle_admission,RAM
        from tools.coopfree.analysis import tape
        from tools.coopbehav.detectors import detect
        endpoint=budget()['updates']['B']
        write(root/'FINAL_DISCOVERY_STARTED.json',dict(unix=time.time(),update=endpoint,checkpoint_sha256=sha(root/f'B_checkpoint_{endpoint:03}.pt')))
        policy,_,_,payload=load(root/f'B_checkpoint_{endpoint:03}.pt');assert payload['update']==endpoint
        b=read(root/'WARM_BINDING.json');index=12;coverage=[];manifest=[]
        from .resource_config import clocks
        b=clocks(root,b)
        for w in worlds('final_test'):
            ident=f'fixedfinal_B_{w["seed"]}';path=root/'jobs'/ident
            events=detect(tape(path/'observable.jsonl.gz',w['seed']))['events']
            manifest.append(dict(id=ident,world=w,events=events))
            ep=None;used=set()
            for event in sorted(events,key=lambda e:(float(e.get('start_s',e.get('time_s',-1))),str(e.get('group',e.get('members',[]))))):
                if ep is None:ep=read(path/'RESULT.json')
                chosen=observed_plan(ep,event,index)
                if chosen is None:continue
                t,plan=chosen;key=(t,tuple(plan['members']))
                if key in used:continue
                used.add(key);cost=whole_group_cost(t,len(plan['members']))
                spent=sum(x['sim_s'] for x in ledger(root) if x['phase']=='discovery')
                if spent+cost>budget()['phases']['discovery']:
                    coverage.append(dict(id=ident,time_s=t,members=plan['members'],covered=False,
                                         reason='insufficient remaining reserve for complete n+2 group',sim_s=cost));continue
                # Reuse the exact immutable n+2 executor. New paths only use the
                # existing owned RAM namespace; never move/delete previous jobs.
                for name in ['joint','all']+[f'revert{i}' for i in range(len(plan['members']))]:
                    jobid=f'planexact{index:03}_{name}_{w["seed"]}'
                    p=root/'jobs'/jobid;q=RAM/jobid
                    assert not p.exists() and not p.is_symlink() and not q.exists()
                    q.mkdir();p.symlink_to(q,target_is_directory=True)
                while not idle_admission(root):time.sleep(10)
                twin,step=twin_episode(ep,t,plan)
                exact_plan(root,b,policy,twin,step,index,role='final_test')
                coverage.append(dict(id=ident,time_s=t,members=plan['members'],covered=True,
                                     sim_s=cost,receipt=f'ACCEL_EXACT_PLAN_{index:03}.json'))
                index+=1
        write(root/'FINAL_DISCOVERY_DETECTORS.json',manifest)
        write(root/'FINAL_DISCOVERY_COMPLETE.json',dict(coverage=coverage,charged_sim_s=charged(root),
            group_rule='unchanged observable100m connected components2..6 containing detector members',
            duration_rule='round measured detector duration up to registered duration grid, cap300s',
            sampling='registered panel order then time/group, deduplicated; complete n+2 only; no outcome-based selection'))
    except BaseException as error:
        write(root/'FINAL_DISCOVERY_FAILED.json',dict(unix=time.time(),error_type=type(error).__name__,physics_retry=False))
        raise
    finally:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('config',type=Path)
    a=p.parse_args();run(a.root,read(a.config))
