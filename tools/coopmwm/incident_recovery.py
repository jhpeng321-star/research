"""MAIN09:58/10:11 recovery; immutable failed attempts are never reused."""
import argparse
import copy
import json
from pathlib import Path
import types
from .protocol import digest, sha


def recovery_budget(old):
    b=copy.deepcopy(old)
    changes=dict(train=25200-691200,discovery=-21000,
        exact_plan_grading=-240000,final_test=115200,paired_branch=345600)
    for k,v in changes.items():b['phases'][k]+=v
    b['allocated']=sum(b['phases'].values());b['unallocated']=b['total']-b['allocated']
    assert b['total']==7500000 and b['unallocated']>=0
    assert b['phases']['discovery']>=120000
    b.update(updates=dict(A=48,B=24,R=24),
        final_checkpoint_rule='primary A24/B24/R24; secondary A48; fixed, no outcome selection')
    return b


def episode_plan(original,train,jobs,arm,u):
    """Keep closed episodes and never-claimed precharges; replace only interrupted."""
    used={w['seed'] for w in original};result=[];replaced=[];adopt=[]
    for w in original:
        ident=f'warm_u{u:03}_{arm}_{w["seed"]}';j=jobs[ident]
        if j['closed'] or not j['claimed']:
            result.append(w)
            if not j['claimed']:adopt.append(ident)
            continue
        options=sorted((v for v in train if v['cell']==w['cell'] and v['seed'] not in used),key=lambda v:v['seed'])
        if not options:raise ValueError('no unused registered same-cell replacement')
        v=options[0];used.add(v['seed']);result.append(v)
        replaced.append(dict(failed_id=ident,failed=w,replacement=v))
    return dict(worlds=sorted(result,key=lambda w:w['seed']),replaced=replaced,
        precharged_unattempted=adopt,additional_sim_s=1200*len(replaced))


def retired_key(record,job_ids):
    raw={k:v for k,v in record.items() if k!='state_id'}
    return any(digest(dict(record=raw,job=j))==record['state_id'] for j in job_ids)


def binding(root):
    r=Path(root);p=r/'VERIFICATION_BETA_BINDING.json'
    from .resource_config import NAME
    if (r/NAME).exists():p=r/NAME
    if not p.exists():p=r/'VERIFICATION_CONTINUATION_BINDING.json'
    if not p.exists():p=r/'MATCH_RECOVERY_BINDING.json'
    b=json.loads((p if p.exists() else r/'INCIDENT_RECOVERY_BINDING.json').read_text())
    assert b['amendment_sha256'] in (r/'AUTHORITY.md').read_text()
    assert b['cpu']['pass_'] and b['cpu']['skips']==b['cpu']['sim_s']==0
    s=Path(b['source_root']);assert Path(__file__).resolve()==s/'pytools/tools/coopmwm/incident_recovery.py'
    for n,h in b['cpu']['source_hashes'].items():assert sha(s/n)==h,n
    return b


def install(root,arm=None):
    from . import warm_runner as wr,accel_storage
    from .warm_protocol import batch_worlds
    from .accel_protocol import SUPERSEDED_CHARGE
    from tools.coopnative import storage
    from .tail_runtime import restore_progress
    from .warm_health import assert_running
    root=Path(root);b=binding(root)
    previous=json.loads((root/'MAIN0508_REPAIR_BINDING_V2.json').read_text())
    accel_storage._reserve=types.FunctionType(storage.reserve.__code__,
        dict(storage.reserve.__globals__,CAP=b['budget']['total']-SUPERSEDED_CHARGE,CAPS=b['budget']['phases']))
    wr.budget=lambda:copy.deepcopy(b['budget'])
    def schedule(u):
        if arm and u==b['resume'][arm]+1:return copy.deepcopy(b['plans'][arm]['worlds'])
        if u==1:return copy.deepcopy(previous['repair_plan']['worlds'])
        return batch_worlds(u)
    wr.batch_worlds=schedule
    def verify(r,a):
        assert a==arm;assert_running(r,a)
        original=json.loads((r/'EXACT_OPERATIONAL_BINDING.json').read_text())
        operational=json.loads((r/'WARM_B_RUNTIME_BINDING_V4.json').read_text())['operational']
        result=dict(original);result.update(operational)
        from .resource_config import clocks
        result=clocks(r,result)
        resume=b['resume'][a]
        assert sha(r/f'{a}_checkpoint_{resume:03}.pt')==b['checkpoints'][a]
        assert not (r/f'{a}_UPDATE_{resume+1:03}.json').exists(),'adopt completed optimizer instead'
        for w in schedule(resume+1):
            key=f'warm_u{resume+1:03}_{a}_{w["seed"]}'
            p=r/'jobs'/b.get('job_id_overrides',{}).get(key,key)
            assert (p/'RESULT.json').exists() or not (p/'CLAIM.json').exists(),'never retry attempted physics'
        # Restore completed updates using their original schedules, not replacement wave.
        def history(u):
            if u==1:return copy.deepcopy(previous['repair_plan']['worlds'])
            old=b.get('history_plans',{}).get(a,{})
            return copy.deepcopy(old[str(u)]) if str(u) in old else batch_worlds(u)
        progress=restore_progress(r,a,resume,history,wr.SEED)
        result.update(budget=b['budget'],awr=json.loads((r/'AWR_BINDING.json').read_text())['config'],
            resume_update=resume,resume_progress=progress,runtime_receipt_suffix=b.get('receipt_suffix','_I1'),
            workers=24 if a=='B' else 4,quota_aware_admission=True,parallel_group_queries=a=='B',
             precharged_unattempted=b['plans'][a]['precharged_unattempted'])
        result['job_id_overrides']=b.get('job_id_overrides',{})
        # MAIN (harden-int, 2026-11-02 evening): resume with bounded fan-out (2 x slots
        # in flight) and R concurrent with B on B's previous completed quota (lag 1).
        result.update(parallel_fanout_factor=b.get('parallel_fanout_factor',2),
            r_concurrent_lag1_quota=b.get('r_concurrent_lag1_quota',True))
        if b.get('b_replay'):
            # MAIN B-REPLAY: live frozen-u3 LS-COLLECT labels count as B bank entries (tag u3).
            result.update(b_replay=True,replay_beta=b.get('replay_beta',1.),
                replay_sources=b.get('replay_sources',{'B':['/home/ubuntu/coopmwm20261102_lscollect1']}),
                demand_phase_end_s=b['demand_phase_end_s'])
        if a in b.get('stop_after_update',{}):result['stop_after_update']=b['stop_after_update'][a]
        if b.get('sumo_verify'):result['sumo_verify']=copy.deepcopy(b['sumo_verify'])
        return result
    wr.verify=verify
    return wr,b


def run(root,component):
    root=Path(root);wr,b=install(root,component if component in ('B','R') else None)
    if component in ('B','R'):return wr.run_arm(root,component)
    if component=='grader':
        from .exact_runtime import supervise
        return supervise(root,4)
    if component=='broker':
        from .capacity_broker import run as broker
        return broker(root,root/('MATCH_BROKER_CONFIG.json' if (root/'MATCH_RECOVERY_BINDING.json').exists() else 'INCIDENT_BROKER_CONFIG.json'))
    raise ValueError(component)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--component',choices=('B','R','broker','grader'),required=True)
    a=p.parse_args();run(a.root,a.component)
