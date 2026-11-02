"""Quota-tail query fan-out, preserving the single plan-start control/grade ABI.

All disjoint eligible groups at one observed instant are queried before any wait.
Resolve in the inherited group order (not predicted gain or completion order).
One accepted group is returned to the existing trust/commit controller; other
forecasts are recorded as searched, not executed. The single window RNG draw
and single beta draw preserve the exact grader's original CRN draw contract.
"""
import copy
from pathlib import Path
import time
import torch
from .protocol import digest
from .storage import read
from .warm_spool import immutable
from .warm_health import assert_running
from . import quota_admission as quota


def _checked(value,r):
    if value['record_sha256']!=digest(r):raise ValueError('parallel plan response identity mismatch')
    if value['accepted'] and value['plan']['members']!=r['group']:
        raise ValueError('parallel plan response group mismatch')
    return value


def withdraw(root,record,job_id):
    """Client-side marker: this decision no longer needs the request.

    Infrastructure-only (never a forecast). The broker may decline it while no
    worker holds it; an already assigned search still finishes once, unchanged.
    """
    folder=Path(root)/'request_withdrawals';folder.mkdir(exist_ok=True)
    immutable(folder/(record['state_id']+'.json'),dict(state_id=record['state_id'],
        record_sha256=digest(record),job=job_id,reason='decided by earlier accepted group in inherited order'))


def query(job,generator,history,rows,commands,now,sample,groups,traffic,workers):
    root=Path(job['queue_root'])
    pending=sum(not (root/'responses'/p.name).exists() for p in (root/'requests').glob('*.json'))
    chosen=quota.choose_group_batch(root,job,now,
        lambda:float(torch.rand((),generator=generator)),pending,len(workers))
    if chosen is None:return None
    p,queried,admission,reservation=chosen
    b=dict(route='wm' if queried else 'policy',sample_probability=p,mixture_mass=p,
        regime=traffic,executed=False,beta=0.,q=0.,eligible_query=True,capacity_admission=admission)
    sample['behaviour']=b
    if not queried:return dict(accepted=False,behaviour=b,reason='parallel window not sampled',record_sha256=None)
    common=dict(world_id=job['world']['seed'],time_s=int(now),rows=rows,commands=commands,
        policy_residuals={v:float(a[0]) for v,a in zip(sample['context']['members'],sample['own_action'])},
        history=history.snapshot(now))
    if sample.get('committed_plans'):common['committed_plans']=copy.deepcopy(sample['committed_plans'])
    for name in ('policy_state','search_learning'):
        if job.get(name) is not None:common[name]=job[name]
    records=[]
    for group in groups:
        r=dict(common,group=group['members']);r['state_id']=digest(dict(record=r,job=job['id']))
        records.append(r)
    # Admissions must exist BEFORE publication so the broker never fabricates a
    # capacity decline for one of these quota-blocking real searches.
    quota.bind_group_batch(reservation,records)
    start=time.monotonic()
    for r in records:immutable(root/'requests'/(r['state_id']+'.json'),r)
    # HARDEN-3: wait in inherited group order and STOP at the first accepted
    # response. The selection rule is unchanged (first accepted in inherited
    # order), but the episode no longer idles until EVERY group of the batch is
    # answered: before this fix an accepted response could not reach the
    # blocking SUMO verification while later groups were still queued behind
    # other episodes' searches (B u4 17:10-18:33 stall). Later groups that are
    # already answered are recorded as before; unanswered ones are recorded as
    # pending and WITHDRAWN so the broker may decline them while unassigned.
    results=[None]*len(records);selected=None
    for i,r in enumerate(records):
        response=root/'responses'/(r['state_id']+'.json')
        while not response.exists():
            assert_running(root,job['arm'])
            if time.time()>job['gpu_deadline']:raise ValueError('registered GPU deadline exceeded')
            time.sleep(.5)
        results[i]=_checked(read(response),r)
        if results[i]['accepted']:selected=i;break
    withdrawn=[]
    for i,r in enumerate(records):
        if results[i] is not None:continue
        response=root/'responses'/(r['state_id']+'.json')
        if response.exists():results[i]=_checked(read(response),r);continue
        withdraw(root,r,job['id'])
        withdrawn.append(r['state_id'])
        results[i]=dict(accepted=None,pending=True,withdrawn=True,candidates_searched=0,imagined_futures=0)
    if selected is None:selected=0
    result=copy.deepcopy(results[selected]);r=records[selected]
    summaries=[dict(state_id=q['state_id'],record_sha256=digest(q),members=q['group'],
        accepted=v['accepted'],selected=i==selected and bool(v['accepted']),pending=bool(v.get('pending')),
        expected_saving=v.get('expected_saving'),candidates_searched=v.get('candidates_searched',0),
        imagined_futures=v.get('imagined_futures',0)) for i,(q,v) in enumerate(zip(records,results))]
    result.update(query_wall_s=time.monotonic()-start,quota_reservation=reservation,
        parallel_queries=summaries,parallel_selection='first accepted in inherited group order; one start per tick')
    b.update(queried_option=result.get('option'),proposal_accepted=result['accepted'],
        query_wall_s=result['query_wall_s'],parallel_query_counts=dict(queries=len(results),
            accepted=sum(bool(v['accepted']) for v in results),withdrawn_pending=len(withdrawn),candidates_searched=sum(v.get('candidates_searched') or 0 for v in results)))
    if result['accepted']:
        result['history']=r['history']
        b.update(option=result['option'],proposal_members=result['plan']['members'],
            expected_saving=result.get('expected_saving'),predicted_rank=result.get('predicted_rank'),
            known_family=result.get('known_family'),duration_s=result['plan']['duration_s'],
            candidates_searched=result.get('candidates_searched'),iterations=result.get('iterations'),
            group_size=len(result['plan']['members']))
    result['behaviour']=b
    return result
