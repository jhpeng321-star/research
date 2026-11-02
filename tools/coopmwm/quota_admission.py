"""Serialized, observable compute admission; never score or predict acceptance."""
import json
from pathlib import Path
import time
from .protocol import digest
from .warm_spool import immutable


def state(root,update):
    root=Path(root);folder=root/'quota_requests'/f'u{update:03}'
    retired=root/'INCIDENT_RETIRED_JOBS.json'
    excluded=set(json.loads(retired.read_text())['jobs']) if retired.exists() else set()
    allowed=None
    repair=root/'MAIN0508_REPAIR_BINDING.json'
    if update==1 and repair.exists():
        allowed={f'warm_u001_B_{w["seed"]}' for w in json.loads(repair.read_text())['repair_plan']['worlds']}
    starts={p.stem for p in (root/'guided_starts'/f'u{update:03}').glob('*.json')
        if json.loads(p.read_text())['job'] not in excluded and
        (allowed is None or json.loads(p.read_text())['job'] in allowed)}
    unresolved=0
    for p in folder.glob('*_RESERVE.json'):
        v=json.loads(p.read_text());token=v['token']
        if v['job'] in excluded:continue
        bound=folder/(token+'_REQUEST.json')
        record=json.loads(bound.read_text()) if bound.exists() else None
        records=(record.get('records',[record]) if record else [])
        if any(r['record_sha256'] in starts for r in records):continue
        decision=folder/(token+'_DECISION.json')
        if decision.exists() and not json.loads(decision.read_text())['executed']:continue
        responses=[root/'responses'/(r['state_id']+'.json') for r in records]
        if responses and all(p.exists() and not json.loads(p.read_text())['accepted'] for p in responses):continue
        unresolved+=1
    return dict(executed=len(starts),inflight=unresolved)


def probability(executed,inflight,minimum,pending,slots):
    blocking=executed+inflight<minimum
    return (.95 if blocking or pending<slots else 0.),blocking


def choose(root,job,now,draw,pending,slots):
    """The RNG draw is made once, while holding the same lock as reservation."""
    import fcntl
    root=Path(root);folder=root/'quota_requests'/f'u{job["update"]:03}'
    folder.mkdir(parents=True,exist_ok=True)
    token=digest(dict(job=job['id'],time_s=int(now)))
    with (folder/'ADMISSION.claim').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        s=state(root,job['update'])
        p,blocking=probability(s['executed'],s['inflight'],job['N_exec'],pending,slots)
        queried=draw()<p
        receipt=dict(s,minimum=job['N_exec'],pending=pending,slots=slots,
            blocking=blocking,nonblocking=not blocking,query_probability=p,
            predicate='executed+potentially executable in-flight < N_exec; otherwise free slot only')
        if queried:immutable(folder/(token+'_RESERVE.json'),dict(token=token,job=job['id'],time_s=int(now),admission=receipt,unix=time.time()))
    return p,queried,receipt,dict(folder=str(folder),token=token) if queried else None


def bind(reservation,record):
    folder=Path(reservation['folder'])
    immutable(folder/(reservation['token']+'_REQUEST.json'),
        dict(state_id=record['state_id'],record_sha256=digest(record)))
    saved=json.loads((folder/(reservation['token']+'_RESERVE.json')).read_text())
    admissions=folder.parent.parent/'quota_admissions';admissions.mkdir(exist_ok=True)
    immutable(admissions/(record['state_id']+'.json'),saved['admission'])


def choose_group_batch(root,job,now,draw,pending,slots):
    """One existing Bernoulli draw/reservation for a parallel group window.

    At most one plan starts at this control tick, so a batch counts as ONE
    potentially executable window, irrespective of its number of WM queries.
    None means the minimum is already executed: use the unchanged normal path.
    """
    import fcntl
    root=Path(root);folder=root/'quota_requests'/f'u{job["update"]:03}'
    folder.mkdir(parents=True,exist_ok=True)
    token=digest(dict(job=job['id'],time_s=int(now)))
    with (folder/'ADMISSION.claim').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        s=state(root,job['update'])
        if s['executed']>=job['N_exec']:return None
        p,blocking=probability(s['executed'],s['inflight'],job['N_exec'],pending,slots)
        queried=draw()<p
        receipt=dict(s,minimum=job['N_exec'],pending=pending,slots=slots,
            blocking=True,nonblocking=False,query_probability=p,parallel_groups=True,
            window_would_block=blocking,
            predicate='quota-unmet admitted window submits all eligible groups; one possible plan-start per window')
        if queried:immutable(folder/(token+'_RESERVE.json'),dict(token=token,job=job['id'],time_s=int(now),admission=receipt,unix=time.time()))
    return p,queried,receipt,dict(folder=str(folder),token=token) if queried else None


def bind_group_batch(reservation,records):
    folder=Path(reservation['folder'])
    immutable(folder/(reservation['token']+'_REQUEST.json'),dict(records=[
        dict(state_id=r['state_id'],record_sha256=digest(r)) for r in records]))
    saved=json.loads((folder/(reservation['token']+'_RESERVE.json')).read_text())
    admissions=folder.parent.parent/'quota_admissions';admissions.mkdir(exist_ok=True)
    for r in records:immutable(admissions/(r['state_id']+'.json'),saved['admission'])


def resolved(reservation,executed):
    immutable(Path(reservation['folder'])/(reservation['token']+'_DECISION.json'),dict(executed=bool(executed)))


def first_update_deviation(root,arm,update,executed,minimum):
    if arm!='B' or executed>=minimum:return None
    reg=Path(root)/'MAIN0548_RUNTIME_BINDING.json'
    if update!=1 or not reg.exists():raise ValueError(f'B plan gate unmet {executed}<{minimum}; no extra SUMO')
    v=json.loads(reg.read_text())
    assert v['plan_amendment_sha256'] in (Path(root)/'AUTHORITY.md').read_text()
    return dict(update=1,executed=executed,minimum=minimum,engineering_deviation=True,
        authority='MAIN05:48 apply retained u1 even if short; no additional physics',
        amendment_sha256=v['plan_amendment_sha256'])
