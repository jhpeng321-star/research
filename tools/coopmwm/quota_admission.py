"""Serialized, observable compute admission; never score or predict acceptance."""
import json
from pathlib import Path
import time
from .protocol import digest
from .warm_spool import immutable


def state(root,update):
    root=Path(root);folder=root/'quota_requests'/f'u{update:03}'
    allowed=None
    repair=root/'MAIN0508_REPAIR_BINDING.json'
    if update==1 and repair.exists():
        allowed={f'warm_u001_B_{w["seed"]}' for w in json.loads(repair.read_text())['repair_plan']['worlds']}
    starts={p.stem for p in (root/'guided_starts'/f'u{update:03}').glob('*.json')
        if allowed is None or json.loads(p.read_text())['job'] in allowed}
    unresolved=0
    for p in folder.glob('*_RESERVE.json'):
        v=json.loads(p.read_text());token=v['token']
        bound=folder/(token+'_REQUEST.json')
        record=json.loads(bound.read_text()) if bound.exists() else None
        if record and record['record_sha256'] in starts:continue
        decision=folder/(token+'_DECISION.json')
        if decision.exists() and not json.loads(decision.read_text())['executed']:continue
        response=root/'responses'/(record['state_id']+'.json') if record else None
        if response and response.exists() and not json.loads(response.read_text())['accepted']:continue
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
