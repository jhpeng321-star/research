"""Exact-zero TRAIN tape checks; reporting only, never a learning filter."""
import gzip
import json
from itertools import zip_longest
from .protocol import digest
from .accel_storage import read,write


def compare(original,branch,start,end):
    out=dict(seconds=0,roster_differences=0,motion_differences=0)
    for al,bl in zip_longest(original,branch):
        if bl is None:break
        if al is None:raise ValueError('without-plan tape exceeds original')
        a,b=json.loads(al),json.loads(bl)
        if a['time_s']!=b['time_s']:raise ValueError('observable tape clocks differ')
        if not start<=a['time_s']<end:continue
        out['seconds']+=1
        av,bv=({r['vid']:r for r in x['vehicles']} for x in (a,b))
        out['roster_differences']+=int(set(av)!=set(bv))
        out['motion_differences']+=sum(any(av[v][k]!=bv[v][k] for k in ('s_m','v_mps','lane')) for v in av.keys()&bv.keys())
    if out['seconds']!=end-start:raise ValueError('incomplete exact-zero observable window')
    return out


def classify(root,receipt,original):
    if receipt['grade_veh_s']!=0:return dict(exact_zero=False,observed_physical_null=False)
    if receipt['arm']!='B' or receipt.get('training_seed')!=20261101:
        raise ValueError('only same-seed B TRAIN receipts admitted')
    key=receipt['grade_id'];destination=root/'physical_nulls'/(key+'.json')
    if destination.exists():
        old=read(destination)
        if old['grade_receipt_sha256']!=receipt['receipt_sha256']:raise ValueError('exact-zero grade changed')
        return old
    op=root/'jobs'/receipt['episode_id']
    bp=root/'exact_grades/jobs'/key/f'attempt{receipt["attempt"]}'
    branch=read(bp/'RESULT.json');start,end=receipt['window']['start_s'],receipt['window']['end_s']
    def steps(result):return {int(s['time_s']):s for s in result['steps'] if start<=s['time_s']<end}
    a,b=steps(original),steps(branch)
    if set(a)!=set(range(start,end)) or set(b)!=set(a):raise ValueError('incomplete exact-zero result window')
    rewards=sum(a[t]['reward']!=b[t]['reward'] for t in a)
    with gzip.open(op/'observable.jsonl.gz','rt') as af,gzip.open(bp/'observable.jsonl.gz','rt') as bf:
        motion=compare(af,bf,start,end)
    value=dict(grade_id=key,grade_receipt_sha256=receipt['receipt_sha256'],update=receipt['update'],
        exact_zero=True,observed_physical_null=not(rewards or motion['roster_differences'] or motion['motion_differences']),
        reward_differences=rewards,**motion,definition='identical sensed whole-road roster/position/speed/lane and per-second reward over exact grade window; not raw/final command equality',
        learning_changes=False,quota_counted=True,imitation=False,library=False)
    write(destination,value)
    return value


def by_update(rows):
    result={}
    for u in sorted({r['update'] for r in rows}):
        values=[r for r in rows if r['update']==u]
        n=sum(r['null_classification']['observed_physical_null'] for r in values)
        z=sum(r['null_classification']['exact_zero'] for r in values)
        result[str(u)]=dict(exact_graded=len(values),exact_zero=z,confirmed_physical_null=n,
            confirmed_null_fraction=n/len(values),denominator='qualified exact-graded B plans, not all executions; zeros inspected, nonzero grades cannot be whole-window physical nulls')
    return result
