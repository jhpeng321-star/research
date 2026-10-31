"""Immutable explicit amendment chain; old failed bytes/STOPs are never removed."""
from pathlib import Path
import hashlib
import json
import os

ROOT=Path('/home/ubuntu/sumocore20261031')


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def amendments():
    mapping={};stops={}
    for p in sorted(ROOT.glob('AMENDMENT_*.json')):
        a=json.loads(p.read_text());mapping[a['old']]=a['new'];stops.update(a['stops'])
        assert digest(ROOT/'jobs'/a['old']/'failure.json')==a['failure_sha256']
    for p in [ROOT/'STOP',*ROOT.glob('STOP.*')]:
        if p.exists():assert stops.get(p.name)==digest(p),('unacknowledged STOP',str(p))
    return mapping


def effective(job,mapping):
    j=dict(job)
    seen=set()
    while j['id'] in mapping:
        assert j['id'] not in seen;seen.add(j['id']);j['id']=mapping[j['id']]
    return j


def manual_zero_step_retry(old,new):
    """Explicit operator action AFTER registration; never called by supervisor."""
    from worker import write
    failure=ROOT/'jobs'/old/'failure.json'
    f=json.loads(failure.read_text());assert f['actual_sim_s']==0
    start=json.loads((failure.parent/'START.json').read_text())
    assert not Path('/proc',str(start['pid'])).exists(), 'old owned worker still exists'
    assert not (ROOT/'jobs'/new).exists()
    rows=[json.loads(x) for x in (ROOT/'ledger.jsonl').read_text().splitlines()]
    assert new not in {r['id'] for r in rows}
    charge=f['job']['charge'];assert sum(r['charge'] for r in rows)+charge<=100000
    stops={p.name:digest(p) for p in [ROOT/'STOP',*ROOT.glob('STOP.*')] if p.exists()}
    assert stops
    with (ROOT/'ledger.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(id=new,charge=charge,kind='registered_zero_step_recovery'))+'\n')
        stream.flush();os.fsync(stream.fileno())
    n=len(list(ROOT.glob('AMENDMENT_*.json')))+1
    write(ROOT/f'AMENDMENT_{n:02d}.json',dict(old=old,new=new,stops=stops,
        failure_sha256=digest(failure),extra_precharge=charge,
        reason='registered mechanical startup repair; no scientific outcome/physics step existed'))
    print(json.dumps(dict(old=old,new=new,extra_precharge=charge,total_precharge=sum(r['charge'] for r in rows)+charge)))


if __name__=='__main__':
    import sys
    manual_zero_step_retry(*sys.argv[1:])
