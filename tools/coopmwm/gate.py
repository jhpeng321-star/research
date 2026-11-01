"""Cross-process executed-guidance quotas; no extra episodes or WM rewards."""
import contextlib
import json
from pathlib import Path
from .storage import read,write


@contextlib.contextmanager
def locked(path):
    import fcntl
    with (Path(path)/'EXECUTION.claim').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        yield


def events(path):
    p=Path(path)/'EVENTS.jsonl'
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []


def counts(path):
    values=events(path)
    return {r:sum(v['regime']==r for v in values) for r in ('merge-dense','ordinary')}


def remaining(config,done,regime):
    quota=config['quota']
    return (quota-sum(done.values()) if isinstance(quota,int)
            else quota[regime]-done[regime])


def initialize(path,quota,sampling):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    value=dict(quota=quota,sampling=sampling)
    if (path/'CONFIG.json').exists():assert read(path/'CONFIG.json')==value
    else:write(path/'CONFIG.json',value)


def sampling(path):
    with locked(path):
        config=read(Path(path)/'CONFIG.json');done=counts(path)
        return {r:config['sampling'][r] if remaining(config,done,r)>0 else 0.
                for r in ('merge-dense','ordinary')}


def execute(path,sample,proposal,previous,generator):
    from .exploration import execute as apply
    with locked(path):
        config=read(Path(path)/'CONFIG.json');done=counts(path)
        traffic=sample['behaviour']['regime']
        allowed=remaining(config,done,traffic)>0
        out=apply(sample,proposal,previous if allowed else 0.,generator)
        if out['behaviour']['executed']:
            value=dict(regime=traffic,option=proposal['option'],
                       ordinal=sum(done.values())+1)
            with (Path(path)/'EVENTS.jsonl').open('a') as f:
                f.write(json.dumps(value,sort_keys=True)+'\n');f.flush()
        return out


def verify(path,episodes):
    """Never advance an update that lacks the registered actual executed count."""
    config=read(Path(path)/'CONFIG.json');done=counts(path)
    actual={r:sum(s['sample'].get('behaviour',{}).get('executed',False)
                  and s['sample']['behaviour']['regime']==r
                  for ep in episodes for s in ep['steps'])
            for r in ('merge-dense','ordinary')}
    if actual!=done:raise ValueError('guidance counter differs from closed SUMO episode events')
    quota=config['quota']
    met=sum(actual.values())>=quota if isinstance(quota,int) else actual==quota
    value=dict(pass_=met,actual=actual,quota=quota,episodes=len(episodes),
               sim_s=len(episodes)*1200,reward_source='SUMO only')
    p=Path(path)/'VERDICT.json'
    if not p.exists():write(p,value)
    if not met:raise ValueError('executed-guidance gate unmet; no PPO update or automatic extra physics: '+str(value))
    return value
