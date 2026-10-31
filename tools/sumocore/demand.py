"""Entrance-request inference from causal counters, not original schedules."""
import hashlib
import numpy as np


def stream(world, rng, name):
    seed=int.from_bytes(hashlib.sha256(f'sumocore:{world}:{rng}:{name}'.encode()).digest()[:8],'big')
    return np.random.default_rng(seed)


def estimate(start, window=60.):
    t0=float(start['t0']);assert t0==300.
    counters=np.asarray(start['observed_queue_history'],float)
    assert counters[-1,0]==t0
    idx=np.flatnonzero(np.isclose(counters[:,0],t0-window))
    assert len(idx)==1
    q0=counters[idx[0],1:];q1=counters[-1,1:]
    entries=[r for r in start['past_entries'] if t0-window<r['time']<=t0]
    inserted=np.array([sum((r['lane']==2)==bool(j) for r in entries) for j in range(2)])
    requested=inserted+q1-q0
    assert (requested>=0).all() and np.allclose(requested,np.round(requested))
    # IDs of future CAVs are not read from episode metadata or scenario files.
    p=sum(bool(r['is_cav']) for r in entries)/len(entries) if entries else 0.
    return dict(window_s=window,inserted=inserted.astype(int).tolist(),
                queue_start=q0.astype(int).tolist(),queue_t0=q1.astype(int).tolist(),
                requested=requested.astype(int).tolist(),rates_per_s=(requested/window).tolist(),
                observed_cav_fraction=p,observed_entries=len(entries))


def sample(start,world,rng,horizon):
    e=estimate(start)
    requests=[]
    for j,name in enumerate(('mainline','ramp')):
        # Initial due queue has observed count but no visible individual identities.
        departures=[300.]*e['queue_t0'][j]
        arrival_rng=stream(world,rng,name+'.arrival')
        rate=e['rates_per_s'][j]
        now=300.
        while rate>0:
            now+=float(arrival_rng.exponential(1./rate))
            if now>=300+horizon:break
            departures.append(now)
        cav_rng=stream(world,rng,name+'.cav')
        for k,t in enumerate(departures):
            requests.append(dict(id=f'{name}.sc.{k}',depart=t,entrance=name,
                                 is_cav=bool(cav_rng.random()<e['observed_cav_fraction'])))
    return sorted(requests,key=lambda r:(r['depart'],r['id'])),e
