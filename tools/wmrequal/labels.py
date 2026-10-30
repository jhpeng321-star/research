"""Recorded-only complete-fleet labels, independent departure RNG, no assignments.

Global numeric identities are FrameBuilder.index_of's append-only registry. Hence
max registry ID + 1 is total insertions, even with a truncated history. At t=300
the existing 300-second counter is the exact cumulative insertion count by source.
These two independent counts must agree. Later-born identities recover all future
insertions. Late-start per-source Q0 is deliberately unavailable without names.
"""
from pathlib import Path
import random
import numpy as np

HORIZONS = (60, 90, 120, 180, 300)
BAD = ('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1',
       'hetdata_truth','vtype')


def safe(path):
    p = Path(path)
    if any(x in str(p).lower() or x in str(p.resolve()).lower() for x in BAD):
        raise ValueError('restricted path')
    return p


def departures(seed, rate, stream):
    rng = random.Random(f'damping_requests_v1:{seed}:{stream}')
    t, result = 0., []
    while True:
        t += rng.expovariate(rate/3600.)
        if t >= 1200:
            break
        result.append(round(t, 6))
    return np.asarray(result)


def crossings(pos, present, lane, valid, ids, dt=.5):
    """Continuous-identity valid-coordinate brackets, ERRSRC corrected kernel."""
    last, previous, events = {}, set(), []
    for t in range(len(pos)):
        slots = np.flatnonzero(present[t])
        current = {int(ids[t,j]): int(j) for j in slots}
        if len(current) != len(slots) or any(i < 0 for i in current):
            raise ValueError('present identities must be unique and known')
        for identity, j in current.items():
            if identity not in previous:
                last.pop(identity, None)
            if valid[t,j]:
                if identity in last:
                    pt, ps = last[identity]
                    if ps < 75.44 <= pos[t,j] and lane[t,j] < 2:
                        f = (75.44-ps)/(float(pos[t,j])-ps)
                        events.append((pt+f*(t-pt))*dt)
                last[identity] = (t, float(pos[t,j]))
        for identity in set(last)-set(current):
            last.pop(identity)
        previous = set(current)
    return np.asarray(events)


def recorded(root, entry, horizon, identities=None):
    seed = int(entry['seed'])
    if 964000 <= seed <= 964199 or (977000 <= seed < 978000 and
            not (977228 <= seed <= 977287 and horizon <= 120)):
        raise ValueError('seed/horizon refused')
    p = safe(Path(root)/entry['dir'])
    load = lambda n: np.load(safe(p/(n+'.npy')), mmap_mode='r', allow_pickle=False)
    ts, vid, mask, kin = (load(n) for n in ('time_s','vid','veh_mask','veh_kin'))
    mask = mask.astype(bool)
    t0 = int(np.searchsorted(ts, entry['warmup_s']))
    steps = int(horizon*2)
    assert ts[t0] == entry['warmup_s'] and len(ts) >= t0+steps+1
    dep = [departures(seed, entry['demand_vph'], 'mainline'),
           departures(seed, 400, 'ramp')]
    times = np.asarray(ts[t0:t0+steps+1])
    due = np.stack([np.searchsorted(d, times, side='right') for d in dep], -1)
    latest = int(vid[:t0+1][mask[:t0+1]].max())
    inserted = np.zeros((steps+1,2), int)
    source_available = float(ts[t0]) == 300.
    if source_available:
        inserted[0] = load('counter')[t0].sum(-1).astype(int)
        assert inserted[0].sum() == latest+1, 'counter/registry disagreement'
    elif identities is not None:
        # Exact S-GAPCHECK ordinal reconstruction from recorded names only.
        seen=np.unique(vid[:t0+1][mask[:t0+1]])
        for entrance,stream in enumerate(('mainline','ramp')):
            ordinals=[int(identities[int(i)].split('.')[1]) for i in seen
                      if i>=0 and identities[int(i)].startswith(stream+'.')]
            inserted[0,entrance]=max(ordinals)+1 if ordinals else 0
        assert inserted[0].sum()==latest+1,'ordinal/registry disagreement'
        source_available=True
    total_inserted = np.zeros(steps+1, int)
    total_inserted[0] = latest+1
    for k in range(1, steps+1):
        t = t0+k
        new = np.flatnonzero(mask[t] & (vid[t] > latest))
        ids = sorted(int(vid[t,j]) for j in new)
        assert ids == list(range(latest+1, latest+len(ids)+1)), 'registry skipped IDs'
        inserted[k] = inserted[k-1]
        for j in new:
            onehot = kin[t,j,1:4]
            assert onehot.sum() > 0, 'new birth lacks entrance coordinate'
            inserted[k, int(onehot.argmax() == 2)] += 1
        latest += len(ids)
        total_inserted[k] = latest+1
    queue = due.sum(-1)-total_inserted
    assert (queue >= 0).all(), 'negative reconstructed queue'
    by_source = due-inserted if source_available else None
    if by_source is not None:
        assert (by_source >= 0).all() and np.array_equal(by_source.sum(-1), queue)
    # Identity-aligned previous cohort and end-step speed, exactly objective14.
    # Exits retain the previous observed speed, as the real BranchTracker does.
    road = np.zeros((steps,3))
    old = np.zeros(steps)
    init_hdv = set(int(vid[t0,j]) for j in np.flatnonzero(
        mask[t0] & ~load('is_cav')[t0].astype(bool)))
    t0_endstep = np.zeros(steps)
    for k in range(steps):
        t = t0+k
        nxt = {int(vid[t+1,j]):j for j in np.flatnonzero(mask[t+1])}
        for j in np.flatnonzero(mask[t]):
            identity = int(vid[t,j]); nj = nxt.get(identity)
            speed = float(kin[t+1,nj,4] if nj is not None else kin[t,j,4])
            cost = .5*max(0., 1-speed/31.29)
            s, oh = float(kin[t,j,0]), kin[t,j,1:4]
            region = 0 if s <= 75.44 else 1  # upstream+merge versus downstream
            road[k,region] += cost
            if identity in init_hdv:
                old[k] += cost
        for j in np.flatnonzero(mask[t+1]):
            if int(vid[t+1,j]) in init_hdv:
                t0_endstep[k] += .5*max(0.,1-float(kin[t+1,j,4])/31.29)
    sl = slice(t0,t0+steps+1)
    events = crossings(kin[sl,:,0],mask[sl],kin[sl,:,1:4].argmax(-1),
                       kin[sl,:,1:4].sum(-1)>0,vid[sl])
    values = {}
    for h in HORIZONS:
        if h > horizon:
            continue
        n = h*2
        # S-GAPCHECK's left-boundary 0.5s rectangle sum (not ledger interpolation).
        source = float(.5*queue[:n].sum())
        r = float(road[:n].sum())
        values[str(h)] = dict(J=r+source, road=r, source=source,
            upstream=float(road[:n,0].sum()), downstream=float(road[:n,1].sum()),
            t0_HDV=float(t0_endstep[:n].sum()),
            queue=float(queue[n]), discharge=int((events<=h).sum()))
    rate = [float(((d <= times[0]) & (d > times[0]-300)).sum())/300 for d in dep]
    return dict(values=values, queue0_total=int(queue[0]),
                queue0=by_source[0].tolist() if by_source is not None else None,
                demand_rate=rate, source_available=source_available,
                registry_count=int(total_inserted[0]),
                counter_registry_equal=float(ts[t0])==300.,
                source_method='recorded_ordinals' if identities is not None else 'counter_registry_conservation',
                input_dir=str(p))
