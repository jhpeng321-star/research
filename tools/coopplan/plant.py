"""Frozen LEVERPROBE full-rerun plant and COOPBANDIT pre-safety executor."""
from collections import defaultdict
from contextlib import contextmanager
import gzip
import time

from tools.coopbandit.protocol import group
from .protocol import canonical, digest, guard, install_seed_scope
from .storage import read, write, verify, admission, charge

ROOT = None
BINDING = None


def setup(root):
    global ROOT, BINDING
    from pathlib import Path
    ROOT = Path(root)
    BINDING = verify(ROOT)
    from tools.leverprobe import backend as probe
    probe.setup(BINDING['runtime'], BINDING['checkpoint'])
    install_seed_scope()
    probe.guard = guard
    from tools.coopbandit.metric_compat import install
    install()


def fresh(root, ident, world, seconds):
    from tools.coopbandit.backend import create
    admission(root)
    path = charge(root,ident,seconds)
    write(path/'START.json',dict(world=world,seconds=seconds))
    return create(path,world,seconds,BINDING['checkpoint']), path


@contextmanager
def free_requests(plan):
    """COOPFREE uses the SAME safety call; fixed targets are not re-anchored in solos."""
    if plan is None or plan.get('id','').split('_')[0] != 'FREE':
        yield
        return
    from tools.leverprobe import intervention
    from tools.coopfree.policy import requests
    original = intervention.requests
    def proposal(p,rows,fixed):
        return requests({v:t for v,t in p['targets'].items() if v in p['members']},rows),'on'
    intervention.requests = proposal
    try:
        yield
    finally:
        intervention.requests = original


def tick(b, event=None, trace=None):
    from tools.coopbandit.backend import tick as frozen_tick
    plan = event['executed_action'] if event is not None else None
    with free_requests(plan):
        return frozen_tick(b,plan=plan,initial=event['initial_rows'] if event else None,
                           start=event['time_s'] if event else None,trace=trace)


def preview_commands(b):
    """Read raw frozen MAPPO anchors using an isolated executor copy, before safety."""
    from copy import deepcopy
    import numpy as np
    import torch
    from wms.control import safety
    exe = deepcopy(b.exe)
    ids = np.asarray(b.env.frame.vid_index)
    keep = np.zeros(len(ids),bool) if b.previous is None else (ids == b.previous) & (ids >= 0)
    for obj in (exe.memory,exe.dwell):
        if obj.present is not None:
            obj.present = obj.present & torch.as_tensor(keep,device=obj.present.device)[None]
    original, captured = safety.safe_accel, []
    def capture(state,accel,**kwargs):
        captured.append(accel.detach().clone())
        return original(state,accel,**kwargs)
    safety.safe_accel = capture
    try:
        exe(b.env.state(),with_lanes=True)
        if len(captured) != 1:
            raise ValueError('exactly one raw MAPPO anchor call required')
        return {v:float(captured[0][0,i]) for i,v in enumerate(b.env.slot_vehicle_ids()) if v is not None}
    finally:
        safety.safe_accel = original


def replay(b, history, until, trace=None):
    while b.env.now < until:
        current = next((e for e in reversed(history) if e['time_s'] <= b.env.now),None)
        tick(b,current,trace)


class Diagnostics:
    """Read-only half-step delay clock + exact 10 s upstream BIC input bins."""
    def __init__(self,b,members):
        self.members = list(members)
        self.delay = dict.fromkeys(members,0.)
        self.speeds = defaultdict(list)
        self.b = b
        import numpy as np
        raw = b.env.history.raw(121)
        kin,mask = raw['kin'],raw['mask']
        first = float(b.env.now)-.5*(len(kin)-1)
        for i in range(len(kin)-1):
            lane = np.argmax(kin[i,:,1:4],axis=-1)
            keep = mask[i] & (lane < 2) & (kin[i,:,0] < 0)
            self.speeds[int((first+.5*i)//10)].extend(float(v) for v in kin[i,:,4][keep])
        original = b.env._after_physics
        def observe(after,*args,**kwargs):
            f = b.env.frame
            lane = np.where(f.junction,f.lane_carry,f.lane)
            keep = f.present & (lane < 2) & (f.s < 0)
            self.speeds[int((after-.5)//10)].extend(float(v) for v in f.v[keep])
            interval = original(after,*args,**kwargs)
            from traffic_shepherd.c4.env import SPEED_LIMIT_MPS
            for vid in members:
                if vid in b.env._table:
                    self.delay[vid] += .5*max(0.,1-b.env._table[vid]['speed_mps']/SPEED_LIMIT_MPS)
            return interval
        b.env._after_physics = observe

    def breakdown(self,t0,end):
        from .bic import breakdown
        values = []
        for k in range((t0-60)//10,end//10):
            vals = self.speeds.get(k,[])
            values.append(sum(vals)/len(vals) if vals else None)
        return breakdown(values)


def branch(root, replay_key, action, horizon):
    """Called in an isolated process: live TraCI remains owned by the parent."""
    from tools.leverprobe.backend import observations
    state = read(root/'states'/f'{replay_key}.json')
    t = state['time_s']
    ident = replay_key+'_'+action['id']
    b,path = fresh(root,ident,state['world'],t+horizon)
    started = time.monotonic()
    try:
        diag = Diagnostics(b,group(state['rows']))
        replay(b,state['history'],t)
        if canonical(b.fingerprint()) != canonical(state['prefix']):
            raise ValueError('BITWISE full executed-history prefix mismatch')
        if canonical(observations(b.env)) != canonical(state['rows']):
            raise ValueError('observable roster mismatch')
        before = dict(b.totals)
        member_before = dict(diag.delay)
        event = dict(time_s=t,executed_action=action,initial_rows=state['rows'])
        with gzip.open(path/'observable.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as tape:
            replay(b,[event],t+horizon,tape)
        metrics = {k:v-before[k] for k,v in b.totals.items()}
        bic = diag.breakdown(t,t+horizon)
        result = dict(team_delay=metrics['team_delay_vs'],breakdown=bic['event'],
                      member_delay={v:diag.delay[v]-member_before[v] for v in member_before},
                      safety={k:v for k,v in metrics.items() if k not in
                              ('team_delay_vs','road_delay_vs','source_wait_vs')})
        write(path/'RESULT.json',dict(outcome=result,metrics=metrics,bic=bic,prefix_pass=True,
              action=action,time_s=t,actual_sim_s=b.actual,wall_s=time.monotonic()-started))
        return result
    finally:
        b.close()


class RerunClient:
    def __init__(self,root,pool):
        self.root,self.pool = root,pool

    def __call__(self,state,action,horizon):
        return self.pool.submit(branch,self.root,state.replay_key,action,horizon).result()
