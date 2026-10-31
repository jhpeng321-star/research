"""Leverprobe plant/executor reused unchanged; richer trace and clock export."""
import gzip
import hashlib
import math
import os
from pathlib import Path
import time

from .protocol import canonical, guard, install_seed_scope, option, sha


def command_record(value, requested):
    """NaN is the executor's native-SUMO/no-command sentinel, not an acceleration."""
    issued=float(value)
    if math.isnan(issued):
        return dict(controller='native_sumo',issued_acceleration_mps2=None,command_present=False)
    if not math.isfinite(issued):
        raise ValueError('infinite issued acceleration')
    return dict(controller='option' if requested else 'policy_mean',
                issued_acceleration_mps2=issued,command_present=True)


def action_record(action):
    out=dict(action)
    absent=[]
    for key,value in action.items():
        if isinstance(value,float) and math.isnan(value):
            if key not in ('mappo_raw','mappo_safe','issued_raw','issued_safe'):
                raise ValueError('unexpected NaN in action '+key)
            out[key]=None
            absent.append(key)
        elif isinstance(value,float) and not math.isfinite(value):
            raise ValueError('infinite action '+key)
    out['native_no_command_fields']=absent
    return out


def setup(binding):
    from tools.leverprobe import backend as probe
    probe.setup(binding['runtime'], binding['checkpoint'])
    install_seed_scope()
    probe.guard = guard


def create(path, world, seconds, checkpoint):
    from tools.leverprobe.backend import Backend
    return Backend(path, world, seconds, checkpoint)


def tick(b, plan=None, initial=None, start=None, trace=None):
    """Same direct/native-lane execution order as leverprobe Backend.advance."""
    import numpy as np
    import torch
    from tools.leverprobe.backend import observations, exact
    from tools.leverprobe.intervention import pre_safety, requests, targets
    now, rows = float(b.env.now), observations(b.env)
    ids = np.asarray(b.env.frame.vid_index)
    keep = np.zeros(len(ids),bool) if b.previous is None else (ids == b.previous) & (ids >= 0)
    b.previous = ids.copy()
    for obj in (b.exe.memory, b.exe.dwell):
        if obj.present is not None:
            obj.present = obj.present & torch.as_tensor(keep,device=obj.present.device)[None]
    active = plan is not None and bool(plan.get('members')) and now < start+plan.get('window',0)
    requested, event = requests(plan,rows,targets(plan,initial)) if active else ({},'baseline')
    audit = []
    with pre_safety(b.exe,b.env,requested,audit):
        cmd, lanes = b.exe(b.env.state(),with_lanes=True)
    if lanes is not None:
        raise ValueError('native lanes required')
    b.command_digest.update(canonical(exact(cmd)).encode())
    if trace is not None:
        slot = {vid:i for i,vid in enumerate(b.env.slot_vehicle_ids())}
        vehicles = []
        for r in rows:
            public = dict(r, length_m=float(b.traci.vehicle.getLength(r['vid'])))
            if r['is_cav']:
                public.update(command_record(np.asarray(cmd).reshape(-1)[slot[r['vid']]],r['vid'] in requested))
            vehicles.append(public)
        trace.write(canonical(dict(time_s=now,vehicles=vehicles,actions=[action_record(a) for a in audit],event=event))+'\n')
    _, info = b.env.step(cmd,lanes=None)
    interval = info['interval']
    road, source = interval['speed_deficit_vehicle_seconds'], interval['source_wait_vehicle_seconds']
    b.totals['road_delay_vs'] += road
    b.totals['source_wait_vs'] += source
    b.totals['team_delay_vs'] += road+source
    for key, field in (('collisions','collision_involvements'),('teleports','starting_teleports'),
                       ('ttc_below_3_vs','ttc_below_3_vehicle_seconds'),('hard_brake','braking_vehicle_seconds')):
        b.totals[key] += interval.get(field,0.)
    b.observe(b.env)
    if info['done'] or any(info['lane_commands'].values()) or any(interval.get(k,0) for k in
                   ('collision_involvements','starting_teleports','dropped_vehicle_steps')):
        raise ValueError('safety/slot/early-end/lane anomaly')
    return dict(requests=len(audit),overrides=sum(r['clamped'] for r in audit),event=event)


def advance(b, until, **kwargs):
    requests, overrides = 0, 0
    while b.env.now < until:
        a = tick(b,**kwargs)
        requests += a['requests']
        overrides += a['overrides']
    return dict(requests=requests,overrides=overrides,
                override_share=overrides/requests if requests else 0.)
