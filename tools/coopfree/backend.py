"""Frozen LEVERPROBE full-rerun plant, unchanged safety and CommandOwnership."""
import math
import numpy as np
import torch
from .protocol import canonical, guard, install_seed_scope
from .execution import Controller, inject


def setup(binding):
    from tools.leverprobe import backend as plant
    plant.setup(binding['runtime'],binding['checkpoint'])
    install_seed_scope()
    plant.guard = guard


def create(path,world,seconds,binding):
    from tools.leverprobe.backend import Backend
    return Backend(path,world,seconds,binding['checkpoint'])


def observations(b):
    from tools.leverprobe.backend import observations as public
    return [dict(r,length_m=float(b.traci.vehicle.getLength(r['vid']))) for r in public(b.env)]


def tick(b, controller=None, trace=None):
    from tools.leverprobe.backend import exact
    controller = controller or Controller()
    now, rows = float(b.env.now), observations(b)
    ids = np.asarray(b.env.frame.vid_index)
    keep = np.zeros(len(ids),bool) if b.previous is None else (ids == b.previous) & (ids >= 0)
    b.previous = ids.copy()
    for obj in (b.exe.memory,b.exe.dwell):
        if obj.present is not None:
            obj.present &= torch.as_tensor(keep,device=obj.present.device)[None]
    audit = []
    with inject(b.exe,b.env,rows,controller,now,audit):
        cmd, lanes = b.exe(b.env.state(),with_lanes=True)
    if lanes is not None:
        raise ValueError('native lanes required')
    b.command_digest.update(canonical(exact(cmd)).encode())
    slot = {vid:i for i,vid in enumerate(b.env.slot_vehicle_ids())}
    active = {r['vid'] for r in audit}
    mean_command = {r['vid']:r['mappo_safe'] for r in audit}
    vehicles = []
    for r in rows:
        v = dict(r)
        if r['is_cav']:
            a = float(np.asarray(cmd).reshape(-1)[slot[r['vid']]])
            if math.isinf(a):
                raise ValueError('infinite command')
            v.update(controller=('setpoint' if r['vid'] in active else 'policy_mean') if math.isfinite(a) else 'native',
                     issued_acceleration_mps2=a if math.isfinite(a) else None,
                     policy_mean_acceleration_mps2=mean_command.get(r['vid'],a) if math.isfinite(a) else None,
                     command_present=math.isfinite(a))
        vehicles.append(v)
    if trace is not None:
        trace.write(canonical(dict(time_s=now,vehicles=vehicles,actions=audit))+'\n')
    # Reuse c4/lane-policy's single-writer lane hook (e5740395). The env keeps
    # SUMO laneChangeMode=513 safety; no direct TraCI command is added here.
    request = getattr(controller, 'lane_request', None)
    lanes = request(b.env) if request is not None else None
    actors = getattr(controller, 'lane_actors', None)
    if lanes is not None and actors is not None:
        # G5 additive ownership contract: lane mode changes only for authorized actors.
        _, info = b.env.step(cmd,lanes=lanes,lane_actors=actors())
    else:
        _, info = b.env.step(cmd,lanes=lanes)  # The only actuator path: CommandOwnership.
    record = getattr(controller, 'observe_lane', None)
    if record is not None:
        record(info)
    interval = info['interval']
    road, source = interval['speed_deficit_vehicle_seconds'], interval['source_wait_vehicle_seconds']
    b.totals['road_delay_vs'] += road
    b.totals['source_wait_vs'] += source
    b.totals['team_delay_vs'] += road+source
    for key,field in (('collisions','collision_involvements'),('teleports','starting_teleports'),
                      ('ttc_below_3_vs','ttc_below_3_vehicle_seconds'),('hard_brake','braking_vehicle_seconds')):
        b.totals[key] += interval.get(field,0.)
    b.observe(b.env)
    if info['done'] or (any(info['lane_commands'].values()) and lanes is None) or any(interval.get(k,0) for k in
                ('collision_involvements','starting_teleports','dropped_vehicle_steps')):
        raise ValueError('early termination/safety/slot/lane anomaly')
    return audit


def advance(b, until, **kwargs):
    while b.env.now < until:
        tick(b,**kwargs)


def motion_changed(reference, actual, tolerance=1e-6):
    """Compare actual physical observations, not requested or safe-command deltas."""
    if set(reference) != set(actual):
        return True
    return any(abs(reference[v][i]-actual[v][i]) > tolerance for v in reference for i in (0,1))


def motion(rows):
    return {r['vid']:[r['s_m'],r['v_mps']] for r in rows}
