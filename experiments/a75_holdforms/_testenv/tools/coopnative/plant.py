"""Frozen SUMO environment without importing/loading any MAPPO actor for learning."""
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import numpy as np
import torch
from .protocol import canonical, checked, sha, guard, install_seed_scope, team_delay, WINDOW, approval_line
from .execution import Controller
from .policy import Policy, encode

SOURCE_SHA = 'b3c031adb6bc2b5c2bca160bea308f83ba5567b97c1cd0e853eddcd8367a8193'
METRICS = ('team_delay_vs','road_delay_vs','source_wait_vs','collisions','teleports','ttc_below_3_vs','hard_brake')


def setup(runtime):
    if os.getpriority(os.PRIO_PROCESS,0) != 19 or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('nice19 CPU only')
    if importlib.metadata.version('eclipse-sumo') != '1.27.1':
        raise ValueError('frozen SUMO required')
    import sumo
    home = str(Path(sumo.__file__).parent)
    os.environ['SUMO_HOME'] = home
    os.environ['PATH'] = home+'/bin:'+str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    root = checked(runtime)/'runtime'
    manifest = root/'SOURCE_MANIFEST.json'
    if sha(manifest) != SOURCE_SHA:
        raise ValueError('runtime manifest mismatch')
    from .protocol import FORBIDDEN
    entries = json.loads(manifest.read_text())
    for name, expected in entries.items():
        if not FORBIDDEN.search(name) and sha(root/name) != expected:
            raise ValueError('runtime mismatch '+name)
    sys.path[:0] = [str(root/'src'),str(root)]
    torch.set_num_threads(1)
    install_seed_scope()


def observations(env):
    """CAV self, range-limited CAV sensing, and existing RSU detections only.

    Frame kinematics are sensor measurements; no generator/driver channels are read.
    The projection is explicit so changing an unsensed HDV cannot affect an actor.
    """
    from traffic_shepherd.c4.state import rsu_detections
    f, ids = env.frame, env.builder.vehicle_ids
    cav = np.flatnonzero(f.present & f.is_cav)
    detected = {r['vehicle_id'] for r in rsu_detections(f,ids)}
    rows = []
    for i in np.flatnonzero(f.present):
        vid = ids[int(f.vid_index[i])]
        if not f.is_cav[i] and vid not in detected and not any(abs(float(f.s[i]-f.s[j])) <= 100 for j in cav):
            continue
        rows.append(dict(vid=vid,s_m=float(f.s[i]),v_mps=float(f.v[i]),
                         lane=int(f.lane_carry[i] if f.junction[i] else f.lane[i]),
                         is_cav=bool(f.is_cav[i]),length_m=float(env.api.vehicle.getLength(vid))))
    return sorted(rows,key=lambda r:r['vid'])


def accumulate(totals, interval):
    totals['road_delay_vs'] += interval['speed_deficit_vehicle_seconds']
    totals['source_wait_vs'] += interval['source_wait_vehicle_seconds']
    totals['team_delay_vs'] += team_delay(interval)
    for key, field in (('collisions','collision_involvements'),('teleports','starting_teleports'),
                       ('ttc_below_3_vs','ttc_below_3_vehicle_seconds'),('hard_brake','braking_vehicle_seconds')):
        totals[key] += interval.get(field,0.)


def episode(job):
    """One precharged episode. Durable claim exists BEFORE any SUMO construction."""
    approval_line(job['approval_line'])
    setup(job['runtime'])
    guard(job['world'],job['role'])
    path = checked(job['path'])
    path.mkdir(parents=True,exist_ok=True)
    allowed = job.get('sim_s',1200)
    stop = job.get('stop',1200)
    with (path/'CLAIM.json').open('x') as stream:
        stream.write(canonical(dict(pid=os.getpid(),job=job['id'],charged=allowed)))
    world = job['world']
    random.seed(world['seed'])
    np.random.seed(world['seed'])
    torch.manual_seed(world['seed'])
    generator = torch.Generator().manual_seed(job['noise_seed'])
    policy = Policy()
    policy.load_state_dict(job['weights'])
    policy.eval()
    from traffic_shepherd.c4.env import CavSumoEnv
    import traci
    from .storage import write
    env, original, actual = None, traci.simulationStep, 0.
    def counted(*args,**kwargs):
        nonlocal actual
        if actual+.5 > allowed:
            raise ValueError('physics exceeds precharge')
        before = traci.simulation.getTime()
        result = original(*args,**kwargs)
        actual += traci.simulation.getTime()-before
        return result
    traci.simulationStep = counted
    try:
        env = CavSumoEnv(k_slots=192,controller='direct',ttc=True,physics_log=False,
                         population='textbook',scenario_root=path/'scenarios')
        physical = hashlib.sha256()
        def physics(env):
            raw = env.history.raw(3)
            for key in ('kin','mask','vid_index'):
                physical.update(np.ascontiguousarray(raw[key][-2:]).tobytes())
        env.reset(seed=world['seed'],demand_vph=world['demand'],penetration=world['penetration'],
                  warmup_s=300,horizon_s=901,controller='direct',warmup_hook=physics)
        controller, steps, metrics = Controller(), [], {}
        totals = dict.fromkeys(METRICS,0.)
        prefix = hashlib.sha256()
        prefix.update(physical.digest())
        comparator = None
        if job['arm'] in ('mappo','ctrl0'):
            from .comparator import Comparator
            comparator = Comparator(job['arm'],job['comparators'])
        requests_count, overrides = 0, 0
        with gzip.open(path/'observable.jsonl.gz','xt',encoding='utf-8') as tape:
            for t in range(300,stop,WINDOW):
                sample = policy.sample(encode(observations(env)),generator,
                                       deterministic=job['arm'] != 'train')
                if job['arm'] == 'native':
                    sample['action'] = [[0.,0.] for _ in sample['action']]
                branch = job.get('branch')
                if branch and t <= branch['time_s']:
                    expected = branch['prefixes'][str(t)]
                    if physical.hexdigest() != expected['physical'] or canonical(totals) != expected['metrics']:
                        raise ValueError('bitwise full-prefix gate failed before intervention')
                    sample = branch['samples'][str(t)]
                    if t == branch['time_s']:
                        sample = json.loads(canonical(sample))
                        for i,vid in enumerate(sample['context']['members']):
                            if vid in branch['members'] and vid not in branch['active']:
                                sample['action'][i] = [0.,0.]
                start_prefix = dict(physical=physical.hexdigest(),metrics=canonical(totals))
                controller.hold(sample,t)
                reward = 0.
                for now in range(t,t+WINDOW):
                    rows = observations(env)
                    info, audit = comparator.step(env) if comparator else controller.step(env,rows,now)
                    physics(env)
                    reward -= team_delay(info['interval'])
                    accumulate(totals,info['interval'])
                    commands = {a['vid']:a for a in audit}
                    for r in rows:
                        a = commands.get(r['vid'])
                        issued = getattr(getattr(env,'ownership',None),'_last',{}).get(r['vid'])
                        # Physical actuator records, not pre-ownership requested commands.
                        acceleration = issued.issued_acceleration_mps2 if issued is not None else None
                        owner = issued.owner if issued is not None else 'native'
                        r.update(controller=('setpoint' if a else job['arm']) if owner!='native' else 'native',
                                 issued_acceleration_mps2=acceleration)
                    tape.write(canonical(dict(time_s=now,vehicles=rows,actions=audit))+'\n')
                    requests_count += len(audit)
                    overrides += sum(a['clamped'] for a in audit)
                    if (info['done'] and now+1 != stop) or info['interval'].get('dropped_vehicle_steps',0) or any(info['lane_commands'].values()):
                        raise ValueError('early termination/slot/lane anomaly')
                    if now+1 in (900,1200):
                        metrics[str(now+1-300)] = dict(totals)
                    if branch and now+1-branch['time_s'] in (120,300,600):
                        metrics['branch'+str(now+1-branch['time_s'])] = dict(totals)
                steps.append(dict(time_s=t,sample=sample,reward=reward,prefix=start_prefix))
        result = dict(id=job['id'],world=world,arm=job['arm'],steps=steps,metrics=metrics,
                      actual_sim_s=actual,charged_sim_s=allowed,prefix_sha256=prefix.hexdigest(),physical_sha256=physical.hexdigest(),
                      requests=requests_count,overrides=overrides,trace_sha256=sha(path/'observable.jsonl.gz'))
        write(path/'RESULT.json',result)
        return result
    finally:
        if env is not None:
            env.close()
        traci.simulationStep = original
        write(path/'CLOSE.json',dict(actual_sim_s=actual,charged_sim_s=allowed))
