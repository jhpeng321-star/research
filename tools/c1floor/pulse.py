"""Exactly verified instantaneous +0.01m/s state pulse, full-prefix reruns."""
import json
import os
from pathlib import Path
import sys
import fcntl
import traceback

BASE=Path('/root/c1floor20261031')
sys.path.insert(0,str(BASE))
import fallback as F
from core import write,digest
ROOT=BASE/'pulse'
F.ROOT=ROOT


def pulse(api,vid):
    ids=sorted(api.vehicle.getIDList())
    speeds={v:api.vehicle.getSpeed(v) for v in ids}
    speed=speeds[vid];accel=api.vehicle.getAcceleration(vid)
    position=api.vehicle.getLanePosition(vid)
    api.vehicle.setPreviousSpeed(vid,speed+.01,accel)
    after=api.vehicle.getSpeed(vid)
    assert abs((after-speed)-.01)<1e-10,'instantaneous pulse amplitude failed'
    assert api.vehicle.getAcceleration(vid)==accel,'acceleration changed'
    assert api.vehicle.getLanePosition(vid)==position,'position changed'
    assert all(api.vehicle.getSpeed(v)==speeds[v] for v in ids if v!=vid),'another vehicle changed'
    return dict(target=vid,speed_before=speed,speed_after=after,delta_mps=after-speed,
                acceleration_before_after=accel,lane_position_before_after=position,
                other_vehicle_speeds_unchanged=len(ids)-1,api='setPreviousSpeed',
                persistent_speed_override=False,pulse_at_s=300.,consumed_by_native_tick_s=300.5)


def reserve(ident):
    rows=[]
    for p in (BASE/'accounting.jsonl',BASE/'fallback/accounting.jsonl',ROOT/'accounting.jsonl'):
        if p.exists():rows.extend(json.loads(s) for s in p.read_text().splitlines())
    assert not any(r['id']==ident for r in rows)
    assert sum(r['sim_s'] for r in rows)+1200<=30000
    with (ROOT/'accounting.jsonl').open('a') as f:
        f.write(json.dumps(dict(id=ident,sim_s=1200,source_sha256=digest(__file__),fallback_sha256=digest(F.__file__)))+'\n')
        f.flush();os.fsync(f.fileno())


def execute(index):
    import torch,traci
    from traffic_shepherd.c4 import branching as B
    from traffic_shepherd.c4.coopmeas import TrajRecorder
    from traffic_shepherd.c4.env import CavSumoEnv
    from worker import signature
    torch.set_num_threads(1)
    assert os.getpriority(os.PRIO_PROCESS,0)==19 and os.environ['CUDA_VISIBLE_DEVICES']==''
    w=F.world(index);ident=f"pulse_s{w['seed']}";out=BASE/'jobs'/ident
    assert not out.exists();F.wait();reserve(ident);out.mkdir()
    original=BASE/'jobs'/f"fallback_b_s{w['seed']}"
    reference=json.loads((original/'result.json').read_text())
    assert reference['ok'] and all(v['identical'] for v in reference['reproduction'].values())
    assert digest(original/'reference.npz')==reference['outputs']['reference.npz']
    counter=F.PhysicsCounter(traci,1200);traci.simulationStep=counter;env=None;effects=[]
    write(out/'START.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],world=w,
                               source_sha256=digest(__file__),reference_receipt_sha256=digest(original/'result.json')))
    try:
        for k,vid in enumerate(reference['targets']):
            F.wait();sub=out/f'alt{k}';sub.mkdir()
            env=CavSumoEnv(k_slots=192,controller='direct',extra_sumo_cli=B.SAVE_STATE_CLI,
                           scenario_root=sub/'scenarios',physics_log=False,population='textbook')
            env.reset(seed=w['seed'],demand_vph=w['demand'],penetration=w['penetration'],
                      warmup_s=300,horizon_s=301,controller='direct')
            assert signature(env)==reference['initial_signature']
            assert F.candidates(env)==reference['targets']
            before=counter.ticks;effect=pulse(env.api,vid);assert counter.ticks==before
            write(out/f'PULSE{k}.json',effect);effects.append(effect)
            rec=TrajRecorder()
            for _ in range(300):
                _,info=env.step(None);assert not info['done'];rec.add(env,float(info['time_s']))
            rec.save(out/f'alt{k}.npz',env);env.close();env=None
        assert counter.ticks*.5==1200
        write(out/'result.json',dict(ok=True,world=w,actual_sim_s=1200,targets=reference['targets'],
              initial_signature=reference['initial_signature'],pulses=effects,
              original_reference=str(original/'reference.npz'),original_reference_sha256=digest(original/'reference.npz'),
              original_receipt_sha256=digest(original/'result.json'),reproduction=reference['reproduction'],
              outputs={p.name:digest(p) for p in out.glob('*.npz')}))
        print('COMPLETE',ident,flush=True)
    except BaseException:
        write(out/'failure.json',dict(actual_sim_s=counter.ticks*.5,traceback=traceback.format_exc()))
        write(ROOT/'STOP',dict(job=ident,reason='pulse failure'));raise
    finally:
        if env is not None:env.close()
        traci.simulationStep=counter.step


if __name__=='__main__':
    with (ROOT/'RUN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if sys.argv[1]=='canary':execute(0)
        else:
            assert json.loads((BASE/'jobs/pulse_s969400/result.json').read_text())['ok']
            for index in range(1,6):execute(index)
