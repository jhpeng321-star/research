"""Owner-authorized fallback with immutable precharges and actual tick counting."""
from pathlib import Path
import fcntl
import json
import os
import sys
import time
import traceback

BASE=Path('/root/c1floor20261031')
ROOT=BASE/'fallback'
R=Path('/root/autodl-tmp/c4/speedhead20261030/e3final')
W=Path('/root/wmdata2')
sys.path[:0]=[str(BASE),str(W),str(R/'runtime/src'),str(R/'runtime'),str(W/'tools/speedhead')]
from core import audit, digest, write, compare_tapes, seed_guard
sys.addaudithook(audit)


def wait():
    while True:
        if (ROOT/'STOP').exists():raise RuntimeError('fallback STOP')
        if os.getloadavg()[0]<=20:return
        p=Path('/sys/fs/cgroup/cpu.stat')
        def usage():return int(dict(s.split() for s in p.read_text().splitlines())['usage_usec'])
        before=usage();t=time.monotonic();time.sleep(5)
        if (usage()-before)/1e6/(time.monotonic()-t)<23:return
        time.sleep(60)


def reserve(ident,charge):
    old=[json.loads(s) for s in (BASE/'accounting.jsonl').read_text().splitlines()]
    ledger=ROOT/'accounting.jsonl'
    rows=[json.loads(s) for s in ledger.read_text().splitlines()] if ledger.exists() else []
    assert not any(r['id']==ident for r in rows),'no duplicate dispatch'
    assert sum(r['sim_s'] for r in old+rows)+charge<=30000
    with ledger.open('a') as f:
        f.write(json.dumps(dict(id=ident,sim_s=charge,source_sha256=digest(__file__)))+'\n')
        f.flush();os.fsync(f.fileno())


def world(index):
    w=json.loads(Path('/root/wmcalib/data/design.json').read_text())['worlds'][index]
    assert w['seed']==969400+index
    from traffic_shepherd.seeds import require_development_case
    from traffic_shepherd.c4.seeds import check_case
    seed_guard(w['seed']);require_development_case(w['seed'],w['demand'],'exposed');check_case(w['seed'],w['demand'])
    return w


def candidates(env):
    f=env.frame
    rows=[(float(f.s[i]),env.builder.vehicle_ids[int(f.vid_index[i])])
          for i in range(len(f.present)) if f.present[i] and not f.is_cav[i] and f.s[i]<=-300]
    return [v for s,v in sorted(rows)[:2]]


class PhysicsCounter:
    def __init__(self,api,limit):
        self.api=api;self.step=api.simulationStep;self.limit=limit;self.ticks=0
        self.target=None;self.release=False;self.effects=[]

    def perturb(self,vid):
        speed=self.api.vehicle.getSpeed(vid)
        self.api.vehicle.setSpeed(vid,speed+.01)
        self.target=vid;self.release=True
        self.effects.append(dict(target=vid,speed_before=speed,speed_request=speed+.01,
                                 tick_before=self.ticks,duration_s=.5))

    def __call__(self,*args,**kw):
        if (self.ticks+1)*.5>self.limit:raise RuntimeError('precharge exhausted')
        result=self.step(*args,**kw);self.ticks+=1
        if self.release:
            self.effects[-1]['speed_after_tick']=self.api.vehicle.getSpeed(self.target)
            self.effects[-1]['release_tick']=self.ticks
            self.api.vehicle.setSpeed(self.target,-1);self.release=False
        return result


def execute(mode,index=0):
    import torch
    import traci
    import numpy as np
    from traffic_shepherd.c4 import branching as B
    from traffic_shepherd.c4.coopmeas import TrajRecorder
    from traffic_shepherd.c4.env import CavSumoEnv
    from worker import signature
    torch.set_num_threads(1)
    assert os.getpriority(os.PRIO_PROCESS,0)==19 and os.environ['CUDA_VISIBLE_DEVICES']==''
    ident='a_canary' if mode=='a' else f'b_s{969400+index}'
    out=BASE/'jobs'/('fallback_'+ident)
    assert not out.exists()
    wait();reserve(ident,900 if mode=='a' else 1800);out.mkdir()
    w=world(index);counter=PhysicsCounter(traci,900 if mode=='a' else 1800);env=None
    traci.simulationStep=counter
    write(out/'START.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],world=w,mode=mode))
    def reset(path):
        nonlocal env
        env=CavSumoEnv(k_slots=192,controller='direct',extra_sumo_cli=B.SAVE_STATE_CLI,
                       scenario_root=path/'scenarios',physics_log=False,population='textbook')
        env.reset(seed=w['seed'],demand_vph=w['demand'],penetration=w['penetration'],
                  warmup_s=300,horizon_s=301,controller='direct')
        return signature(env)
    def continuation(path):
        rec=TrajRecorder()
        for _ in range(300):
            _,info=env.step(None);rec.add(env,float(info['time_s']))
            assert not info['done']
        rec.save(path,env)
    try:
        initial=reset(out);targets=candidates(env)
        if mode=='a':
            B.save_state(env,out/'start.xml');snap=B.snapshot_env(env)
        continuation(out/'reference.npz')
        ref=BASE/'reference'/f"s{w['seed']}_native"/'traj_base.npz'
        comparison=compare_tapes(out/'reference.npz',ref)
        write(out/'REPRODUCTION.json',comparison)
        assert all(v['identical'] for v in comparison.values()),'original reference reproduction failed'
        if mode=='a':
            wait();B.load_state(env,out/'start.xml',snap);B.prepare_branch_mode(env)
            assert signature(env)==initial
            continuation(out/'same_seed.npz')
            same=compare_tapes(out/'reference.npz',out/'same_seed.npz')
            write(out/'IDENTITY.json',same)
            assert all(v['identical'] for v in same.values()),'COOPMEASURE path fails on registered world'
        else:
            assert len(targets)==2,'no geometry-qualified pair; no replacement'
            for k,vid in enumerate(targets):
                env.close();wait();sub=out/f'alt{k}';sub.mkdir()
                assert reset(sub)==initial,'trunk signature differs before perturbation'
                assert candidates(env)==targets
                counter.perturb(vid)
                continuation(out/f'alt{k}.npz')
                write(out/f'PERTURB{k}.json',dict(**counter.effects[-1],delta_mps=.01,
                       original_seed=w['seed'],initial_signature=initial))
        result=dict(ok=True,mode=mode,world=w,actual_sim_s=counter.ticks*.5,targets=targets,
                    perturbations=counter.effects,
                    initial_signature=initial,reproduction=comparison,
                    outputs={p.name:digest(p) for p in out.glob('*.npz')})
        write(out/'result.json',result);print(json.dumps(result),flush=True)
    except BaseException:
        write(out/'failure.json',dict(actual_sim_s=counter.ticks*.5,traceback=traceback.format_exc()))
        raise
    finally:
        if env is not None:env.close()
        traci.simulationStep=counter.step


if __name__=='__main__':
    with (ROOT/'RUN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if sys.argv[1]=='a':execute('a')
        else:
            assert (BASE/'jobs/fallback_a_canary/failure.json').exists()
            for i in ([0] if sys.argv[1]=='b_canary' else range(1,11)):
                if i>0:assert json.loads((BASE/'jobs/fallback_b_s969400/result.json').read_text())['ok']
                execute('b',i)
