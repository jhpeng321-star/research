"""Original native WMDATA rerun + fresh-process, RNG-only saved-state branches."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
import traceback

ROOT = Path('/root/c1floor20261031')
R = Path('/root/autodl-tmp/c4/speedhead20261030/e3final')
W = Path('/root/wmdata2')
sys.path[:0] = [str(ROOT), str(W), str(R/'runtime/src'), str(R/'runtime'),
                str(W/'tools/speedhead')]
from core import allowed, digest, write, seed_guard, rng_only, compare_tapes, audit
sys.addaudithook(audit)


def wait():
    while True:
        assert not (ROOT/'STOP').exists()
        if os.getloadavg()[0] <= 20:break
        # Host load includes workloads outside this 25-CPU container. Measure
        # its actual quota consumption; one worker leaves >1 quota CPU free.
        def usage():
            return int(dict(s.split() for s in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())['usage_usec'])
        before=usage();begin=time.monotonic();time.sleep(5)
        busy=(usage()-before)/1e6/(time.monotonic()-begin)
        if busy < 23:
            print('SPARE_CAPACITY',dict(host_load=os.getloadavg(),busy_cpus=busy),flush=True)
            break
        print('YIELD',dict(host_load=os.getloadavg(),busy_cpus=busy),flush=True)
        time.sleep(60)
    assert not (ROOT/'STOP').exists()


def signature(env):
    import numpy as np
    raw = env.history.raw(1)
    h = hashlib.sha256()
    for k in sorted(raw):
        h.update(k.encode()); h.update(np.asarray(raw[k]).tobytes())
    h.update(json.dumps(sorted((r.vehicle_id,r.depart_s,r.route) for r in env.requests)).encode())
    h.update(json.dumps(sorted(env.cav_ids)).encode())
    return h.hexdigest()


def run(job):
    import torch
    import numpy as np
    from traffic_shepherd.c4 import branching as B
    from traffic_shepherd.c4 import coopmeas as C
    from traffic_shepherd import simulation
    from traffic_shepherd.seeds import require_development_case
    from traffic_shepherd.c4.seeds import check_case
    from tools.ctrlarm.zero import run_e3
    from observer import Measurement, audit_lanes
    torch.set_num_threads(1)
    assert os.getpriority(os.PRIO_PROCESS,0) == 19
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    rows = [json.loads(s) for s in (ROOT/'accounting.jsonl').read_text().splitlines()]
    assert sum(r['sim_s'] for r in rows) <= 30000
    assert len([r for r in rows if r['id']==job['id'] and r['sim_s']==job['charge']])==1
    for seed in (job['seed'],*job['rng_seeds']):
        seed_guard(seed); require_development_case(seed,job['demand'],'exposed'); check_case(seed,job['demand'])
    out = ROOT/'jobs'/job['id']
    assert not out.exists()
    wait()
    out.mkdir()
    write(out/'START.json', dict(job=job,pid=os.getpid(),nice=19,start_ticks=Path('/proc/self/stat').read_text().split()[21]))
    # Count each actual .5 s simulation advance, including native warmup.
    import traci
    step = traci.simulationStep
    ticks = [0]
    def counted(*args,**kwargs):
        if .5*(ticks[0]+1) > job['charge']:
            raise RuntimeError('attempt precharge exhausted')
        result=step(*args,**kwargs)
        ticks[0]+=1
        return result
    traci.simulationStep=counted
    launch=simulation.start_sumo
    def start(*args,**kwargs):
        kwargs['extra_cli']=[*(kwargs.get('extra_cli') or []),*B.SAVE_STATE_CLI]
        return launch(*args,**kwargs)
    # Both env.reset's imported launch and branching.relaunch use identical flags.
    simulation.start_sumo=start
    class Capture(Measurement):
        def start(self,env):
            prior=env.__dict__.get('_after_physics')
            super().start(env)
            B.save_state(env,out/'start.xml')
            recorder,observe=env.recorder,env._after_physics
            env.recorder=None
            env.__dict__.pop('_after_physics',None)
            if prior is not None:env._after_physics=prior
            try:
                self.snap=B.snapshot_env(env)
                self.initial=signature(env)
            finally:
                env.recorder=recorder; env._after_physics=observe
    m=Capture(); m.path=out
    try:
        with audit_lanes() as lanes:
            result=run_e3(dict(job,arm='native',warmup=300,horizon=300),
                          {'policy':str(R/'inputs/policy_latest.pt')},out,measurement=m)
        assert all(v==0 for v in lanes.values()),lanes
        assert .5*ticks[0]==600, ticks
        ref=ROOT/'reference'/f"s{job['seed']}_native"
        original=json.loads((ref/'result.json').read_text())
        assert digest(ref/'traj_base.npz')==original['output_hashes']['traj_base.npz']
        comparison=compare_tapes(out/'traj_base.npz',ref/'traj_base.npz')
        write(out/'REPRODUCTION.json',comparison)
        assert all(r['identical'] for r in comparison.values()),'original recorded continuation not bitwise'
        env=m.env
        patched,n=rng_only((out/'start.xml').read_bytes())
        with (out/'start_rng_fresh.xml').open('xb') as f:
            f.write(patched);f.flush();os.fsync(f.fileno())
        branches=[]
        if job['canary']:
            branches.append(('same_seed',job['seed'],out/'start.xml'))
        branches += [(f'alt{k}',seed,out/'start_rng_fresh.xml') for k,seed in enumerate(job['rng_seeds'])]
        summaries={}
        for name,seed,state in branches:
            wait()
            env.__dict__.pop('_after_physics',None)
            B.restore_env(env,m.snap)  # run_e3.close cleared its own API handle
            env.seed=seed  # affects ONLY the fresh SUMO launch; restore reinstates world seed
            restore=B.load_state(env,state,m.snap)
            assert env.seed==job['seed'] and signature(env)==m.initial
            rec=C.TrajRecorder()
            # Match the frozen observer's genuine t300 boundary row if present.
            with np.load(out/'traj_base.npz',allow_pickle=False) as tape:
                has_initial=bool(np.any(tape['t']==300))
            if has_initial:
                raw=env.history.raw(1)
                mask=raw['mask'][-1];vi=raw['vid_index'][-1];kin=raw['kin'][-1]
                slots=np.flatnonzero(mask & (vi>=0));k=kin[slots]
                lane=np.where(k[:,1:4].max(-1)>.5,k[:,1:4].argmax(-1),-1)
                for key,val in dict(t=np.full(len(slots),300.),vid=vi[slots].astype(np.int32),
                                    s=k[:,0].astype(np.float32),v=k[:,4].astype(np.float32),
                                    a=k[:,5].astype(np.float32),lane=lane.astype(np.int8)).items():
                    rec.cols[key].append(val)
            safety=dict(collisions=0,teleports=0,dropped=0)
            for elapsed in range(300):
                _,info=env.step(None)
                rec.add(env,env.now)
                interval=info['interval']
                safety['collisions']+=interval.get('collision_involvements',0)
                safety['teleports']+=interval.get('starting_teleports',0)
                safety['dropped']+=interval.get('dropped_vehicle_steps',0)
                assert env.now==301+elapsed
            rec.save(out/f'{name}.npz',env)
            same=compare_tapes(out/'traj_base.npz',out/f'{name}.npz')
            if name=='same_seed':
                assert all(r['identical'] for r in same.values()),'saved RNG canary not bitwise'
            summaries[name]=dict(rng_seed=seed,restore=restore,safety=safety,
                identical=all(r['identical'] for r in same.values()),sha256=digest(out/f'{name}.npz'))
            write(out/f'{name}.json',summaries[name])
        assert .5*ticks[0]==job['charge']
        result=dict(ok=True,job=job,actual_sim_s=.5*ticks[0],reproduction=comparison,
                    branches=summaries,rng_elements_removed=n,initial_signature=m.initial,
                    saved_state_sha256=digest(out/'start.xml'),fresh_state_sha256=digest(out/'start_rng_fresh.xml'),
                    lane_audit=lanes,recorded_sha256=digest(ref/'traj_base.npz'))
        write(out/'result.json',result)
        return result
    except BaseException:
        write(out/'failure.json',dict(job=job,actual_sim_s=.5*ticks[0],traceback=traceback.format_exc()))
        raise
    finally:
        if hasattr(m,'env'):m.env.close()
        traci.simulationStep=step
        simulation.start_sumo=launch


if __name__=='__main__':
    run(json.loads(Path(sys.argv[1]).read_text()))
