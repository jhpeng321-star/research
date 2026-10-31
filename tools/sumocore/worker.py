"""One precharged observable-only simulation; safe on resume, never retries."""
from pathlib import Path
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import os
import sys
import time
import traceback

ROOT=Path('/home/ubuntu/sumocore20261031')
R=Path('/home/ubuntu/e3final/runtime')
sys.path[:0]=[str(R/'src'),str(R)]


def write(path,doc):
    with path.open('x') as f:
        json.dump(doc,f,indent=1,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def run(job):
    assert os.getpriority(os.PRIO_PROCESS,0)==19
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    from admission import amendments
    amendments()
    while os.getloadavg()[0]>24:
        time.sleep(20);amendments()
    os.environ['C4_POLICY_STOCHASTIC']='0';os.environ['C4_POLICY_ACCEL_NOISE']='0'
    os.environ['PATH']='/home/ubuntu/miniconda3/bin:'+os.environ['PATH']
    assert importlib.metadata.version('eclipse-sumo')=='1.27.1'
    rows=[json.loads(x) for x in (ROOT/'ledger.jsonl').read_text().splitlines()]
    assert len({r['id'] for r in rows})==len(rows)
    assert sum(r['charge'] for r in rows)<=100000
    assert any(r['id']==job['id'] and r['charge']>=job['horizon'] for r in rows)
    out=ROOT/'jobs'/job['id']
    out.mkdir(parents=True,exist_ok=False)
    write(out/'START.json',dict(job=job,pid=os.getpid(),unix=time.time(),
        code={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}))
    import torch
    torch.set_num_threads(1)
    from prepare_bundle import prepare
    from model_env import ObservableEnv
    env=None
    started=time.monotonic()
    try:
        bundle,config=prepare(job,out)
        with (out/'worker.log').open('x') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            env=ObservableEnv(k_slots=512,controller='native',physics_log=False,ttc=False,
                              population='textbook',use_subscriptions=False)
            env.initialize(bundle,config,seed=job['rng'],arm=job['arm'],out=out,horizon=job['horizon'])
            write(out/'initial_audit.json',env.initial_audit)
            policy=None
            if job['arm']=='mappo':
                from wms.control.policy import load_policy_fn
                checkpoint=Path('/home/ubuntu/mappospeed/production2/policy_latest.pt')
                assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()=='d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca'
                policy=load_policy_fn(checkpoint,device='cpu')
                if hasattr(policy,'reset'):policy.reset()
            for tick in range(job['horizon']):
                env.advance(policy)
                if tick%30==29:
                    # Append progress durably without replacing prior receipts.
                    with (out/'progress.jsonl').open('a') as f:
                        f.write(json.dumps(dict(actual_sim_s=env.actual_steps*.5,time=env.now))+'\n');f.flush();os.fsync(f.fileno())
            env.save(out)
            assert env.now==300+job['horizon']
            assert not env.lane_command_counts.get('issued')
            write(out/'result.json',dict(ok=True,job=job,actual_sim_s=env.actual_steps*.5,
                cumulative=env.cumulative,initial_audit=env.initial_audit,
                safety=dict(env.totals),commands=dict(env.command_counts),
                scientific_valid=not (env.totals.get('collision_involvements') or env.totals.get('starting_teleports')),
                wall_s=time.monotonic()-started,input_receipt=bundle['receipt']))
    except BaseException:
        actual=env.actual_steps*.5 if env is not None and hasattr(env,'actual_steps') else 0.
        # initialize() performs no simulationStep; a launched-but-unadvanced SUMO costs0.
        write(out/'failure.json',dict(job=job,actual_sim_s=actual,traceback=traceback.format_exc()))
        raise
    finally:
        if env is not None:env.close()
    return job['id']


if __name__=='__main__':
    plan=json.loads((ROOT/'plan.json').read_text())
    run(next(j for j in plan['jobs'] if j['id']==sys.argv[1]))
