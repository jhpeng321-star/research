"""MAIN23:25 immutable recovery, frozen scientific adapter and full-rerun plant.

Only unfinished physics is rerun. Cached picks must reproduce their complete
observable payload and prefix. Evaluation branches are scheduled after episode
physics; they never affect a pick. All abandoned physics remains charged.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .design import BUDGET, CONTROLLERS, measurement_indices, worlds
from .episode import ARMS, episode
from .recovery import plan
from .rpc import FilePredictor
from .storage import Ledger, canonical, exclusive, write

TOTAL = 499260
DROPPED = (969125, 969126, 969127)
GPU_CAP = 14400
ROOT = Path(os.environ.get('C1SWITCH_MAIN_ROOT','/home/ubuntu/c1switch20261031/recovery_v8'))


def amended_plan(audit):
    old = plan(audit)
    out = dict(old)
    out['per_arm'] = {a:dict(r) for a,r in old['per_arm'].items()}
    r = out['per_arm']['random']
    if r['charged_sim_s'] != 0:
        raise ValueError('random trim requires prospectively unstarted arm')
    r['remaining_sim_s'] -= 2700
    r['projected_total_sim_s'] -= 2700
    out['projected_total_sim_s'] -= 2700
    if out['projected_total_sim_s'] != TOTAL:
        raise ValueError('unexpected recovery projection')
    prior_gpu = sum(x['cumulative_gpu_s'] for x in old['gpu'].values())
    if prior_gpu >= GPU_CAP:
        raise ValueError('shared owner GPU cap exhausted')
    out.update(status='CPU_VALIDATION_REQUIRED',dispatch_authorized=False,
               reasons=[],dropped_random_seeds=list(DROPPED),shared_gpu_cap_s=GPU_CAP,
               prior_gpu_s=prior_gpu,remaining_gpu_s=GPU_CAP-prior_gpu)
    return out


def imported_rows(audit):
    partial = {x['id'] for x in audit['cpuQ']['jobs'] if not x['result_present']}
    out = []
    for row in audit['cpuQ']['accounting']:
        row = dict(row)
        for ident in partial:
            if row['id'].startswith(ident+'_'):
                arm,rest = row['id'].split('_',1)
                row['id'] = arm+'_abandoned_v7_'+rest
                break
        out.append(row)
    if len({r['id'] for r in out}) != len(out):
        raise ValueError('duplicate recovery debit')
    return out


def tasks():
    return [(a,w) for a in ('bb','wm')+ARMS[2:] for w in worlds('main')
            if not (a=='random' and w['seed'] in DROPPED)]


def verify_pick(old, new):
    for key in ('j','pick','incumbent','belief_sha256','prefix','predicted_costs'):
        if old[key] != new[key]:
            raise ValueError('replayed committed pick mismatch: '+key)


def reusable_branch(row, spec):
    if row['prefix'] != spec['prefix'] or not all(k in row for k in ('60','90','end')):
        raise ValueError('retained branch/live prefix mismatch')
    return row


class RecoveryLedger(Ledger):
    def __init__(self, arm):
        super().__init__(ROOT)
        self.arm = arm

    def charge(self, ident, sim_s, stage):
        import fcntl
        with (ROOT/'ACCOUNTING.mutex').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            binding = json.loads((ROOT/'BINDING.json').read_text())
            used = sum(r['sim_s'] for r in self.rows() if r['id'].startswith(self.arm+'_'))
            if (ROOT/'STOP').exists() or stage != 'main':
                raise ValueError('recovery durable STOP or invalid stage')
            if used+sim_s > binding['per_arm'][self.arm]:
                raise ValueError('recovery per-arm budget exceeded')
            if sum(r['sim_s'] for r in self.rows())+sim_s > TOTAL:
                raise ValueError('registered recovery total exceeded')
            super().charge(ident,sim_s,stage)


class Predictor:
    def __init__(self, arm):
        self.arm = arm
        self.binding = json.loads((ROOT/(arm+'_BINDING.json')).read_text())
        self.live = FilePredictor(ROOT/'rpc'/arm,self.binding,timeout_s=1200)

    def predict(self, payload, ident):
        from .core import medians
        from .observable import anonymous_payload
        cached = ROOT/'cache'/self.arm/(ident+'.json')
        if cached.exists():
            record = json.loads(cached.read_text())
            if canonical(record['belief']) != canonical(anonymous_payload(payload)):
                raise ValueError('cached inference observable history mismatch')
            return medians(record['samples'])
        return self.live.predict(payload, ident)


def backend(world, arm, ident, inputs):
    from .main_backend import MainBackend
    path = ROOT/'jobs'/ident
    path.mkdir()
    write(path/'START.json',dict(world=world,arm=arm,pid=os.getpid(),id=ident))
    return MainBackend(path,world,RecoveryLedger(arm),'main',ident,inputs)


def run_episode(world, arm, inputs):
    ident = f'{arm}_s{world["seed"]}_episode'
    path = ROOT/'jobs'/ident
    env = backend(world,arm,ident,inputs)
    prior = ROOT/'prior_jobs'/ident
    try:
        def receipt(kind,row):
            if kind == 'pick' and (prior/f'pick_j{row["j"]}.json').exists():
                verify_pick(json.loads((prior/f'pick_j{row["j"]}.json').read_text()),row)
            write(path/f'{kind}_j{row["j"]}.json',row)
        def measurement(schedule,prefix):
            j = len(schedule)
            if j not in measurement_indices(world['seed']):
                raise ValueError('unregistered evaluation clock')
            results = {}
            for c in CONTROLLERS:
                bid = f'{arm}_s{world["seed"]}_j{j}_{c}'
                spec = dict(id=bid,world=world,arm=arm,schedule=schedule,
                            alternative=c,prefix=prefix)
                result_path = ROOT/'jobs'/bid/'RESULT.json'
                if result_path.exists():
                    results[c] = reusable_branch(json.loads(result_path.read_text()),spec)
                else:
                    write(ROOT/'specs'/(bid+'.json'),spec)
                    results[c] = dict(pending=True,prefix=prefix)
            return results
        result = episode(env,arm,world,predictor=Predictor(arm) if arm in ('wm','bb') else None,
                         measurement=measurement,receipt=receipt)
        write(path/'PHYSICS.json',result)
    finally:
        env.close()


def run_branch(spec, inputs):
    from .rerun import replay
    row = replay(lambda:backend(spec['world'],spec['arm'],spec['id'],inputs),
                 spec['schedule'],spec['alternative'],expected_prefix=spec['prefix'])
    row.pop('belief')
    write(ROOT/'jobs'/spec['id']/'RESULT.json',row)


def finalize(arm, world):
    path = ROOT/'jobs'/f'{arm}_s{world["seed"]}_episode'
    if (path/'RESULT.json').exists():
        return
    row = json.loads((path/'PHYSICS.json').read_text())
    for d in row['decisions']:
        if 'branches' not in d:
            continue
        for c in CONTROLLERS:
            bid = f'{arm}_s{world["seed"]}_j{d["j"]}_{c}'
            spec = dict(prefix=d['prefix'],alternative=c)
            d['branches'][c] = reusable_branch(json.loads((ROOT/'jobs'/bid/'RESULT.json').read_text()),spec)
        write(path/f'measurement_j{d["j"]}.json',d)
    write(path/'RESULT.json',row)


def supervisor():
    root = ROOT/'recovery'
    with exclusive(root):
        if (root/'START.json').exists() or (ROOT/'STOP').exists():
            raise ValueError('existing recovery dispatch or STOP')
        write(root/'START.json',dict(pid=os.getpid(),unix=time.time()))
        for phase in ('episodes','branches'):
            if phase == 'episodes':
                todo = [(f'{a}_s{w["seed"]}_episode',['episode','--arm',a,'--seed',str(w['seed'])])
                        for a,w in tasks() if not (ROOT/'jobs'/f'{a}_s{w["seed"]}_episode'/'RESULT.json').exists()]
            else:
                todo = [(p.stem,['branch','--spec',str(p)]) for p in sorted((ROOT/'specs').glob('*.json'))]
            active, done = {}, []
            while todo or active:
                if (ROOT/'STOP').exists():
                    return
                while todo and len(active)<14:
                    ident,args = todo.pop(0)
                    with (root/(ident+'.log')).open('x') as log:
                        p = subprocess.Popen([sys.executable,'-B','-u','-m','tools.c1switch.resume']+args,
                                             stdout=log,stderr=subprocess.STDOUT)
                    active[ident] = p
                    write(root/('OWN_'+ident+'.json'),dict(pid=p.pid,id=ident))
                for ident,p in list(active.items()):
                    code = p.poll()
                    if code is None:
                        continue
                    if code:
                        write(ROOT/'STOP',dict(reason='recovery child failed',id=ident,code=code))
                        return
                    done.append(ident)
                    del active[ident]
                for arm in ('bb','wm'):
                    if all((ROOT/'jobs'/f'{arm}_s{w["seed"]}_episode'/'pick_j9.json').exists() for w in worlds('main')):
                        if not (ROOT/'rpc'/arm/'COMPLETE').exists():
                            write(ROOT/'rpc'/arm/'COMPLETE',dict(all160_picks=True,unix=time.time()))
                write(root/'STATUS.json',dict(phase=phase,pending=len(todo),active={k:p.pid for k,p in active.items()},done=done))
                time.sleep(.5)
        for arm,world in tasks():
            finalize(arm,world)
        write(root/'COMPLETE.json',dict(unix=time.time(),episodes=125,branches=480))


def gpu_serve(root, binding_path):
    """One model allocation, charged against a prospectively partitioned shared cap."""
    import signal
    from .adapter import create
    from .core import medians
    from .design import sha
    from .rpc import validate_objective
    binding = json.loads(binding_path.read_text())
    validate_objective(binding)
    if sha(Path(__file__).with_name('adapter.py')) != binding['adapter_sha256']:
        raise ValueError('frozen adapter hash mismatch')
    cap = binding['recovery_allocation_s']
    if cap <= 0 or binding['shared_gpu_prior_s']+sum(binding['shared_allocations_s'].values()) > GPU_CAP:
        raise ValueError('shared GPU allocation exceeds owner cap')
    began = time.monotonic()
    with exclusive(root):
        if (root/'GPU_STARTED.json').exists():
            raise ValueError('allocation already started')
        write(root/'GPU_STARTED.json',dict(pid=os.getpid(),unix=time.time(),cap_s=cap))
        def expired(*args):
            raise TimeoutError('shared GPU allocation exhausted')
        signal.signal(signal.SIGALRM,expired)
        signal.setitimer(signal.ITIMER_REAL,cap)
        try:
            model = create(binding,device='cuda')
            reference = json.loads((root/'PARITY_INPUT.json').read_text())
            samples = model.predict(reference['belief'],K=8,horizon_s=60)
            if samples != reference['samples']:
                raise ValueError('gpuA anonymous-draw raw-K8 bitwise parity failed')
            write(root/'PARITY.json',dict(pass_=True,bitwise=True,unix=time.time()))
            write(root/'READY.json',dict(pid=os.getpid(),attestation=model.attestation,unix=time.time()))
            while not (root/'COMPLETE').exists():
                if (root/'STOP').exists():
                    raise ValueError('GPU STOP')
                for p in sorted(root.glob('*.request.json')):
                    result = p.with_name(p.name.replace('.request.','.result.'))
                    if result.exists():
                        continue
                    req = json.loads(p.read_text());key = req.pop('request_sha256')
                    if canonical(req) != key or req['binding'] != binding or 'validation_seed' in req:
                        raise ValueError('recovery RPC identity mismatch')
                    samples = model.predict(req['belief'],K=8,horizon_s=60)
                    medians(samples)
                    write(result,dict(request_sha256=key,binding=binding,samples=samples,
                                      elapsed_gpu_s=time.monotonic()-began))
                time.sleep(.1)
        except BaseException as exc:
            write(root/'STOP',dict(reason=str(exc),pid=os.getpid()))
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            write(root/'GPU_FINISHED.json',dict(pid=os.getpid(),elapsed_s=time.monotonic()-began,unix=time.time()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=('episode','branch','supervisor'))
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--spec',type=Path)
    args = parser.parse_args()
    try:
        binding = json.loads((ROOT/'BINDING.json').read_text())
        if binding['total_sim_s'] != TOTAL or binding['authorization'] != 'MAIN23:25':
            raise ValueError('recovery authorization mismatch')
        if args.mode == 'supervisor':
            supervisor()
            return
        from .sumo import setup
        inputs = setup()
        if args.mode == 'branch':
            run_branch(json.loads(args.spec.read_text()),inputs)
        else:
            world = next(w for a,w in tasks() if a==args.arm and w['seed']==args.seed)
            with exclusive(ROOT/'tasks'/f'{args.arm}_s{args.seed}'):
                run_episode(world,args.arm,inputs)
    except BaseException as exc:
        write(ROOT/'STOP',dict(reason=str(exc),pid=os.getpid()))
        raise


if __name__ == '__main__':
    main()
