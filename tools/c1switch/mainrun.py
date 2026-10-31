"""Write-once cpuQ episode jobs and isolated full-prefix counterfactual children."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .design import BUDGET, CONTROLLERS, measurement_indices, worlds
from .episode import ARMS, episode
from .rerun import replay
from .rpc import FilePredictor
from .storage import Ledger, exclusive, write

ROOT = Path('/home/ubuntu/c1switch20261031/main_v4')


class MainLedger(Ledger):
    def __init__(self, arm):
        super().__init__(ROOT)
        self.arm = arm

    def charge(self, ident, sim_s, stage):
        import fcntl
        # All dispatchers use this interprocess mutex, including rerun children.
        with (ROOT/'ACCOUNTING.mutex').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if (ROOT/'STOP').exists():
                raise ValueError('durable main STOP')
            used = sum(r['sim_s'] for r in self.rows() if r['id'].startswith(self.arm+'_'))
            if stage != 'main' or used+sim_s > BUDGET[self.arm]:
                raise ValueError('per-arm registered budget exceeded')
            super().charge(ident,sim_s,stage)


def backend(world, arm, ident, inputs):
    from .main_backend import MainBackend
    path = ROOT/'jobs'/ident
    path.mkdir()  # No automatic retry of partial jobs.
    write(path/'START.json',dict(world=world,arm=arm,pid=os.getpid(),id=ident))
    return MainBackend(path,world,MainLedger(arm),'main',ident,inputs)


def branch(spec, inputs):
    row = replay(lambda:backend(spec['world'],spec['arm'],spec['id'],inputs),
                 spec['schedule'],spec['alternative'],expected_prefix=spec['prefix'])
    row.pop('belief')  # Evaluation output has no role in inference.
    write(ROOT/'jobs'/spec['id']/'RESULT.json',row)


def run(world, arm, inputs):
    ident = f'{arm}_s{world["seed"]}_episode'
    env = backend(world,arm,ident,inputs)
    path = ROOT/'jobs'/ident
    try:
        predictor = None
        if arm in ('wm','bb'):
            binding = json.loads((ROOT/f'{arm}_BINDING.json').read_text())
            predictor = FilePredictor(ROOT/'rpc'/arm,binding,timeout_s=900)
        def measurement(schedule,prefix):
            results = {}
            j = len(schedule)
            if j not in measurement_indices(world['seed']):
                raise ValueError('unregistered measurement clock')
            for c in CONTROLLERS:
                bid = f'{arm}_s{world["seed"]}_j{j}_{c}'
                spec = dict(id=bid,world=world,arm=arm,schedule=schedule,alternative=c,prefix=prefix)
                target = path/f'branch_j{j}_{c}.json'
                write(target,spec)
                with (path/f'branch_j{j}_{c}.log').open('x') as log:
                    subprocess.run([sys.executable,'-B','-u','-m','tools.c1switch.mainrun',
                        'branch','--spec',str(target)],stdout=log,stderr=subprocess.STDOUT,check=True)
                results[c] = json.loads((ROOT/'jobs'/bid/'RESULT.json').read_text())
            return results
        def receipt(kind,row):
            write(path/f'{kind}_j{row["j"]}.json',row)
        result = episode(env,arm,world,predictor=predictor,measurement=measurement,receipt=receipt)
        write(path/'RESULT.json',result)
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode',choices=('episode','branch'))
    p.add_argument('--arm',choices=ARMS)
    p.add_argument('--seed',type=int)
    p.add_argument('--spec',type=Path)
    args = p.parse_args()
    from .sumo import setup
    inputs = setup()
    binding = json.loads((ROOT/'BINDING.json').read_text())
    if binding['total_sim_s'] != 478410 or not binding['main_authorized']:
        raise ValueError('main authorization binding')
    try:
        if args.mode == 'branch':
            branch(json.loads(args.spec.read_text()),inputs)
        else:
            world = next(w for w in worlds('main') if w['seed'] == args.seed)
            task = ROOT/'tasks'/f'{args.arm}_s{args.seed}'
            with exclusive(task):
                run(world,args.arm,inputs)
    except BaseException as exc:
        write(ROOT/'STOP',dict(reason=str(exc),pid=os.getpid(),mode=args.mode,arm=args.arm,seed=args.seed))
        raise


if __name__ == '__main__':
    main()
