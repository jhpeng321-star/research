"""Canary, Step0 native-trunk branching, and gated nine-arm switching episodes."""
import argparse
import json
import os
from pathlib import Path
import random
import time
import traceback

from .core import branch_all, choose
from .design import CONTROLLERS, checked, worlds
from .rpc import FilePredictor
from .storage import Ledger, exclusive, write


def run(root, mode, world, arm=None):
    from .sumo import Backend, setup
    inputs = setup()  # All imports/dependencies checked before any precharge.
    if not json.loads((root/'EXPOSURE.json').read_text())['pass']:
        raise ValueError('fresh-world exposure not verified')
    if mode != 'canary' and not json.loads((root/'jobs/canary_s969100/result.json').read_text())['pass']:
        raise ValueError('native restore canary required')
    ident = f'{mode}_s{world["seed"]}'+(f'_{arm}' if arm else '')
    out = root/'jobs'/ident
    if out.exists() or (root/'STOP').exists():
        raise ValueError('prior attempt/STOP; no redispatch')
    if mode == 'main':
        if not json.loads((root/'STEP0_GATE.json').read_text())['pass']:
            raise ValueError('Step0 headroom gate')
        binding = json.loads((root/'MODEL_SELECTION.json').read_text())
        if not binding.get('registered_before_switching'):
            raise ValueError('model rule not resolved/registered')
    out.mkdir(parents=True)
    write(out/'START.json', dict(world=world, arm=arm, mode=mode, pid=os.getpid(), unix=time.time()))
    backend = None
    try:
        backend = Backend(out, world, Ledger(root), 'step0' if mode != 'main' else 'main', ident, inputs)
        if mode == 'canary':
            start = backend.save('canary')
            backend.advance(90)
            live, live_tape = backend.metrics(), backend.digest.hexdigest()
            backend.load(start)
            backend.advance(90)
            restored, restored_tape = backend.metrics(), backend.digest.hexdigest()
            passed = live == restored and live_tape == restored_tape
            result = dict(pass_=passed, live=live, restored=restored,
                          live_tape_sha256=live_tape, restored_tape_sha256=restored_tape)
            result['pass'] = result.pop('pass_')
            write(out/'result.json', dict(result, actual_sim_s=backend.actual))
            if not passed:
                write(root/'STOP', dict(reason='native save/load bitwise gate failed', job=ident))
            return
        decisions = []
        predictor = None
        if arm in ('wm', 'bb'):
            predictor = FilePredictor(root/'rpc'/arm, binding[arm])
        for j in range(10):
            costs = None
            incumbent = backend.handover.name
            belief = backend.belief.payload()
            if predictor:
                costs = predictor.predict(belief, f'{ident}_d{j}')
                pick = choose(costs, backend.handover.name)
            elif mode == 'step0':
                pick = 'native'
            elif arm == 'random':
                pick = random.Random(f'0:{world["seed"]}:{j}').choice(CONTROLLERS)
            elif arm in CONTROLLERS:
                pick = arm
            elif arm != 'oracle':
                raise ValueError('unknown arm')
            if mode == 'step0' or arm in ('wm', 'bb', 'oracle'):
                branches, continuations = branch_all(backend, horizon=60 if arm == 'oracle' else 90,
                                                     repeat_native=mode == 'step0' and j == 0)
                if arm == 'oracle':
                    pick = choose({c:branches[c]['60']['team_delay_vs'] for c in CONTROLLERS}, incumbent)
                row = dict(decision=j, t_s=300+60*j, pick=pick, belief=belief, predicted=costs, branches=branches)
                write(out/f'decision_{j:02d}.json', row)
                decisions.append(row)
                backend.load(continuations[pick])
            else:
                backend.select(pick)
                backend.advance(60)
        write(out/'result.json', dict(ok=True, world=world, arm=arm, metrics=backend.metrics(),
                                      switches=backend.handover.switches, actual_sim_s=backend.actual,
                                      decisions=len(decisions)))
    except BaseException:
        failure = dict(error=traceback.format_exc(), actual_sim_s=backend.actual if backend else None, job=ident)
        write(out/'failure.json', failure)
        write(root/'STOP', failure)
        raise
    finally:
        if backend:
            backend.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('canary', 'step0', 'main'))
    p.add_argument('--root', type=Path, default=Path('/root/c1switch20261031'))
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--arm', choices=('wm', 'bb', 'random', 'oracle')+CONTROLLERS)
    a = p.parse_args()
    root = checked(a.root)
    with exclusive(root):
        run(root, a.mode, worlds('main' if a.mode == 'main' else 'step0')[a.index], a.arm)


if __name__ == '__main__':
    main()
