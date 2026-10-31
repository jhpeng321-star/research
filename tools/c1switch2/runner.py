"""Write-once episode/branch jobs; approval and dependency gates precede setup."""
import argparse
import importlib
import json
import os

from tools.c1switch.design import checked, sha
from tools.c1switch.storage import Ledger, exclusive, write

from .design import MODEL_ARMS, PROTOCOL, assigned, budget, guard, local_budget, worlds
from .episode import episode
from .gate import admission
from .rerun import replay


def load_plugin(spec):
    module, attribute = spec['entry'].split(':')
    mod = importlib.import_module(module)
    if sha(mod.__file__) != spec['sha256']:
        raise ValueError('dependency source changed')
    return getattr(mod, attribute)


class ArmLedger(Ledger):
    def __init__(self, root, arm, binding):
        super().__init__(root)
        self.arm, self.binding = arm, binding

    def charge(self, ident, sim_s, stage):
        import fcntl
        with (self.root/'ACCOUNTING.mutex').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            rows = self.rows()
            # Arm is encoded explicitly in the precharge ID, including RL names.
            used = sum(r['sim_s'] for r in rows if r['id'].startswith(self.arm+'__'))
            if (self.root/'STOP').exists() or stage != 'main':
                raise ValueError('durable STOP or wrong accounting stage')
            limits = local_budget(self.binding['cpu_host'], self.binding.get('include_phys') is True)
            if not ident.startswith(self.arm+'__') or used+sim_s > limits[self.arm]:
                raise ValueError('registered per-arm sim-s exhausted')
            if sum(r['sim_s'] for r in rows)+sim_s > sum(limits.values()):
                raise ValueError('registered total sim-s exhausted')
            # v7's Ledger hard-codes its old 500k cap. Preserve its append/fsync,
            # duplicate refusal and no-refund semantics under v2's bound total.
            if sim_s <= 0 or any(r['id'] == ident for r in rows):
                raise ValueError('invalid/duplicate reservation')
            with self.path.open('a', encoding='utf-8') as stream:
                import time
                stream.write(json.dumps(dict(id=ident, sim_s=sim_s, stage=stage, unix=time.time()))+'\n')
                stream.flush()
                os.fsync(stream.fileno())


def admitted(root, *, phase='main'):
    binding = json.loads((root/'BINDING.json').read_text())
    evidence = json.loads((root/'GATES.json').read_text())
    registration = root/'REGISTRATION.md'
    if sha(registration) != binding.get('registration_sha256'):
        raise ValueError('registered approval snapshot changed')
    status = admission(registration.read_text(encoding='utf-8'), binding, evidence, phase=phase)
    if not status['dispatch_authorized']:
        raise ValueError('S-C1SWITCH2 dispatch blocked: '+json.dumps(status))
    if binding['protocol'] != PROTOCOL or binding['sim_s'] != budget(binding.get('include_phys') is True):
        raise ValueError('runtime protocol mismatch')
    return binding


def backend(root, binding, arm, world, ident):
    guard(world)
    if (arm == 'canary' and binding['cpu_host'] != 'cpuQ') or (
            arm != 'canary' and assigned(world['seed']) != binding['cpu_host']):
        raise ValueError('world belongs to other CPU host')
    # Setup/imports that might allocate SUMO occur ONLY after admitted().
    runtime = load_plugin(binding['runtime'])(binding)
    path = root/'jobs'/ident
    path.mkdir()  # Partial jobs are never retried automatically.
    write(path/'START.json', dict(id=ident, world=world, arm=arm, pid=os.getpid()))
    return runtime.create_backend(path, world, ArmLedger(root, arm, binding), 'main', ident, arm=arm)


def run_episode(root, binding, arm, world):
    ident = f'{arm}__s{world["seed"]}__episode'
    env = backend(root, binding, arm, world, ident)
    path = root/'jobs'/ident
    try:
        predictor = load_plugin(binding['predictors'][arm])(binding, root) if arm in MODEL_ARMS else None
        selector = load_plugin(binding['selectors'][arm])(binding) if arm in ('fleet_ppo', 'cav_mappo') else None
        def receipt(kind, row):
            write(path/f'{kind}_j{row["j"]}.json', row)
        result = episode(env, arm, world, predictor=predictor, selector=selector, receipt=receipt)
        write(path/'PHYSICS.json', result)
        for item in result['measurement_specs']:
            alternatives = [(c, c) for c in binding['library']]
            if arm == 'cav_mappo':
                alternatives.append(('chosen_joint', item['chosen']))
            for controller, action in alternatives:
                bid = f'{arm}__s{world["seed"]}__j{item["j"]}__{controller}'
                write(root/'specs'/(bid+'.json'), dict(id=bid, arm=arm, world=world,
                    schedule=item['schedule'], prefix=item['prefix'], alternative=action))
    finally:
        env.close()


def run_branch(root, binding, spec):
    if spec['arm'] not in binding['sim_s'] or spec['arm'] == 'canary':
        raise ValueError('unregistered branch arm')
    row = replay(lambda: backend(root, binding, spec['arm'], spec['world'], spec['id']),
                 spec['schedule'], spec['alternative'], expected_prefix=spec['prefix'])
    write(root/'jobs'/spec['id']/'RESULT.json', row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('episode', 'branch', 'canary'))
    p.add_argument('--root', required=True)
    p.add_argument('--arm')
    p.add_argument('--seed', type=int)
    p.add_argument('--spec')
    a = p.parse_args()
    root = checked(a.root)
    binding = admitted(root, phase='canary' if a.mode == 'canary' else 'main')
    # All admission precedes plugin import, setup, process spawn or physics.
    if a.mode == 'canary':
        from .canary import run
        with exclusive(root/'tasks'/'canary'):
            run(root, binding)
        return
    if a.mode == 'episode':
        if a.arm not in binding['sim_s'] or a.arm == 'canary':
            raise ValueError('unregistered arm')
        world = next(w for w in worlds() if w['seed'] == a.seed)
        with exclusive(root/'tasks'/f'{a.arm}__s{a.seed}'):
            run_episode(root, binding, a.arm, world)
    else:
        spec = json.loads(checked(a.spec).read_text())
        with exclusive(root/'tasks'/spec['id']):
            run_branch(root, binding, spec)


if __name__ == '__main__':
    main()
