"""Bounded own-child dispatch. cpuQ/gpuR world partitions are disjoint and fixed."""
import argparse
import os
import subprocess
import sys
import time

from tools.c1switch.design import checked
from tools.c1switch.storage import exclusive, write

from .design import assigned, worlds
from .runner import admitted


def run(root, host, workers):
    if host not in ('cpuQ', 'gpuR') or not 1 <= workers <= (14 if host == 'cpuQ' else 4):
        raise ValueError('unregistered host/CPU ceiling')
    binding = admitted(root)
    if os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise ValueError('nice19 required')
    if binding.get('cpu_host') != host:
        raise ValueError('runtime host binding mismatch')
    with exclusive(root/'supervisor'):
        # The prior incarnation's receipts are retained; recovery requires a new
        # prospective reconciliation, never silently skipping abandoned charges.
        if (root/'supervisor'/'START.json').exists() or (root/'STOP').exists():
            raise ValueError('existing dispatch/STOP; no automatic retry')
        write(root/'supervisor'/'START.json', dict(pid=os.getpid(), host=host, workers=workers))
        for phase in ('episodes', 'branches'):
            if phase == 'episodes':
                todo = [(f'{a}__s{w["seed"]}', ['episode', '--arm', a, '--seed', str(w['seed'])])
                        for a in binding['sim_s'] if a != 'canary'
                        for w in worlds() if assigned(w['seed']) == host]
            else:
                todo = [(p.stem, ['branch', '--spec', str(p)]) for p in sorted((root/'specs').glob('*.json'))]
            active = {}
            while todo or active:
                if (root/'STOP').exists():
                    # No peer signals and no kill-by-name. Own children finish
                    # their already-precharged interval and refuse their next one.
                    for process, log in active.values():
                        process.wait()
                        log.close()
                    raise ValueError('durable STOP')
                while todo and len(active) < workers:
                    if os.getloadavg()[0] > (28 if host == 'cpuQ' else 20):
                        break  # Yield spare CPU capacity without touching peer PIDs.
                    # Recheck authority/gates before every spawn, not just startup.
                    admitted(root)
                    ident, args = todo.pop(0)
                    log = (root/'logs'/(ident+'.log')).open('x')
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
                    process = subprocess.Popen([sys.executable, '-B', '-u', '-m',
                        'tools.c1switch2.runner', *args, '--root', str(root)],
                        stdout=log, stderr=subprocess.STDOUT, env=env)
                    active[ident] = (process, log)
                for ident, (process, log) in list(active.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    log.close()
                    del active[ident]
                    if code:
                        write(root/'STOP', dict(reason='child failed', id=ident, code=code))
                        break
                time.sleep(.25)
        write(root/'supervisor'/'COMPLETE.json', dict(host=host, pid=os.getpid()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--host', choices=('cpuQ', 'gpuR'), required=True)
    p.add_argument('--workers', type=int, required=True)
    args = p.parse_args()
    run(checked(args.root), args.host, args.workers)
