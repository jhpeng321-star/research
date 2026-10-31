"""Bounded spawn waves: precharge once, no automatic retry or peer signals."""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing
import os
import shutil
import time
import traceback

from .protocol import safe_path
from .storage import PrechargedEpisode, append


def admitted(root, workers):
    free_disk = shutil.disk_usage(root).free
    available = next(int(line.split()[1])*1024 for line in
                     safe_path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    return (available >= 8*1024**3 and free_disk >= 2*1024**3 and
            os.getloadavg()[0]+workers <= .8*os.cpu_count())


def job(state, world, args):
    import torch
    from .learning import Selector
    from .train import episode
    torch.set_num_threads(1)
    model = Selector()
    model.load_state_dict(state)
    budget = PrechargedEpisode(safe_path(args.out)/str(world['seed']), world, args.variant)
    try:
        transitions = episode(model, world, args, budget)
        return dict(ok=True, transitions=transitions, actual=budget.actual)
    except BaseException:
        return dict(ok=False, error=traceback.format_exc(), actual=budget.actual)


def wave(model, worlds, args, budget, deadline):
    root = safe_path(args.out)
    pending, results, cursor, failed, paused = {}, {}, 0, False, False
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        while cursor < len(worlds) or pending:
            stop = time.monotonic() >= deadline or (root/'STOP_REQUESTED').exists()
            if stop:
                failed = True
                if not (root/'STOP_REQUESTED').exists():
                    with (root/'STOP_REQUESTED').open('x', encoding='utf-8') as stream:
                        stream.write('24h wall deadline; no retry\n')
            while not failed and cursor < len(worlds) and len(pending) < args.workers:
                if not admitted(root, len(pending)+1):
                    if not paused:
                        append(root/'status.jsonl', dict(status='DISPATCH_PAUSED_RESOURCE_PRIORITY',
                               load_1m=os.getloadavg()[0], cores=os.cpu_count(),
                               max_load_fraction=.8, pending=len(pending), next_seed=worlds[cursor]['seed']))
                        paused = True
                    break
                if paused:
                    append(root/'status.jsonl', dict(status='DISPATCH_RESUMED_RESOURCE_PRIORITY',
                           load_1m=os.getloadavg()[0], cores=os.cpu_count(), pending=len(pending)))
                    paused = False
                world = worlds[cursor]
                budget.reserve(world, args.variant)
                future = pool.submit(job, model.state_dict(), world, args)
                pending[future] = (cursor, world)
                cursor += 1
            if not pending:
                if failed:
                    raise RuntimeError('dispatch stopped before complete wave')
                time.sleep(10)
                continue
            ready, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            for future in ready:
                index, world = pending.pop(future)
                try:
                    result = future.result()
                except BaseException:
                    result = dict(ok=False, actual=None, error=traceback.format_exc())
                if result['actual'] is not None:
                    budget.finish(result['actual'], f'{args.variant}:{world["seed"]}')
                if not result['ok']:
                    failed = True
                    append(root/'failure.jsonl', dict(world=world, **result))
                    with (root/'STOP_REQUESTED').open('a', encoding='utf-8') as stream:
                        stream.write('worker failure; no retry\n')
                else:
                    results[index] = result['transitions']
            if failed and not pending:
                raise RuntimeError('incomplete production wave; all charges retained, no retry')
    return [results[i] for i in range(len(worlds))]
