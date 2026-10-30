"""Explicit exploratory adapter with update-boundary, memory-safe ES routing."""
from contextlib import ExitStack
import argparse
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch


def safe_workers(total, reserved, *, maximum=4, fraction=.24):
    """Bound allocators plus 768MiB/context, below both90% and45GiB."""
    ceiling = min(.90 * total, 45 * 1024**3)
    for workers in range(maximum, 1, -1):
        bound = reserved + workers * fraction * total + (workers + 1) * 768 * 1024**2
        if bound <= ceiling:
            return workers, bound
    return 1, reserved


def parallel_requested(path):
    if not path.is_file():
        return False
    doc = json.loads(path.read_text())
    if doc.get('label') != 'EXPLORATORY' or doc.get('forward_workers') != 2:
        raise ValueError('invalid exploratory worker admission')
    return True


def run(authority, output, *, exploratory=False, base_module=None, parallel_initial=False,
        continuation_proof=None):
    from tools.candB import pilot_cli as base
    from wms.control import parallel_futures
    from tools.candB.pilot_telemetry_2305 import telemetry
    from wms.control.dreamer import DreamerTrainer
    import torch

    base = base_module or base
    base.authority.require_opt_in(exploratory)
    output = Path(output)
    real_pool = parallel_futures.FuturePool
    state = dict(parallel=False, actual='serial', boundary=None, workers=1, memory_bound=None,
                 dense_workers=1, dense_slots=1, pool_memory=[])

    class RetainedPool(real_pool):
        def close(self):
            # Preserve the owned common payload and trainer snapshot as receipts.
            (self.root / 'STOP').touch()
            for process in self.processes:
                process.wait()
            for log in self.logs:
                log.close()
            state['pool_memory'].append(dict(root=str(self.root), workers=[
                dict(pid=process.pid, **json.loads((self.root / str(i) / 'memory.json').read_text()))
                for i, process in enumerate(self.processes)
                if (self.root / str(i) / 'memory.json').is_file()]))

    class Router:
        @staticmethod
        def object(target, name, replacement):
            if target is parallel_futures and name == 'FuturePool':
                serial_pool = replacement

                def routed(trainer, common):
                    if common.get('kind') == 'backward':
                        if not state['parallel']:
                            return RetainedPool(trainer, common)
                        torch.cuda.empty_cache()
                        total = torch.cuda.get_device_properties(trainer.device).total_memory
                        reserved = torch.cuda.memory_reserved(trainer.device)
                        slots = 1
                        for candidate in (3, 2, 1):
                            bound = reserved + total * (candidate*.24+(3-candidate)*.04) + 4*768*1024**2
                            if bound <= min(.90*total, 45*1024**3):
                                slots = candidate
                                break
                        state.update(dense_workers=3, dense_slots=slots)
                        os.environ.update(WMS_SPEED_FUTURE_WORKERS='3', WMS_SPEED_FUTURE_VJP_SLOTS=str(slots))
                        return RetainedPool(trainer, common)
                    if not common.get('eslong') or common.get('kind') != 'forward':
                        raise ValueError('exploratory recipe routes ES forward/dense backward only')
                    if state['parallel']:
                        torch.cuda.empty_cache()
                        total = torch.cuda.get_device_properties(trainer.device).total_memory
                        workers, bound = safe_workers(total, torch.cuda.memory_reserved(trainer.device))
                        state.update(workers=workers, memory_bound=bound)
                        os.environ['WMS_SPEED_FORWARD_WORKERS'] = str(workers)
                        if workers > 1:
                            state['actual'] = f'{workers}-forward-workers'
                            return RetainedPool(trainer, common)
                        state['actual'] = 'serial-budget-fallback'
                    return serial_pool(trainer, common)

                return patch.object(target, name, routed)
            if target is DreamerTrainer and name == 'update':
                def boundary(trainer, *args, **kwargs):
                    # Latch once, before any work/RNG consumption in this update.
                    if continuation_proof and not (output / 'CONTINUATION_PROOF.json').exists():
                        baseline_sha = hashlib.sha256((output / 'frozen_scoring_baseline.json').read_bytes()).hexdigest()
                        if baseline_sha != continuation_proof['parent_sha256']['frozen_scoring_baseline.json']:
                            raise ValueError('same-phase frozen baseline bytes changed')
                        continuation_proof['retained_baseline_sha256'] = baseline_sha
                        with (output / 'CONTINUATION_PROOF.json').open('x') as stream:
                            json.dump(continuation_proof, stream, indent=1)
                    state['parallel'] = parallel_initial or parallel_requested(output / 'TWO_FORWARD_WORKERS.json')
                    state['actual'] = 'serial'
                    state.update(workers=1, memory_bound=None, pool_memory=[])
                    state['boundary'] = trainer.update_count
                    os.environ.update(WMS_SPEED_FORWARD_WORKERS='4' if state['parallel'] else '1',
                                      WMS_SPEED_FUTURE_WORKERS='3' if state['parallel'] else '1',
                                      WMS_SPEED_FUTURE_VJP_SLOTS='2' if state['parallel'] else '1',
                                      WMS_SPEED_FUTURE_BUDGET_CAP='.90' if state['parallel'] else '.44')
                    metrics = replacement(trainer, *args, **kwargs)
                    metrics['eslong_forward_workers'] = state['workers']
                    metrics['pilot_recipe'] = dict(latched_at_update=state['boundary'],
                        requested_workers=4 if state['parallel'] else 1, actual=state['actual'],
                        allocator_fraction=.24, memory_bound_bytes=state['memory_bound'],
                        memory_ceiling='min(90% device,45GiB), contexts included',
                        dense_workers=state['dense_workers'], dense_vjp_slots=state['dense_slots'])
                    metrics['pilot_worker_memory'] = list(state['pool_memory'])
                    return metrics
                return patch.object(target, name, boundary)
            return patch.object(target, name, replacement)

    # Only this adapter's patch factory is routed. Other modules retain mock.patch.
    with ExitStack() as stack:
        stack.enter_context(telemetry())
        stack.enter_context(patch.object(base, 'patch', Router))
        return base.run(authority, str(output), 'pilot', exploratory=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exploratory-owner-20261030', action='store_true', required=True)
    p.add_argument('--authority', required=True)
    p.add_argument('output')
    a = p.parse_args()
    # Bind the operational overlay independently of the unchanged scientific preset.
    print(json.dumps(dict(label='EXPLORATORY', operational_adapter_sha256=hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest())), flush=True)
    raise SystemExit(run(a.authority, a.output, exploratory=True))


if __name__ == '__main__':
    main()
