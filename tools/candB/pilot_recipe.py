"""Explicit exploratory adapter with update-boundary, memory-safe ES routing."""
from contextlib import ExitStack
import argparse
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch


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
    from wms.control.dreamer import DreamerTrainer
    import torch

    base = base_module or base
    base.authority.require_opt_in(exploratory)
    output = Path(output)
    real_pool = parallel_futures.FuturePool
    state = dict(parallel=False, actual='serial', boundary=None)

    class RetainedPool(real_pool):
        def close(self):
            # Preserve the owned common payload and trainer snapshot as receipts.
            (self.root / 'STOP').touch()
            for process in self.processes:
                process.wait()
            for log in self.logs:
                log.close()

    class Router:
        @staticmethod
        def object(target, name, replacement):
            if target is parallel_futures and name == 'FuturePool':
                serial_pool = replacement

                def routed(trainer, common):
                    if not common.get('eslong') or common.get('kind') != 'forward':
                        raise ValueError('exploratory recipe routes only ES forward tasks')
                    if state['parallel']:
                        total = torch.cuda.get_device_properties(trainer.device).total_memory
                        bound = (torch.cuda.memory_reserved(trainer.device) + total * .48
                                 + 3 * 768 * 1024**2)
                        if bound <= .90 * total:
                            state['actual'] = 'two-forward-workers'
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
                    state['boundary'] = trainer.update_count
                    os.environ.update(WMS_SPEED_FORWARD_WORKERS='2' if state['parallel'] else '1',
                                      WMS_SPEED_FUTURE_BUDGET_CAP='.90' if state['parallel'] else '.44')
                    metrics = replacement(trainer, *args, **kwargs)
                    metrics['eslong_forward_workers'] = 2 if state['actual'] == 'two-forward-workers' else 1
                    metrics['pilot_recipe'] = dict(latched_at_update=state['boundary'],
                        requested_workers=2 if state['parallel'] else 1, actual=state['actual'])
                    return metrics
                return patch.object(target, name, boundary)
            return patch.object(target, name, replacement)

    # Only this adapter's patch factory is routed. Other modules retain mock.patch.
    with ExitStack() as stack:
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
