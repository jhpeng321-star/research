"""Fresh exploratory lean pilot: unchanged dense K8, first4 ES CRNs, bound panel4."""
import argparse
from contextlib import ExitStack
from functools import wraps
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

from tools.candB import pilot_cli as base, pilot_lean_authority as authority
from tools.candB import pilot_lean_worker


def memory_bound(total, reserved, peer_bytes, workers, fraction):
    # Peer measurement plus2GiB growth allowance; include all CUDA contexts.
    bound = reserved + workers*fraction*total + (workers+1)*768*1024**2 + peer_bytes + 2*1024**3
    if bound > min(.90*total,45*1024**3):
        raise ValueError('lean aggregate memory bound including peers exceeds safe ceiling')
    return bound


def es_prefix_gradient(original, trainer, imagined, weights, scale):
    context = trainer._eslong_context
    if len(context['cache']['items']) != 8:
        raise ValueError('lean ES needs original eight-future dense cache')
    context['cache'] = dict(context['cache'], items=context['cache']['items'][:4])
    grads, metrics = original(trainer, imagined, weights, scale)
    metrics.update(lean_es_future_indices=[0, 1, 2, 3], lean_dense_futures=8)
    return grads, metrics


def run(path, output, *, exploratory=False):
    base.authority.require_opt_in(exploratory)
    doc = authority.verify(path, exploratory=True)
    if authority.LABEL not in Path(output).name:
        raise ValueError('lean output must include EXPLORATORY-LEAN')
    for key, value in authority.ENV.items():
        if os.environ.get(key) != value:
            raise ValueError('lean environment identity mismatch: '+key)
    from wms.control import eslong, policy14_cli as cli, parallel_futures, policy14, execution
    from wms.control.dreamer import DreamerTrainer
    from tools.candB.pilot_telemetry_2305 import telemetry
    from tools.candB.rankcheck import measurement_prefix
    import torch

    original_gradient, original_pair = eslong.gradient, eslong.paired_loss_difference
    original_panel, dumps = cli.load_gate_panel, json.dumps
    real_pool = parallel_futures.FuturePool
    began = time.monotonic()
    state = dict(phase=0, workers=[], capture=None, raw=None, own_worker_pids=set())

    def load_panel(path, val):
        panel = original_panel(path, val)  # Validate ALL original16 before taking prefix.
        if panel['starts'][:4] != doc['lean_recipe']['panel_starts']:
            raise ValueError('lean panel starts differ from bound first4')
        return dict(panel, n=4, starts=panel['starts'][:4])

    def paired(*args, **kwargs):
        # Only this scalar ES reduction uses K4; dense objective/scoring stay K8.
        with measurement_prefix(4):
            return original_pair(*args, **kwargs)

    def gradient(*args, **kwargs):
        return es_prefix_gradient(original_gradient, *args, **kwargs)

    class RetainedPool(real_pool):
        def __init__(self, trainer, common):
            self.lean_role=('perturbed_ES' if common.get('eslong') else
                ('unperturbed_forward' if common['kind']=='forward' else 'dense_backward'))
            if trainer.device.type == 'cuda':
                torch.cuda.empty_cache()
                samples = Path(output).parent/'GPU.jsonl'
                sample = json.loads(samples.read_text().splitlines()[-1])
                peers = 0
                for line in sample['apps'].splitlines():
                    pid, used = (int(x.strip()) for x in line.split(','))
                    if pid != os.getpid() and pid not in state['own_worker_pids']:
                        peers += used*1024**2
                kind=common['kind']
                workers=parallel_futures.worker_count(kind)
                fraction=float(os.environ['WMS_SPEED_FORWARD_MEMORY_FRACTION' if kind=='forward' else 'WMS_SPEED_FUTURE_MEMORY_FRACTION'])
                memory_bound(torch.cuda.get_device_properties(trainer.device).total_memory,
                    torch.cuda.memory_reserved(trainer.device),peers,workers,fraction)
            super().__init__(trainer, common)
            state['own_worker_pids'].update(p.pid for p in self.processes)

        def map(self, tasks):
            tasks = list(tasks)
            for index, task in enumerate(tasks):
                directory = self.root / str(index % len(self.processes))
                (directory / f'{index}.json').write_text(dumps(task))
                (directory / f'{index}.ready').touch()
            for index in range(len(tasks)):
                directory = self.root / str(index % len(self.processes))
                while not (directory / f'{index}.done').exists():
                    for process in self.processes:
                        if process.poll() is not None:
                            raise RuntimeError(f'owned lean worker exited; retained receipts at {self.root}')
                    time.sleep(.05)
                error = directory / f'{index}.error'
                if error.exists():
                    raise RuntimeError(error.read_text())
                yield torch.load(directory / f'{index}.pt', weights_only=False)

        def close(self):
            (self.root / 'STOP').touch()
            for process in self.processes:
                process.wait()
            for log in self.logs:
                log.close()
            state['workers'].append(dict(root=str(self.root), role=self.lean_role, workers=[
                dict(pid=p.pid, **json.loads((self.root / str(i) / 'memory.json').read_text()),
                    executed=json.loads((self.root / str(i) / 'executed.json').read_text())
                    if (self.root / str(i) / 'executed.json').exists() else None)
                for i, p in enumerate(self.processes)
                if (self.root / str(i) / 'memory.json').is_file()]))

    class Router:
        @staticmethod
        def object(target, name, replacement):
            if target is DreamerTrainer and name == 'load_checkpoint':
                def load(trainer, *args, **kwargs):
                    result = replacement(trainer, *args, **kwargs)
                    summary = Path(output, 'RUN_SUMMARY.json')
                    data = json.loads(summary.read_text())
                    data.update(label=authority.LABEL, lean_recipe=doc['lean_recipe'])
                    summary.write_text(dumps(data, indent=1))
                    return result
                return patch.object(target, name, load)
            if target is DreamerTrainer and name == 'update':
                def update(trainer, *args, **kwargs):
                    if state['phase'] == 0:
                        state['startup_s'] = time.monotonic()-began
                        Path(output, 'LEAN_STARTUP.json').write_text(dumps(dict(
                            label=authority.LABEL, startup_s=state['startup_s'],
                            phase_origin_update=trainer.update_count, recipe=doc['lean_recipe']), indent=1))
                    state.update(workers=[], raw=None, capture=dict(totals=None, minima=None))
                    try:
                        metrics = replacement(trainer, *args, **kwargs)
                    finally:
                        captured = state['capture']
                        state['capture'] = None
                    state['phase'] += 1
                    metrics.update(label=authority.LABEL, startup_s=state['startup_s'],
                        pilot_worker_memory=state['workers'], lean_training_raw_J=state['raw'],
                        lean_recipe=doc['lean_recipe'])
                    if captured['totals'] is not None:
                        count, nonzero, brake = captured['totals'].cpu().tolist()
                        metrics['executed_residual_training'] = dict(
                            scope='coordinator training rollouts incl replay; final acting-CAV ticks',
                            count=count, fraction_nonzero=nonzero/max(1,count), mean_braking=brake/max(1,count),
                            min_delta_mps2=float(captured['minima'].cpu()))
                    transported=pilot_lean_worker.aggregate(state['workers'])
                    if transported is not None:
                        metrics['executed_residual_training']=transported
                    trainer.save_checkpoint(Path(output, 'policy_latest.pt'))
                    return metrics
                return patch.object(target, name, update)
            if target is json and name == 'dumps':
                def serialize(value, *args, **kwargs):
                    text = replacement(value, *args, **kwargs)
                    if isinstance(value, dict) and ('actor_loss' in value or 'status' in value):
                        result = json.loads(text)
                        result.update(label=authority.LABEL, lean_recipe=doc['lean_recipe'])
                        text = dumps(result, *args, **kwargs)
                    return text
                return patch.object(target, name, serialize)
            return patch.object(target, name, replacement)

    original_actor, original_controller = DreamerTrainer.actor_objective, policy14.make_cav_controller
    def actor(trainer, imagined, *args, **kwargs):
        result = original_actor(trainer, imagined, *args, **kwargs)
        if state['capture'] is not None and 'gated300_long_J' in imagined:
            from wms.control.gated300 import reference_index
            n = imagined['n_base']
            refs = reference_index(imagined['variants'])[:n]
            components = imagined['components']
            j = (components['delay']+components['source_wait']).sum(1).mean(0)
            long_j = imagined['gated300_long_J']
            state['raw'] = dict(raw_dJ_vs_CTRL0_dense=float((j[:n]-j[refs]).mean()),
                raw_dJ_vs_CTRL0_long=float((long_j[:n]-long_j[refs]).mean()),
                dense_horizon_s=imagined['imagination_horizon_s'], long_horizon_s=300,
                timing='pre-optimizer training start, K8; NOT frozen independent post-update rescore')
        return result

    def controller(*args, **kwargs):
        callback = original_controller(*args, **kwargs)
        @wraps(callback)
        def measured(frame):
            result = callback(frame)
            pilot_lean_worker.observe(frame,result,kwargs.get('act_mask'))
            capture = state['capture']
            if capture is not None:
                with torch.no_grad():
                    zeros = torch.zeros_like(frame['s_m'])
                    reference = execution.executed_residual(frame, zeros, zeros, residual_reference='ctrl0')
                    reference = ((frame['v_mps']+.5*reference).clamp_min(0.)-frame['v_mps'])/.5
                    mask = frame['present'] & frame['is_cav'] & (frame['lane'] >= 0)
                    if kwargs.get('act_mask') is not None:
                        mask = mask & kwargs['act_mask']
                    delta = (result[..., 3]-reference)[mask]
                    if delta.numel():
                        values = torch.stack((delta.new_tensor(delta.numel()), (delta!=0).sum(), (-delta).clamp_min(0).sum())).double()
                        capture['totals'] = values if capture['totals'] is None else capture['totals']+values
                        minimum = delta.min()
                        capture['minima'] = minimum if capture['minima'] is None else torch.minimum(capture['minima'], minimum)
            return result
        return measured

    with ExitStack() as stack:
        stack.enter_context(telemetry())
        # Telemetry's actor wrapper must also feed lean raw-J capture.
        original_actor = DreamerTrainer.actor_objective
        stack.enter_context(patch.object(DreamerTrainer, 'actor_objective', actor))
        stack.enter_context(patch.object(policy14, 'make_cav_controller', controller))
        stack.enter_context(patch.object(base, 'patch', Router))
        stack.enter_context(patch.object(cli, 'load_gate_panel', load_panel))
        stack.enter_context(patch.object(eslong, 'gradient', gradient))
        stack.enter_context(patch.object(eslong, 'paired_loss_difference', paired))
        stack.enter_context(patch.object(parallel_futures, 'FuturePool', RetainedPool))
        return base.run(path, output, 'pilot', exploratory=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(authority.base.FLAG, action='store_true', required=True)
    parser.add_argument('--authority', required=True)
    parser.add_argument('output')
    args = parser.parse_args()
    raise SystemExit(run(args.authority, args.output, exploratory=True))


if __name__ == '__main__':
    main()
