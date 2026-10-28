"""Opt-in, process-isolated future execution; no shared RNG or mutable WM state.

Workers preserve row geometry and the full-K outer VJP. Only the coordinator
adds parameter gradients, in the original (start, future) order. Files contain
owned trusted tensors/models only. Workers exit cooperatively through STOP.
"""
from __future__ import annotations

import json
import gc
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

import torch

# Windows venv launchers may insert an intermediate parent process.
_REPARENTING = os.name != 'nt'


def worker_count():
    raw = os.environ.get('WMS_SPEED_FUTURE_WORKERS', '1')
    if not raw.isdigit() or not 1 <= int(raw) <= 8:
        raise ValueError('WMS_SPEED_FUTURE_WORKERS must be an integer in [1,8]')
    return int(raw)


def parallel_forward():
    return os.environ.get('WMS_SPEED_PARALLEL_FORWARD', '') == '1'


def vjp_slots():
    count = worker_count()
    raw = os.environ.get('WMS_SPEED_FUTURE_VJP_SLOTS', str(count))
    if not raw.isdigit() or not 1 <= int(raw) <= count:
        raise ValueError('WMS_SPEED_FUTURE_VJP_SLOTS must be in [1, workers]')
    return int(raw)


def parent_alive(root):
    """Linux reparents orphan workers; stop this owned pool without signals."""
    expected = os.environ.get('WMS_SPEED_FUTURE_PARENT_PID')
    if _REPARENTING and expected and os.getppid() != int(expected):
        (root/'STOP').touch()
        return False
    return True


class VJPGate:
    """Keep replay forwards concurrent while bounding live large recomputations.

    One measured 9-row replay forward reserves <.86 GB, versus 12.14 GB for
    recompute/backward. A worker raises its allocator limit only while owning
    one atomic directory token. It frees its graph/cache and lowers the limit
    BEFORE releasing the token. No arithmetic, shapes or checkpoint changes.
    """
    def __init__(self, trainer, root):
        self.trainer, self.root, self.token = trainer, root, None
        self.low = float(os.environ.get('WMS_SPEED_REPLAY_FORWARD_MEMORY_FRACTION', '.022'))
        self.high = float(os.environ.get('WMS_SPEED_FUTURE_MEMORY_FRACTION', '.10'))

    def acquire(self):
        while self.token is None:
            if not parent_alive(self.root) or (self.root/'STOP').exists():
                raise RuntimeError('own worker STOP requested while waiting for VJP slot')
            for index in range(vjp_slots()):
                token = self.root/'vjp_slots'/str(index)
                try:
                    token.mkdir()
                except FileExistsError:
                    continue
                self.token = token
                break
            if self.token is None:
                time.sleep(.02)
        if self.trainer.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(self.high, self.trainer.device)

    def release(self):
        if self.token is None:
            return
        if self.trainer.device.type == 'cuda':
            # Called outside _vjp, after its graph-bearing frame has returned.
            gc.collect()
            torch.cuda.empty_cache()
            total = torch.cuda.get_device_properties(self.trainer.device).total_memory
            if torch.cuda.memory_reserved(self.trainer.device) > self.low * total:
                # Keep the token until this worker exits, freeing its resources.
                raise RuntimeError('VJP worker retained more than its idle memory budget')
            torch.cuda.set_per_process_memory_fraction(self.low, self.trainer.device)
        self.token.rmdir()
        self.token = None


def _snapshot(trainer):
    # Deliberately exclude datasets, optimizer state, validation panels, and real
    # return providers. All execution inputs to imagine are passed explicitly.
    keys = ('world_models policy group_encoder critic target_critic actor_parameters '
            'device _training_imagination config counterfactual use_lanes lane_rules '
            'controller_factory observation_builder reward_fn reward_weights safety '
            'exploration update_count coop_anchor uncertainty long_credit_requested '
            'long_credit_trusted_s paircrit').split()
    result = object.__new__(type(trainer))
    result.__dict__.update({k: getattr(trainer, k) for k in keys if hasattr(trainer, k)})
    result.output_dir = None
    return result


class FuturePool:
    def __init__(self, trainer, common):
        import cloudpickle  # Optional dependency, used only by this opt-in mode.
        if trainer.output_dir is None:
            raise ValueError('parallel futures requires an owned output_dir')
        self.root = Path(trainer.output_dir) / 'future_workers' / uuid.uuid4().hex
        self.root.mkdir(parents=True)
        self.processes, self.logs = [], []
        env = dict(os.environ)
        env['WMS_SPEED_POOL_KIND'] = common['kind']
        env['WMS_SPEED_FUTURE_PARENT_PID'] = str(os.getpid())
        if common['kind'] == 'forward':
            env['WMS_SPEED_FUTURE_MEMORY_FRACTION'] = env.get(
                'WMS_SPEED_FORWARD_MEMORY_FRACTION', '0.04')
        if trainer.device.type == 'cuda':
            torch.cuda.empty_cache()
            total = torch.cuda.get_device_properties(trainer.device).total_memory
            fraction = float(env.get('WMS_SPEED_FUTURE_MEMORY_FRACTION', '0.10'))
            # CUDA contexts/libraries are outside the allocator fraction. Reserve
            # 768 MiB per worker in addition to the coordinator's current pool.
            slots = vjp_slots() if common['kind'] == 'backward' else worker_count()
            low = float(env.get('WMS_SPEED_REPLAY_FORWARD_MEMORY_FRACTION', '.022'))
            if not 0 < low <= fraction <= .6:
                raise ValueError('memory fractions must satisfy 0 < idle <= active <= .6')
            bound = torch.cuda.memory_reserved(trainer.device) + total * (
                slots * fraction + (worker_count() - slots) * low) + (worker_count() + 1) * 768 * 1024**2
            if bound > .60 * total:
                raise ValueError(f'parallel memory budget exceeds 60%: {bound / total:.3f}')
        (self.root/'vjp_slots').mkdir()
        with (self.root / 'trainer.pkl').open('wb') as stream:
            cloudpickle.dump(_snapshot(trainer), stream)
        torch.save(common, self.root / 'common.pt')
        try:
            for index in range(worker_count()):
                (self.root / str(index)).mkdir()
                log = (self.root / f'worker{index}.log').open('w')
                self.logs.append(log)
                process = subprocess.Popen(
                    [sys.executable, '-B', '-m', __name__, str(self.root), str(index)],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    env=env,
                    creationflags=0x08000000 if os.name == 'nt' else 0)
                self.processes.append(process)
            (self.root / 'PIDS.json').write_text(json.dumps([p.pid for p in self.processes]))
        except BaseException:
            self.close()
            raise

    def map(self, tasks):
        tasks = list(tasks)
        for index, task in enumerate(tasks):
            directory = self.root / str(index % len(self.processes))
            (directory / f'{index}.json').write_text(json.dumps(task))
            # Readiness is separate from file creation, including on network FS.
            (directory / f'{index}.ready').touch()
        for index in range(len(tasks)):
            directory = self.root / str(index % len(self.processes))
            ready = directory / f'{index}.done'
            while not ready.exists():
                for process in self.processes:
                    if process.poll() is not None:
                        raise RuntimeError(f'future worker exited ({process.returncode}); see {self.root}')
                time.sleep(.05)
            error = directory / f'{index}.error'
            if error.exists():
                raise RuntimeError(error.read_text())
            output = directory / f'{index}.pt'
            result = torch.load(output, weights_only=False)
            if tasks[index].__class__ is int:  # forward tensors are large; owned scratch only
                output.unlink()
            yield result

    def close(self):
        (self.root / 'STOP').touch()
        for process in self.processes:
            process.wait()
        for log in self.logs:
            log.close()
        for name in ('trainer.pkl', 'common.pt'):
            (self.root / name).unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _vjp(trainer, common, task, gate=None):
    from contextlib import nullcontext
    from .dreamer import _restore_rng
    from .imagination_memory import (_RandomTape, _subset, _saved_tensor_spill,
                                     _slice_shape, _trace_memory)
    batch, cfg, kwargs, before = (common[k] for k in ('batch', 'cfg', 'kwargs', 'before'))
    total = batch.state['s_m'].shape[0]
    n_rows = kwargs['variants'].n_rows if kwargs['variants'] is not None else total
    start, stop, future = task
    small, variants, starts, rows = _subset(batch, kwargs['variants'], start, stop)
    replay = _RandomTape(common['tape'], (starts, rows, total, n_rows))
    _restore_rng(before, trainer.device)
    spill = _saved_tensor_spill(trainer) if cfg.imagination_offload_saved_tensors else nullcontext()
    parameters = tuple(p for p in trainer.actor_parameters if p.requires_grad)
    _trace_memory(trainer, 'future_start', start=start, future=future)
    with torch.enable_grad(), replay, spill:
        result = trainer.imagine(small, **{**kwargs, 'variants': variants},
            _memory_replay=True, _memory_config=cfg, _memory_full_batch=batch,
            _memory_start_index=starts, _memory_row_index=rows,
            _memory_future_cache=common['cache'], _memory_active_future=future)
        _trace_memory(trainer, 'future_forward_done', start=start, future=future)
        # Dict construction can use hash-randomized sets (reward components).
        # Match the coordinator's output paths, never the worker's dict order.
        leaves = common['spec'].flatten_up_to(result)
        outputs, vectors = [], []
        for slot, vector in zip(common['slots'], common['grad_outputs']):
            value = leaves[slot]
            if vector is not None and value.requires_grad:
                outputs.append(value)
                vectors.append(_slice_shape(vector, value.shape, starts, rows, total, n_rows))
        if replay.position != len(common['tape']):
            raise RuntimeError('LONGMEM random call trace shortened during parallel replay')
        if gate is not None:
            gate.acquire()
        grads = torch.autograd.grad(outputs, parameters, vectors, allow_unused=True)
        _trace_memory(trainer, 'future_vjp_done', start=start, future=future)
    # CPU transport preserves bytes. Coordinator copies to its device before addition.
    return {'grads': [None if g is None else g.detach().cpu() for g in grads],
            'spill': getattr(trainer, 'memory_spill_stats', None)}


def backward_futures(trainer, batch, cfg, kwargs, tape, before, cache, slots, grad_outputs, spec):
    from .imagination_memory import _trace_memory
    common = dict(kind='backward', batch=batch, cfg=cfg, kwargs=kwargs, tape=tape,
                  before=before, cache=cache, slots=slots, grad_outputs=grad_outputs, spec=spec)
    total = batch.state['s_m'].shape[0]
    micro = cfg.imagination_microbatch or total
    tasks = [(start, min(total, start + micro), future)
             for start in range(0, total, micro) for future in range(len(cache['items']))]
    accumulated = [None] * len([p for p in trainer.actor_parameters if p.requires_grad])
    with FuturePool(trainer, common) as pool:
        for task, result in zip(tasks, pool.map(tasks)):
            for i, grad in enumerate(result['grads']):
                if grad is not None:
                    grad = grad.to(trainer.device)
                    accumulated[i] = grad if accumulated[i] is None else accumulated[i] + grad
            if result['spill'] is not None:
                trainer.memory_spill_stats = result['spill']
            _trace_memory(trainer, 'parallel_future_collected', start=task[0], future=task[2])
    return tuple(accumulated)


def _forward(trainer, common, future):
    from .dreamer import _restore_rng
    from .imagination_memory import _RandomTape
    cache, tape = {'items': []}, []
    _restore_rng(common['before'], trainer.device)
    with torch.no_grad(), _RandomTape(tape, offload=True):
        trainer.imagine(common['batch'], **common['kwargs'], _memory_replay=True,
            _memory_config=common['cfg'], _memory_full_batch=common['batch'],
            _memory_future_cache=cache, _memory_collect_future=future)
    return cache, tape


def forward_futures(trainer, batch, cfg, kwargs, before):
    from .dreamer import _restore_rng
    from .imagination_memory import _RandomTape
    from .futures import FuturesConfig
    futures_cfg = FuturesConfig.from_trainer(cfg)
    count = len(trainer.world_models) * (futures_cfg.n_samples if futures_cfg.active else 1)
    common = dict(kind='forward', batch=batch, cfg=cfg, kwargs=kwargs, before=before)
    cache, tape = {'items': []}, []
    with FuturePool(trainer, common) as pool:
        for index, (part, draws) in enumerate(pool.map(range(count))):
            prefix = part['tape_prefix']
            if index == 0:
                tape.extend(draws[:prefix])
                cache['advanced_rng'] = part['advanced_rng']
            tape.extend(draws[prefix:])
            item = part['items'][0]
            item['tape_end'] = len(tape)
            cache['items'].append(item)
    total = batch.state['s_m'].shape[0]
    n_rows = kwargs['variants'].n_rows if kwargs['variants'] is not None else total
    starts, rows = torch.arange(total, device=trainer.device), torch.arange(n_rows, device=trainer.device)
    _restore_rng(before, trainer.device)
    with torch.no_grad(), _RandomTape(tape, (starts, rows, total, n_rows)):
        result = trainer.imagine(batch, **kwargs, _memory_replay=True, _memory_config=cfg,
            _memory_full_batch=batch, _memory_future_cache=cache, _memory_active_future=-1,
            _memory_row_index=rows)
    return result, tape, cache


def _main(root, index):
    import cloudpickle
    root, directory = Path(root), Path(root) / index
    # Configure allocator BEFORE deserializing any CUDA tensors.
    fraction = float(os.environ.get('WMS_SPEED_FUTURE_MEMORY_FRACTION', '0.10'))
    gated = os.environ.get('WMS_SPEED_POOL_KIND') == 'backward' and vjp_slots() < worker_count()
    if gated:
        fraction = float(os.environ.get('WMS_SPEED_REPLAY_FORWARD_MEMORY_FRACTION', '.022'))
    if not 0 < fraction <= .6:
        raise ValueError('worker memory fraction must be in (0,.6]')
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)
    with (root / 'trainer.pkl').open('rb') as stream:
        trainer = cloudpickle.load(stream)
    trainer.output_dir = directory
    common = torch.load(root / 'common.pt', weights_only=False)
    gate = VJPGate(trainer, root) if gated else None
    completed = set()
    while parent_alive(root) and not (root / 'STOP').exists():
        ready = sorted(directory.glob('*.ready'), key=lambda p: int(p.stem))
        for path in ready:
            if (root / 'STOP').exists():
                break
            if path.stem in completed:
                continue
            try:
                task = json.loads(path.with_suffix('.json').read_text())
                result = (_vjp(trainer, common, task, gate) if common['kind'] == 'backward'
                          else _forward(trainer, common, task))
                torch.save(result, path.with_suffix('.pt'))
                del result
                if trainer.device.type == 'cuda':
                    (directory / 'memory.json').write_text(json.dumps({
                        'peak_allocated': torch.cuda.max_memory_allocated(),
                        'peak_reserved': torch.cuda.max_memory_reserved()}))
            except BaseException:
                path.with_suffix('.error').write_text(traceback.format_exc())
            finally:
                if gate is not None:
                    gate.release()
            path.with_suffix('.done').touch()
            completed.add(path.stem)
        time.sleep(.05)


if __name__ == '__main__':
    _main(*sys.argv[1:])
