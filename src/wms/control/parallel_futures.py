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


def worker_count(kind=None):
    """Worker processes. ``kind='forward'`` may be sized separately (S-LONGMEM300 memory
    option ``WMS_SPEED_FORWARD_WORKERS``: the no-grad forward pool and the VJP pool have
    different per-worker memory at 300 s); unset keeps the single registered count."""
    raw = os.environ.get('WMS_SPEED_FUTURE_WORKERS', '1')
    if kind == 'forward' and os.environ.get('WMS_SPEED_FORWARD_WORKERS'):
        raw = os.environ['WMS_SPEED_FORWARD_WORKERS']
    if not raw.isdigit() or not 1 <= int(raw) <= 8:
        raise ValueError('WMS_SPEED_FUTURE_WORKERS must be an integer in [1,8]')
    return int(raw)


def parallel_forward():
    return os.environ.get('WMS_SPEED_PARALLEL_FORWARD', '') == '1'


def budget_cap():
    """Aggregate allocator budget ceiling as a fraction of the device (default .60).

    S-LONGMEM300 (memory/infrastructure only): a dedicated 24 GB card running the
    300 s long credit needs a larger ceiling than the shared-card default. The
    default and every arithmetic path are unchanged when the variable is unset.
    """
    raw = os.environ.get('WMS_SPEED_FUTURE_BUDGET_CAP', '.60')
    value = float(raw)
    if not .0 < value <= .95:
        raise ValueError('WMS_SPEED_FUTURE_BUDGET_CAP must be in (0,.95]')
    return value


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

    One measured 9-row replay ends below .86 GB reserved, but its transient
    forward peak is higher. Limits must cover that peak, not the end value.
    A worker raises its allocator limit only while owning
    one atomic directory token. It frees its graph/cache and lowers the limit
    BEFORE releasing the token. No arithmetic, shapes or checkpoint changes.
    """
    def __init__(self, trainer, root):
        self.trainer, self.root, self.token = trainer, root, None
        self.device_index = (trainer.device.index if trainer.device.index is not None
                             else torch.cuda.current_device()) if trainer.device.type == 'cuda' else None
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
            torch.cuda.set_per_process_memory_fraction(self.high, self.device_index)

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
            torch.cuda.set_per_process_memory_fraction(self.low, self.device_index)
        self.token.rmdir()
        self.token = None


def _snapshot(trainer):
    # Deliberately exclude datasets, optimizer state, validation panels, and real
    # return providers. All execution inputs to imagine are passed explicitly.
    keys = ('world_models policy group_encoder critic target_critic actor_parameters '
            'device _training_imagination config counterfactual use_lanes lane_rules '
            'controller_factory observation_builder reward_fn reward_weights safety '
            'exploration update_count coop_anchor uncertainty long_credit_requested '
            'long_credit_trusted_s paircrit gated300 world_model_ids horizon_curriculum '
            '_longgrad_phase _long_credit_reference').split()
    result = object.__new__(type(trainer))
    result.__dict__.update({k: getattr(trainer, k) for k in keys if hasattr(trainer, k)})
    result._longgrad_optimizer_groups = [tuple(g['params']) for g in trainer.actor_optimizer.param_groups]
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
        # Source-tree runs (pytest pythonpath=src / editable installs) are not
        # inherited by the worker subprocess; pass the package import root
        # explicitly. Operational only: no arithmetic or task payload change.
        src_root = str(Path(__file__).resolve().parents[2])
        parts = [src_root] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else [])
        env['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(parts))
        if common['kind'] == 'forward':
            env['WMS_SPEED_FUTURE_MEMORY_FRACTION'] = env.get(
                'WMS_SPEED_FORWARD_MEMORY_FRACTION', '0.04')
        pool_size = worker_count(common['kind'])
        if trainer.device.type == 'cuda':
            torch.cuda.empty_cache()
            total = torch.cuda.get_device_properties(trainer.device).total_memory
            fraction = float(env.get('WMS_SPEED_FUTURE_MEMORY_FRACTION', '0.10'))
            # CUDA contexts/libraries are outside the allocator fraction. Reserve
            # 768 MiB per worker in addition to the coordinator's current pool.
            slots = vjp_slots() if common['kind'] == 'backward' else pool_size
            low = float(env.get('WMS_SPEED_REPLAY_FORWARD_MEMORY_FRACTION', '.022'))
            if not 0 < low <= fraction <= budget_cap():
                raise ValueError('memory fractions must satisfy 0 < idle <= active <= budget cap')
            bound = torch.cuda.memory_reserved(trainer.device) + total * (
                slots * fraction + (pool_size - slots) * low) + (pool_size + 1) * 768 * 1024**2
            if bound > budget_cap() * total:
                raise ValueError(f'parallel memory budget exceeds {budget_cap():.0%}: {bound / total:.3f}')
        (self.root/'vjp_slots').mkdir()
        with (self.root / 'trainer.pkl').open('wb') as stream:
            cloudpickle.dump(_snapshot(trainer), stream)
        torch.save(common, self.root / 'common.pt')
        try:
            for index in range(pool_size):
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
                        errors = sorted(self.root.glob('[0-7]/*.error'))
                        detail = '\n' + errors[0].read_text() if errors else ''
                        raise RuntimeError(f'future worker exited ({process.returncode}); see {self.root}{detail}')
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


def _vjp(trainer, common, task, gate=None, *, transport=True):
    from contextlib import nullcontext
    from .dreamer import _restore_rng
    from .imagination_memory import (_RandomTape, _subset, _saved_tensor_spill,
                                     _slice_shape, _trace_memory, _lm300_snapshot)
    batch, cfg, kwargs, before = (common[k] for k in ('batch', 'cfg', 'kwargs', 'before'))
    total = batch.state['s_m'].shape[0]
    n_rows = kwargs['variants'].n_rows if kwargs['variants'] is not None else total
    start, stop, future = task[:3]
    horizon = task[3] if len(task) > 3 else cfg.horizon
    from .eslong import active as es_active
    if es_active(trainer):
        from .longskip import dense_replay_horizon
        # Includes the unknown-support fallback: no ES backward can build a
        # full long trajectory graph, even with entirely zero dense credit.
        horizon = min(horizon, dense_replay_horizon(trainer, batch, cfg))
    small, variants, starts, rows = _subset(batch, kwargs['variants'], start, stop)
    replay = _RandomTape(common['tape'], (starts, rows, total, n_rows))
    _restore_rng(before, trainer.device)
    spill = _saved_tensor_spill(trainer) if cfg.imagination_offload_saved_tensors else nullcontext()
    parameters = tuple(p for p in trainer.actor_parameters if p.requires_grad)
    from .longgrad import StateTrace, future_record
    tracing = getattr(trainer, '_longgrad_phase', None) == 'long'
    trace = StateTrace() if tracing else None
    trainer._longgrad_state_trace = trace
    _trace_memory(trainer, 'future_start', start=start, future=future)
    _lm300_snapshot(trainer, 'start')
    began = time.perf_counter()
    with torch.enable_grad(), replay, spill:
        result = trainer.imagine(small, **{**kwargs, 'variants': variants},
            _memory_replay=True, _memory_config=cfg, _memory_full_batch=batch,
            _memory_start_index=starts, _memory_row_index=rows,
            _memory_future_cache=common['cache'], _memory_active_future=future,
            _memory_replay_horizon=horizon)
        peaks = ({'forward_peak_allocated_bytes': torch.cuda.max_memory_allocated(trainer.device),
                  'forward_peak_reserved_bytes': torch.cuda.max_memory_reserved(trainer.device)}
                 if trainer.device.type == 'cuda' else {})
        _trace_memory(trainer, 'future_forward_done', start=start, future=future, **peaks)
        _lm300_snapshot(trainer, 'forward_done')
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
    diagnostic = future_record(trainer, start, future, parameters, grads, trace) if tracing else None
    trainer._longgrad_state_trace = None
    return {'grads': [None if g is None else g.detach().cpu() if transport else g.detach()
                      for g in grads],
            'elapsed': time.perf_counter() - began, 'horizon': horizon,
            'longgrad': diagnostic,
            'spill': getattr(trainer, 'memory_spill_stats', None)}


def backward_futures(trainer, batch, cfg, kwargs, tape, before, cache, slots, grad_outputs, spec,
                     support=None):
    from contextlib import nullcontext
    from .imagination_memory import _trace_memory
    from .longskip import replay_horizon, record_replay
    common = dict(kind='backward', batch=batch, cfg=cfg, kwargs=kwargs, tape=tape,
                  before=before, cache=cache, slots=slots, grad_outputs=grad_outputs, spec=spec)
    total = batch.state['s_m'].shape[0]
    micro = cfg.imagination_microbatch or total
    from .imagination_memory import _subset
    tasks, skipped = [], []
    for start in range(0, total, micro):
        stop = min(total, start + micro)
        _, _, _, rows = _subset(batch, kwargs['variants'], start, stop)
        for future in range(len(cache['items'])):
            horizon = replay_horizon(trainer, batch, cfg, cache, slots, grad_outputs,
                                     spec, future, rows)
            task = (start, stop, future, horizon)
            (tasks if horizon else skipped).append(task)
    params = [p for p in trainer.actor_parameters if p.requires_grad]
    accumulated = [None] * len(params)
    support = {} if support is None else support

    def collect(task, result):
        used = [g is not None for g in result['grads']]
        if task[3] == cfg.horizon or all(used):
            support[tuple(task[:3])] = used
        for i, grad in enumerate(result['grads']):
            if grad is not None:
                grad = grad.to(trainer.device)
                accumulated[i] = grad if accumulated[i] is None else accumulated[i] + grad
        if result['spill'] is not None:
            trainer.memory_spill_stats = result['spill']
        if result.get('longgrad') is not None:
            trainer._longgrad_futures.append(result['longgrad'])
        record_replay(trainer, task[0], task[2], result.get('horizon', task[3]), cfg.horizon,
                      result['elapsed'])
        _trace_memory(trainer, 'parallel_future_collected', start=task[0], future=task[2])

    # Avoid creating workers for an empty long pass. Serial and pooled execution
    # use the identical planner, VJP and ordered accumulation code.
    context = FuturePool(trainer, common) if tasks and worker_count() > 1 else nullcontext()
    with context as pool:
        results = pool.map(tasks) if pool else (
            _vjp(trainer, common, task, transport=False) for task in tasks)
        for task, result in zip(tasks, results):
            collect(task, result)
    for task in skipped:
        known = support.get(tuple(task[:3]))
        if known is None and any(g is None for g in accumulated):
            # Preserve None vs materialized zero (Adam semantics). A first-ever
            # all-zero backward has no known parameter connectivity; conservatively
            # execute it. A prior dense pass supplies support for the long pass.
            fallback = (*task[:3], cfg.horizon)
            if worker_count() > 1:
                # A pool coordinator may have only a small allocator allowance;
                # unknown-support fallbacks must obey the worker VJP budget too.
                with FuturePool(trainer, common) as pool:
                    collect(fallback, next(iter(pool.map([fallback]))))
            else:
                collect(fallback, _vjp(trainer, common, fallback, transport=False))
            continue
        if known is not None:
            for i, used in enumerate(known):
                if used and accumulated[i] is None:
                    accumulated[i] = torch.zeros_like(params[i])
        record_replay(trainer, task[0], task[2], 0, cfg.horizon, 0.)
        _trace_memory(trainer, 'future_zero_vjp_skipped', start=task[0], future=task[2])
    return tuple(accumulated)


def _forward(trainer, common, future):
    from .dreamer import _restore_rng
    from .imagination_memory import _RandomTape
    cache, tape = {'items': []}, []
    from .eslong import active as es_active, calibration
    _restore_rng(common['before'], trainer.device)
    with torch.no_grad(), calibration(trainer, common['batch']) as probe, _RandomTape(tape, offload=True):
        trainer.imagine(common['batch'], **common['kwargs'], _memory_replay=True,
            _memory_config=common['cfg'], _memory_full_batch=common['batch'],
            _memory_future_cache=cache, _memory_collect_future=future)
    if es_active(trainer):
        cache['eslong_sensitivity'] = probe.get('sensitivity', 0.)
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
                if 'eslong_sensitivity' in part:
                    cache.update(eslong_sensitivity=part['eslong_sensitivity'], tape_prefix=prefix)
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
    if not 0 < fraction <= budget_cap():
        raise ValueError('worker memory fraction must be in (0,budget cap]')
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
                if common.get('eslong'):
                    from .eslong import forward_task, forward_stacked_task
                    function = forward_stacked_task if common.get('eslong_stack') else forward_task
                    result = function(trainer, common, task)
                else:
                    result = (_vjp(trainer, common, task, gate) if common['kind'] == 'backward'
                              else _forward(trainer, common, task))
                torch.save(result, path.with_suffix('.pt'))
                del result
            except BaseException:
                path.with_suffix('.error').write_text(traceback.format_exc())
            finally:
                if trainer.device.type == 'cuda':
                    (directory / 'memory.json').write_text(json.dumps({
                        'peak_allocated': torch.cuda.max_memory_allocated(),
                        'peak_reserved': torch.cuda.max_memory_reserved()}))
                if gate is not None:
                    gate.release()
            path.with_suffix('.done').touch()
            completed.add(path.stem)
        time.sleep(.05)


if __name__ == '__main__':
    _main(*sys.argv[1:])
