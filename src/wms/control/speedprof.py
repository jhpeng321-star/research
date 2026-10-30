"""Default-off execution/diagnostic controls; no objective or estimator changes."""
from contextlib import contextmanager
from functools import wraps
import json
import os
import time


def rescore_every():
    raw = os.environ.get('WMS_SPEED_RESCORE_EVERY', '1')
    if not raw.isdigit() or int(raw) < 1:
        raise ValueError('WMS_SPEED_RESCORE_EVERY must be a positive integer')
    return int(raw)


def skip_grad_diagnostics():
    return os.environ.get('WMS_SPEED_SKIP_GRAD_DIAGNOSTICS') == '1'


def rescore_due(trainer, switches, phase_update, *, final=False):
    every = rescore_every()
    if every == 1:
        return True
    # Native residual actors return identically zero entropy. Thinning their
    # scores can change only the inert entropy-weight bookkeeping. Never permit
    # that argument for a stochastic actor with active improvement annealing.
    native = getattr(trainer.policy, 'residual_on_native', False)
    if (not native and trainer.config.entropy_weight != 0.
            and float(switches['entropy_decay_on_improvement']) != 1.):
        raise ValueError('rescore cadence requires diagnostic-only scoring (no active entropy annealing)')
    return (final or phase_update % every == 0
            or phase_update == int(switches['signal_gate_updates'])
            or trainer.update_count == int(switches['signal_gate_updates']))


def panel_key(trainer, val, panel, device):
    """Only reuse the immediately preceding matched native panel, never a rollout row.

    The scoring paths have identical row-local reductions when matched scoring
    is enabled. Restrict to non-relabelled validation and bind live parameters,
    WM identities, all execution configs and the frozen scoring phase.
    """
    if (os.environ.get('WMS_SPEED_REUSE_PANEL') != '1'
            or getattr(val, 'relabel', True)
            or not getattr(trainer, 'gated300', {}).get('config', {}).get('score_training_horizon')):
        return None
    modules = (*trainer.world_models, trainer.policy, trainer.group_encoder)
    versions = tuple((id(m), tuple((id(p), p._version) for p in (*m.parameters(), *m.buffers())))
                     for m in modules)
    configs = tuple(repr(getattr(trainer, name, None)) for name in (
        'config', 'reward_weights', 'counterfactual', 'gated300', 'horizon_curriculum',
        'exploration', '_gatemeas_score_position', 'world_model_ids'))
    return id(val), repr(panel), str(device), trainer.update_count, versions, configs


@contextmanager
def phase(trainer, name, **details):
    """Wall/CPU timing only; no device sync, RNG draw, or global profiler attach."""
    if os.environ.get('WMS_SPEED_PROFILE') != '1' or trainer.output_dir is None:
        yield
        return
    start, cpu = time.monotonic(), time.process_time()
    try:
        yield
    finally:
        record = dict(phase=name, update=trainer.update_count, seconds=time.monotonic()-start,
                      cpu_seconds=time.process_time()-cpu, epoch=time.time(), **details)
        with (trainer.output_dir/'speedprof_execution.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')


def profiled(name):
    def decorate(function):
        @wraps(function)
        def measured(trainer, *args, **kwargs):
            with phase(trainer, name):
                return function(trainer, *args, **kwargs)
        return measured
    return decorate


def receipt():
    return dict(rescore_every=rescore_every(), reuse_panel=os.environ.get('WMS_SPEED_REUSE_PANEL') == '1',
                profile=os.environ.get('WMS_SPEED_PROFILE') == '1',
                skip_grad_diagnostics=skip_grad_diagnostics(),
                native_entropy_bookkeeping='diagnostic-only when cadence > 1')
