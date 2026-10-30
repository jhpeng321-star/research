"""Default-OFF, head-only antithetic long credit. No rollout autograd here."""
from contextlib import contextmanager, nullcontext
from dataclasses import fields, is_dataclass, replace
from types import SimpleNamespace
import math
import os

import torch

from .speedprof import profiled


def active(trainer):
    from .gated300 import active as qualified
    return (getattr(trainer, 'gated300', {}).get('config', {}).get('estimator', 'pathwise')
            == 'antithetic' and qualified(trainer))


def execution_config(trainer, cfg, batch):
    if not active(trainer):
        return cfg
    from .longskip import dense_replay_horizon
    if dense_replay_horizon(trainer, batch, cfg) >= cfg.horizon:
        raise ValueError('antithetic dense replay boundary must be shorter than the long horizon')
    # Force exact cached-prefix VJPs, even if the runtime omitted LONGSKIP flags.
    return replace(cfg, imagination_offload_futures=True,
                   imagination_replay_dense_prefix=True, imagination_skip_zero_vjp=True)


def head_parameters(trainer):
    head = getattr(trainer.policy, 'residual_head', None)
    if not isinstance(head, torch.nn.Linear) or head.out_features != 1:
        raise ValueError('antithetic long credit requires a linear residual output head')
    params = tuple(head.parameters())
    if not all(p.requires_grad for p in params):
        raise ValueError('antithetic residual output head must be trainable')
    return params


@contextmanager
def calibration(trainer, batch):
    """Local command Jacobian RMS at frozen t0, not a 300s sensitivity.

    The hook is also the integration seam for a changed residual decode law:
    policy.eslong_decode_slope(latent) may supply its local command slope.
    """
    stats = {}
    if not active(trainer):
        yield stats
        return
    head_parameters(trainer)
    n = batch.state['s_m'].shape[0]
    mask = (batch.state['present'] & batch.state['is_cav'] & (batch.state['lane'] >= 0))

    def observe(module, args, output):
        if stats:
            return
        features, latent = args[0][:n].detach(), output[:n].detach().squeeze(-1)
        radius = min(-trainer.policy.accel_min, trainer.policy.accel_max)
        decode = getattr(trainer.policy, 'eslong_decode_slope', None)
        slope = decode(latent) if decode else radius * (1 - latent.tanh().square())
        jac2 = slope.square() * (features.square().sum(-1) + 1)
        stats['sensitivity'] = float(jac2[mask].mean().sqrt()) if bool(mask.any()) else 0.

    handle = trainer.policy.residual_head.register_forward_hook(observe)
    try:
        yield stats
    finally:
        handle.remove()


def choose_sigma(trainer, sensitivity):
    cfg = trainer.gated300['config']
    params = head_parameters(trainer)
    theta = torch.cat([p.detach().flatten().double().cpu() for p in params])
    rms = float(theta.square().mean().sqrt())
    relative_scale = max(rms, 1 / math.sqrt(theta.numel()))
    exploration = trainer.exploration
    std = (exploration.rho_at(trainer._curriculum_position(advance=False))
           if exploration is not None and exploration.mode == 'additive_accel' else 0.)
    sigma = cfg.get('es_sigma', 0.)
    if not sigma:
        if std <= 0 or sensitivity <= 0:
            raise ValueError('automatic ES sigma needs positive additive exploration and head sensitivity; '
                             'supply long_credit.es_sigma explicitly for an inert t0 batch')
        sigma = std / sensitivity
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError('ES sigma must be finite and positive')
    return sigma, dict(eslong_sigma=sigma, eslong_head_parameter_rms=rms,
        eslong_sigma_relative_head_scale=sigma / relative_scale,
        eslong_local_command_sensitivity=sensitivity,
        eslong_predicted_command_rms=sigma * sensitivity, eslong_exploration_std=std)


@contextmanager
def perturb(trainer, direction, sigma):
    params = head_parameters(trainer)
    saved = [p.detach().clone() for p in params]
    try:
        with torch.no_grad():
            offset = 0
            for p, original in zip(params, saved):
                p.copy_(original + sigma * direction[offset:offset + p.numel()].reshape_as(p).to(p))
                offset += p.numel()
        yield
    finally:
        with torch.no_grad():
            for p, original in zip(params, saved):
                p.copy_(original)


def future_tape(tape, cache, future):
    prefix = cache['tape_prefix']
    begin = prefix if future == 0 else cache['items'][future - 1]['tape_end']
    return tape[:prefix] + tape[begin:cache['items'][future]['tape_end']]


def stack_sides():
    """Execution-only, checkpoint-neutral switch. Zero retains the serial path."""
    raw = os.environ.get('WMS_SPEED_ES_STACK_SIDES', '0')
    if raw not in ('0', '2', '4'):
        raise ValueError('WMS_SPEED_ES_STACK_SIDES must be 0, 2 or 4')
    return int(raw)


def stacked_variants(variants, sides, *, share_reference=False):
    """Repeat ROWS, not starts: every side keeps the original stochastic plans.

    Optional reference sharing is limited to native/CTRL0 rows with no actors.
    Learned all-CAV/unperturbed treatment rows are never aliases of a perturbation.
    """
    from .counterfactual import KIND_BASE, KIND_CTRL0
    original, owners, selections, shared = [], [], [], {}
    for side in range(sides):
        selection = []
        for row in range(variants.n_rows):
            reference = (share_reference and int(variants.row_kind[row]) in (KIND_BASE, KIND_CTRL0)
                         and not bool(variants.act_mask[row].any()))
            if not reference or row not in shared:
                slot = len(original)
                original.append(row)
                owners.append(side)
                if reference:
                    shared[row] = slot
            selection.append(shared[row] if reference else len(original) - 1)
        selections.append(torch.tensor(selection, device=variants.row_start.device))
    rows = variants.row_start.new_tensor(original)
    # Side-specific base maps are restored before ANY credit reduction.
    stacked = replace(variants, row_start=variants.row_start[rows],
                      row_kind=variants.row_kind[rows], act_mask=variants.act_mask[rows])
    return stacked, rows, owners, selections


@contextmanager
def row_heads(trainer, common, sides, selections, owners):
    """Functional side heads with original actor/group kernel geometry.

    Shape-dependent GEMM and CPU vectorized tanh tails otherwise change command
    bits even with an unchanged dot product. Physics still runs ONE stacked set.
    """
    params = head_parameters(trainer)
    values = []
    for direction, sign in sides:
        offset, tensors = 0, []
        for p in params:
            eps = common['directions'][direction][offset:offset + p.numel()].reshape_as(p).to(p)
            tensors.append(p.detach() + sign * common['sigma'] * eps)
            offset += p.numel()
        values.append(tensors)

    sources = [selection.new_tensor([r for r, slot in enumerate(selection.tolist())
                                    if owners[slot] == side])
               for side, selection in enumerate(selections)]

    def map_structure(function, value):
        if is_dataclass(value) and not isinstance(value, type):
            return replace(value, **{f.name: map_structure(function, getattr(value, f.name))
                                     for f in fields(value)})
        if isinstance(value, SimpleNamespace):
            return SimpleNamespace(**{k: map_structure(function, v) for k, v in vars(value).items()})
        from torch.utils._pytree import tree_map
        return tree_map(function, value)

    def side_call(function, args, kwargs, side):
        selection = selections[side]
        def select(x):
            return (x.index_select(0, selection.to(x.device))
                    if torch.is_tensor(x) and x.ndim and x.shape[0] == len(owners) else x)
        # Dataclass observations can also be nested inside positional arguments.
        return function(*(map_structure(select, x) for x in args),
                        **{k: map_structure(select, v) for k, v in kwargs.items()})

    def merge(outputs):
        from torch.utils._pytree import tree_map
        def combine(*values):
            if not torch.is_tensor(values[0]):
                return values[0]
            output = values[0].new_empty((len(owners), *values[0].shape[1:]))
            for selection, source, value in zip(selections, sources, values):
                output.index_copy_(0, selection[source], value.index_select(0, source))
            return output
        if is_dataclass(outputs[0]):
            return replace(outputs[0], **{f.name: merge([getattr(x, f.name) for x in outputs])
                                          for f in fields(outputs[0])})
        if isinstance(outputs[0], SimpleNamespace):
            return SimpleNamespace(**{k: merge([getattr(x, k) for x in outputs])
                                      for k in vars(outputs[0])})
        return tree_map(combine, *outputs)

    telemetry = [[] for _ in sides]

    def policy(*args, **kwargs):
        outputs = []
        for side, tensors in enumerate(values):
            parameters = dict(zip(('residual_head.weight', 'residual_head.bias'), tensors))
            def call(*a, **kw):
                return torch.func.functional_call(trainer.policy, parameters, a, kw)
            outputs.append(side_call(call, args, kwargs, side))
            telemetry[side].append(outputs[-1])
        return merge(outputs)

    def grouped(function):
        return lambda *a, **kw: merge([side_call(function, a, kw, s) for s in range(len(sides))])

    hooks = dict(_eslong_policy_call=policy, _eslong_group_call=grouped(trainer.group_encoder),
                 _eslong_observe_call=grouped(trainer.observation_builder),
                 _eslong_row_local=grouped, _eslong_side_policy_outputs=telemetry)
    saved = {key: getattr(trainer, key, None) for key in hooks}
    trainer.__dict__.update(hooks)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                delattr(trainer, key)
            else:
                setattr(trainer, key, value)


def split_outputs(result, variants, selections, telemetry=None):
    """Restore original row layout BEFORE gated300 sums, references or quantiles."""
    from torch.utils._pytree import tree_map
    total = result['rewards'].shape[-1]
    outputs = []
    for side, rows in enumerate(selections):
        def select(axis):
            return lambda x: (x.index_select(axis, rows.to(x.device))
                if torch.is_tensor(x) and x.ndim > axis and x.shape[axis] == total else x)
        out = {}
        for name, value in result.items():
            if name == 'variants':
                out[name] = variants
            elif name == 'states':
                out[name] = tree_map(select(0), value)
            elif name == 'disagreement':
                out[name] = tree_map(select(1), value)
            else:
                out[name] = tree_map(select(2), value)
        if out.get('entrance_zone') is not None:
            cav = torch.stack([torch.stack([s['present'] & s['is_cav'] for s in states[:-1]])
                               for states in out['states']])[..., :out['n_base'], :]
            zone = out['entrance_zone'][..., :out['n_base'], :]
            out['entrance_zone_fraction'] = float((zone & cav).sum()) / max(1, int(cav.sum()))
        if telemetry is not None:
            # Shared physical reference rows still have SIDE-SPECIFIC unused
            # actor predictions in diagnostics. Reconstruct those exact bytes;
            # only the state/WM physics of an actor-free row may be shared.
            commands = torch.stack([p[0] for p in telemetry[side]]).unsqueeze(0).detach()
            if 'antiexploit_commands' in out:
                out['antiexploit_commands']['deviation'] = commands
            if 'action_audit' in out:
                out['action_audit']['audit_action'] = commands
                out['action_audit']['audit_log_prob'] = torch.stack([
                    getattr(p, 'log_prob_accel', p[1]) for p in telemetry[side]]).unsqueeze(0).detach()
        outputs.append(out)
    return outputs


@profiled('es_forward_stacked')
def forward_stacked_task(trainer, common, task, *, full_output=False):
    """S head perturbations, one future, original starts/tape and side-local credit."""
    from .dreamer import _restore_rng
    from .imagination_memory import _RandomTape
    sides, future = task
    batch, kwargs, cfg = (common[k] for k in ('batch', 'kwargs', 'cfg'))
    variants = kwargs['variants']
    total = batch.state['s_m'].shape[0]
    # Sharing native rows is exact at the execution level for residual actors;
    # arbitrary user policies may couple their rows, so retain their full layout.
    share = bool(getattr(trainer.policy, 'residual_on_native', False))
    stacked, rows, owners, selections = stacked_variants(variants, len(sides), share_reference=share)
    tape = future_tape(common['tape'], common['cache'], future)
    replay = _RandomTape(tape, (torch.arange(total), rows, total, variants.n_rows))
    _restore_rng(common['before'], trainer.device)
    with torch.no_grad(), row_heads(trainer, common, sides, selections, owners), replay:
        outputs = trainer.imagine(batch, **{**kwargs, 'variants': stacked},
            _memory_replay=True, _memory_config=cfg, _memory_full_batch=batch,
            _memory_future_cache={'items': []}, _memory_collect_future=future,
            _eslong_stack=(variants, selections))
    if replay.position != len(tape):
        raise RuntimeError('ES stacked antithetic CRN trace shortened')
    if full_output:
        return outputs
    return [dict(samples=out['gated300_long_difference'].detach().cpu(),
                 actions=out['actions'][:, 0, :total].detach().cpu(),
                 mask=out['action_mask'][:, 0, :total].detach().cpu()) for out in outputs]


@profiled('es_forward')
def forward_task(trainer, common, task, *, full_output=False):
    """One perturbed future on the existing process-isolated forward workers."""
    from .dreamer import _restore_rng
    from .imagination_memory import _RandomTape
    direction, sign, future = task
    batch, kwargs, cfg = (common[k] for k in ('batch', 'kwargs', 'cfg'))
    total = batch.state['s_m'].shape[0]
    n_rows = kwargs['variants'].n_rows
    selection = (torch.arange(total), torch.arange(n_rows), total, n_rows)
    tape = future_tape(common['tape'], common['cache'], future)
    replay = _RandomTape(tape, selection)
    cache = {'items': []}
    _restore_rng(common['before'], trainer.device)
    with torch.no_grad(), perturb(trainer, common['directions'][direction], sign * common['sigma']), replay:
        out = trainer.imagine(batch, **kwargs, _memory_replay=True, _memory_config=cfg,
            _memory_full_batch=batch, _memory_future_cache=cache, _memory_collect_future=future)
    if replay.position != len(tape):
        raise RuntimeError('ES antithetic CRN trace shortened')
    if full_output:
        return out
    # Transport only the credit and t0 executions; never keep M rollout caches.
    return dict(samples=out['gated300_long_difference'].detach().cpu(),
                actions=out['actions'][:, 0, :total].detach().cpu(),
                mask=out['action_mask'][:, 0, :total].detach().cpu())


def paired_loss_difference(trainer, plus, minus, imagined, coefficients, scale):
    """Registered per-side gain gates + symmetric pair gate, then joint q10.

    Negative differences use q10(-difference)>0 (equivalently q90(diff)<0).
    This makes direction reversal exact; a one-sided pair gate would bias ES.
    """
    from .gated300 import gate, SIGMA
    from .antiexploit import tail_credit
    sigma = SIGMA[imagined['gated300_horizon_s']]
    eligible = coefficients != 0
    gain_only = trainer.gated300['config'].get('mode') == 'gain_only'
    p = gate(plus, sigma, eligible, gain_only=gain_only)[0]
    m = gate(minus, sigma, eligible, gain_only=gain_only)[0]
    _, on, _, median = gate(plus - minus, sigma, eligible, gain_only=False)
    p, m = (torch.where(on.unsqueeze(0), x, 0.) for x in (p, m))
    def value(samples):
        selected, _ = tail_credit(imagined, samples, coefficients, enabled=True)
        return -(selected.mean((0, 1)) * coefficients).sum()
    delta = trainer.gated300['config']['w_long'] * (value(p) - value(m)) / scale
    return delta, dict(accepted=int(on.sum()), rejected=int(eligible.sum()) - int(on.sum()),
                      on=on.cpu().tolist(), median=[
                          [float(x) if math.isfinite(float(x)) else None for x in row]
                          for row in median.cpu()])


def estimate(directions, differences, sigma):
    """Gaussian antithetic estimate; rejected directions stay in denominator M."""
    return (differences.to(directions)[:, None] * directions).mean(0) / (2 * sigma)


def gradient(trainer, imagined, weights, scale):
    from .dreamer import _rng_state, _restore_rng
    from .counterfactual import row_coefficients
    from .parallel_futures import FuturePool, worker_count, parallel_forward
    context = trainer._eslong_context
    sigma, metrics = choose_sigma(trainer, context['sensitivity'])
    dimension = sum(p.numel() for p in head_parameters(trainer))
    m = trainer.gated300['config'].get('es_directions', 2)
    # A per-update private generator needs no new checkpoint state and never
    # consumes the dense imagination RNG stream. Update count is checkpointed.
    generator = torch.Generator().manual_seed(7_000_027 * (trainer.update_count + 1))
    directions = torch.randn(m, dimension, generator=generator)
    common = {**context, 'kind': 'forward', 'eslong': True,
              'directions': directions, 'sigma': sigma}
    # Workers need the CRN boundaries, never the unperturbed full-K state cache.
    common['cache'] = dict(tape_prefix=context['cache']['tape_prefix'],
        items=[dict(tape_end=item['tape_end']) for item in context['cache']['items']])
    count = len(context['cache']['items'])
    sides = [(i, sign) for i in range(m) for sign in (1, -1)]
    stacking = stack_sides()
    common['eslong_stack'] = bool(stacking)
    tasks = ([(sides[start:start + stacking], k)
              for start in range(0, len(sides), stacking) for k in range(count)] if stacking else
             [(i, sign, k) for i, sign in sides for k in range(count)])
    coefficients = row_coefficients(imagined['variants'], weights,
        imagined['returns'].shape[-1], imagined['n_base'])
    after = _rng_state(trainer.device)
    old_training = trainer._training_imagination
    deltas, receipts, responses = [], [], []
    try:
        trainer._training_imagination = True
        pooled = trainer.device.type == 'cuda' or parallel_forward()
        manager = FuturePool(trainer, common) if pooled else nullcontext()
        with manager as pool:
            function = forward_stacked_task if stacking else forward_task
            results = pool.map(tasks) if pool else (function(trainer, common, t) for t in tasks)
            if stacking:
                # The pool yields groups in side-group/future order. Reconstruct
                # the original direction/sign/future ordering without reductions.
                groups, ordered = iter(results), []
                for start in range(0, len(sides), stacking):
                    futures = [next(groups) for _ in range(count)]
                    ordered.extend(futures[k][s] for s in range(len(futures[0])) for k in range(count))
                results = iter(ordered)
            else:
                results = iter(results)
            for _ in range(m):
                sides = [[next(results) for _ in range(count)] for _ in range(2)]
                samples = [torch.cat([r['samples'] for r in side]) for side in sides]
                delta, receipt = paired_loss_difference(trainer, *samples, imagined,
                                                        coefficients, scale)
                deltas.append(delta)
                receipts.append(receipt)
                mask = sides[0][0]['mask'] & sides[1][0]['mask']
                diff = (sides[0][0]['actions'] - sides[1][0]['actions']) / 2
                responses.append(float(diff[mask].square().mean().sqrt()) if bool(mask.any()) else 0.)
    finally:
        trainer._training_imagination = old_training
        _restore_rng(after, trainer.device)
        trainer._eslong_context = None
    vector = estimate(directions, torch.stack(deltas), sigma)
    by_id, offset = {}, 0
    for p in head_parameters(trainer):
        by_id[id(p)] = vector[offset:offset + p.numel()].reshape_as(p).to(p)
        offset += p.numel()
    grads = tuple(by_id.get(id(p)) for p in trainer.actor_parameters if p.requires_grad)
    metrics.update(eslong_directions=m, eslong_dimension=dimension,
        eslong_pairs=receipts, eslong_accepted_pairs=sum(r['accepted'] for r in receipts),
        eslong_rejected_pairs=sum(r['rejected'] for r in receipts),
        eslong_accepted_directions=sum(r['accepted'] > 0 for r in receipts),
        eslong_t0_executed_half_difference_rms=responses,
        eslong_perturbed_forwards=2 * m * count, eslong_forwards_per_update=(1 + 2 * m) * count,
        eslong_forward_workers=worker_count('forward') if pooled else 0,
        eslong_loss_differences=[float(d) for d in deltas])
    if stacking:
        metrics.update(eslong_stack_sides=stacking, eslong_stacked_forward_calls=len(tasks))
    return grads, metrics


def backward(trainer, dense_loss, grads):
    from .longgrad import phase, combine, group_norms
    from .longskip import metrics as replay_metrics
    params = tuple(p for p in trainer.actor_parameters if p.requires_grad)
    with phase(trainer, 'dense'):
        dense = (torch.autograd.grad(dense_loss, params, allow_unused=True)
                 if dense_loss.requires_grad else (None,) * len(params))
    combined, metrics = combine(dense, grads, balance=trainer.config.long_grad_balance)
    for p, g in zip(params, combined):
        p.grad = g
    metrics.update(grad_norm_long_es=metrics['grad_norm_long'],
        grad_long_balance=trainer.config.long_grad_balance, grad_long_futures=[],
        grad_group_norms_dense=group_norms(trainer, params, dense),
        grad_group_norms_long=group_norms(trainer, params, grads),
        grad_group_norms_combined=group_norms(trainer, params, combined))
    metrics.update(replay_metrics(trainer))
    return metrics
