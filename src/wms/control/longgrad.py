"""Objective-preserving long-credit optimizer preconditioning and diagnostics.

The balance switch changes the optimizer direction, NOT the registered scalar
objective. OFF retains the original combined backward (including roundoff).
"""
from contextlib import contextmanager
import math

import torch


def norm(grads):
    return math.sqrt(sum(float(g.detach().double().square().sum())
                         for g in grads if g is not None))


def groups(trainer):
    optimizer_groups = (tuple(g['params'] for g in trainer.actor_optimizer.param_groups)
                        if hasattr(trainer, 'actor_optimizer') else trainer._longgrad_optimizer_groups)
    result = {f'optimizer_{i}': tuple(params) for i, params in enumerate(optimizer_groups)}
    for name in ('residual_head', 'own_encoder', 'report_encoder', 'attention'):
        module = getattr(trainer.policy, name, None)
        if module is not None:
            result[name] = tuple(module.parameters())
    result['group_encoder'] = tuple(trainer.group_encoder.parameters())
    return result


def group_norms(trainer, params, grads):
    by_id = {id(p): g for p, g in zip(params, grads)}
    return {name: norm(by_id.get(id(p)) for p in values)
            for name, values in groups(trainer).items()}


def combine(dense, long, *, balance):
    """One global norm cap at ratio 1, with no epsilon or fitted constant."""
    nd, nl = norm(dense), norm(long)
    if not math.isfinite(nd) or not math.isfinite(nl):
        raise ValueError('nonfinite dense/long actor gradient; optimizer not stepped')
    dot = sum(float((a.detach().double() * b.detach().double()).sum())
              for a, b in zip(dense, long) if a is not None and b is not None)
    cosine = max(-1., min(1., dot / nd / nl)) if nd and nl else 0.
    scale = min(1., nd / nl) if balance and nl else 1.
    out = []
    for a, b in zip(dense, long):
        if b is None:
            out.append(a)
        elif a is None:
            out.append(b * scale)
        else:
            out.append(a + b * scale)
    return out, dict(grad_norm_dense=nd, grad_norm_long=nl, grad_cosine=cosine,
                     grad_long_scale=scale, grad_norm_long_scaled=nl * scale)


@contextmanager
def phase(trainer, name):
    old = getattr(trainer, '_longgrad_phase', None)
    trainer._longgrad_phase = name
    try:
        yield
    finally:
        trainer._longgrad_phase = old


def actor_backward(trainer, dense_loss, long_loss, total_loss):
    params = tuple(p for p in trainer.actor_parameters if p.requires_grad)
    trainer._longgrad_futures = []
    if long_loss is None or not long_loss.requires_grad:
        total_loss.backward()
        dense = tuple(p.grad for p in params)
        long = (None,) * len(params)
        _, metrics = combine(dense, long, balance=False)
    else:
        # Keep the terms separate even when one dwarfs the other: subtracting
        # dense from an already-rounded total would lose the dense direction.
        with phase(trainer, 'dense'):
            dense = (torch.autograd.grad(dense_loss, params, retain_graph=True, allow_unused=True)
                     if dense_loss.requires_grad else (None,) * len(params))
        balance = trainer.config.long_grad_balance
        with phase(trainer, 'long'):
            long = torch.autograd.grad(long_loss, params, retain_graph=not balance, allow_unused=True)
        combined, metrics = combine(dense, long, balance=balance)
        if balance:
            for p, grad in zip(params, combined):
                p.grad = grad
        else:
            # Diagnostics must not change the legacy OFF update's arithmetic.
            total_loss.backward()
    metrics['grad_group_norms_dense'] = group_norms(trainer, params, dense)
    metrics['grad_group_norms_long'] = group_norms(trainer, params, long)
    metrics['grad_group_norms_combined'] = group_norms(trainer, params, (p.grad for p in params))
    metrics['grad_long_balance'] = trainer.config.long_grad_balance
    metrics['grad_long_futures'] = trainer._longgrad_futures
    return metrics


class StateTrace:
    """Passive full-tensor hooks at controller inputs; never modify a cotangent.

    Shared tensor hooks observe accumulated adjoints, not independent per-step
    parameter contributions. Checkpoint recomputes can repeat a hook; retain the
    maximum per (step, field, row), rather than double count recomputation.
    """
    def __init__(self):
        self.values = {}

    def watch(self, state, step):
        for key in ('s_m', 'v_mps', 'a_mps2'):
            value = state.get(key)
            if value is None or not value.requires_grad:
                continue

            def record(grad, index=(step, key)):
                size = grad.detach().double().square().flatten(1).sum(-1)
                prior = self.values.get(index)
                self.values[index] = size if prior is None else torch.maximum(prior, size)
            value.register_hook(record)

    def summary(self):
        if not self.values:
            return {'max_norm': 0., 'peak_step': None, 'top_steps': [], 'row_max_norms': []}
        steps = sorted({step for step, _ in self.values})
        values = torch.stack([sum(v for (s, _), v in self.values.items() if s == step)
                              for step in steps]).sqrt().cpu()
        norms = values.square().sum(-1).sqrt()
        finite = bool(torch.isfinite(values).all())
        # JSON-safe even if a future's internal adjoint is nonfinite. The actor
        # gradient guard remains fail-closed; telemetry never silently repairs it.
        order = norms.argsort(descending=True)[:8].tolist()
        return {'finite': finite, 'max_norm': float(norms.max()) if finite else None,
                'peak_step': steps[order[0]],
                'top_steps': [{'step': steps[i], 'norm': float(norms[i]) if finite else None}
                              for i in order],
                'row_max_norms': [float(v) if math.isfinite(float(v)) else None
                                  for v in values.max(0).values]}


def future_record(trainer, start, future, params, grads, trace):
    return dict(start=start, future=future, grad_norm=norm(grads),
                group_norms=group_norms(trainer, params, grads),
                observed_state_cotangent=trace.summary())
