"""S-LONGWM physical consistency and paired effects; measured gradient parity.

The loop adapter applies this only to the current cycle's complete real pairs.
Scales and weights follow c4/longwm@908c6f50, not behavioral constants.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .losses.aggregate import (AggregateLossConfig, MERGE_S1_M, _recorded_kin,
                               soft_cell_fields, soft_crossings)
from .losses.agg_horizon import AggHorizonConfig, region_delay_queue

HORIZONS_S = (60., 120., 180., 240., 300.)


def traffic_views(s, v, lane, mask, initial, *, dt_s, valid, continuity=None):
    cfg = AggregateLossConfig(dt_s=dt_s, cell_temp_m=5.)
    fields = soft_cell_fields(s, v, lane, mask, cfg=cfg)
    region = region_delay_queue(s, v, lane, mask, cfg=AggHorizonConfig(
        dt_s=dt_s, region_temp_m=5.))
    prev_s = torch.cat((initial['s_m'].unsqueeze(1), s[:, :-1]), 1)
    prev_mask = torch.cat((initial['present'].unsqueeze(1), mask[:, :-1]), 1)
    continuity = prev_mask & mask if continuity is None else continuity
    return {'density': fields['density'], 'speed': fields['mean_speed'],
            'queue': region['queue'],
            'flow': soft_crossings(prev_s, s, continuity, (MERGE_S1_M,), 5.),
            'delay': region['delay'], 'valid': valid}


def horizon_features(view, horizons, dt_s):
    out = {k: [] for k in ('density', 'speed', 'queue', 'flow', 'delay')}
    for horizon in horizons:
        n = int(round(horizon / dt_s))
        valid = view['valid'][:, :n].to(view['delay'].dtype)
        for name in out:
            value = (view[name][:, :n] * valid.unsqueeze(-1)).sum(1)
            if name in ('density', 'speed', 'queue'):
                value = value / valid.sum(1, keepdim=True).clamp_min(1)
            out[name].append(value)
    return {name: torch.stack(values, 1) for name, values in out.items()}


def consistency_components(ctx, batch):
    t, road = ctx.targets, ctx.targets.road
    if ctx.c0 != 0 or ctx.c1 != t.n_steps:
        raise ValueError('longwm requires one complete pathwise unroll')
    horizons = [h for h in HORIZONS_S if h <= t.n_steps * t.dt_s + 1e-6]
    if not horizons or road is None or 'road_valid' not in batch:
        raise ValueError('longwm needs >=60s explicit recorded road observations')
    valid = batch['road_valid'].to(t.states.device).bool()[:, 1:t.n_steps + 1]
    if valid.shape != (t.states.shape[0], t.n_steps) or not bool(valid.all()):
        raise ValueError('longwm refuses incomplete recorded futures')
    k = None if road.full else road.k
    ps, pv, pl, pm = (ctx.pred[n][..., :k] if k is not None else ctx.pred[n]
                      for n in ('positions', 'speeds', 'lanes', 'masks'))
    initial = {n: v[..., :k] if k is not None else v for n, v in ctx.current.items()
               if n in ('s_m', 'present')}
    pf = horizon_features(traffic_views(ps, pv, pl, pm.bool(), initial,
                         dt_s=t.dt_s, valid=valid), horizons, t.dt_s)
    with torch.no_grad():
        rs, rl, rv = _recorded_kin(t.states, road, 0, t.n_steps + 1)
        rf = horizon_features(traffic_views(rs[:, 1:], rv[:, 1:], rl[:, 1:],
            road.present[:, 1:t.n_steps + 1],
            {'s_m': rs[:, 0], 'present': road.present[:, 0]}, dt_s=t.dt_s,
            valid=valid, continuity=road.cont[:, :t.n_steps]), horizons, t.dt_s)
    out, scales = {}, {}
    for name in pf:
        scale = rf[name].square().mean().sqrt()
        if float(scale) == 0.:
            scale = (pf[name].detach() - rf[name]).square().mean().sqrt()
        scale = scale.clamp_min(torch.finfo(ps.dtype).eps)
        scales[name] = float(scale.detach())
        out[name] = F.smooth_l1_loss(pf[name] / scale, rf[name] / scale)
        if 'pair_ref' in batch:
            ref = batch['pair_ref'].to(ps.device).long()
            arm = ref != torch.arange(len(ref), device=ref.device)
            if bool(arm.any()):
                pred = (pf[name] - pf[name][ref])[arm]
                real = (rf[name] - rf[name][ref])[arm]
                out['paired_' + name] = F.smooth_l1_loss(pred / scale, real / scale)
    ctx.cache['longwm_feature_scales'] = scales
    return out


def gradient_parity(base, components, parameters):
    def norm(loss):
        if not loss.requires_grad:
            return 0.
        grads = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)
        return sum(float(g.detach().double().square().sum()) for g in grads if g is not None) ** .5

    base_norm = norm(base)
    norms = {n: norm(v) for n, v in components.items()}
    active = sum(v > 0 for v in norms.values())
    weights = {n: base_norm / (active * v) if active and v > 0 else 0. for n, v in norms.items()}
    if not all(torch.isfinite(torch.tensor(w)) for w in weights.values()):
        raise RuntimeError('non-finite longwm gradient-parity weight')
    return weights, {'base_gradient_norm': base_norm, 'gradient_norms': norms,
                     'weights': weights, 'raw_losses': {n: float(v.detach()) for n, v in components.items()}}
