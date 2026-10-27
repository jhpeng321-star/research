"""Default-off K=8 paired long credit with world-split, spread-scaled CQR.

Pinball fits are conditional quantiles of standardized paired errors, partitioned
by fitting-world empirical quantiles of paired mean/width. The scale feature is
the observed K-sample interquantile width; a world-
max finite-sample correction follows S-CQR/S-GATECAL (80% intervals). Neither
spread nor an encoder vote is treated as calibrated confidence by itself.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import math

import numpy as np
import torch

from .start_states import checked_path

K = 8
ALPHA = .2


class MissingCostClocks(ValueError):
    """Converted real pairs lack per-tick source-queue clocks: no CQR can be fitted."""


def conditional_scales(mean, width, model):
    bins = np.searchsorted(model.get('condition_edges', []), mean / width, side='left')
    return (np.asarray(model.get('conditional_lower', [model['lower_scale']]))[bins],
            np.asarray(model.get('conditional_upper', [model['upper_scale']]))[bins])


def world_role(world):
    # Preserve the already registered paired-critic development exclusion.
    if int(hashlib.sha256(('965140:' + str(world)).encode()).hexdigest(), 16) % 5 == 0:
        return 'validation'
    return ('fit', 'calibration', 'validation')[int(hashlib.sha256(
        ('LOOPLONG-2:' + str(world)).encode()).hexdigest(), 16) % 3]


def validate_model(worlds, mean, width, real, model):
    """Untouched-world coverage and WITHIN-world paired-benefit ranking.

    Compare paired action effects only within a world, never absolute levels
    across worlds. Single-contrast worlds supply no ranking evidence. No
    validation label changes a fitted quantile/correction.
    """
    unique = sorted(set(worlds))
    if model['correction_scale'] is None or not unique:
        return {'qualified': False, 'world_coverage': None, 'ranking_lower95': None}
    lower, upper = conditional_scales(mean, width, model)
    lo = mean + (lower - model['correction_scale']) * width
    hi = mean + (upper + model['correction_scale']) * width
    center = (lo + hi) / 2
    coverage, ranking = [], []
    for world in unique:
        mask = worlds == world
        coverage.append(bool(((lo[mask] <= real[mask]) & (real[mask] <= hi[mask])).all()))
        actual, predicted = real[mask], center[mask]
        dr = actual[:, None] - actual[None, :]
        dp = predicted[:, None] - predicted[None, :]
        informative = np.triu(np.ones(dr.shape, dtype=bool), 1) & (dr != 0)
        if informative.any():
            ranking.append(float(((np.sign(dp[informative]) == np.sign(dr[informative]))
                                  * 1. + (dp[informative] == 0) * .5).mean()))
    lower = None
    if len(ranking) >= 2:
        rng = np.random.default_rng(20261027)
        lower = float(np.quantile(np.asarray(ranking)[rng.integers(
            len(ranking), size=(2000, len(ranking)))].mean(1), .025))
    covered = float(np.mean(coverage))
    return {'qualified': covered >= 1 - ALPHA and lower is not None and lower > .5,
            'world_coverage': covered, 'ranking_lower95': lower,
            'ranking_worlds': len(ranking), 'worlds': unique,
            'ranking': 'within-world paired-benefit ordering; world-cluster lower95 > chance'}


def weighted_quantile(values, weights, q):
    order = np.argsort(values, kind='stable')
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    return float(v[np.searchsorted(np.cumsum(w), q * w.sum(), side='left')])


def sample_location_width(samples):
    a = np.asarray(samples, float)
    if a.ndim != 2 or a.shape[1] != K or not np.isfinite(a).all():
        raise ValueError('finite paired K=8 samples required')
    # Machine epsilon is a numerical divisor, not a behavior threshold.
    return a.mean(1), np.maximum(np.quantile(a, .9, axis=1) - np.quantile(a, .1, axis=1),
                                np.finfo(np.float32).eps)


def fit_cqr(rows, binding):
    models = {}
    for horizon in sorted({r['horizon_s'] for r in rows}):
        subset = [r for r in rows if r['horizon_s'] == horizon]
        worlds = np.asarray([str(r['world']) for r in subset])
        roles = np.asarray([world_role(w) for w in worlds])
        fit, cal = roles == 'fit', roles == 'calibration'
        mean, width = sample_location_width([r['samples'] for r in subset])
        real = np.asarray([r['real_benefit'] for r in subset], float)
        if not np.isfinite(real).all():
            raise ValueError('nonfinite CQR outcome')
        if not fit.any() or not cal.any():
            models[str(int(horizon))] = {'lower_scale': 0., 'upper_scale': 0.,
                'correction_scale': None, 'qualified': False,
                'reason': 'no disjoint fitting/calibration worlds'}
            continue
        residual = (real - mean) / width
        weights = np.array([1. / sum(worlds[fit] == w) for w in worlds[fit]])
        lo, hi = [weighted_quantile(residual[fit], weights, q) for q in (.1, .9)]
        # The partitions and pinball minimizers see fitting worlds only. This
        # corrects conditional WM bias; raw K spread alone never sets the bound.
        count = min(4, max(1, int(math.sqrt(len(set(worlds[fit]))))))
        edges = np.unique([weighted_quantile((mean / width)[fit], weights, q)
                           for q in np.arange(1, count) / count])
        bins = np.searchsorted(edges, mean / width, side='left')
        lower, upper = [], []
        for b in range(len(edges) + 1):
            selected = bins[fit] == b
            values = residual[fit][selected]
            lower.append(weighted_quantile(values, weights[selected], .1) if len(values) else lo)
            upper.append(weighted_quantile(values, weights[selected], .9) if len(values) else hi)
        conditional = {'condition_edges': edges.tolist(), 'conditional_lower': lower,
                       'conditional_upper': upper, 'lower_scale': lo, 'upper_scale': hi}
        qlo, qhi = conditional_scales(mean, width, conditional)
        scores = np.maximum(mean + qlo * width - real, real - mean - qhi * width) / width
        world_scores = [float(scores[cal & (worlds == w)].max()) for w in sorted(set(worlds[cal]))]
        rank = math.ceil((len(world_scores) + 1) * (1 - ALPHA))
        correction = max(0., sorted(world_scores)[rank - 1]) if rank <= len(world_scores) else None
        model = {**conditional,
            'correction_scale': correction, 'fit_worlds': sorted(set(worlds[fit])),
            'calibration_worlds': sorted(set(worlds[cal])), 'rank': rank,
            'qualified': False}
        validation = roles == 'validation'
        model['validation'] = validate_model(worlds[validation], mean[validation],
                                             width[validation], real[validation], model)
        model['qualified'] = model['validation']['qualified']
        models[str(int(horizon))] = model
    return {'schema': 'looplong_cqr/3', 'k': K, 'alpha': ALPHA,
            'binding': binding, 'horizons': models,
            'rule': 'fit-only conditional pinball residual/width; disjoint world-max CQR finite rank'}


def bounded_difference(difference, model, coefficients):
    """[K,H,N] paired benefit -> conservative signed row contribution.

    Negative cooperation coefficients use the upper endpoint before subtraction;
    otherwise subtracting two lower bounds could manufacture cooperation credit.
    """
    if difference.shape[0] != K:
        raise ValueError('long credit needs exactly K=8 paired futures')
    if model['correction_scale'] is None:
        # No finite rank: long effects supply no positive credit or gradient.
        return torch.zeros_like(difference)
    mean = difference.mean(0)
    width = (torch.quantile(difference, .9, dim=0) -
             torch.quantile(difference, .1, dim=0)).clamp_min(torch.finfo(difference.dtype).eps)
    bins = torch.bucketize((mean / width).detach().contiguous(),
                           mean.new_tensor(model.get('condition_edges', [])))
    lower = mean.new_tensor(model.get('conditional_lower', [model['lower_scale']]))[bins]
    upper = mean.new_tensor(model.get('conditional_upper', [model['upper_scale']]))[bins]
    lo = mean + (lower - model['correction_scale']) * width
    hi = mean + (upper + model['correction_scale']) * width
    bound = torch.where(coefficients >= 0, lo, hi)
    if not model['qualified']:
        # Finite but unvalidated bounds may only penalize, never reward an
        # apparent benefit, including negatively weighted cooperation terms.
        bound = torch.where(coefficients >= 0, bound.clamp_max(0), bound.clamp_min(0))
    # Native continuation is exactly zero, including its uncertainty.
    bound = torch.where((difference == 0).all(0), torch.zeros_like(bound), bound)
    return bound.unsqueeze(0).expand_as(difference)


def pessimistic_difference(trainer, imagined, difference, base, has_base, coefficients):
    artifact = trainer.long_credit
    h = int(imagined['imagination_horizon_s'])
    trusted = int(getattr(trainer, 'long_credit_trusted_s', 60))
    if h <= trusted:
        return difference
    model = artifact['horizons'].get(str(h))
    if model is None:
        raise ValueError('no horizon-matched loop CQR model')
    if (not getattr(trainer, 'long_credit_on_policy', False) or
            trainer.update_count != getattr(trainer, 'long_credit_bound_update', None)
            or not imagined.get('deterministic_policy', False)):
        # Calibration replays the deterministic collection actor. Exploratory
        # actions have no matching real labels, even before the first update.
        model = {**model, 'qualified': False}
    if not bool(has_base[:imagined['n_base']].all()):
        raise ValueError('long credit requires paired native rows for every policy start')
    rewards = imagined['rewards']
    short = rewards.clone()
    short[:, trusted:] = 0.
    long = rewards - short
    # Finite-horizon accrued effects only: a critic tail is not K-sample evidence.
    def paired(ret):
        return ret - torch.where(has_base, ret[..., base].detach(), torch.zeros_like(ret))
    short_returns = torch.flip(torch.flip(short, (1,)).cumsum(1), (1,))
    # Calibration targets are the complete (trusted,h] paired benefit. Use that same
    # interval (not uncalibrated shorter suffixes or TD-lambda shrinkage) as the
    # long component of every decision's advantage.
    long_benefit = paired(long.sum(1, keepdim=True))
    bound = bounded_difference(long_benefit, model, coefficients)
    # Only POLICY/B contrasts were validated. ONLY/GROUP/FLIP/NOT cannot
    # inherit that qualification without their own real probe pairs.
    pessimistic = bounded_difference(long_benefit, {**model, 'qualified': False}, coefficients)
    eligible = torch.arange(difference.shape[-1], device=difference.device) < imagined['n_base']
    bound = torch.where(eligible, bound, pessimistic)
    return paired(short_returns) + bound.expand_as(difference)


def hold_metrics(imagined):
    """Zero-tolerance sampled upstream group slowing runs, not a causal event label."""
    actions, masks = imagined['actions'], imagined['action_mask']
    # Only the deployed all-CAV rows; paired native rows are not actor samples.
    n = imagined['n_base']
    eligible = masks[..., :n, :]
    upstream = torch.stack([torch.stack([s['s_m'] < 0 for s in member[:-1]])
                             for member in imagined['states']])[..., :n, :]
    slow = eligible & upstream & (actions[..., :n, :] < 0)
    definition = '>=2 acting upstream CAV negative-command steps, uninterrupted >=30s'
    variants = imagined.get('variants')
    if variants is not None and all(i in variants.base_row for i in range(n)):
        native = [variants.base_row[i] for i in range(n)]
        speed = torch.stack([torch.stack([s['v_mps'] for s in member[:-1]])
                             for member in imagined['states']])
        present = torch.stack([torch.stack([s['present'] for s in member[:-1]])
                               for member in imagined['states']]).bool()
        initial_cav = torch.stack([member[0]['present'] & member[0]['is_cav']
                                   for member in imagined['states']])[:, None, :n]
        # Stable t0 IDs only: entrant-slot correspondence may diverge after
        # different queue admissions. A zero command can still hold speed below
        # native, so acceleration sign alone is not a metering descriptor.
        slow = (eligible & initial_cav & upstream & present[..., native, :]
                & (speed[..., :n, :] < speed[..., native, :]))
        definition = '>=2 t0 acting upstream CAVs slower than matched native, uninterrupted >=30s'
    grouped = slow.sum(-1) >= 2
    longest = torch.zeros_like(grouped[:, 0], dtype=torch.long)
    run = longest.clone()
    count = 0
    for t in range(grouped.shape[1]):
        run = torch.where(grouped[:, t], run + 1, 0)
        longest = torch.maximum(longest, run)
        count += int((run == 30).sum())
    return {'metering_like_holds_ge30s': count,
            'metering_like_sampled_starts': grouped.shape[0] * grouped.shape[2],
            'metering_like_max_hold_s': int(longest.max()),
            'metering_like_definition': definition}


def fit_from_loop(trainer, source, pairing, policy_returns, native_returns, policy_sha, wm_sha,
                  horizons, wm_config):
    """No-grad current-policy K futures from causal real starts; no future commands as inputs.

    Recorded per-tick full-population/source costs must reconcile to authoritative
    exports before any partial-horizon target is used. Missing clocks refuse.
    """
    import json
    from .paircrit import PairedProvider
    from .real_returns14 import NPZRealReturnProvider
    from .start_states import StartStateSampler
    from .counterfactual import Variants, KIND_ALL, KIND_BASE, acting_cavs
    from .dreamer import _rng_state, _restore_rng
    from wms.data.c4_pairs import PairedBranchSource
    from wms.config.load import load_train_config

    p, b = NPZRealReturnProvider(policy_returns), NPZRealReturnProvider(native_returns)
    PairedProvider(p, b, pairing, policy_sha)
    proof = json.loads(checked_path(pairing).read_text())
    cfg = load_train_config(str(checked_path(wm_config)))
    pairs = PairedBranchSource(checked_path(source), horizon_steps=600,
        history_s=cfg.data.history_s, history_stride_s=cfg.data.history_stride_s,
        entries_v2=True, junction_fix=cfg.data.junction_fix,
        merge_pivot_history=cfg.model.merge_decision == 'first_passage', name='loop14_pairs',
        arm_kinds=('B', 'POLICY'))
    sampler = StartStateSampler(pairs.dataset)
    by_key = {g['pair_key']: g for g in pairs.groups}
    old, curriculum, training = trainer.config, trainer.horizon_curriculum, trainer._training_imagination
    uncertainty = trainer.uncertainty
    rng = _rng_state(trainer.device)
    rows = []
    try:
        trainer.horizon_curriculum, trainer.uncertainty = None, None
        trainer._training_imagination = False
        for i, item in enumerate(proof['pairs']):
            group = by_key[item['pair_key']]
            native_row = group['rows'][0]
            policy_row = next(r for r in group['rows'][1:] if
                pairs.dataset._loader.entries[r]['branch']['kind'] == 'POLICY')
            curves = []
            for row, target in ((policy_row, p._returns[i]), (native_row, b._returns[i])):
                loader = pairs.dataset._loader
                ep_index, tick = loader.index[row]
                ep = loader.episode(ep_index)
                if not ep.has('entry_backlog'):
                    raise MissingCostClocks('converted real pairs have no entry_backlog clocks')
                kin = ep.array('veh_kin')[tick + 1:tick + 601]
                mask = ep.array('veh_mask')[tick + 1:tick + 601]
                queue = ep.array('entry_backlog')[tick:tick + 600]
                if len(kin) != 600 or queue.shape != (600, 2):
                    raise ValueError('complete 300s observable cost clocks required')
                costs = .5 * ((np.maximum(0., 1. - kin[..., 4] / 31.29) * mask).sum(-1) + queue.sum(-1))
                if not np.isclose(costs.sum(), -float(target), rtol=1e-6, atol=1e-3):
                    raise ValueError('recorded cost clocks do not reconcile to authoritative J')
                curves.append(costs)
            for h in horizons:
                trainer.config = replace(old, horizon=int(h), future_samples=K,
                    future_driver_scale=1., uncertainty_weight=0., low_penetration_horizon_cap=False,
                    entry_pad_max=max(old.entry_pad_max, math.ceil(old.entry_pad_max * h / old.horizon)))
                start = sampler._batch([sampler._load(policy_row)], trainer.device)
                with torch.no_grad():
                    start = trainer._entry_slots(start, torch.Generator().manual_seed(710003 + i))
                    cav = acting_cavs(start.state)
                    variants = Variants(torch.tensor([0, 0], device=trainer.device),
                        torch.cat((cav, torch.zeros_like(cav))),
                        torch.tensor([KIND_ALL, KIND_BASE], device=trainer.device), 1, {0: 1}, {}, [])
                    image = trainer.imagine(start, variants=variants, deterministic=True, difference_only=True)
                trusted = trainer.long_credit_trusted_s
                benefit = (image['rewards'][:, trusted:, 0] - image['rewards'][:, trusted:, 1]).sum(1)
                real = float((curves[1][2*trusted:int(2*h)] - curves[0][2*trusted:int(2*h)]).sum())
                rows.append({'world': item['seed'], 'horizon_s': h,
                             'samples': benefit.cpu().tolist(), 'real_benefit': real})
    finally:
        trainer.config, trainer.horizon_curriculum = old, curriculum
        trainer._training_imagination, trainer.uncertainty = training, uncertainty
        _restore_rng(rng, trainer.device)
    binding = {'wm_sha256': wm_sha, 'policy_sha256': policy_sha,
               'pairs_sha256': hashlib.sha256(checked_path(pairing).read_bytes()).hexdigest(),
                'objective': 'source_integral_plus_road_delay',
                'safe_horizon_s': trainer.long_credit_trusted_s}
    return fit_cqr(rows, binding)
