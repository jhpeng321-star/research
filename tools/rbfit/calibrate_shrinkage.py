"""Learn population contamination from pre-t0 FIT motion, never action outcomes.

Prefix240..270 posterior predicts270..300 motion. Optimize the CRPS of the
physical acceleration-profile mixture r*posterior+(1-r)*population, with
r=n/(n+learned_prior_count). The population is the existing empirical-Bayes
FIT fit, so this is internal calibration, NOT independent validation.
Later300..360 motion is scored only after the policy is frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import gzip
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from idm_fit import (FIT_WORLDS, extract_rows, native_episode, next_speed,
                     observed, read_episode, summarize, write_json)
from idm_mixture import moments


def posterior_weight(n, coordinate):
    """Compact optimization coordinate prior_count/(1+prior_count), endpoints exact."""
    n = np.asarray(n, dtype=float)
    if coordinate == 1:
        return np.zeros_like(n)
    strength = coordinate / (1-coordinate)
    return np.divide(n, n+strength, out=np.zeros_like(n), where=n+strength > 0)


def crps_coefficients(values, target, prior, posterior):
    """Exact CRPS(r)=c+b*r+a*r²; O(rows*profiles*log(profiles))."""
    order = values.argsort(axis=1)
    x = np.take_along_axis(values, order, axis=1)
    absolute = abs(values-target[:, None])

    def score(weights):
        sw = weights[order]
        return absolute@weights - np.sum(sw*x*(2*sw.cumsum(axis=1)-sw-1), axis=1)

    c, end, mid = score(prior), score(posterior), score((prior+posterior)/2)
    return np.stack((c, 4*mid-3*c-end, 2*(c+end-2*mid)), axis=-1)


def select_coordinate(coefficients, counts):
    def objective(coordinate):
        r = posterior_weight(counts, coordinate)
        return float(np.sum(coefficients[:, 0]+r*coefficients[:, 1]+r*r*coefficients[:, 2]))
    fit = minimize_scalar(objective, bounds=(0, 1), method='bounded',
                          options={'xatol': 1e-10})
    candidates = (0., float(fit.x), 1.)
    best = min(candidates, key=objective)
    return best, objective(best)


def predict(rows, support):
    return (next_speed(rows[:, 3, None], rows[:, 5, None], rows[:, 6, None],
                       support, rows[0, 9])-rows[:, 3, None])/rows[:, 9, None]


def bayes(rows, support, prior, sigma):
    if not len(rows):
        return prior
    sse = ((predict(rows, support)-observed(rows)[:, None])**2).sum(0)
    joint = np.log(prior)-sse/(2*sigma*sigma)
    return np.exp(joint-logsumexp(joint))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--fit', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if not args.out.parent.is_dir():
        raise ValueError('Output parent must exist')
    args.out.mkdir(exist_ok=False)
    started = time.time()
    population_bytes = (args.fit/'idm_population.json').read_bytes()
    pop = json.loads(population_bytes)
    if pop['schema'] != 'wm_phys.idm_population.v1' or pop['distribution'] != 'finite_support_mixture':
        raise ValueError('Expected frozen finite-support physical population')
    support, prior = np.array(pop['support']), np.array(pop['weights'])
    sigma = pop['sigma_accel_mps2']
    coefficients, ns, keys, worlds, sources = [], [], [], [], []
    for world in FIT_WORLDS:
        arrays, provenance = read_episode(native_episode(args.root, world), world)
        rows = extract_rows(arrays)[0]
        sources.append(provenance)
        prefix = rows[(rows[:, 0] >= 240) & (rows[:, 0] < 270)]
        suffix = rows[(rows[:, 0] >= 270) & (rows[:, 0] < 300)]
        for vid in np.unique(suffix[:, 1]).astype(int):
            train, target = prefix[prefix[:, 1] == vid], suffix[suffix[:, 1] == vid]
            post = bayes(train, support, prior, sigma)
            cc = crps_coefficients(predict(target, support), observed(target), prior, post)
            coefficients.append(cc.sum(0))
            ns.append(len(train))
            keys.append(f'{world}:{vid}')
            worlds.append(world)
        print(json.dumps({'phase': 'internal_calibration', 'world': world}), flush=True)
    coefficients, ns, worlds = np.array(coefficients), np.array(ns), np.array(worlds)
    coordinate, total_score = select_coordinate(coefficients, ns)
    # Freeze the policy BEFORE inspecting later validation motion.
    policy = {'schema': 'wm_phys.idm_shrinkage.v1', 'population_only': coordinate == 1,
        'prior_count': None if coordinate == 1 else coordinate/(1-coordinate),
        'formula': 'posterior_fraction=n_observations/(n_observations+prior_count); zero evidence -> population',
        'idm_population_sha256': hashlib.sha256(population_bytes).hexdigest(),
        'fit_worlds': list(FIT_WORLDS), 'conditioning_end_s': 300,
        'prefix_interval_s': [240, 270], 'calibration_interval_s': [270, 300],
        'objective': 'Physical acceleration-profile CRPS; no runtime residual noise',
        'qualification': 'Internal empirical-Bayes calibration, not coverage or action-effect qualification'}
    write_json(args.out/'idm_shrinkage.json', policy)
    (args.out/'idm_population.json').write_bytes(population_bytes)
    world_crossfits = []
    for world in FIT_WORLDS:
        train, test = worlds != world, worlds == world
        c, _ = select_coordinate(coefficients[train], ns[train])
        r = posterior_weight(ns[test], c)
        cc = coefficients[test]
        world_crossfits.append({'world': world, 'coordinate': c,
            'crps_sum': float(np.sum(cc[:, 0]+r*cc[:, 1]+r*r*cc[:, 2])),
            'note': 'Policy scalar crossfit only; empirical-Bayes population still shared'})
    names = ('population', 'raw_posterior', 'shrunk_posterior')
    scores = {name: {'error': [], 'crps': [], 'covered': [], 'width': []} for name in names}
    driver_diagnostics = {}
    # Schema-compatible, fully materialized joint posteriors for UNC's own
    # committed FittedHiddenState reader; no new runtime policy is required.
    posterior_stream = gzip.open(args.out/'idm_posteriors.json.gz', 'xt', encoding='utf-8')
    header = {'schema': 'wm_phys.idm_posteriors.v1', 'parameter_order': pop['parameter_order'],
        'conditioning_end_s': 300, 'identity_key': 'world:observable_vid',
        'distribution': 'finite_support_mixture', 'fallback': 'idm_population.json',
        'shrinkage_applied': True, 'shrinkage_policy': 'idm_shrinkage.json'}
    posterior_stream.write(json.dumps(header)[:-1]+',"drivers":{')
    first_driver = True
    for world in FIT_WORLDS:
        arrays, _ = read_episode(native_episode(args.root, world), world)
        rows, histories, _ = extract_rows(arrays)
        train = rows[(rows[:, 0] >= 240) & (rows[:, 0] < 300)]
        target = rows[(rows[:, 0] >= 300) & (rows[:, 0] < 360)]
        fitted = {}
        for vid, duration in histories.items():
            rr = train[train[:, 1] == vid]
            post = bayes(rr, support, prior, sigma)
            r = float(posterior_weight(len(rr), coordinate))
            mixed = (1-r)*prior+r*post
            mixed /= mixed.sum()
            fitted[vid] = (post, mixed, r)
            indices = np.flatnonzero(mixed > 0)
            driver = dict(**moments(support, mixed), support_indices=indices.tolist(),
                weights=mixed[indices].tolist(), n_observations=len(rr),
                observed_history_s=duration, posterior_fraction=r, delta=4)
            if not first_driver:
                posterior_stream.write(',')
            posterior_stream.write(json.dumps(f'{world}:{vid}')+':'+json.dumps(driver))
            first_driver = False
        for vid in np.unique(target[:, 1]).astype(int):
            rr, tt = train[train[:, 1] == vid], target[target[:, 1] == vid]
            post, mixed, r = fitted.get(vid, (prior, prior, 0.))
            values, y = predict(tt, support), observed(tt)
            order = values.argsort(axis=1)
            x = np.take_along_axis(values, order, axis=1)
            driver_diagnostics[f'{world}:{vid}'] = {'n_observations': len(rr), 'posterior_fraction': r}
            for name, w in zip(names, (prior, post, mixed)):
                sw = w[order]
                cw = sw.cumsum(axis=1)
                crps = abs(values-y[:, None])@w - np.sum(sw*x*(2*cw-sw-1), axis=1)
                lo = x[np.arange(len(x)), (cw >= .025).argmax(axis=1)]
                hi = x[np.arange(len(x)), (cw >= .975).argmax(axis=1)]
                scores[name]['error'].extend(values@w-y)
                scores[name]['crps'].extend(crps)
                scores[name]['covered'].extend((y >= lo) & (y <= hi))
                scores[name]['width'].extend(hi-lo)
        print(json.dumps({'phase': 'later_FIT_validation', 'world': world}), flush=True)
    posterior_stream.write('}}\n')
    posterior_stream.close()
    result = {name: {'n': len(s['crps']), 'expected_acceleration_error_mps2': summarize(s['error']),
        'physical_profile_acceleration_crps': float(np.mean(s['crps'])),
        'physical_profile_95pct_coverage': float(np.mean(s['covered'])),
        'mean_interval_width_mps2': float(np.mean(s['width']))} for name, s in scores.items()}
    result.update(elapsed_s=time.time()-started, policy=policy,
                  internal_calibration_crps_sum=total_score, scalar_world_crossfits=world_crossfits,
                  qualification=policy['qualification'])
    write_json(args.out/'validation.json', result)
    write_json(args.out/'driver_shrinkage.json', driver_diagnostics)
    write_json(args.out/'provenance.json', {'sources': sources,
        'idm_population_sha256': policy['idm_population_sha256'],
        'idm_posteriors_gzip_sha256': hashlib.sha256((args.out/'idm_posteriors.json.gz').read_bytes()).hexdigest()})
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
