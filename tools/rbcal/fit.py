"""Fit physical latent spread from native FIT motion/counters only (CPU)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln, logsumexp

from tools.wmdecide.rbeval_adapter import safe_path, sha256

FIT_WORLDS = tuple(range(969400, 969432))


def bounded(objective):
    """Global compact scan, then bounded local refinement, including endpoints."""
    grid = np.linspace(0., 1., 65)
    values = [objective(x) for x in grid]
    i = int(np.argmin(values))
    lo, hi = grid[max(0, i-1)], grid[min(64, i+1)]
    result = minimize_scalar(objective, bounds=(lo, hi), method='bounded',
                             options={'xatol': 1e-10})
    return min((0., 1., float(grid[i]), float(result.x)), key=objective)


def weighted_crps_terms(values, targets, prior, other):
    """Sum CRPS(prior + r*(other-prior)) = c + b*r + a*r²."""
    order = values.argsort(axis=1)
    x = np.take_along_axis(values, order, axis=1)
    absolute = np.abs(values-targets[:, None]).sum(0)

    def score(weights):
        w = weights[order]
        return float(absolute@weights - np.sum(w*x*(2*w.cumsum(1)-w-1)))

    c, end, mid = score(prior), score(other), score((prior+other)/2)
    return np.array([c, 4*mid-3*c-end, 2*(c+end-2*mid)])


def quadratic_fit(coefficients):
    c, b, a = np.asarray(coefficients).sum(0)
    candidates = [0., 1.]
    if a > 0:
        candidates.append(float(np.clip(-b/(2*a), 0., 1.)))
    return min(candidates, key=lambda r: c+b*r+a*r*r)


def concentration_fit(coefficients, counts):
    coefficients, counts = np.asarray(coefficients), np.asarray(counts)

    def score(coordinate):
        if coordinate == 1:
            r = np.zeros_like(counts, dtype=float)
        else:
            alpha = coordinate/(1-coordinate)
            r = np.divide(counts, counts+alpha, out=np.zeros_like(counts, dtype=float),
                          where=counts+alpha > 0)
        return float(np.sum(coefficients[:, 0]+r*coefficients[:, 1]+r*r*coefficients[:, 2]))

    coordinate = bounded(score)
    return None if coordinate == 1 else coordinate/(1-coordinate)


def count_nll(count, target, dispersion):
    """Same-mean Gamma-Poisson. Dispersion=0 is exact Poisson endpoint."""
    mean = np.asarray(count, float)+.5  # Equal 150 s prefix/target exposures.
    target = np.asarray(target, float)
    if dispersion == 0:
        return mean-target*np.log(mean)+gammaln(target+1)
    shape = mean/dispersion
    logp = -np.log1p(dispersion)
    logq = np.log(dispersion)-np.log1p(dispersion)
    return -(gammaln(target+shape)-gammaln(shape)-gammaln(target+1)
             + shape*logp+target*logq)


def arrival_fit(observations, excluded=None):
    rows = [r for r in observations if r['world'] != excluded]
    count, target = np.array([r['prefix'] for r in rows]), np.array([r['target'] for r in rows])
    # Infinity has infinite log score for positive observed counts. Use a compact
    # coordinate and avoid evaluating the divergent endpoint numerically.
    objective = lambda u: float(count_nll(count, target, u/(1-u)).mean()) if u < 1 else np.inf
    coordinate = bounded(objective)
    return float(coordinate/(1-coordinate))


def fit_motion(root, population, policy):
    # Only trusted existing FIT selector is imported; all reads remain allowlisted.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'rbfit'))
    from tools.rbfit.idm_fit import native_episode, read_episode, extract_rows
    from tools.rbint.motion_posterior import chain_gaps, profile_next_speed

    support, prior = np.array(population['support']), np.array(population['weights'])
    prior /= prior.sum()
    sigma, strength = population['sigma_accel_mps2'], policy['prior_count']
    records, sources, evidence = [], [], {}
    for world in FIT_WORLDS:
        arrays, source = read_episode(native_episode(root, world), world)
        sources.append(source)
        rows = extract_rows(arrays, start=240, stop=300)[0]
        # chain_gaps expects this exact observable dictionary shape.
        rows = chain_gaps(rows, arrays)
        prefix = rows[rows[:, 0]+rows[:, 9] <= 270]
        suffix = rows[rows[:, 0] >= 270]
        drivers = sorted(set(prefix[:, 1].astype(int)) | set(suffix[:, 1].astype(int)))
        posts, ns, raw_posts = {}, {}, {}
        for vid in drivers:
            train = prefix[prefix[:, 1] == vid]
            n = len(train)
            if n:
                prediction = profile_next_speed(train[:, 3, None], train[:, 5, None],
                    train[:, 6, None], support, train[0, 9])
                errors = (prediction-train[:, 4, None])/train[:, 9, None]
                logw = np.log(prior)-np.square(errors).sum(0)/(2*sigma*sigma)
                raw = np.exp(logw-logsumexp(logw))
                fraction = n/(n+strength)
                post = fraction*raw+(1-fraction)*prior
            else:
                raw, post = prior, prior
            posts[vid], raw_posts[vid], ns[vid] = post, raw, n
        evidenced = [vid for vid in drivers if ns[vid]]
        # Each observed driver contributes one expected latent categorical count;
        # uncertainty is retained in its soft posterior, never a truth class.
        counts = sum((posts[vid] for vid in evidenced), np.zeros_like(prior))
        evidence[str(world)] = dict(soft_counts=counts.tolist(), drivers=len(evidenced))
        for vid in np.unique(suffix[:, 1]).astype(int):
            target = suffix[suffix[:, 1] == vid]
            predicted = profile_next_speed(target[:, 3, None], target[:, 5, None],
                target[:, 6, None], support, target[0, 9])
            values = (predicted-target[:, 3, None])/target[:, 9, None]
            actual = (target[:, 4]-target[:, 3])/target[:, 9]
            own = weighted_crps_terms(values, actual, prior, posts[vid])
            other_n = len(evidenced)-int(ns[vid] > 0)
            other_counts = counts-posts[vid] if ns[vid] else counts
            other = other_counts/other_n if other_n else prior
            mix = weighted_crps_terms(values, actual, prior, other)
            records.append(dict(world=world, vid=int(vid), rows=len(target), prefix_rows=ns[vid],
                                own=own.tolist(), mix=mix.tolist(), other_drivers=other_n))
        print(json.dumps(dict(world=world, drivers=len(drivers), target_rows=len(suffix))), flush=True)
    return records, sources, evidence


def fit_parameters(motion, arrivals, excluded=None):
    selected = [r for r in motion if r['world'] != excluded]
    return dict(posterior_retention=quadratic_fit([r['own'] for r in selected]),
        entrant_concentration=concentration_fit([r['mix'] for r in selected],
                                               [r['other_drivers'] for r in selected]),
        arrival_variance_multiplier=arrival_fit(arrivals, excluded))


def objective_summary(motion, arrivals, parameters):
    r = parameters['posterior_retention']
    n = sum(v['rows'] for v in motion)
    cc = np.array([v['own'] for v in motion]).sum(0)
    alpha = parameters['entrant_concentration']
    mc = np.array([v['mix'] for v in motion])
    counts = np.array([v['other_drivers'] for v in motion])
    fraction = np.zeros_like(counts, float) if alpha is None else np.divide(
        counts, counts+alpha, out=np.zeros_like(counts, float), where=counts+alpha > 0)
    count, target = np.array([v['prefix'] for v in arrivals]), np.array([v['target'] for v in arrivals])
    return dict(motion_rows=n, motion_drivers=len(motion),
        population_profile_crps=float(cc[0]/n), baseline_profile_crps=float(cc.sum()/n),
        calibrated_profile_crps=float((cc[0]+r*cc[1]+r*r*cc[2])/n),
        entrant_profile_crps=float(np.sum(mc[:, 0]+fraction*mc[:, 1]+fraction*fraction*mc[:, 2])/n),
        baseline_count_nll=float(count_nll(count, target, 1.).mean()),
        calibrated_count_nll=float(count_nll(count, target, parameters['arrival_variance_multiplier']).mean()))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--population', type=Path, required=True)
    p.add_argument('--shrinkage', type=Path, required=True)
    p.add_argument('--arrivals', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    for path in (args.root, args.population, args.shrinkage, args.arrivals, args.out):
        safe_path(path)
    if not args.out.parent.is_dir() or args.out.exists():
        raise ValueError('exclusive new output directory under existing parent required')
    args.out.mkdir()
    started = time.time()
    pop = json.loads(args.population.read_bytes())
    policy = json.loads(args.shrinkage.read_bytes())
    if policy['idm_population_sha256'] != sha256(args.population):
        raise ValueError('population/shrinkage mismatch')
    artifact = json.loads(args.arrivals.read_bytes())
    if set(artifact['worlds']) != {str(w) for w in FIT_WORLDS}:
        raise ValueError('FIT32 causal arrival artifact only')
    arrivals = []
    for world in FIT_WORLDS:
        obs = artifact['worlds'][str(world)]
        counts = np.array(obs['counts'])
        if counts.shape != (30, 2) or obs['edges_s'] != list(range(0, 301, 10)):
            raise ValueError('arrival bin contract differs')
        for e in range(2):
            arrivals.append(dict(world=world, entrance=e, prefix=int(counts[:15, e].sum()),
                                 target=int(counts[15:, e].sum())))
    motion, sources, evidence = fit_motion(args.root, pop, policy)
    parameters = fit_parameters(motion, arrivals)
    result = dict(schema='wm_phys.spread_calibration.v1', version='rb2-cal-v1',
        base_source='22cc32f65a80af482a1a277c5f22377fa727d0a3', fit_worlds=list(FIT_WORLDS),
        conditioning_end_s=300, parameters=parameters,
        idm_population_sha256=sha256(args.population), shrinkage_sha256=sha256(args.shrinkage),
        arrivals_sha256=sha256(args.arrivals), fitter_sha256=sha256(Path(__file__)),
        qualification='internal physical input calibration; fresh traffic outcome evaluation required',
        population_shared_with_crossfits=True, learned_corrections=False)
    crossfits = []
    for world in FIT_WORLDS:
        fitted = fit_parameters(motion, arrivals, excluded=world)
        crossfits.append(dict(world=world, parameters=fitted,
            scores=objective_summary([r for r in motion if r['world'] == world],
                                    [r for r in arrivals if r['world'] == world], fitted)))
    from tools.wmdecide.report import write_json
    for name, value in [('calibration.json', result), ('motion_objectives.json', motion),
        ('arrival_records.json', arrivals), ('prefix_evidence.json', evidence),
        ('validation.json', dict(full_fit=objective_summary(motion, arrivals, parameters),
             crossfits=crossfits, sources=sources, elapsed_s=time.time()-started))]:
        write_json(args.out/name, value)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
