"""Causal prefix request-count check; reads one existing FIT counter artifact."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
from scipy.stats import nbinom, poisson

from wms.control.fitted_hidden import read_artifact


def forecast(prefix_count, exposure_s, horizon_s, *, family):
    if (prefix_count < 0 or int(prefix_count) != prefix_count
            or not np.isfinite(exposure_s) or exposure_s <= 0
            or not np.isfinite(horizon_s) or horizon_s <= 0):
        raise ValueError('invalid causal count forecast')
    shape = prefix_count + .5
    if family == 'gamma_poisson':
        return nbinom(shape, exposure_s / (exposure_s + horizon_s))
    if family == 'posterior_mean_poisson':
        return poisson(shape * horizon_s / exposure_s)
    if family == 'mle_poisson':
        return poisson(prefix_count * horizon_s / exposure_s)
    raise ValueError('unknown count forecast family')


def discrete_crps(distribution, target):
    # CRPS on unit-spaced integer support via its CDF identity. Squared tail
    # probabilities above this quantile are negligible at FIT count magnitudes.
    end = max(int(target), int(distribution.ppf(1 - 1e-12)))
    n = np.arange(end + 1)
    return float(np.where(n < target, distribution.cdf(n)**2,
                          distribution.sf(n)**2).sum())


def scores(distribution, target):
    lo, hi = [int(v) for v in distribution.ppf([.05, .95])]
    logp = float(distribution.logpmf(target))
    mean, variance = float(distribution.mean()), float(distribution.var())
    return dict(mean=mean, variance=variance,
                nll=-logp if np.isfinite(logp) else None,
                impossible=not np.isfinite(logp), crps=discrete_crps(distribution, target),
                interval90=[lo, hi], covered90=bool(lo <= target <= hi),
                discrete_nominal90=float(distribution.cdf(hi)-distribution.cdf(lo-1)),
                midpoint_pit=float(distribution.cdf(target-1)+.5*distribution.pmf(target)),
                standardized_squared_residual=(target-mean)**2/variance if variance else None)


def analyze(artifact):
    expected = {str(w) for w in range(969400, 969432)}
    if (artifact['schema'] != 'wm_phys.observed_arrivals.v1'
            or artifact['generator_family'] != 'piecewise_constant_poisson'
            or artifact['conditioning_end_s'] != 300
            or set(artifact['worlds']) != expected):
        raise ValueError('only frozen native FIT32 causal counters are permitted')
    families = ('gamma_poisson', 'posterior_mean_poisson', 'mle_poisson')
    records = []
    for world in sorted(expected):
        observation = artifact['worlds'][world]
        counts = np.asarray(observation['counts'], dtype=float)
        if (counts.shape != (30, 2) or not np.isfinite(counts).all()
                or (counts < 0).any() or not np.equal(counts, np.rint(counts)).all()
                or observation['edges_s'] != list(range(0, 301, 10))
                or not np.array_equal(counts.sum(0), observation['total'])):
            raise ValueError('invalid frozen counter bins')
        prefix, target = counts[:15].sum(0).astype(int), counts[15:].sum(0).astype(int)
        for entrance, name in enumerate(('mainline', 'ramp')):
            records.append(dict(world=int(world), entrance=name,
                prefix_count=int(prefix[entrance]), target=int(target[entrance]),
                exposure_s=150, horizon_s=150,
                scores={family:scores(forecast(int(prefix[entrance]), 150., 150., family=family),
                                      int(target[entrance])) for family in families}))
    indices = np.random.default_rng(20261101).integers(0, 32, size=(10000, 32))
    summaries = {}
    for entrance in ('mainline', 'ramp', 'both'):
        selected = [r for r in records if entrance == 'both' or r['entrance'] == entrance]
        summary = {}
        for family in families:
            values = [r['scores'][family] for r in selected]
            numeric = ('mean', 'variance', 'crps', 'covered90', 'discrete_nominal90', 'midpoint_pit')
            summary[family] = {n:float(np.mean([v[n] for v in values])) for n in numeric}
            impossible = sum(v['impossible'] for v in values)
            summary[family]['impossible'] = impossible
            summary[family]['nll'] = ('infinite' if impossible else float(np.mean([v['nll'] for v in values])))
            residuals = [v['standardized_squared_residual'] for v in values]
            summary[family]['standardized_squared_residual'] = (
                float(np.mean(residuals)) if all(v is not None for v in residuals) else None)
        differences = {}
        for metric in ('nll', 'crps'):
            # Both entrances from each world stay together in every resample.
            delta = np.array([r['scores']['gamma_poisson'][metric]
                              - r['scores']['posterior_mean_poisson'][metric] for r in selected])
            delta = delta.reshape(32, -1).mean(1)
            differences[metric] = dict(mean=float(delta.mean()),
                ci95=np.quantile(delta[indices].mean(1), [.025, .975]).tolist())
        summary['gamma_minus_same_mean_poisson'] = differences
        summaries[entrance] = summary
    return dict(schema='rbunc_arrival_prefix/1', worlds=32, entrances=2,
                prefix_s=[0, 150], prediction_s=[150, 300],
                summaries=summaries, records=records,
                qualification='FIT-only causal count diagnostic; not independent qualification or traffic calibration',
                physical_sampling_scale_changed=False)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--artifact', type=Path,
                        default=Path('docs/paper_results/rbunc/observed_arrivals.json'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not args.out.parent.is_dir() or args.out.exists():
        raise ValueError('output must be a NEW file in an existing directory')
    artifact, digest = read_artifact(args.artifact)
    result = analyze(artifact)
    result.update(source_sha256=digest,
                  analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  design_sha256=hashlib.sha256(Path('docs/paper_results/RBUNC_ARRIVAL_PREFIX_DESIGN.md').read_bytes()).hexdigest())
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=1, allow_nan=False)
    print(json.dumps(dict(output=str(args.out), summaries=result['summaries']), indent=1))


if __name__ == '__main__':
    main()
