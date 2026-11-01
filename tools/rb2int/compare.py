"""Six predeclared matched model contrasts; all arms/futures remain world clusters."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from tools.rb2int.predictions import read_model
from tools.rb2int.uncertainty import PairedWorldBootstrap
from tools.rbeval.core import ARMS, HS, crps, digest, load, save_new
from tools.rbint.score_endpoints import world_errors, read_short_baseline

MODELS = ('WM2', 'FULL-v4', 'FULL-v5', 'FULL-v5-LP')
CONTRASTS = (('FULL-v5', 'WM2'), ('FULL-v5', 'FULL-v4'),
             ('FULL-v5-LP', 'WM2'), ('FULL-v5-LP', 'FULL-v4'),
             ('FULL-v5-LP', 'FULL-v5'), ('FULL-v4', 'WM2'))


def features(predictions, real, worlds):
    lookup = {(r['seed'], r['arm']): r for r in real}
    shape = (len(worlds), len(MODELS), len(HS))
    delay, fair, bias = (np.zeros(shape) for _ in range(3))
    # Last model-like column is the recorded rate, repeated across futures only
    # to preserve paired transforms. It is never counted as eight real worlds.
    rate_futures = np.zeros((len(worlds), 8, len(MODELS)+1, len(HS)))
    for i, seed in enumerate(worlds):
        for j, h in enumerate(HS):
            clock = str(h)
            actual_rate = []
            for arm in ARMS:
                target = lookup[seed, arm]
                y = target['clocks'][clock]['team_delay_vs']
                re = target['bins']['upstream']['events'][clock]['event']
                if re is None:
                    raise ValueError('missing factual upstream support')
                actual_rate.append(float(re))
                for m, label in enumerate(MODELS):
                    samples = predictions[label][seed, arm]
                    if len(samples) != 8:
                        raise ValueError('only complete paired K8 forecasts are scoreable')
                    x = [s['clocks'][clock]['team_delay_vs'] for s in samples]
                    delay[i, m, j] += crps(x, y)/len(ARMS)
                    fair[i, m, j] += crps(x, y, fair=True)/len(ARMS)
                    bias[i, m, j] += (np.mean(x)-y)/len(ARMS)
                    events = [s['bins']['upstream']['events'][clock]['event'] for s in samples]
                    if any(e is None for e in events):
                        raise ValueError('missing predicted upstream support')
                    rate_futures[i, :, m, j] += np.asarray(events, float)/len(ARMS)
            rate_futures[i, :, -1, j] = np.mean(actual_rate)
    contrasts = lambda values: np.concatenate([values[:, MODELS.index(a)]-values[:, MODELS.index(b)]
                                              for a, b in CONTRASTS], axis=1)
    return dict(delay=delay, fair=fair, bias=bias, delay_contrasts=contrasts(delay),
                fair_contrasts=contrasts(fair), rate_futures=rate_futures)


def rate_transform(values):
    values = values.reshape(len(values), len(MODELS)+1, len(HS))
    return np.concatenate([np.abs(values[:, MODELS.index(a)]-values[:, -1])-
                           np.abs(values[:, MODELS.index(b)]-values[:, -1])
                           for a, b in CONTRASTS], axis=1)


def label_bands(rows):
    return [dict(candidate=a, baseline=b, horizon_s=h, **row)
            for (a, b, h), row in zip(((a, b, h) for a, b in CONTRASTS for h in HS), rows)]


def switching(delay, rates, endpoint, *, complete):
    failures, missing = [], []
    if len(delay) != 3 or any(r['mean'] is None for r in delay):
        missing.append('three-horizon delay evidence missing')
    elif any(r['mean'] >= 0 for r in delay):
        failures.append('delay CRPS not lower at every horizon')
    rate = next((r for r in rates if r['horizon_s'] == 300), None)
    if rate is None or rate['mean'] is None:
        missing.append('300s breakdown-distance evidence missing')
    elif rate['mean'] >= 0:
        failures.append('300s breakdown rate not closer to recorded')
    if endpoint is None or endpoint['mean'] is None or not endpoint.get('all_worlds_supported', False):
        missing.append('complete common-cohort HDV60 evidence missing')
    elif endpoint['mean'] > 0:
        failures.append('HDV60 position error worse')
    point = 'FAIL' if failures else 'NOT_ESTABLISHED' if missing else 'PASS'
    supported = (complete and point == 'PASS' and all(r['high'] is not None and r['high'] < 0 for r in delay)
                 and rate['high'] is not None and rate['high'] < 0
                 and endpoint['high'] is not None and endpoint['high'] <= 0)
    return dict(point_status=point, point_failures=failures, missing_evidence=missing,
        simultaneous_support='PASS' if supported else 'NOT_ESTABLISHED',
        switching_eligibility='ELIGIBLE' if supported else 'FAIL' if failures else 'NOT_ESTABLISHED',
        complete_registered_scope=complete)


def attach_reserved_wm2_endpoints(predictions, inputs, real):
    short, audit = read_short_baseline([Path(p) for p in inputs], real)
    attached = []
    for (seed, arm), samples in predictions['WM2'].items():
        if all('endpoint60' in s for s in samples):
            continue
        if not 969432 <= seed <= 969439 or (seed, arm) not in short:
            continue
        for full, prefix in zip(samples, short[seed, arm]):
            if full['initial_sha256'] != prefix['initial_sha256']:
                raise ValueError('WM2 prefix start differs')
            for reg in ('upstream', 'merge', 'ramp'):
                old = full['bins'][reg]
                new = prefix['prefix60_bins'][reg]
                if old['post_counts'][:6] != new['post_counts'][:6] or any(
                        (a is None) != (b is None) or a is not None and not np.isclose(a, b, rtol=0, atol=1e-8)
                        for a, b in zip(old['speed'][6:12], new['speed'][:6])):
                    raise ValueError('WM2 short prediction is not prefix-matched')
            full['endpoint60'] = prefix['endpoint60']
        attached.append(dict(seed=seed, arm=arm))
    return dict(attachment='analysis-only; original prediction bytes unchanged', attached=attached, audit=audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True)
    parser.add_argument('--real', required=True)
    parser.add_argument('--factual-endpoints')
    parser.add_argument('--wm2-reserved-endpoints', nargs='+')
    parser.add_argument('--scope', choices=('primary54', 'all58', 'reserved8'), default='primary54')
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    manifest = load(args.runs)
    registration = load(manifest['registration_path'])
    requested = registration[dict(primary54='replicated_primary_worlds',
        all58='final_comparison_worlds', reserved8='reserved8')[args.scope]]
    real = load(args.real)['episodes']
    predictions, audits = {}, {}
    for label in MODELS:
        predictions[label], audits[label] = read_model(args.runs, label, real, requested, partial=args.allow_partial)
    worlds = sorted(set.intersection(*(set(a['complete_worlds']) for a in audits.values())))
    metadata = {r['seed']: dict(seed=r['seed'], demand=r['demand'], penetration=r['penetration']) for r in real}
    boot = PairedWorldBootstrap([metadata[s] for s in worlds])
    values = features(predictions, real, worlds)
    delay = label_bands(boot.bands(values['delay_contrasts']))
    fair = label_bands(boot.bands(values['fair_contrasts']))
    futures = values['rate_futures'].reshape(len(worlds), 8, (len(MODELS)+1)*len(HS))
    rates = label_bands(boot.bands(futures.mean(1), futures=futures, transform=rate_transform))
    prefix = (attach_reserved_wm2_endpoints(predictions, args.wm2_reserved_endpoints, real)
              if args.wm2_reserved_endpoints else None)
    endpoint_rows, endpoint_values = {}, []
    factual = load(args.factual_endpoints) if args.factual_endpoints else None
    if factual is not None and (factual['schema'] != 'wmphys-factual-endpoints/1' or
            factual['horizon_s'] != 60 or factual['t0_s'] != 300):
        raise ValueError('factual endpoint contract differs')
    for a, b in CONTRASTS:
        supported = [s for s in worlds if factual is not None and all(
            'endpoint60' in e for label in (a, b) for arm in ARMS for e in predictions[label][s, arm])]
        rows = world_errors(predictions[a], predictions[b], factual['episodes'], supported) if supported else []
        endpoint_rows[a, b] = rows
        lookup = {r['seed']: r['median_difference_m'] for r in rows}
        endpoint_values.append([lookup.get(s, np.nan) for s in worlds])
    endpoints = []
    for (a, b), band in zip(CONTRASTS, boot.bands(np.asarray(endpoint_values).T)):
        rows = endpoint_rows[a, b]
        counts = Counter()
        for row in rows:
            counts.update(row['counts'])
        endpoints.append(dict(candidate=a, baseline=b, **band, counts=dict(counts), per_world=rows,
            all_worlds_supported=len(rows) == len(worlds) and all(r['median_difference_m'] is not None for r in rows)))
    eligibility = {}
    for label in ('FULL-v5', 'FULL-v5-LP'):
        eligibility[label] = switching([r for r in delay if (r['candidate'], r['baseline']) == (label, 'WM2')],
            [r for r in rates if (r['candidate'], r['baseline']) == (label, 'WM2')],
            next(r for r in endpoints if (r['candidate'], r['baseline']) == (label, 'WM2')),
            complete=worlds == requested)
    out = Path(args.out)
    if not out.parent.is_dir():
        raise ValueError('new receipt in existing parent required')
    report = dict(schema='rb2int-four-model-comparison/1', scope=args.scope,
        requested_worlds=requested, matched_complete_worlds=worlds, complete_registered_scope=worlds == requested,
        stratum_counts=boot.stratum_counts, replicated_scored_strata=boot.replicated,
        delay_crps=delay, fair_crps_sensitivity=fair, upstream_rate_distance=rates, hdv60=endpoints,
        switching=eligibility, input_audits=audits, wm2_prefix_attachment=prefix,
        real_sha256=digest(args.real), factual_endpoints_sha256=digest(args.factual_endpoints) if factual else None,
        approved_run_registry_sha256=digest(args.runs), bootstrap=10000, bootstrap_seed=20261101,
        source_sha256=digest(__file__),
        ci_scope='fixed registered scope' if worlds == requested else 'available matched worlds only; attrition prevents qualification')
    save_new(out, report)
    print(json.dumps(dict(receipt=str(out), scope=args.scope, worlds=len(worlds), switching=eligibility), indent=1))


if __name__ == '__main__':
    main()
