"""Distributional gate and diagnostic report for identity-bound rbeval/1 tasks.

Baselines use only recorded development worlds and leave the entire target
world out. Every uncertainty calculation resamples whole worlds, preserving
all arms/time bins/futures. Sparse cells are never promoted to qualification.
"""
import argparse
from collections import defaultdict
import csv
import itertools
import json
from pathlib import Path

import numpy as np

if __package__:
    from .core import (ARMS, HERE, HS, REGIONS, SEEDS, breakdown, crps, digest, load,
                       onset_vector, safe, save_new, wilson)
else:
    from core import (ARMS, HERE, HS, REGIONS, SEEDS, breakdown, crps, digest, load,
                      onset_vector, safe, save_new, wilson)


def groups(rows, keys):
    out = defaultdict(list)
    for row in rows:
        out[tuple(row[k] for k in keys)].append(row)
    return out


def avg(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.mean(x)) if x else None


def write_csv(out, name, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (out/(name+'.csv')).open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class WorldBootstrap:
    """Stratified whole-world bootstrap + centered standardized max-stat bands.

    The frozen demand×penetration composition is preserved. Singleton cells
    cannot give uncertainty; their estimates remain descriptive. Features are
    compared in joint resamples, rather than bootstrapping arms independently.
    """
    def __init__(self, real, B=10000):
        self.seeds = sorted({e['seed'] for e in real if e['seed'] in SEEDS})
        self.index = {s: i for i, s in enumerate(self.seeds)}
        meta = {e['seed']: (e['demand'], e['penetration'], e['seed'] >= 969432) for e in real}
        self.B = B
        rng = np.random.default_rng(20261031)
        self.weights = np.zeros((B, len(self.seeds)))
        for _, ids in groups([dict(seed=s, cell=meta[s]) for s in self.seeds], ['cell']).items():
            ix = [self.index[x['seed']] for x in ids]
            draws = rng.multinomial(len(ix), np.full(len(ix), 1/len(ix)), size=B)
            self.weights[:, ix] = draws
        # One resampled future-index vector per world, shared across arms and
        # metrics, includes finite-K Monte Carlo uncertainty without breaking pairs.
        self.future_weights = rng.multinomial(8, np.full(8, 1/8), size=(B, len(self.seeds)))/8

    def bands(self, features, alpha=.05, chunk_size=256):
        """features=[[(seed,value),...], ...]; return joint family intervals."""
        if not features:
            return []
        matrix = np.full((len(self.seeds), len(features)), np.nan)
        for j, pairs in enumerate(features):
            by = defaultdict(list)
            for pair in pairs:
                seed, value = pair[:2]
                if value is not None and np.isfinite(value):
                    by[seed].append(value)
            for seed, vals in by.items():
                matrix[self.index[seed], j] = np.mean(vals)
        present = np.isfinite(matrix)
        n = present.sum(0)
        estimate = np.divide(np.nansum(matrix, axis=0), n, out=np.full(len(features), np.nan), where=n > 0)
        # Bound working memory independently of the onset-CDF family's width;
        # every chunk still contributes to ONE joint maximum on the same draws.
        sd = np.zeros(len(features))
        valid = np.zeros(len(features), bool)
        maxstat = np.zeros(self.B)
        for start in range(0, len(features), chunk_size):
            stop = min(start+chunk_size, len(features))
            denom = self.weights @ present[:, start:stop].astype(float)
            draws = np.divide(self.weights @ np.nan_to_num(matrix[:, start:stop]), denom,
                              out=np.full((self.B, stop-start), np.nan), where=denom > 0)
            samples_by_seed = defaultdict(dict)
            for j, pairs in enumerate(features[start:stop]):
                by_seed = defaultdict(list)
                for pair in pairs:
                    if len(pair) == 3:
                        values = np.asarray(pair[2], float)
                        assert values.shape == (8,) and np.isfinite(values).all()
                        by_seed[pair[0]].append(values)
                for seed, values in by_seed.items():
                    values = np.mean(values, axis=0)
                    samples_by_seed[seed][j] = values-values.mean()
            for seed, columns in samples_by_seed.items():
                i = self.index[seed]
                deviations = np.zeros((8, stop-start))
                for j, values in columns.items():
                    deviations[:, j] = values
                delta = self.future_weights[:, i] @ deviations
                draws += np.divide(delta*self.weights[:, i:i+1], denom,
                                   out=np.zeros_like(draws), where=denom > 0)
            sd[start:stop] = np.nanstd(draws, axis=0, ddof=1)
            good = (n[start:stop] >= 2) & (sd[start:stop] > 1e-14) & np.isfinite(draws).all(0)
            valid[start:stop] = good
            if good.any():
                chunk_max = np.max(np.abs((draws[:, good]-estimate[start:stop][good])/sd[start:stop][good]), axis=1)
                maxstat = np.maximum(maxstat, chunk_max)
        critical = None
        if valid.any():
            critical = float(np.quantile(maxstat, 1-alpha))
        result = []
        for j in range(len(features)):
            good = bool(valid[j])
            result.append(dict(mean=float(estimate[j]) if n[j] else None, worlds=int(n[j]),
                low=float(estimate[j]-critical*sd[j]) if good else None,
                high=float(estimate[j]+critical*sd[j]) if good else None,
                uncertainty_available=good, maxstat_critical=critical))
        return result


def read_predictions(inputs, real, partial):
    lookup = {(e['seed'], e['arm']): e for e in real}
    tasks = {}
    identities = []
    model_contract = None
    for root in inputs:
        root = safe(root)
        for path in sorted(root.rglob('w*_k*.json')):
            task = load(path)
            assert task['schema'] == 'rbeval/1'
            identity_path = path.parent/'IDENTITY.json'
            identity = load(identity_path)
            assert task['identity_sha256'] == digest(identity_path)
            contract = {k: identity[k] for k in ('model', 'wm_sha256', 'config_sha256', 'K', 'arms', 'mix', 'native_lanes',
                'sources', 'switches', 'candidate_sha256', 'export_sha256', 'real_receipt_sha256', 'seed', 'update', 't0', 'horizon')}
            contract.update({k: identity[k] for k in ('external_manifest_sha256', 'adapter_sha256',
                                                     'construction_loader_sha256') if k in identity})
            assert contract['K'] == 8 and contract['arms'] == list(ARMS)
            assert contract['real_receipt_sha256'] == digest(HERE/'real_observables.json')
            if model_contract is None:
                model_contract = contract
            assert model_contract == contract, 'mixed model contracts'
            key = (task['seed'], task['future'])
            assert key not in tasks, ('duplicate task (never pool duplicate-device copies)', key)
            assert key[0] in SEEDS and key[1] in range(8)
            assert task['arms'] == list(ARMS) and len(task['episodes']) == 5
            assert {e['arm'] for e in task['episodes']} == set(ARMS)
            for e in task['episodes']:
                assert e['seed'] == key[0] and e['future'] == key[1]
                r = lookup[e['seed'], e['arm']]
                assert e['initial_sha256'] == r['initial_sha256']
                for reg in REGIONS:
                    b = e['bins'][reg]
                    assert len(b['speed']) == 66 and len(b['post_counts']) == 60
                    assert b['speed'][:6] == r['bins'][reg]['speed'][:6]
                    assert all(isinstance(n, int) and n >= 0 and
                        ((v is None) == (n == 0)) and (v is None or np.isfinite(v))
                        for v, n in zip(b['speed'][6:], b['post_counts']))
                    for h in HS:
                        d = breakdown(b['speed'][:6+h//10])
                        stored = b['events'][str(h)]
                        assert (d['event'], d['onset_s'], d.get('preexisting', False)) == (
                            stored['event'], stored['onset_s'], stored.get('preexisting', False))
                for h in HS:
                    c = e['clocks'][str(h)]
                    assert np.isfinite(list(c.values())).all()
                    assert abs(c['team_delay_vs']-c['road_delay_vs']-c['source_wait_vs']) < 1e-4
            tasks[key] = task
            identities.append(dict(task=str(path), sha256=digest(path), identity_sha256=digest(identity_path)))
    expected = {(s, k) for s in SEEDS for k in range(8)}
    missing = sorted(expected-set(tasks))
    if not partial:
        assert not missing, ('incomplete panel', missing)
    # Never score a partly observed K ensemble as K8.
    complete = [s for s in SEEDS if all((s, k) in tasks for k in range(8))]
    pred = {}
    for s in complete:
        for arm in ARMS:
            pred[s, arm] = [next(e for e in tasks[s, k]['episodes'] if e['arm'] == arm) for k in range(8)]
    return pred, dict(model=model_contract, tasks=len(tasks), complete_worlds=complete,
                     missing_tasks=missing, inputs=identities)


def rank_stats(x, y, rng):
    x = np.asarray(x, float)
    less, equal = int((x < y).sum()), int((x == y).sum())
    # Include the factual value in the random tie ordering; then jitter within rank.
    rank = less+int(rng.integers(equal+1))
    return dict(range_coverage=int(x.min() <= y <= x.max()), randomized_range_coverage=int(0 < rank < len(x)), rank=rank,
                randomized_pit=float((rank+rng.random())/(len(x)+1)),
                below_all=int(y < x.min()), above_all=int(y > x.max()))


def eligible_peers(real, e, cell=False):
    """No reserved label and no row from the target world may enter climatology."""
    return [r for r in real if r['seed'] < 969432 and r['seed'] != e['seed'] and
            (not cell or (r['arm'], r['demand'], r['penetration']) ==
             (e['arm'], e['demand'], e['penetration']))]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', nargs='+', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--allow-partial', action='store_true')
    p.add_argument('--bootstrap', type=int, default=10000)
    args = p.parse_args()
    assert args.bootstrap == 10000, 'frozen gate bootstrap count'
    out = safe(args.out)
    assert out.parent.is_dir()
    out.mkdir(exist_ok=False)
    real = load(HERE/'real_observables.json')['episodes']
    pred, audit = read_predictions(args.inputs, real, args.allow_partial)
    save_new(out/'audit.json', audit)
    lookup = {(e['seed'], e['arm']): e for e in real}
    targets = [e for e in real if (e['seed'], e['arm']) in pred]
    bootstrap = WorldBootstrap(real, args.bootstrap)
    rng = np.random.default_rng(20261031)
    rows, events, timing, energy = [], [], [], []
    cdf_features, cdf_meta = [], []

    def peers(e, cell=False):
        return eligible_peers(real, e, cell)

    def distribution(e, h, metric, x, y, pooled, cell, bin_index=None, support=None):
        meta = dict(seed=e['seed'], split='reserved8' if e['seed'] >= 969432 else 'development12',
                    demand=e['demand'], penetration=e['penetration'], arm=e['arm'], horizon_s=h,
                    metric=metric, bin=bin_index)
        good = [v for v in x if v is not None and np.isfinite(v)]
        complete = len(good) == 8 and y is not None
        row = dict(meta, real=y, wm_mean=avg(good), K_valid=len(good),
                   wm_crps=crps(good, y) if complete else None,
                   wm_fair_crps=crps(good, y, True) if complete else None,
                   pooled_crps=crps(pooled, y) if y is not None else None,
                   cell_crps=crps(cell, y) if y is not None else None,
                    cell_baseline_n=len(cell))
        if support is not None:
            real_n, future_n = support
            row.update(real_vehicle_rows=real_n, wm_mean_vehicle_rows=avg(future_n),
                       wm_empty_probability=avg(float(n == 0) for n in future_n))
        if complete:
            row.update(rank_stats(good, y, rng))
            row['error'] = avg(good)-y
            row['within_K_sd'] = float(np.std(good, ddof=1))
        for base in ('pooled', 'cell'):
            row['wm_minus_'+base] = row['wm_crps']-row[base+'_crps'] if row['wm_crps'] is not None and row[base+'_crps'] is not None else None
        rows.append(row)

    for e in targets:
        samples = pred[e['seed'], e['arm']]
        pool, cell = peers(e), peers(e, True)
        for h in HS:
            for metric in ('team_delay_vs', 'road_delay_vs', 'source_wait_vs'):
                distribution(e, h, metric, [s['clocks'][str(h)][metric] for s in samples],
                    e['clocks'][str(h)][metric], [r['clocks'][str(h)][metric] for r in pool],
                    [r['clocks'][str(h)][metric] for r in cell])
            for reg in REGIONS:
                re = e['bins'][reg]['events'][str(h)]
                draws = [s['bins'][reg]['events'][str(h)] for s in samples]
                unknown = sum(d['event'] is None for d in draws)
                meta = dict(seed=e['seed'], split='reserved8' if e['seed'] >= 969432 else 'development12',
                    demand=e['demand'], penetration=e['penetration'], arm=e['arm'], horizon_s=h, region=reg)
                event = dict(meta, real_event=re['event'], real_preexisting=re.get('preexisting', False),
                    real_onset_s=re['onset_s'], wm_event_probability=avg(float(d['event']) for d in draws if d['event'] is not None) if not unknown else None,
                    wm_preexisting_probability=avg(float(d.get('preexisting', False)) for d in draws),
                    wm_unscorable=unknown)
                event['wm_event_draws'] = [d['event'] for d in draws]
                if event['wm_event_probability'] is not None and re['event'] is not None:
                    event['rate_error'] = event['wm_event_probability']-float(re['event'])
                events.append(event)
                rv = onset_vector(re, h)
                vectors = [onset_vector(d, h) for d in draws]
                cv = [onset_vector(r['bins'][reg]['events'][str(h)], h) for r in cell]
                cv = [v for v in cv if v is not None]
                if rv is not None and all(v is not None for v in vectors) and cv:
                    wm, climate = np.mean(vectors, axis=0), np.mean(cv, axis=0)
                    # Integrated Brier over the discrete onset CDF; categorical
                    # pre-existing/censored probabilities retained separately.
                    ibs_wm = float(np.mean((wm[1:-1]-rv[1:-1])**2))
                    ibs_cell = float(np.mean((climate[1:-1]-rv[1:-1])**2))
                    timing.append(dict(meta, wm_ibs=ibs_wm, cell_ibs=ibs_cell, wm_minus_cell=ibs_wm-ibs_cell))
                    for j, (a, b) in enumerate(zip(wm, rv)):
                        cdf_meta.append(dict(meta, category='preexisting' if j == 0 else 'censored' if j == len(wm)-1 else 'cdf',
                                             time_s=(j-1)*10 if 0 < j < len(wm)-1 else None))
                        cdf_features.append((e['seed'], float(a-b), [float(v[j]-b) for v in vectors]))
                        cdf_meta[-1].update(real_cdf=float(b), wm_cdf=float(a))
                speed_vectors = []
                for j in range(h//10):
                    k = 6+j
                    x = [s['bins'][reg]['speed'][k] for s in samples]
                    y = e['bins'][reg]['speed'][k]
                    distribution(e, h, 'speed_'+reg, x, y,
                        [r['bins'][reg]['speed'][k] for r in pool if r['bins'][reg]['speed'][k] is not None],
                        [r['bins'][reg]['speed'][k] for r in cell if r['bins'][reg]['speed'][k] is not None], j,
                        (e['bins'][reg]['post_counts'][j], [s['bins'][reg]['post_counts'][j] for s in samples]))
                    if y is not None and all(v is not None for v in x):
                        speed_vectors.append((x, y))
                if speed_vectors:
                    x = np.array([v[0] for v in speed_vectors]).T
                    y = np.array([v[1] for v in speed_vectors])
                    norm = np.sqrt(x.shape[1])
                    score = np.linalg.norm(x-y, axis=1).mean()/norm-.5*np.linalg.norm(x[:, None]-x, axis=2).mean()/norm
                    energy.append(dict(meta, joint_speed_energy=float(score), supported_bins=len(y), total_bins=h//10))

    write_csv(out, 'distribution_world_bins', rows)
    write_csv(out, 'events_worlds', events)
    write_csv(out, 'timing_worlds', timing)
    write_csv(out, 'energy_diagnostic', energy)
    # Collapse time bins within each world first: a speed field is one world score.
    keys = ['split', 'demand', 'penetration', 'arm', 'horizon_s', 'metric']
    field_rows = []
    score_keys = ('wm_crps', 'wm_fair_crps', 'pooled_crps', 'cell_crps', 'wm_minus_pooled', 'wm_minus_cell',
                  'range_coverage', 'randomized_range_coverage', 'error', 'below_all', 'above_all', 'within_K_sd')
    for group, rs in groups(rows, ['seed']+keys).items():
        field_rows.append(dict(zip(['seed']+keys, group)) | {k: avg(r.get(k) for r in rs) for k in score_keys} |
                          dict(supported_bins=sum(r['wm_crps'] is not None for r in rs), total_bins=len(rs),
                               missing_baseline_bins=sum(r['cell_crps'] is None for r in rs)))
    write_csv(out, 'distribution_worlds', field_rows)
    summary, crps_features, crps_refs = [], [], []
    for level_keys in (['split', 'arm', 'horizon_s', 'metric'], keys):
        for group, rs in groups(field_rows, level_keys).items():
            row = dict(zip(level_keys, group), level='cell' if 'demand' in level_keys else 'arm',
                       worlds=len(rs), supported_bins=sum(r['supported_bins'] for r in rs),
                       total_bins=sum(r['total_bins'] for r in rs))
            row.update({k: avg(r[k] for r in rs) for k in score_keys})
            summary.append(row)
            for base in ('pooled', 'cell'):
                crps_features.append([(r['seed'], r['wm_minus_'+base]) for r in rs])
                crps_refs.append((row, base))
    # Frozen speed+delay family: includes all regions/horizons/cells and diagnostics.
    for band, (row, base) in zip(bootstrap.bands(crps_features), crps_refs):
        row.update({f'wm_minus_{base}_{k}': v for k, v in band.items() if k in ('low', 'high', 'uncertainty_available')})
    write_csv(out, 'distribution_summary', summary)

    rates, rate_features = [], []
    ekeys = ['split', 'demand', 'penetration', 'arm', 'horizon_s', 'region']
    for group, rs in groups(events, ekeys).items():
        valid = [r for r in rs if r['real_event'] is not None]
        n, k = len(valid), sum(bool(r['real_event']) for r in valid)
        point = wilson(k, n)
        simultaneous = wilson(k, n, .05/(180 if rs[0]['region'] == 'upstream' else 360))
        row = dict(zip(ekeys, group), worlds=n, real_rate=k/n if n else None,
            wm_rate=avg(r['wm_event_probability'] for r in rs),
            real_preexisting=avg(float(r['real_preexisting']) for r in rs),
            wm_preexisting=avg(r['wm_preexisting_probability'] for r in rs),
            real_onset_mean_s=avg(r['real_onset_s'] for r in rs if r['real_event']),
            wilson_low=point[0], wilson_high=point[1], simultaneous_low=simultaneous[0], simultaneous_high=simultaneous[1],
            precision_sufficient=bool(n and (simultaneous[1]-simultaneous[0])/2 <= .1))
        row['rate_inside_real_interval'] = bool(row['wm_rate'] is not None and n and simultaneous[0] <= row['wm_rate'] <= simultaneous[1])
        rates.append(row)
        rate_features.append([(r['seed'], r.get('rate_error'),
            [float(v)-float(r['real_event']) for v in r['wm_event_draws']])
            if r['wm_unscorable'] == 0 and r['real_event'] is not None else
            (r['seed'], None) for r in rs])
    for row, band in zip(rates, bootstrap.bands(rate_features)):
        row.update({'paired_rate_'+k: v for k, v in band.items()})
    write_csv(out, 'rates', rates)

    def grouped_bands(records, keys, value):
        grouped = list(groups(records, keys).items())
        bands = bootstrap.bands([[(r['seed'], r[value]) for r in rs] for _, rs in grouped])
        return [dict(zip(keys, key)) | band for (key, rs), band in zip(grouped, bands)]

    timing_summary = grouped_bands(timing, ekeys, 'wm_minus_cell')
    write_csv(out, 'timing_summary', timing_summary)
    cdf_rows = [meta | dict(error=pair[1], seed=pair[0]) for meta, pair in zip(cdf_meta, cdf_features)]
    cdf_keys = ekeys+['category', 'time_s']
    cdf_groups = list(groups(cdf_rows, cdf_keys).items())
    cdf_real_bands = bootstrap.bands([[(r['seed'], r['real_cdf']) for r in rs] for _, rs in cdf_groups])
    onset_bands = []
    for (key, rs), band in zip(cdf_groups, cdf_real_bands):
        wm = avg(r['wm_cdf'] for r in rs)
        onset_bands.append(dict(zip(cdf_keys, key)) | band | dict(wm_cdf=wm,
            wm_inside_real_band=band['low'] <= wm <= band['high'] if band['low'] is not None else None))
    write_csv(out, 'onset_cdf_bands', onset_bands)

    # Expected shifts: world pairing is preserved across all eight sampled futures.
    effects = []
    event_lookup = {(r['seed'], r['arm'], r['horizon_s'], r['region']): r for r in events}
    for e in targets:
        if e['arm'] not in ('mappo', 'pi', 'circles'):
            continue
        for reference_arm in ('native', 'ctrl0'):
            ref = lookup[e['seed'], reference_arm]
            for h in HS:
                for quantity in ('delay', 'event'):
                    if quantity == 'delay':
                        y = e['clocks'][str(h)]['team_delay_vs']-ref['clocks'][str(h)]['team_delay_vs']
                        x = [a['clocks'][str(h)]['team_delay_vs']-b['clocks'][str(h)]['team_delay_vs']
                             for a, b in zip(pred[e['seed'], e['arm']], pred[e['seed'], reference_arm])]
                        wm = avg(x)
                    else:
                        a = event_lookup[e['seed'], e['arm'], h, 'upstream']
                        b = event_lookup[e['seed'], reference_arm, h, 'upstream']
                        if a['real_event'] is None or b['real_event'] is None or a['wm_event_probability'] is None or b['wm_event_probability'] is None:
                            continue
                        y = float(a['real_event'])-float(b['real_event'])
                        wm = a['wm_event_probability']-b['wm_event_probability']
                    effects.append(dict(seed=e['seed'], split='reserved8' if e['seed'] >= 969432 else 'development12',
                        demand=e['demand'], penetration=e['penetration'], arm=e['arm'], reference=reference_arm,
                        horizon_s=h, quantity=quantity, real_effect=y, wm_effect=wm, bias=wm-y))
    effect_keys = ['split', 'demand', 'penetration', 'arm', 'reference', 'horizon_s', 'quantity']
    effect_summary = []
    effect_groups = list(groups(effects, effect_keys).items())
    real_bands = bootstrap.bands([[(r['seed'], r['real_effect']) for r in rs] for _, rs in effect_groups])
    bias_bands = bootstrap.bands([[(r['seed'], r['bias']) for r in rs] for _, rs in effect_groups])
    for (group, rs), rb, bb in zip(effect_groups, real_bands, bias_bands):
        wm = avg(r['wm_effect'] for r in rs)
        resolved = rb['low'] is not None and (rb['low'] > 0 or rb['high'] < 0)
        effect_summary.append(dict(zip(effect_keys, group)) | dict(worlds=len(rs), real_shift=rb['mean'], wm_shift=wm,
            real_low=rb['low'], real_high=rb['high'], bias_low=bb['low'], bias_high=bb['high'],
            predicted_inside_real_ci=rb['low'] <= wm <= rb['high'] if rb['low'] is not None else None,
            real_sign_resolved=resolved, resolved_sign_agrees=bool(np.sign(wm) == np.sign(rb['mean'])) if resolved else None))
    write_csv(out, 'expected_shifts', effect_summary)
    write_csv(out, 'expected_shifts_worlds', effects)
    order = []
    # Bootstrap group-mean rankings (not individual-world effect signs).
    for key, rs in groups(effects, ['split', 'demand', 'penetration', 'reference', 'horizon_s', 'quantity']).items():
        arms = groups(rs, ['arm'])
        worlds = sorted({r['seed'] for r in rs})
        pairs = list(itertools.combinations(sorted(arms), 2))
        def concordance(weights):
            hits = []
            for a, b in pairs:
                def weighted(arm, col):
                    vals = arms[arm]
                    ws = np.array([weights[bootstrap.index[r['seed']]] for r in vals])
                    return np.dot(ws, [r[col] for r in vals])/ws.sum() if ws.sum() else np.nan
                dy = weighted(a, 'real_effect')-weighted(b, 'real_effect')
                dx = weighted(a, 'wm_effect')-weighted(b, 'wm_effect')
                if np.isfinite(dy) and dy != 0:
                    hits.append(float(np.sign(dx) == np.sign(dy)))
            return avg(hits)
        estimate = concordance(np.ones(len(bootstrap.seeds)))
        # Bonferroni conservative simultaneous bound over frozen 12×2×3×2 groups.
        draws = [concordance(w) for w in bootstrap.weights] if len(worlds) >= 2 else []
        good = [v for v in draws if v is not None]
        lower = float(np.quantile(good, .05/144)) if len(good) == bootstrap.B else None
        order.append(dict(zip(['split', 'demand', 'penetration', 'reference', 'horizon_s', 'quantity'], key)) |
            dict(worlds=len(worlds), concordance=estimate, simultaneous_lower=lower))
    write_csv(out, 'expected_order', order)

    # Distribution-free simultaneous exchangeability bands: Hoeffding on bounded
    # whole-world rank-bin/coverage averages. Valid with arbitrary within-world
    # arm/time dependence; do not assume IID future/arm/time rows. Wide is honest.
    calibration = []
    for key, rs in groups(rows, ['split', 'metric', 'horizon_s']).items():
        rs = [r for r in rs if 'rank' in r]
        world = groups(rs, ['seed'])
        if not world:
            continue
        weights = np.array([len(v) for v in world.values()], float)
        # 2 splits × 6 fields × 3 horizons × (9 rank bins + coverage).
        radius = float(np.sqrt(np.log(2*360/.05)*np.sum(weights**2)/(2*weights.sum()**2)))
        lo, hi = max(0., 7/9-radius), min(1., 7/9+radius)
        observed = avg(r['randomized_range_coverage'] for r in rs)
        for rank in range(9):
            pl, ph = max(0., 1/9-radius), min(1., 1/9+radius)
            fraction = sum(r['rank'] == rank for r in rs)/len(rs)
            calibration.append(dict(zip(['split', 'metric', 'horizon_s'], key)) |
                dict(rank_bin=rank, rank_fraction=fraction, pit_simultaneous_low=float(pl), pit_simultaneous_high=float(ph),
                     randomized_range_coverage=observed, inclusive_range_coverage=avg(r['range_coverage'] for r in rs), exchangeable_range_expected=7/9,
                     range_simultaneous_low=float(lo), range_simultaneous_high=float(hi),
                     worlds=len(world), exchangeability_band_method='whole-world Hoeffding union bound; arbitrary within-world dependence'))
    write_csv(out, 'calibration', calibration)
    failures = []
    if audit['missing_tasks']:
        failures.append('incomplete K8 panel')
    if len({(e['demand'], e['penetration']) for e in targets}) < 12:
        failures.append('missing registered demand/penetration cells')
    if any(not r['precision_sufficient'] for r in rates if r['region'] == 'upstream') or not rates:
        failures.append('simultaneous upstream rate precision insufficient')
    if any(not r['rate_inside_real_interval'] for r in rates if r['region'] == 'upstream'):
        failures.append('upstream point rate outside simultaneous real intervals')
    primary = [r for r in summary if r['level'] == 'cell' and r['metric'] in ('team_delay_vs', *['speed_'+s for s in REGIONS])]
    if not primary or any(r.get('wm_minus_'+b+'_high') is None or r['wm_minus_'+b+'_high'] >= 0 for r in primary for b in ('pooled', 'cell')):
        failures.append('cell-level delay/speed CRPS superiority unproven or failed')
    if any(r['supported_bins'] != r['total_bins'] for r in primary):
        failures.append('speed support/missingness prevents complete field scoring')
    if not timing_summary or any(r['high'] is None or r['high'] >= 0 for r in timing_summary):
        failures.append('timing IBS superiority unproven or failed')
    if not onset_bands or any(r['wm_inside_real_band'] is not True for r in onset_bands):
        failures.append('onset CDF simultaneous real bands insufficient or violated')
    if any(r['paired_rate_low'] is None or not (r['paired_rate_low'] <= 0 <= r['paired_rate_high'])
           for r in rates if r['region'] == 'upstream'):
        failures.append('paired upstream WM-real rate compatibility unproven or failed (includes K8 MC)')
    if not order or any(r['simultaneous_lower'] is None or r['simultaneous_lower'] <= .5 for r in order):
        failures.append('expected-shift arm-order evidence insufficient or failed')
    if not effect_summary or any(r['predicted_inside_real_ci'] is not True or r['resolved_sign_agrees'] is False for r in effect_summary):
        failures.append('expected shifts unproven or outside paired real intervals')
    if any(not (r['pit_simultaneous_low'] <= r['rank_fraction'] <= r['pit_simultaneous_high']) or not
           (r['range_simultaneous_low'] <= r['randomized_range_coverage'] <= r['range_simultaneous_high']) for r in calibration):
        failures.append('PIT/range outside simultaneous world-cluster exchangeability bands')
    verdict = dict(schema='rbeval-gate/1', verdict='NOT_QUALIFIED',
        pilot_only=True, reasons=failures, complete_worlds=len(audit['complete_worlds']),
        bootstrap=10000, bootstrap_seed=20261031,
        bootstrap_method='demand/penetration/split-stratified whole-world centered standardized max-stat',
        rate_family_primary=180, rate_family_diagnostic=360,
        scope='Fixed overnight diagnostic, not a powered qualification panel',
        caveat='One factual continuation per world cannot identify same-state irreducible randomness')
    save_new(out/'gate.json', verdict)
    print(json.dumps(verdict, indent=1))


if __name__ == '__main__':
    main()
