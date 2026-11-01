"""Cached-only S-BDRANK: observable breakdown cost, ranking, and fixed decisions."""
from collections import Counter, defaultdict
import csv
import hashlib
import io
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent
OUT = ROOT/'docs/paper_results/bdrank'
BD = BASE/'c4-wmbreakdown/docs/paper_results/wmbreakdown'
V4 = BASE/'c4-rb-int/docs/rbint/noon/20261101T093000+0800/FULL-v4/distribution'
TEACH = BASE/'c4-wmteach/docs/paper_results/wmteach'
RB2 = BASE/'c4-rb2-int/docs/rb2int'
BAD = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1',
       'hetdata_truth', 'vehicle_params', 'driver_params', 'vtype')
ARMS = ('native', 'ctrl0', 'mappo', 'pi', 'circles')
PREF = ('mappo', 'native', 'ctrl0', 'pi', 'circles')
HS = (120, 300, 600)
REGIONS = ('upstream', 'merge', 'ramp')
INPUTS = {}
CHECKS = Counter()
WORLD_META = {}


def safe(p):
    p = Path(p)
    assert not any(s in str(p).lower() for s in BAD), p
    return p


def read(p):
    p = safe(p)
    b = p.read_bytes()
    INPUTS[str(p)] = hashlib.sha256(b).hexdigest()
    return b


def load(p):
    return json.loads(read(p))


def csvread(p):
    return list(csv.DictReader(io.StringIO(read(p).decode('utf-8-sig'))))


def emit(name, rows):
    assert rows, name
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with safe(OUT/(name+'.csv')).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows({k: ('NA' if v is None else json.dumps(v) if isinstance(v, (list, dict)) else v)
                    for k, v in r.items()} for r in rows)


def save(name, obj):
    safe(OUT/name).write_text(json.dumps(obj, indent=1, allow_nan=False)+'\n', encoding='utf-8')


def groups(rows, keys):
    result = defaultdict(list)
    for r in rows:
        result[tuple(r[k] for k in keys)].append(r)
    return result


def avg(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def correlation(x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def slope(x, y):
    if len(x) < 2 or np.var(x) == 0:
        return None
    return float(np.mean((np.array(x)-np.mean(x))*(np.array(y)-np.mean(y)))/np.var(x))


def interval(rows, numerator, denominator=None, frame=None):
    """Ratio of totals, clustered by world, retaining every scored row per world."""
    by = groups(rows, ['seed'])
    ids = sorted({r['seed'] for r in (rows if frame is None else frame)})
    cells = defaultdict(list)
    for i, s in enumerate(ids):
        cells[WORLD_META[s]].append(i)
    if not ids:
        return dict(ci_low=None, ci_high=None, ci_status='no_scored_worlds', bootstrap_empty=10000)
    contributing = Counter(WORLD_META[s] for s in ids if by[s,])
    if any(n < 2 for n in contributing.values()) or any(len(v) < 2 for v in cells.values()):
        return dict(ci_low=None, ci_high=None, ci_status='NA_singleton_stratum', bootstrap_empty=None)
    num = np.array([sum(float(r[numerator]) for r in by[s,]) for s in ids])
    den = np.array([sum(float(r[denominator]) for r in by[s,]) if denominator else len(by[s,]) for s in ids])
    rng = np.random.default_rng(20261101)
    a = np.zeros(10000)
    b = np.zeros(10000)
    for ix in cells.values():
        sampled = rng.choice(ix, size=(10000, len(ix)), replace=True)
        a += num[sampled].sum(1)
        b += den[sampled].sum(1)
    missing = int((b == 0).sum())
    if missing:
        return dict(ci_low=None, ci_high=None, ci_status='NA_empty_informative_draws', bootstrap_empty=missing)
    lo, hi = np.quantile(a/b, [.025, .975])
    return dict(ci_low=float(lo), ci_high=float(hi), ci_status='fixed_cell_world_percentile', bootstrap_empty=0)


def detector():
    ref = load(ROOT/'tools/rbeval/references.json')['breakdown']
    assert hashlib.sha256(ref['code'].encode()).hexdigest() == ref['code_sha256']
    assert ref['code_sha256'] == '6de4df039e108508cf71f94abfe2e81136d5a9a502bb4803980411c0fa5b3f10'
    ns = {'np': np}
    exec(compile(ref['code'], '<registered-LRDIAG-only>', 'exec'), ns)
    return ns['breakdown'], ref['code_sha256']


def event_metrics(d, h):
    assert d['event'] is not None
    e, pre = bool(d['event']), bool(d.get('preexisting', False))
    assert not (e and pre)
    depth = -float(d['drop_mps']) if e else None
    return dict(occurrence=float(e), preexisting=float(pre),
                onset_conditional=float(d['onset_s']) if e else None,
                depth_conditional=depth, depth_expected=depth if e else 0.,
                onset_rmst=0. if pre else float(d['onset_s']) if e else float(h))


def verify_event(d, fresh):
    for key in ('event', 'onset_s', 'preexisting'):
        assert d.get(key, False if key == 'preexisting' else None) == fresh.get(key, False if key == 'preexisting' else None)
    for key in ('drop_mps', 'bic_improvement'):
        if d.get(key) is not None:
            assert abs(d[key]-fresh[key]) < 1e-8
    CHECKS['independent_detector_decisions'] += 1


def real_data(fn):
    episodes = []
    for name in ('wmdata_observables.json', 'wmdata2_observables.json'):
        episodes += load(BD/name)['episodes']
    assert len(episodes) == 500
    assert len({(e['seed'], e['arm']) for e in episodes}) == 500
    rows = []
    for e in episodes:
        s = e['seed']
        assert 969400 <= s <= 969439 or 969550 <= s <= 969599
        WORLD_META[s] = (e['demand'], e['penetration'], e['split'])
        meta = {k: e[k] for k in ('corpus', 'split', 'seed', 'arm', 'demand', 'penetration')}
        for reg in REGIONS:
            for h in HS:
                d = e['bins'][reg]['events'][str(h)]
                verify_event(d, fn(e['bins'][reg]['speed'][:6+h//10]))
                c = e['clocks'][str(h)]
                assert abs(c['team_delay_vs']-c['road_delay_vs']-c['source_wait_vs']) < 1e-4
                CHECKS['real_cost_additivity'] += 1
                rows.append(meta | dict(region=reg, horizon_s=h, status='preexisting' if d.get('preexisting') else 'event' if d['event'] else 'no_event',
                    onset_s=d['onset_s'], depth_mps=-d['drop_mps'] if d['event'] else None,
                    duration_s=None, queue_extent_m=None, **event_metrics(d, h),
                    **{k: c[k] for k in ('team_delay_vs', 'road_delay_vs', 'source_wait_vs')},
                    J120=e['clocks']['120']['team_delay_vs'], J300=e['clocks']['300']['team_delay_vs'], J600=e['clocks']['600']['team_delay_vs']))
    emit('real_episodes', rows)
    return episodes, rows


def cost_analysis(rows):
    base = ['corpus', 'split', 'region', 'horizon_s']
    summaries, association, onset_summaries, contrast = [], [], [], []
    for extra in ([], ['demand', 'penetration'], ['arm'], ['demand', 'penetration', 'arm']):
        keys = base+extra
        for vals, rs in groups(rows, keys).items():
            meta = dict(zip(keys, vals)) | dict(level='+'.join(extra) or 'pooled')
            for status, ss in groups(rs, ['status']).items():
                summaries.append(meta | dict(status=status[0], episodes=len(ss), worlds=len({r['seed'] for r in ss}),
                    mean_J=avg(r['team_delay_vs'] for r in ss),
                    mean_road=avg(r['road_delay_vs'] for r in ss), mean_source=avg(r['source_wait_vs'] for r in ss),
                    mean_onset_s=avg(r['onset_conditional'] for r in ss), mean_depth_mps=avg(r['depth_mps'] for r in ss),
                    **interval(ss, 'team_delay_vs', frame=rs)))
            ev = [r for r in rs if r['status'] == 'event']
            no = [r for r in rs if r['status'] == 'no_event']
            for field in ('onset_s', 'depth_mps'):
                x, y = [r[field] for r in ev], [r['team_delay_vs'] for r in ev]
                association.append(meta | dict(signal=field, event_n=len(ev), no_event_n=len(no),
                    event_minus_noevent_J=avg(y)-avg(r['team_delay_vs'] for r in no) if ev and no else None,
                    correlation=correlation(x, y), slope_veh_s_per_unit=slope(x, y)))
            if extra == ['demand', 'penetration']:
                for t, ss in groups(ev, ['onset_s']).items():
                    onset_summaries.append(meta | dict(onset_s=t[0], n=len(ss), mean_J=avg(r['team_delay_vs'] for r in ss),
                        mean_depth_mps=avg(r['depth_mps'] for r in ss)))
            # Cell x controller standardization avoids compositional event/no-event contrasts.
            supported = []
            for _, cell in groups(rs, ['demand', 'penetration', 'arm']).items():
                a = [r for r in cell if r['status'] == 'event']
                b = [r for r in cell if r['status'] == 'no_event']
                if a and b:
                    supported.append((len(a)+len(b), avg(r['team_delay_vs'] for r in a)-avg(r['team_delay_vs'] for r in b)))
            contrast.append(meta | dict(supported_cell_arm_groups=len(supported), supported_episodes=sum(n for n, _ in supported),
                standardized_event_minus_noevent_J=sum(n*x for n, x in supported)/sum(n for n, _ in supported) if supported else None,
                interpretation='descriptive association; not causal breakdown cost'))
    emit('cost_by_status', summaries)
    emit('cost_associations', association)
    emit('cost_by_exact_onset', onset_summaries)
    emit('cost_standardized_contrasts', contrast)

    pairs = []
    for _, rs in groups(rows, ['seed', 'region', 'horizon_s']).items():
        for a, b in itertools.combinations(sorted(rs, key=lambda r: r['arm']), 2):
            excluded = a['status'] == 'preexisting' or b['status'] == 'preexisting'
            pairs.append({k: a[k] for k in ('corpus', 'split', 'seed', 'demand', 'penetration', 'region', 'horizon_s')} |
                dict(arm_a=a['arm'], arm_b=b['arm'], excluded_preexisting=excluded,
                    delta_J=a['team_delay_vs']-b['team_delay_vs'],
                    occurrence=a['occurrence']-b['occurrence'],
                    timing=((a['onset_conditional'] or 0)-(b['onset_conditional'] or 0))/a['horizon_s'],
                    severity=a['depth_expected']-b['depth_expected']))
    emit('cost_controller_pairs', pairs)
    fits = []
    for extra in ([], ['demand', 'penetration']):
        keys = base+extra
        for vals, allpairs in groups(pairs, keys).items():
            rs = [r for r in allpairs if not r['excluded_preexisting']]
            meta = dict(zip(keys, vals)) | dict(level='cell' if extra else 'pooled', pairs=len(rs), excluded_pairs=len(allpairs)-len(rs),
                worlds=len({r['seed'] for r in rs}))
            if not rs:
                fits.append(meta | dict(status='no_nonpreexisting_pairs'))
                continue
            X = np.array([[r[k] for k in ('occurrence', 'timing', 'severity')] for r in rs])
            y = np.array([r['delta_J'] for r in rs])
            total = float(y@y)
            r2 = {0: 0.}
            for mask in range(1, 8):
                xx = X[:, [j for j in range(3) if mask & (1 << j)]]
                pred = xx@np.linalg.lstsq(xx, y, rcond=None)[0]
                r2[mask] = 1-float((y-pred)@(y-pred))/total if total else 0.
            shap = np.zeros(3)
            for perm in itertools.permutations(range(3)):
                mask = 0
                for j in perm:
                    new = mask | (1 << j)
                    shap[j] += (r2[new]-r2[mask])/6
                    mask = new
            beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
            cv_sse = 0.
            for s in {r['seed'] for r in rs}:
                test = np.array([r['seed'] == s for r in rs])
                b = np.linalg.lstsq(X[~test], y[~test], rcond=None)[0] if (~test).any() else np.zeros(3)
                residual = y[test]-X[test]@b
                cv_sse += float(residual@residual)
            fits.append(meta | dict(status='descriptive_pair_regression', design_rank=int(rank),
                paired_squared_delay_sum=total, R2_occurrence=r2[1], R2_occurrence_timing=r2[3], R2_full=r2[7],
                shapley_occurrence=float(shap[0]), shapley_timing=float(shap[1]), shapley_severity=float(shap[2]),
                residual_fraction=1-r2[7], leave_world_out_R2=1-cv_sse/total if total else None,
                beta_occurrence_veh_s=float(beta[0]), beta_timing_per_horizon_veh_s=float(beta[1]),
                beta_depth_veh_s_per_mps=float(beta[2]), ci_status='not_claimed_descriptive_regression'))
    emit('cost_variance_decomposition', fits)
    # Confirm controller occurrence shifts against both historical references.
    shifts = []
    for _, rs in groups(rows, ['corpus', 'split', 'region', 'horizon_s', 'seed']).items():
        by = {r['arm']: r for r in rs}
        for ref in ('native', 'ctrl0'):
            for arm, r in by.items():
                if arm == ref:
                    continue
                b = by[ref]
                shifts.append({k: r[k] for k in base+['seed', 'demand', 'penetration']} |
                    dict(reference=ref, arm=arm, delta_event=r['occurrence']-b['occurrence'], delta_J=r['team_delay_vs']-b['team_delay_vs']))
    summary = []
    for extra in ([], ['demand', 'penetration']):
        keys = base+['reference', 'arm']+extra
        for vals, rs in groups(shifts, keys).items():
            ci = interval(rs, 'delta_event')
            ci = {k: 100*v if k in ('ci_low', 'ci_high') and v is not None else v for k, v in ci.items()}
            summary.append(dict(zip(keys, vals)) | dict(level='cell' if extra else 'pooled', worlds=len(rs),
                probability_shift_pp=100*avg(r['delta_event'] for r in rs), delta_J=avg(r['delta_J'] for r in rs),
                **ci))
    emit('occurrence_controller_shifts', summary)
    return fits, association, contrast


def predictions(fn, real):
    audit = load(V4/'audit.json')
    assert audit['tasks'] == 160 and audit['complete_worlds'] == list(range(969420, 969440))
    factual = {(e['seed'], e['arm']): e for e in real}
    drawrows, cache, manifest, seconds = [], {}, [], []
    for entry in audit['inputs']:
        p = Path(entry['task'])
        task = load(p)
        assert INPUTS[str(p)] == entry['sha256']
        ident = load(p.parent/'IDENTITY.json')
        assert INPUTS[str(p.parent/'IDENTITY.json')] == entry['identity_sha256'] == task['identity_sha256']
        assert ident['model'] == 'WM-PHYS-FULL-v4-gpu' and ident['K'] == 8 and ident['arms'] == list(ARMS)
        assert any('phys_code_22cc32f65a80' in k for k in ident['sources'])
        key = task['seed'], task['future']
        assert key not in cache
        cache[key] = task
        seconds.append(task['seconds'])
        manifest.append(dict(path=str(p), sha256=entry['sha256'], seed=task['seed'], future=task['future'], identity_sha256=entry['identity_sha256']))
        for e in task['episodes']:
            r = factual[e['seed'], e['arm']]
            assert e['initial_sha256'] == r['initial_sha256']
            for reg in REGIONS:
                assert e['bins'][reg]['speed'][:6] == r['bins'][reg]['speed'][:6]
                for h in HS:
                    d = e['bins'][reg]['events'][str(h)]
                    verify_event(d, fn(e['bins'][reg]['speed'][:6+h//10]))
                    c = e['clocks'][str(h)]
                    assert abs(c['team_delay_vs']-c['road_delay_vs']-c['source_wait_vs']) < 1e-4
                    CHECKS['forecast_cost_additivity'] += 1
                    drawrows.append(dict(model='FULL-v4', seed=e['seed'], future=e['future'], arm=e['arm'],
                        split='development12' if e['seed'] < 969432 else 'reserved8', demand=r['demand'], penetration=r['penetration'],
                        region=reg, horizon_s=h, mean_delay=c['team_delay_vs'], **event_metrics(d, h)))
    assert set(cache) == {(s, k) for s in range(969420, 969440) for k in range(8)}
    emit('controller_future_draws', drawrows)
    rows = []
    for _, ds in groups(drawrows, ['seed', 'arm', 'region', 'horizon_s']).items():
        a = ds[0]
        assert len(ds) == 8 and {d['future'] for d in ds} == set(range(8))
        h, reg = a['horizon_s'], a['region']
        r = factual[a['seed'], a['arm']]
        target = event_metrics(r['bins'][reg]['events'][str(h)], h) | dict(mean_delay=r['clocks'][str(h)]['team_delay_vs'])
        row = {k: a[k] for k in ('model', 'seed', 'split', 'demand', 'penetration', 'arm', 'region', 'horizon_s')}
        for signal in ('occurrence', 'preexisting', 'onset_conditional', 'depth_conditional', 'onset_rmst', 'depth_expected', 'mean_delay'):
            row['pred_'+signal] = avg(d[signal] for d in ds)
            row['real_'+signal] = target[signal]
        row['event_futures'] = int(sum(d['occurrence'] for d in ds))
        rows.append(row)
    emit('controller_predictions', rows)
    save('controller_cache_manifest.json', dict(model='FULL-v4', source='22cc32f65a80af482a1a277c5f22377fa727d0a3',
        tasks=manifest, mean_task_seconds=avg(seconds), total_recorded_worker_gpu_minutes=sum(seconds)/60))
    return rows


def sign(x, signal):
    return 0 if abs(x) <= (1e-8 if 'delay' in signal or 'depth' in signal else 0) else 1 if x > 0 else -1


def pair_record(meta, a, b, signal, target):
    px, py = a.get('pred_'+signal), b.get('pred_'+signal)
    rx, ry = a.get('real_'+target), b.get('real_'+target)
    available = all(v is not None for v in (px, py, rx, ry))
    ps, rs = (sign(px-py, signal), sign(rx-ry, target)) if available else (None, None)
    # Onset scores have opposite preference direction from cost/probability/depth.
    if available and signal.startswith('onset') != target.startswith('onset'):
        ps = -ps
    both_distinct = available and ps != 0 and rs != 0
    factual_distinct = available and rs != 0
    category = ('missing' if not available else 'both_tie' if ps == rs == 0 else 'predicted_only_tie' if ps == 0 else
                'real_only_tie' if rs == 0 else 'concordant' if ps == rs else 'discordant')
    return meta | dict(item_a=a['arm'], item_b=b['arm'], signal=signal, target=target,
        predicted_a=px, predicted_b=py, real_a=rx, real_b=ry, category=category,
        comparable=int(both_distinct), concordant=int(category == 'concordant'),
        factual_distinct=int(factual_distinct), tie_adjusted_numerator=float(category == 'concordant')+.5*float(category == 'predicted_only_tie'))


def ranking(rows, prefix):
    pairrows = []
    keys = ['model', 'split', 'seed', 'region', 'horizon_s']
    for _, rs in groups(rows, keys).items():
        for a, b in itertools.combinations(sorted(rs, key=lambda r: r['arm']), 2):
            meta = {k: a[k] for k in keys+['demand', 'penetration']}
            for signal in ('occurrence', 'onset_rmst', 'depth_expected', 'mean_delay', 'onset_conditional', 'depth_conditional'):
                for target in dict.fromkeys((signal, 'mean_delay')):
                    pairrows.append(pair_record(meta, a, b, signal, target))
    emit(prefix+'_pairs', pairrows)
    summary = []
    base = ['model', 'split', 'region', 'horizon_s', 'signal', 'target']
    for extra in ([], ['demand', 'penetration'], ['item_a', 'item_b'], ['seed']):
        for vals, rs in groups(pairrows, base+extra).items():
            c = Counter(r['category'] for r in rs)
            n, den = sum(r['concordant'] for r in rs), sum(r['comparable'] for r in rs)
            fden = sum(r['factual_distinct'] for r in rs)
            summary.append(dict(zip(base+extra, vals)) | dict(level='+'.join(extra) or 'pooled', pairs=len(rs),
                worlds=len({r['seed'] for r in rs}), concordant=c['concordant'], discordant=c['discordant'],
                predicted_only_ties=c['predicted_only_tie'], real_only_ties=c['real_only_tie'], both_ties=c['both_tie'], missing=c['missing'],
                concordance=n/den if den else None, tie_adjusted_concordance=sum(r['tie_adjusted_numerator'] for r in rs)/fden if fden else None,
                **interval(rs, 'concordant', 'comparable'),
                **{'tie_adjusted_'+k: v for k, v in interval(rs, 'tie_adjusted_numerator', 'factual_distinct').items()}))
    emit(prefix+'_concordance', summary)
    return summary


RULES = dict(probability=('pred_occurrence',), onset=('negative_rmst',), severity=('pred_depth_expected',),
             delay=('pred_mean_delay',), breakdown_then_delay=('pred_occurrence', 'pred_mean_delay'))


def decision_rows(rows):
    results = []
    for _, rs in groups([r for r in rows if r['region'] == 'upstream'], ['model', 'split', 'seed', 'horizon_s']).items():
        by = {r['arm']: r for r in rs}
        assert set(by) == set(ARMS)
        oracle = min(r['real_mean_delay'] for r in rs)
        baseline = by['mappo']['real_mean_delay']
        for rule, fields in RULES.items():
            def key(r):
                return tuple(-r['pred_onset_rmst'] if k == 'negative_rmst' else r[k] for k in fields)
            best = min(key(r) for r in rs)
            tied = [a for a in PREF if key(by[a]) == best]
            chosen = tied[0]
            r = by[chosen]
            results.append({k: r[k] for k in ('model', 'split', 'seed', 'demand', 'penetration', 'horizon_s')} |
                dict(rule=rule, chosen=chosen, tied_actions=tied, tie_size=len(tied), selected_J=r['real_mean_delay'],
                    MAPPO_J=baseline, oracle_J=oracle, saving=baseline-r['real_mean_delay'], regret=r['real_mean_delay']-oracle,
                    oracle_saving=baseline-oracle,
                    min_saving_among_prediction_ties=min(baseline-by[a]['real_mean_delay'] for a in tied),
                    max_saving_among_prediction_ties=max(baseline-by[a]['real_mean_delay'] for a in tied)))
    return results


def decision_summary(rows):
    summary = []
    for extra in ([], ['demand', 'penetration']):
        keys = ['model', 'split', 'horizon_s', 'rule']+extra
        for vals, rs in groups(rows, keys).items():
            meta = dict(zip(keys, vals)) | dict(level='cell' if extra else 'pooled', worlds=len(rs),
                tie_worlds=sum(r['tie_size'] > 1 for r in rs), choices=dict(Counter(r['chosen'] for r in rs)))
            for metric in ('saving', 'regret', 'oracle_saving', 'min_saving_among_prediction_ties', 'max_saving_among_prediction_ties'):
                summary.append(meta | dict(metric=metric, mean=avg(r[metric] for r in rs), **interval(rs, metric)))
    return summary


def decisions(rows):
    # Physically persist development-only selection before even constructing reserved decisions.
    dev = decision_rows([r for r in rows if r['split'] == 'development12'])
    devsummary = decision_summary(dev)
    selected = {}
    for h in HS:
        means = {rule: avg(r['saving'] for r in dev if r['horizon_s'] == h and r['rule'] == rule) for rule in RULES}
        selected[str(h)] = dict(rule=max(RULES, key=lambda k: means[k]), development_mean_savings=means,
            basis='maximum development12 mean saving only; fixed rule order breaks equal development scores')
    save('development_selection.json', dict(source='FULL-v4/22cc32f6', horizon_selection=selected,
        development_worlds=list(range(969420, 969432)), reserved_labels_used=False))
    confirmation = decision_rows([r for r in rows if r['split'] == 'reserved8'])
    allrows = dev+confirmation
    for r in allrows:
        r['selected_on_development'] = r['rule'] == selected[str(r['horizon_s'])]['rule']
    emit('controller_decisions', allrows)
    summary = devsummary+decision_summary(confirmation)
    emit('controller_decision_summary', summary)
    paired = []
    for _, rs in groups(allrows, ['seed', 'horizon_s']).items():
        incumbent = next(r for r in rs if r['rule'] == 'delay')
        for r in rs:
            paired.append({k: r[k] for k in ('model', 'split', 'seed', 'demand', 'penetration', 'horizon_s', 'rule')} |
                          dict(extra_saving_vs_delay=r['saving']-incumbent['saving'], changed_choice=r['chosen'] != incumbent['chosen']))
    emit('controller_rule_contrasts', paired)
    contrast = []
    for vals, rs in groups(paired, ['model', 'split', 'horizon_s', 'rule']).items():
        contrast.append(dict(zip(('model', 'split', 'horizon_s', 'rule'), vals)) |
            dict(extra_saving_vs_delay=avg(r['extra_saving_vs_delay'] for r in rs), changed_choices=sum(r['changed_choice'] for r in rs),
                 **interval(rs, 'extra_saving_vs_delay')))
    emit('controller_rule_contrast_summary', contrast)
    return summary, selected


def options():
    v4 = load(TEACH/'SNAPSHOT_1793521615.json')
    v5 = load(TEACH/'V5_QUALITY_SNAPSHOT_1793521619.json')
    cached_audit = load(OUT/'option_cache_audit.json')
    labels = {(r['state_id'], r['horizon_s']): r for r in v4['labels'] if r['role'] == 'QUALITY_ONLY'}
    assert len(labels) == 24 and v4['completed_future_tasks'] == v5['completed_futures'] == 96
    v4records = {(r['state_id'], r['horizon_s']): r for r in v4['files']['QUALITY_RECORDS.json']}
    v5records = {(r['state_id'], r['horizon_s']): r for r in v5['files']['QUALITY_RECORDS.json']}
    assert set(labels) == set(v4records) == set(v5records)
    # Cell IDs are causal selection metadata from the fixed one-state-per-cell panel.
    cells = {s: i for i, s in enumerate(v4['files']['IDENTITY.json']['quality_states'])}
    rows, decision = [], []
    for model, snapshot, records in [('FULL-v4', v4, v4records), ('FULL-v5', v5, v5records)]:
        ident = snapshot['files']['IDENTITY.json']
        audit = cached_audit['versions'][model]
        assert len(audit['tasks']) == 96 and all(t['all_breakdown_missing'] for t in audit['tasks'])
        assert audit['source'] == ident['source']
        means = {(r['state_id'], r['horizon_s']): r for r in audit['labels']}
        assert ident['K'] == 8 and ident['source'].startswith('22cc32f6' if model == 'FULL-v4' else 'b2df7458')
        for key, label in labels.items():
            r = records[key]
            absolute = means[key]['expected_delay']
            acts = [o for o, available in zip(('O0', 'O1', 'O2', 'O3', 'O4'), label['available']) if available]
            assert len(acts) == r['candidates'] == len(r['predicted_D']) == len(r['exact_D'])
            assert label['K'] == 8 and len(set(label['future_ids'])) == 8
            s, h = r['world_id'], r['horizon_s']
            WORLD_META[s] = ('diagnostic_cell_'+str(cells[r['state_id']]), 'one_state_per_cell', 'EXAM12')
            for i, a in enumerate(acts):
                if model == 'FULL-v4':
                    assert abs(label['effects'][a]-r['predicted_D'][i]) < 1e-8
                assert abs(absolute['O0']-absolute[a]-r['predicted_D'][i]) < 1e-6
                assert abs(r['exact_D'][i]-v4records[key]['exact_D'][i]) < 1e-8
                rows.append(dict(model=model, split='diagnostic_EXAM12', seed=s, state_id=r['state_id'],
                    demand='cell_'+str(cells[r['state_id']]), penetration='one_state_per_cell', arm=a, region='upstream', horizon_s=h,
                    pred_mean_delay=-r['predicted_D'][i], real_mean_delay=-r['exact_D'][i],
                    pred_absolute_mean_delay=absolute[a],
                    delay_clock='relative MAPPO J; common baseline cancels',
                    pred_occurrence=None, real_occurrence=None, pred_onset_rmst=None, real_onset_rmst=None,
                    pred_depth_expected=None, real_depth_expected=None, pred_onset_conditional=None, real_onset_conditional=None,
                    pred_depth_conditional=None, real_depth_conditional=None))
            chosen = max(acts, key=lambda a: r['predicted_D'][acts.index(a)])
            saving = r['exact_D'][acts.index(chosen)]
            assert chosen == r['choice'] and abs(saving-r['exact_saving']) < 1e-8
            decision.append(dict(model=model, split='diagnostic_EXAM12', seed=s, horizon_s=h, state_id=r['state_id'],
                rule='delay', chosen=chosen, available_options=acts, saving=saving, regret=max(r['exact_D'])-saving,
                oracle_saving=max(r['exact_D']), gate_saving=r['exact_gated_saving'], agreement=chosen == r['exact_choice']))
    emit('option_predictions', rows)
    concordance = ranking(rows, 'option')
    emit('option_decisions', decision)
    summary = []
    for vals, rs in groups(decision, ['model', 'split', 'horizon_s', 'rule']).items():
        for metric in ('saving', 'regret', 'oracle_saving', 'gate_saving', 'agreement'):
            summary.append(dict(zip(('model', 'split', 'horizon_s', 'rule'), vals)) | dict(metric=metric,
                worlds=len(rs), mean=avg(r[metric] for r in rs), **interval(rs, metric)))
    emit('option_decision_summary', summary)
    # No breakdown proxy is derived from delay or generic student queue features.
    canary = v4['files']['CANARY_PASS.json']
    minutes = {model: sum(t['wall_s'] for t in entry['tasks'])/60 for model, entry in cached_audit['versions'].items()}
    return summary, concordance, dict(v4_canary_future_wall_s=canary.get('wall_s'),
        v4_estimated_96_future_gpu_minutes=96*canary['wall_s']/60 if canary.get('wall_s') else None,
        estimated_repeat_worker_gpu_minutes=minutes,
        exact_detector_labels_available=False, imagined_detector_fields_available=False)


def round2_status():
    # Fixed snapshot: metadata only while the registered panel is incomplete.
    a = load(RB2/'COLLECT_20261101T161643+0800_gpuA.json')
    b = load(RB2/'COLLECT_20261101T161643+0800_gpuR.json')
    plan = load(RB2/'FINAL_RUN_PLAN.json')
    complete = a['complete']+b['complete']
    jobs = len(plan['jobs'])
    assert jobs == 216
    status = dict(snapshot='20261101T161643+0800', complete_world_jobs=len(complete), expected_world_jobs=jobs,
        controller_v5_scored=False, reason='registered round2 panel incomplete; no partial-panel outcome reads',
        peer_jobs_touched=False, inference_launched=False)
    save('round2_availability.json', status)
    return status


def main():
    assert ROOT.name == 'c4-bdrank' and OUT.parent.is_dir()
    OUT.mkdir(exist_ok=True)
    fn, sha = detector()
    real, rr = real_data(fn)
    fits, assocs, costs = cost_analysis(rr)
    print('Q1 complete', len(rr), flush=True)
    pr = predictions(fn, real)
    ranking(pr, 'controller')
    decision, selected = decisions(pr)
    print('Q2/Q3 controllers complete', len(pr), flush=True)
    opdecision, oprank, missing = options()
    rb2 = round2_status()
    controller_minutes = json.loads((OUT/'controller_cache_manifest.json').read_text())['total_recorded_worker_gpu_minutes']
    blockers = [
        dict(item='real_duration_and_queue_extent', status='STOP', missing='registered detector defines only event/onset/step depth, not recovery/duration/queue extent',
             inference_needed='none for existing real speed bins; new preregistered definitions and spatial observable reduction required', gpu_minutes=0),
        dict(item='controller_duration_and_queue_extent', status='STOP', missing='no registered duration or queue extent; regional speed means cannot identify queue length',
             inference_needed='definition first; existing bins may support a duration metric without inference; if no separate spatial archive exists,160 five-controller H600 futures saving position/speed/lane/presence are required for queue extent',
             gpu_minutes=controller_minutes,
             estimate_basis='conditional cost of missing spatial forecasts from total observed160-task occupied GPU time; definition/instrumentation overhead excluded'),
        dict(item='option_breakdown_onset_severity_and_rules_a_b_c_e', status='STOP',
             missing='WMTEACH scalar delay caches; runtime serializes breakdown=None; no saved regional speed/prehistory or exact option detector labels',
             inference_needed='96 paired H300 K8 option forecasts/version saving 10s region speed+60s prehistory; exact archived trajectories also needed, otherwise forbidden new SUMO',
             gpu_minutes=sum(missing['estimated_repeat_worker_gpu_minutes'].values()),
             estimate_basis='observed complete96-task totals: v4=71.4883/v5=91.7741 occupied GPU-min; excludes instrumentation overhead and exact-label recovery'),
        dict(item='controller_FULL_v5', status='PENDING', missing=f"{rb2['complete_world_jobs']}/{rb2['expected_world_jobs']} registered world jobs complete at frozen snapshot",
             inference_needed='wait for existing owner jobs; no redispatch', gpu_minutes=0),
    ]
    emit('blocked_items', blockers)
    save('summary.json', dict(detector_sha256=sha, checks=dict(CHECKS), real_episodes=500, prediction_tasks=160,
        new_sumo_sim_s=0, new_training=0, new_inference=0, process_signals=0,
        selected_development_rules=selected, option_missing=missing, round2=rb2))
    save('input_manifest.json', dict(inputs=[dict(path=k, sha256=v) for k, v in INPUTS.items()],
        detector_sha256=sha, explicit_path_guard=True))
    print(json.dumps(dict(checks=dict(CHECKS), selected=selected, option_missing=missing), indent=1), flush=True)


if __name__ == '__main__':
    main()
