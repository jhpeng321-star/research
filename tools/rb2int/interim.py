"""Outcome-blind complete-cell interim v5/v4 analysis from collected K8 only."""
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from tools.rb2int.compare import propagate_rate_distance
from tools.rb2int.credibility import summarize, ratio_bands
from tools.rb2int.predictions import read_model
from tools.rb2int.uncertainty import PairedWorldBootstrap
from tools.rbint.score_endpoints import world_errors
from tools.rbeval.core import ARMS, HS, REGIONS, crps, digest, load, save_new
from tools.wmdecide.rbeval_adapter import observations

MODELS = ('FULL-v4', 'FULL-v5')
PAIRS = tuple(combinations(ARMS, 2))
SHIFTS = tuple((a, b) for b in ('native', 'ctrl0') for a in ARMS if a != b)


def scopes(complete, registration, metadata):
    """Membership/completion only: never select a cell using its forecast score."""
    complete = set(complete)
    cells = defaultdict(list)
    for s in registration['replicated_primary_worlds']:
        cells[metadata[s]['demand'], metadata[s]['penetration']].append(s)
    # The frozen design permits available-case diagnostics if replication is
    # rebuilt using scored worlds alone. Do not wait for unscored peers or let
    # them supply bootstrap variation. This is never a complete primary54 claim.
    scored_groups = [[s for s in group if s in complete] for group in cells.values()]
    replicated = sorted(s for group in scored_groups if len(group) >= 2 for s in group)
    result = {}
    if set(registration['reserved8']) <= complete:
        result['reserved8'] = registration['reserved8']
    if replicated:
        result['available_replicated_cells'] = replicated
    return result


def prefix_cost(task, horizon, field='real_J'):
    bins = np.asarray(task['costs']['time_bins_30s'][field], float)
    if bins.shape != (len(ARMS), 20) or not np.isfinite(bins).all():
        raise ValueError('complete finite arm-indexed 30s costs required')
    for h in HS:
        if not np.allclose(bins[:, :h//30].sum(1), task['costs'][str(h)][field], rtol=1e-10, atol=1e-6):
            raise ValueError('saved short increments differ from registered clocks')
    return bins[:, :horizon//30].sum(1)


def bounded_direction(predicted, low, high):
    """Team-effect true sign is identified only when its source-wait bounds resolve."""
    if low > high or not np.isfinite([predicted, low, high]).all():
        raise ValueError('finite ordered factual effect bounds required')
    truth = 1 if low > 0 else -1 if high < 0 else 0 if low == high == 0 else None
    return None if truth is None else float(np.sign(predicted) == truth)


def short_quality(audits, targets, worlds, metadata):
    lookup = {(r['seed'], r['arm']): r for r in targets['episodes']}
    boot = PairedWorldBootstrap([metadata[s] for s in worlds])
    samples, provenance = {}, []
    for model in MODELS:
        for item in audits[model]['inputs']:
            path = Path(item['task'])
            if digest(path) != item['sha256']:
                raise ValueError('short-horizon task changed since validation')
            task = load(path)
            if task['seed'] not in worlds:
                continue
            for h in (30, 60, 90):
                samples[model, task['seed'], task['future'], h] = (
                    prefix_cost(task, h), prefix_cost(task, h, 'real_road'))
            provenance.append(item)
    rows, features, labels = [], [], []
    for model in MODELS:
        for h in (30, 60, 90):
            subset = []
            for seed in worlds:
                team = np.mean([samples[model, seed, k, h][0] for k in range(8)], axis=0)
                road = np.mean([samples[model, seed, k, h][1] for k in range(8)], axis=0)
                for a, b in PAIRS:
                    ia, ib = ARMS.index(a), ARMS.index(b)
                    ta, tb = lookup[seed, a], lookup[seed, b]
                    if not ta['team_delay_bounds_vs'] or not tb['team_delay_bounds_vs']:
                        raise ValueError('audited short factual bounds required')
                    alo, ahi = ta['team_delay_bounds_vs'][str(h)]
                    blo, bhi = tb['team_delay_bounds_vs'][str(h)]
                    resolved = bounded_direction(team[ia]-team[ib], alo-bhi, ahi-blo)
                    actual_road = ta['road_delay_vs'][str(h)]-tb['road_delay_vs'][str(h)]
                    row = dict(seed=seed, model=model, horizon_s=h, arm=a, reference=b,
                        predicted_team_effect_vs=float(team[ia]-team[ib]),
                        real_team_effect_bounds_vs=[alo-bhi, ahi-blo],
                        resolved_correct=resolved, resolved=float(resolved is not None),
                        accuracy_lower=0. if resolved is None else resolved,
                        accuracy_upper=1. if resolved is None else resolved,
                        road_direction_correct=float(np.sign(road[ia]-road[ib]) == np.sign(actual_road)))
                    subset.append(row);rows.append(row)
            for field in ('accuracy_lower', 'accuracy_upper', 'resolved', 'resolved_correct', 'road_direction_correct'):
                features.append((subset, field));labels.append(dict(model=model, horizon_s=h, metric=field))
    summaries = [dict(label, **band) for label, band in zip(labels, ratio_bands(boot, features))]
    for row in summaries:
        if row['low'] is not None:
            row['low'], row['high'] = max(0.,row['low']), min(1.,row['high'])
    return dict(summary=summaries, per_pair=rows, inputs=provenance,
        target_limit='No30/60/90 recorded total clocks. Exact road replay checked at120s; source waiting bounded by observed cumulative120. Team direction partially identified, never imputed.',
        uncertainty='whole-world fixed-cell, conditional on saved K8; family includes both models/all3horizons')


def analyze(runs, real_path, factual_path, short_path, worlds, scope, out):
    real = load(real_path)['episodes']
    lookup = {(r['seed'], r['arm']): r for r in real}
    meta = {r['seed']:dict(seed=r['seed'], demand=r['demand'], penetration=r['penetration']) for r in real}
    pred, audits = {}, {}
    for model in MODELS:
        pred[model], audits[model] = read_model(runs, model, real, worlds)
    boot = PairedWorldBootstrap([meta[s] for s in worlds])
    delay = np.zeros((len(worlds), 2, 3))
    bias = np.zeros_like(delay)
    signed = np.zeros((len(worlds), 8, 2, 3))
    shifts = np.zeros((len(worlds), 8, 3, 3, len(SHIFTS)))
    direction = np.zeros((len(worlds), 2, 3))
    actual_rates = np.zeros((len(worlds), 3))
    for i, seed in enumerate(worlds):
        for j, h in enumerate(HS):
            y = {a:lookup[seed,a]['clocks'][str(h)]['team_delay_vs'] for a in ARMS}
            re = [lookup[seed,a]['bins']['upstream']['events'][str(h)]['event'] for a in ARMS]
            if any(x is None for x in re):
                raise ValueError('missing factual event support')
            actual_rates[i,j] = np.mean(re)
            for m, model in enumerate(MODELS):
                x = {a:np.array([e['clocks'][str(h)]['team_delay_vs'] for e in pred[model][seed,a]]) for a in ARMS}
                delay[i,m,j] = np.mean([crps(x[a],y[a]) for a in ARMS])
                bias[i,m,j] = np.mean([x[a].mean()-y[a] for a in ARMS])
                events = [[e['bins']['upstream']['events'][str(h)]['event'] for e in pred[model][seed,a]] for a in ARMS]
                if any(e is None for arm in events for e in arm):
                    raise ValueError('missing predicted event support')
                signed[i,:,m,j] = np.mean(events,axis=0)-actual_rates[i,j]
                direction[i,m,j] = np.mean([np.sign(x[a].mean()-x[b].mean()) == np.sign(y[a]-y[b]) for a,b in PAIRS])
                for p,(a,b) in enumerate(SHIFTS):
                    shifts[i,:,m,j,p] = x[a]-x[b]
            for p,(a,b) in enumerate(SHIFTS):
                shifts[i,:,2,j,p] = y[a]-y[b]
    delay_bands = boot.bands(delay[:,1]-delay[:,0])
    sbands = boot.bands(signed.mean(1).reshape(len(worlds),-1), futures=signed.reshape(len(worlds),8,-1))
    rate_bands = [propagate_rate_distance(sbands[3+j],sbands[j]) for j in range(3)]
    for row in rate_bands:
        row.update(family_size=3,signed_bias_family_size=6)
    shift_bands = boot.bands(shifts.mean(1).reshape(len(worlds),-1), futures=shifts.reshape(len(worlds),8,-1))
    shift_rows = [dict(model=model,horizon_s=h,arm=a,reference=b,**band)
        for (model,h,a,b),band in zip(((m,h,a,b) for m in (*MODELS,'recorded') for h in HS for a,b in SHIFTS),shift_bands)]
    direction_bands = boot.bands(direction.reshape(len(worlds),-1))
    factual = load(factual_path)
    errors = world_errors(pred['FULL-v5'],pred['FULL-v4'],factual['episodes'],worlds)
    endpoints = boot.bands(np.array([[r['median_difference_m']] for r in errors],float))[0]
    endpoints['per_world'] = errors
    credibility = {}
    for model in MODELS:
        data = observations(pred[model],real,ARMS,HS,REGIONS)
        credibility[model] = summarize(data,[meta[s] for s in worlds])[0]
    short = short_quality(audits,load(short_path),worlds,meta)
    report = dict(schema='rb2int-interim/1',label='INTERIM',scope=scope,worlds=worlds,
        stratum_counts=boot.stratum_counts,replicated=boot.replicated,delay_crps_difference=delay_bands,
        absolute_delay_crps={m:delay[:,i].mean(0).tolist() for i,m in enumerate(MODELS)},
        mean_delay_bias={m:bias[:,i].mean(0).tolist() for i,m in enumerate(MODELS)},
        breakdown_rate_distance_difference=rate_bands,
        breakdown_rates={m:(signed[:,:,i].mean((0,1))+actual_rates.mean(0)).tolist() for i,m in enumerate(MODELS)},
        recorded_breakdown_rates=actual_rates.mean(0).tolist(),hdv60_difference=endpoints,
        controller_shifts=shift_rows,direction_accuracy=[dict(model=m,horizon_s=h,**band)
            for (m,h),band in zip(((m,h) for m in MODELS for h in HS),direction_bands)],
        credibility=credibility,short_horizon=short,input_audits=audits,
        source_sha256=digest(__file__),short_targets_sha256=digest(short_path),
        approved_runs_sha256=digest(runs),real_sha256=digest(real_path),
        bootstrap=10000,bootstrap_seed=20261101,
        ci_scope='available complete cells only; not final qualification; no repeated-look sequential guarantee',
        switching_eligibility='NOT_ESTABLISHED; INTERIM v5/v4 does not replace registered four-model comparison')
    save_new(out,report)
    return report


def interval(row, scale=1., digits=1, singleton=False):
    point = 'NA' if row.get('mean') is None else f"{row['mean']*scale:,.{digits}f}"
    if row.get('low') is None:
        return point+'; CI '+('NA (singleton)' if singleton else 'NA (insufficient support/variance)')
    return point+f" [95% CI {row['low']*scale:,.{digits}f}, {row['high']*scale:,.{digits}f}]"


def render(report, stamp):
    singleton = not report['replicated']
    ci = lambda row,scale=1.,digits=1: interval(row,scale,digits,singleton)
    lines = [f"## INTERIM v5 versus v4 — {stamp} ({report['scope']})", '',
        f"**{len(report['worlds'])} matched complete K8 worlds; all five arms.** This is an interim diagnostic, not model selection or switching authorization.",
        'Worlds: '+', '.join(map(str,report['worlds']))+'.',
        'CIs use10,000 paired fixed-cell whole-world resamples; delay/HDV/accuracy condition on stored K8. Rate/effect means include paired future resampling. Each metric family is simultaneous across its displayed horizons/models; this interim two-model family is separate from the final registered six-contrast family. No sequential/repeated-look confidence claim.',
        'Strata: '+str(report['stratum_counts'])+'.', '', '### Delay, breakdown and HDV60']
    for model in MODELS:
        lines.append('- '+model+' delay CRPS120/300/600='+ '/'.join(f'{v:,.0f}' for v in report['absolute_delay_crps'][model])+'veh-s; mean bias='+ '/'.join(f'{v:,.0f}' for v in report['mean_delay_bias'][model])+'veh-s.')
    for i,h in enumerate(HS):
        lines.append(f'- {h}s v5−v4 delay CRPS: '+ci(report['delay_crps_difference'][i])+'veh-s; breakdown absolute rate-distance: '+ci(report['breakdown_rate_distance_difference'][i],100,2)+'pp.')
    lines.append('- v5−v4 HDV60 mean-world-median position-error difference: '+ci(report['hdv60_difference'])+'m.')
    lines += ['', '### Controller-effect signs and credibility']
    for model in MODELS:
        rows = report['credibility'][model]
        for h in HS:
            select = lambda target,metric: next(r for r in rows if (r['target'],r['metric'],r['horizon_s'])==(target,metric,h))
            calls = select('selected_preference','outcome');forecast = select('selected_preference','probability')
            coverage = next(r for r in rows if r['target']=='delay' and r['metric']=='covered' and r['horizon_s']==h and r['interval']=='range')
            accuracy = next(r for r in report['direction_accuracy'] if r['model']==model and r['horizon_s']==h)
            lines.append(f"- {model}/{h}s:0.8calls n={calls['count']}, observed win "+ci(calls,100)+'% vs forecast '+ci(forecast,100)+'%; range coverage '+ci(coverage,100)+'%; all10pair direction accuracy '+ci(accuracy,100)+'%.')
        for a,b in SHIFTS:
            pred = [r for r in report['controller_shifts'] if r['model']==model and (r['arm'],r['reference'])==(a,b)]
            actual = [r for r in report['controller_shifts'] if r['model']=='recorded' and (r['arm'],r['reference'])==(a,b)]
            lines.append(f'- {model} {a}−{b} expected delay effect120/300/600: '+' / '.join(ci(r) for r in pred)+'veh-s; recorded '+' / '.join(ci(r) for r in actual)+'veh-s. Negative means lower delay.')
    lines += ['', '### Short-horizon decision quality:30/60/90s',
        'Exact total-delay accuracy is **partially identified**: the recorded data saved total/source clocks only at120/300/600s. Reconstructed road clocks match recorded120s road delay; unknown short source waiting is bounded between0 and each arm\'s recorded120s source waiting. The intervals below bound all-pair team-effect accuracy without inventing a source trajectory. Road-only accuracy is a separate diagnostic.',
        'These are five recorded controller options, not unrecorded cooperation membership/group interventions. Cooperation/planner short-horizon qualification remains NOT ESTABLISHED.']
    rows = report['short_horizon']['summary']
    for model in MODELS:
        for h in (30,60,90):
            get = lambda metric: next(r for r in rows if (r['model'],r['horizon_s'],r['metric'])==(model,h,metric))
            lines.append(f'- {model}/{h}s: team direction accuracy identification lower '+ci(get('accuracy_lower'),100)+'%, upper '+ci(get('accuracy_upper'),100)+'%; resolved-pair fraction '+ci(get('resolved'),100)+'%; resolved-only accuracy '+ci(get('resolved_correct'),100)+'%; road-only direction accuracy '+ci(get('road_direction_correct'),100)+'%.')
    lines += ['', 'Strict paired P(beats incumbent)>=0.8 plus lower expected delay; ties are non-wins. K8 min–max range is not a90% interval (continuous exchangeability reference7/9). Zero-selected worlds remain in resampling. Full gate/switching: NOT ESTABLISHED at this interim.']
    return '\n'.join(lines)+'\n'
