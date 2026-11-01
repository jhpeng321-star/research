"""Complete-K paired development summaries; singleton cells have no valid CIs."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from tools.rbeval.core import ARMS, HS, REGIONS, crps, digest, load, onset_vector, save_new


def mean(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def compare(panel, targets, *, expected_worlds=tuple(range(969420,969432))):
    assert expected_worlds and all(w in range(969420,969432) for w in expected_worlds)
    assert panel['worlds'] == list(expected_worlds) and panel['K'] == 2
    real = {(e['seed'],e['arm']): e for e in load(Path(__file__).resolve().parents[1]/'rbeval/real_observables.json')['episodes']
            if e['seed'] in panel['worlds']}
    lc = defaultdict(lambda:np.zeros(60))
    with (targets/'lc_counts_10s.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            seed = int(row['seed'])
            if seed in panel['worlds']:
                lc[seed,row['arm']][int(row['bin_index'])] += int(row['count'])
    predictions, receipts, audit = {}, {}, []
    for params, records in panel['records'].items():
        assert [r['world'] for r in records] == panel['worlds']
        for record in records:
            world = record['world']
            root = Path(record['directory'])
            complete = load(root/'LP_COMPLETE.json')
            assert complete['K'] == 2 and complete['world'] == world and complete['params'] == params
            identity = load(root/'IDENTITY.json')
            assert identity['K'] == 2 and identity['planned_K'] == 8 and identity['worlds'] == [world]
            identity_sha = digest(root/'IDENTITY.json')
            for k in range(2):
                path = root/f'w{world}_k{k}.json'
                task = load(path)
                assert digest(path) == complete['tasks'][k]['sha256']
                assert task['identity_sha256'] == identity_sha
                assert task['schema'] == 'rb2lphys.development/1' and task['seed'] == world and task['future'] == k
                assert task['future_sampling']['population_audit']['conserved']
                predictions[params,world,k] = task
                receipts[params,world,k] = task['future_sampling']
                audit.append(dict(params=params,world=world,future=k,path=str(path),sha256=digest(path)))
    for world in panel['worlds']:
        for k in range(2):
            a,b = receipts['off',world,k],receipts['learned_v1',world,k]
            for name in ('seed','arrival_sample','demand_rate_draws','observable_hashes','motion_rows_by_observed_id','fitted_artifacts'):
                assert a[name] == b[name], 'CRN/source mismatch: '+name
            assert a['population_audit']['due'] == b['population_audit']['due']
    world_rows, cells = [], defaultdict(list)
    for params in panel['records']:
        for world in panel['worlds']:
            tasks = [predictions[params,world,k] for k in range(2)]
            episodes = [{e['arm']:e for e in task['episodes']} for task in tasks]
            for arm in ARMS:
                r = real[world,arm]
                es = [e[arm] for e in episodes]
                for h in HS:
                    clock = str(h)
                    x = [e['clocks'][clock]['team_delay_vs'] for e in es]
                    y = r['clocks'][clock]['team_delay_vs']
                    draws = [e['bins']['upstream']['events'][clock] for e in es]
                    event = r['bins']['upstream']['events'][clock]
                    valid = event['event'] is not None and all(d['event'] is not None for d in draws)
                    row = dict(params=params,world=world,arm=arm,horizon_s=h,
                        demand=r['demand'],penetration=r['penetration'],K=2,
                        delay_crps=crps(x,y),delay_fair_crps=crps(x,y,True),delay_bias=mean(x)-y,
                        delay_mean=mean(x),delay_real=y,
                        range_coverage=float(min(x)<=y<=max(x)),
                        upstream_rate=mean([float(d['event']) for d in draws]) if valid else None,
                        upstream_real=float(event['event']) if valid else None,
                        onset_ibs=None)
                    if valid:
                        v = np.mean([onset_vector(d,h) for d in draws],0)
                        rv = onset_vector(event,h)
                        row['onset_ibs'] = float(np.mean((v[1:-1]-rv[1:-1])**2))
                        cells[params,r['demand'],r['penetration'],arm,h].append(row)
                    for region in REGIONS:
                        scores = []
                        for j in range(6,6+h//10):
                            yy = r['bins'][region]['speed'][j]
                            xx = [e['bins'][region]['speed'][j] for e in es]
                            if yy is not None and all(v is not None for v in xx):scores.append(crps(xx,yy))
                        row['speed_crps_'+region] = mean(scores)
                        row['speed_supported_bins_'+region] = len(scores)
                    real_counts = lc[world,arm][:h//10]
                    counts = np.array([task['costs']['lp_lane_changes_10s'][ARMS.index(arm)][:h//10] for task in tasks])
                    row['lc_count_crps'] = crps(counts.sum(-1),float(real_counts.sum()))
                    row['lc_10s_crps'] = mean([crps(counts[:,j],real_counts[j]) for j in range(h//10)])
                    row['lc_mean'] = float(counts.sum(-1).mean())
                    row['lc_real'] = float(real_counts.sum())
                    for key in ('lp_ramp_lane_road','lp_ramp_source'):
                        row[key] = mean([t['costs'][clock][key][ARMS.index(arm)] for t in tasks])
                    world_rows.append(row)
    summary = []
    numeric = ('delay_crps','delay_fair_crps','delay_bias','range_coverage','upstream_rate','upstream_real',
        'onset_ibs','speed_crps_upstream','speed_crps_merge','speed_crps_ramp','lc_count_crps',
        'lc_10s_crps','lc_mean','lc_real','lp_ramp_lane_road','lp_ramp_source')
    for params in panel['records']:
        for h in HS:
            rs = [r for r in world_rows if r['params']==params and r['horizon_s']==h]
            row = dict(params=params,horizon_s=h,worlds=len(expected_worlds),arms=5,K=2,**{n:mean([r[n] for r in rs]) for n in numeric})
            row['upstream_cell_rate_distance_pp'] = 100*mean([abs(mean([r['upstream_rate'] for r in rows])-mean([r['upstream_real'] for r in rows]))
                for key,rows in cells.items() if key[0]==params and key[-1]==h])
            row['confidence_interval'] = None
            row['uncertainty_note'] = 'singleton demand×penetration cells; whole-panel95%CIs unavailable'
            summary.append(row)
    effects = []
    for params in panel['records']:
        for h in HS:
            rows = {(r['world'],r['arm']):r for r in world_rows if r['params']==params and r['horizon_s']==h}
            for arm in ARMS[1:]:
                predicted = [rows[w,arm]['delay_mean']-rows[w,'native']['delay_mean'] for w in panel['worlds']]
                observed = [rows[w,arm]['delay_real']-rows[w,'native']['delay_real'] for w in panel['worlds']]
                effects.append(dict(params=params,horizon_s=h,arm=arm,predicted_effect=mean(predicted),
                    recorded_effect=mean(observed),point_sign_agreement=bool(np.sign(mean(predicted))==np.sign(mean(observed))),
                    confidence_interval=None,interpretation='descriptive expected point effect; sign resolution not established'))
    before = next(r for r in summary if r['params']=='off' and r['horizon_s']==300)
    after = next(r for r in summary if r['params']=='learned_v1' and r['horizon_s']==300)
    improves = all(next(r for r in summary if r['params']=='learned_v1' and r['horizon_s']==h)['delay_crps'] <
                   next(r for r in summary if r['params']=='off' and r['horizon_s']==h)['delay_crps'] for h in (300,600))
    eligible = improves and after['upstream_cell_rate_distance_pp']<=before['upstream_cell_rate_distance_pp']
    return dict(schema='rb2lphys.comparison.v1',summary=summary,effects=effects,world_rows=world_rows,
        audit=audit,development_screen_pass=eligible,final_qualification=False,
        fit_worlds=list(range(969400,969432)),selection_worlds=panel['worlds'],
        limits=['K2 subset of K8 planning, not K8 qualification.',
                'Ramp-lane road loss and ramp-source waiting reported separately; not full ramp-origin delay.',
                'LC10s reference half-open endpoint convention differs at exact boundaries; count totals are primary.'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--panel', type=Path,required=True)
    p.add_argument('--targets',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args = p.parse_args()
    assert args.out.is_dir() and not (args.out/'COMPARISON.json').exists()
    result = compare(load(args.panel),args.targets)
    save_new(args.out/'COMPARISON.json',result)
    for name in ('summary','effects','world_rows'):
        with (args.out/(name+'.csv')).open('x',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=list(result[name][0]));writer.writeheader();writer.writerows(result[name])
    print(json.dumps(dict(summary=result['summary'],development_screen_pass=result['development_screen_pass'])),flush=True)


if __name__ == '__main__':
    main()
