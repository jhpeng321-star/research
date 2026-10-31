"""Decompose saved SUMOCORE setting-ablation effects on FIT worlds only.

No raw rollout/settings/assignment files, simulation, inference or fit. Outcome
diagnostics NEVER feed the motion-only shrinkage estimator.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forecasts', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if not args.out.parent.is_dir():
        raise ValueError('Output parent must exist')
    args.out.mkdir(exist_ok=False)
    raw = args.forecasts.read_bytes()
    data = json.loads(raw)
    rows = {}
    for episode in data['episodes']:
        job = episode['job']
        if not 969400 <= job['world'] <= 969431:
            continue
        key = (job['world'], job['variant'], job['arm'], job['rng'])
        if key in rows:
            raise ValueError('Duplicate effective forecast')
        rows[key] = episode['cumulative']
    worlds = sorted({key[0] for key in rows})
    seeds = sorted({key[3] for key in rows})
    if len(rows) != len(worlds)*2*3*len(seeds):
        raise ValueError('Incomplete matched FIT panel')
    cells = []
    for horizon in ('120', '300', '600'):
        for world in worlds:
            for arm in ('mappo', 'pi'):
                by_variant, absolute_change = {}, {}
                for variant in ('default', 'inferred'):
                    by_variant[variant] = {component: statistics.mean(
                        rows[world, variant, arm, seed][horizon][component]
                        - rows[world, variant, 'native', seed][horizon][component]
                        for seed in seeds) for component in ('source', 'road', 'team')}
                for name in ('native', arm):
                    absolute_change[name] = {component: statistics.mean(
                        rows[world, 'inferred', name, seed][horizon][component]
                        - rows[world, 'default', name, seed][horizon][component]
                        for seed in seeds) for component in ('source', 'road', 'team')}
                change = {c: by_variant['inferred'][c]-by_variant['default'][c]
                          for c in ('source', 'road', 'team')}
                assert abs(change['source']+change['road']-change['team']) < 1e-7
                cells.append(dict(world=world, horizon=int(horizon), arm=arm,
                    effect_by_variant=by_variant, inferred_minus_default_effect=change,
                    inferred_minus_default_absolute=absolute_change))
    summary = []
    for horizon in (120, 300, 600):
        for arm in ('mappo', 'pi'):
            cc = [c for c in cells if c['horizon'] == horizon and c['arm'] == arm]
            mean = lambda fn: statistics.mean(fn(c) for c in cc)
            summary.append(dict(horizon=horizon, arm=arm, n_worlds=len(cc),
                effect_by_variant={v: {k: mean(lambda c: c['effect_by_variant'][v][k])
                    for k in ('source', 'road', 'team')} for v in ('default', 'inferred')},
                inferred_minus_default_effect={k: mean(lambda c: c['inferred_minus_default_effect'][k])
                    for k in ('source', 'road', 'team')},
                inferred_minus_default_absolute={a: {k: mean(lambda c: c['inferred_minus_default_absolute'][a][k])
                    for k in ('source', 'road', 'team')} for a in ('native', arm)},
                leave_one_world_out_change={str(drop): statistics.mean(
                    c['inferred_minus_default_effect']['team'] for c in cc if c['world'] != drop)
                    for drop in worlds}))
    report = dict(schema='wm_phys.sumocore_fit_diagnosis.v1', worlds=worlds, seeds=seeds,
        source_sha256=hashlib.sha256(raw).hexdigest(), n_effective_futures=len(rows),
        qualification='Descriptive FIT-only outcome decomposition; NOT a causal parameter attribution or calibration input',
        cells=cells, summary=summary)
    (args.out/'diagnosis.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'worlds': worlds, 'summary_600': [r for r in summary if r['horizon'] == 600]}))


if __name__ == '__main__':
    main()
