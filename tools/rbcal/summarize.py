"""Compact engineering receipt for development calibration experiments."""
import argparse
import json
from pathlib import Path

import numpy as np

from tools.wmdecide.report import write_json


def compact(rows):
    result = {}
    for h in (120, 300, 600):
        selected = [r for r in rows if r['horizon_s'] == h]
        record = {}
        for target in ('delay', 'paired_delay'):
            crps = next(r['mean'] for r in selected if r['target'] == target and r['metric'] == 'crps')
            coverage = {str(r['nominal_level']):r['mean'] for r in selected if r['target'] == target
                        and r['metric'] == 'covered' and r['interval'] == 'central'}
            record[target] = dict(crps=crps, coverage=coverage)
        for target in ('preference', 'selected_preference'):
            record[target] = {r['metric']:r['mean'] for r in selected if r['target'] == target
                               and r['metric'] in ('brier', 'probability', 'outcome')}
            brier = next(r for r in selected if r['target'] == target and r['metric'] == 'brier')
            record[target]['calls'] = brier['count']
        record['switch_rate'] = next(r['mean'] for r in selected if r['target'] == 'picker' and r['metric'] == 'switched')
        result[str(h)] = record
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--fit', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    data = json.loads(args.results.read_bytes())
    fit = json.loads((args.fit/'calibration.json').read_bytes())
    validation = json.loads((args.fit/'validation.json').read_bytes())
    crossfits = validation['crossfits']
    result = dict(parameters=fit['parameters'], full_fit=validation['full_fit'],
        crossfit_count_nll_baseline=float(np.mean([r['scores']['baseline_count_nll'] for r in crossfits])),
        crossfit_count_nll_calibrated=float(np.mean([r['scores']['calibrated_count_nll'] for r in crossfits])),
        scalar_crossfit_ranges={name:[min(r['parameters'][name] for r in crossfits),
                                    max(r['parameters'][name] for r in crossfits)]
                               for name in fit['parameters']}, physical=compact(data['physical']))
    if 'diagnostic_oof_scale' in data:
        result.update(diagnostic_oof_scale=compact(data['diagnostic_oof_scale']),
                      full_fit_outcome_scales=data['full_fit_scale'])
    lines = ['# S-RB2-CAL development receipt', '',
        'Internal development evidence only. Full-world CIs unavailable: demand × penetration cells are unreplicated.', '',
        '## Learned physical parameters', '',
        f"- Posterior retention: {fit['parameters']['posterior_retention']:.8g}.",
        f"- Gamma-rate variance multiplier: {fit['parameters']['arrival_variance_multiplier']:.8g}.",
        f"- Entrant concentration: {fit['parameters']['entrant_concentration']} (null means fixed-population infinity).",
        f"- Count NLL, whole-world-excluded scalar fits: {result['crossfit_count_nll_baseline']:.6f} → {result['crossfit_count_nll_calibrated']:.6f}.", '',
        'The population itself was fitted on FIT32; scalar crossfits are not independent population validation.', '',
        '## Development traffic outcomes', '']
    for name in ('physical', 'diagnostic_oof_scale'):
        if name not in result:
            continue
        lines += [f'### {name}', '']
        if name != 'physical':
            lines += ['NOT a physical candidate: mean-preserving outcome-scale limitation experiment, scalar fitted excluding entire target world.', '']
        for horizon, record in result[name].items():
            lines.append(f'**{horizon} s**')
            for target in ('delay', 'paired_delay'):
                row = record[target]
                coverage = '/'.join(f"{100*row['coverage'][str(level)]:.1f}" for level in (.5, .8, .9))
                lines.append(f"- {target}: CRPS {row['crps']:.2f} veh-s; 50/80/90% coverage {coverage}%.")
            for target in ('preference', 'selected_preference'):
                row = record[target]
                brier = 'NA' if row['brier'] is None else f"{row['brier']:.5f}"
                lines.append(f"- {target}: Brier {brier}, n={row['calls']}; forecast={row['probability']}, observed={row['outcome']}.")
            lines.append('')
    lines += ['Strict paired ties are non-wins. Selected-call sets differ across methods.',
              'Reliability diagrams and exact decile support are adjacent to each raw outcome RESULTS.json.',
              'Reserved8 outcomes were not scored or used for parameter selection. No outcome-coverage qualification is implied.']
    write_json(args.out.with_suffix('.json'), result)
    with args.out.open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines)+'\n')
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
