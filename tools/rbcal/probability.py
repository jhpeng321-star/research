"""Source-bound, dev-fitted strict-win reliability map (not a dynamics repair)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.wmdecide.decision import SWITCH_CONFIDENCE, summarize
from tools.wmdecide.metrics import brier, WorldIntervals, reliability
from tools.wmdecide.report import write_json, write_csv, reliability_svg

DEV = tuple(range(969420, 969432))


def fingerprint(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def fit_retention(rows, *, excluded=None):
    if any(r['seed'] not in DEV for r in rows):
        raise ValueError('only development12 can fit probability calibration')
    rows = [r for r in rows if r['seed'] != excluded]
    midpoint = np.array([(1-r['predicted_tie_probability'])/2 for r in rows])
    probability, outcome = np.array([r['probability'] for r in rows]), np.array([r['outcome'] for r in rows])
    contrast = probability-midpoint
    denominator = contrast@contrast
    return float(np.clip(contrast@(outcome-midpoint)/denominator, 0, 1)) if denominator else 0.


def calibrated_probability(probability, tie_probability, retention):
    if (not all(np.isfinite(v) and 0 <= v <= 1 for v in (probability, tie_probability, retention))
            or probability+tie_probability > 1+1e-12):
        raise ValueError('invalid strict-win probability/tie/retention')
    return float(retention*probability+(1-retention)*(1-tie_probability)/2)


def calibrated_decide(actions, incumbent, *, horizon_s, calibration, source_fingerprint):
    """Opt-in wrapper: preserve trajectories and the registered 0.8/mean gate."""
    if (calibration.get('schema') != 'wmdecide.probability_calibration.v1'
            or calibration.get('source_fingerprint') != source_fingerprint
            or calibration.get('fit_worlds') != list(DEV)):
        raise ValueError('probability calibration/source or development binding mismatch')
    retention = calibration['retention_by_horizon'][str(horizon_s)]
    result = summarize(actions)
    if incumbent not in result['actions']:
        raise ValueError('incumbent must be included')
    for pair in result['pairwise']:
        pair['raw_probability_beats'] = pair['probability_beats']
        pair['probability_beats'] = calibrated_probability(pair['probability_beats'],
            pair['probability_tie'], retention)
    current = result['actions'][incumbent]['team_delay']['mean']
    eligible = [p['action'] for p in result['pairwise'] if p['reference'] == incumbent
                and p['probability_beats'] >= SWITCH_CONFIDENCE
                and result['actions'][p['action']]['team_delay']['mean'] < current]
    selected = min(eligible, key=lambda a:(result['actions'][a]['team_delay']['mean'], a)) if eligible else incumbent
    return dict(selected=selected, incumbent=incumbent, switched=selected != incumbent,
        eligible=sorted(eligible), confidence_level=SWITCH_CONFIDENCE,
        probability_calibration=calibration['source_fingerprint'], summary=result)


def evaluate(events, distributions):
    pairs = [r for r in events if r['target'] == 'preference']
    if {r['seed'] for r in pairs} != set(DEV):
        raise ValueError('all development12 worlds required')
    lookup = {(r['seed'], r['horizon_s'], r['action'], r['reference']):r for r in pairs}
    means = {(r['seed'], r['horizon_s'], r['action']):r for r in distributions if r['target'] == 'delay'}
    horizons = sorted({r['horizon_s'] for r in pairs})
    full = {str(h):fit_retention([r for r in pairs if r['horizon_s'] == h]) for h in horizons}
    oof = {world:{str(h):fit_retention([r for r in pairs if r['horizon_s'] == h], excluded=world)
                  for h in horizons} for world in DEV}
    scored, choices, selected_events = [], [], []
    for row in pairs:
        adapted = dict(row)
        adapted['raw_probability'] = row['probability']
        adapted['probability'] = calibrated_probability(row['probability'], row['predicted_tie_probability'],
                                                       oof[row['seed']][str(row['horizon_s'])])
        adapted['brier'] = brier(adapted['probability'], adapted['outcome'])
        scored.append(adapted)
    for world in DEV:
        for h in horizons:
            arms = sorted(a for s, horizon, a in means if s == world and horizon == h)
            for incumbent in arms:
                current = means[world, h, incumbent]
                eligible = []
                for action in arms:
                    if action == incumbent:
                        continue
                    direct = lookup.get((world, h, action, incumbent))
                    reverse = lookup.get((world, h, incumbent, action))
                    row = direct or reverse
                    tie = row['predicted_tie_probability']
                    p = row['probability'] if direct else max(0., 1-tie-row['probability'])
                    calibrated = calibrated_probability(p, tie, oof[world][str(h)])
                    candidate = means[world, h, action]
                    if calibrated >= SWITCH_CONFIDENCE and candidate['expected_delay'] < current['expected_delay']:
                        eligible.append((candidate['expected_delay'], action, calibrated, p))
                selected = min(eligible) if eligible else (current['expected_delay'], incumbent, None, None)
                expected, action, probability, raw = selected
                saving = current['outcome']-means[world, h, action]['outcome']
                choices.append(dict(seed=world, horizon_s=h, incumbent=incumbent, selected=action,
                    switched=action != incumbent, observed_saving=saving,
                    expected_saving=current['expected_delay']-expected))
                if action != incumbent:
                    event = dict(current, target='selected_preference', action=action, reference=incumbent,
                        probability=probability, raw_probability=raw, outcome=int(saving > 0),
                        brier=brier(probability, int(saving > 0)))
                    selected_events.append(event)
    summary = []
    for h in horizons:
        good = [r for r in scored if r['horizon_s'] == h]
        selected = [r for r in selected_events if r['horizon_s'] == h]
        old = [r for r in events if r['target'] == 'selected_preference' and r['horizon_s'] == h]
        summary.append(dict(horizon_s=h, raw_pair_brier=float(np.mean([r['brier'] for r in pairs if r['horizon_s'] == h])),
            oof_pair_brier=float(np.mean([r['brier'] for r in good])),
            raw_selected_calls=len(old), raw_selected_brier=float(np.mean([r['brier'] for r in old])) if old else None,
            oof_selected_calls=len(selected),
            oof_selected_brier=float(np.mean([r['brier'] for r in selected])) if selected else None,
            oof_selected_probability=float(np.mean([r['probability'] for r in selected])) if selected else None,
            oof_selected_observed_win=float(np.mean([r['outcome'] for r in selected])) if selected else None,
            oof_switch_rate=float(np.mean([r['switched'] for r in choices if r['horizon_s'] == h])),
            oof_observed_saving=float(np.mean([r['observed_saving'] for r in choices if r['horizon_s'] == h]))))
    return full, oof, scored, selected_events, choices, summary


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if not args.out.parent.is_dir() or args.out.exists():
        raise ValueError('new probability output required')
    audit = json.loads((args.report/'audit.json').read_bytes())
    observations = json.loads((args.report/'observations.json').read_bytes())
    if {r['seed'] for r in observations['events']} != set(DEV):
        raise ValueError('development-only observations required')
    full, oof, pairs, selected, choices, summary = evaluate(observations['events'], observations['distributions'])
    contract = audit['model']
    artifact = dict(schema='wmdecide.probability_calibration.v1', fit_worlds=list(DEV),
        source_fingerprint=fingerprint(contract), source_model=contract['model'],
        retention_by_horizon=full, objective='all unordered pair strict-win Brier',
        qualification='development fit; source bound; no independent decision qualification')
    args.out.mkdir()
    for name, value in [('calibration.json', artifact), ('crossfits.json', oof),
        ('summary.json', summary), ('events.json', pairs+selected), ('decisions.json', choices)]:
        write_json(args.out/name, value)
    diagrams = []
    intervals = WorldIntervals({w:('development12',) for w in DEV})
    for h in full:
        for target, rows in [('preference', pairs), ('selected_preference', selected)]:
            rows = [r for r in rows if r['horizon_s'] == int(h)]
            for row in reliability(rows, intervals):
                row.update(observed_low=None, observed_high=None, uncertainty_available=False)
                diagrams.append(dict(row, scope='pooled', target=target, region='',
                                     horizon_s=int(h), split='development12'))
    write_csv(args.out/'reliability.csv', diagrams)
    for target in ('preference', 'selected_preference'):
        if any(r['target'] == target and r['count'] for r in diagrams):
            with (args.out/f'reliability_{target}.svg').open('x', encoding='utf-8') as f:
                f.write(reliability_svg(diagrams, target, 'Source-bound OOF probability calibration'))
    print(json.dumps(dict(artifact=artifact, summary=summary), indent=1))


if __name__ == '__main__':
    main()
