"""Matched-world WM-PHYS vs WM2 switching diagnostics; no inference or fitting.

Intervals condition on the saved K8 ensembles. All arms stay together inside a
world, with EVAL's frozen demand/penetration/split-stratified bootstrap weights.
Missing endpoints/partial panels never imply switching eligibility.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from tools.rbeval.core import ARMS, HS, crps, digest, load, save_new
from tools.rbeval.score import WorldBootstrap, read_predictions


def paired_features(candidate, baseline, real, worlds):
    lookup = {(e['seed'], e['arm']): e for e in real}
    delay, cp, bp, rp = [], [], [], []
    for seed in worlds:
        dr, cpr, bpr, rpr = [], [], [], []
        for h in HS:
            clock = str(h)
            differences, cv, bv, rv = [], [], [], []
            for arm in ARMS:
                target = lookup[seed, arm]
                observed = target['clocks'][clock]['team_delay_vs']
                c = [e['clocks'][clock]['team_delay_vs'] for e in candidate[seed, arm]]
                b = [e['clocks'][clock]['team_delay_vs'] for e in baseline[seed, arm]]
                differences.append(crps(c, observed)-crps(b, observed))
                ce = [e['bins']['upstream']['events'][clock]['event'] for e in candidate[seed, arm]]
                be = [e['bins']['upstream']['events'][clock]['event'] for e in baseline[seed, arm]]
                re = target['bins']['upstream']['events'][clock]['event']
                if re is None or any(v is None for v in ce+be):
                    raise ValueError('missing upstream event support prevents switching comparison')
                cv.append(np.mean(ce)); bv.append(np.mean(be)); rv.append(float(re))
            dr.append(np.mean(differences))
            cpr.append(np.mean(cv)); bpr.append(np.mean(bv)); rpr.append(np.mean(rv))
        delay.append(dr); cp.append(cpr); bp.append(bpr); rp.append(rpr)
    return [np.asarray(v, dtype=float).reshape(len(worlds), len(HS)) for v in (delay, cp, bp, rp)]


def summarize_features(features, worlds, bootstrap):
    delay, cp, bp, rp = features
    if not worlds:
        return dict(worlds=[], status='NO_COMPLETE_PAIRED_K8_WORLD')
    weights = bootstrap.weights[:, [bootstrap.index[s] for s in worlds]]
    denominator = weights.sum(-1, keepdims=True)
    norm = np.divide(weights, denominator, out=np.full_like(weights, np.nan), where=denominator > 0)
    delay_point = delay.mean(0)
    rate_point = np.abs(cp.mean(0)-rp.mean(0))-np.abs(bp.mean(0)-rp.mean(0))
    draws = np.concatenate((norm@delay, np.abs(norm@cp-norm@rp)-np.abs(norm@bp-norm@rp)), axis=1)
    point = np.r_[delay_point, rate_point]
    sd = draws.std(0, ddof=1)
    valid = (len(worlds) >= 2) & np.isfinite(draws).all(0) & (sd > 1e-14)
    critical = (float(np.quantile(np.abs((draws[:, valid]-point[valid])/sd[valid]).max(1), .95))
                if valid.any() else None)
    rows = []
    for i, metric in enumerate(('delay_crps_difference', 'upstream_rate_distance_difference')):
        for j, h in enumerate(HS):
            k = i*len(HS)+j
            rows.append(dict(metric=metric, horizon_s=h, mean=float(point[k]),
                low=float(point[k]-critical*sd[k]) if valid[k] else None,
                high=float(point[k]+critical*sd[k]) if valid[k] else None,
                uncertainty_available=bool(valid[k]),
                units='veh-s' if i == 0 else 'probability_fraction',
                candidate_rate=float(cp[:, j].mean()) if i else None,
                baseline_rate=float(bp[:, j].mean()) if i else None,
                real_rate=float(rp[:, j].mean()) if i else None))
    return dict(worlds=worlds, status='DIAGNOSTIC_ONLY', comparisons=rows,
        simultaneous_family_size=6, bootstrap_maxstat_critical=critical,
        all_three_crps_point_improvements=bool((delay_point < 0).all()),
        upstream_rate300_point_closer=bool(rate_point[list(HS).index(300)] < 0),
        hdv60_position_error=dict(status='MISSING_ENDPOINT_EVIDENCE', mean=None, low=None, high=None),
        switching_eligibility='NOT_ESTABLISHED')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', nargs='+', required=True)
    parser.add_argument('--baseline', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if not out.parent.is_dir() or out.exists():
        raise ValueError('new report in an existing parent required')
    real_path = Path(__file__).resolve().parents[1]/'rbeval/real_observables.json'
    real = load(real_path)['episodes']
    candidate, ca = read_predictions([Path(p) for p in args.candidate], real, partial=True)
    baseline, ba = read_predictions([Path(p) for p in args.baseline], real, partial=True)
    if ba['model'] is None or ba['model']['model'] != 'wm2':
        raise ValueError('baseline must be the registered WM2 inference contract')
    common = sorted(set(ca['complete_worlds']) & set(ba['complete_worlds']))
    bootstrap = WorldBootstrap(real, 10000)
    summaries = {}
    for split, worlds in (
            ('development12', [s for s in common if s < 969432]),
            ('reserved8', [s for s in common if s >= 969432])):
        summaries[split] = summarize_features(paired_features(candidate, baseline, real, worlds), worlds, bootstrap)
    out.mkdir()
    save_new(out/'COMPARISON.json', dict(schema='wmphys-paired-comparison/1',
        real_receipt_sha256=digest(real_path), source_sha256=digest(Path(__file__)),
        candidate_audit=ca, baseline_audit=ba, summaries=summaries,
        bootstrap=10000, seed=20261031, stratified='demand x penetration x split',
        uncertainty='whole worlds; conditional on saved K8 ensembles; no finite-K or model-error qualification',
        qualification='diagnostic; missing endpoints and sparse cells cannot establish switching eligibility'))
    print(json.dumps({k:dict(worlds=v['worlds'], status=v['status']) for k,v in summaries.items()}))


if __name__ == '__main__':
    main()
