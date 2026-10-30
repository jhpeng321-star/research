"""Registered CCT-2 direction-aware confident-gain test, CPU only.

Pools the two held-out verdict panels (pairs300 + validation) of one exact WM;
selection/calibration panels are refused. CCT-1 stays reported unchanged.
Output is a WM-hash-bound authority file; no model selection/adoption.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re

import numpy as np

from tools.gated300.cct1 import SIGMA, checked_path, horizon_pairs

SCHEMA = 'gated300_cct2/1'
MIN_GAIN = 20
MIN_AGREEMENT = .80
H_G_HORIZONS = (120, 180, 300)
DRAWS = 2000


def _wm(document):
    wm = document.get('wm_sha256')
    if not isinstance(wm, str) or re.fullmatch('[0-9a-f]{64}', wm) is None:
        raise ValueError('CCT-2 requires the exact WM checkpoint SHA256')
    return wm


def direction_counts(pairs, threshold):
    """dJ = arm - native cost: negative predicted/real dJ means improvement."""
    a = np.asarray(pairs, dtype=float).reshape(-1, 2)
    gain = a[:, 0] < -threshold
    harm = a[:, 0] > threshold
    signable = a[:, 1] != 0
    gain_right = a[:, 1] < 0
    harm_right = a[:, 1] > 0
    return a, gain, harm, signable, gain_right, harm_right


def bootstrap_ci(groups, threshold):
    """World-cluster percentile interval of the gain sign agreement, seed 0."""
    counts = []
    for group in groups.values():
        _, gain, _, signable, right, _ = direction_counts(group, threshold)
        counts.append([int((gain & signable & right).sum()), int((gain & signable).sum())])
    draws = []
    rng = np.random.default_rng(0)
    counts = np.asarray(counts, dtype=int).reshape(-1, 2)
    for _ in range(DRAWS):
        if not len(counts):
            break
        numerator, denominator = counts[rng.integers(0, len(counts), len(counts))].sum(0)
        if denominator:
            draws.append(float(numerator / denominator))
    return np.quantile(draws, [.025, .975]).tolist() if draws else [None, None]


def horizon_result(groups, sigma):
    threshold = 2 * sigma
    pairs = [p for group in groups.values() for p in group]
    a, gain, harm, signable, gain_right, harm_right = direction_counts(pairs, threshold)
    counted = gain & signable
    wrong = counted & ~gain_right
    agreement = float(gain_right[counted].mean()) if counted.any() else None
    # No wrong gain calls means zero missed magnitude, not an undefined statistic.
    wrong_size = float(np.abs(a[wrong, 1]).mean()) if wrong.any() else 0.0
    harm_counted = harm & signable
    harm_wrong = harm_counted & ~harm_right
    harm_agreement = float(harm_right[harm_counted].mean()) if harm_counted.any() else None
    passed = int(gain.sum()) >= MIN_GAIN and agreement is not None and agreement >= MIN_AGREEMENT
    return dict(sigma=sigma, threshold=threshold, n_pairs=len(a), n_worlds=len(groups),
        n_gain=int(gain.sum()), n_gain_sign=int(counted.sum()),
        gain_sign_agreement=agreement, gain_sign_ci95=bootstrap_ci(groups, threshold),
        wrong_gain_mean_abs_real=wrong_size, verdict='PASS' if passed else 'FAIL',
        harm=dict(n_harm=int(harm.sum()), n_harm_sign=int(harm_counted.sum()),
            harm_sign_agreement=harm_agreement,
            wrong_harm_mean_abs_real=float(np.abs(a[harm_wrong, 1]).mean())
            if harm_wrong.any() else 0.0))


def pool(documents, horizon):
    """Merge every arm/start of both panels into shared world clusters."""
    groups = {}
    for document in documents:
        for key, group in horizon_pairs(document['rows'], horizon).items():
            groups.setdefault(key, []).extend(group)
    return groups


def compute(pairs300, validation, *, sources=None):
    if pairs300.get('role') != 'pairs300':
        raise ValueError('CCT-2 pools the pairs300 verdict panel; selection/calibration refused')
    if validation.get('role') != 'validation':
        raise ValueError('CCT-2 pools the validation verdict panel; selection/calibration refused')
    wm = _wm(pairs300)
    if _wm(validation) != wm:
        raise ValueError('both CCT-2 panels must name the same exact WM')
    if (pairs300.get('entrant_control') != validation.get('entrant_control')
            or validation.get('deterministic_crn') is False
            or pairs300.get('deterministic_crn') is False):
        raise ValueError('CCT-2 panels disagree on entrant/CRN inference')
    documents = (pairs300, validation)
    horizons = {}
    for horizon, sigma in SIGMA.items():
        pooled = horizon_result(pool(documents, horizon), sigma)
        pooled['panels'] = {
            name: horizon_result(horizon_pairs(document['rows'], horizon), sigma)
            for name, document in zip(('pairs300', 'validation'), documents)}
        horizons[str(horizon)] = pooled
    h_g = 0
    for horizon in H_G_HORIZONS:
        if horizons[str(horizon)]['verdict'] == 'PASS':
            h_g = horizon
    result = dict(schema=SCHEMA, wm_sha256=wm, panels=['pairs300', 'validation'],
        policy_authorized=True, H_G=h_g, horizons=horizons,
        bootstrap=dict(unit='world', draws=DRAWS, seed=0, ci=.95),
        pass_rule=f'|G_h| >= {MIN_GAIN} and sign agreement on G_h >= {MIN_AGREEMENT}',
        h_g_rule='largest passing h of 120/180/300; no contiguous-prefix requirement',
        zero_real_rule='exclude from sign counts only; |G_h| uses all confident-gain pairs',
        sources=sources or {}, entrant_control=pairs300.get('entrant_control'),
        deterministic_crn=pairs300.get('deterministic_crn'))
    objective = pairs300.get('objective_contract')
    if objective is not None or validation.get('objective_contract') is not None:
        if not isinstance(objective, dict) or objective != validation.get('objective_contract'):
            raise ValueError('CCT-2 panels disagree on objective contract')
        result['objective_contract'] = objective
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pairs300')
    parser.add_argument('validation')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    documents, sources = [], {}
    for name in ('pairs300', 'validation'):
        raw = checked_path(getattr(args, name)).read_bytes()
        documents.append(json.loads(raw))
        sources[name] = dict(path=str(getattr(args, name)),
                             sha256=hashlib.sha256(raw).hexdigest())
    result = compute(*documents, sources=sources)
    if result['wm_sha256'] not in str(checked_path(args.out)):
        raise ValueError('CCT-2 output filename must contain the full WM sha256')
    with checked_path(args.out).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    summary = {k: v for k, v in result.items() if k != 'horizons'}
    summary['horizons'] = {h: {k: v for k, v in row.items() if k != 'panels'}
                           for h, row in result['horizons'].items()}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
