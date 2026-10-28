"""Registered CCT-1, CPU only; existing horizon-gate rows are the sole input.

Calibration/validation results are informational. Only an explicitly identified
pairs300 verdict panel can produce policy authority. No model selection/adoption.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

SCHEMA = 'gated300_cct1/1'
SIGMA = {90: 25.38, 120: 41.22, 180: 42.11, 300: 43.91}
BAD_PATH = re.compile(r'sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vtype', re.I)


def checked_path(value):
    path = Path(value)
    if BAD_PATH.search(str(path)) or BAD_PATH.search(str(path.resolve())):
        raise ValueError('restricted artifact path')
    return path


def world_key(row):
    key = row.get('world', row.get('pair_key', row.get('seed')))
    if key is None:
        raise ValueError('CCT-1 needs a world/seed/pair_key for cluster bootstrap')
    match = re.match(r'w(\d+)', str(key))
    return match.group(1) if match else str(key).split('@')[0]


def horizon_pairs(rows, horizon):
    groups = {}
    for row in rows:
        key = world_key(row)
        values = row.get('dJ', {}).get(str(horizon))
        if values is None:
            continue
        # Support both one arm [pred, real] and the gate's grouped [[pred, real], ...].
        if len(values) == 2 and all(isinstance(v, (float, int)) for v in values):
            values = [values]
        for pair in values:
            if (len(pair) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                     or not math.isfinite(v) for v in pair)):
                raise ValueError('dJ must contain finite [predicted, real] cost differences')
            groups.setdefault(key, []).append(pair)
    return groups


def sign_counts(pairs, threshold):
    a = np.asarray(pairs, dtype=float).reshape(-1, 2)
    confident = np.abs(a[:, 0]) > threshold
    signable = a[:, 1] != 0
    correct = np.sign(a[:, 0]) == np.sign(a[:, 1])
    return a, confident, signable, correct


def horizon_result(groups, sigma):
    threshold = 2 * sigma
    pairs = [p for group in groups.values() for p in group]
    a, confident, signable, correct = sign_counts(pairs, threshold)
    counted = confident & signable
    wrong = counted & ~correct
    agreement = float(correct[counted].mean()) if counted.any() else None
    coverage = float(confident.mean()) if len(a) else 0.0
    # No wrong calls means zero missed harm, not an undefined/failed condition.
    wrong_size = float(np.abs(a[wrong, 1]).mean()) if wrong.any() else 0.0
    # Bootstrap WORLD clusters, keeping all arms and starts of a sampled world.
    counts = []
    for group in groups.values():
        _, c, s, right = sign_counts(group, threshold)
        counts.append([int((c & s & right).sum()), int((c & s).sum())])
    draws = []
    rng = np.random.default_rng(0)
    counts = np.asarray(counts, dtype=int).reshape(-1, 2)
    for _ in range(2000):
        if not len(counts):
            break
        numerator, denominator = counts[rng.integers(0, len(counts), len(counts))].sum(0)
        if denominator:
            draws.append(float(numerator / denominator))
    ci = np.quantile(draws, [.025, .975]).tolist() if draws else [None, None]
    passed = agreement is not None and agreement >= .80 and coverage >= .25 and wrong_size <= threshold
    return dict(sigma=sigma, threshold=threshold, n_pairs=len(a), n_worlds=len(groups),
        n_confident=int(confident.sum()), n_confident_sign=int(counted.sum()),
        n_wrong=int(wrong.sum()), coverage=coverage, sign_agreement=agreement,
        sign_ci95=ci, wrong_mean_abs_real=wrong_size, verdict='PASS' if passed else 'FAIL',
        all_pairs_sign_agreement=float(correct[signable].mean()) if signable.any() else None)


def compute(document, *, panel, source_sha256=None):
    if panel not in ('calibration', 'validation', 'pairs300'):
        raise ValueError('unknown panel')
    role = document.get('role')
    if role in ('calibration', 'validation') and panel != role:
        raise ValueError('cannot relabel an informational panel as a verdict panel')
    wm = document.get('wm_sha256')
    if not isinstance(wm, str) or re.fullmatch('[0-9a-f]{64}', wm) is None:
        raise ValueError('CCT-1 requires the exact WM checkpoint SHA256')
    horizons = {str(h): horizon_result(horizon_pairs(document['rows'], h), sigma)
                for h, sigma in SIGMA.items()}
    trusted = 0
    for h in SIGMA:
        if horizons[str(h)]['verdict'] != 'PASS':
            break
        trusted = h
    return dict(schema=SCHEMA, wm_sha256=wm, panel=panel,
        policy_authorized=panel == 'pairs300', H_C=trusted, horizons=horizons,
        bootstrap=dict(unit='world', draws=2000, seed=0, ci=.95),
        zero_real_rule='exclude from sign counts only; coverage uses all pairs',
        source_sha256=source_sha256, entrant_control=document.get('entrant_control'),
        deterministic_crn=document.get('deterministic_crn'), samples=document.get('samples'),
        wm_config_sha256=document.get('wm_config_sha256'),
        legacy_threshold=document.get('threshold'), legacy_horizons=document.get('horizons', {}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('--panel', choices=('calibration', 'validation', 'pairs300'), required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    raw = checked_path(args.input).read_bytes()
    result = compute(json.loads(raw), panel=args.panel, source_sha256=hashlib.sha256(raw).hexdigest())
    with checked_path(args.out).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'legacy_horizons'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
