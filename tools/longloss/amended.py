"""Owner-amended verdict (REGISTRATION 21:38, before any S-LONGLOSSTEST score).

helps iff (a) |300 s mean paired dJ error| falls with world-bootstrap CI excluding 0, OR
(b) mean sign agreement over 120/180/300 s rises with world-bootstrap CI excluding 0.
Short horizons (10/30/60/90 s) are reported, not gating. Also a metering-slope check:
OLS slope of WM-predicted on real paired dJ at 300 s (world-cluster bootstrap CI).
Sign agreement follows horizon_gate.sign_agreement (pairs with real dJ == 0 dropped).
Usage: python amended.py wm1_gate.json candidate_gate.json [out.json]
"""
import json
import re
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0].rsplit('\\', 1)[0])
from paired_statistics import bias, compare  # noqa: E402

LONG, SHORT = ('120', '180', '300'), ('10', '30', '60', '90')
DRAWS, SEED = 2000, 0


def by_world(doc):
    out = {}
    for row in doc['rows']:
        m = re.match(r'w(\d+)m', row['pair_key'])
        assert m, row['pair_key']
        out.setdefault(int(m[1]), []).append(row)
    return out


def arrays(rows, h):
    return np.asarray([v for r in rows for v in r['dJ'][h]], float)


def sign_hits(rows, h):
    a = arrays(rows, h)
    keep = a[:, 1] != 0
    return (np.sign(a[keep, 0]) == np.sign(a[keep, 1])).astype(float)


def mean_long_sign(rows_list):
    vals = []
    for h in LONG:
        hits = np.concatenate([sign_hits(r, h) for r in rows_list])
        vals.append(hits.mean())
    return float(np.mean(vals))


def slope(rows_list):
    a = np.concatenate([arrays(r, '300') for r in rows_list])
    x, y = a[:, 1], a[:, 0]
    if x.std() == 0:
        return float('nan')
    return float(np.polyfit(x, y, 1)[0])


def boot(ref_w, cand_w, stat):
    keys = sorted(ref_w)
    assert keys == sorted(cand_w), 'scoring panel changed'
    point = stat([ref_w[k] for k in keys], [cand_w[k] for k in keys])
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(DRAWS):
        idx = rng.integers(len(keys), size=len(keys))
        draws.append(stat([ref_w[keys[i]] for i in idx], [cand_w[keys[i]] for i in idx]))
    draws = np.asarray(draws, float)
    draws = draws[np.isfinite(draws)]
    return point, np.quantile(draws, [.025, .975]).tolist()


def evaluate(reference, candidate):
    original = compare(reference, candidate)
    rw, cw = by_world(reference), by_world(candidate)
    sign_point, sign_ci = boot(rw, cw, lambda r, c: mean_long_sign(c) - mean_long_sign(r))
    slope_diff, slope_diff_ci = boot(rw, cw, lambda r, c: slope(c) - slope(r))
    s_ref, s_ref_ci = boot(rw, rw, lambda r, c: slope(r))
    s_cand, s_cand_ci = boot(cw, cw, lambda r, c: slope(r))
    a_ok = original['absolute_300s_bias_reduction'] > 0 and original['reduction_ci95'][0] > 0
    b_ok = sign_point > 0 and sign_ci[0] > 0
    per_world = {}
    for k in sorted(cw):
        a = arrays(cw[k], '300')
        b = arrays(rw[k], '300')
        if len(a) >= 3 and a[:, 1].std() > 0:
            per_world[k] = {'n': len(a), 'candidate_slope': float(np.polyfit(a[:, 1], a[:, 0], 1)[0]),
                            'wm1_slope': float(np.polyfit(b[:, 1], b[:, 0], 1)[0])}
    short = {h: {'wm1': reference['horizons'][h]['sign_agreement'],
                 'candidate': candidate['horizons'][h]['sign_agreement']} for h in SHORT + LONG}
    return {
        'original_rule': {k: original[k] for k in ('verdict', 'absolute_300s_bias_reduction',
                                                   'reduction_ci95', 'short_signs_pass',
                                                   'n_worlds', 'n_pair_groups')},
        'amended_rule': {
            'verdict': 'helps' if (a_ok or b_ok) else 'does_not_help',
            'a_300s_abs_bias_reduction': original['absolute_300s_bias_reduction'],
            'a_ci95': original['reduction_ci95'], 'a_pass': a_ok,
            'b_mean_long_sign_change': sign_point, 'b_ci95': sign_ci, 'b_pass': b_ok,
            'b_wm1_mean_long_sign': mean_long_sign(list(rw.values())),
            'b_candidate_mean_long_sign': mean_long_sign(list(cw.values())),
            'short_horizons_gating': False},
        'sign_agreement_by_horizon': short,
        'metering_slope_300s': {
            'definition': 'OLS slope of WM-predicted on real paired dJ at 300 s, pooled pairs; '
                          'world-cluster bootstrap 2000 draws seed 0',
            'wm1': s_ref, 'wm1_ci95': s_ref_ci, 'candidate': s_cand, 'candidate_ci95': s_cand_ci,
            'change': slope_diff, 'change_ci95': slope_diff_ci,
            'per_world_n_ge3': per_world},
        'paired_bias_and_spread': {'wm1': bias(reference), 'candidate': bias(candidate)},
        'bootstrap': {'unit': 'world_seed', 'draws': DRAWS, 'seed': SEED},
    }


if __name__ == '__main__':
    ref = json.load(open(sys.argv[1]))
    cand = json.load(open(sys.argv[2]))
    out = evaluate(ref, cand)
    text = json.dumps(out, indent=1)
    if len(sys.argv) > 3:
        open(sys.argv[3], 'w').write(text)
    print(text)
