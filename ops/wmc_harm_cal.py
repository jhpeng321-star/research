"""Fit the WM group-harm calibration on SUMO-verified families (zero SUMO, zero WM calls).

    python ops/wmc_harm_cal.py CAL_INPUT.json CAL_WM.json OUT.json [quantile]

Per group g: SUMO increase (s/veh, paired vs u0, WMVERIFY) ~ a + b * WM increase (least squares over families).
Conservative margin = the `quantile` (default 0.8) of the leave-one-out residuals (SUMO - LOO prediction), floored at 0.
Hold-out check: leave-one-out allowed (all calibrated groups <= 10 s/veh) vs SUMO ALLOWED, raw WM vs calibrated,
plus Spearman of calibrated vs SUMO lane-2 harm. Hold-only families (no lane-change role) are the fitting set.
"""
import json
import sys
from pathlib import Path

import numpy as np

GROUPS = ('lane0', 'lane1', 'lane2', 'hdv', 'cav')
BOUND = 10.0


def fit(x, y):
    A = np.c_[np.ones(len(x)), x]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0]), float(coef[1])


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main(cal_in, cal_wm, out, q=0.8):
    ci = json.loads(Path(cal_in).read_text())
    wm = json.loads(Path(cal_wm).read_text())['families']
    rows = []
    for r in ci['rows']:
        s = wm.get(r['family'], {}).get('summary') or {}
        if 'group_inc' not in s or r['spec'] is None:
            continue
        hold = all(x['lc_target'] is None for x in r['spec']['roles'])
        rows.append(dict(family=r['family'], hold_only=hold, sumo=r['sumo_inc'], wm=s['group_inc'],
                         sumo_allowed=r['allowed'], sumo_diff_vs_inc=r['sumo_diff_vs_inc'],
                         wm_gain_vs_inc=s['gain_vs_inc'], wm_feasible=s['feasible']))
    fitset = [r for r in rows if r['hold_only']]
    cal, loo = {}, {r['family']: {} for r in fitset}
    for g in GROUPS:
        x = np.array([r['wm'][g] for r in fitset])
        y = np.array([r['sumo'][g] for r in fitset])
        a, b = fit(x, y)
        res = []
        for i, r in enumerate(fitset):
            m = np.arange(len(fitset)) != i
            ai, bi = fit(x[m], y[m])
            loo[r['family']][g] = ai + bi * x[i]
            res.append(y[i] - loo[r['family']][g])
        margin = max(0.0, float(np.quantile(res, q)))
        cal[g] = dict(a=a, b=b, margin=margin, loo_rmse=float(np.sqrt(np.mean(np.square(res)))), n=len(fitset))
    check = []
    for r in fitset:
        raw_ok = all(r['wm'][g] <= BOUND for g in GROUPS)
        cal_ok = all(loo[r['family']][g] + cal[g]['margin'] <= BOUND for g in GROUPS)
        check.append(dict(family=r['family'], sumo_allowed=r['sumo_allowed'], wm_raw_allowed=raw_ok,
                          loo_cal_allowed=cal_ok, sumo_lane2=r['sumo']['lane2'], wm_lane2=r['wm']['lane2'],
                          loo_cal_lane2=loo[r['family']]['lane2'] + cal['lane2']['margin']))

    def acc(key):
        return dict(correct=sum(c[key] == c['sumo_allowed'] for c in check), n=len(check),
                    false_allowed=[c['family'] for c in check if c[key] and not c['sumo_allowed']],
                    false_refused=[c['family'] for c in check if not c[key] and c['sumo_allowed']])
    doc = dict(source=dict(cal_input=str(cal_in), cal_wm=str(cal_wm)), quantile=q, bound_s=BOUND, harm_cal=cal,
               holdout=dict(method='leave-one-out over hold-only verified families', raw=acc('wm_raw_allowed'),
                            calibrated=acc('loo_cal_allowed'),
                            spearman_lane2_raw=spearman([c['wm_lane2'] for c in check], [c['sumo_lane2'] for c in check]),
                            spearman_lane2_loo_cal=spearman([c['loo_cal_lane2'] for c in check],
                                                            [c['sumo_lane2'] for c in check]),
                            spearman_delay_vs_inc=spearman([-r['wm_gain_vs_inc'] for r in fitset],
                                                           [r['sumo_diff_vs_inc'] for r in fitset])),
               families=check, excluded_lane_change=[r['family'] for r in rows if not r['hold_only']])
    Path(out).write_text(json.dumps(doc, indent=1))
    print(json.dumps(dict(cal={g: {k: round(v, 2) for k, v in c.items()} for g, c in cal.items()},
                          holdout=doc['holdout']), indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], *(float(x) for x in sys.argv[4:]))
