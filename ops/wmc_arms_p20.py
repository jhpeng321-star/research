"""WM-C ARMS for the p20 setting (owner 2026-11-08 17:22; zero WM / SUMO calls).

Group size 1-6 CAV roles (not capped by penetration). One pick per group-size band (1 / 2 / 3 / 4-6) where a
feasible plan is predicted to beat u0, plus the best overall. D3200 = delay priority (score vs the incumbent),
D2800 = fairness priority (calibrated every-group increase <= 10 s/veh first). Also reports the WM feasibility
rate (share of start states where the plan's members exist) per group size over every screened plan.

    python ops/wmc_arms_p20.py OUT.json HARM_CAL.json D3200=SEARCH_D3200_p20.json D2800=SEARCH_D2800_p20.json
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, '.')
from tools.wzprop import strategy_search as SS  # noqa: E402

BANDS = (('1', (1,)), ('2', (2,)), ('3', (3,)), ('4-6', (4, 5, 6)))
N_RESERVE = 2


def eligible(r):
    s = r['strategy']
    return (not r.get('comparator') and not r.get('banned') and not SS.banned(s)
            and (r.get('feasible') or 0) > 0 and r['gain_vs_base'] > 0)


def cal(r, hc):
    if r.get('cal_group_inc'):
        return r['cal_group_inc'], bool(r['fair_cal'])
    ci = SS.calibrate(r['group_inc'], hc)
    return ci, all(v <= SS.GROUP_BOUND_S for v in ci.values())


def order(rows, mode):
    if mode == 'delay':
        return sorted(rows, key=lambda r: (not r['gain_vs_inc'] > 0, not r['gain_vs_base_lcb'] > 0,
                                           -r['score_vs_inc']))
    return sorted(rows, key=lambda r: (not (r['_fair'] and r['gain_vs_inc'] > 0), not r['_fair'],
                                       not r['gain_vs_base_lcb'] > 0, max(r['_ci'].values()), -r['score_vs_inc']))


def label(r, mode):
    a = ('beats incumbent' if r['gain_vs_inc'] > 0 else 'beats u0 only (not the incumbent)')
    b = 'LCB>0 vs u0' if r['gain_vs_base_lcb'] > 0 else 'LCB<=0 vs u0'
    return f"{mode}-priority; {a}; {b}; {'calibrated fair' if r['_fair'] else 'NOT calibrated fair'}"


def pick(res, hc, mode):
    rows = [r for r in res['final'] if eligible(r)]
    for r in rows:
        r['_ci'], r['_fair'] = cal(r, hc)
    ranked = order(rows, mode)
    out = []
    for band, sizes in BANDS:
        b = [r for r in ranked if len(r['strategy']['roles']) in sizes]
        if b:
            b[0]['_band'] = band
            out.append(b[0])
    if ranked and ranked[0] not in out:
        ranked[0]['_band'] = 'best-overall'
        out.append(ranked[0])
    elif ranked:
        ranked[0]['_band'] += '+best-overall'
    out.sort(key=ranked.index)
    reserve = [r for r in ranked if r not in out][:N_RESERVE]
    return out, reserve


def feasibility(res):
    by = defaultdict(list)
    for r in res.get('screened') or []:
        by[r['n_roles']].append(r)
    by_f = defaultdict(list)
    for r in res['final']:
        if not r.get('comparator'):
            by_f[len(r['strategy']['roles'])].append(r['feasible'])
    return {str(n): dict(n_screened=len(v), share_ever_feasible=round(sum(x['feasible'] > 0 for x in v) / len(v), 3),
                         mean_feasible_rate=round(sum(x['feasible'] for x in v) / len(v), 3),
                         share_beats_u0=round(sum(x['feasible'] > 0 and x['gain_vs_base'] > 0 for x in v) / len(v), 3),
                         finals_mean_feasible_rate=(round(sum(by_f[n]) / len(by_f[n]), 3) if by_f[n] else None))
            for n, v in sorted(by.items())}


def main(out, hc_path, *cells):
    hc = json.loads(Path(hc_path).read_text())['harm_cal']
    doc = {}
    for arg in cells:
        demand, path = arg.split('=', 1)
        res = json.loads(Path(path).read_text())
        mode = 'delay' if demand == 'D3200' else 'fair'
        picks, reserve = pick(res, hc, mode)
        pred = {r['strategy']['name']: dict(
            band=r.get('_band', 'reserve'), label=label(r, mode), roles=len(r['strategy']['roles']),
            lanes=sorted({x['lane'] for x in r['strategy']['roles']}), trigger=r['strategy']['trigger'],
            gate=r['strategy'].get('gate'), start_gate=r['strategy'].get('start_gate'),
            phases=r['strategy'].get('phases'), feasible=r.get('feasible'),
            gain_vs_base=r['gain_vs_base'], gain_vs_base_lcb=r['gain_vs_base_lcb'], gain_vs_inc=r['gain_vs_inc'],
            gain_vs_inc_lcb=r.get('gain_vs_inc_lcb'), group_inc_raw=r.get('group_inc'), group_inc_cal=r['_ci'],
            calibrated_fair=r['_fair']) for r in picks + reserve}
        inc = next((r for r in res['final'] if r['strategy']['name'] == SS.INCUMBENT['name']), None)
        meta = dict(source=str(path), source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                    search_states='p20 native TRAIN tapes (regime_STATES_p20.json sha 9dc83319)',
                    selection=f'{mode} priority; one pick per group-size band 1/2/3/4-6 + best overall',
                    n_run=len(picks), evaluated=res['evaluated'], wm_rollouts=res['wm_rollouts'],
                    n_states=res['n_states'], n_final_states=res['n_final_states'],
                    feasibility_by_group_size=feasibility(res), harm_cal=str(hc_path),
                    harm_cal_note='fitted on S-WMVERIFY D3200_p60 hold-only families (LOO 12/16); applied at p20 as is',
                    incumbent_pred=(dict(gain_vs_base=inc['gain_vs_base'], feasible=inc['feasible'],
                                         group_inc_cal=cal(inc, hc)[0]) if inc else None))
        doc[f'{demand}_p20'] = dict(wm=[r['strategy'] for r in picks], wm_reserve=[r['strategy'] for r in reserve],
                                    wm_predictions=pred, incumbent=SS.INCUMBENT, banned_comparator=SS.SOLO2,
                                    banned_v_w=list(SS.BANNED_V_W), bound_s=SS.GROUP_BOUND_S, provenance=meta)
        print(demand, mode, 'run', len(picks), 'reserve', len(reserve), 'inc', meta['incumbent_pred'])
        for n, v in pred.items():
            print('  %-24s %-14s roles=%d lanes=%s feas=%.2f gb=%.0f lcb=%.0f gi=%.0f worst_cal=%.1f fair=%s' % (
                n, v['band'], v['roles'], v['lanes'], v['feasible'], v['gain_vs_base'], v['gain_vs_base_lcb'],
                v['gain_vs_inc'], max(v['group_inc_cal'].values()), v['calibrated_fair']))
        for n, f in meta['feasibility_by_group_size'].items():
            print('  size', n, f)
    Path(out).write_text(json.dumps(doc, indent=1))
    print('ARMS sha256', hashlib.sha256(Path(out).read_bytes()).hexdigest())


if __name__ == '__main__':
    main(*sys.argv[1:])
