"""Build ARMS.json from finished STRATEGY_SEARCH_<cell>.json files (zero WM / SUMO calls).

    python ops/wmc_arms_from_search.py OUT.json STRATEGY_SEARCH_D3200_p60.json [...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.wzprop import strategy_search as SS  # noqa: E402

PROVENANCE = dict(checkpoint='6da9d333 (FAILED qualification A1-A5; proposal generator only, no WM-accuracy claim)',
                  code='code_v6 = port 7138ba9a + wzprop 9e1a8d01 (search); selection c4/wz-wm-propose b1927fef',
                  crn_fix='NOT landed: rows within one WM call are not CRN-paired',
                  run='gpuA strategy_v3_r2 (r1 OOM receipt kept)')


def main(out, paths, k=4):
    arms = {}
    for p in paths:
        res = json.loads(Path(p).read_text())
        cell = res['cell']
        by = {r['key']: r for r in res['final']}
        c = dict(SS.wm_arms(res, k), structured=SS.structured_arm(k), random=SS.random_arm(k),
                 reference=SS.reference_arm(), incumbent=SS.INCUMBENT, banned_comparator=SS.SOLO2,
                 banned_v_w=list(SS.BANNED_V_W), bound_s=SS.GROUP_BOUND_S,
                 best={a: (v or {}).get('strategy') for a, v in res['best'].items()},
                 best_beating_incumbent=(res['best_beating_incumbent'] or {}).get('strategy'),
                 comparators=res['comparators'], fairness=res['fairness'],
                 provenance=dict(PROVENANCE, evaluated=res['evaluated'], wm_rollouts=res['wm_rollouts'],
                                 n_final_states=res['n_final_states']))
        c['wm_predictions'] = {s['name']: {key: by[SS.strategy_key(s)].get(key) for key in (
            'gain_vs_base', 'gain_vs_base_lcb', 'gain_vs_inc', 'worst_group_inc', 'worst_group_inc_group',
            'cvar90_inc', 'bounded', 'feasible')}
            for g in ('wm', 'wm_fairer', 'wm_bounded', 'wm_fair') for s in c[g]}
        arms[cell] = c
        for g in ('wm', 'wm_fairer', 'wm_bounded'):
            print(cell, g, [(s['name'], c['wm_predictions'][s['name']]['gain_vs_base'],
                             c['wm_predictions'][s['name']]['worst_group_inc']) for s in c[g]])
    Path(out).write_text(json.dumps(arms, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
