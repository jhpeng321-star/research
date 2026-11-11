"""AMENDMENT 60, part 60B (S-WMSCORE), step 1: WM scores with the EXACT A42 (A47 run) search scoring - zero SUMO.

The A42 candidates (and the A45 diverse rows) come from the A47 search `final` list (STRATEGY_SEARCH_D2800_p60.json,
sha256 b8e451ee...). Their scores are stored there and are reused. This script computes the same `final`-stage score
for further plans with the unchanged search code:

    per, n = strategy_search.evaluate(wm, states, plans, K_final, H, groups=True)    # = the search's final call
    row    = strategy_search.summarize(per[key], lam, **sk)                            # score = row['score']

with the same code (code_v12r7b + the A47 execution-only split dispatch wm_backend), the same checkpoint, the same
start states (data/regime_STATES.json, valid rows of cell D2800_p60, file order; 18 <= n_final_states so all are
used, as in the search), K_final / H / lam / fairness settings read from the stored search `config`, the same WM
arguments (wma:WMA_READY.json, map_spec maps/wz3to2/MAP_SPEC.json, device cpu, max_rows 16, WZPROP_SPLIT_ACTIONS=4).
Every input is sha256-checked (fail closed). Identity plans (stored A42 rows) are scored in the same call and
compared with their stored scores by the caller (a60_score_analyse.py).

    python -I a60_wmscore.py --code-dir <code_split> --ckpt <wm_s.pt> --search <A47 search json>
                             --randplans <RANDPLANS_D2800_p60.json> --identity KEY [--identity KEY ...]
                             --workers N --out <new json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

SHA = dict(
    strategy_search='02e119c11a056c13bc54309c5f71e0ca8dd73aeb3c49c5a901e008fda36f54fc',
    wm_backend_split='728fc917cc1f8f9ca7f78f835871642b5143dbc5d4edf5fb4907ea25c07793a4',
    wma_ready='c7952b42b34b67e90bdfb85e368784cfbfc54e670cf6cc960f522351fdd38b1a',
    states='74aef8f8047a073e16097aa1e104fbf72ab32ca1dd16c7a53f4ad58c70a154e3',
    map_spec='386400b78d27e50a376ee28a5d18d25a5baeb27fd4bb0436fb171e5ccafd4e0d',
    ckpt='6da9d3337aad78799c2e6ac97c87e79b927346011f55861e98887bb89728c063',
    search='b8e451eeca45ad407606d72d00675f1862adf44634987ea55bed22e6761f582a',
    randplans='687663b078c768643d56758b42dc8113d297be1d606ecaf7994217964ec3dbaa',
)
CELL = 'D2800_p60'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check(name, path):
    got = sha(path)
    if got != SHA[name]:
        raise SystemExit(f'{name}: {path} sha256 {got[:12]} != {SHA[name][:12]}')
    return got


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--code-dir', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--search', required=True)
    p.add_argument('--randplans', required=True)
    p.add_argument('--identity', action='append', default=[])
    p.add_argument('--workers', type=int, required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args(argv)
    out = Path(a.out).resolve()
    if out.exists():
        raise SystemExit(f'{out}: exists (never overwritten)')
    code = Path(a.code_dir).resolve()
    ckpt = Path(a.ckpt).resolve()
    shas = dict(strategy_search=check('strategy_search', code/'tools'/'wzprop'/'strategy_search.py'),
                wm_backend_split=check('wm_backend_split', code/'tools'/'wzprop'/'wm_backend.py'),
                wma_ready=check('wma_ready', code/'WMA_READY.json'),
                states=check('states', code/'data'/'regime_STATES.json'),
                map_spec=check('map_spec', code/'maps'/'wz3to2'/'MAP_SPEC.json'),
                ckpt=check('ckpt', ckpt), search=check('search', a.search), randplans=check('randplans', a.randplans))
    search = json.loads(Path(a.search).read_text(encoding='utf-8'))
    rand = json.loads(Path(a.randplans).read_text(encoding='utf-8'))
    cfg = search['config']
    os.environ['WZPROP_SPLIT_ACTIONS'] = '4'
    os.chdir(code)
    sys.path[:0] = [str(code/'src'), str(code)]
    from tools.wzprop import strategy_search as SS
    from tools.wzprop import wm_backend as W
    if Path(SS.__file__).resolve() != (code/'tools'/'wzprop'/'strategy_search.py').resolve():
        raise SystemExit(f'imported {SS.__file__}')
    # module globals exactly as search_cell sets them (only LANE2_EXTRA_S / REC_W enter summarize)
    SS.UNCOND, SS.F_VS_BASE = bool(cfg.get('unconditional')), bool(cfg.get('f_vs_base'))
    SS.NOV_W = float(cfg.get('novelty_weight', 0.0))
    SS.MIN_ROLES = int(cfg.get('min_roles', 0))
    SS.LANE2_EXTRA_S = float(cfg.get('lane2_extra_margin_s', 0.0))
    SS.REC_W = float(cfg.get('rec_weight', 0.0))
    SS.ONSET_MAX = cfg.get('onset_max')
    SS.SEARCH_GRID = cfg.get('search_grid')
    lam = float(cfg['lam'])
    sk = dict(fair_tol=float(cfg['fair_tol']), bound_s=float(cfg['bound_s']), bound_stat=cfg['bound_stat'],
              fairer_total=cfg['fairer_total'], harm_cal=cfg['harm_cal'])
    states_all = [m for m in json.loads(Path('data/regime_STATES.json').read_text()) if m.get('valid')]
    st = [m for m in states_all if m['cell'] == CELL]
    if int(cfg['n_final_states']) and len(st) > int(cfg['n_final_states']):
        raise SystemExit('more states than n_final_states: the search would have sampled; not reproduced here')
    if len(st) != int(search['n_final_states']):
        raise SystemExit(f'{len(st)} states, search used {search["n_final_states"]}')
    final_by_key = {r['key']: r for r in search['final']}
    plans, roles = [], {}
    for k in a.identity:
        plans.append(dict(final_by_key[k]['strategy']))
        roles[k] = 'identity'
    for x in rand['plans']:
        s = dict(x['strategy'])
        if SS.uncond(s) != s or SS.strategy_key(s) != x['key']:
            raise SystemExit(f'{x["family"]}: not an unconditional strategy with key {x["key"]}')
        if x['key'] in final_by_key:
            raise SystemExit(f'{x["family"]}: already in the search final list')
        plans.append(s)
        roles[x['key']] = 'a56_random'
    kw = dict(checkpoint=str(ckpt), map_spec='maps/wz3to2/MAP_SPEC.json', device='cpu', max_rows=16)
    wm = W.ParallelWM('wma:WMA_READY.json', a.workers, api={}, local_kw=dict(kw, device='cpu'),
                      mem_fraction=None, **kw)
    t0 = time.time()
    per, n = SS.evaluate(wm, st, plans, int(cfg['K_final']), cfg['H'], groups=bool(cfg['fairness']))
    rows = []
    for s in plans:
        k = SS.strategy_key(s)
        rows.append(dict(strategy=s, key=k, family=s['name'], role=roles[k], descriptor=list(SS.descriptor(s)),
                         novelty=SS.novelty(s), banned=SS.banned(s), per_state=per[k], **SS.summarize(per[k], lam, **sk)))
    doc = dict(schema='coopmwm.a60.wmscore.v1', amendment='60B', cell=CELL, input_sha256=shas,
               settings=dict(K_final=int(cfg['K_final']), H=cfg['H'], lam=lam, groups=bool(cfg['fairness']), sk=sk,
                             n_states=len(st), state_seeds=[m['seed'] for m in st], wm_kwargs=kw,
                             split_actions=4, workers=a.workers),
               backend=wm.meta, rollouts=n, wall_s=round(time.time()-t0, 1), new_sumo_sim_s=0, rows=rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, default=str)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(out=str(out), sha256=sha(out), rollouts=n, wall_s=doc['wall_s'],
                          scores={r['family']: r['score'] for r in rows})))
    wm.pool.shutdown()


if __name__ == '__main__':
    main()
