"""AMENDMENT 56 (S-RANDCTRL60) plan draw - zero physics, no world model.

Draws the 12 random-control plans with the amendment-42/47 search's OWN random plan generator
(tools/wzprop/strategy_search.py random_strategy + uncond, code_v12r7b, sha256 02e119c1...) under the A47 search
config (search_grid, onset_max, dense, hold_only, role_counts, max_roles, unconditional, min_roles), RNG
random.Random(20261111), taking the first 12 DISTINCT draws that pass the static filters the A42 selection applies
before verification: not a comparator (wall_g_solo2 / solo2 keys), not banned (15 m/s hold), executable by the
registered A37 executor (tools.coopmwm.a37_cellbank.executor_refusal is None), >= min_roles roles.
Only the `config` entry of the search file is read (file sha256 checked); no WM rollout, no score.

    python ops/a56_draw.py --code-dir <code_v12r7b> --search <A47 STRATEGY_SEARCH_D2800_p60.json> --out <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

SEED = 20261111
N_PLANS = 12
MAX_DRAWS = 10_000
GENERATOR_SHA256 = '02e119c11a056c13bc54309c5f71e0ca8dd73aeb3c49c5a901e008fda36f54fc'
SEARCH_SHA256 = 'b8e451eeca45ad407606d72d00675f1862adf44634987ea55bed22e6761f582a'
CELL = 'D2800_p60'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_generator(code_dir):
    code_dir = Path(code_dir)
    gen = code_dir/'tools'/'wzprop'/'strategy_search.py'
    if sha(gen) != GENERATOR_SHA256:
        raise SystemExit(f'{gen}: sha256 is not the A47 search generator {GENERATOR_SHA256[:12]}')
    import tools
    tools.__path__.append(str(code_dir/'tools'))          # tools.wzprop from the search code copy (read only)
    from tools.wzprop import strategy_search as S
    if Path(S.__file__).resolve() != gen.resolve():
        raise SystemExit(f'imported {S.__file__}, expected {gen}')
    return S


def draw(S, cfg, seed=SEED, n=N_PLANS):
    from tools.coopmwm import a37_cellbank as C
    S.SEARCH_GRID = cfg['search_grid']
    S.ONSET_MAX = cfg['onset_max']
    S.UNCOND = bool(cfg['unconditional'])
    min_roles = int(cfg.get('min_roles', 0))
    rc = tuple(cfg['role_counts']) if cfg.get('role_counts') else None
    comp = {S.strategy_key(c) for c in S.COMPARATORS}
    rng = random.Random(seed)
    plans, seen, rejects, k = [], set(), [], 0
    while len(plans) < n:
        if k >= MAX_DRAWS:
            raise SystemExit('draw limit reached')
        k += 1
        s = S.uncond(S.random_strategy(rng, cfg['max_roles'], bool(cfg['dense']), bool(cfg['hold_only']), rc))
        key = S.strategy_key(s)
        why = ('min_roles' if len(s['roles']) < min_roles else 'comparator' if key in comp
               else 'banned' if S.banned(s) else C.executor_refusal(s) or ('duplicate' if key in seen else None))
        if why is not None:
            rejects.append(dict(draw=k, key=key, why=why))
            continue
        seen.add(key)
        if key != s['name'].split('_', 1)[1]:
            raise SystemExit(f'{s["name"]}: name/key mismatch')
        lanes = sorted({r['lane'] for r in s['roles']})
        plans.append(dict(random_rank=len(plans)+1, draw=k, family=s['name'], key=key, n_roles=len(s['roles']),
                          lanes=lanes, closing_lane_used=2 in lanes, descriptor=list(S.descriptor(s)), strategy=s,
                          strategy_sha256=C.spec_sha(s)))
    return plans, rejects, k


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--code-dir', required=True)
    p.add_argument('--search', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args(argv)
    if sha(a.search) != SEARCH_SHA256:
        raise SystemExit(f'{a.search}: not the A47 search file {SEARCH_SHA256[:12]}')
    cfg = json.loads(Path(a.search).read_text(encoding='utf-8'))['config']   # config only; nothing else is read
    S = load_generator(a.code_dir)
    plans, rejects, k = draw(S, cfg)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f'{out}: exists (one draw only; never overwritten)')
    doc = dict(schema='coopmwm.a56.randplans.v1', amendment='56', cell=CELL, seed=SEED, n_plans=N_PLANS,
               generator=dict(file='tools/wzprop/strategy_search.py (code_v12r7b, the A47 search code)',
                              sha256=GENERATOR_SHA256, calls='uncond(random_strategy(rng, max_roles, dense, hold_only, '
                                                             'role_counts)) with SEARCH_GRID / ONSET_MAX / UNCOND set'),
               search=dict(file_sha256=SEARCH_SHA256, read='config only'),
               config={x: cfg[x] for x in ('search_grid', 'onset_max', 'dense', 'hold_only', 'role_counts',
                                           'max_roles', 'unconditional', 'min_roles')},
               filters='not comparator, not banned, a37_cellbank.executor_refusal is None, >= min_roles, distinct',
               n_draws=k, rejects=rejects, wm_score_used=False, plans=plans)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(out=str(out), sha256=sha(out), n_draws=k, n_rejects=len(rejects),
                          families=[x['family'] for x in plans])))


if __name__ == '__main__':
    main()
