"""E1R pool recovery: deterministic replay of the A75 S2b candidate generator (ZERO WM rollouts, ZERO SUMO).

The S2b search (a75ss.search_cell over strategy_search 02e119c1) draws every candidate from ONE random.Random(seed)
stream. Generation 0 (rule seeds + dense seeds + n_init random plans) depends on the seed only. In generations 1-4
each child is SS.mutate(rng, rng.choice(parents), ...) where `parents` depends on WM scores we do not re-run. The
replay therefore treats (len(parents), parent identity) as unknowns: for each child it tries every previously
evaluated strategy as the parent and accepts one whose output key equals the NEXT key in the recorded `screened`
order (16-hex sha256 strategy keys => a match is a verification, not a guess). Children that the search filtered
(mutate -> None, duplicate key, < min_roles, banned) produce no screened key; those branches are explored by DFS.
Output: full strategy definitions for all screened keys, each verified by strategy_key(strategy) == recorded key.

usage: python -I replay_pool.py SEARCH_JSON OUT_JSON
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'src_ref' / 'tree'))
sys.setrecursionlimit(100000)

import a75ss as A  # noqa: E402
from tools.wzprop import strategy_search as SS  # noqa: E402

LOG_TOTALS = (229, 322, 416, 512, 607)   # 'evaluated' after gen 0..4 (LOG_D2800_p60.jsonl)
N_SCREEN_POOL = 18   # valid D2800_p60 states in data/regime_STATES.json (count only; consumed by rng.sample)


def main(src, out):
    res = json.loads(Path(src).read_text())
    cfg = res['config']
    exp = [r['key'] for r in res['screened']]
    A.install(cfg)
    SS.UNCOND, SS.F_VS_BASE = bool(cfg.get('unconditional')), bool(cfg.get('f_vs_base'))
    SS.NOV_W = float(cfg.get('novelty_weight', 0.0))
    SS.MIN_ROLES = int(cfg.get('min_roles', 0))
    SS.LANE2_EXTRA_S = float(cfg.get('lane2_extra_margin_s', 0.0))
    SS.ONSET_MAX = cfg.get('onset_max')
    SS.SEARCH_GRID = cfg.get('search_grid')
    assert not cfg.get('extra_seeds')
    rng = random.Random(cfg['seed'])
    rng.sample(list(range(N_SCREEN_POOL)), min(N_SCREEN_POOL, int(cfg['n_screen_states'])))
    dense, hold_only = bool(cfg['dense']), bool(cfg['hold_only'])
    pop = [s for s in SS.seeds() if not (hold_only and any(r['lc_target'] is not None for r in s['roles']))]
    if dense:
        pop += SS.dense_seeds()
    if SS.ONSET_MAX is not None:
        pop = [s for s in pop if all(int(r['onset_s']) <= int(SS.ONSET_MAX) for r in s['roles'])]
    if SS.SEARCH_GRID:
        pop = [s for s in pop if all(r[k] in SS.SEARCH_GRID[k] for r in s['roles'] for k in SS.SEARCH_GRID)]
    rc = tuple(cfg['role_counts']) if cfg.get('role_counts') else None
    pop += [SS.random_strategy(rng, cfg['max_roles'], dense, hold_only, rc) for _ in range(int(cfg['n_init']))]

    evaluated = {}       # key -> strategy (insertion order = screened order)

    def admit(c, batch):
        """run() filters for one candidate; returns key if it would be newly evaluated, else None."""
        c = SS.uncond(c)
        if c is None or len(c['roles']) < SS.MIN_ROLES:
            return None, None
        k = SS.strategy_key(c)
        if k in evaluated or SS.banned(c):
            return None, None
        return k, c                         # duplicate inside one batch: run() overwrites (position of first,
                                            # strategy/name of last), reproduced by the dict assignment below

    batch = {}
    for c in pop:
        k, cc = admit(c, batch)
        if k is not None:
            batch[k] = cc
    for k, c in batch.items():
        evaluated[k] = c
    n0 = len(evaluated)
    if list(evaluated) != exp[:n0]:
        raise SystemExit(f'generation 0 mismatch: {n0} keys; first diff at '
                         f'{next(i for i, (a, b) in enumerate(zip(evaluated, exp)) if a != b)}')
    print('gen0 ok', n0, flush=True)
    pos = n0
    nk, ni, pm = int(cfg['n_children']), int(cfg['n_immigrants']), cfg['p_mutate']

    for gen in range(int(cfg['generations'])):
        parents_pool = list(evaluated.values())
        # number of new keys this generation = keys until the next generation's block; unknown split -> solve jointly
        found = None
        budget = nk + ni - (LOG_TOTALS[gen + 1] - LOG_TOTALS[gen])   # filtered children this generation (LOG)
        for n_par in range(1, 80):
            sol = solve_gen(rng.getstate(), n_par, parents_pool, evaluated, exp, pos, nk, ni, pm, dense, hold_only,
                            rc, cfg, budget)
            if sol is not None:
                found = (n_par, sol)
                break
        if found is None:
            raise SystemExit(f'generation {gen + 1}: no consistent replay')
        n_par, (state, new) = found
        rng.setstate(state)
        for k, c in new:
            evaluated[k] = c
        pos += len(new)
        print(f'gen{gen + 1} ok n_parents={n_par} new={len(new)} total={len(evaluated)}', flush=True)
    keys = list(evaluated)
    if keys != exp:
        raise SystemExit(f'final mismatch {len(keys)} vs {len(exp)}')
    for k, c in evaluated.items():
        assert SS.strategy_key(c) == k
    Path(out).write_text(json.dumps(dict(schema='e1r.pool_replay.v1', source=str(src), n=len(keys),
                                         strategies=[dict(key=k, strategy=c) for k, c in evaluated.items()]),
                                    indent=1))
    print('ALL', len(keys), 'keys verified', flush=True)


def solve_gen(state, n_par, pool, evaluated, exp, pos, nk, ni, pm, dense, hold_only, rc, cfg, budget):
    """DFS over the nk children of one generation (+ ni immigrants); returns (rng state after, [(key, strat)])."""
    rng = random.Random()
    gen_end_limit = len(exp)

    def immigrants(st, pos_, batch, left):
        rng.setstate(st)
        out = []
        for _ in range(ni):
            c = SS.random_strategy(rng, cfg['max_roles'], dense, hold_only, rc)
            c = SS.uncond(c)
            if c is None or len(c['roles']) < SS.MIN_ROLES:
                left -= 1
                continue
            k = SS.strategy_key(c)
            if k in evaluated or k in batch or SS.banned(c) or k in dict(out):
                left -= 1
                continue
            if pos_ + len(out) >= gen_end_limit or exp[pos_ + len(out)] != k:
                return None
            out.append((k, c))
        if left != 0:
            return None
        return rng.getstate(), out

    def dfs(j, st, pos_, batch, left):
        if j == nk:
            r = immigrants(st, pos_, batch, left)
            if r is None:
                return None
            st2, imm = r
            return st2, list(batch.items()) + imm
        rng.setstate(st)
        idx_state = None
        rng._randbelow(n_par)                 # rng.choice(parents) consumption (identity solved below)
        idx_state = rng.getstate()
        outcomes = {}                         # (key or None, rng state) -> (strategy)
        for p in pool:
            rng.setstate(idx_state)
            t = SS.mutate(rng, p, pm, dense, hold_only)
            st_after = rng.getstate()
            k = None
            c = None
            if t is not None:
                c = SS.uncond(t)
                if c is not None and len(c['roles']) >= SS.MIN_ROLES:
                    kk = SS.strategy_key(c)
                    if not (kk in evaluated or kk in batch or SS.banned(c)):
                        k = kk
            if k is not None and (pos_ >= len(exp) or exp[pos_] != k):
                continue                      # produces a key that is not the recorded next key: impossible
            outcomes.setdefault((k, st_after), c)
        # matched outcomes first, then filtered ones
        for (k, st_after), c in sorted(outcomes.items(), key=lambda kv: kv[0][0] is None):
            if k is not None:
                b2 = dict(batch)
                b2[k] = c
                r = dfs(j + 1, st_after, pos_ + 1, b2, left)
            else:
                if left <= 0:
                    continue
                r = dfs(j + 1, st_after, pos_, batch, left - 1)
            if r is not None:
                return r
        return None

    return dfs(0, state, pos, {}, budget)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
