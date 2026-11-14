"""A96 S4-p20 box check (CPU, no WM, no GPU; run from the code tree with PYTHONPATH=src:.:a85ship). Copy of
a85/check_s4_gen0.py (e1ded2ab) for the p20 cell: states file and cell are arguments, and the filter-off reference is
the recorded S3b (p20, same a75ss code path) generation-0 keys instead of S2b (keys only are read).
(1) the S4-p20 cfg equals the S4 (p60) cfg byte for byte;
(2) filter OFF + S3b cfg + p20 states reproduces the recorded S3b generation-0 keys (229, as LOG line 1);
(3) filter ON + S4-p20 cfg + p20 states: generation 0 has no plan with role lanes {1,2}; n_init random plans kept.
usage: python -B check_s4p20_gen0.py S3B_STRATEGY_SEARCH.json S4P20_CFG.json STATES.json CELL S4_CFG.json"""
import json
import random
import sys

from tools.wzprop import strategy_search as SS
import a75ss as A
import a85s4 as S4


def gen0(cfg, excl, n_states):
    cfg = dict(SS.DEFAULTS, **cfg)
    A.uninstall()
    A.install(cfg)
    S4.set_globals(cfg)
    SS.ONSET_MAX, SS.SEARCH_GRID = cfg.get('onset_max'), cfg.get('search_grid')
    flt = S4.LaneFilter(excl)
    _scr, pop, _rc = S4.initial_population(random.Random(cfg['seed']), list(range(n_states)), cfg, flt)
    keys = {}
    for c in pop:
        c = SS.uncond(c)
        if c is None or len(c['roles']) < SS.MIN_ROLES:
            continue
        k = SS.strategy_key(c)
        if k not in keys and not SS.banned(c):
            keys[k] = c
    return cfg, flt, pop, keys


def main(s3b_out, cfg_path, states_path, cell, s4_cfg_path):
    assert open(cfg_path, 'rb').read() == open(s4_cfg_path, 'rb').read(), 'cfg differs from S4'
    res = json.load(open(s3b_out))
    s3b_cfg = {k: v for k, v in res['config'].items()}
    states = [m for m in json.load(open(states_path)) if m.get('valid') and m['cell'] == cell]
    n = len(states)
    assert n == 30, n
    _c, _f, _p, keys = gen0(s3b_cfg, (), n)
    rec = [r['key'] for r in res['screened']][:len(keys)]
    assert len(keys) == 229 and list(keys) == rec, 'S3b gen-0 keys not reproduced'
    s4 = json.load(open(cfg_path))
    cfg, flt, pop, keys4 = gen0(s4, S4.excluded_sets(s4), n)
    assert not any(S4.lane_set(s) == {1, 2} for s in pop)
    assert sum(s['name'].startswith('rand_') for s in pop) == int(cfg['n_init'])
    print('GEN0_OK cell', cell, 'states', n, 's3b_keys_reproduced', len(rec), 's4p20_pop', len(pop),
          's4p20_admitted', len(keys4), 'rejected_seeds', len(flt.rejected['seed']),
          'rejected_random', flt.rejected['random'])


if __name__ == '__main__':
    main(*sys.argv[1:6])
