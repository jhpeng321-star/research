"""A85 S4 box check (CPU, no WM, no GPU; run from the code tree with PYTHONPATH=src:.:a85ship):
(1) filter OFF + S2b cfg reproduces the recorded S2b generation-0 keys (keys only are read from the S2b output);
(2) filter ON + S4 cfg: generation 0 has no plan with role lanes {1,2}; n_init random plans kept.
usage: python -B check_s4_gen0.py S2B_STRATEGY_SEARCH.json S4_CFG.json"""
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


def main(s2b_out, s4_cfg_path):
    res = json.load(open(s2b_out))
    s2b_cfg = {k: v for k, v in res['config'].items()}
    states = [m for m in json.load(open('data/regime_STATES.json')) if m.get('valid') and m['cell'] == 'D2800_p60']
    n = len(states)
    _c, _f, _p, keys = gen0(s2b_cfg, (), n)
    rec = [r['key'] for r in res['screened']][:len(keys)]
    assert len(keys) == 229 and list(keys) == rec, 'S2b gen-0 keys not reproduced'
    s4 = json.load(open(s4_cfg_path))
    cfg, flt, pop, keys4 = gen0(s4, S4.excluded_sets(s4), n)
    assert not any(S4.lane_set(s) == {1, 2} for s in pop)
    assert sum(s['name'].startswith('rand_') for s in pop) == int(cfg['n_init'])
    print('GEN0_OK s2b_keys_reproduced', len(rec), 's4_pop', len(pop), 's4_admitted', len(keys4),
          'rejected_seeds', len(flt.rejected['seed']), 'rejected_random', flt.rejected['random'])


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
