"""A96 S5-p20 box check = A86 check_s5_gen0.py (d1239ddb) with ONE change: the S5 part (3) uses the run's own states
file and cell (D2800_p20, regime_STATES_p20.json 7860a5d1) instead of the bundle's D2800_p60 states. Parts (1)-(2)
(code identity vs S2b/S4 on the bundle's D2800_p60 states) are unchanged. CPU only, no WM, no GPU.
usage: python -B check_s5p20_gen0.py S2B_GEN0_KEYS.json S2B_CFG.json S4_CFG.json S5_CFG.json EXPECTED_S5P20_GEN0_SHA \
       P20_STATES.json"""
import hashlib
import json
import random
import sys
from collections import Counter

from tools.wzprop import strategy_search as SS
import a75ss as A
import a85s4 as S4
import a86s5 as S5


def gen0(mod, cfg, excl, states):
    cfg = dict(SS.DEFAULTS, **cfg)
    A.uninstall()
    A.install(cfg)
    mod.set_globals(cfg)
    SS.ONSET_MAX, SS.SEARCH_GRID = cfg.get('onset_max'), cfg.get('search_grid')
    flt = mod.LaneFilter(excl)
    _scr, pop, _rc = mod.initial_population(random.Random(cfg['seed']), states, cfg, flt)
    keys = {}
    for c in pop:
        c = SS.uncond(c)
        if c is None or len(c['roles']) < SS.MIN_ROLES:
            continue
        k = SS.strategy_key(c)
        if k not in keys and not SS.banned(c):
            keys[k] = c
    return cfg, flt, pop, keys


def load(p):
    raw = json.load(open(p))
    return raw.get('config', raw)


def main(keys_f, s2b_f, s4_f, s5_f, want, p20_f):
    states = [m for m in json.load(open('data/regime_STATES.json')) if m.get('valid') and m['cell'] == 'D2800_p60']
    rec = json.load(open(keys_f))['keys']
    _c, _f, _p, keys = gen0(S5, load(s2b_f), (), states)
    assert list(keys) == rec and len(rec) == 229, 'S2b gen-0 keys not reproduced'
    s4 = load(s4_f)
    _c, f4, _p, k4 = gen0(S4, s4, S4.excluded_sets(s4), states)
    _c, f5, _p, k5 = gen0(S5, s4, S5.excluded_sets(s4), states)
    assert list(k4) == list(k5) and len(k5) == 221 and f4.rejected == f5.rejected, 'S4 gen-0 differs'
    p20 = [m for m in json.load(open(p20_f)) if m.get('valid') and m['cell'] == 'D2800_p20']      # A96-S5P20
    assert len(p20) == 30, 'p20 states'                                                             # A96-S5P20
    s5 = load(s5_f)
    cfg, flt, pop, keys = gen0(S5, s5, S5.excluded_sets(s5), p20)                                   # A96-S5P20
    assert len(pop) == int(cfg['n_init']) and all(s['name'].startswith('rand_') for s in pop)
    assert all(0 in S5.lane_set(s) for s in pop)
    assert not any(S5.lane_set(s) in S5.excluded_sets(s5) for s in pop)
    assert S5.niching_cfg(s5) == dict(parents=True, finalists=True, top_cap=2)
    got = hashlib.sha256(json.dumps(list(keys)).encode()).hexdigest()
    assert want == 'PRINT' or got == want, f'S5-p20 gen-0 key sha {got} != {want}'
    niches = Counter(S5.niche_label(s) for s in pop)
    print('GEN0_OK s2b_keys_reproduced', len(rec), 's4_admitted', len(k5), 's5p20_pop', len(pop), 's5p20_admitted',
          len(keys), 'rejected_seeds', len(flt.rejected['seed']), 'rejected_random', flt.rejected['random'],
          'niches', len(niches), 'gen0_key_sha', got[:16])
    if want == 'PRINT':
        print('FULL_SHA', got)
        print('NICHES', json.dumps(dict(sorted(niches.items()))))


if __name__ == '__main__':
    main(*sys.argv[1:7])
