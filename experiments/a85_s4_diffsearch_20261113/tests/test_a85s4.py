"""A85 S4 search-side tests (CPU only, zero SUMO, zero GPU, StubWM fakes only). Run:
  C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -I -B -m pytest -q -p no:cacheprovider <this dir>
"""
import hashlib
import json
import random
import sys
from pathlib import Path

import pytest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SHIP = ROOT / 'a85ship'
TREE = ROOT / 'testtree'          # tools/ + data/regime_STATES.json copied from the A75 box image (shas pinned below)
for p in (str(SHIP), str(TREE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from tools.wzprop import strategy_search as SS  # noqa: E402
from tools.wzprop import wm_backend as W  # noqa: E402
import a75ss as A  # noqa: E402
import a85s4 as S4  # noqa: E402

S2B_CFG = ROOT / 'cfg' / 'S2B_REF' / 'A75S2B_SEARCH_CFG.json'
S4_CFG = ROOT / 'cfg' / 'A85S4_SEARCH_CFG.json'
S2B_OUT = Path('E:/research-dense/runs/active/C4-V7FT-20261017/a75_s2b/STRATEGY_SEARCH_D2800_p60.json')
PINS = {
    SHIP / 'a75ss.py': 'cae54dc11be5f5220f19eb7399efac2699c7cdf363a90b7fa202766d5b906bc0',
    SHIP / 'a75_wmforms.py': '3af8f45d9b0a90d0ff3fb1e9931bb2a1af1f2d3b8759f6d2ba6c9c1908e476bd',
    SHIP / 'wm_fix_crn.py': '62e8e38f9ce7894d956314e6e030c03ed54f4becf1bef450da3c4b949960f46b',
    TREE / 'tools' / 'wzprop' / 'strategy_search.py': '02e119c11a056c13bc54309c5f71e0ca8dd73aeb3c49c5a901e008fda36f54fc',
    TREE / 'tools' / 'wzprop' / 'wm_backend.py': '728fc917cc1f8f9ca7f78f835871642b5143dbc5d4edf5fb4907ea25c07793a4',
    TREE / 'data' / 'regime_STATES.json': '74aef8f8047a073e16097aa1e104fbf72ab32ca1dd16c7a53f4ad58c70a154e3',
    S2B_CFG: '8f71eb5534093333c4fa042143daf4b66de81b727580975aed72ea398fc2491c',
}
S2B_GEN0 = 229                     # 'evaluated' after generation 0 in the S2b LOG_D2800_p60.jsonl
CLOSED = frozenset({1, 2})


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    raw = json.loads(Path(p).read_text(encoding='utf-8'))
    return raw.get('config', raw)


def merged(cfg):
    return dict(SS.DEFAULTS, **cfg)


def states(cell='D2800_p60'):
    d = json.loads((TREE / 'data' / 'regime_STATES.json').read_text())
    return [m for m in d if m.get('valid') and m['cell'] == cell]


def admit_order(pop, seen=None):
    """run()'s candidate filters (uncond, min_roles, not yet evaluated, not banned) -> keys in evaluation order."""
    seen = dict(seen or {})
    out = {}
    for c in pop:
        c = SS.uncond(c)
        if c is None or len(c['roles']) < SS.MIN_ROLES:
            continue
        k = SS.strategy_key(c)
        if k in seen or SS.banned(c):
            continue
        out[k] = c
    return out


def gen0(cfg, excl):
    cfg = merged(cfg)
    A.install(cfg)
    S4.set_globals(cfg)
    SS.ONSET_MAX = cfg.get('onset_max')
    SS.SEARCH_GRID = cfg.get('search_grid')
    rng = random.Random(cfg['seed'])
    flt = S4.LaneFilter(excl)
    screen, pop, rc = S4.initial_population(rng, states(), cfg, flt)
    return cfg, rng, flt, pop, rc


def small(cfg, **kw):
    c = dict(cfg, n_init=24, generations=2, n_children=12, n_immigrants=6, n_screen_states=4, n_final_states=6,
             finalists=6, K_screen=2, K_final=2)
    c.update(kw)
    return c


@pytest.fixture(autouse=True)
def _clean():
    A.uninstall()
    yield
    A.uninstall()


def test_pinned_inputs():
    for p, h in PINS.items():
        assert sha(p) == h, p


def test_cfg_differs_only_in_seed_and_filter():
    a, b = load(S2B_CFG), load(S4_CFG)
    assert sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k)) == ['exclude_lane_sets', 'seed']
    assert b['seed'] == 20261185 and b['exclude_lane_sets'] == [[1, 2]]
    assert S4.excluded_sets(b) == {CLOSED}


def test_lane_semantics():
    """lane 0 = closing lane (closing-lane field speed reads lane 0); the hand-written duo12 plans hold in {1, 2}."""
    assert SS.field_speed([dict(edge='up', lane=0, x=SS.FIELD_X[1] - 1.0, v=3.0)]) == 3.0
    assert SS.field_speed([dict(edge='up', lane=1, x=SS.FIELD_X[1] - 1.0, v=3.0)]) is None
    lib = {s['name']: s for s in SS.seeds() + SS.dense_seeds()}
    for n in ('wall_g_duo12', 'wall_g_duo12_gate0', 'duo12_v18', 'duo12_v19', 'duo12_v20', 'duo12_v21'):
        assert S4.lane_set(lib[n]) == CLOSED, n
    assert all(S4.lane_set(c) == {2} for c in SS.COMPARATORS)


def test_filter_disabled_reproduces_s2b_generation0_keys():
    cfg, _rng, flt, pop, _rc = gen0(load(S2B_CFG), ())
    keys = list(admit_order(pop))
    rec = [r['key'] for r in json.loads(S2B_OUT.read_text())['screened']]
    assert len(keys) == S2B_GEN0
    assert keys == rec[:S2B_GEN0]
    assert flt.rejected == dict(seed=[], random=0, mutation=0)


@pytest.mark.parametrize('which', ['s2b', 's4_without_filter'])
def test_filter_disabled_search_cell_identical_to_a75ss(which):
    cfg = load(S2B_CFG) if which == 's2b' else {k: v for k, v in load(S4_CFG).items() if k != 'exclude_lane_sets'}
    cfg = small(merged(cfg))
    st = states()
    A.install(cfg)
    ref = A.search_cell(W.StubWM(), st, dict(cfg))
    A.install(cfg)
    got = S4.search_cell(W.StubWM(), st, dict(cfg))
    assert 'a85' not in got
    assert json.dumps(ref, sort_keys=True, default=str) == json.dumps(got, sort_keys=True, default=str)


def test_s4_generation0_full_scale():
    cfg, _rng, flt, pop, _rc = gen0(load(S4_CFG), S4.excluded_sets(load(S4_CFG)))
    assert not any(S4.lane_set(s) == CLOSED for s in pop)
    _c2, _r2, _f2, pop_ref, _ = gen0(load(S4_CFG), ())
    n_seed_ref = len(pop_ref) - cfg['n_init']
    assert len(pop) == n_seed_ref - len(flt.rejected['seed']) + cfg['n_init']        # all 216 random plans kept
    assert sum(s['name'].startswith('rand_') for s in pop) == cfg['n_init']
    assert set(flt.rejected['seed']) >= {'wall_g_duo12', 'duo12_v18', 'duo12_v20'}
    assert flt.rejected['random'] > 0


def test_s4_children_and_immigrants_full_scale():
    """Generations 1-4 generator calls at the configured sizes (parents: any evaluated plans; no WM needed)."""
    cfg, rng, flt, pop, rc = gen0(load(S4_CFG), S4.excluded_sets(load(S4_CFG)))
    pool = list(admit_order(pop).values())
    dense, hold_only = bool(cfg['dense']), bool(cfg['hold_only'])
    for _g in range(int(cfg['generations'])):
        parents = rng.sample(pool, 30)
        kids = [flt.child(rng, parents, cfg, dense, hold_only) for _ in range(int(cfg['n_children']))]
        kids += [flt.random(rng, cfg, dense, hold_only, rc) for _ in range(int(cfg['n_immigrants']))]
        assert len(kids) == int(cfg['n_children']) + int(cfg['n_immigrants'])
        assert not any(k is not None and S4.lane_set(k) == CLOSED for k in kids)
        pool += [k for k in admit_order(kids).values()]
    assert flt.rejected['mutation'] > 0 and flt.rejected['random'] > 0


def test_s4_search_cell_with_stub(monkeypatch):
    cfg = small(merged(load(S4_CFG)))
    seen = []
    orig = SS.evaluate

    def spy(wm, sts, cands, K, H, **kw):
        seen.extend(cands)
        return orig(wm, sts, cands, K, H, **kw)
    monkeypatch.setattr(SS, 'evaluate', spy)
    A.install(cfg)
    res = S4.search_cell(W.StubWM(), states(), dict(cfg))
    comp = {SS.strategy_key(c) for c in SS.COMPARATORS}
    cands = [c for c in seen if SS.strategy_key(c) not in comp]
    assert cands and not any(S4.lane_set(c) == CLOSED for c in cands)
    assert not any(S4.lane_set(r['strategy']) == CLOSED for r in res['final'] if not r['comparator'])
    a85 = res['a85']
    n_seed = a85['generated_per_generation'][0] - cfg['n_init']
    assert n_seed > 0
    assert a85['generated_per_generation'] == [n_seed + cfg['n_init']] + \
        [cfg['n_children'] + cfg['n_immigrants']] * cfg['generations']
    assert a85['exclude_lane_sets'] == [[1, 2]] and a85['a85s4_sha256'] == sha(SHIP / 'a85s4.py')


def test_main_routes_through_s4(monkeypatch):
    called = {}
    monkeypatch.setattr(A, 'search_cell', A.search_cell)
    monkeypatch.setattr(A, 'main', lambda argv=None: called.setdefault('sc', A.search_cell))
    S4.main([])
    assert called['sc'] is S4.search_cell
