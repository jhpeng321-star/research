"""A86 S5 search-side tests (CPU only, zero SUMO, zero GPU, StubWM fakes only). Extends tests/test_a85s4.py (S4).
Run: C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -I -B -m pytest -q -p no:cacheprovider <this dir>
"""
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SHIP = ROOT / 'ship'
TREE = ROOT / 'testtree'          # tools/ + data/regime_STATES.json (copied from the S4 prep; shas pinned below)
for p in (str(SHIP), str(TREE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from tools.wzprop import strategy_search as SS  # noqa: E402
from tools.wzprop import wm_backend as W  # noqa: E402
import a75ss as A  # noqa: E402
import a85s4 as S4  # noqa: E402
import a86s5 as S5  # noqa: E402

S2B_CFG = ROOT / 'cfg' / 'S2B_REF' / 'A75S2B_SEARCH_CFG.json'
S4_CFG = ROOT / 'cfg' / 'S4_REF' / 'A85S4_SEARCH_CFG.json'
S5_CFG = ROOT / 'cfg' / 'A86S5_SEARCH_CFG.json'
S2B_DIR = Path('E:/research-dense/runs/active/C4-V7FT-20261017/a75_s2b')
S2B_OUT = S2B_DIR / 'STRATEGY_SEARCH_D2800_p60.json'
A80 = Path('E:/research-dense/experiments/a82_p20_fourarm_20261113/ship/a80_s2b.py')
PINS = {
    SHIP / 'a75ss.py': 'cae54dc11be5f5220f19eb7399efac2699c7cdf363a90b7fa202766d5b906bc0',
    SHIP / 'a75_wmforms.py': '3af8f45d9b0a90d0ff3fb1e9931bb2a1af1f2d3b8759f6d2ba6c9c1908e476bd',
    SHIP / 'wm_fix_crn.py': '62e8e38f9ce7894d956314e6e030c03ed54f4becf1bef450da3c4b949960f46b',
    SHIP / 'a85s4.py': '7ca82976e7183ce205d0591470a60281b553c2d03e8643d479d739b86c419261',
    TREE / 'tools' / 'wzprop' / 'strategy_search.py': '02e119c11a056c13bc54309c5f71e0ca8dd73aeb3c49c5a901e008fda36f54fc',
    TREE / 'tools' / 'wzprop' / 'wm_backend.py': '728fc917cc1f8f9ca7f78f835871642b5143dbc5d4edf5fb4907ea25c07793a4',
    TREE / 'data' / 'regime_STATES.json': '74aef8f8047a073e16097aa1e104fbf72ab32ca1dd16c7a53f4ad58c70a154e3',
    S2B_CFG: '8f71eb5534093333c4fa042143daf4b66de81b727580975aed72ea398fc2491c',
    S4_CFG: '0a80c0f4c6a8858faf4430c8e55be04a9abc77779337cd91921e1df873b82f5e',
    A80: '48c5c448c2a85d4e2020def54cb17259979c89f2e0cc341829d6e99de0d0e289',
}
S2B_GEN0 = 229
EXCL5 = {frozenset({1}), frozenset({2}), frozenset({1, 2})}
SIZE_KEYS = ('n_init', 'generations', 'n_children', 'n_immigrants')
SEED_NAMES = 28                    # hand-library seeds left after hold_only / search_grid (all lack a lane-0 role)


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


def gen0(mod, cfg, excl):
    cfg = merged(cfg)
    A.install(cfg)
    mod.set_globals(cfg)
    SS.ONSET_MAX = cfg.get('onset_max')
    SS.SEARCH_GRID = cfg.get('search_grid')
    rng = random.Random(cfg['seed'])
    flt = mod.LaneFilter(excl)
    screen, pop, rc = mod.initial_population(rng, states(), cfg, flt)
    return cfg, rng, flt, pop, rc


def small(cfg, **kw):
    c = dict(cfg, n_init=24, generations=2, n_children=12, n_immigrants=6, n_screen_states=4, n_final_states=6,
             finalists=6, K_screen=2, K_final=2)
    c.update(kw)
    return c


def no_niching(cfg):
    return {k: v for k, v in cfg.items() if k != 'niching'}


def dumps(res, drop_sha=True):
    res = json.loads(json.dumps(res, sort_keys=True, default=str))
    if drop_sha and 'a85' in res:
        res['a85'].pop('a85s4_sha256', None)
        res['a85'].pop('a86s5_sha256', None)
    return json.dumps(res, sort_keys=True)


def has_lane0(s):
    return 0 in S5.lane_set(s)


@pytest.fixture(autouse=True)
def _clean():
    A.uninstall()
    yield
    A.uninstall()


# ------------------------------------------------------------------------------------------ inputs / cfg
def test_pinned_inputs():
    for p, h in PINS.items():
        assert sha(p) == h, p


def test_form_family_is_a80_verbatim():
    import ast
    def body(path):
        src = Path(path).read_text(encoding='utf-8')
        f = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == 'form_family'][0]
        return ast.get_source_segment(src, f).split('\n')[1:]
    assert body(A80) == body(SHIP / 'a86s5.py')


def test_cfg_differs_from_s4_only_in_registered_keys():
    a, b = load(S4_CFG), load(S5_CFG)
    diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    assert diff == sorted(['exclude_lane_sets', 'seed', 'niching'] + list(SIZE_KEYS))
    assert b['seed'] == 20261187 and b['exclude_lane_sets'] == [[1], [2], [1, 2]]
    assert S5.excluded_sets(b) == EXCL5
    assert S5.niching_cfg(b) == dict(parents=True, finalists=True, top_cap=2)
    # scoring / per-plan evaluation keys are untouched
    for k in ('H', 'K_screen', 'K_final', 'n_screen_states', 'n_final_states', 'finalists', 'lam', 'novelty_weight',
              'harm_cal', 'fair_tol', 'bound_s', 'bound_stat', 'fairer_total', 'n_elite', 'p_mutate'):
        assert a.get(k) == b.get(k), k
    # immigrant share not below S2b's (36 of 96)
    s2b = load(S2B_CFG)
    share = lambda c: c['n_immigrants'] / (c['n_children'] + c['n_immigrants'])  # noqa: E731
    assert share(b) >= share(s2b)


def test_lane_semantics():
    assert SS.field_speed([dict(edge='up', lane=0, x=SS.FIELD_X[1] - 1.0, v=3.0)]) == 3.0
    assert SS.field_speed([dict(edge='up', lane=1, x=SS.FIELD_X[1] - 1.0, v=3.0)]) is None
    assert all(S5.lane_set(c) == {2} for c in SS.COMPARATORS)


# ------------------------------------------------------------------------------------------ niching off == a85s4
def test_niching_off_reproduces_s4_and_s2b_generation0_keys():
    _c, _r, flt, pop, _ = gen0(S5, load(S2B_CFG), ())
    keys = list(admit_order(pop))
    rec = [r['key'] for r in json.loads(S2B_OUT.read_text())['screened']]
    assert len(keys) == S2B_GEN0 and keys == rec[:S2B_GEN0]
    s4 = load(S4_CFG)
    _c, _r, f4, pop4, _ = gen0(S4, s4, S4.excluded_sets(s4))
    A.uninstall()
    _c, _r, f5, pop5, _ = gen0(S5, s4, S5.excluded_sets(s4))
    assert list(admit_order(pop4)) == list(admit_order(pop5)) and len(admit_order(pop5)) == 221
    assert f4.rejected == f5.rejected


@pytest.mark.parametrize('which', ['s2b', 's4', 's5_without_niching'])
def test_niching_off_search_cell_identical_to_a85s4(which):
    cfg = {'s2b': lambda: load(S2B_CFG), 's4': lambda: load(S4_CFG),
           's5_without_niching': lambda: no_niching(load(S5_CFG))}[which]()
    cfg = small(merged(cfg))
    st = states()
    A.install(cfg)
    ref = S4.search_cell(W.StubWM(), st, dict(cfg))
    A.install(cfg)
    got = S5.search_cell(W.StubWM(), st, dict(cfg))
    assert 'a86' not in got
    assert dumps(ref) == dumps(got)


# ------------------------------------------------------------------------------------------ S5 filter
def test_s5_generation0_full_scale():
    cfg5 = load(S5_CFG)
    cfg, _rng, flt, pop, _rc = gen0(S5, cfg5, S5.excluded_sets(cfg5))
    assert len(pop) == cfg['n_init'] == 72
    assert all(s['name'].startswith('rand_') for s in pop)
    assert all(has_lane0(s) and S5.lane_set(s) not in EXCL5 for s in pop)
    assert len(flt.rejected['seed']) == SEED_NAMES and flt.rejected['random'] > 0
    assert {'wall_g_solo2', 'wall_g_duo12', 'g2_stagger40', 'duo12_v18'} <= set(flt.rejected['seed'])
    assert len(admit_order(pop)) == 72


def test_s5_children_and_immigrants_full_scale():
    cfg5 = load(S5_CFG)
    cfg, rng, flt, pop, rc = gen0(S5, cfg5, S5.excluded_sets(cfg5))
    pool = list(admit_order(pop).values())
    dense, hold_only = bool(cfg['dense']), bool(cfg['hold_only'])
    for _g in range(int(cfg['generations'])):
        parents = rng.sample(pool, 30)
        kids = [flt.child(rng, parents, cfg, dense, hold_only) for _ in range(int(cfg['n_children']))]
        kids += [flt.random(rng, cfg, dense, hold_only, rc) for _ in range(int(cfg['n_immigrants']))]
        assert len(kids) == int(cfg['n_children']) + int(cfg['n_immigrants']) == 32
        assert all(k is None or has_lane0(k) for k in kids)
        pool += list(admit_order(kids).values())


def _s5_stub(monkeypatch=None, **kw):
    cfg = small(merged(load(S5_CFG)), **kw)
    A.install(cfg)
    return cfg, S5.search_cell(W.StubWM(), states(), dict(cfg))


def test_s5_search_cell_with_stub_filter_and_comparators(monkeypatch):
    seen = []
    orig = SS.evaluate

    def spy(wm, sts, cands, K, H, **kw):
        seen.append(list(cands))
        return orig(wm, sts, cands, K, H, **kw)
    monkeypatch.setattr(SS, 'evaluate', spy)
    cfg, res = _s5_stub()
    comp = {SS.strategy_key(c) for c in SS.COMPARATORS}
    screened = [c for batch in seen[:-1] for c in batch]
    assert screened and all(has_lane0(c) for c in screened)                 # no comparator is screened in S5
    assert not any(SS.strategy_key(c) in comp for c in screened)
    finals = seen[-1]
    assert {SS.strategy_key(c) for c in finals} >= comp                     # comparators still scored in finals
    fc = [r for r in res['final'] if r['comparator']]
    assert sorted(r['strategy']['name'] for r in fc) == ['solo2', 'wall_g_solo2']
    assert all(has_lane0(r['strategy']) for r in res['final'] if not r['comparator'])
    assert set(res['comparators']) == {'solo2', 'wall_g_solo2'}
    a85 = res['a85']
    assert a85['generated_per_generation'] == [cfg['n_init']] + [cfg['n_children'] + cfg['n_immigrants']] * cfg['generations']
    assert a85['exclude_lane_sets'] == [[1], [1, 2], [2]] and a85['a86s5_sha256'] == sha(SHIP / 'a86s5.py')
    a86 = res['a86']
    assert a86['definition'] == S5.NICHE_DEFINITION and sum(a86['gen0_niches'].values()) == cfg['n_init']
    assert len(a86['parent_niches']) == cfg['generations']


def test_s5_search_deterministic():
    _c, r1 = _s5_stub()
    A.uninstall()
    _c, r2 = _s5_stub()
    assert dumps(r1, drop_sha=False) == dumps(r2, drop_sha=False)


# ------------------------------------------------------------------------------------------ niching: parents
def _row(name, lanes, form='other', score=0.0, feasible=1.0, comparator=False, banned=False, zone='full'):
    roles = []
    for i, ln in enumerate(lanes):
        r = dict(lane=ln, rank=0, v_w=20.0, dmax=1.0, hold_s=90, onset_s=0, lc_target=None)
        if form == 'gap' and i == 0:
            r['start'] = 'gap'
        if form == 'pulse' and i == 0:
            r['pulse_on_s'] = 10
        if form == 'rolling' and i == 1:
            r['onset_s'] = 20
        roles.append(r)
    s = dict(name=name, trigger='lanes', zone=zone, roles=roles)
    return dict(strategy=s, key=name, score=score, feasible=feasible, comparator=comparator, banned=banned,
                gain_vs_base=score, gain_vs_base_lcb=score)


def test_niche_order_round_robin_and_tiers():
    rows = [_row('a1', [0, 1], 'gap', 9), _row('a2', [0, 1], 'gap', 8), _row('b1', [0, 2], 'gap', 7),
            _row('a3', [0, 1], 'gap', 6), _row('c1', [0], 'pulse', 5), _row('b2', [0, 2], 'gap', 4)]
    got = [r['key'] for r in S5.niche_order(rows)]
    assert got == ['a1', 'b1', 'c1', 'a2', 'b2', 'a3']
    assert [r['key'] for r in S5.niche_order(rows, cap=1)] == ['a1', 'b1', 'c1']
    # activation tiers are kept: the non-activated block stays behind
    tier = {'a1': 1, 'a2': 1, 'b1': 1, 'a3': 0, 'c1': 0, 'b2': 0}
    got = [r['key'] for r in S5.niche_order(rows, tier=lambda r: tier[r['key']])]
    assert got == ['a1', 'b1', 'a2', 'a3', 'c1', 'b2']
    assert S5.niche_order([]) == []


def test_parents_span_niches(monkeypatch):
    """Each generation the elite slice (first n_elite parents) holds one plan per niche (as many niches as the
    activated tier offers), in the round-robin order of the ranked list; with niching off it is the plain top-N."""
    calls = []
    orig = S5.LaneFilter.child

    def spy(self, rng, parents, cfg, dense, hold_only):
        calls.append(list(parents))
        return orig(self, rng, parents, cfg, dense, hold_only)
    monkeypatch.setattr(S5.LaneFilter, 'child', spy)
    spans = {}
    for mode in ('on', 'off'):
        calls.clear()
        cfg = small(merged(load(S5_CFG)), n_init=48)
        if mode == 'off':
            cfg = no_niching(cfg)
        A.install(cfg)
        res = S5.search_cell(W.StubWM(), states(), dict(cfg))
        n_el = int(cfg['n_elite'])
        per_gen = [calls[g * cfg['n_children']] for g in range(cfg['generations'])]
        spans[mode] = [len({S5.niche(p) for p in ps[:n_el]}) for ps in per_gen]
        if mode == 'on':
            n_niche_avail = len({S5.niche_label(r['strategy']) for r in res['final']})
            assert all(s == min(n_el, s if n_niche_avail >= n_el else n_niche_avail) for s in spans['on'])
            assert all(s == n_el for s in spans['on'])                              # >= 6 niches available here
            assert res['a86']['parent_niches'] and all(len(x) >= n_el for x in res['a86']['parent_niches'])
        A.uninstall()
    assert sum(spans['on']) >= sum(spans['off'])


# ------------------------------------------------------------------------------------------ niching: TOP-12 cap
def test_capped_top_on_s2b_final():
    final = json.loads(S2B_OUT.read_text())['final']
    capped, glob = S5.capped_top(final, 12, 2)
    # the recomputed global top = the unchanged a75ss writer output of S2b
    rec = json.loads((S2B_DIR / 'TOP_OFFER_CONDITIONS_D2800_p60.json').read_text())
    assert [r['key'] for r in glob] == [r['strategy_key'] for r in rec]
    assert len(capped) == 12
    c = Counter(S5.niche(r['strategy']) for r in capped)
    assert max(c.values()) <= 2
    assert capped == S5.capped_top(final, 12, 2)[0]                              # deterministic
    elig = {r['key'] for r in final if not r['comparator'] and not r['banned'] and A.activated(r)}
    assert all(r['key'] in elig for r in capped)


def test_capped_top_cap_and_fill_order():
    final = [_row('x1', [0, 1], 'gap', 10), _row('x2', [0, 1], 'gap', 9), _row('x3', [0, 1], 'gap', 8),
             _row('cmp', [2], 'other', 7.5, comparator=True), _row('y1', [0], 'pulse', 7),
             _row('na', [0, 2], 'gap', 6.5, feasible=0.0), _row('y2', [0], 'pulse', 6), _row('z1', [0, 2], 'zone', 5,
                                                                                            zone='mid')]
    capped, glob = S5.capped_top(final, 12, 2)
    assert [r['key'] for r in glob] == ['x1', 'x2', 'x3', 'y1', 'y2', 'z1']
    assert [r['key'] for r in capped] == ['x1', 'y1', 'z1', 'x2', 'y2']             # x3 dropped by the cap
    capped, _ = S5.capped_top(final, 2, 2)
    assert [r['key'] for r in capped] == ['x1', 'y1']


def test_write_niche_top_files(tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    shutil.copyfile(S2B_OUT, out / 'STRATEGY_SEARCH_D2800_p60.json')
    shutil.copyfile(S2B_DIR / 'TOP_OFFER_CONDITIONS_D2800_p60.json', out / 'TOP_OFFER_CONDITIONS_D2800_p60.json')
    g_bytes = (out / 'TOP_OFFER_CONDITIONS_D2800_p60.json').read_bytes()
    capped, glob = S5.write_niche_top(out, 'D2800_p60', 12, 2)
    assert not (out / 'TOP_OFFER_CONDITIONS_D2800_p60.json').exists()
    assert (out / 'TOP_OFFER_CONDITIONS_GLOBAL_UNCAPPED_D2800_p60.json').read_bytes() == g_bytes
    rows = json.loads((out / 'TOP_OFFER_CONDITIONS_NICHECAP2_FOR_SUMO_D2800_p60.json').read_text())
    g_rows = json.loads(g_bytes)
    assert len(rows) == 12 and all(list(r) == list(g_rows[0]) for r in rows)    # same writer row keys
    assert [r['strategy_key'] for r in rows] == [r['key'] for r in capped]
    # rows of plans present in both lists are identical to the writer's rows
    by = {r['strategy_key']: r for r in g_rows}
    assert all(r == by[r['strategy_key']] for r in rows if r['strategy_key'] in by)
    doc = json.loads((out / 'NICHE_TOP_D2800_p60.json').read_text())
    assert doc['for_sumo'] == 'TOP_OFFER_CONDITIONS_NICHECAP2_FOR_SUMO_D2800_p60.json'
    assert max(Counter(r['niche'] for r in doc['niche_capped']).values()) <= 2


def _fake_main_factory(tmp_out):
    def fake(argv=None):
        o = Path(tmp_out)
        o.mkdir()
        shutil.copyfile(S2B_OUT, o / 'STRATEGY_SEARCH_D2800_p60.json')
        shutil.copyfile(S2B_DIR / 'TOP_OFFER_CONDITIONS_D2800_p60.json', o / 'TOP_OFFER_CONDITIONS_D2800_p60.json')
        fake.sc = A.search_cell
    return fake


@pytest.mark.parametrize('niching', [True, False])
def test_main_routes_and_post_processes(monkeypatch, tmp_path, niching):
    cfgf = tmp_path / 'cfg.json'
    cfg = load(S5_CFG) if niching else no_niching(load(S5_CFG))
    cfgf.write_text(json.dumps(cfg))
    out = tmp_path / 'out'
    fake = _fake_main_factory(out)
    monkeypatch.setattr(A, 'search_cell', A.search_cell)
    monkeypatch.setattr(A, 'main', fake)
    S5.main(['--states', 'x', '--cell', 'D2800_p60', '--config', str(cfgf), '--out', str(out), '--ckpt', 'y'])
    assert fake.sc is S5.search_cell
    names = sorted(p.name for p in out.iterdir())
    if niching:
        assert names == ['NICHE_TOP_D2800_p60.json', 'STRATEGY_SEARCH_D2800_p60.json',
                         'TOP_OFFER_CONDITIONS_GLOBAL_UNCAPPED_D2800_p60.json',
                         'TOP_OFFER_CONDITIONS_NICHECAP2_FOR_SUMO_D2800_p60.json']
    else:
        assert names == ['STRATEGY_SEARCH_D2800_p60.json', 'TOP_OFFER_CONDITIONS_D2800_p60.json']


def test_niching_cfg_validation():
    assert S5.niching_cfg({}) is None and S5.niching_cfg({'niching': None}) is None
    with pytest.raises(ValueError):
        S5.niching_cfg({'niching': {'definition': 'other', 'parents': True}})
    with pytest.raises(ValueError):
        S5.niching_cfg({'niching': {'definition': S5.NICHE_DEFINITION, 'typo': 1}})
