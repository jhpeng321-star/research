"""A75 search-side tests (CPU, zero SUMO, zero GPU). Run from anywhere:
  python -I -B -m pytest -q E:/research-dense/experiments/a75_holdforms/search/tests
"""
import copy
import json
import random
import sys
from pathlib import Path

import pytest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TREE = ROOT / 'ref' / 'code_v12r7b'
SHIP = ROOT / 'ship'
TESTENV = ROOT.parent / '_testenv'          # executor test environment (read only, other agent)
for p in (str(SHIP), str(TREE / 'src'), str(TREE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from tools.wzprop import strategy_search as SS  # noqa: E402
from tools.wzprop import wm_backend as W  # noqa: E402
import a75ss as A  # noqa: E402

CFG_DIR = ROOT / 'cfg'
OLD_CFGS = {
    'A47': Path('E:/research-dense/experiments/claim_validation_20261113_0231/inputs/A47_SEARCH_CFG.json'),
    'A68': Path('E:/research-dense/runs/active/C4-V7FT-20261017/a68_research60c/search/A68_SEARCH_CFG.json'),
    'A69': Path('E:/research-dense/runs/active/C4-V7FT-20261017/a68_research60c/search_lc/A69_SEARCH_L_CFG.json'),
    'A73': Path('E:/research-dense/runs/active/C4-V7FT-20261017/a73_search_p20/cfg/A73_SEARCH_L20_CFG.json'),
}


def load_cfg(p):
    raw = json.loads(Path(p).read_text(encoding='utf-8'))
    return dict(SS.DEFAULTS, **raw.get('config', raw))


def set_globals(cfg):
    SS.SEARCH_GRID = cfg.get('search_grid')
    SS.ONSET_MAX = cfg.get('onset_max')


def gen_stream(cfg, n=300):
    """Deterministic generator stream: random strategies, then mutation chains (the search's own calls)."""
    set_globals(cfg)
    rng = random.Random(cfg['seed'])
    dense, hold_only = bool(cfg['dense']), bool(cfg['hold_only'])
    rc = tuple(cfg['role_counts']) if cfg.get('role_counts') else None
    out = [SS.random_strategy(rng, cfg['max_roles'], dense, hold_only, rc) for _ in range(n)]
    seeds = [x for x in SS.seeds() if not (hold_only and any(r['lc_target'] is not None for r in x['roles']))]
    if SS.ONSET_MAX is not None:
        seeds = [x for x in seeds if all(int(r['onset_s']) <= int(SS.ONSET_MAX) for r in x['roles'])]
    if SS.SEARCH_GRID:
        seeds = [x for x in seeds if all(r[k] in SS.SEARCH_GRID[k] for r in x['roles'] for k in SS.SEARCH_GRID)]
    pool = list(out) + seeds
    kids = []
    for _ in range(n):
        k = SS.mutate(rng, rng.choice(pool), cfg['p_mutate'], dense, hold_only)
        kids.append(k)
        if k is not None:
            pool.append(k)
    return out + kids


@pytest.fixture(autouse=True)
def _clean():
    A.uninstall()
    yield
    A.uninstall()


@pytest.mark.parametrize('name', sorted(OLD_CFGS))
def test_old_cfg_byte_identical_candidates(name):
    cfg = load_cfg(OLD_CFGS[name])
    ref = gen_stream(cfg)
    ref_desc = [None if s is None else (SS.strategy_key(s), list(SS.descriptor(s)), SS.novelty(s)) for s in ref]
    A.install(cfg)
    got = gen_stream(cfg)
    got_desc = [None if s is None else (SS.strategy_key(s), list(SS.descriptor(s)), SS.novelty(s)) for s in got]
    assert json.dumps(ref, sort_keys=True) == json.dumps(got, sort_keys=True)
    assert ref_desc == got_desc
    assert sum(s is not None for s in got) > 300


def _states(cell='D2800_p60'):
    d = json.loads((TREE / 'data' / 'regime_STATES.json').read_text())
    return [m for m in d if m.get('valid') and m['cell'] == cell]


def small(cfg, **kw):
    c = dict(cfg, n_init=24, generations=2, n_children=12, n_immigrants=6, n_screen_states=4, n_final_states=6,
             finalists=6, K_screen=2, K_final=2)
    c.update(kw)
    return c


@pytest.mark.parametrize('name', ['A47', 'A68'])
def test_old_cfg_search_cell_identical_with_stub(name):
    cfg = small(load_cfg(OLD_CFGS[name]))
    st = _states()
    ref = SS.search_cell(W.StubWM(), st, dict(cfg))
    A.install(cfg)
    got = A.search_cell(W.StubWM(), st, dict(cfg))
    assert json.dumps(ref, sort_keys=True, default=str) == json.dumps(got, sort_keys=True, default=str)


def stage_cfg(name):
    return load_cfg(CFG_DIR / name)


def test_stage1_generator_widened_and_valid():
    cfg = stage_cfg('A75_SEARCH_CFG.json')
    A.install(cfg)
    got = [s for s in gen_stream(cfg, 400) if s is not None and len(s['roles']) >= cfg['min_roles']]
    seen = {k: set() for k in ('v_w', 'hold_s', 'onset_s')}
    zones, cds, nroles = set(), set(), set()
    for s in got:
        assert A.holdform_refusal(s) is None
        assert not A.is_extended(s) or s['zone'] in ('mid', 'merge')
        assert 2 <= len(s['roles']) <= 4
        assert all(r['lc_target'] is None for r in s['roles'])
        assert int(s['cooldown_s']) in cfg['cooldown_grid']
        assert int(s['cooldown_s']) >= A._need(s) // 2
        assert s['name'].split('_', 1)[1] == SS.strategy_key(s)
        for r in s['roles']:
            for k in seen:
                seen[k].add(r[k])
        zones.add(s['zone'])
        cds.add(s['cooldown_s'])
        nroles.add(len(s['roles']))
    assert seen['hold_s'] == set(cfg['search_grid']['hold_s'])
    assert seen['v_w'] == set(cfg['search_grid']['v_w'])
    assert seen['onset_s'] == set(cfg['search_grid']['onset_s'])
    assert zones == set(cfg['zones']) and cds == set(cfg['cooldown_grid']) and nroles == {2, 3, 4}
    keys = [SS.strategy_key(s) for s in got]
    assert len(set(keys)) == len({json.dumps({k: s[k] for k in s if k != 'name'}, sort_keys=True) for s in got})


def test_stage2_generator_forms_canonical_and_executable():
    cfg = stage_cfg('A75B_SEARCH_CFG.json')
    A.install(cfg)
    got = [s for s in gen_stream(cfg, 500) if s is not None]
    letters = set()
    for s in got:
        assert A.holdform_refusal(s) is None, s
        for r in s['roles']:
            assert r == A.canonical_role(r)
            if r.get('start') == 'gap':
                assert r['lane'] == 1 and r['hold_s'] in cfg['holdforms']['gap']['hold_s']
                assert r['gap_dx_lo'] < r['gap_dx_hi']
        if s.get('zone_d') is not None:
            assert s['zone'] == 'custom' and s['zone_d'][0] - s['zone_d'][1] >= 200.0
        letters |= set(A.form_letters(s))
        assert s['name'].split('_', 1)[1] == SS.strategy_key(s)
    assert letters == {'g', 'd', 't', 'p'}
    assert any(s.get('zone_d') for s in got)


def _executor():
    if not (TESTENV / 'tools' / 'coopmwm' / 'a75_holdforms.py').exists():
        pytest.skip('executor test environment not available')
    p = str(TESTENV)
    saved = {k: v for k, v in sys.modules.items() if k == 'tools' or k.startswith('tools.')}
    for k in saved:
        del sys.modules[k]
    sys.path.insert(0, p)
    try:
        import tools.coopmwm.a75_holdforms as H
        return H
    finally:
        sys.path.remove(p)
        for k in [k for k in sys.modules if k == 'tools' or k.startswith('tools.')]:
            sys.modules['_exec_' + k] = sys.modules.pop(k)
        sys.modules.update(saved)


def test_keys_refusals_and_bank_match_executor():
    H = _executor()
    for name in ('A75_SEARCH_CFG.json', 'A75B_SEARCH_CFG.json'):
        cfg = stage_cfg(name)
        A.install(cfg)
        SS.UNCOND = True
        got = [SS.uncond(s) for s in gen_stream(cfg, 250) if s is not None]
        for s in got:
            assert H.executor_refusal(s) is None
            assert H.strategy_key(H.canonical(s)) == SS.strategy_key(s)
            assert H.is_extended(s) == A.is_extended(s)
        plans = [dict(family=f'p{i}', strategy=dict(s, name=f'p{i}')) for i, s in enumerate(got[:60])]
        doc = H.bank_doc(plans, 'D2800_p60')
        adm, refused = H.a75_bank_plans(doc, 'D2800_p60')
        assert not refused and len(adm) == len(plans)
        for p, s in zip(sorted(adm, key=lambda p: int(p['family'][1:])), got[:60]):
            assert p['key'] == SS.strategy_key(s)
        conds = [t['offer_condition'] for t in doc['clean5c']['templates']]
        assert conds == [A.offer_condition(dict(s, name=f'p{i}')) for i, s in enumerate(got[:60])]
        A.uninstall()
    # refusal mirror on hand-made bad specs
    base = dict(trigger='lanes', zone='far', spread_m=300.0, cooldown_s=60, slots=1, gate=None, start_gate=None,
                phases=None, roles=[dict(lane=1, rank=0, v_w=18.0, dmax=1.5, hold_s=10, onset_s=0, lc_target=None)])
    bads = [dict(base, zone='custom'), dict(base, zone='nowhere'),
            dict(base, roles=[dict(base['roles'][0], start='gap')]),
            dict(base, roles=[dict(base['roles'][0], lane=2, start='gap', gap_dx_lo=0.0, gap_dx_hi=10.0, gap_arm_s=5)]),
            dict(base, roles=[dict(base['roles'][0], taper_in_s=8, taper_out_s=8)]),
            dict(base, roles=[dict(base['roles'][0], pulse_on_s=5)]),
            dict(base, roles=[dict(base['roles'][0], hold_s=0)]),
            dict(base, roles=[dict(base['roles'][0], lc_target=2, dv_rel=2.0)]),
            dict(base, zone='custom', zone_d=[100.0, 400.0])]
    for b in bads:
        assert A.holdform_refusal(b) == H.holdform_refusal(H.canonical(b)) and A.holdform_refusal(b) is not None


def test_example_plans_keys():
    ex = json.loads((ROOT.parent / 'EXAMPLE_PLANS.json').read_text())
    tpl = {t['family']: t for t in ex['bank']['clean5c']['templates']}
    for row in ex['bank']['rows']:
        c = tpl[row['family']]['offer_condition']
        roles = [A.canonical_role(dict(lane=r['sumo_lane'], rank=r['rank'], v_w=r['v_w'], dmax=r['dmax'],
                                       hold_s=r['hold_s'], onset_s=r['onset_s'], lc_target=r['lc_target_sumo'],
                                       **{k: r[k] for k in A.ROLE_DEFAULTS if k in r})) for r in c['roles']]
        s = dict(trigger=c['trigger'], zone=c['zone'], spread_m=c['spread_m'], cooldown_s=c['cooldown_s'],
                 slots=c['slots'], gate=c['gate'], start_gate=c['start_gate'], phases=c['phases'], roles=roles)
        if c.get('zone_d') is not None:
            s['zone_d'] = c['zone_d']
        assert A.strategy_key(s) == row['strategy_key'], row['family']
        assert A.holdform_refusal(s) is None


def test_member_specs_carry_form_and_plain_unchanged():
    cfg = stage_cfg('A75B_SEARCH_CFG.json')
    A.install(cfg)
    cavs = [dict(id=f'c{i}', edge='up', lane=ln, x=x, v=25.0) for i, (ln, x) in
            enumerate([(1, 2400.0), (2, 2380.0), (0, 2390.0), (1, 2300.0)])]
    s = dict(name='t', trigger='lanes', zone='merge', spread_m=300.0, cooldown_s=60, slots=1, gate=None,
             start_gate=None, phases=None,
             roles=[dict(lane=1, rank=0, v_w=0.0, dmax=1.5, hold_s=10, onset_s=0, lc_target=None, start='gap',
                         gap_dx_lo=-10.0, gap_dx_hi=40.0, gap_arm_s=60, dv_rel=4.0),
                    dict(lane=2, rank=0, v_w=18.0, dmax=1.0, hold_s=30, onset_s=20, lc_target=None)])
    act = SS.activation(s, cavs)
    assert [m['id'] for m in act] == ['c0', 'c1']
    assert act[0]['params']['form'] == dict(start='gap', gap_dx_lo=-10.0, gap_dx_hi=40.0, gap_arm_s=60, dv_rel=4.0)
    assert 'form' not in act[1]['params'] and act[1]['params']['start_s'] == 20
    plain = dict(s, zone='full', roles=[s['roles'][1]])
    A.uninstall()
    ref = SS.activation(dict(plain), cavs)
    A.install(cfg)
    assert SS.activation(dict(plain), cavs) == ref
    # zones mirror the executor windows
    assert A.zone_bounds(dict(zone='custom', zone_d=[1000.0, 400.0])) == (1696.0, 2296.0)
    assert A.zone_bounds(dict(zone='merge')) == (2096.0, 2596.0)


def test_rank_activated_first():
    cfg = stage_cfg('A75_SEARCH_CFG.json')
    A.install(cfg)
    act = dict(feasible=0.5, score=-50.0, score_vs_inc=-50.0, novelty=0.0)
    never = dict(feasible=0.0, score=0.0, score_vs_inc=0.0, novelty=0.0)
    for a in ('delay', 'D', 'F', 'bounded'):
        assert SS.rank_key(a)(act) < SS.rank_key(a)(never)
    A.uninstall()
    assert SS.rank_key('delay')(act) > SS.rank_key('delay')(never)      # the registered defect


def test_search_cell_stage_cfgs_with_stub_activated_first():
    st = _states()
    for name in ('A75_SEARCH_CFG.json', 'A75B_SEARCH_CFG.json'):
        cfg = small(stage_cfg(name))
        A.install(cfg)
        r1 = A.search_cell(W.StubWM(), st, copy.deepcopy(cfg))
        A.install(cfg)
        r2 = A.search_cell(W.StubWM(), st, copy.deepcopy(cfg))
        assert json.dumps(r1, sort_keys=True, default=str) == json.dumps(r2, sort_keys=True, default=str)
        flags = [A.activated(r) for r in r1['final']]
        assert flags == sorted(flags, reverse=True)
        for r in r1['final']:
            assert A.holdform_refusal(r['strategy']) is None
        A.uninstall()
