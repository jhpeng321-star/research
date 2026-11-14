"""SUMO-free tests of the claim-validation build (no simulation, no world model, no network, no registered tree).
Every number used as a 'result' below is a synthetic fixture made inside the test."""
from __future__ import annotations

import ast
import copy
import gzip
import inspect
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import box_driver  # noqa: E402
import cvlib as L  # noqa: E402
import e1_bank  # noqa: E402
import e1_order  # noqa: E402
import e1_pool  # noqa: E402
import e1_random_arm  # noqa: E402
import e1_verify  # noqa: E402
import e2_analyse  # noqa: E402
import e2_build  # noqa: E402
import e2_select  # noqa: E402
import e2_replay  # noqa: E402
import stats as S  # noqa: E402

GRID = dict(v_w=[18.0, 19.0, 20.0, 21.0, 22.0], dmax=[1.0, 1.5], hold_s=[90, 120, 180], onset_s=[0, 20])
CFG = dict(search_grid=dict(v_w=[18, 19, 20, 21, 22], dmax=[1.0, 1.5], hold_s=[90, 120, 180], onset_s=[0, 20, 40]),
           onset_max=30)
REAL_BANK = Path('E:/agent-work/a42-run/offload/root/a42_banks/A42_BIG_BANK_D2800_p60.json')

_OPENED, _WATCH = [], [False]


def _hook(event, args):
    if _WATCH[0] and event == 'open':
        _OPENED.append(str(args[0]))


sys.addaudithook(_hook)


# ------------------------------------------------------------------------------------------------ E1 pool
def test_grid_applies_onset_max():
    assert L.load_grid(CFG) == GRID


def test_pool_is_deterministic_and_seed_dependent():
    a, b = e1_pool.generate_pool(11, GRID), e1_pool.generate_pool(11, GRID)
    assert a == b
    c = e1_pool.generate_pool(23, GRID)
    assert [e['desc_hash'] for e in a['pool']] != [e['desc_hash'] for e in c['pool']]


@pytest.mark.parametrize('seed', L.SEARCH_SEEDS)
def test_pool_stratification_legality_and_distinctness(seed):
    got = e1_pool.generate_pool(seed, GRID)
    pool = got['pool']
    assert len(pool) == 120
    assert {n: sum(1 for e in pool if e['size'] == n) for n in L.SIZES} == {3: 30, 4: 30, 5: 30, 6: 30}
    assert len({e['desc_hash'] for e in pool}) == 120
    for e in pool:
        s = e['strategy']
        assert len(s['roles']) == e['size']
        assert L.static_legality(s, GRID) is None and L.executor_refusal(s) is None
        assert all(r['lc_target'] is None for r in s['roles'])                      # hold-only
        assert e['key'] == L.strategy_key(s) and e['desc_hash'] == L.desc_hash(s)
        assert s['name'] == f'e1s{seed}_{e["key"]}'
        assert s['trigger'] == 'lanes' or e['size'] == 3                            # a wall has at most 3 roles
    # generation order and every attempt are kept; accepted attempts are exactly the pool, in order
    acc = [a for a in got['attempts'] if a['status'] == 'accepted']
    assert [a['desc_hash'] for a in acc] == [e['desc_hash'] for e in pool]
    assert [e['index'] for e in pool] == list(range(120))
    assert all(a['reason'] for a in got['attempts'] if a['status'] == 'rejected')
    assert [e['size'] for e in pool[:4]] == [3, 4, 5, 6] or any(a['status'] == 'rejected' for a in got['attempts'][:4])


def test_dedupe_by_normalised_descriptor(monkeypatch):
    r = [L.make_role(0, 0, 18, 1.0, 90, 0), L.make_role(1, 0, 19, 1.5, 120, 20), L.make_role(2, 1, 20, 1.0, 180, 0)]
    a = L.make_strategy('a', 'lanes', 'far', 300.0, r)
    b = L.make_strategy('b', 'lanes', 'far', 300.0, list(reversed(r)))               # same plan, roles permuted
    assert L.desc_hash(a) == L.desc_hash(b) and L.strategy_key(a) != L.strategy_key(b)
    w1 = L.make_strategy('w', 'wall', 'far', 200.0, [L.make_role(i, 0, 18, 1.0, 90, 0) for i in range(3)])
    w2 = dict(w1, spread_m=300.0)                                                   # the wall trigger ignores spread
    assert L.desc_hash(w1) == L.desc_hash(w2)
    assert L.desc_hash(a) != L.desc_hash(dict(a, zone='near'))
    c = copy.deepcopy(a)
    c['roles'][0]['onset_s'] = 20
    assert L.desc_hash(a) != L.desc_hash(c)
    # the generator records a repeated draw as a rejected duplicate and keeps drawing
    real, n = e1_pool.draw, [0]

    def twice(rng, size, grid):
        n[0] += 1
        if n[0] == 2:
            return copy.deepcopy(twice.first)
        s = real(rng, size, grid)
        if n[0] == 1:
            twice.first = copy.deepcopy(s)
        return s
    monkeypatch.setattr(e1_pool, 'draw', twice)
    got = e1_pool.generate_pool(5, GRID, sizes=(3,), per_size=4)
    assert got['attempts'][1]['reason'] == 'duplicate' and len(got['pool']) == 4
    assert len({e['desc_hash'] for e in got['pool']}) == 4


def test_static_legality_reasons():
    r = [L.make_role(i, 0, 18, 1.0, 90, 0) for i in range(3)]
    ok = L.make_strategy('x', 'lanes', 'far', 300.0, r)
    assert L.static_legality(ok, GRID) is None
    assert L.static_legality(dict(ok, roles=r[:2]), GRID) == 'role_count'
    assert L.static_legality(dict(ok, roles=[dict(r[0], v_w=25.0)]+r[1:]), GRID) == 'v_w_off_grid'
    assert L.static_legality(dict(ok, roles=[dict(r[0], onset_s=40)]+r[1:]), GRID) == 'onset_s_off_grid'
    assert L.static_legality(dict(ok, roles=[dict(r[0], lc_target=1)]+r[1:]), GRID) == 'non_hold_plan'
    assert L.static_legality(dict(ok, roles=[r[0], r[0], r[2]]), GRID) == 'duplicate_role'
    assert L.static_legality(dict(ok, spread_m=400.0), GRID) == 'spread'
    assert L.static_legality(dict(ok, cooldown_s=10), GRID) == 'cooldown'
    assert L.static_legality(dict(ok, trigger='wall', roles=[dict(r[0], rank=1)]+r[1:]), GRID) == 'wall_rank'
    assert L.static_legality(dict(ok, gate=dict(x=1)), GRID) == 'gate_or_phases'


# ------------------------------------------------------------------------------------------------ Random arm
def test_random_arm_module_cannot_touch_files():
    tree = ast.parse(Path(e1_random_arm.__file__).read_text(encoding='utf-8'))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module)
    assert imports == {'hashlib', 'random'}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
            {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not names & {'open', 'Path', 'os', 'read_text', 'read_bytes', 'glob', 'listdir', 'exec', 'eval',
                        '__import__', 'importlib', 'subprocess', 'socket'}
    assert list(inspect.signature(e1_random_arm.random_order).parameters) == ['pool_hashes', 'seed']


def test_random_order_is_reproducible_and_order_free():
    hashes = [e['desc_hash'] for e in e1_pool.generate_pool(11, GRID)['pool']]
    _OPENED.clear()
    _WATCH[0] = True
    try:
        a = e1_random_arm.random_order(hashes, 11)
        b = e1_random_arm.random_order(list(reversed(hashes)), 11)
    finally:
        _WATCH[0] = False
    assert _OPENED == []                                   # the function opened no file at all
    assert a == b and sorted(a) == sorted(hashes) and a != hashes and a != sorted(hashes)
    assert e1_random_arm.random_order(hashes, 23) != a
    with pytest.raises(ValueError):
        e1_random_arm.random_order(hashes+hashes[:1], 11)


def _write_pools(tmp_path, seeds):
    cfg = tmp_path/'cfg.json'
    cfg.write_text(json.dumps(dict(config=CFG)), encoding='utf-8')
    out = tmp_path/'out'
    assert e1_pool.main(['--cfg', str(cfg), '--out', str(out), '--seeds']+[str(s) for s in seeds]) == 0
    return out


def test_random_command_reads_only_the_pool(tmp_path, capsys):
    out = _write_pools(tmp_path, [11])
    # decoys that a leaking implementation could pick up
    for name in ('WM_SCORES_seed11.json', 'VERIFY_RESULTS_seed11.json', 'CONFIRM_RESULTS.json'):
        for d in (out/'e1'/'wm_scores', out/'e1'/'order', out/'e1'/'pool', out):
            d.mkdir(parents=True, exist_ok=True)
            (d/name).write_text('{"poison": true}', encoding='utf-8')
    _OPENED.clear()
    _WATCH[0] = True
    try:
        assert e1_order.main(['random', '--out', str(out), '--seeds', '11', '--dry-run']) == 0
    finally:
        _WATCH[0] = False
    data = sorted({Path(p).name for p in _OPENED if str(tmp_path) in str(Path(p))})
    assert data == ['POOL_seed11.json']
    # and the result is exactly the pure function of (hashes, seed)
    hashes = [e['desc_hash'] for e in e1_pool.load_pool(out, 11)['pool']]
    assert e1_order.random_doc(out, 11)['order'] == e1_random_arm.random_order(hashes, 11)
    capsys.readouterr()


def test_pool_resume_keeps_identical_and_refuses_different(tmp_path, capsys):
    out = _write_pools(tmp_path, [11])
    p = e1_pool.pool_path(out, 11)
    before = p.read_bytes()
    _write_pools(tmp_path, [11])                           # resume: kept
    assert p.read_bytes() == before
    p.write_text(p.read_text(encoding='utf-8').replace('"cell"', '"cell" '), encoding='utf-8')
    with pytest.raises(L.Refused):
        _write_pools(tmp_path, [11])
    capsys.readouterr()


# ------------------------------------------------------------------------------------------------ WM arm
def test_wm_rank():
    assert e1_order.wm_rank(['c', 'a', 'b', 'd'], dict(a=1.0, b=3.0, c=None, d=1.0)) == ['b', 'a', 'd', 'c']
    with pytest.raises(L.Refused):
        e1_order.wm_rank(['a', 'b'], dict(a=1.0))


def _fake_scores(out, seed, tmp_path):
    """Synthetic two-stage WM scores (fixture). Screen score = -generation index, so the finalists are the first
    twelve drawn; final score = +index for those, so the WM top six is pool[11], pool[10], ..., pool[6]."""
    pool = e1_pool.load_pool(out, seed)
    rows = [dict(desc_hash=e['desc_hash'], key=e['key'], screen_summary=dict(score=-float(e['index'])),
                 screen_activated=True, finalist=e['index'] < 12, activated=(e['index'] != 7) if e['index'] < 12 else None,
                 summary=dict(score=float(e['index'])) if e['index'] < 12 else None) for e in pool['pool']]
    d = tmp_path/'scores'
    d.mkdir(exist_ok=True)
    (d/f'WM_SCORES_seed{seed}.json').write_text(json.dumps(dict(
        mode='two_stage', pool_doc_sha256=pool['doc_sha256'], search_seed=seed, wm_checkpoint_sha256=L.WM_CKPT_SHA256,
        rows=rows, finalist_hashes=[e['desc_hash'] for e in pool['pool'][:12]], cfg_sha256='x', wm_rollouts=0,
        counters=None, n_final_states=18, K=4, H=300, n_screen_states=4, K_screen=4, finalists=12)), encoding='utf-8')
    return d, pool


def test_two_stage_rank():
    rows = [dict(desc_hash=h, finalist=f, summary=dict(score=fs) if f else None, screen_summary=dict(score=ss))
            for h, f, fs, ss in (('a', True, 1.0, 9.0), ('b', True, 5.0, 8.0), ('c', False, None, 2.0),
                                 ('d', True, 5.0, 7.0), ('e', False, None, 3.0), ('f', True, 0.0, 6.0),
                                 ('g', True, -1.0, 5.0), ('h', True, 2.0, 4.0))]
    got = e1_order.wm_two_stage_rank(list('abcdefgh'), rows, list('abdfgh'))
    assert got == ['b', 'd', 'h', 'a', 'f', 'g', 'e', 'c']          # final score; tie b/d by screen rank; rest by screen
    with pytest.raises(L.Refused):
        e1_order.wm_two_stage_rank(list('abcdefgh'), rows, list('abdfg'))


def test_e1_chain_on_synthetic_results(tmp_path, capsys):
    seeds = [11, 23]
    out = _write_pools(tmp_path, seeds)
    for s in seeds:
        d, _ = _fake_scores(out, s, tmp_path)
    args = ['--out', str(out), '--seeds']+[str(s) for s in seeds]
    assert e1_order.main(['random']+args) == 0
    assert e1_order.main(['wm-jobs']+args) == 0
    assert e1_order.main(['wm-ingest', '--scores-dir', str(d)]+args) == 0
    assert e1_verify.main(['plan']+args) == 0
    plan = L.verify_seal(L.read_json(out/'e1'/'verify'/'VERIFY_PLAN.json'))
    assert plan['symbolic'] is True
    res_dir = tmp_path/'res'
    res_dir.mkdir()
    worlds = [f'w{i:02d}' for i in range(24)]
    groups = {g: 1.0 for g in S.CO_PRIMARY_GROUPS}
    for sp in plan['seeds']:
        pool = e1_pool.load_pool(out, sp['search_seed'])
        wm = L.read_json(e1_order.order_path(out, 'WM', sp['search_seed']))
        assert wm['top'] == [e['desc_hash'] for e in pool['pool'][11:5:-1]]         # fixture: final score = index
        assert wm['unscored'] == 1 and wm['mode'] == 'two_stage'                    # pool[7] never activated
        # a candidate chosen by both arms is ONE job set; every candidate has one wave of 24 worlds + one U0 wave
        assert sp['n_unique'] == 12-sp['n_shared'] == len(sp['waves'])-1
        keys = [k for w in sp['waves'] for k in w['cache_keys']]
        assert len(keys) == len(set(keys)) == 24*len(sp['waves'])
        u0 = {w: 1000.0+i for i, w in enumerate(worlds)}
        plans = {}
        for j, c in enumerate(sp['candidates']):
            gain = 50.0 if 'WM' in c['selected_by'] and c['selected_by']['WM'] in (1, 3) else -20.0
            plans[c['family']] = dict(J={w: u0[w]-gain-(i % 3) for i, w in enumerate(worlds)}, group_increase=groups)
        if sp['search_seed'] == 23:                                                 # a harmed group blocks admission
            first = next(c['family'] for c in sp['candidates'] if c['selected_by'].get('WM') == 1)
            plans[first]['group_increase'] = dict(groups, **{'origin:L2': 10.5})
        (res_dir/f'VERIFY_RESULTS_seed{sp["search_seed"]}.json').write_text(json.dumps(dict(u0=u0, plans=plans)),
                                                                            encoding='utf-8')
    assert e1_bank.main(['admit', '--results-dir', str(res_dir)]+args) == 0
    assert e1_bank.main(['build', '--confirm-worlds', '8']+args) == 0
    frozen = L.verify_seal(L.read_json(out/'e1'/'bank'/'FROZEN_BANKS.json'))
    rows = {(b['search_seed'], b['arm'], b['budget']): b for b in frozen['banks']}
    assert len(rows) == 2*2*3 and frozen['max_plans'] is None
    assert [rows[(11, 'WM', b)]['n_admitted'] for b in (2, 4, 6)] == [1, 2, 2]      # WM ranks 1 and 3 pass
    assert [rows[(23, 'WM', b)]['n_admitted'] for b in (2, 4, 6)] == [0, 1, 1]      # rank 1 blocked by the L2 group
    assert rows[(23, 'WM', 2)]['is_u0'] and rows[(23, 'WM', 2)]['file'] is None     # empty bank := U0
    for b in frozen['banks']:
        assert set(b['families']) <= {f for f in b['families']} and b['tested'] == b['tested'][:b['budget']]
        if not b['is_u0']:
            doc = L.read_json(out/'e1'/'banks'/b['file'])
            got = L.bank_strategies(doc)
            assert [p['family'] for p in got] == b['families']                      # executor priority order
            assert b['priority'] == sorted(b['priority'], reverse=True)
    # identical banks share one confirmation run; U0 banks have none
    names = [w['wave'] for w in frozen['confirm_waves']]
    assert len(names) == len(set(names)) == 1+len({b['sha256'] for b in frozen['banks']
                                                    if b['primary'] and not b['is_u0']})
    assert names[0] == 'CONF_U0'
    # the registered E1 primary: admitted count at budget 6, WM - Random, exact permutation
    pa = frozen['primary_admitted_count']
    assert pa['WM'] == [rows[(sd, 'WM', 6)]['n_admitted'] for sd in seeds]
    assert pa['wm_minus_random'] == [a-b for a, b in zip(pa['WM'], pa['RANDOM'])]
    assert pa['permutation'] == S.exact_paired_permutation(pa['wm_minus_random'])
    cw = [f'c{i}' for i in range(8)]
    confirm = dict(u0={w: 900.0+i for i, w in enumerate(cw)},
                   banks={b['sha256']: {w: 900.0+i-30.0*b['n_admitted'] for i, w in enumerate(cw)}
                          for b in frozen['banks'] if b['primary'] and not b['is_u0']})
    (res_dir/'CONFIRM_RESULTS.json').write_text(json.dumps(confirm), encoding='utf-8')
    assert e1_bank.main(['analyse', '--results-dir', str(res_dir)]+args) == 0
    an = L.verify_seal(L.read_json(out/'e1'/'bank'/'E1_ANALYSIS.json'))
    b6 = an['by_budget']['6']
    # expected from the frozen banks: every admitted plan is worth -30 s in the confirmation fixture
    exp = [-30.0*(rows[(sd, 'WM', 6)]['n_admitted']-rows[(sd, 'RANDOM', 6)]['n_admitted']) for sd in seeds]
    assert b6['primary'] and b6['wm_minus_random_per_seed'] == exp and all(x <= 0 for x in exp)
    for sd in seeds:
        if rows[(sd, 'RANDOM', 6)]['is_u0']:
            assert b6['bank_minus_u0'][f'RANDOM_seed{sd}']['note'] == 'bank is U0'
    assert b6['permutation']['assignments'] == 4
    assert b6['permutation']['p_less'] == S.exact_paired_permutation(exp)['p_less']
    # --max-plans is an explicit cap in executor order
    pool = e1_pool.load_pool(out, 11)
    adm = L.read_json(out/'e1'/'bank'/'ADMISSION_seed11.json')['verdicts']
    order = L.read_json(e1_order.order_path(out, 'WM', 11))['top']
    by = {e['desc_hash']: e for e in pool['pool']}
    full = e1_bank.bank_for(order, adm, by, 6)
    assert e1_bank.bank_for(order, adm, by, 6, max_plans=1) == full[:1]
    # resume: every stage re-run keeps its files
    assert e1_bank.main(['build', '--confirm-worlds', '8']+args) == 0
    # the box adapter finds the waves of one block
    ws = box_driver.find_waves(plan, 'e1v_seed11')
    assert len(ws) == len(plan['seeds'][0]['waves']) and ws[0]['kind'] == 'U0'
    assert len(box_driver.find_waves(frozen, 'conf')) == len(names)
    extra = dict(confirm_waves=[dict(frozen['confirm_waves'][0]), dict(wave='E2C_F', kind='BANK', block='conf')])
    assert len(box_driver.merge_waves([frozen, extra], 'conf')) == len(names)+1     # CONF_U0 kept once
    extra['confirm_waves'][0]['n_worlds'] = 9
    with pytest.raises(L.Refused):
        box_driver.merge_waves([frozen, extra], 'conf')
    assert box_driver.world_indices(ws[0], 24) == list(range(24))
    with pytest.raises(L.Refused):
        box_driver.world_indices(ws[0], 23)
    capsys.readouterr()


def test_seed_blocks_are_parameters_and_banned_ranges_refused(tmp_path):
    p = tmp_path/'blocks.json'
    p.write_text(json.dumps({'11': dict(start=962000, n=24)}), encoding='utf-8')
    with pytest.raises(L.Refused):
        L.load_seed_blocks(p, [11], 24)
    for bad in (964100, 977123, 989650):
        with pytest.raises(L.Refused):
            L.check_world_seeds([bad])
    p.write_text(json.dumps({'11': list(range(5, 29))}), encoding='utf-8')         # arbitrary non-banned fixture
    assert L.load_seed_blocks(p, [11], 24)['11'] == list(range(5, 29))
    with pytest.raises(L.Refused):
        L.load_seed_blocks(p, [23], 24)                                              # no default block exists


# ------------------------------------------------------------------------------------------------ E2
def _fixture_bank():
    def plan(name, lb, roles, cd):
        s = L.make_strategy(name, 'lanes', 'far', 300.0, roles, cd)
        return dict(family=name, strategy=s, lb95_veh_s=lb)
    return L.bank_doc([
        plan('p_a', 900.0, [L.make_role(0, 0, 18, 1.0, 90, 0), L.make_role(1, 0, 19, 1.0, 90, 20),
                            L.make_role(2, 0, 21, 1.5, 180, 0)], 180),
        plan('p_b', 400.0, [L.make_role(0, 1, 22, 1.5, 120, 20), L.make_role(1, 1, 22, 1.0, 90, 20),
                            L.make_role(2, 1, 20, 1.0, 120, 0), L.make_role(2, 0, 18, 1.0, 90, 0)], 200),
        plan('p_c', 100.0, [L.make_role(0, 0, 20, 1.0, 90, 0), L.make_role(1, 0, 20, 1.0, 90, 0),
                            L.make_role(2, 0, 20, 1.0, 90, 0)], 90)])


def _check_e2(bank):
    plans, banks, diffs = e2_build.build(bank)
    assert set(banks) == {'E2_S_BANK.json'} | {f'E2_H{v}_BANK.json' for v in (18, 19, 20, 21, 22)}
    by = {p['family']: p for p in plans}
    for name, b in banks.items():
        assert [p['source_family'] for p in b['plans']] == [p['family'] for p in plans]
        for c in b['plans']:
            f = by[c['source_family']]
            kinds = {x.split('.')[-1] for x in e2_build.diff_fields(f['strategy'], c['strategy'])}
            assert c['lb95_veh_s'] == f['lb95_veh_s']
            assert c['strategy']['cooldown_s'] == f['strategy']['cooldown_s']
            assert all(r['onset_s'] == 0 for r in c['strategy']['roles'])
            assert S.planned_resource(c['strategy']) == S.planned_resource(f['strategy'])
            assert [r['hold_s'] for r in c['strategy']['roles']] == [r['hold_s'] for r in f['strategy']['roles']]
            assert [(r['lane'], r['rank'], r['dmax']) for r in c['strategy']['roles']] == \
                   [(r['lane'], r['rank'], r['dmax']) for r in f['strategy']['roles']]
            if b['arm'] == 'S':
                assert kinds <= {'onset_s'}                                         # only onsets (+ ids / names)
                assert [r['v_w'] for r in c['strategy']['roles']] == [r['v_w'] for r in f['strategy']['roles']]
                assert bool(kinds) == any(r['onset_s'] for r in f['strategy']['roles'])
            else:
                assert kinds <= {'onset_s', 'v_w'}
                assert {r['v_w'] for r in c['strategy']['roles']} == {float(b['arm'][1:])}
            assert L.static_legality(c['strategy'], GRID) is None
        # the written bank reads back as the same plans in the same executor priority order
        assert [p['family'] for p in L.bank_strategies(b['doc'])] == [p['family'] for p in b['plans']]
    assert len(diffs) == 6*len(plans)
    return plans, banks, diffs


def test_e2_invariants_on_fixture_bank():
    plans, banks, diffs = _check_e2(_fixture_bank())
    s = {d['original']: d for d in diffs if d['kind'] == 'S'}
    assert s['p_c']['identical_to_original'] and not s['p_a']['identical_to_original']
    assert s['p_a']['differing_fields'] == ['roles[1].onset_s']
    assert e2_build.onset_report(plans)['plans_with_nonzero_onset'] == 2


def test_e2_checker_catches_a_violation():
    plans, _, _ = e2_build.build(_fixture_bank())
    f = plans[0]
    for kind, mutate in (('S', lambda s: s['roles'][0].update(hold_s=120)),
                         ('S', lambda s: s['roles'][0].update(v_w=22.0)),
                         ('H', lambda s: s['roles'][0].update(dmax=1.5)),
                         ('H', lambda s: s.update(cooldown_s=999)),
                         ('H', lambda s: s['roles'].pop()),
                         ('S', lambda s: s.update(zone='near'))):
        c = e2_build.sync_copy(f) if kind == 'S' else e2_build.uniform_hold(f, 20)
        mutate(c['strategy'])
        with pytest.raises(L.Refused):
            e2_build.check_pair(f, c, kind)
    c = e2_build.sync_copy(f)
    c['strategy']['roles'][1]['onset_s'] = 20                                       # an onset left in place
    with pytest.raises(L.Refused):
        e2_build.check_pair(f, c, 'S')


@pytest.mark.skipif(not REAL_BANK.is_file(), reason='registered original bank not on this host')
def test_e2_invariants_on_the_registered_bank():
    assert L.sha_file(REAL_BANK) == L.ORIGINAL_BANK_SHA256
    bank = L.read_json(REAL_BANK)
    plans, banks, diffs = _check_e2(bank)
    assert len(plans) == 4
    rep = e2_build.onset_report(plans)
    assert rep['any_nonzero_onset'] and rep['plans_with_nonzero_onset'] == 4       # every original plan staggers
    assert not any(d['identical_to_original'] for d in diffs)
    # the mirror reproduces the registered bank document for the original plans
    assert L.canon(L.bank_doc(plans)['rows']) == L.canon(sorted(
        bank['rows'], key=lambda r: [p['family'] for p in plans].index(r['family'])))


def test_replay_selection_and_mechanics():
    plans, bank, pairs = e2_replay.one_shot_bank(_fixture_bank())
    got = {p['family']: p['strategy'] for p in L.bank_strategies(bank)}
    assert len(got) == 6 and all(s['cooldown_s'] == e2_replay.ONE_SHOT_COOLDOWN_S > L.HORIZON_S
                                 for s in got.values())
    assert pairs['p_a'] == dict(F1='ONE_p_a', S1='ONE_S_p_a', onsets_differ=True,
                                planned=dict(vehicles=3, hold_s_sum=360))
    assert pairs['p_c']['onsets_differ'] is False
    assert e2_replay.first_launch([]) is None
    assert e2_replay.first_launch([dict(t=601, family='p_a', members=['a'])]) is None
    acts = [dict(t=420, family='p_b', members=['x', 'y']), dict(t=420, family='p_c', members=['z']),
            dict(t=350, family='p_a', members=['a', 'b', 'c'])]
    assert e2_replay.first_launch(acts) == dict(t=350, family='p_a', members=['a', 'b', 'c'])
    assert e2_replay.first_launch(acts[:2])['family'] == 'p_b'                      # same second: priority order
    disc = {f'w{i}': dict(world_index=i, acts=acts if i % 2 == 0 else []) for i in range(30)}
    disc['w1'] = dict(world_index=1, acts=acts[:2])
    states, none = e2_replay.select(disc, pairs)
    assert len(states) == 16 and len({s['world_id'] for s in states}) == 16 and len(none) == 14
    assert e2_replay.select({f'w{i}': dict(world_index=i, acts=acts) for i in range(40)}, pairs)[0].__len__() == 24
    waves = e2_replay.branch_waves(states)
    assert waves[0]['kind'] == 'U0' and waves[0]['world_indices'] == sorted(s['world_index'] for s in states)
    assert sum(len(w['world_indices']) for w in waves) == 3*len(states)
    assert {w['family'] for w in waves[1:]} == {'ONE_p_a', 'ONE_S_p_a', 'ONE_p_b', 'ONE_S_p_b'}
    st = states[0]

    def res(**over):
        act = dict(t=st['t_star'], members=st['members'])
        r = dict(U0=dict(J=100.0, activations=0, prefix_digest='d'),
                 F1=dict(J=90.0, acts=[dict(act, family=st['F1'])], prefix_digest='d'),
                 S1=dict(J=95.0, acts=[dict(act, family=st['S1'])], prefix_digest='d'))
        for k, v in over.items():
            r[k] = dict(r[k], **v)
        return r
    assert e2_replay.check_world(st, res())['ok']
    assert not e2_replay.check_world(st, res(S1=dict(prefix_digest='e')))['ok']
    assert not e2_replay.check_world(st, res(F1=dict(acts=[])))['ok']
    assert not e2_replay.check_world(st, res(U0=dict(activations=1)))['ok']
    late = [dict(t=st['t_star']+1, family=st['S1'], members=st['members'])]
    assert not e2_replay.check_world(st, res(S1=dict(acts=late)))['ok']
    other = [dict(t=st['t_star'], family=st['F1'], members=['q'])]
    assert not e2_replay.check_world(st, res(F1=dict(acts=other)))['ok']
    results = {s['world_id']: res() for s in states}
    results[states[1]['world_id']] = res(S1=dict(prefix_digest='e'))                # excluded, not silently used
    an = e2_replay.analyse(states, results)
    assert an['n_states'] == 16 and an['n_valid'] == 15
    assert an['F1_minus_S1']['diff'] == -5.0 and an['F1_minus_U0']['diff'] == -10.0


def test_e2_analyse_contrasts_and_effort():
    plans, banks, _ = e2_build.build(_fixture_bank())
    worlds = [f'w{i}' for i in range(6)]
    arms = dict(F='shaF', S='shaS', **{f'H{v}': f'shaH{v}' for v in (18, 19, 20, 21, 22)})
    shift = dict(F=-40.0, S=-25.0, H18=-10.0, H19=-12.0, H20=-30.0, H21=-5.0, H22=0.0)
    u0 = {w: 500.0+7*i for i, w in enumerate(worlds)}
    res = dict(u0=u0, arms=arms, banks={arms[a]: {w: u0[w]+shift[a]+(i % 2) for i, w in enumerate(worlds)}
                                        for a in arms})
    # Development tuning -> H* = lowest mean delay (H20); ties -> lower speed; confirmation waves U0 / F / S / H*
    man = dict(v_common=[18, 19, 20, 21, 22], tune_worlds=6, original_bank=dict(name='ORIG.json', sha256='shaF'),
               banks={f'E2_{a}_BANK.json': dict(arm=a, sha256=arms[a]) for a in arms if a != 'F'})
    arm, table = e2_select.select(man, res)
    assert arm == 'H20' and [r['arm'] for r in table] == ['H18', 'H19', 'H20', 'H21', 'H22']
    tie = dict(res, banks=dict(res['banks'], shaH19=res['banks']['shaH20']))
    assert e2_select.select(man, tie)[0] == 'H19'
    with pytest.raises(L.Refused):
        e2_select.select(dict(man, tune_worlds=24), res)
    cw = e2_select.confirm_waves(man, arm, 96)
    assert [w['wave'] for w in cw] == ['CONF_U0', 'E2C_F', 'E2C_S', 'E2C_H20'] and {w['block'] for w in cw} == {'conf'}
    tw = e2_build.tune_waves(banks, 24)
    assert [w['wave'] for w in tw] == ['DEV_U0']+[f'E2T_H{v}' for v in (18, 19, 20, 21, 22)]
    assert len(box_driver.find_waves(dict(tune_waves=tw), 'dev')) == 6
    one = e2_analyse.analyse(dict(v_common=[18, 19, 20, 21, 22]), res, {}, selected='H20')
    assert one['primary']['arm'] == 'H20' and one['primary']['diff'] == -10.0 and list(one['F_minus_H']) == ['H20']
    out = e2_analyse.analyse(dict(v_common=[18, 19, 20, 21, 22]), res, {})
    assert out['F_minus_S']['diff'] == -15.0
    assert out['F_minus_H_worst_case']['arm'] == 'H20' and out['F_minus_H_worst_case']['F_better_than_every_H']
    assert out['vs_u0']['H22']['diff'] == pytest.approx(0.5)
    specs = {p['family']: p['strategy'] for p in plans}
    acts = [dict(t=400, family='p_a', members=['a', 'b', 'c'], release={'b': [10, 'left'], 'c': [100, 'lane']}),
            dict(t=1150, family='p_c', members=['a', 'd', 'e'], release={})]
    eff = S.control_effort(acts, specs)
    # p_a: a 90 s; b released at 10 s before its onset 20 -> never held; c held 100 of 180; p_c: 3 x 50 s (horizon)
    assert eff == dict(activations=2, activations_by_family=dict(p_a=1, p_c=1), hold_vehicle_seconds=340.0,
                       unique_cavs=4, full_groups=1, truncated_groups=1)
    assert e2_analyse.effort(dict(w0=acts, w1=[]), specs)['per_episode_mean']['activations'] == 1.0


# ------------------------------------------------------------------------------------------------ statistics
def test_bootstrap_self_difference_is_exactly_zero():
    x = [812.5, 1033.25, 640.0, 977.125, 1210.0, 733.5, 1002.0, 688.75]
    z = S.paired_bootstrap(x, x)
    assert z['diff'] == 0.0 and z['ci95'] == [0.0, 0.0] and z['pct'] == 0.0 and z['pct_ci95'] == [0.0, 0.0]
    assert z['worlds_lower'] == z['worlds_higher'] == 0
    assert S.unc_boot_ci([0.0]*24) == (0.0, 0.0)
    t = S.two_way_bootstrap([[0.0]*5]*6, reps=200)
    assert t['diff'] == 0.0 and t['ci95'] == [0.0, 0.0]
    assert S.two_way_bootstrap([[0.0]*5, [0.0]*7], reps=200, crossed=False)['ci95'] == [0.0, 0.0]
    with pytest.raises(ValueError):
        S.two_way_bootstrap([[0.0]*5, [0.0]*7], reps=10)                            # crossed needs a rectangle


def test_bootstrap_sign_and_percent_convention():
    ref = [100.0, 200.0, 300.0, 400.0]
    z = S.paired_bootstrap([r-10.0 for r in ref], ref)
    assert z['diff'] == -10.0 and z['ci95'] == [-10.0, -10.0]                       # treatment - reference
    assert z['pct'] == pytest.approx(-4.0)                                          # diff of means / reference mean
    assert z['pct_ci95'][0] <= z['pct'] <= z['pct_ci95'][1] < 0
    assert S.paired_bootstrap(ref, [r-10.0 for r in ref])['diff'] == 10.0
    assert S.paired_bootstrap(ref[:3]+[450.0], ref) == S.paired_bootstrap(ref[:3]+[450.0], ref)   # reproducible
    w = S.paired_by_world(dict(b=1.0, a=3.0), dict(a=1.0, b=1.0))
    assert w['worlds'] == ['a', 'b'] and w['diff'] == 1.0
    with pytest.raises(ValueError):
        S.paired_by_world(dict(a=1.0, b=1.0), dict(a=1.0, c=1.0))


def test_exact_permutation_known_answers():
    p = S.exact_paired_permutation([-1, -2, -3, -4, -5, -6])
    assert p['assignments'] == 64 and p['p_two_sided'] == 2/64 and p['p_less'] == 1/64
    assert S.exact_paired_permutation([1, 2, 3, 4, 5, 6])['p_less'] == 1.0
    assert S.exact_paired_permutation([1, -1])['p_two_sided'] == 1.0
    assert S.exact_paired_permutation([0.0]*6) == dict(S.exact_paired_permutation([0.0]*6), p_two_sided=1.0, p_less=1.0)
    # equal magnitudes: sum of 6 signs; P(|sum| >= 4) = (1 + 6 + 6 + 1) / 64, P(sum <= -4) = 7 / 64
    q = S.exact_paired_permutation([-1, -1, -1, -1, -1, 1])
    assert q['p_two_sided'] == 14/64 and q['p_less'] == 7/64
    # one-sided value by hand for three pairs: sums <= -4 among {+-1 +-2 +-3} are -6 and -4 -> 2 / 8
    assert S.exact_paired_permutation([-1, -2, -3])['p_less'] == 1/8
    assert S.exact_paired_permutation([1, -2, -3])['p_less'] == 2/8


def test_holm_known_answers():
    assert S.holm(dict(a=0.01, b=0.04)) == dict(a=0.02, b=0.04)
    assert S.holm(dict(a=0.04, b=0.03)) == dict(b=0.06, a=0.06)                     # monotone step-down
    assert S.holm(dict(a=0.6, b=0.9)) == dict(a=1.0, b=1.0)


def test_admission_rule():
    worlds = [f'w{i:02d}' for i in range(24)]
    u0 = {w: 1000.0+i for i, w in enumerate(worlds)}
    good = {w: u0[w]-50.0-(i % 4) for i, w in enumerate(worlds)}
    g = {k: 2.0 for k in S.CO_PRIMARY_GROUPS}
    v = S.unc_admit(good, u0, g)
    assert v['PASS'] and v['ci95'][1] < 0 and v['lb95_gain_veh_s'] == -v['ci95'][1] > 0
    assert S.unc_admit(good, u0, dict(g, **{'class:hdv': 10.0}))['PASS']            # exactly +10 s is allowed
    assert not S.unc_admit(good, u0, dict(g, **{'class:hdv': 10.01}))['PASS']
    assert not S.unc_admit(good, u0, {k: 2.0 for k in S.CO_PRIMARY_GROUPS[:4]})['PASS']   # a missing group
    assert not S.unc_admit(u0, u0, g)['PASS']                                       # upper bound 0 is not < 0
    noisy = {w: u0[w]+(-200.0 if i % 2 else 190.0) for i, w in enumerate(worlds)}
    assert not S.unc_admit(noisy, u0, g)['PASS']
    with pytest.raises(ValueError):
        S.unc_admit(dict(list(good.items())[:23]), u0, g)
    assert S.unc_boot_ci([1.0, 2.0, 3.0]*8) == S.unc_boot_ci([1.0, 2.0, 3.0]*8)


def test_tail_and_braking(tmp_path):
    assert S.cvar90(list(range(1, 21))) == 19.5                                     # ceil(0.1 * 20) = 2 largest
    assert S.cvar90([5.0]) == 5.0 and S.cvar90(list(range(1, 12))) == 10.5          # ceil(1.1) = 2
    t = S.paired_cvar90(dict(a=[9.0, 'L0', 1], b=[1.0, 'L1', 0], x=[99.0, 'L2', 0]), dict(a=4.0, b=2.0, y=50.0))
    assert t == dict(n_common=2, treatment=9.0, reference=4.0)                      # common vehicles only
    assert S.hard_brake_vs([-3.0, -3.01, -5.0, 0.0, 2.0]) == 2                      # strictly below -3.0
    log = tmp_path/'vehicles.jsonl.gz'
    rows = [dict(veh=0, id='a', cav=1, origin='L0', depart_delay_s=0.0, first_t=300),
            dict(t=300, k=[0], s=[1.0], v=[20.0], a=[-3.5], lane=[0]),
            dict(veh=1, id='b', cav=0, origin='L1', depart_delay_s=0.0, first_t=301),
            dict(t=301, k=[0, 1], s=[2.0, 0.0], v=[18.0, 20.0], a=[-4.0, 0.0], lane=[0, 1]),
            dict(t=302, k=[0, 1], s=[3.0, 1.0], v=[17.0, 20.0], a=[-1.0, -3.2], lane=[0, 1])]
    with gzip.open(log, 'wt', encoding='utf-8') as f:
        f.write('\n'.join(json.dumps(r) for r in rows)+'\n')
    assert S.log_hard_brake_vs(log) == 3 and S.log_hard_brake_vs(log, t1=302) == 2
    log2 = tmp_path/'other.jsonl.gz'
    rows2 = rows[:4]+[dict(t=302, k=[0, 1], s=[9.0, 9.0], v=[1.0, 1.0], a=[0.0, 0.0], lane=[0, 1])]
    with gzip.open(log2, 'wt', compresslevel=1, encoding='utf-8') as f:
        f.write('\n'.join(json.dumps(r) for r in rows2)+'\n')
    assert S.log_prefix_digest(log, 302) == S.log_prefix_digest(log2, 302)          # same content before t*
    assert S.log_prefix_digest(log, 303) != S.log_prefix_digest(log2, 303)
    assert S.selftest() == dict(selftest='PASS')


def test_mirror_reproduces_registered_keys():
    if not REAL_BANK.is_file():
        pytest.skip('registered original bank not on this host')
    keys = [L.strategy_key(p['strategy']) for p in L.bank_strategies(L.read_json(REAL_BANK))]
    assert keys == ['3a452b1348b01083', '2922c89d0b31d4c8', 'bb1cb39d47f0a524', 'd7251d07b4f749e0']
    assert L.crosscheck(None, [])['status'] == 'UNAVAILABLE'
