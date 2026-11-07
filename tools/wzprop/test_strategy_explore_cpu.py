"""CPU tests of strategy_explore (stub WM only; no SUMO, GPU or ssh).

    C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m pytest -q tools/wzprop/test_strategy_explore_cpu.py
"""
import hashlib
import itertools
import json
import math
import random
import sys
from pathlib import Path

import pytest

from tools.wzprop import fakes
from tools.wzprop import geometry as G
from tools.wzprop import strategy_explore as X
from tools.wzprop import strategy_search as SS
from tools.wzprop import wm_backend as W

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# sha256 at 65aeefc3 (git blob bytes, LF line endings)
SHA_AT_BASE = {'strategy_search.py': '763e7ae692bf8813051740750a64684ed4b749f8f2dbfbf493d6a4e26698b306',
               '__main__.py': 'f22ff2c1279340f3a52124367c39149448b6f0b66d496e672ef0b9e3aa7a0361'}
SMALL = dict(n_init=16, generations=2, n_children=12, n_screen_states=6, K_screen=2, finalists=4, n_final_states=0,
             K_final=3, novelty_k=5)


def moment(seed, t0=450, cavs=None):
    cavs = fakes.fake_cavs(seed, 0.6, t0) if cavs is None else cavs
    return dict(cell='D3200_p60', demand=3200, penetration=0.6, seed=seed, window=0, t0=t0, valid=True,
                sid=f'ex_D3200_p60_s{seed}_t{t0}', inserted_t0=100 + t0, approach_speed=12.0, closing_lane_slow=3,
                n_zone_cavs=len(G.zone_cavs(cavs)), variants={}, union=[], dither_ids={}, all_cavs_t0=cavs)


def states(n=8, seed0=989300):
    return [moment(seed0 + i, 450 + 10 * (i % 2)) for i in range(n)]


def stub():
    """Deterministic pseudo-random WM rows (planted world with small through-lane cost -> some low-harm plans)."""
    return W.StubWM(world=fakes.PlantedWorld(through_cost=5.0))


@pytest.fixture(scope='module')
def result():
    return X.search_cell(stub(), states(), dict(SMALL))


def _check_child(c):
    assert X.allowed(c)
    SS.validate(c)
    X.wmverify_check(c)
    assert not SS.banned(c) and all(float(r['v_w']) != 15.0 for r in c['roles'])
    assert c['slots'] == 1 and float(c['spread_m']) in (200.0, 300.0) and c['cooldown_s'] <= 240


# ------------------------------------------------------------------------------------------ hard constraint
@pytest.mark.parametrize('name', sorted(SHA_AT_BASE))
def test_protected_modules_unchanged(name):
    data = (HERE / name).read_bytes().replace(b'\r\n', b'\n')
    assert hashlib.sha256(data).hexdigest() == SHA_AT_BASE[name]


# ------------------------------------------------------------------------------------------ WMVERIFY mirror
def _ok(spec, **kw):
    try:
        X.wmverify_check(spec, **kw)
        return True
    except ValueError:
        return False


def test_wmverify_mirror_known_valid_and_invalid_specs():
    inc = dict(SS.INCUMBENT)
    good = dict(inc, name='g2x')
    assert _ok(good)
    assert all(_ok(dict(s, name='s_' + s['name'])) for s in SS.seeds())
    assert not _ok(inc)                                         # reserved name (WMVERIFY maps the incumbent itself)
    assert not _ok(dict(good, name='a_minus_b')) and not _ok(dict(good, name='1abc'))
    assert not _ok(dict(good, spread_m=250.0)) and _ok(dict(good, spread_m=300.0))
    assert not _ok(dict(good, slots=2))
    assert not _ok(dict(good, cooldown_s=300)) and _ok(dict(good, cooldown_s=240))
    assert not _ok(dict(good, cooldown_s=120.0))                # int only
    assert not _ok(dict(good, members=['veh1']))                # unknown spec key (no vehicle ids)
    assert not _ok(dict(good, roles=[dict(good['roles'][0], id='veh1')]))
    assert not _ok(dict(good, roles=[dict(good['roles'][0], lane=2.0)]))
    assert not _ok(dict(good, roles=[dict(good['roles'][0], hold_s=120.0)]))
    assert not _ok(dict(good, roles=[dict(good['roles'][0], v_w=15.0)]))   # banned
    assert not _ok({k: v for k, v in good.items() if k != 'zone'})
    assert _ok({k: v for k, v in good.items() if k not in ('gate', 'start_gate', 'phases')})
    alt = next(s for s in SS.seeds() if s['name'] == 'wall_g_alt21')
    assert _ok(dict(alt, name='alt')) and not _ok(dict(alt, name='alt', phases=((0,), (1,))))
    lc0 = dict(SS.seeds()[0], name='lc10', trigger='lanes', roles=[SS._role(1, SS.WALL_G, lc=0)])
    assert _ok(lc0) and not _ok(lc0, lane_executor='plant') and not _ok(lc0, lane_executor='foo')
    assert _ok(dict(lc0, roles=[SS._role(1, SS.WALL_G, lc=2)]), lane_executor='plant')


# ------------------------------------------------------------------------------------------ operators
def test_generated_children_pass_validate_and_wmverify_rules():
    rng = random.Random(7)
    pool = SS.seeds() + [SS.random_strategy(rng) for _ in range(40)]
    cfg = dict(X.DEFAULTS)
    kids, st = X.make_children(rng, pool, 300, cfg, set())
    assert len(kids) == 300 and st['immigrants'] == 45
    assert all(st[k] > 0 for k in ('crossover', 'team', 'early', 'struct', 'mutate'))
    for c in kids:
        _check_child(c)
    assert len({SS.strategy_key(c) for c in kids}) == len(kids)
    seen_ops = dict(removed=False, mixed=False, flipped=False, lc=False, six=False)
    for i in range(400):
        p = pool[i % len(pool)]
        c = X.mutate_struct(rng, p)
        if c is not None:
            _check_child(c)
            seen_ops['removed'] |= len(c['roles']) < len(p['roles'])
            seen_ops['mixed'] |= bool(c.get('phases')) and any(len(g) >= 2 for g in c['phases'])
            seen_ops['flipped'] |= c['trigger'] != p['trigger']
            seen_ops['lc'] |= X.has_lane_roles(c) != X.has_lane_roles(p)
        q = X.crossover(rng, p, pool[(7 * i + 3) % len(pool)])
        if q is not None:
            _check_child(q)
            seen_ops['six'] |= len(q['roles']) == 6
            assert len({(r['lane'], r['rank']) for r in q['roles']}) == len(q['roles']) <= 6
    assert all(seen_ops.values()), seen_ops


def test_gap_open_and_early_change_operators():
    rng = random.Random(11)
    n_team = n_early = 0
    for i in range(200):
        parent = SS.seeds()[i % len(SS.seeds())] if i % 2 else None
        t = X.gap_open(rng, parent)
        if t is not None:
            n_team += 1
            _check_child(t)
            ch = [r for r in t['roles'] if r['lc_target'] is not None]
            assert any(r['lc_target'] is None and any(c['lc_target'] == r['lane'] and c['onset_s'] > r['onset_s']
                                                      for c in ch) for r in t['roles'])
        e = X.early_change(rng, parent)
        if e is not None:
            n_early += 1
            _check_child(e)
            assert e['zone'] in ('far', 'full') and X.has_lane_roles(e)
            assert all(r['onset_s'] == 0 for r in e['roles'] if r['lc_target'] is not None)
    assert n_team > 50 and n_early > 50


# ------------------------------------------------------------------------------------------ distance / novelty
def test_genome_distance_properties():
    by = {s['name']: s for s in SS.seeds()}
    solo1, solo2 = by['wall_g_solo1'], by['wall_g_solo2']
    assert X.genome_distance(solo1, solo2) > 0 and X.genome_distance(solo2, solo2) == 0.0
    duo = by['wall_g_duo12']
    rev = dict(duo, roles=list(reversed(duo['roles'])))
    assert X.genome_distance(duo, rev) == 0.0                    # role order is matched away
    rng = random.Random(3)
    pool = SS.seeds() + [SS.random_strategy(rng) for _ in range(30)]
    for a, b in itertools.combinations(pool[:25], 2):
        dab = X.genome_distance(a, b)
        assert dab == X.genome_distance(b, a) and dab >= 0
        assert X.genome_distance(a, a) == 0.0
    for n in (1, 2, 3, 4, 5):
        C = [[rng.random() for _ in range(n)] for _ in range(n)]
        brute = min(sum(C[i][p[i]] for i in range(n)) for p in itertools.permutations(range(n)))
        assert abs(X.assignment_cost(C) - brute) < 1e-12


def test_novelty_prefers_far_strategies():
    by = {s['name']: s for s in SS.seeds()}
    ref = [dict(strategy=s) for s in SS.seeds()]
    near = dict(by['wall_g_solo2'], cooldown_s=150, name='near')
    rng = random.Random(5)
    far = None
    for _ in range(50):
        c = SS.random_strategy(rng)
        if len(c['roles']) >= 5:
            far = c
            break
    assert X.novelty(dict(strategy=far), ref, k=3) > X.novelty(dict(strategy=near), ref, k=3)
    assert X.novelty(dict(strategy=by['wall_g_solo2']), ref, k=1) > 0      # itself is skipped


# ------------------------------------------------------------------------------------------ lane gate
def test_lane_gate_formula_numbers():
    ego_q = dict(s=1000.0, v=1.5)                    # queued ego wanting into a 27 m/s lane (needs ~190 m rear gap)
    fol = lambda gap: dict(s=1000.0 - 5.0 - gap, v=27.0, lane=1)    # noqa: E731
    need_rear = 3.0 + 1.25 * 27.0 + 6.0 * (27.0 - 1.5)
    assert abs(need_rear - 189.75) < 1e-9
    assert not X.target_gap_ok(ego_q, [fol(180.0)], 1) and X.target_gap_ok(ego_q, [fol(190.0)], 1)
    ego_f = dict(s=1000.0, v=15.0)                   # fast ego into a 2 m/s queue: front need 1 + 18 + 4.5 * 13
    lead = lambda gap: dict(s=1000.0 + 5.0 + gap, v=2.0, lane=1)    # noqa: E731
    assert not X.target_gap_ok(ego_f, [lead(77.0)], 1) and X.target_gap_ok(ego_f, [lead(78.0)], 1)
    assert not X.target_gap_ok(ego_f, [dict(s=1002.0, v=15.0, lane=1)], 1)      # alongside blocks
    assert X.target_gap_ok(ego_f, [], 1) and X.target_gap_ok(ego_f, [dict(lead(1.0), lane=2)], 1)
    m = dict(id='c', edge='up', lane=0, x=2000.0, v=10.0)
    far = dict(id='h', edge='up', lane=1, x=2000.0 - 150.0, v=30.0)            # beyond 100 m sensing: not seen
    assert X.lane_gate_ok(m, 1, [m, far]) and not X.lane_gate_ok(m, 2, [m]) and X.lane_gate_ok(m, 0, [m])
    assert X.lane_gate_ok(dict(m, lane=1), 2, [])                                # free target lane
    edge = dict(id='e', edge='up', lane=1, x=G.X_B - 15.0, v=10.0)
    assert not X.lane_gate_ok(edge, 2, [edge])                                   # edge guard v*1 s + 10 m
    assert X.lane_gate_ok(dict(edge, lane=0), 1, [edge])                         # leaving the closing lane is exempt


def test_lane_gate_crosscheck_against_lanes_py():
    torch = pytest.importorskip('torch')
    src = str(ROOT / 'src')
    if src not in sys.path:
        sys.path.insert(0, src)
    lanes = pytest.importorskip('wms.control.lanes')
    rules = lanes.LaneRules()
    assert (rules.gap_base_m, rules.gap_time_s, rules.closing_time_s, rules.edge_guard_m) == (
        X.LANE_RULES['gap_base_m'], X.LANE_RULES['gap_time_s'], X.LANE_RULES['closing_time_s'],
        X.LANE_RULES['edge_guard_m'])
    assert rules.rear == (X.LANE_RULES['rear_gap_base_m'], X.LANE_RULES['rear_gap_time_s'],
                          X.LANE_RULES['rear_closing_time_s'])
    rng = random.Random(17)
    n_ok = 0
    for _ in range(400):
        k = rng.randint(1, 7)
        rows = [dict(s=rng.uniform(0, 200), v=rng.choice([rng.uniform(0, 3), rng.uniform(10, 30)]),
                     lane=rng.randint(0, 2), length=rng.choice([5.0, 4.5, 12.0])) for _ in range(k)]
        ego = rng.randrange(k)
        target = rng.choice([x for x in (rows[ego]['lane'] - 1, rows[ego]['lane'] + 1) if 0 <= x <= 2])
        t = lambda key, kind=float: torch.tensor([[kind(r[key]) for r in rows]])  # noqa: E731
        lane = t('lane', int)
        goal = lane.clone()
        goal[0, ego] = target
        ref = bool(lanes._target_gap_ok(t('s'), t('v'), t('length'), torch.ones_like(lane, dtype=torch.bool), lane,
                                        goal, rules)[0, ego])
        mine = X.target_gap_ok(rows[ego], [r for i, r in enumerate(rows) if i != ego], target)
        assert mine == ref
        n_ok += ref
    assert 20 < n_ok < 380                         # both outcomes exercised


def test_gate_aware_admission_and_descriptor():
    cfg = dict(X.DEFAULTS)
    lc = dict(SS.seeds()[-1])                      # g2_gapopen12 (has a lane change)
    hold = dict(SS.INCUMBENT)
    base = dict(feasible=1.0, worst_group_inc=1.0, gain_vs_base=50.0)
    assert not X.novel_admits(dict(base, strategy=lc, gate_pass_rate=0.2), cfg)
    assert not X.novel_admits(dict(base, strategy=lc, gate_pass_rate=None), cfg)
    assert X.novel_admits(dict(base, strategy=lc, gate_pass_rate=0.35), cfg)
    assert X.novel_admits(dict(base, strategy=hold, gate_pass_rate=None), cfg)
    assert not X.novel_admits(dict(base, strategy=hold, worst_group_inc=6.0), cfg)        # 5 s/veh (WM ~2x low)
    assert not X.novel_admits(dict(base, strategy=hold, feasible=0.2), cfg)
    assert X.fine_descriptor(lc, 0.05) != X.fine_descriptor(lc, 0.7) and X.fine_descriptor(hold)[-1] == 'hold'
    gp, n = X.gate_pass(lc, states(6))
    assert n > 0 and 0.0 <= gp <= 1.0 and X.gate_pass(hold, states(3)) == (None, 0)


# ------------------------------------------------------------------------------------------ search / archives
def test_min_feasible_keeps_never_activating_out_of_archives(result):
    for a in X.ARCHIVES:
        assert all(e['feasible'] >= 0.25 for e in result['archives'][a])
    dead = [moment(989300 + i, cavs=[c for c in fakes.fake_cavs(989300 + i, 0.6, 450)
                                     if c['edge'] != 'up' or float(c['x']) < G.X_B - 1900.0]) for i in range(4)]
    cfg = dict(SMALL, generations=1, n_init=8, n_children=6)
    res = X.search_cell(stub(), dead, cfg)
    assert res['evaluated'] > 0 and all(v == 0 for v in res['archive_cells'].values())
    assert res['selection']['wm_novel'] == []
    res0 = X.search_cell(stub(), dead, dict(cfg, min_feasible=0.0))
    assert res0['archive_cells']['delay'] > 0                     # the same never-activating plans without the gate


def test_deterministic_for_seed(result):
    again = X.search_cell(stub(), states(), dict(SMALL))
    assert json.dumps(again, sort_keys=True, default=str) == json.dumps(result, sort_keys=True, default=str)
    other = X.search_cell(stub(), states(), dict(SMALL, seed=5))
    assert [r['key'] for r in other['final']] != [r['key'] for r in result['final']]


def test_end_to_end_schema_with_novel_archive(result):
    ss = SS.search_cell(stub(), states(), dict(SS.DEFAULTS, **{k: v for k, v in SMALL.items() if k in SS.DEFAULTS}))
    assert set(ss) <= set(result) and result['schema'] == ss['schema']
    assert result['explore_schema'] == X.EXPLORE_SCHEMA
    assert 'novel' in result['archives'] and 'novel' in result['archive_cells'] and 'novel' in result['best']
    assert result['archive_cells']['novel'] > 0
    assert set(result['archives']) == set(X.ARCHIVES)
    for e in result['archives']['novel']:
        assert e['worst_group_inc'] <= 5.0 and e['feasible'] >= 0.25
        assert e['gate_pass_rate'] is None or e['gate_pass_rate'] >= 0.3
    sel = result['selection']
    assert sel['group'] == 'wm_novel' and len(sel['wm_novel']) == len(sel['tags']) > 0
    assert all('gate_pass_rate' in r and 'novelty' in r for r in result['final'])
    for c in result['final']:
        if not c['comparator']:
            assert X.allowed(c['strategy'])
    json.dumps(result, default=str)


def test_diverse_selection_on_search_result(result):
    sel = result['selection']
    refs = X.comparator_refs()
    rk = {SS.strategy_key(s) for s in refs}
    by = {r['key']: r for r in result['final']}
    assert len({t['cluster'] for t in sel['tags']}) == len(sel['tags'])
    assert len({SS.strategy_key(s) for s in sel['wm_novel']}) == len(sel['wm_novel'])
    for s, t in zip(sel['wm_novel'], sel['tags']):
        k = SS.strategy_key(s)
        r = by[k]
        assert k not in rk and not SS.banned(s) and X.allowed(s) and set(s) <= X.SPEC_KEYS
        assert r['feasible'] >= 0.25 and r['worst_group_inc'] <= 5.0 and X.gate_ok(r, result['config'])
        assert min(X.genome_distance(s, c) for c in refs) > sel['delta']
        assert t['support'] == X.support_tag(s)[0]


def _row(s, score, **kw):
    r = dict(strategy=s, key=SS.strategy_key(s), comparator=False, banned=SS.banned(s), feasible=1.0,
             worst_group_inc=1.0, gain_vs_base=score, gain_vs_base_lcb=score - 10, score=score, gate_pass_rate=None,
             novelty=1.0)
    r.update(kw)
    return r


def test_diverse_selection_excludes_and_clusters():
    rng = random.Random(23)
    good = []
    while len(good) < 14:
        s = SS.random_strategy(rng)
        s = dict(s, name='ok_' + SS.strategy_key(s))
        if X.allowed(s) and min(X.genome_distance(s, c) for c in X.comparator_refs()) > 1.0 \
                and not X.has_lane_roles(s):
            good.append(s)
    lc = next(s for s in (dict(SS.random_strategy(rng), name='lc') for _ in range(200))
              if X.has_lane_roles(s) and X.allowed(s) and SS.strategy_key(s) not in {SS.strategy_key(g) for g in good})
    final = [_row(dict(SS.INCUMBENT), 999.0, comparator=True), _row(dict(SS.SOLO2), 998.0, comparator=True, banned=True),
             _row(good[0], 997.0, feasible=0.1), _row(good[1], 996.0, worst_group_inc=8.0),
             _row(lc, 995.0, gate_pass_rate=0.1),
             _row(dict(SS.INCUMBENT, name='inc_copy', cooldown_s=150), 994.0)]
    final += [_row(s, 100.0 - i) for i, s in enumerate(good[2:])]
    res = dict(config=dict(X.DEFAULTS), final=final)
    sel = X.select_diverse_for_sumo(res, 6)
    keys = [SS.strategy_key(s) for s in sel['wm_novel']]
    bad = {SS.strategy_key(s) for s in (SS.INCUMBENT, SS.SOLO2, good[0], good[1], lc,
                                        dict(SS.INCUMBENT, cooldown_s=150))}
    assert keys and not set(keys) & bad
    ex = sel['excluded']
    assert ex['comparator'] == 2 and ex['infeasible'] == 1 and ex['harm'] == 1 and ex['lane_gate'] == 1
    assert ex['near_comparator'] == 1
    assert len(keys) == len(set(keys)) == len({t['cluster'] for t in sel['tags']}) == 6
    tiers = {t['support'] for t in sel['tags']}
    assert tiers <= {'in', 'out'}
    assert X.select_diverse_for_sumo(res, 6) == sel                           # deterministic
    # passing the low-gate lane plan's rate over the threshold readmits it
    final[4] = _row(lc, 995.0, gate_pass_rate=0.5)
    sel2 = X.select_diverse_for_sumo(dict(config=dict(X.DEFAULTS), final=final), 6)
    assert sel2['excluded']['lane_gate'] == 0


def test_cli_end_to_end_with_stub(tmp_path):
    sp = tmp_path / 'states.json'
    sp.write_text(json.dumps(states(6)))
    out = tmp_path / 'run'
    cfg = json.dumps(dict(SMALL, generations=1, n_children=8))
    assert X.main(['--states', str(sp), '--cells', 'D3200_p60', '--out', str(out), '--wm', 'stub',
                   '--wm-arg', 'noise_sd=60', '--config', cfg, '--k', '2']) == 0
    res = json.loads((out / 'STRATEGY_SEARCH_D3200_p60.json').read_text())
    arms = json.loads((out / 'ARMS.json').read_text())['D3200_p60']
    assert 'novel' in res['archives'] and 'selection' in res
    assert 'wm_novel' in arms and len(arms['wm_novel']) == len(arms['wm_novel_tags'])
    for s in arms['wm_novel']:
        X.wmverify_check(s)
    assert (out / 'LOG_D3200_p60.jsonl').read_text().strip()
