"""CPU tests for amendment 45 (zero SUMO): diverse structural-class selection from the same search `final` (classes from
the descriptor, the A42 top 6 never selected again, cap 12, deterministic ties by `final` order, empty case)."""
import json
from copy import deepcopy

import pytest

from tools.coopmwm import a42_bigbank as B
from tools.coopmwm.test_a42_bigbank_cpu import FIX_SHA, as_p60, r7b


def test_descriptor_fields_and_class_of_a_row():
    x = r7b()['final'][0]
    cls = B.structural_class(x)
    assert B.CLASS_FIELDS == ('n_roles', 'n_lanes', 'staggered', 'zone', 'phases')
    assert cls == dict(n_roles=6, n_lanes=3, staggered=1, zone='full', phases=False)
    assert B.class_key(cls) == 'n_roles=6|n_lanes=3|staggered=1|zone=full|phases=False'
    bad = dict(x, descriptor=[5]+list(x['descriptor'][1:]))
    with pytest.raises(ValueError, match='differs from the strategy'):
        B.structural_class(bad)
    with pytest.raises(ValueError, match='fields'):
        B.structural_class(dict(x, descriptor=x['descriptor'][:9]))


def test_diverse_on_r7b_one_per_class_never_top6_ordered_by_score():
    doc = B.select(as_p60(r7b()), FIX_SHA)
    div = doc['diverse']
    top = {c['key'] for c in doc['candidates']}
    cands = div['candidates']
    assert cands and len(cands) <= 12 and div['cap'] == 12
    assert not top & {c['key'] for c in cands}
    keys = [c['class_key'] for c in cands]
    assert len(keys) == len(set(keys))
    scores = [c['score'] for c in cands]
    assert scores == sorted(scores, reverse=True)
    assert [c['diverse_rank'] for c in cands] == list(range(1, len(cands)+1))
    ranking = {e['key']: e for e in doc['ranking']}
    for c in cands:
        assert ranking[c['key']]['skip'] is None and ranking[c['key']]['wm_rank'] == c['wm_rank']
        # the class best: no eligible non-top-6 row of the same class has a higher score
        same = [x for x in doc['ranking'] if x['skip'] is None and x['key'] not in top
                and B.class_key(B.structural_class(next(r for r in r7b()['final'] if r['key'] == x['key'])))
                == c['class_key']]
        assert max(x['score'] for x in same) == c['score'] and c['class_counts']['eligible_not_top'] == len(same)
    sc = div['skip_counts']
    assert sc['not_eligible'] == doc['n_skipped'] and sc['in_top_k'] == 6
    assert sc['not_eligible']+sc['in_top_k']+sc['not_class_best']+div['n_classes'] == doc['search']['n_final']
    assert sc['over_cap'] == max(0, div['n_classes']-12)
    assert div['descriptor_fields'][4] == 'staggered' and div['class_fields'] == list(B.CLASS_FIELDS)


def test_cap_is_applied_in_class_score_order():
    s = as_p60(r7b())
    sel = B.select_candidates(s, B.CELL, k=B.K)
    full = B.select_diverse(s, sel)
    assert full['n_classes'] >= 3
    two = B.select_diverse(s, sel, cap=2)
    assert [c['key'] for c in two['candidates']] == [c['key'] for c in full['candidates'][:2]]
    assert two['skip_counts']['over_cap'] == full['n_classes']-2


def test_cap_12_with_many_classes():
    s = as_p60(r7b())
    proto = next(x for x in s['final'] if x['feasible'] > 0 and not x['comparator'])
    rows = []
    for i in range(30):                       # 30 distinct classes: n_roles 2..6 x zone x staggered
        n, zone, stag = 2+i % 5, ('near', 'far', 'full')[(i//5) % 3], i//15
        st = deepcopy(proto['strategy'])
        st['roles'] = [dict(st['roles'][0], onset_s=20*stag*(r == 0)) for r in range(n)]
        st['zone'] = zone
        st['name'] = f'syn_{i:02}'
        d = list(proto['descriptor'])
        d[0], d[1], d[4], d[5] = n, 1, stag, zone
        rows.append(dict(proto, strategy=st, key=f'{i:016x}', descriptor=d, score=100.0-i))
    s['final'] = rows
    from tools.coopmwm import oracle_plan as O
    for r in s['final']:
        r['key'] = O.strategy_key(r['strategy'])
    doc = B.select(s, 'x')
    div = doc['diverse']
    top = [c['key'] for c in doc['candidates']]
    assert len(top) == 6 and div['n_selected'] == 12 and div['n_classes'] == 24
    assert div['n_classes'] == len({B.class_key(B.structural_class(r)) for r in rows if r['key'] not in top})
    assert div['skip_counts']['over_cap'] == div['n_classes']-12


def test_deterministic_and_ties_follow_final_order():
    s = as_p60(r7b())
    a = B.select(s, FIX_SHA)['diverse']
    b = B.select(json.loads(json.dumps(s)), FIX_SHA)['diverse']
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    # two eligible non-top rows of one class with the same score: the earlier one in `final` is the class best
    sel = B.select_candidates(s, B.CELL, k=B.K)
    top = {c['key'] for c in sel['candidates']}
    by_cls = {}
    for i, x in enumerate(s['final']):
        e = next(r for r in sel['ranking'] if r['key'] == x['key'])
        if e['skip'] is None and x['key'] not in top:
            by_cls.setdefault(B.class_key(B.structural_class(x)), []).append(i)
    ck, idx = next((k, v) for k, v in by_cls.items() if len(v) >= 2)
    i, j = idx[0], idx[1]
    s2 = deepcopy(s)
    s2['final'][i]['score'] = s2['final'][j]['score'] = max(s['final'][i]['score'], s['final'][j]['score'])
    got = {c['class_key']: c for c in B.select(s2, 'x')['diverse']['candidates']}
    assert got[ck]['final_index'] == min(i, j) and got[ck]['key'] == s2['final'][min(i, j)]['key']
    s3 = deepcopy(s2)
    s3['final'][i], s3['final'][j] = s3['final'][j], s3['final'][i]
    got3 = {c['class_key']: c for c in B.select(s3, 'x')['diverse']['candidates']}
    assert got3[ck]['key'] == s3['final'][min(i, j)]['key'] != got[ck]['key']


def test_empty_when_every_eligible_row_is_in_the_top6():
    s = as_p60(r7b())
    sel = B.select_candidates(s, B.CELL, k=B.K)
    top = {c['key'] for c in sel['candidates']}
    s['final'] = [x for x in s['final'] if x['key'] in top or x['comparator'] or not x['feasible'] > 0]
    doc = B.select(s, 'x')
    assert doc['n_selected'] == 6 and doc['diverse']['candidates'] == [] and doc['diverse']['n_classes'] == 0
    assert doc['diverse']['skip_counts']['in_top_k'] == 6
    s['final'] = [dict(x, feasible=0.0) for x in s['final']]           # STOP search: nothing eligible at all
    doc = B.select(s, 'x')
    assert doc['STOP'] and doc['diverse']['candidates'] == []
