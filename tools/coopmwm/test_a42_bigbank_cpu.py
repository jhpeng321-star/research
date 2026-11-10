"""CPU tests for amendment 42 (zero SUMO): the candidate extractor + STOP rule on the r7b search output (fixture), the
A37 executor on 3-6 role plans (static admission and a fake-row activation), realised group size, runner ranges."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from tools.coopmwm import a37_cellbank as C
from tools.coopmwm import a42_bigbank as B

FIX = Path(__file__).resolve().parent/'fixtures'/'a42_r7b_STRATEGY_SEARCH_D2800_p20.json'
FIX_SHA = '03c274e05b7a2260a8a3c239a4ecbb5606a0a257e5e8647377f5d73e0dd9db61'


def r7b():
    assert hashlib.sha256(FIX.read_bytes()).hexdigest() == FIX_SHA
    return json.loads(FIX.read_text(encoding='utf-8'))


def as_p60(search):
    return dict(search, cell=B.CELL)            # the rule is cell-agnostic; fixture relabelled only


def test_r7b_registered_rule_skips_never_activated_rows():
    """MAIN ruling 2026-11-10 22:40 (pre-outcome): feasibility-0 rows are skipped like comparator / banned / refused."""
    doc = B.select(as_p60(r7b()), FIX_SHA)
    assert doc['skip_counts'] == {'comparator': 2, 'never_activated': 14}
    assert doc['n_skipped'] == 16 and doc['n_eligible'] == 31 and doc['search']['n_final'] == 47
    assert doc['k'] == 6 and doc['n_selected'] == len(doc['candidates']) == 6
    assert [c['wm_rank'] for c in doc['candidates']] == [15, 16, 17, 18, 19, 20]
    assert [c['key'][:8] for c in doc['candidates']] == ['406b1c8d', '10aa9d29', 'bbe66edd', '14c15871', '5a64ff4a',
                                                         '35092be9']
    assert all(c['wm']['feasible'] > 0 and c['n_roles'] >= 4 and c['usable_larger_group'] for c in doc['candidates'])
    scores = [c['score'] for c in doc['candidates']]
    assert scores == sorted(scores, reverse=True)
    assert doc['n_usable_larger_in_candidates'] == 6 and not doc['STOP'] and doc['verdict'] == 'CONTINUE'
    assert doc['n_usable_larger_in_whole_final_list'] == 31
    assert {s['name'] for s in doc['skipped'] if s['skip'] == 'comparator'} == {'wall_g_solo2', 'solo2'}
    assert all(s['score'] == 0.0 for s in doc['skipped'] if s['skip'] == 'never_activated')
    assert sum(1 for r in doc['ranking'] if r['skip'] is None) == 31 and len(doc['ranking']) == 47


def test_stop_when_no_row_ever_activates():
    s = as_p60(r7b())
    s['final'] = [dict(x, feasible=0.0) for x in s['final']]
    doc = B.select(s, 'x')
    assert doc['candidates'] == [] and doc['STOP'] and doc['verdict'] == B.STOP_TEXT
    assert doc['skip_counts'] == {'comparator': 2, 'never_activated': 45}


def test_select_refuses_other_cell_and_key_mismatch():
    import pytest
    with pytest.raises(ValueError):
        B.select(r7b(), FIX_SHA)                                       # cell D2800_p20 is not D2800_p60
    s = as_p60(r7b())
    s['final'][0] = dict(s['final'][0], key='0'*16)
    with pytest.raises(ValueError):
        B.select(s, FIX_SHA)


def test_stop_rule_continues_when_one_candidate_is_a_feasible_larger_group():
    s = as_p60(r7b())
    feas = next(x for x in s['final'] if x['feasible'] > 0 and len(x['strategy']['roles']) >= 3 and not x['comparator'])
    s['final'] = [dict(feas, score=1.0)] + [dict(x, feasible=0.0) for x in s['final'] if x is not feas]
    doc = B.select(s, 'x')
    assert not doc['STOP'] and doc['n_usable_larger_in_candidates'] == 1 and doc['candidates'][0]['key'] == feas['key']
    assert len(doc['candidates']) == 1                                # fewer than K eligible rows: all of them, no padding


def test_seal_detects_edits():
    doc = B.seal(B.select(as_p60(r7b()), FIX_SHA))
    assert B.doc_sha(doc) == doc['doc_sha256']
    doc['candidates'][0]['score'] = 9.0
    assert B.doc_sha(doc) != doc['doc_sha256']


def test_executor_admits_every_r7b_shape():
    rows = B.shape_check([x['strategy'] for x in r7b()['final']])
    shapes = Counter((r['trigger'], r['n_roles']) for r in rows)
    assert shapes == Counter({('lanes', 5): 14, ('lanes', 4): 12, ('wall', 3): 10, ('lanes', 6): 7, ('lanes', 3): 2,
                              ('wall', 1): 2})
    assert all(r['admitted'] and not r['refused'] for r in rows)


def test_executor_refuses_lane_change_and_phases_shapes_only():
    s = dict(r7b()['final'][0]['strategy'])
    lc = dict(s, roles=[dict(s['roles'][0], lc_target=1)] + s['roles'][1:])
    ph = dict(s, phases=[[0, 1], [2, 3]])
    rows = B.shape_check([lc, ph])
    assert [r['pre_refusal'] for r in rows] == ['non_hold_plan', 'gate_or_phases'] and not any(r['admitted'] for r in rows)


def _cluster():
    rows = []
    for sumo in (0, 1, 2):
        for k, s in enumerate((-1000.0, -1060.0)):
            rows.append(dict(vid=f'c{sumo}{k}', s_m=s-sumo, v_mps=25.0, lane=2-sumo, is_cav=True))
    return rows


def test_three_to_six_role_plans_activate_with_distinct_members_on_fake_rows():
    rows = _cluster()
    cmds = {r['vid']: 0.0 for r in rows}
    seen = Counter()
    for x in r7b()['final']:
        s = x['strategy']
        if len(s['roles']) < 3:
            continue
        core = C.CellBankCore(C.bank_doc([dict(family=s['name'], strategy=s, lb95_veh_s=0.0)], B.CELL), B.CELL)
        accel, lanes = core.step(rows, cmds, 300)
        assert len(core.acts) == 1
        mem = core.acts[0]['members']
        assert len(mem) == len(s['roles']) == len(set(mem))
        assert len(accel) == sum(int(r['onset_s']) == 0 for r in s['roles'])
        assert set(lanes) == set(mem)
        seen[len(mem)] += 1
    assert set(seen) == {3, 4, 5, 6}


def test_realised_group_size_counts_members_released_before_onset_as_not_held():
    s = dict(name='p', roles=[dict(lane=0, rank=0, onset_s=0), dict(lane=1, rank=0, onset_s=20),
                              dict(lane=2, rank=0, onset_s=20)], phases=None)
    rec = dict(acts=[dict(family='p', members=['a', 'b', 'c'], release={'a': [90, 'hold_end'], 'b': [10, 'at_drop']}),
                     dict(family='p', members=['d', 'e', 'f'], release={})])
    rows = B.realised_sizes(rec, {'p': s})
    assert [(r['planned'], r['held']) for r in rows] == [(3, 2), (3, 3)]
    g = B.group_size_summary(rows)
    assert g['n_held_ge3'] == 1 and g['share_held_ge3'] == .5 and g['planned_size_counts'] == {'3': 2}
    assert B.group_size_summary([])['share_held_ge3'] is None


def test_runner_registers_the_a42_blocks_and_keeps_older_probes_unregistered():
    from tools.coopmwm import map_scenarios as ms
    from tools.coopmwm import nm_eval_runner as R
    R.install_eval_ranges(ms)
    assert R.A42_RANGES == ((992000, 992099), (998600, 998699))
    assert R._eval_base(992023) == 992000 and R._eval_base(998695) == 998600
    assert R._eval_base(991900) is None and R._eval_base(998599) is None and R._eval_base(998700) is None
    w = ms.world(992005, 2800, .6)
    assert w['repeat'] == 5 and w['cell_key'] == ms.cell_key(2800, .6) and ms.seed_scope(998600) == 'heldout'


def test_a37_executor_digest_unchanged():
    assert C.a37_digest().startswith('0a902437')


def test_a43_precondition_frozen_bank_executor_refuses_lane_change_roles():
    """AMENDMENT 43 precondition (zero SUMO): the frozen BANK_ALWAYS executor (oracle_plan.bank_plans, used by
    a37_cellbank.CellBankCore) refuses any plan with an lc_target role ('non_hold_plan', oracle_plan.py:406-407),
    even with a correct strategy key; LC plans therefore cannot run without an executor change (A43 STOP)."""
    from tools.coopmwm import oracle_plan as O
    s = dict(r7b()['final'][0]['strategy'])
    lc = dict(s, name='lc_plan', roles=[dict(s['roles'][0], lc_target=0)] + s['roles'][1:])
    hold = dict(lc, roles=[dict(r, lc_target=None) for r in lc['roles']])
    bank = C.bank_doc([dict(family='lc_plan', strategy=hold, lb95_veh_s=0.0)], B.CELL)
    t = bank['clean5c']['templates'][0]
    t['offer_condition']['roles'][0]['lc_target_sumo'] = 0
    t['template']['source']['strategy_key'] = O.strategy_key(lc)
    bank['rows'][0]['strategy_key'] = O.strategy_key(lc)
    core = C.CellBankCore(bank, B.CELL)
    assert core.plans == [] and core.refused == {'non_hold_plan': ['lc_plan']}
    assert C.executor_refusal(lc) == 'non_hold_plan'
