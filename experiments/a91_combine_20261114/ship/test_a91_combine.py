"""A91 CPU tests (zero SUMO). Run in the A91 tree: python -m pytest -q ops/test_a91_combine.py"""
import json
import os
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
os.environ.setdefault('A91_TURNS_SHA256', '0'*64)

import a81_bigconfirm as B  # noqa: E402
import a91_combine as M  # noqa: E402


def test_constants_and_caps():
    assert M.SEEDS_C == tuple(range(988600, 988696)) and M.RESERVED_C == (988696, 988699)
    assert (M.MECH_CAP, M.CONFIRM_CAP, M.CAP_SIM_S) == (14_400, 576_000, 590_400)
    assert M.ARMS == ('L2_ONLY', 'BANK_TURNS', 'ROLL_BOTH', 'DUO_HOLD_60C', 'MAPPO60C')
    assert len(M.MECH_ARMS) == 6 and M.MECH_EPISODES == 12
    for lo, hi in B.FORBIDDEN+((988200, 988599), (988400, 988499), (992000, 992999), (996000, 996999)):
        assert not (M.BLOCK[0] <= hi and M.BLOCK[1] >= lo)


def test_point_repoints_a81_names():
    M.point()
    assert B.AMENDMENT == '91' and B.PHASE == 'combine_a91' and B.ARMS == M.ARMS and B.CONFIRM_N == 96
    assert B.STAGE_CAP == dict(mech=14_400, confirm=576_000) and B.CAP_SIM_S == 590_400
    assert B.PREFIX == dict(i='a91i_', c='a91c_') and B.MECH_FILE == 'A91_MECH.json'
    assert B.RESULTS_FILE == 'A91_RESULTS_D2800_p60.json' and B.PANEL_C == 'A91_PANEL_C.json'
    assert B.waves is M.waves and B.analyse is M.analyse and B.mech_compare is M.mech_compare
    assert B.core is M.core and B._runner_check is M._runner_check and B.prepare is M.prepare
    assert B.BANK_ARM == 'A75D2_BANK_60C' and B.BANK_SHA256.startswith('c325ec1d')


def test_block_free_against_runner_rows():
    table = [(988400, 988499, 988400), (988600, 988699, 988600)]
    assert B.block_free(M.BLOCK, table)
    with pytest.raises(ValueError):
        B.block_free(M.BLOCK, table+[(988650, 988660, 988650)])


def _row(t, r, diff, hi, all_pass=True, kind='x'):
    return dict(treatment=t, reference=r, kind=kind, total_minus_ref=dict(diff=diff, ci95=[diff-1, hi]),
                groups=dict(all_pass=all_pass), tail_cvar90=None)


def test_primary_verdict():
    v = M.primary_verdict([_row('L2_ONLY', 'DUO_HOLD_60C', -5, -0.1)])
    assert v['supported'] and v['verdict'] == 'SUPPORTED' and v['group_rule_reference'] == 'DUO_HOLD_60C'
    v = M.primary_verdict([_row('L2_ONLY', 'DUO_HOLD_60C', -5, -0.1, all_pass=False)])
    assert not v['supported'] and v['delay_pass'] and not v['group_rule_pass']
    v = M.primary_verdict([_row('L2_ONLY', 'DUO_HOLD_60C', -5, 0.0)])
    assert not v['supported'] and not v['delay_pass']


def test_contrast_list():
    got = [(a, r, k) for a, r, k in M.CONTRASTS]
    assert got[0] == ('L2_ONLY', 'DUO_HOLD_60C', 'primary')
    assert ('BANK_TURNS', 'L2_ONLY', 'secondary') in got and ('L2_ONLY', 'ROLL_BOTH', 'secondary_replication_a89') in got
    assert {a for a, r, _ in got if r == 'MAPPO60C'} == {'L2_ONLY', 'BANK_TURNS', 'ROLL_BOTH', 'DUO_HOLD_60C'}
    assert all(a != r for a, r, _ in got)


def test_summary_layout():
    rows = [_row(a, r, -1, -0.5, kind=k) for a, r, k in M.CONTRASTS]
    v = dict(analysis_sha256='x'*64, primary=M.primary_verdict(rows), contrasts=rows,
             arms=dict(BANK_TURNS=dict(overlap=dict(concurrent_s=3))))
    s = M.summary(v)
    json.dumps(s)
    assert s['primary']['verdict'] == 'SUPPORTED' and len(s['secondary']) == 2 and s['bank_turns_concurrent_s'] == 3


def _act(fam, t, rel):
    return dict(family=fam, t=t, members=list(rel), release={k: v for k, v in rel.items() if v is not None})


def test_overlap_stats():
    r = dict(acts=[_act('L2', 400, {'c1': [40, 'hold_end']}),
                   _act('TP', 420, {'c2': [120, 'hold_end'], 'c3': [60, 'at_drop']}),
                   _act('L2', 600, {'c4': None})])
    ov = M.overlap_stats(r, horizon=1200)
    assert ov['active_s_by_family'] == {'L2': 40+600, 'TP': 120}
    assert ov['concurrent_s'] == 20 + 0          # 420-439 (L2 400-439 & TP 420-539); 600+ vs TP ended 540
    assert ov['acts_started_while_other_active'] == 1 and ov['cav_in_two_sessions'] == 0
    r['acts'].append(_act('TP', 410, {'c1': [10, 'hold_end']}))
    assert M.overlap_stats(r, horizon=1200)['cav_in_two_sessions'] == 1


def test_split_family():
    r = dict(acts=[_act('A', 1, {'x': [1, 'hold_end']}), _act('B', 2, {'y': [1, 'hold_end']})],
             holdforms=[dict(family='A', vid='x'), dict(family='B', vid='y')], refused={})
    a = M.split_family(r, 'A')
    assert [x['family'] for x in a['acts']] == ['A'] and [e['vid'] for e in a['holdforms']] == ['x']
    assert len(r['acts']) == 2


TURNS = HERE/'a91'/M.TURNS_BANK
ARMS89 = HERE/'a91'/M.ARMS_BANK


@pytest.mark.skipif(not TURNS.is_file(), reason='turns bank not built in this tree')
def test_turns_bank_derivation():
    from tools.coopmwm import a75_holdforms as H
    t = json.loads(TURNS.read_text(encoding='utf-8'))
    a = json.loads(ARMS89.read_text(encoding='utf-8'))
    canon = lambda x: json.dumps(x, sort_keys=True)  # noqa: E731
    for f in M.BANK_FAMILIES['BANK_TURNS']:
        assert [canon(x) for x in t['rows'] if x['family'] == f] == [canon(x) for x in a['rows'] if x['family'] == f]
        assert [canon(x) for x in t['clean5c']['templates'] if x['family'] == f] == \
            [canon(x) for x in a['clean5c']['templates'] if x['family'] == f]
    assert len(t['rows']) == 2 and len(t['clean5c']['templates']) == 2 and t['schema'] == a['schema']
    core = H.make_core('A75_BANK', M.CELL, t)
    assert [p['family'] for p in core.plans] == [M.L2_FAMILY, M.TAPER_FAMILY]          # executor priority: L2 first
    assert [type(c).__name__ for c in core.cores] == ['LcStrategyCore', 'HoldFormCore']
    for arm, fam in M.ARM_FAMILY.items():
        assert [p['family'] for p in H.make_core('A75_PLAN', M.CELL, a, family=fam).plans] == [fam]


@pytest.mark.skipif(not ARMS89.is_file(), reason='arms bank not in this tree')
def test_union_plan_is_not_executable():
    """Part 1b evidence: the literal one-plan union of the L2 role and the four TAPER roles is admitted by the executor
    but select_members can never fill it (two lane-2 rank-0 roles need the same CAV)."""
    from tools.coopmwm import a75_holdforms as H
    a = json.loads(ARMS89.read_text(encoding='utf-8'))
    tp = next(x for x in a['clean5c']['templates'] if x['family'] == M.TAPER_FAMILY)['offer_condition']
    l2 = next(x for x in a['clean5c']['templates'] if x['family'] == M.L2_FAMILY)['offer_condition']
    cond = dict(tp, roles=list(l2['roles'])+list(tp['roles']))
    spec = H.spec_of('union_test', cond)
    assert H.executor_refusal(spec) is None
    lo, hi = H.zone_bounds(spec)
    cavs = [dict(id=f'c{i}', edge='up', lane=ln, x=lo+50.0+10*i, v=20.0)
            for i, ln in enumerate((2, 2, 2, 1, 1, 0, 0))]
    assert H.select_members(spec, cavs, cavs) is None
    assert H.select_members(H.spec_of('tp', tp), cavs, cavs) is not None       # TAPER alone fills on the same rows
