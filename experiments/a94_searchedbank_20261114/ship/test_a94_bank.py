"""A94 / A95 CPU tests (zero SUMO). Run in the A94 tree: python -m pytest -q ops/test_a94_bank.py"""
import json
import os
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
os.environ.setdefault('A94_HAND_SHA256', '0'*64)

import a81_bigconfirm as B  # noqa: E402
import a94_bank as M  # noqa: E402


def test_constants_and_caps():
    assert M.SEEDS_C == tuple(range(988700, 988892)) and M.RESERVED_C == (988892, 988899)
    assert M.HAND_RUN and (M.MECH_CAP, M.CONFIRM_CAP, M.CAP_SIM_S) == (14_400, 921_600, 936_000)
    assert M.ARMS == ('BANK_SEARCHED', 'BANK_HAND', 'DUO_HOLD_60C', 'MAPPO60C')
    assert len(M.MECH_ARMS) == 5 and M.MECH_EPISODES == 10 and M.MECH_EPISODES*M.HORIZON == 12_000
    assert M.PRIMARY_CONTRAST == ('BANK_SEARCHED', 'BANK_HAND')
    for lo, hi in B.FORBIDDEN+((988200, 988599), (988400, 988499), (988600, 988699), (992000, 992999),
                               (996000, 996999), (964000, 964199), (989600, 989699)):
        assert not (M.BLOCK[0] <= hi and M.BLOCK[1] >= lo)


def test_no_hand_layout():
    """A94_HAND_SHA256=none: 3 arms, primary BANK_SEARCHED - DUO_HOLD_60C, caps 9,600 / 691,200 (fresh interpreter);
    unset refuses."""
    import subprocess
    code = ("import sys; sys.path[:0]=%r; import a94_bank as M; "
            "assert not M.HAND_RUN and M.ARMS == ('BANK_SEARCHED', 'DUO_HOLD_60C', 'MAPPO60C'); "
            "assert M.PRIMARY_CONTRAST == ('BANK_SEARCHED', 'DUO_HOLD_60C') and M.CONTRASTS[0][2] == 'primary'; "
            "assert (M.MECH_EPISODES, M.CONFIRM_CAP, M.CAP_SIM_S) == (8, 691_200, 705_600); print('ok')"
            % [str(HERE), str(HERE.parent)])
    env = dict(os.environ, A94_HAND_SHA256='none')
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and r.stdout.strip() == 'ok', r.stderr[-1500:]
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=dict(env, A94_HAND_SHA256=''))
    assert r.returncode != 0


def test_point_repoints_a81_names():
    M.point()
    assert B.AMENDMENT == '94' and B.PHASE == 'searchedbank_a94' and B.ARMS == M.ARMS and B.CONFIRM_N == 192
    assert B.STAGE_CAP == dict(mech=14_400, confirm=921_600) and B.CAP_SIM_S == 936_000
    assert B.PREFIX == dict(i='a94i_', c='a94c_') and B.MECH_FILE == 'A94_MECH.json'
    assert B.RESULTS_FILE == 'A94_RESULTS_D2800_p60.json' and B.PANEL_C == 'A94_PANEL_C.json'
    assert B.waves is M.waves and B.analyse is M.analyse and B.mech_compare is M.mech_compare
    assert B.core is M.core and B._runner_check is M._runner_check and B.prepare is M.prepare
    assert B.BANK_ARM == 'A75D2_BANK_60C' and B.BANK_SHA256.startswith('c325ec1d')


def test_block_free_against_runner_rows():
    table = [(988600, 988699, 988600), (988700, 988899, 988700)]
    assert B.block_free(M.BLOCK, table)
    with pytest.raises(ValueError):
        B.block_free(M.BLOCK, table+[(988850, 988860, 988850)])


def _row(t, r, diff, hi, all_pass=True, kind='x'):
    return dict(treatment=t, reference=r, kind=kind, total_minus_ref=dict(diff=diff, ci95=[diff-1, hi]),
                groups=dict(all_pass=all_pass), tail_cvar90=None)


def test_primary_verdict():
    v = M.primary_verdict([_row('BANK_SEARCHED', 'BANK_HAND', -5, -0.1)])
    assert v['supported'] and v['verdict'] == 'SUPPORTED' and v['group_rule_reference'] == 'BANK_HAND'
    assert v['direction'] == 'SEARCHED_BETTER'
    v = M.primary_verdict([_row('BANK_SEARCHED', 'BANK_HAND', -5, -0.1, all_pass=False)])
    assert not v['supported'] and v['delay_pass'] and not v['group_rule_pass'] and v['direction'] == 'NOT_SEPARATED'
    v = M.primary_verdict([_row('BANK_SEARCHED', 'BANK_HAND', -5, 0.0)])
    assert not v['supported'] and not v['delay_pass']
    v = M.primary_verdict([_row('BANK_SEARCHED', 'BANK_HAND', 9, 12)])          # ci95 [8, 12]: hand better
    assert v['direction'] == 'REFERENCE_BETTER' and v['verdict'] == 'NOT SUPPORTED'
    v = M.primary_verdict([_row('BANK_SEARCHED', 'DUO_HOLD_60C', -5, -0.1)], ('BANK_SEARCHED', 'DUO_HOLD_60C'))
    assert v['group_rule_reference'] == 'DUO_HOLD_60C' and v['supported']


def test_contrast_list():
    got = list(M.contrasts(True))
    assert got[0] == ('BANK_SEARCHED', 'BANK_HAND', 'primary')
    assert [(a, r) for a, r, k in got if k == 'secondary'] == [
        ('BANK_SEARCHED', 'DUO_HOLD_60C'), ('BANK_HAND', 'DUO_HOLD_60C'), ('BANK_SEARCHED', 'MAPPO60C'),
        ('BANK_HAND', 'MAPPO60C')]
    assert ('DUO_HOLD_60C', 'MAPPO60C', 'descriptive') in got and all(a != r for a, r, _ in got)
    assert M.CONTRASTS == M.contrasts(True)
    got = list(M.contrasts(False))
    assert got[0] == ('BANK_SEARCHED', 'DUO_HOLD_60C', 'primary') and all('BANK_HAND' not in (a, r) for a, r, _ in got)


def test_summary_layout():
    rows = [_row(a, r, -1, -0.5, kind=k) for a, r, k in M.CONTRASTS]
    v = dict(analysis_sha256='x'*64, primary=M.primary_verdict(rows), contrasts=rows,
             arms=dict(BANK_SEARCHED=dict(overlap=dict(concurrent_s=3), activations_by_family={'a': 1}),
                       BANK_HAND=dict(overlap=dict(concurrent_s=0), activations_by_family={'h': 2}),
                       DUO_HOLD_60C={}, MAPPO60C={}))
    s = M.summary(v)
    json.dumps(s)
    assert s['primary']['verdict'] == 'SUPPORTED' and len(s['secondary']) == 4
    assert s['banks']['BANK_SEARCHED']['concurrent_s'] == 3 and sorted(s['banks']) == ['BANK_HAND', 'BANK_SEARCHED']


def _act(fam, t, rel):
    return dict(family=fam, t=t, members=list(rel), release={k: v for k, v in rel.items() if v is not None})


def test_overlap_and_engaged():
    r = dict(acts=[_act('L2', 400, {'c1': [40, 'hold_end']}),
                   _act('TP', 420, {'c2': [120, 'hold_end'], 'c3': [60, 'at_drop']}),
                   _act('L2', 600, {'c4': None})])
    ov = M.overlap_stats(r, horizon=1200)
    assert ov['active_s_by_family'] == {'L2': 40+600, 'TP': 120}
    assert ov['concurrent_s'] == 20
    assert ov['acts_started_while_other_active'] == 1 and ov['cav_in_two_sessions'] == 0
    assert ov['seconds_by_n_plans_active'] == {1: 40+600+120-2*20, 2: 20}
    eng = M.engaged_hist(r, horizon=1200)
    # 400-419: c1; 420-439: c1 c2 c3; 440-479: c2 c3; 480-539: c2; 600-1199: c4
    assert eng == {1: 20+60+600, 2: 40, 3: 20}
    r['acts'].append(_act('TP', 410, {'c1': [10, 'hold_end']}))
    assert M.overlap_stats(r, horizon=1200)['cav_in_two_sessions'] == 1


def test_offers_by_plan():
    got = M.offers_by_plan({'a|fire': 2, 'a|member_busy': 3, 'b|fire': 1, 'b|{"x": 1}': 4})
    assert got == {'a': {'fire': 2, 'member_busy': 3}, 'b': {'fire': 1, '{"x": 1}': 4}}


def test_split_family():
    r = dict(acts=[_act('A', 1, {'x': [1, 'hold_end']}), _act('B', 2, {'y': [1, 'hold_end']})],
             holdforms=[dict(family='A', vid='x'), dict(family='B', vid='y')], refused={})
    a = M.split_family(r, 'A')
    assert [x['family'] for x in a['acts']] == ['A'] and [e['vid'] for e in a['holdforms']] == ['x']
    assert len(r['acts']) == 2


def _doc(fams):
    return dict(schema='s', rows=[dict(family=f, strategy_key=f'k{f}', lb95_veh_s=1.0) for f in fams],
                clean5c=dict(templates=[dict(family=f, cell='D2800_p60', template=dict(source=dict(strategy_key=f'k{f}')))
                                        for f in fams]))


def test_families():
    assert M.families(dict(BANK_SEARCHED=_doc(['a', 'b']), BANK_HAND=_doc(['h']))) == \
        dict(BANK_SEARCHED=('a', 'b'), BANK_HAND=('h',))
    bad = _doc(['a', 'b'])
    bad['clean5c']['templates'].pop()
    with pytest.raises(ValueError):
        M.families(dict(BANK_SEARCHED=bad))


S5 = Path('/root/a86adm/root')
S4 = Path('/root/a85adm/root')
S5_BANK_SHA = '577a0564944b2062e441d7ee603e7fe8090bede52a5dd6e8774d41ed9cb2d8b6'
S4_BANK_SHA = 'a428cd1d320196578d55c84eea4b9b391f5e5c03928692a90dff0ac96667d7a8'
HAVE_S = (S5/'A80_ADMISSION_D2800_p60.json').is_file() and (S4/'A80_ADMISSION_D2800_p60.json').is_file()


def _build(*args):
    import subprocess
    cmd = [sys.executable, '-I', '-c',
           f"import sys; sys.path[:0]={[str(HERE.parent/'pytools'), str(HERE), str(HERE.parent)]!r}; "
           f"sys.argv=sys.argv[1:]; exec(open({str(HERE/'a94'/'build_bank_a94.py')!r}).read())", 'b', *map(str, args)]
    return subprocess.run(cmd, capture_output=True, text=True)


@pytest.mark.skipif(not HAVE_S, reason='S5 / S4 admissions not on this host')
def test_builder_searched_and_hand_standins(tmp_path):
    """Builder: searched (S5 + S4, pinned); hand with stand-ins: one root (S4 as a fake hand root), two roots (S5 + S4
    as fake hand roots: the bank rule across both), a synthetic no-plan root (NO_HAND); a wrong sha stops."""
    bank = 'a80_banks/A80_BANK_D2800_p60.json'
    r = _build('searched', '--s5-bank', S5/bank, '--s4-bank', S4/bank, '--out-dir', tmp_path/'s')
    assert r.returncode == 0, r.stdout[-2000:]+r.stderr[-2000:]
    rec = json.loads((tmp_path/'s'/'A94_BANK_SEARCHED_BUILD.json').read_text())
    s5 = json.loads((S5/bank).read_text())
    assert rec['plan_ids'][:3] == [x['family'] for x in s5['rows']]
    assert len(rec['plan_ids']) == 6-len(rec['dropped_duplicates'])
    r = _build('searched', '--s5-bank', S5/bank, '--out-dir', tmp_path/'s5')           # S5-only option
    assert r.returncode == 0, r.stdout[-2000:]+r.stderr[-2000:]
    r5 = json.loads((tmp_path/'s5'/'A94_BANK_SEARCHED_BUILD.json').read_text())
    assert r5['mode'] == 's5only' and r5['inputs']['S4'] is None and r5['bytes_equal_s5_bank'] is True
    assert r5['plan_ids'] == [x['family'] for x in s5['rows']] and r5['output']['sha256'] == S5_BANK_SHA
    assert rec['mode'] == 's5+s4' and rec['bytes_equal_s5_bank'] is False
    r = _build('hand', '--hand', f'{S4}={S4_BANK_SHA}', '--out-dir', tmp_path/'h1')
    assert r.returncode == 0, r.stdout[-2000:]+r.stderr[-2000:]
    h1 = json.loads((tmp_path/'h1'/'A94_BANK_HAND_BUILD.json').read_text())
    assert h1['status'] == 'HAND' and h1['bytes_equal_single_source'] is True
    r = _build('hand', '--hand', f'{S5}={S5_BANK_SHA}', '--hand', f'{S4}={S4_BANK_SHA}', '--out-dir', tmp_path/'h2')
    assert r.returncode == 0, r.stdout[-2000:]+r.stderr[-2000:]
    h2 = json.loads((tmp_path/'h2'/'A94_BANK_HAND_BUILD.json').read_text())
    assert h2['status'] == 'HAND' and 1 <= len(h2['plan_ids']) <= 4 and h2['choice']['chosen']
    assert set(h2['plan_ids']) == set(h2['choice']['chosen'])
    fake = tmp_path/'noroot'
    fake.mkdir()
    adm = json.loads((S5/'A80_ADMISSION_D2800_p60.json').read_text())
    adm.update(bank=None, admitted=[], n_admitted=0)
    (fake/'A80_ADMISSION_D2800_p60.json').write_text(json.dumps(adm))
    r = _build('hand', '--hand', f'{fake}=none', '--out-dir', tmp_path/'h0')
    assert r.returncode == 0 and 'NO_HAND' in r.stdout, r.stdout[-2000:]+r.stderr[-2000:]
    assert not (tmp_path/'h0'/'A94_BANK_HAND_D2800_p60.json').exists()
    r = _build('hand', '--hand', f'{S4}={S5_BANK_SHA}', '--out-dir', tmp_path/'bad')
    assert r.returncode != 0 and 'STOP' in r.stdout and not (tmp_path/'bad').exists()
