"""Zero-SUMO tests of the A96 parts (c)/(d) driver (ops/a96_p20.py) and builder (ops/a96/build_bank_a96.py).
Layout constants of both parts and both arm sets are checked in fresh interpreters (the arm set is fixed at import)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
os.environ.setdefault('A96_PART', 'c')
os.environ.setdefault('A96_HAND_SHA256', '0'*64)

import a96_p20 as M  # noqa: E402


def _probe(part, hand, code):
    env = dict(os.environ, A96_PART=part, A96_HAND_SHA256=hand)
    pre = f"import sys; sys.path[:0]={[str(HERE), str(HERE.parent)]!r}; import a96_p20 as M; "
    return subprocess.run([sys.executable, '-c', pre+code], capture_output=True, text=True, env=env)


@pytest.mark.parametrize('part,hand', [('c', '0'*64), ('c', 'none'), ('d', '0'*64), ('d', 'none')])
def test_layouts(part, hand):
    code = ("import json; print(json.dumps(dict(arms=M.ARMS, primary=M.PRIMARY, sec=M.SECONDARY, desc=M.DESCRIPTIVE, "
            "block=M.BLOCK, seeds=[M.seeds_of(192)[0], M.seeds_of(192)[-1]], caps=M.caps(192), mech=M.MECH_EPISODES, "
            "attr=M.RUNNER_ATTR, pre=M.PREFIX, files=[M.PANEL_FILE, M.MECH_FILE, M.RESULTS_FILE, M.BANK_PIN])))")
    r = _probe(part, hand, code)
    assert r.returncode == 0, r.stderr[-2000:]
    d = json.loads(r.stdout)
    s = 'BANK_SEARCHED_P20' if part == 'c' else 'A82_BANK'
    h = 'BANK_HAND_P20'
    if hand == 'none':
        assert d['arms'] == [s, 'DUO_HOLD', 'MAPPO20C'] and d['primary'] == [s, 'DUO_HOLD'] and d['mech'] == 10
        assert d['sec'] == [[s, 'DUO_HOLD'], [s, 'MAPPO20C']]
        assert d['caps'] == dict(mech=14_400, confirm=691_200, total=705_600)
    else:
        assert d['arms'] == [s, h, 'DUO_HOLD', 'MAPPO20C'] and d['primary'] == [s, h] and d['mech'] == 12
        assert d['sec'] == [[s, 'DUO_HOLD'], [h, 'DUO_HOLD'], [s, 'MAPPO20C'], [h, 'MAPPO20C']]
        assert d['caps'] == dict(mech=14_400, confirm=921_600, total=936_000)
    assert d['desc'] == [['DUO_HOLD', 'MAPPO20C']] and 'U0' not in d['arms']
    if part == 'c':
        assert d['block'] == [984400, 984599] and d['seeds'] == [984400, 984591] and d['attr'] == 'A96C_RANGES'
    else:
        assert d['block'] == [988200, 988599] and d['seeds'] == [988200, 988391] and d['attr'] == 'A82_RANGES'
    assert d['pre']['c'] == f'a96{part}c_' and all(f.startswith(f'A96{part.upper()}_') for f in d['files'])


def test_env_required():
    assert _probe('x', 'none', 'print(1)').returncode != 0
    assert _probe('c', '', 'print(1)').returncode != 0


def test_block_free_and_forbidden():
    lo, hi = M.BLOCK
    other = [(997100, 997199, 997100)] + ([(988200, 988599, 988200)] if M.PART == 'c' else [])
    assert M.block_free(M.BLOCK, other+[(lo, hi, lo)])
    for bad in ([(lo, hi, lo), (lo+100, lo+110, lo+100)], [], [(lo, hi, lo+1)]):
        with pytest.raises(ValueError):
            M.block_free(M.BLOCK, bad)
    for b0, b1 in M.FORBIDDEN:
        assert hi < b0 or lo > b1


def test_patch_tables_a96c():
    import importlib.util
    spec = importlib.util.spec_from_file_location('patch96c', HERE/'a96'/'patch_tables_a96c.py')
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    base = ('x\n' + P.ANCHOR + 'EVAL_HELDOUT_RANGES = ((1, 2, 3),) + tuple(\n' + P.TAIL + '\n')
    out = P.runner(base)
    assert 'A96C_RANGES = ((984400, 984599),)' in out and P.runner(out) == out
    assert out.count('for lo, hi in A96C_RANGES') == 1
    with pytest.raises(SystemExit):
        P.runner(base.replace('x\n', 'Y = ((984500, 984510),)\n'))


def test_runner_carries_the_block():
    from tools.coopmwm import nm_eval_runner as R
    if not hasattr(R, 'A96C_RANGES'):
        pytest.skip('not the patched A96 (c) runner')
    assert tuple(R.A96C_RANGES) == ((984400, 984599),) and M.block_free(M.BLOCK, R.EVAL_HELDOUT_RANGES)
    assert tuple(R.A82_RANGES) == ((988200, 988599),)


def _rec(**kw):
    r = dict(world_id='w1', world=dict(seed=997300, a=1), J_veh_s=10.5, safety=dict(collisions=0, teleports=0, hard_brakes=3),
             executor_sha256_loaded=M.FROZEN_U0_SHA256, a37=dict(activations=2, activations_by_family={'x': 2}, issued=5,
                                                                  acts=[1]))
    r.update(kw)
    return r


def test_ident_row_and_verdict():
    rep = dict(ok=True, n=3)
    assert M.ident_row('U0', _rec(), _rec(), rep, dict(rep))['PASS']
    assert not M.ident_row('U0', _rec(J_veh_s=10.6), _rec(), rep, rep)['PASS']
    good = _rec(a75_sha256_loaded=M.A75_SHA256)
    rows = [M.ident_row(a, good, good, rep, rep) for a in M.IDENT_ARMS for _ in range(2)]
    assert M.mech_verdict(rows)[0] == 'PASS'
    assert M.mech_verdict(rows[:-1])[0] == 'FAIL'
    assert M.mech_verdict(rows, ['BANK_HAND_P20 w: x'])[0] == 'FAIL'


def _row(t, r, diff, lo, hi, all_pass=True, kind='x'):
    return dict(treatment=t, reference=r, kind=kind, total_minus_ref=dict(diff=diff, ci95=[lo, hi]),
                groups=dict(all_pass=all_pass), tail_cvar90=None)


def test_primary_direction():
    s, h = M.PRIMARY
    v = M.primary_verdict(_row(s, h, -5, -9, -1))
    assert v['verdict'] == 'SUPPORTED' and v['direction'] == 'SEARCHED_BETTER' and v['group_rule_reference'] == h
    assert M.primary_verdict(_row(s, h, -5, -9, -1, all_pass=False))['direction'] == 'NOT_SEPARATED'
    assert M.primary_verdict(_row(s, h, -1, -3, 0.5))['verdict'] == 'NOT SUPPORTED'
    assert M.primary_verdict(_row(s, h, 5, 1, 9))['direction'] == 'REFERENCE_BETTER'
    assert M.contrasts()[0] == (s, h, 'primary') and len(M.contrasts()) == 6


def _act(fam, t, rel):
    return dict(family=fam, t=t, members=list(rel), release={k: v for k, v in rel.items() if v is not None})


def test_overlap_engaged_offers():
    r = dict(acts=[_act('L2', 400, {'c1': [40, 'hold_end']}),
                   _act('TP', 420, {'c2': [120, 'hold_end'], 'c3': [60, 'at_drop']}),
                   _act('L2', 600, {'c4': None})])
    ov = M.overlap_stats(r, horizon=1200)
    assert ov['concurrent_s'] == 20 and ov['seconds_by_n_plans_active'] == {1: 720, 2: 20}
    assert M.engaged_hist(r, horizon=1200) == {1: 680, 2: 40, 3: 20}
    assert M.offers_by_plan({'a|fire': 2, 'a|member_busy': 3}) == {'a': {'fire': 2, 'member_busy': 3}}


def test_summary_layout():
    rows = [_row(a, r, -1, -2, -0.5, kind=k) for a, r, k in M.contrasts()]
    v = dict(analysis_sha256='x'*64, primary=M.primary_verdict(rows[0]), contrasts=rows,
             arms={a: (dict(overlap=dict(concurrent_s=1), activations_by_family={}) if a in M.BANK_ARMS else {})
                   for a in M.ARMS})
    s = M.summary(v)
    json.dumps(s)
    assert len(s['secondary']) == 4 and sorted(s['banks']) == sorted(M.BANK_ARMS) and 'direction' in s['primary']


def test_rules_text():
    assert 'paired' in M.PRIMARY_RULE and '10,000' in M.PRIMARY_RULE and M.PRIMARY[1] in M.PRIMARY_RULE
    assert M.PREFIX['c']+'*' in M.DROP_RULE and 'a75d2' not in M.DROP_RULE


# ------------------------------------------------------------------------------------------------ builder (stand-ins)

S60 = dict(S5=(Path('/root/a86adm/root'), '577a0564944b2062e441d7ee603e7fe8090bede52a5dd6e8774d41ed9cb2d8b6'),
           S4=(Path('/root/a85adm/root'), 'a428cd1d320196578d55c84eea4b9b391f5e5c03928692a90dff0ac96667d7a8'))
HAVE_60 = all((r/'A80_ADMISSION_D2800_p60.json').is_file() for r, _ in S60.values())


def _fake_p20_root(dst, src_root, empty=False):
    """Synthetic p20 admission root from a 60 % A80 admission (TEST ONLY): bank rows' regime and template cell rewritten
    to D2800_p20 (plans unchanged), receipt cell / bank sha updated. empty=True: a receipt with no bank."""
    adm = json.loads((src_root/'A80_ADMISSION_D2800_p60.json').read_text())
    adm['cell'] = 'D2800_p20'
    (dst/'a80_banks').mkdir(parents=True)
    if empty:
        adm.update(bank=None, admitted=[], n_admitted=0)
        sha = 'none'
    else:
        b = json.loads((src_root/'a80_banks'/'A80_BANK_D2800_p60.json').read_text())
        for r in b['rows']:
            r['regime'] = 'D2800_p20'
        for t in b['clean5c']['templates']:
            t['cell'] = 'D2800_p20'
        text = json.dumps(b, indent=1, sort_keys=True)+'\n'
        (dst/'a80_banks'/'A80_BANK_D2800_p20.json').write_text(text, newline='\n')
        sha = hashlib.sha256(text.encode()).hexdigest()
        adm['bank'] = dict(adm['bank'], file_sha256=sha)
    (dst/'A80_ADMISSION_D2800_p20.json').write_text(json.dumps(adm))
    return f'{dst}={sha}'


def _build(*args):
    cmd = [sys.executable, '-I', '-c',
           f"import sys; sys.path[:0]={[str(HERE.parent/'pytools'), str(HERE), str(HERE.parent)]!r}; "
           f"sys.argv=sys.argv[1:]; exec(open({str(HERE/'a96'/'build_bank_a96.py')!r}).read())", 'b', *map(str, args)]
    return subprocess.run(cmd, capture_output=True, text=True)


@pytest.mark.skipif(not HAVE_60, reason='60 % admissions (stand-in sources) not on this host')
def test_builder_standins(tmp_path):
    a = _fake_p20_root(tmp_path/'s4', S60['S4'][0])
    b = _fake_p20_root(tmp_path/'s5', S60['S5'][0])
    e = _fake_p20_root(tmp_path/'e', S60['S5'][0], empty=True)
    base = ['--cell', 'D2800_p20', '--prefix', 'TEST']
    r = _build('searched', *base, '--src', a, '--src', b, '--out-dir', tmp_path/'o2')
    assert r.returncode == 0, r.stdout[-1500:]+r.stderr[-1500:]
    rec = json.loads((tmp_path/'o2'/'TEST_BANK_SEARCHED_BUILD.json').read_text())
    assert len(rec['plan_ids']) == 6 and rec['plan_ids'][3:] == ['rand_8b08c77a63f7d622', 'mut_069966492c9abdcc',
                                                                  'mut_97dda2be6b45ccf0']
    r = _build('searched', *base, '--src', e, '--src', b, '--out-dir', tmp_path/'o1')
    assert r.returncode == 0 and json.loads((tmp_path/'o1'/'TEST_BANK_SEARCHED_BUILD.json').read_text())[
        'bytes_equal_single_source'] is True
    r = _build('searched', *base, '--src', e, '--out-dir', tmp_path/'o0')
    assert r.returncode == 8 and 'NO_SEARCHED' in r.stdout and not (tmp_path/'o0').exists()
    r = _build('hand', *base, '--hand', a, '--out-dir', tmp_path/'h1')
    assert r.returncode == 0 and 'BUILD_BANK_A96_OK' in r.stdout
    r = _build('hand', *base, '--hand', e, '--out-dir', tmp_path/'h0')
    assert r.returncode == 0 and 'NO_HAND' in r.stdout and not (tmp_path/'h0'/'TEST_BANK_HAND_D2800_p20.json').exists()
    r = _build('hand', *base, '--hand', a.split('=')[0]+'='+'1'*64, '--out-dir', tmp_path/'bad')
    assert r.returncode != 0 and not (tmp_path/'bad').exists()
