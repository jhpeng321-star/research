"""Zero-SUMO tests of the A82 part-B driver (ops/a82_fourarm.py). Optional host-root checks run when A82F_TEST_ROOT
points at a set-up root (bindings mirrored, IDENT_U0 bound)."""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import a82_fourarm as M  # noqa: E402
import a75_d2forms as D2  # noqa: E402


def test_caps_sizes_seeds():
    assert M.caps(96) == dict(mech=9_600, confirm=460_800, total=470_400)
    assert M.caps(192) == dict(mech=9_600, confirm=921_600, total=931_200)
    assert M.caps(384) == dict(mech=9_600, confirm=1_843_200, total=1_852_800)
    assert M.seeds_of(192) == tuple(range(988200, 988392)) and M.seeds_of(384)[-1] == 988583
    with pytest.raises(ValueError):
        M.seeds_of(100)
    assert M.ARMS == ('A82_BANK', 'DUO_HOLD', 'MAPPO20C', 'U0') and M.PRIMARY == ('A82_BANK', 'DUO_HOLD')
    assert set(M.SECONDARY) == {('A82_BANK', 'MAPPO20C'), ('DUO_HOLD', 'MAPPO20C'), ('MAPPO20C', 'U0')}
    assert M.PHASE == 'fourarm_a82p20' and M.PREFIX['c'] == 'a82fc_' and M.PREFIX['i'] == 'a82fi_'


def test_block_free_and_forbidden():
    lo, hi = M.BLOCK
    assert M.block_free(M.BLOCK, [(997100, 997199, 997100), (lo, hi, lo)])
    for bad in ([(lo, hi, lo), (988000, 988250, 988000)], [], [(lo, hi, lo+1)]):
        with pytest.raises(ValueError):
            M.block_free(M.BLOCK, bad)
    with pytest.raises(ValueError):
        M.block_free((989650, 989700), [(989650, 989700, 989650)])
    for b0, b1 in M.FORBIDDEN:
        assert hi < b0 or lo > b1


def test_runner_carries_the_block():
    from tools.coopmwm import nm_eval_runner as R
    if not hasattr(R, 'A82_RANGES'):
        pytest.skip('not the patched A82 runner')
    assert tuple(R.A82_RANGES) == (M.BLOCK,) and M.block_free(M.BLOCK, R.EVAL_HELDOUT_RANGES)
    assert tuple(R.A76_RANGES) == ((997100, 997199), (997200, 997299), (997300, 997399))


def test_patch_tables_a82():
    import importlib.util
    spec = importlib.util.spec_from_file_location('patch82', HERE/'a82'/'patch_tables_a82.py')
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    base = ('x\n' + P.ANCHOR + 'EVAL_HELDOUT_RANGES = ((1, 2, 3),) + tuple(\n' + P.TAIL + '\n')
    out = P.runner(base)
    assert 'A82_RANGES = ((988200, 988599),)' in out and P.runner(out) == out
    assert out.count('for lo, hi in A82_RANGES') == 1
    with pytest.raises(SystemExit):
        P.runner(base.replace('x\n', 'Y = ((988300, 988310),)\n'))


def _rec(**kw):
    r = dict(world_id='w1', world=dict(seed=997300, a=1), J_veh_s=10.5, safety=dict(collisions=0, teleports=0, hard_brakes=3),
             executor_sha256_loaded=M.FROZEN_U0_SHA256, a37=dict(activations=2, activations_by_family={'x': 2}, issued=5,
                                                                  acts=[1]))
    r.update(kw)
    return r


def test_ident_row_and_verdict():
    rep = dict(ok=True, n=3)
    ok = M.ident_row('U0', _rec(), _rec(), rep, dict(rep))
    assert ok['PASS'] and ok['report_equal']
    assert not M.ident_row('U0', _rec(J_veh_s=10.6), _rec(), rep, rep)['PASS']
    assert not M.ident_row('U0', _rec(), _rec(executor_sha256_loaded='f'*64), rep, rep)['PASS']
    assert not M.ident_row('U0', _rec(a37=dict(activations=3)), _rec(), rep, rep)['PASS']
    assert not M.ident_row('U0', _rec(), _rec(), rep, dict(ok=True, n=4))['PASS']
    duo = M.ident_row('DUO_HOLD', _rec(), _rec(), rep, None)
    assert duo['PASS'] and not duo['report_compared']
    assert not M.ident_row('A75_BANK_A76', _rec(a75_sha256_loaded='0'*64), _rec(), rep, rep)['PASS']
    good = _rec(a75_sha256_loaded=M.A75_SHA256)
    rows = [M.ident_row(a, good, good, rep, rep) for a in M.IDENT_ARMS for _ in range(2)]
    assert M.mech_verdict(rows)[0] == 'PASS'
    assert M.mech_verdict(rows[:-1])[0] == 'FAIL'
    bad = deepcopy(rows)
    bad[0]['PASS'] = False
    assert M.mech_verdict(bad)[0] == 'FAIL'


def test_primary_and_reading():
    row = dict(treatment='A82_BANK', reference='DUO_HOLD', kind='primary',
               total_minus_ref=dict(diff=-5.0, ci95=[-9.0, -1.0]), groups=dict(all_pass=True))
    assert M.primary_verdict(row)['verdict'] == 'SUPPORTED'
    assert M.primary_verdict(dict(row, groups=dict(all_pass=False)))['verdict'] == 'NOT SUPPORTED'
    assert M.primary_verdict(dict(row, total_minus_ref=dict(diff=-1, ci95=[-3.0, 0.5])))['verdict'] == 'NOT SUPPORTED'
    r = M.reading([row, dict(row, treatment='MAPPO20C', reference='U0', total_minus_ref=dict(diff=2, ci95=[1.0, 3.0]))])
    assert r['A82_BANK-DUO_HOLD']['saves'] and r['MAPPO20C-U0']['worse']


def test_rules_text():
    assert 'paired percentile bootstrap 10,000' in M.PRIMARY_RULE and 'seed' in M.PRIMARY_RULE
    assert 'DUO_HOLD' in M.PRIMARY_RULE and 'five-group 10 s' in M.PRIMARY_RULE
    assert 'a82fc_*' in M.DROP_RULE and 'a75d2' not in M.DROP_RULE
    assert M.retag([dict(job_id='x'), dict(job_id='y')], 't') == [dict(job_id='t_0000'), dict(job_id='t_0001')]


@pytest.mark.skipif(not os.environ.get('A82F_TEST_ROOT'), reason='needs a set-up host root (A82F_TEST_ROOT)')
def test_host_arm_specs_differ_only_where_registered():
    """U0 vs the MAPPO20C path (IDENT_U0) differ only in the six executor fields + job ids; DUO_HOLD = a32 on U0."""
    root = Path(os.environ['A82F_TEST_ROOT'])
    D, _ = M.setup(root, need_panel=False)
    comp = M.comparator(root)
    _, p20c = M.ident_panels(os.environ.get('A82F_TEST_A76_ROOT', '/root/a76p20/root'))
    u0 = M.M76.u0_specs(p20c, root, comp, 't')
    ip = M.policy_specs(p20c, root, comp, 'IDENT_U0', 't')
    six = {'comparator_kind', 'label', 'executor_path', 'executor_sha256', 'binding_sha256', 'binding_path'}
    for a, b in zip(u0, ip):
        assert a['job_id'] == b['job_id'] and {k for k in a if a[k] != b[k]} <= {'checkpoint'}
        diff = {k for k in set(a['checkpoint']) | set(b['checkpoint']) if a['checkpoint'].get(k) != b['checkpoint'].get(k)}
        assert diff <= six and b['checkpoint']['executor_sha256'] == M.FROZEN_U0_SHA256
    duo = M.duo_specs(p20c, root, 'd')
    assert all(s['arm'] == 'DUO_HOLD' and s['checkpoint']['kind'] == 'a32' for s in duo)
    assert [s['world']['seed'] for s in duo[:2]] == [993100, 993101]
    assert D2.check_d2 is not None
