"""A89 CPU tests (zero SUMO). Run in the A89 tree: python -m pytest -q ops/test_a89_sixarm.py"""
import json
import os
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
os.environ.setdefault('A89_ARMS_SHA256', '0'*64)

import a81_bigconfirm as B  # noqa: E402
import a89_sixarm as M  # noqa: E402


def test_constants_and_caps():
    assert M.SEEDS_C == tuple(range(988400, 988496)) and M.RESERVED_C == (988496, 988499)
    assert (M.MECH_CAP, M.CONFIRM_CAP, M.CAP_SIM_S) == (16_800, 691_200, 708_000)
    assert len(M.ARMS) == 6 and set(M.ARMS) == {'ROLL_BOTH', 'ROLL_L0_ONLY', 'ROLL_L2_ONLY', 'TAPER', 'DUO_HOLD_60C',
                                                 'MAPPO60C'}
    for lo, hi in B.FORBIDDEN:
        assert not (M.BLOCK[0] <= hi and M.BLOCK[1] >= lo)
    assert not (M.BLOCK[0] <= 988391 and M.BLOCK[1] >= 988200)          # the block reserved for another test


def test_point_repoints_a81_names():
    M.point()
    assert B.AMENDMENT == '89' and B.PHASE == 'sixarm_a89' and B.ARMS == M.ARMS and B.CONFIRM_N == 96
    assert B.STAGE_CAP == dict(mech=16_800, confirm=691_200) and B.CAP_SIM_S == 708_000
    assert B.PREFIX == dict(i='a89i_', c='a89c_') and B.MECH_FILE == 'A89_MECH.json'
    assert B.waves is M.waves and B.analyse is M.analyse and B.mech_compare is M.mech_compare
    assert B.core is M.core and B._runner_check is M._runner_check and B.prepare is M.prepare
    assert B.BANK_ARM == 'A75D2_BANK_60C' and B.BANK_SHA256.startswith('c325ec1d')


def test_block_free_against_runner_rows():
    table = [(990100, 990499, 990100), (988400, 988499, 988400)]
    assert B.block_free(M.BLOCK, table)
    with pytest.raises(ValueError):
        B.block_free(M.BLOCK, table+[(988450, 988460, 988450)])


def _row(t, r, diff, hi, all_pass=True):
    return dict(treatment=t, reference=r, total_minus_ref=dict(diff=diff, ci95=[diff-1, hi]),
                groups=dict(all_pass=all_pass), tail_cvar90=None)


def _rows(u_l0, u_l2, u_t, g=True):
    return [_row('ROLL_BOTH', 'ROLL_L0_ONLY', -5, u_l0), _row('ROLL_BOTH', 'ROLL_L2_ONLY', -5, u_l2),
            _row('TAPER', 'MAPPO60C', -5, u_t, g)]


def test_verdicts():
    abl, tap = M.verdicts(_rows(-1, -0.5, -2))
    assert abl['supported'] and abl['verdict'] == 'COOPERATION SUPPORTED' and tap['confirmed']
    abl, tap = M.verdicts(_rows(-1, 0.0, -2, g=False))
    assert not abl['supported'] and abl['pass_vs_l0'] and not abl['pass_vs_l2']
    assert not tap['confirmed'] and tap['delay_pass'] and not tap['group_rule_pass']
    abl, tap = M.verdicts(_rows(3, -1, 1))
    assert not abl['supported'] and not tap['confirmed']


def test_interaction_identity():
    """(B - M) - [(L0 - M) + (L2 - M)] == (B + M) - (L0 + L2) per world (the vector fed to fe.paired_stats)."""
    import numpy as np
    rng = np.random.default_rng(0)
    b, m, l0, l2 = (rng.normal(30000, 2000, 96) for _ in range(4))
    assert np.allclose((b-m)-((l0-m)+(l2-m)), (b+m)-(l0+l2))


def test_contrast_list():
    got = [(a, r) for a, r, _ in M.CONTRASTS]
    assert ('ROLL_BOTH', 'ROLL_L0_ONLY') in got and ('ROLL_BOTH', 'ROLL_L2_ONLY') in got
    assert ('TAPER', 'MAPPO60C') in got and ('TAPER', 'DUO_HOLD_60C') in got and ('DUO_HOLD_60C', 'MAPPO60C') in got
    assert all(a != r for a, r in got)


ARMS_FILE = HERE/'a89'/'A89_ARMS_D2800_p60.json'


@pytest.mark.skipif(not ARMS_FILE.is_file(), reason='arms bank not built in this tree')
def test_arms_bank_derivation():
    from tools.coopmwm import a75_holdforms as H
    doc = json.loads(ARMS_FILE.read_text(encoding='utf-8'))
    t = {x['family']: x for x in doc['clean5c']['templates']}
    both = t[M.ARM_FAMILY['ROLL_BOTH']]['offer_condition']
    l0 = t[M.ARM_FAMILY['ROLL_L0_ONLY']]['offer_condition']
    l2 = t[M.ARM_FAMILY['ROLL_L2_ONLY']]['offer_condition']
    assert [r['sumo_lane'] for r in both['roles']] == [0, 2]
    assert l0['roles'] == [both['roles'][0]] and l2['roles'] == [both['roles'][1]]
    assert l2['roles'][0]['onset_s'] == 20 and l0['roles'][0]['v_w'] == 24.0 and l2['roles'][0]['v_w'] == 16.0
    for c in (l0, l2):
        assert {k: v for k, v in c.items() if k != 'roles'} == {k: v for k, v in both.items() if k != 'roles'}
    for arm, fam in M.ARM_FAMILY.items():
        core = H.make_core('A75_PLAN', M.CELL, doc, family=fam)
        assert [p['family'] for p in core.plans] == [fam]
    assert len(t[M.ARM_FAMILY['TAPER']]['offer_condition']['roles']) == 4
