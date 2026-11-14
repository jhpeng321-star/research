"""Amendment 35 transfer (CPU only, no SUMO): the runner admits the frozen bank rows under their source cell
D2800_p20 when ck['bank_cell_key'] is set, the world stays the transfer cell, and every pre-a35 spec is unchanged."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.coopmwm import a32_arms as A
from tools.coopmwm import learned_trigger as L
from tools.coopmwm import map_scenarios as ms
from tools.coopmwm import nm_eval_runner as R
from tools.coopmwm.test_learned_trigger_cpu import fam_bank, keyed, synthetic_model, wm_rows

UNION = Path('E:/research-dense/runs/active/C4-V7FT-20261017/wmverify_unc/GUIDANCE_C4SIT_priv_UNION_UNC.json')


def test_bank_cell_absent_key_is_world_cell():
    assert R.bank_cell({}, 'D2800_p20') == 'D2800_p20'
    assert R.bank_cell({'cell_key': 'D2400_p20'}, 'D2400_p20') == 'D2400_p20'


def test_bank_cell_transfer_and_refusal():
    assert R.bank_cell({'bank_cell_key': 'D2800_p20'}, 'D3200_p20') == 'D2800_p20'
    with pytest.raises(ValueError):
        R.bank_cell({'bank_cell_key': 'D3200_p20'}, 'D3200_p20')


def test_a35_eval_ranges_registered():
    R.install_eval_ranges(ms)
    for lo, d, p in ((998000, 2400, .2), (998100, 3200, .2), (998200, 2800, .6), (998300, 2800, .4)):
        w = ms.world(lo+5, d, p)
        assert w['repeat'] == 5 and w['cell_key'] == ms.cell_key(d, p)
        assert ms.seed_scope(lo+95) == 'heldout'
    w = ms.world(993105, 2800, .2)                    # pre-a35 worlds unchanged
    assert w['repeat'] == 105


def test_transfer_cell_has_no_own_rows_but_source_rows_run():
    fams = ('mut_a', 'mut_b')
    doc = fam_bank(fams)
    model = keyed(synthetic_model(fams, threshold=0.01), doc)
    other = L.LearnedCore(doc, 'D3200_p20', model)
    assert other.plans == []                          # without bank_cell_key a transfer cell admits nothing
    src = L.LearnedCore(doc, R.bank_cell({'bank_cell_key': 'D2800_p20'}, 'D3200_p20'), model)
    assert [p['family'] for p in src.plans] == [p['family'] for p in L.LearnedCore(doc, 'D2800_p20', model).plans]
    assert src.plans


@pytest.mark.skipif(not UNION.is_file(), reason='UNION_UNC not on this host')
def test_duo_transfer_uses_source_row():
    doc = json.loads(UNION.read_text(encoding='utf-8'))
    with pytest.raises(ValueError):
        A.make_core('DUO_HOLD', 'D2400_p20', union_doc=doc)
    core = A.make_core('DUO_HOLD', R.bank_cell({'bank_cell_key': 'D2800_p20'}, 'D2400_p20'), union_doc=doc)
    assert [p['family'] for p in core.plans] == ['duo12_v20']
    cmds = {r['vid']: 0.0 for r in wm_rows(speed=25.0) if r['is_cav']}
    accel, _ = core.step(wm_rows(speed=25.0), cmds, 300)
    assert set(accel) == {'c0', 'c1'}
