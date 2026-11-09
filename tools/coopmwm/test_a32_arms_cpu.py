"""Amendment 32 arms (CPU only, no SUMO): same executor as LEARNED_TRIGGER, only the start rule / plan set differs."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.coopmwm import a32_arms as A
from tools.coopmwm import learned_trigger as L
from tools.coopmwm import oracle_plan as O
from tools.coopmwm.test_learned_trigger_cpu import fam_bank, keyed, synthetic_model, wm_rows

CELL = 'D2800_p20'
UNION = Path('E:/research-dense/runs/active/C4-V7FT-20261017/wmverify_unc/GUIDANCE_C4SIT_priv_UNION_UNC.json')


def run(core, rows, ts=range(300, 306)):
    out = []
    cmds = {r['vid']: 0.0 for r in rows if r['is_cav']}
    for t in ts:
        out.append(core.step(rows, cmds, t))
    return out


def test_solo_spec_is_the_incumbent():
    assert O.strategy_key(A.SOLO_SPEC) == A.SOLO_KEY
    core = A.make_core('SOLO_HOLD', CELL)
    assert [p['family'] for p in core.plans] == ['wall_g_solo2'] and core.plans[0]['row_class'] == 'unconditional'


def test_solo_acts_whenever_feasible_lane2_member_only():
    rows = wm_rows(speed=25.0)
    core = A.make_core('SOLO_HOLD', CELL)
    steps = run(core, rows)
    accel, lanes = steps[0]
    # WM lane 0 == SUMO lane 2: only the lane-2 member holds; law clip(22 - 25, -1, 1) = -1
    assert accel == {'c0': -1.0}
    assert len(core.acts) == 1 and core.acts[0]['members'] == ['c0']
    assert all(len(a) == 1 for a, _ in steps)   # holds on, no second activation before the cooldown


@pytest.mark.skipif(not UNION.is_file(), reason='UNION_UNC not on this host')
def test_duo_is_the_union_rule_row():
    doc = json.loads(UNION.read_text(encoding='utf-8'))
    core = A.make_core('DUO_HOLD', CELL, union_doc=doc)
    assert [(p['family'], p['key'], p['plan_source']) for p in core.plans] == [('duo12_v20', A.DUO_KEY, 'rule')]
    accel, _ = run(core, wm_rows(speed=25.0))[0]
    assert set(accel) == {'c0', 'c1'}            # SUMO lanes 2 and 1 hold at 20 m/s


def test_bank_always_equals_learned_trigger_with_gate_open():
    fams = ('mut_a', 'mut_b')
    doc = fam_bank(fams)
    rows = wm_rows(speed=25.0)
    hold = keyed(synthetic_model(fams, threshold=0.99), doc)      # p = 0.5 < 0.99: LEARNED_TRIGGER never fires
    always = A.make_core('BANK_ALWAYS', CELL, bank_doc=doc, model=hold)
    lt_hold = L.LearnedCore(doc, CELL, hold)
    lt_fire = L.LearnedCore(doc, CELL, keyed(synthetic_model(fams, threshold=0.01), doc))
    assert [p['family'] for p in always.plans] == [p['family'] for p in lt_hold.plans]
    a, f, h = run(always, rows), run(lt_fire, rows), run(lt_hold, rows)
    assert a == f and all(x == ({}, {}) for x in h)
    assert [x['members'] for x in always.acts] == [x['members'] for x in lt_fire.acts]
    assert always.scores == []                  # the classifier is never evaluated


def test_digest_stable_and_hook_wraps():
    d = A.a32_digest()
    assert d == A.a32_digest() and len(d) == 64

    class C:
        def apply(self, rows, commands, now):
            return {}
    hook = A.make_hook('SOLO_HOLD', CELL)
    c = C()
    hook(c, None, None)
    rows = wm_rows(speed=25.0)
    out = c.apply(rows, {r['vid']: 0.3 for r in rows if r['is_cav']}, 300)
    assert out['c0']['requested_raw'] == -1.0 and out['c0']['oracle_plan'] == 'wall_g_solo2'
    assert hook.receipts()['a32_arm'] == 'SOLO_HOLD'
