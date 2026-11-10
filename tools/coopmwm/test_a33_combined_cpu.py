"""Amendment 33 COMBINED (CPU only, no SUMO): LEARNED_TRIGGER executor on WM fd385a2b + duo12_v20 (appended last),
retrained-classifier gate; classifier data helpers."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.coopmwm import a32_arms
from tools.coopmwm import a33_classifier as K
from tools.coopmwm import a33_combined as C
from tools.coopmwm import learned_trigger as L
from tools.coopmwm import oracle_plan as O
from tools.coopmwm.test_learned_trigger_cpu import synthetic_model, wm_rows

CELL = 'D2800_p20'
BASE = Path('E:/research-dense/runs/active/C4-V7FT-20261017')
WM = BASE/'paper_data/train/_setup_cpuZ/GUIDANCE_R6_WMONLY_PRIORITY.json'
UNION = BASE/'wmverify_unc/GUIDANCE_C4SIT_priv_UNION_UNC.json'
U5 = BASE/'paper_data/train/_setup_cpuZ/GUIDANCE_UNION_R6_UNC_NOF2B_PRIORITY.json'
HAVE = WM.is_file() and UNION.is_file()


def docs():
    return json.loads(WM.read_text(encoding='utf-8')), json.loads(UNION.read_text(encoding='utf-8'))


def model6(threshold):
    m = synthetic_model(C.FAMILIES, threshold=threshold)
    wm, un = docs()
    plans, _ = C.ordered_plans(C.combined_bank(wm, un, CELL), CELL)
    m['strategy_keys'] = {p['family']: p['key'] for p in plans}
    return m


def run(core, rows, ts=range(300, 306)):
    cmds = {r['vid']: 0.0 for r in rows if r['is_cav']}
    return [core.step(rows, cmds, t) for t in ts]


def test_constants():
    assert C.DUO_KEY == a32_arms.DUO_KEY and C.DUO_FAMILY == a32_arms.DUO_FAMILY
    assert C.DUO_BANK_SHA256 == a32_arms.DUO_BANK_SHA256
    assert C.FAMILIES == tuple(sorted(C.FAMILIES)) and len(C.FAMILIES) == 6
    assert len(L.feature_names(list(C.FAMILIES))) == 69


@pytest.mark.skipif(not HAVE, reason='bank docs not on this host')
def test_bank_shas_and_order():
    assert K.T.sha(WM) == C.WM_BANK_SHA256 and K.T.sha(UNION) == C.DUO_BANK_SHA256
    wm, un = docs()
    plans, refused = C.ordered_plans(C.combined_bank(wm, un, CELL), CELL)
    lt, _ = O.bank_plans(wm, CELL, 'wm')
    assert not refused
    # the LEARNED_TRIGGER WM order, then duo12_v20 last
    assert [p['family'] for p in plans] == [p['family'] for p in lt] + [C.DUO_FAMILY]
    assert plans[-1]['key'] == C.DUO_KEY and plans[-1]['plan_source'] == 'rule'
    # identical plans (keys/specs) to the amendment-27 ORACLE union 5f6796c5 when it is present
    if U5.is_file():
        u5, _ = O.bank_plans(json.loads(U5.read_text(encoding='utf-8')), CELL, None)
        assert [(p['family'], p['key']) for p in u5] == [(p['family'], p['key']) for p in plans]


@pytest.mark.skipif(not HAVE, reason='bank docs not on this host')
def test_core_gate_and_receipts():
    wm, un = docs()
    doc = C.combined_bank(wm, un, CELL)
    rows = wm_rows(speed=25.0)
    never = C.CombinedCore(doc, CELL, model6(0.99))      # synthetic p = 0.5
    fire = C.CombinedCore(doc, CELL, model6(0.01))
    assert [p['family'] for p in fire.plans][-1] == C.DUO_FAMILY
    assert all(x == ({}, {}) for x in run(never, rows))
    run(fire, rows)
    assert fire.scores and all(s[3] for s in fire.scores)
    r = fire.receipts()
    assert r['executor'] == 'combined' and r['order'][-1] == C.DUO_FAMILY
    # a model without duo12_v20 is refused (fail closed)
    m5 = synthetic_model(C.WM_FAMILIES, threshold=0.5)
    m5['strategy_keys'] = {p['family']: p['key'] for p in O.bank_plans(wm, CELL, 'wm')[0]}
    with pytest.raises(ValueError):
        C.CombinedCore(doc, CELL, m5)


@pytest.mark.skipif(not HAVE, reason='bank docs not on this host')
def test_duo_alone_matches_duo_hold_when_only_duo_fires():
    """With the gate open only for duo12_v20, COMBINED acts exactly like DUO_HOLD (same executor)."""
    wm, un = docs()
    doc = C.combined_bank(wm, un, CELL)
    m = model6(0.5)
    n = len(m['coef'])
    m['coef'] = [0.0]*n
    m['intercept'] = -10.0
    m['coef'][L.feature_names(list(C.FAMILIES)).index('plan_' + C.DUO_FAMILY)] = 40.0
    m['mean'], m['scale'] = [0.0]*n, [1.0]*n
    rows = wm_rows(speed=25.0)
    comb = C.CombinedCore(doc, CELL, m)
    duo = a32_arms.make_core('DUO_HOLD', CELL, union_doc=un)
    assert run(comb, rows) == run(duo, rows)
    assert [a['members'] for a in comb.acts] == [a['members'] for a in duo.acts]


def test_widen_and_digest():
    old = list(C.WM_FAMILIES)
    f = [1.0]*63 + [0.0, 1.0, 0.0, 0.0, 0.0]
    w = K.widen(f, old)
    assert len(w) == 69 and w[:63] == f[:63]
    assert w[63:] == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]          # duo12_v20 sorts first
    assert C.a33_digest() == C.a33_digest()


def test_acts_doc():
    def rec(j, acts=()):
        return dict(J_veh_s=j, a32=dict(acts=[dict(t=t, family=C.DUO_FAMILY, members=['a', 'b']) for t in acts]))
    acts = K.acts_doc({(991600, 'DUO_HOLD'): rec(100.0, (400, 700)), (991600, 'U0'): rec(150.0),
                       (991601, 'DUO_HOLD'): rec(200.0, (500,)), (991601, 'U0'): rec(150.0)})
    assert [(a['seed'], a['t'], a['diff']) for a in acts] == [(991600, 400, -50.0), (991600, 700, -50.0),
                                                                (991601, 500, 50.0)]
    assert all(a['ref_arms'] == ['u0'] and a['lookback'] == 10 for a in acts)
    with pytest.raises(ValueError):
        K.acts_doc({(993100, 'DUO_HOLD'): rec(1.0, (400,)), (993100, 'U0'): rec(1.0)})


def test_build_rows_bg_weight_union():
    orig = [dict(tape='x', t=300, family='f', seed=991000, kind='harm', y=0, w=4.0),
            dict(tape='x', t=305, family='f', seed=991000, kind='bg', y=0, w=1.0)]
    rows, _ = K.build_rows(orig, [], [])
    assert [r['w'] for r in rows if r['kind'] == 'bg'] == [K.T.BG_SHARE*4.0]


def test_runner_wiring_and_ranges():
    from tools.coopmwm import map_scenarios as ms, nm_eval_runner as R
    R.install_eval_ranges(ms)
    assert 'COMBINED' in R.A33_ARMS
    assert ms.seed_scope(999500) == 'heldout' and ms.seed_scope(991600) == 'heldout'
    assert ms.world(999500, 2800, 0.2)['repeat'] == 0 and ms.world(993100, 2800, 0.2)['repeat'] == 100
    assert ms.seed_scope(999700) is None and ms.seed_scope(991900) is None
