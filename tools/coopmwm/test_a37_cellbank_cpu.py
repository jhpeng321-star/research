"""Amendment 37 cell-bank arms (CPU only, no SUMO): candidate selection from the WM ranking only, the BANK_ALWAYS
executor on the cell's own plans, the UNC admission rule, the runner hook and the verification seed ranges."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.coopmwm import a32_arms as A32
from tools.coopmwm import a37_cellbank as C
from tools.coopmwm import map_scenarios as ms
from tools.coopmwm import nm_eval_runner as R
from tools.coopmwm import oracle_plan as O
from tools.coopmwm.test_learned_trigger_cpu import wm_rows

CELL = 'D3200_p20'
SEARCH = Path('E:/research-dense/tmp/wmc_p20_r3/fetched')
OPS = Path(__file__).resolve().parents[2]/'ops'/'a37'


def spec(name, roles, **kw):
    s = dict(name=name, trigger='wall', zone='full', spread_m=200.0, cooldown_s=120, slots=1, gate=None,
             start_gate=None, phases=None, roles=roles)
    s.update(kw)
    return s


def role(lane, v_w=20.0, dmax=1.0, hold_s=120, onset_s=0, lc_target=None):
    return dict(lane=lane, rank=0, v_w=v_w, dmax=dmax, hold_s=hold_s, onset_s=onset_s, lc_target=lc_target)


DUO = spec('p_duo', [role(1, 20.0), role(2, 21.0)])
SOLO = spec('p_solo', [role(2, 22.0)], cooldown_s=90)


def plans(*specs, lb=None):
    return [dict(family=s['name'], strategy=s, lb95_veh_s=(lb or {}).get(s['name'], 0.0)) for s in specs]


def run(core, rows, ts=range(300, 306)):
    cmds = {r['vid']: 0.0 for r in rows if r['is_cav']}
    return [core.step(rows, cmds, t) for t in ts]


def test_k_is_fixed_by_the_verification_cap():
    assert C.k_per_cell() == 9 and C.k_per_cell(1) == 12          # 2 cells: (600,000 - 24,000) // 57,600 - 1
    assert (C.k_per_cell()+1)*C.N_SEEDS*C.EPISODE_SIM_S*2+C.VERIFY_RESERVE_SIM_S <= C.VERIFY_CAP_SIM_S


def test_executor_refusal_classes():
    assert C.executor_refusal(DUO) is None
    assert C.executor_refusal(spec('x', [role(1)], phases=[[0]])) == 'gate_or_phases'
    assert C.executor_refusal(spec('x', [role(1)], gate=7.0)) == 'gate_or_phases'
    assert C.executor_refusal(spec('x', [role(1, lc_target=2)])) == 'non_hold_plan'
    assert C.executor_refusal(spec('x', [role(1)], start_gate='empty100')) is None


@pytest.mark.parametrize('cell', C.CELLS)
def test_candidates_are_the_wm_ranking_only(cell):
    path = SEARCH/f'STRATEGY_SEARCH_{cell}.json'
    if not path.is_file():
        pytest.skip('search file not on this host')
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == C.SEARCH_SHA256[cell]
    sel = C.select_candidates(json.loads(raw), cell)
    assert len(sel['candidates']) == 9
    scores = [c['score'] for c in sel['candidates']]
    assert scores == sorted(scores, reverse=True)
    assert all(C.executor_refusal(c['strategy']) is None for c in sel['candidates'])
    ranked = [r for r in sel['ranking'] if r['skip'] is None][:9]
    assert [r['name'] for r in ranked] == [c['family'] for c in sel['candidates']]
    assert not any(r['skip'] is None and r['score'] > scores[-1] and r['name'] not in [c['family'] for c in sel['candidates']]
                   for r in sel['ranking'])
    committed = json.loads((OPS/f'CANDIDATES_{cell}.json').read_text(encoding='utf-8'))
    assert [c['family'] for c in committed['candidates']] == [c['family'] for c in sel['candidates']]


def test_bank_doc_rows_are_unconditional_and_keyed():
    doc = C.bank_doc(plans(DUO, SOLO, lb={'p_solo': 50.0, 'p_duo': 10.0}), CELL)
    core = C.CellBankCore(doc, CELL)
    assert [p['family'] for p in core.plans] == ['p_solo', 'p_duo']          # priority = lb95 desc
    assert all(p['row_class'] == 'unconditional' for p in core.plans) and core.refused == {}
    assert [p['key'] for p in core.plans] == [O.strategy_key(SOLO), O.strategy_key(DUO)]
    assert C.CellBankCore(doc, 'D2400_p20').plans == []                       # rows bind their own cell only


def test_start_rule_is_the_bank_always_rule():
    assert C.CellBankCore._gate is A32.AlwaysCore._gate
    core = C.make_core('CELL_BANK', CELL, bank=C.bank_doc(plans(DUO), CELL))
    accel, lanes = run(core, wm_rows(speed=25.0))[0]
    assert accel == {'c0': -1.0, 'c1': -1.0}            # SUMO lanes 2 / 1 (WM 0 / 1): clip(v_w - 25, -1, 1)
    assert core.acts[0]['family'] == 'p_duo' and core.acts[0]['row_class'] == 'unconditional'
    assert core.receipts()['executor'] == 'cell_bank_always'


def test_a37_plan_runs_exactly_one_family():
    bank = C.bank_doc(plans(DUO, SOLO), CELL)
    core = C.make_core('A37_PLAN', CELL, bank=bank, family='p_solo')
    assert [p['family'] for p in core.plans] == ['p_solo']
    assert run(core, wm_rows(speed=25.0))[0][0] == {'c0': -1.0}
    with pytest.raises(ValueError):
        C.make_core('A37_PLAN', CELL, bank=bank, family='nope')
    with pytest.raises(ValueError):
        C.make_core('CELL_BANK', CELL, bank=C.bank_doc([], CELL))     # empty bank never runs (CELL_BANK := U0)


def test_bank_doc_refuses_unexecutable_plans():
    with pytest.raises(ValueError):
        C.bank_doc(plans(spec('x', [role(1)], phases=[[0]])), CELL)


def test_hook_wraps_and_never_commands_hdvs():
    class Ctl:
        def apply(self, rows, commands, now):
            return {}
    hook = C.make_hook('CELL_BANK', CELL, bank=C.bank_doc(plans(SOLO), CELL))
    c = Ctl()
    hook(c, None, None)
    rows = wm_rows(speed=25.0)
    out = c.apply(rows, {r['vid']: 0.3 for r in rows if r['is_cav']}, 300)
    assert set(out) == {'c0'} and out['c0']['oracle_plan'] == 'p_solo'
    assert not any(v.startswith('h') for v in out)
    assert hook.receipts()['a37_arm'] == 'CELL_BANK'


def test_digest_stable():
    d = C.a37_digest()
    assert d == C.a37_digest() and len(d) == 64


def test_boot_ci_is_the_regimeu0_bootstrap():
    import random
    rng = random.Random(1)
    diffs = [rng.gauss(-100, 500) for _ in range(24)]
    try:
        from tools.coopmwm import regimeu0
    except Exception:               # noqa: BLE001  (heavy plant imports on some hosts)
        pytest.skip('regimeu0 not importable here')
    assert C.boot_ci(diffs) == regimeu0.boot_ci(diffs)


def groups(all_pass, worst=3.0):
    return dict(all_pass=all_pass, groups={'origin:L2': dict(increase_mean_s=worst, registered=True),
                                           'class:hdv': dict(increase_mean_s=-1.0, registered=True)})


def test_unc_verdict_rule():
    seeds = list(range(991700, 991724))
    u0 = {s: 50_000.0+100*i for i, s in enumerate(seeds)}
    good = {s: u0[s]-2_000.0+(50 if i % 2 else -50) for i, s in enumerate(seeds)}
    v = C.unc_verdict(good, u0, groups(True))
    assert v['PASS'] and v['ci95'][1] < 0 and v['lb95_gain_veh_s'] == -v['ci95'][1] and v['worst_group'] == 'origin:L2'
    assert not C.unc_verdict(good, u0, groups(False, worst=12.0))['PASS']              # not ALLOWED
    noisy = {s: u0[s]+(3_000 if i % 2 else -3_100) for i, s in enumerate(seeds)}
    assert not C.unc_verdict(noisy, u0, groups(True))['PASS']                          # CI crosses 0
    with pytest.raises(ValueError):
        C.unc_verdict({s: good[s] for s in seeds[:12]}, {s: u0[s] for s in seeds[:12]}, groups(True))


def test_admitted_plans_order_and_failures_dropped():
    cands = [dict(family='p_duo', strategy=DUO), dict(family='p_solo', strategy=SOLO)]
    out = C.admitted_plans(cands, {'p_duo': dict(PASS=True, lb95_gain_veh_s=10.0),
                                   'p_solo': dict(PASS=True, lb95_gain_veh_s=40.0)})
    assert [p['family'] for p in out] == ['p_solo', 'p_duo']
    assert C.admitted_plans(cands, {'p_duo': dict(PASS=False, lb95_gain_veh_s=1.0)}) == []


def test_runner_hook_checks_bank_sha_digest_arm_and_cell(tmp_path):
    bank = tmp_path/'bank.json'
    bank.write_text(json.dumps(C.bank_doc(plans(DUO, SOLO), CELL)), encoding='utf-8')
    ck = dict(a37_arm='A37_PLAN', cell_key=CELL, a37_sha256=C.a37_digest(), a37_family='p_duo',
              a37_bank=dict(path=str(bank), file_sha256=hashlib.sha256(bank.read_bytes()).hexdigest()))
    world = dict(cell_key=CELL)
    hook, loaded = R.a37_hook(ck, world, 'A37_PLAN')
    assert loaded == C.a37_digest() and [p['family'] for p in hook.core.plans] == ['p_duo']
    for bad in (dict(a37_arm='CELL_BANK'), dict(a37_sha256='0'*64), dict(bank_cell_key='D2800_p20'),
                dict(a37_bank=dict(path=str(bank), file_sha256='0'*64)), dict(cell_key='D2400_p20')):
        with pytest.raises(ValueError):
            R.a37_hook(dict(ck, **bad), world, 'A37_PLAN')
    with pytest.raises(ValueError):
        R.a37_hook({k: v for k, v in ck.items() if k != 'a37_family'}, world, 'A37_PLAN')
    hook, _ = R.a37_hook(dict(ck, a37_arm='CELL_BANK'), world, 'CELL_BANK')
    assert {p['family'] for p in hook.core.plans} == {'p_duo', 'p_solo'}


def test_a37_verification_ranges_registered_and_others_unchanged():
    R.install_eval_ranges(ms)
    w = ms.world(991705, 3200, .2)
    assert w['repeat'] == 5 and w['cell_key'] == ms.cell_key(3200, .2) and ms.seed_scope(991723) == 'heldout'
    assert ms.world(991812, 2400, .2)['repeat'] == 12
    assert ms.world(998105, 3200, .2)['repeat'] == 5 and ms.world(993105, 2800, .2)['repeat'] == 105
    assert R._eval_base(991699) == 991600 and R._eval_base(991900) is None
