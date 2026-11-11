"""CPU-only tests for the amendment-63 driver (no simulator): names / budget / cap, arms, verdict rules, the STRICT_BANK
build rule (synthetic sources) and the smoke receipt rule."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a42_bigp60 as D42  # noqa: E402
import a63_strictbank as A63  # noqa: E402
import nm_finaleval as N  # noqa: E402

from tools.coopmwm import a37_cellbank as C  # noqa: E402
from tools.coopmwm.test_a37_cellbank_cpu import role, spec  # noqa: E402

D = A63.D


def test_names_budget_arms():
    assert (D.A42_PHASE, D.A42_CAP_SIM_S, D.STOP_A42, D.JOB_PREFIX, D.A42Runtime.phase, D.BANK_DIR) == \
        ('strictbank_a63', 480_000, 'STOP_A63', 'a63', 'strictbank_a63', 'a63_banks')
    assert (D42.A42_PHASE, D42.JOB_PREFIX) == ('bigp60_a42', 'a42')                 # the A42 module untouched
    b = A63.budget_plan()
    assert b['planned'] == 1_200+4*96*1_200 == 462_000 and b['margin'] == 18_000
    assert A63.ARMS == dict(BIG_PART_L01=(A63.BIG_BANK, [0, 1]), STRICT_BANK=(A63.STRICT_BANK, None),
                            STRICT_PART_L2=(A63.STRICT_BANK, [2]), STRICT_PART_L01=(A63.STRICT_BANK, [0, 1]))
    assert A63.BOOTSTRAP['seed'] == 20261104 and A63.BOOTSTRAP['reps'] == 10000 and A63.LIMIT_S == 5.0
    assert sorted(A63.STRICT_FAMILIES) == sorted(['rand_2c80042490ec698f', 'rand_bb1cb39d47f0a524',
                                                  'rand_cfa9b70a3782d48e', 'mut_2922c89d0b31d4c8'])


def test_verdict_rules():
    assert A63.p1_verdict([-5., -1.]) == 'L0+L1 PAIR SAVES'
    assert A63.p1_verdict([-5., 0.]) == 'NOT SHOWN' and A63.p1_verdict([1., 2.]) == 'NOT SHOWN'
    assert A63.p2_verdict([-5., -1.]) == 'OTHER LANES HELP'
    assert A63.p2_verdict([1., 5.]) == 'HURT'
    assert A63.p2_verdict([-1., 1.]) == 'NOT RESOLVED' and A63.p2_verdict([0., 1.]) == 'NOT RESOLVED'
    ok, no = dict(meets_limit=True, saves=True), dict(meets_limit=False, saves=True)
    assert A63.p3_verdict(dict(STRICT_BANK=ok, STRICT_PART_L2=no, STRICT_PART_L01=dict(meets_limit=True, saves=False))) \
        == 'JOINT-ONLY FEASIBLE'
    assert A63.p3_verdict(dict(STRICT_BANK=ok, STRICT_PART_L2=ok, STRICT_PART_L01=no)) == 'NOT SHOWN'
    assert A63.p3_verdict(dict(STRICT_BANK=no, STRICT_PART_L2=no, STRICT_PART_L01=no)) == 'NOT SHOWN'


def test_worst_group():
    g = {k: dict(increase_mean_s=v, ci95=[v-1., v+1.5]) for k, v in zip(N.CO_PRIMARY_GROUPS, (1., 4.9, -2., 0., 3.))}
    g['origin:undefined'] = dict(increase_mean_s=99., ci95=[0, 1])                  # reported only, never the worst
    assert A63.worst_group(dict(groups=g)) == ('origin:L1', 4.9, [4.9-1., 4.9+1.5])


def _fams(monkeypatch, names):
    monkeypatch.setattr(A63, 'STRICT_FAMILIES', tuple(names))


def test_strict_doc_copies_templates_and_orders_by_lb95(monkeypatch):
    a = spec('p_a', [role(0, 20.0), role(2, 21.0)])
    b = spec('p_b', [role(1, 19.0), role(2, 22.0)])
    x = spec('p_x', [role(1, 18.0), role(0, 22.0)])
    src1 = C.bank_doc([dict(family='p_a', strategy=a, lb95_veh_s=10.), dict(family='p_x', strategy=x, lb95_veh_s=3.)],
                      'D2800_p60')
    src2 = C.bank_doc([dict(family='p_b', strategy=b, lb95_veh_s=0.), dict(family='p_a', strategy=a, lb95_veh_s=0.)],
                      'D2800_p60')
    _fams(monkeypatch, ['p_a', 'p_b'])
    doc, origin = A63.strict_doc(dict(S1=src1, S2=src2), dict(p_a=10., p_b=50.),
                                 dict(plans=[dict(family='p_b', strategy=b)]))
    assert [r['family'] for r in doc['rows']] == ['p_b', 'p_a'] and [r['lb95_veh_s'] for r in doc['rows']] == [50., 10.]
    want = C.bank_doc([dict(family='p_b', strategy=b, lb95_veh_s=50.), dict(family='p_a', strategy=a, lb95_veh_s=10.)],
                      'D2800_p60')
    assert doc == want and origin == dict(p_a=['S1', 'S2'], p_b=['S2'])
    core = C.make_core('CELL_BANK', 'D2800_p60', doc)
    assert [p['family'] for p in core.plans] == ['p_b', 'p_a']
    bad = C.bank_doc([dict(family='p_a', strategy=spec('p_a', [role(0, 21.0), role(2, 21.0)]), lb95_veh_s=0.)],
                     'D2800_p60')
    with pytest.raises(N.Blocked):                       # the same family with a different offer condition
        A63.strict_doc(dict(S1=src1, S2=src2, S3=bad), dict(p_a=10., p_b=50.), dict(plans=[]))
    with pytest.raises(N.Blocked):                       # an A56 plan whose stored template is not its strategy
        A63.strict_doc(dict(S1=src1, S2=src2), dict(p_a=10., p_b=50.), dict(plans=[dict(family='p_b', strategy=dict(x, name='p_b'))]))


def test_verified_lb95(monkeypatch):
    _fams(monkeypatch, ['f1', 'f2', 'f3'])
    adm42 = dict(priority=[['f1', 5.], ['f2', 2.]])
    adm45 = dict(priority=[['f2', 2.], ['g', 9.]])
    red = dict(verdicts=dict(f3=dict(PASS=True, lb95_gain_veh_s=1.5), h=dict(PASS=False, lb95_gain_veh_s=0.)))
    assert A63.verified_lb95(adm42, adm45, red) == dict(f1=5., f2=2., f3=1.5)
    with pytest.raises(N.Blocked):
        A63.verified_lb95(dict(priority=[['f1', 5.], ['f2', 3.]]), adm45, red)       # inconsistent
    with pytest.raises(N.Blocked):
        A63.verified_lb95(adm42, adm45, dict(verdicts=dict(f3=dict(PASS=False, lb95_gain_veh_s=1.))))


def test_part_receipt_rule():
    acts = [dict(family='p', members=['a', 'b', 'c'], on_lane=['a', 'b'], t=300)]
    good = dict(member_lanes=[0, 1], a57_sha256='d', acts=acts, commanded_lanes={'a': 0, 'b': 1},
                vehicles_with_accel_command=2)
    assert A63.part_receipt_ok(good, [1, 0], 'd', {'p'})
    assert not A63.part_receipt_ok(dict(good, commanded_lanes={'a': 0, 'c': 2}), [0, 1], 'd', {'p'})
    assert not A63.part_receipt_ok(good, [2], 'd', {'p'})
    assert not A63.part_receipt_ok(good, [0, 1], 'x', {'p'})
    assert not A63.part_receipt_ok(good, [0, 1], 'd', {'q'})
    assert not A63.part_receipt_ok(dict(good, vehicles_with_accel_command=3), [0, 1], 'd', {'p'})
