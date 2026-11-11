"""CPU-only tests for the amendment-64 driver (no simulator): names / budget / arms, the plan-variant rule, the
implied-speed smoke rule, the part receipt rule and the Holm P3 rule."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a42_bigp60 as D42  # noqa: E402
import a64_hand3fast as A64  # noqa: E402
import nm_finaleval as N  # noqa: E402

D = A64.D
PLANS = A64.PLANS_DIR


def test_names_budget_arms():
    assert (D.A42_PHASE, D.A42_CAP_SIM_S, D.STOP_A42, D.JOB_PREFIX, D.A42Runtime.phase, D.BANK_DIR) == \
        ('hand3fast_a64', 480_000, 'STOP_A64', 'a64', 'hand3fast_a64', 'a64_banks')
    assert (D42.A42_PHASE, D42.JOB_PREFIX) == ('bigp60_a42', 'a42')
    b = A64.budget_plan()
    assert b['planned'] == 1_200+4*96*1_200 == 462_000 and b['margin'] == 18_000
    assert A64.ARMS == dict(HAND3_F23=('F23', None), HAND3_F26=('F26', None), F23_PART_L2=('F23', [2]),
                            F26_PART_L2=('F26', [2]))
    assert A64.BOOTSTRAP == dict(reps=10000, level=.95, seed=20261104, method='paired_percentile')
    assert {v: s for v, (_, s) in A64.VARIANTS.items()} == dict(F23=23.0, F26=26.0)


def test_pinned_plans_are_the_variant_rule():
    src = json.loads((PLANS/'A64_HAND3_F23_D2800_p60.json').read_text(encoding='utf-8'))
    for v, (name, vw) in A64.VARIANTS.items():
        p = PLANS/name
        assert N.file_sha(p) == A64.PLAN_SHA256[v]
        doc = json.loads(p.read_text(encoding='utf-8'))
        assert A64.role_speeds(doc) == {0: 20.0, 1: 20.0, 2: vw}
        back = A64.variant_doc(doc, 20.0)        # undo the one change -> the A59 HAND3 key 978a6170
        assert back['rows'][0]['strategy_key'] == '978a6170333d159a'
        diff = A64.json_diff(back, doc)
        assert [d[0] for d in diff] == ['/clean5c/templates/0/offer_condition/roles/1/v_w',
                                        '/clean5c/templates/0/template/source/strategy_key', '/rows/0/strategy_key']
        assert D.C37.bank_text(A64.variant_doc(back, vw)) == p.read_text(encoding='utf-8')
    assert src['rows'][0]['family'] == 'HAND3'


def test_part_receipt_rule():
    dg = 'x'*64
    part = dict(member_lanes=[2], a57_sha256=dg, vehicles_with_accel_command=1, commanded_lanes={'a': 2},
                acts=[dict(family='HAND3', t=400, members=['b', 'a', 'c'], on_lane=['a'])])
    assert A64.part_receipt_ok(part, [2], dg)
    assert not A64.part_receipt_ok(dict(part, commanded_lanes={'b': 1}), [2], dg)
    assert not A64.part_receipt_ok(dict(part, acts=[dict(part['acts'][0], family='other')]), [2], dg)
    assert not A64.part_receipt_ok(part, [2], 'y'*64)


ROLES = [dict(lane=1, v_w=20.0, dmax=1.0), dict(lane=2, v_w=23.0, dmax=1.0), dict(lane=0, v_w=20.0, dmax=1.0)]


def _act(vid, r):
    return dict(vid=vid, requested_raw=r, oracle_plan='HAND3')


def test_implied_speeds_and_check():
    acts = [dict(t=400, members=['b', 'a', 'c'])]
    actions = {400: [_act('b', 1.0), _act('a', 0.5), _act('c', -0.25)], 401: [_act('a', 0.25)]}
    speeds = {400: {'b': 10.0, 'a': 22.5, 'c': 20.25}, 401: {'a': 22.75}}
    chk = A64.implied_speeds(acts, ROLES, actions, speeds)
    assert chk['n_mismatch'] == 0 and chk['unmapped'] == 0
    assert chk['per_lane']['2'] == dict(n_interior=2, n_saturated=0, implied_min=23.0, implied_max=23.0,
                                        registered_v_w=23.0)
    assert chk['per_lane']['1']['n_saturated'] == 1 and chk['commanded_lanes'] == {'0': 1, '1': 1, '2': 2}
    assert A64.speed_check_ok('HAND3_F23', chk)
    assert not A64.speed_check_ok('HAND3_F26', chk)            # wrong registered speed
    assert not A64.speed_check_ok('F23_PART_L2', chk)          # lanes 0 / 1 commanded in a part arm
    part = A64.implied_speeds(acts, ROLES, {400: [_act('a', 0.5)]}, speeds)
    assert A64.speed_check_ok('F23_PART_L2', part)
    bad = A64.implied_speeds(acts, ROLES, {400: [_act('a', 0.4)]}, speeds)     # implies 22.9, not 23
    assert bad['n_mismatch'] == 1 and not A64.speed_check_ok('F23_PART_L2', bad)


def test_holm_p3():
    rng = np.random.default_rng(0)
    big = -500+50*rng.standard_normal(96)
    null = 30*rng.standard_normal(96)
    out = A64.holm_p3(dict(F23=big, F26=null), A64.BOOTSTRAP)
    assert out['F23']['holm_step'] == 1 and out['F23']['holm_alpha'] == .025 and out['F23']['verdict'] == 'OTHER LANES HELP'
    assert out['F26']['holm_step'] == 2 and out['F26']['verdict'] == 'NOT RESOLVED'
    out = A64.holm_p3(dict(F23=null, F26=-big), A64.BOOTSTRAP)
    assert out['F26']['verdict'] == 'HURT' and out['F23']['verdict'] == 'NOT RESOLVED'
    assert A64.p1_verdict([-2., -1.]) == 'SAVES' and A64.p1_verdict([-2., 0.]) == 'NOT SHOWN'


def test_group_rows():
    g = {k: dict(increase_mean_s=v, ci95=[v-1., v+1.]) for k, v in zip(N.CO_PRIMARY_GROUPS, (1., 4.5, 6., 0., 9.5))}
    r = A64.group_rows(dict(groups=g))
    assert r['origin:L2']['within_10s_point'] and not r['origin:L2']['within_5s_point']
    assert r['origin:L1']['within_5s_point'] and not r['origin:L1']['within_5s_ci_upper']
    assert not r['class:cav']['within_10s_ci_upper']
