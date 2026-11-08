"""CLEAN5C4 situational WM guidance (owner REGISTRATION 2026-11-08 19:31 + MAIN amendment): per (plan, onset bin)
admission; bins from the acting CAV's own sensing (one function offline and in offer_ok; unobserved = own bin);
B targets / R none / same templates; no admission -> report, no doc."""
import json
from pathlib import Path

import pytest

from . import clean5b_restart as c5b, fullpol_bank as fb, fullpol_situational as fs

C4SIT = Path(__file__).resolve().parents[2]/'clean5c'/'c4sit'


def _r(vid, lane, s, v, cav=True):
    return dict(vid=vid, lane=lane, s_m=s, v_mps=v, is_cav=cav, length_m=5.)


def test_sensor_range_is_the_runtime_one():
    from .merge_features import SENSOR_RANGE_M
    assert fs.SENSOR_RANGE_M == SENSOR_RANGE_M


def test_bins_rule():
    assert fs.situation_bin(dict(drop_speed=14.9, v_lane_ahead500={'0': 30., '1': 24.9})) == dict(
        cong='congested', lane_diff='L2faster')
    assert fs.situation_bin(dict(drop_speed=15., v_lane_ahead500={'0': 30., '1': 25.})) == dict(
        cong='free', lane_diff='L2notfaster')


def test_onset_bin_own_sensing_and_unobserved():
    own = _r('a', 0, -560., 20.)
    near = [_r('b', 0, -520., 30.), _r('c', 1, -500., 10., False)]
    far = [_r('x', 1, -300., 99.), _r('y', 2, 50., 1.)]                     # beyond 100 m: never seen
    assert fs.onset_bin(own, near+far) == dict(cong='free', lane_diff='L2faster')    # drop 20 m/s, 25 - 10
    slow = [dict(r, v_mps=5.) for r in near]
    assert fs.onset_bin(dict(own, v_mps=5.), slow) == dict(cong='congested', lane_diff='L2notfaster')
    assert fs.onset_bin(_r('a', 0, -900., 20.), near) == fs.UNOBSERVED     # drop area out of view
    assert fs.onset_bin(own, [near[0]]) == fs.UNOBSERVED                   # no lane-1 vehicle ahead
    assert fs.onset_bin(None, near) == fs.UNOBSERVED
    assert fs.onset_bin(own, [dict(near[0], v_mps=None), near[1]]) == fs.UNOBSERVED


def test_offer_ok_uses_onset_bin_and_default_unchanged():
    cond = dict(trigger='lanes', own_wm_lanes=[0], zone_s=[-1700., 0.], spread_m=200.,
                roles=[dict(wm_lane=0), dict(wm_lane=1)])
    own = _r('a', 0, -560., 20.)
    rows = [_r('b', 1, -500., 10.), _r('c', 0, -520., 30., False)]
    assert fb.offer_ok(cond, own, rows) is True
    got = fs.onset_bin(own, rows)
    assert got == dict(cong='free', lane_diff='L2faster')
    sit = lambda *bins: dict(cond, situation=dict(rule=fs.RULE, bins=list(bins)))  # noqa: E731
    assert fb.offer_ok(sit(got), own, rows) is True
    assert fb.offer_ok(sit(dict(cong='congested', lane_diff='L2faster')), own, rows) is False
    assert fb.offer_ok(sit(fs.UNOBSERVED), own, [rows[0]]) is False           # unobserved: never offered


def _rows(arm, effs, cell='D2800_p20', members=2, b=None):
    return [dict(cluster=('t', 'r', cell, k), cell=cell, pen=int(cell.split('_p')[1]), arm=arm, t=300, members=members,
                 eff=e, bin=b or dict(cong='free', lane_diff='L2notfaster')) for k, e in enumerate(effs)]


def test_admission_rule():
    ok = dict(wm=True, representable=True, allowed=True, cells=[])
    rows = (_rows('good', [-10., -12., -9., -11.]) + _rows('wide', [-10., 30., -9., 5.])
            + _rows('few', [-10., -12.]) + _rows('solo', [-10., -12., -9.], members=1)
            + _rows('struct', [-10., -12., -9., -11.]) + _rows('unfair', [-10., -12., -9., -11.])
            + _rows('blind', [-10., -12., -9., -11.], b=dict(fs.UNOBSERVED)))
    status = dict(good=ok, wide=ok, few=ok, solo=ok, struct=dict(ok, wm=False), unfair=dict(ok, allowed=False), blind=ok)
    got = {p['family']: p for p in fs.admit(rows, status)}
    assert got['good']['admitted'] and got['good']['pool'] == 'all_penetrations' and got['good']['effect']['hi'] < 0
    assert got['wide']['reasons'] == ['ci_upper>=0'] and got['few']['reasons'] == ['clusters<3']
    assert got['solo']['reasons'] == ['not_multi_cav'] and got['struct']['reasons'] == ['not_wm_sourced']
    assert got['unfair']['reasons'] == ['not_fairness_allowed'] and 'blind' not in got     # unobserved never admitted
    assert fs.cboot(rows[:4]) == fs.cboot(rows[:4])
    p20 = [dict(r, cluster=('t', 'r', 'D3200_p20', k % 10)) for k, r in enumerate(_rows('many', [-5.]*40, cell='D3200_p20'))]
    assert fs.admit(p20, dict(many=ok))[0]['pool'] == 'p20'


def test_activations_need_an_onset_bin_for_every_activation():
    u0 = dict(bins=[[0.]*8]*90, acts=[])
    arm = dict(bins=[[1.]*8]*90, acts=[dict(t=400, members=['a', 'b'])])
    eps = {('mc', 'wmv_mc', 'D2800_p20', 1, 'u0'): u0, ('mc', 'wmv_mc', 'D2800_p20', 1, 'm'): arm}
    b = dict(cong='free', lane_diff='L2faster')
    rows = fs.activations(eps, {('wmv_mc', 'D2800_p20', 1, 'm', 400): b})
    assert rows[0]['bin'] == b and rows[0]['eff'] == pytest.approx(8.*30)    # 30 bins x 8 groups, window 300 s
    with pytest.raises(ValueError, match='no own-sensing onset bin'):
        fs.activations(eps, {})


def test_plan_status_scope(tmp_path):
    d = tmp_path/'A'
    d.mkdir()
    (d/'PLAN.json').write_text(json.dumps(dict(families=dict(m=dict(cells={'D3200_p20': 'wm'})))))
    (d/'VERIFY.json').write_text(json.dumps(dict(families=dict(m=dict(n2_parity=dict(verdict='identical'))))))
    (d/'ANALYSIS.json').write_text(json.dumps(dict(cells={'D3200_p20': dict(m=dict(n=2)), 'D3200_p40': dict(m=dict(n=2))})))
    (d/'ANALYSIS_ALLOWED.json').write_text(json.dumps(dict(cells={'D3200_p20': dict(m={})})))
    assert fs.plan_status([d], allowed_scope='p20')['m']['allowed'] is True
    assert fs.plan_status([d], allowed_scope='all')['m']['allowed'] is False
    assert fs.plan_status([d], structured=['m'])['m']['wm'] is False


def test_recorded_result_no_admission_and_check():
    """The recorded own-sensing run (v2: wmv_mc + wmv_p20 + wmv_rep, ALLOWED scope p20) admitted nothing."""
    rep = json.loads((C4SIT/'GUIDANCE_C4SIT_p20.json.NO_ADMISSION.json').read_text(encoding='utf-8'))
    assert rep['rule'] == 'fullpol-clean5c4-situational-v2' and not any(p['admitted'] for p in rep['pairs'])   # own-sensing record
    best = [p for p in rep['pairs'] if p['family'] == 'mut_0584e488a8bf93c5' and p['effect']['hi'] < 0]
    assert best and all('not_fairness_allowed' in p['reasons'] for p in best)
    assert not (C4SIT/'GUIDANCE_C4SIT_p20.json').exists()
    doc = dict(situational=dict(rule=fs.RULE, pairs=[dict(family='m', admitted=True)]), rows=[dict(
        arm='B', regime='D2800_p20', family='m')], clean5c=dict(templates=[dict(cell='D2800_p20', offer_condition=dict(
            situation=dict(rule=fs.RULE, bins=[dict(cong='free', lane_diff='L2faster')])))]))
    assert c5b.check_p20_guidance(doc, c5b.P20_CELLS, wm=True) == ('D2400_p20', 'D3200_p20')
    doc['clean5c']['templates'][0]['offer_condition']['situation']['rule'] = 'v1'
    with pytest.raises(ValueError, match='CLEAN5C4'):
        fs.check(doc, c5b.P20_CELLS)
