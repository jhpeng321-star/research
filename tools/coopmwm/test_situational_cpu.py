"""CLEAN5C4 situational WM guidance (owner REGISTRATION 2026-11-08 19:31): per (plan, onset bin) admission,
offer only in admitted bins (same feature rule offline and at runtime), B targets / R none / same templates."""
# ruff: noqa: F811 (imported pytest fixtures are test parameters)
import json
from pathlib import Path

import pytest

from . import clean5a_restart as c5, clean5b_restart as c5b, fullpol_bank as fb, fullpol_situational as fs
from .test_lc_families_cpu import wz  # noqa: F401
from .test_track2_backend_cpu import cfg, cpu_threads  # noqa: F401

DOC = Path(__file__).resolve().parents[2]/'clean5c'/'c4sit'/'GUIDANCE_C4SIT_p20.json'


def _doc():
    return json.loads(DOC.read_text(encoding='utf-8'))


def _r(vid, lane, s, v, cav=True):
    return dict(vid=vid, lane=lane, s_m=s, v_mps=v, is_cav=cav, length_m=5.)


def test_bins_offline_rule():
    assert fs.situation_bin(dict(drop_speed=14.9, v_lane_ahead500={'0': 30., '1': 24.9})) == dict(
        cong='congested', lane_diff='L2faster')
    assert fs.situation_bin(dict(drop_speed=15., v_lane_ahead500={'0': 30., '1': 25.})) == dict(
        cong='free', lane_diff='L2notfaster')
    assert fs.situation_bin(dict(drop_speed=None, v_lane_ahead500={'0': None, '1': 3.})) == dict(
        cong='free', lane_diff='L2notfaster')                               # missing -> the offline defaults


def test_runtime_situation_matches_reduce_act_features():
    own = _r('a', 0, -400., 20.)
    rows = [_r('b', 0, -100., 30.), _r('c', 1, -200., 10., False), _r('d', 1, -500., 99.),   # d behind: not ahead
            _r('e', 2, 50., 6.), _r('f', 1, 100., 1.)]                                       # f past the drop
    sit = fs.runtime_situation(own, rows)
    assert sit['drop_speed'] == pytest.approx((20.+30.+10.+99.+6.)/5)     # -600 <= s < S_DROP, own included
    assert sit['v_lane_ahead500'] == {'0': pytest.approx(25.), '1': 10., '2': 6.}
    assert fs.situation_bin(sit) == dict(cong='free', lane_diff='L2faster')


def test_offer_ok_situation_gate_and_default_unchanged():
    cond = dict(trigger='lanes', own_wm_lanes=[0], zone_s=[-1700., 0.], spread_m=200.,
                roles=[dict(wm_lane=0), dict(wm_lane=1)])
    own = _r('a', 0, -400., 20.)
    rows = [_r('b', 1, -350., 22.)]
    assert fb.offer_ok(cond, own, rows) is True
    free = dict(cond, situation=dict(rule=fs.RULE, bins=[dict(cong='free', lane_diff='L2notfaster')]))
    cong = dict(cond, situation=dict(rule=fs.RULE, bins=[dict(cong='congested', lane_diff='L2notfaster')]))
    assert fb.offer_ok(free, own, rows) is True and fb.offer_ok(cong, own, rows) is False
    slow = rows+[_r('h', 2, -100., 5., False), _r('i', 1, -50., 5., False)]
    assert fb.offer_ok(free, own, slow) is False                              # congested now: free bin not offered
    sit = fs.situation_bin(fs.runtime_situation(own, slow))
    assert sit['cong'] == 'congested'
    assert fb.offer_ok(dict(cond, situation=dict(rule=fs.RULE, bins=[sit])), own, slow) is True
    assert fb.offer_ok(free, own, [dict(rows[0], v_mps=None)]) is False     # unobserved speed: fail closed


def _rows(arm, effs, cell='D2800_p20', members=2):
    return [dict(cluster=('t', 'r', cell, k), cell=cell, pen=int(cell.split('_p')[1]), arm=arm, t=300, members=members,
                 eff=e, bin=dict(cong='free', lane_diff='L2notfaster')) for k, e in enumerate(effs)]


def test_admission_rule():
    ok = dict(wm=True, representable=True, allowed=True, cells=[])
    rows = (_rows('good', [-10., -12., -9., -11.]) + _rows('wide', [-10., 30., -9., 5.])
            + _rows('few', [-10., -12.]) + _rows('solo', [-10., -12., -9.], members=1)
            + _rows('struct', [-10., -12., -9., -11.]) + _rows('unfair', [-10., -12., -9., -11.]))
    status = dict(good=ok, wide=ok, few=ok, solo=ok, struct=dict(ok, wm=False), unfair=dict(ok, allowed=False))
    got = {p['family']: p for p in fs.admit(rows, status)}
    assert got['good']['admitted'] and got['good']['pool'] == 'all_penetrations' and got['good']['effect']['hi'] < 0
    assert got['wide']['reasons'] == ['ci_upper>=0'] and got['few']['reasons'] == ['clusters<3']
    assert got['solo']['reasons'] == ['not_multi_cav'] and got['struct']['reasons'] == ['not_wm_sourced']
    assert got['unfair']['reasons'] == ['not_fairness_allowed']
    assert fs.cboot(rows[:4]) == fs.cboot(rows[:4])                       # fresh RNG per estimate
    p20 = _rows('many', [-5.]*40, cell='D3200_p20')
    p20 = [dict(r, cluster=('t', 'r', 'D3200_p20', k % 10)) for k, r in enumerate(p20)]
    assert fs.admit(p20, dict(many=ok))[0]['pool'] == 'p20'                # >= 30 activations, >= 8 clusters


def test_built_doc_and_launch_path(cfg, wz, monkeypatch):
    from .lc_families import CONFIG_KEY
    doc = _doc()
    assert [(r['regime'], r['family'], r['arm']) for r in doc['rows']] == [('D2800_p20', 'mut_006c290e03ad3d3c', 'B')]
    adm = [p for p in doc['situational']['pairs'] if p['admitted']]
    assert [(p['family'], p['bin']) for p in adm] == [('mut_006c290e03ad3d3c', dict(cong='free', lane_diff='L2notfaster'))]
    assert adm[0]['effect']['hi'] < 0 and adm[0]['effect']['n'] == 15
    assert 'g2_gap1' in doc['situational']['structured_excluded']
    assert c5b.check_p20_guidance(doc, c5b.P20_CELLS, wm=True) == ('D2400_p20', 'D3200_p20')
    with pytest.raises(ValueError, match='LOWCAV'):
        c5b.check_p20_guidance(doc, c5b.P20_CELLS, wm=False)
    out = dict(cfg)
    out.update(c5b.keys_c(desc=wz, guidance_doc=doc, provenance='c4sit test'))
    tpl = out[CONFIG_KEY]['generator_args']['templates']
    assert tpl['B'] == tpl['R'] and len(tpl['B']) == 1
    assert tpl['B'][0]['offer_condition']['situation']['bins'] == [dict(cong='free', lane_diff='L2notfaster')]
    assert c5.check_guidance(out, ('D2400_p20', 'D3200_p20'), c5b.P20_CELLS)['regimes'] == ['D2800_p20']
    seen = {}
    monkeypatch.setattr(c5b.c5, 'prepare', lambda source, *, root, variant, guidance_doc, **kw: seen.update(v=variant))
    c5b.prepare_c_p20('src', root='/r/c4', guidance_doc=doc)
    assert seen['v']['guidance_missing_ok'] == ('D2400_p20', 'D3200_p20')
    assert seen['v']['newmap_extra']['p20_guidance_sources']['D2800_p20'] == 'wm_situational'
    bad = json.loads(json.dumps(doc))
    bad['clean5c']['templates'][0]['offer_condition'].pop('situation')
    with pytest.raises(ValueError, match='CLEAN5C4'):
        fs.check(bad, c5b.P20_CELLS)
