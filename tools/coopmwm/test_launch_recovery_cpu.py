import json
from pathlib import Path

import pytest

from . import composite_health as c, warm_health as h, safety_diagnosis as d
from .exact_grade import _seal
from .protocol import sha


def put(path,value):
    path.write_text(json.dumps(value),encoding='utf-8')


def setup(root):
    (root/'AUTHORITY.md').write_text('authorized-1230 authorized-original',encoding='utf-8')
    put(root/c.R_PAUSE,dict(rule='SAFETY_COMPOSITE_R_u003',evidence=dict(arm='R')))
    put(root/'SAFETY_RULE_0957.json',dict(authority=c.AUTHORITY,TTC_threshold=1.5,
        bootstrap=10000,bootstrap_seed=20261101,alpha=.05,amendment_sha256='authorized-original'))
    put(root/c.R_RULE,_seal(dict(authority=c.R_AUTHORITY,arm='R',composite_action='report_only',
        collision_teleport_action='pause_on_any',original_pause_sha256=sha(root/c.R_PAUSE),
        amendment_sha256='authorized-1230'),'receipt_sha256'))


def records(monkeypatch,*,collisions=0,teleports=0):
    episodes=[dict(id='R1',world=dict(seed=986000)),dict(id='R2',world=dict(seed=986001))]
    references=[dict(id='A1',world=dict(seed=986000)),dict(id='A2',world=dict(seed=986001))]
    def episode(root,ident,*args):
        is_r=ident.startswith('R')
        return dict(world=dict(seed=986000 if ident.endswith('1') else 986001),
            collisions=collisions if is_r else 0,teleports=teleports if is_r else 0,
            hard_brake=1,ttc_below_1p5_events=50 if is_r else 0,critical_total=51 if is_r else 1)
    monkeypatch.setattr(d,'episode',episode)
    return episodes,references


def test_r_reports_real_failed_composite_and_admits_without_rewriting_result(tmp_path,monkeypatch):
    setup(tmp_path)
    episodes,references=records(monkeypatch)
    assert c.braking(tmp_path,'R',3,episodes,references,h.pause)
    out=json.loads((tmp_path/'R_SAFETY_COMPOSITE_003.json').read_text())
    assert out['release'] is False  # original statistical finding remains visible
    assert out['metrics']['critical_total']['ci95']==[50,50]
    assert out['optimization_admitted'] is True and out['composite_action']=='report_only'
    assert c.released(tmp_path,tmp_path/c.R_PAUSE)
    assert not c.released(tmp_path,tmp_path/'PAUSE_SAFETY_COMPOSITE_B_u003.json')
    assert not c.released(tmp_path,tmp_path/'PAUSE_SAFETY_COMPOSITE_R_u004.json')


@pytest.mark.parametrize('arm,amended',[('R',False),('B',True),('B',False)])
def test_original_rule_default_and_b_unchanged(tmp_path,monkeypatch,arm,amended):
    setup(tmp_path)
    if not amended:(tmp_path/c.R_RULE).unlink()
    episodes,references=records(monkeypatch)
    with pytest.raises(h.HealthPause):c.braking(tmp_path,arm,4,episodes,references,h.pause)


@pytest.mark.parametrize('metric',['collisions','teleports'])
def test_any_r_collision_or_teleport_still_pauses_even_with_equal_a_baseline(tmp_path,monkeypatch,metric):
    setup(tmp_path)
    episodes,references=records(monkeypatch,**{metric:1})
    with pytest.raises(h.HealthPause):c.braking(tmp_path,'R',4,episodes,references,h.pause)
    # Independent live guard: a nonzero A benchmark cannot mask an R event.
    monkeypatch.setattr(h,'assert_running',lambda *args:None)
    totals=dict(collisions=0,teleports=0);totals[metric]=1
    with pytest.raises(h.HealthPause):
        h.live_safety(tmp_path,'R',totals,986000,
            dict(by_world={'986000':totals},unmatched_zero_event_benchmark=False,source='test'))


def test_modified_or_unregistered_amendment_refuses(tmp_path):
    setup(tmp_path)
    (tmp_path/'AUTHORITY.md').write_text('no amendment')
    with pytest.raises(ValueError,match='unregistered'):c.r_report_only(tmp_path,'R')
    assert c.r_report_only(tmp_path,'B') is False


def test_native_wave_adopts_closed_episodes_before_any_reservation(tmp_path,monkeypatch):
    from . import accel_runner as runner
    monkeypatch.setattr(runner,'ledger',lambda root:[dict(id='clean1_warm_u003_R_986000')])
    path=tmp_path/'jobs/clean1_warm_u003_R_986000';path.mkdir(parents=True)
    saved=dict(id=path.name,world=dict(seed=986000),arm='R',steps=[dict(complete=True)])
    put(path/'RESULT.json',saved)
    class Policy:
        def state_dict(self):raise AssertionError('closed result must be adopted first')
    monkeypatch.setattr(runner,'reserve',lambda *args:pytest.fail('completed physics recharged'))
    assert runner.wave(tmp_path,{},dict(R=Policy()),[dict(seed=986000)],
        'train','clean1_warm_u003','v2v_train',False)==[saved]
