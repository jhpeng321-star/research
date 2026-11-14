import pytest


def test_empty_actionable_list_never_loads_snapshot_or_runs_sumo(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    from . import ls_collection as ls, ls_feasible as lf
    from .protocol import digest
    from .exact_grade import verify_sealed
    key='a'*64
    record=dict(state_id=key,rows=[],commands={})
    envelope=dict(record=record,world_id=986000,time_s=300)
    response=dict(collection_top8=[dict(id=str(i)) for i in range(8)])
    gate=dict(original_k=8,available_k=0,inactionable=[str(i) for i in range(8)],members_t0=None)
    monkeypatch.setattr(lf,'shortlist',lambda *a:([],gate))
    monkeypatch.setattr(ls,'complete_snapshot',lambda *a:(_ for _ in ()).throw(AssertionError('no snapshot')))
    result=ls.label(tmp_path,envelope,response,dict(nullfix_active=True,accel_bounds=[-3,2.6]))
    verify_sealed(result,'receipt_sha256')
    assert result['status']=='inactionable' and result['actual_sim_s']==0
    assert result['grade_veh_s'] is None and result['original_reservation_retained']
    assert (tmp_path/'labels'/(key+'.json')).exists()


def test_skip_never_fabricates_an_exact_grade():
    pytest.importorskip('torch')
    from .ls_feasible import skipped
    r=skipped(dict(state_id='b'*64),dict(time_s=300),dict(original_k=8))
    assert r['status'] not in ('accepted','rejected')
    assert r['J_policy_veh_s'] is r['J_candidates_veh_s'] is r['grade_veh_s'] is None


def test_real_gate_keeps_braking_and_records_observable_policy_raw():
    pytest.importorskip('torch')
    from .ls_feasible import shortlist
    from .test_nullfix_cpu import gate_plan, sat_rows
    record=dict(rows=sat_rows(),commands=dict(c1=1.5,c2=1.5),
        policy_residuals=dict(c1=.1,c2=.2))
    up=gate_plan('up',('c1','c2'),1.)
    brake=gate_plan('brake',('c1','c2'),-4.)
    candidates,metadata=shortlist(record,[up,brake],[-3.,2.6])
    assert up['id'] in metadata['inactionable']
    assert candidates == [brake]
    assert metadata['members_t0']['c1']['policy_raw_mps2'] == 1.6
