import pytest
from .exact_grade import _seal
from .solo_seed2_links import views,merge


def test_batched_member_views_preserve_correct_candidate_and_sealed_provenance():
    source=_seal(dict(verification_id='batch',snapshot_sha256='snapshot',window=dict(start_s=300,end_s=570),
        J_policy_veh_s=100.,J_candidates_veh_s=[90.,120.],branches=[dict(branch='policy'),dict(branch='cand1'),dict(branch='cand2')]),'receipt_sha256')
    s=_seal(dict(group=['a','b'],verification=source),'supplement_sha256')
    a,b=views(s)
    assert a['grade_veh_s']==10 and b['grade_veh_s']==-20
    assert a['J_candidates_veh_s']==[90.] and b['J_candidates_veh_s']==[120.]
    assert b['branches'][1]['branch']=='cand2' and b['source_receipt_sha256']==source['receipt_sha256']
    assert a['verification_id']!=b['verification_id'] and a['new_sim_s']==0
    s['group']=['wrong']
    with pytest.raises(ValueError):views(s)


def test_absent_supplement_preserves_original_object_and_keys(tmp_path):
    c=_seal(dict(verification_id='v',cooperation=dict(complete=True)),'confirmation_sha256')
    original={'v':c};out=merge(tmp_path,original)
    assert out==original and 'replication_solo' not in out['v']['cooperation']
