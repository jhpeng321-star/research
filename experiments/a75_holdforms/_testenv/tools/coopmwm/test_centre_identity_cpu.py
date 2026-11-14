"""Centre dispatch removes mode before hashing; accept only that exact envelope change."""
import copy
from .replay_bank import centre_identity
from .protocol import digest


def test_exact_mode_stripped_response_identity_is_valid():
    r=dict(state_id='state',mode='centre',world_id=986000,group=['a','b'],history=[])
    sha=digest(r);prepared={k:v for k,v in r.items() if k!='mode'}
    assert centre_identity(r,sha,dict(record_sha256=digest(prepared),candidate_scores=0))
    assert centre_identity(r,sha,dict(record_sha256=sha,candidate_scores=0))


def test_no_other_request_change_or_candidate_scoring_is_accepted():
    r=dict(state_id='state',mode='centre',world_id=986000,group=['a','b'])
    sha=digest(r);prepared={k:v for k,v in r.items() if k!='mode'}
    v=dict(record_sha256=digest(prepared),candidate_scores=0)
    changed=copy.deepcopy(r);changed['world_id']=986001
    assert not centre_identity(changed,sha,v)
    assert not centre_identity(r,sha,dict(v,candidate_scores=8))
    assert not centre_identity(r,sha,dict(v,record_sha256='wrong'))
    assert not centre_identity(dict(prepared),sha,v)
