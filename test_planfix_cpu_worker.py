"""Frozen PolicyExecutor ABI and off-CAV sentinel contract; no model/physics."""
import numpy as np
import pytest
import hashlib,json
from planfix_cpu_worker import native_requests,training_finished

ROWS=[dict(vid='c0',is_cav=True),dict(vid='h0',is_cav=False),dict(vid='c1',is_cav=True)]

def test_query_uses_export_paired_api_and_preserves_hdv_nan_sentinel():
    class Export:
        def __call__(self,state):raise AssertionError('bare export call returns acceleration only')
        def act_with_lane(self,state):return np.array([[1.2,np.nan,-.3]]),None
    assert native_requests(Export(),{},ROWS)=={'c0':1.2,'c1':-.3}

def test_nonfinite_acting_export_is_refused_before_a_wm_search():
    class Export:
        def act_with_lane(self,state):return np.array([[np.nan,np.nan,-.3]]),None
    with pytest.raises(AssertionError):native_requests(Export(),{},ROWS)

def complete_receipts(root):
    for role in ('A','B','R'):
        checkpoint=root/(role+'_checkpoint_024.pt');checkpoint.write_bytes(b'fixture checkpoint')
        row=dict(arm=role,update=24,config_sha256='registered',checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        row['update_sha256']=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        (root/(role+'_UPDATE_024.json')).write_text(json.dumps(row))

def test_idle_guard_requires_all_three_sealed_final_checkpoints(tmp_path):
    complete_receipts(tmp_path)
    assert training_finished(tmp_path,dict(runtime_sha256='registered'))
    assert not training_finished(tmp_path,dict(runtime_sha256='different_runtime'))
    (tmp_path/'R_checkpoint_024.pt').write_bytes(b'changed checkpoint')
    assert not training_finished(tmp_path,dict(runtime_sha256='registered'))

def test_idle_guard_refuses_edited_final_receipt(tmp_path):
    complete_receipts(tmp_path)
    path=tmp_path/'B_UPDATE_024.json';row=json.loads(path.read_text());row['update']=23;path.write_text(json.dumps(row))
    assert not training_finished(tmp_path,dict(runtime_sha256='registered'))
