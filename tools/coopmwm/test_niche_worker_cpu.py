import hashlib
import json
import pytest
from .niche_worker import adopted_checks


def test_adoption_requires_exact_same_numerical_bytes_and_exact_receipt(tmp_path):
    (tmp_path/'numeric.py').write_bytes(b'frozen')
    (tmp_path/'completed.json').write_text(json.dumps(dict(pass_=True,max_cost_error=0.)))
    h=hashlib.sha256(b'frozen').hexdigest()
    binding=dict(hashes={'numeric.py':h},adopted_joint_checks=dict(
        numerical_hashes={'numeric.py':h},receipts={'gate.json':dict(file='completed.json',
            sha256=hashlib.sha256((tmp_path/'completed.json').read_bytes()).hexdigest())}))
    assert adopted_checks(tmp_path,binding,('numeric.py',),('gate.json',))['gate.json']['adopted']
    (tmp_path/'numeric.py').write_bytes(b'changed')
    with pytest.raises(AssertionError):adopted_checks(tmp_path,binding,('numeric.py',),('gate.json',))
    (tmp_path/'numeric.py').write_bytes(b'frozen')
    (tmp_path/'completed.json').write_text(json.dumps(dict(pass_=True,max_cost_error=1.)))
    with pytest.raises(AssertionError):adopted_checks(tmp_path,binding,('numeric.py',),('gate.json',))
    binding['adopted_joint_checks']['receipts']['gate.json']['sha256']=hashlib.sha256((tmp_path/'completed.json').read_bytes()).hexdigest()
    with pytest.raises(AssertionError):adopted_checks(tmp_path,binding,('numeric.py',),('gate.json',))


def test_adoption_rejects_changed_receipt_and_missing_gate_set(tmp_path):
    h=hashlib.sha256(b'x').hexdigest();(tmp_path/'n').write_bytes(b'x')
    binding=dict(hashes={'n':h},adopted_joint_checks=dict(numerical_hashes={'n':h},receipts={}))
    with pytest.raises(AssertionError):adopted_checks(tmp_path,binding,('n',),('gate',))
    assert adopted_checks(tmp_path,dict(hashes={}),(),())=={}
