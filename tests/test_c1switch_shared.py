"""Shared GPU cap and completed-physics retention, without SUMO or a GPU."""
import json

import pytest

from tools.c1switch import shared
from tools.c1switch.design import CONTROLLERS, sha
from tools.c1switch.storage import canonical, write


def test_completed_physics_without_branches_is_never_replayed(tmp_path):
    assert not shared.physics_done(tmp_path)
    write(tmp_path/'PHYSICS.json',dict(schedule=['pi']*10))
    write(tmp_path/'CLOSE.json',dict(actual_sim_s=900.,charged_sim_s=900.))
    assert shared.physics_done(tmp_path)
    write(tmp_path/'CLOSE.json',dict(actual_sim_s=899.,charged_sim_s=899.))
    with pytest.raises(ValueError,match='accounting'):
        shared.physics_done(tmp_path)


def test_both_models_share_one_alarm_and_one_elapsed_debit(tmp_path,monkeypatch):
    from tools.c1switch import adapter
    calls=[];alarms=[]
    attestation=dict(sources={},checkpoint_sha256='c',wm_config_sha256='d',candidate_sha256='e',mappo_sha256='f')
    for arm in ('bb','wm'):
        binding=dict(model=arm,adapter_sha256=sha(adapter.__file__))
        write(tmp_path/(arm+'_BINDING.json'),binding)
        write(tmp_path/(arm+'_ATTESTATION.json'),attestation)
        req=dict(id=arm,belief={},binding=binding,K=8,horizon_s=60)
        key=canonical(req);req['request_sha256']=key
        write(tmp_path/'rpc'/arm/(key+'.request.json'),req)
    write(tmp_path/'SHARED_BUDGET.json',dict(prior_gpu_s=1000,cap_s=100))
    class Model:
        def __init__(self,arm):self.arm=arm;self.attestation=attestation
        def predict(self,*args,**kwargs):
            calls.append(self.arm)
            write(tmp_path/'rpc'/self.arm/'COMPLETE',{})
            return {c:[float(i)]*8 for i,c in enumerate(CONTROLLERS)}
    monkeypatch.setattr(adapter,'create',lambda b,device:Model(b['model']))
    monkeypatch.setattr(shared,'GPU',str(tmp_path))
    monkeypatch.setattr(shared.signal,'SIGALRM',14,raising=False)
    monkeypatch.setattr(shared.signal,'ITIMER_REAL',0,raising=False)
    monkeypatch.setattr(shared.signal,'signal',lambda *a:None)
    monkeypatch.setattr(shared.signal,'setitimer',lambda timer,value:alarms.append(value),raising=False)
    shared.serve()
    assert calls==['bb','wm']
    assert alarms==[100,0]  # No separate BB/WM cap or idle double debit.
    finish=json.loads((tmp_path/'GPU_FINISHED.json').read_text())
    assert finish['cumulative_gpu_s']==1000+finish['elapsed_s']
    assert len(list(tmp_path.glob('rpc/*/*.result.json')))==2


def test_shared_cap_cannot_exceed_owner_budget(tmp_path,monkeypatch):
    write(tmp_path/'SHARED_BUDGET.json',dict(prior_gpu_s=14000,cap_s=401))
    monkeypatch.setattr(shared,'GPU',str(tmp_path))
    with pytest.raises(ValueError,match='owner GPU cap'):
        shared.serve()
