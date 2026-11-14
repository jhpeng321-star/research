"""Actuation evidence observes the real safety boundary without changing commands."""
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tools.coopfree.execution import inject
from .acceleration import requests


@pytest.mark.parametrize('offset,base_action', [(-.4,.15),(.5,1.5)])
def test_receipt_counterfactual_measures_actual_issued_delta_and_keeps_commands(offset, base_action, monkeypatch):
    safety = ModuleType('wms.control.safety')
    calls = []
    def safe(state, raw, **kwargs):
        calls.append(raw.clone())
        return raw.clamp(-1.2, 1.2)
    safety.safe_accel = safe
    control = ModuleType('wms.control'); control.safety = safety
    package = ModuleType('wms'); package.control = control
    for name, module in [('wms',package),('wms.control',control),('wms.control.safety',safety)]:
        monkeypatch.setitem(sys.modules,name,module)
    original = safe
    executor = SimpleNamespace(accel_min=-3., accel_max=2.6)
    env = SimpleNamespace(slot_vehicle_ids=lambda:['a','hdv'])
    rows = [dict(vid='a',is_cav=True),dict(vid='hdv',is_cav=False)]
    outcomes = []
    for enabled in (False, True):
        sample = dict(context=dict(members=['a'], move_offsets={'a':offset}),
            action=[[base_action+offset]], own_action=[[base_action+offset]], coop_layer={})
        if enabled:
            sample['coop_layer'].update(base_action=[[base_action]],applied_correction=[[offset]])
        controller = SimpleNamespace(apply=lambda rows, commands, now:requests(sample,commands))
        audit=[];before=len(calls)
        with inject(executor,env,rows,controller,300.,audit):
            issued = safety.safe_accel({},torch.tensor([[0.,.3]]))
        assert safety.safe_accel is original
        assert len(calls)-before == (3 if enabled else 2)
        assert {r['vid'] for r in audit} == {'a'}
        outcomes.append(issued)
        if enabled:
            receipt, = audit
            without = min(1.2,max(-1.2,base_action))
            assert receipt['without_library_issued_safe'] == pytest.approx(without)
            assert receipt['library_issued_delta_mps2'] == pytest.approx(float(issued[0,0])-without, abs=1e-6)
            assert receipt['requested_raw']-receipt['without_library_raw'] == pytest.approx(offset)
            # Saturated moves are correctly logged as zero issued change; they
            # must never satisfy the smoke's nonzero-actuation admission alone.
            assert (abs(receipt['library_issued_delta_mps2']) > 1e-6) == (base_action < 1.2)
            from .clean2_receipts import update_evidence
            episode=dict(id='actual_fake_tick',steps=[dict(time_s=300,sample=sample,library_actuation_audit=audit)])
            evidence=update_evidence('B',[episode],dict(bank_trigger={}),dict(receipt={}),{})
            assert evidence['actuation']['full_strength_zero_residual_member_ticks'] == 1
            assert evidence['actuation'].get('physically_changed_issued_commands',0) == (base_action < 1.2)
        else:
            assert 'library_issued_delta_mps2' not in audit[0]
    assert torch.equal(*outcomes)
    assert outcomes[1][0,1] == pytest.approx(.3)  # HDV command never changed


def test_actual_registered_safety_filter_receipt_is_observational_only():
    from wms.control import safety
    original=safety.safe_accel
    state=dict(present=torch.tensor([[True,True]]),is_cav=torch.tensor([[True,False]]),
        lane=torch.tensor([[0,1]]),s_m=torch.tensor([[-50.,-20.]]),
        v_mps=torch.tensor([[15.,12.]]),length_m=torch.tensor([[5.,5.]]))
    before={k:v.clone() for k,v in state.items()}
    executor=SimpleNamespace(accel_min=-3.,accel_max=2.6)
    env=SimpleNamespace(slot_vehicle_ids=lambda:['a','hdv'])
    rows=[dict(vid='a',is_cav=True),dict(vid='hdv',is_cav=False)]
    outcomes=[]
    for enabled in (False,True):
        sample=dict(context=dict(members=['a'],move_offsets={'a':-.4}),action=[[-.25]],coop_layer={})
        if enabled:sample['coop_layer']['base_action']=[[.15]]
        controller=SimpleNamespace(apply=lambda rows,commands,now:requests(sample,commands))
        audit=[]
        with inject(executor,env,rows,controller,300.,audit):
            issued=safety.safe_accel(state,torch.tensor([[0.,.2]]),accel_max=1.2)
        outcomes.append(issued)
        assert safety.safe_accel is original
        assert all(torch.equal(state[k],v) for k,v in before.items())
        if enabled:
            assert audit[0]['library_issued_delta_mps2']==pytest.approx(-.4)
            assert audit[0]['without_library_issued_safe']==pytest.approx(.15)
    assert torch.equal(*outcomes)
    assert outcomes[1][0,1]==pytest.approx(.2)
