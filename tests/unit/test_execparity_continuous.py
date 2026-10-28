"""Same-step law, deployment zero identity, and stock TraCI timing counterexample."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from wms.control.continuous import (BACKEND_CONTRACT, native_relative, project_native,
                                    scalar_native_relative, write_residuals)
from wms.control.execution import executed_residual, authority_guard
from wms.model.cav import EXEC_CAP_KEY


def test_tensor_scalar_law_agree_on_unconstrained_capped_braking_and_emergency():
    rng = np.random.default_rng(20261028)
    speed = rng.uniform(0.,30.,10000)
    cap = rng.uniform(-7.,1.2,10000)
    cap = np.maximum(cap,-speed/.5)
    native = rng.uniform(0.,1.,10000)*(cap+speed/.5)-speed/.5
    r = rng.uniform(-1.2,1.2,10000)
    st = {"v_mps":torch.tensor(speed), EXEC_CAP_KEY:torch.tensor(cap)}
    actual = native_relative(st,torch.tensor(r),torch.tensor(native)).numpy()
    expected = np.array([scalar_native_relative(v,v+.5*a,c,x)
                         for v,a,c,x in zip(speed,native,cap,r)])
    np.testing.assert_allclose(actual,expected,atol=2e-14,rtol=1e-13)


def test_positive_residual_has_credit_until_the_actual_cap_then_zero_gradient():
    native = torch.tensor([-.6,.2,-6.])
    st = {"v_mps":torch.full((3,),10.),EXEC_CAP_KEY:torch.tensor([.3,.3,-5.])}
    r = torch.tensor([.04,.2,.04],requires_grad=True)
    out = executed_residual(st,r,native)
    torch.testing.assert_close(out,torch.tensor([-.56,.3,-5.96]))
    out.sum().backward()
    assert torch.equal(r.grad,torch.tensor([1.,0.,1.]))


def test_zero_tensor_identity_and_trainable_zero_initialisation():
    native = torch.tensor([-.8,.1,.3,-6.,-0.],dtype=torch.float64)
    st = {"v_mps":torch.full((5,),10.),EXEC_CAP_KEY:torch.tensor([.3,.3,.3,-5.,.3])}
    r = torch.zeros_like(native,requires_grad=True)
    out = executed_residual(st,r,native)
    assert out.detach().numpy().tobytes() == native.numpy().tobytes()
    out.sum().backward()
    assert torch.equal(r.grad[:2],torch.ones(2,dtype=torch.float64))
    for eps in (1e-2,1e-5,1e-8):
        tiny = executed_residual(st,torch.full_like(native,eps),native)
        assert torch.max(torch.abs(tiny-native)) <= eps*(1.+1e-6)


class SameStepFixture:
    """Fixture, NOT stock SUMO: hook runs after native LC speed is computed."""
    def __init__(self):
        self.speed = 10.
        self.position = 0.
        self.pending = 0.
        self.calls = []
    def getNativeResidualContract(self): return BACKEND_CONTRACT
    def getSpeedMode(self,vid): return 27
    def setNativeAccelerationResidual(self,vid,r,hi,lo,dt):
        self.calls.append((vid,r,hi,lo,dt))
        self.pending = r
    def advance(self,native_accel,cap):
        self.speed += .5*scalar_native_relative(self.speed,self.speed+.5*native_accel,
                                               cap,self.pending)
        self.position += .5*self.speed
        self.pending = 0.
        return self.position,self.speed


def test_zero_residual_deployment_writer_bit_identical_to_native_with_yielding():
    native,zero = SameStepFixture(),SameStepFixture()
    a,b = [],[]
    for acc,cap in ((.2,.3),(-.8,.3),(-1.1,.2),(.1,.3),(-6.,-5.)):
        assert write_residuals(SimpleNamespace(vehicle=zero),{"cav":0.},-3.,1.2) == {}
        a.append(native.advance(acc,cap))
        b.append(zero.advance(acc,cap))
    assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    assert zero.calls == []


def test_small_residual_preserves_same_step_yielding_in_backend_fixture():
    domain = SameStepFixture()
    write_residuals(SimpleNamespace(vehicle=domain),{"cav":.00001},-3.,1.2)
    _,speed = domain.advance(-.8,.3)
    assert speed == pytest.approx(10.+.5*(-.8+.00001),abs=1e-14)
    assert len(domain.calls) == 1


def test_stock_api_rejected_before_writes_and_zero_needs_no_backend():
    class Stock:
        def getSpeedWithoutTraCI(self,vid): raise AssertionError("stale getter must not be used")
        def setAcceleration(self,*args): raise AssertionError("absolute takeover forbidden")
    api = SimpleNamespace(vehicle=Stock())
    assert write_residuals(api,{"cav":0.},-3.,1.2) == {}
    with pytest.raises(ValueError,match="completed-step cache"):
        write_residuals(api,{"cav":.01},-3.,1.2)


def test_cached_no_traci_speed_is_not_next_native_and_leaves_a_switch():
    # At t=300, the last native speed and actual speed are both 10. The
    # *upcoming* LC yield selects 9.6. A pre-step cached getter cannot see it.
    speed, cached, next_native, eps = 10.,10.,9.6,1e-8
    native_acc = (next_native-speed)/.5
    stale_command = (cached-speed)/.5+eps
    assert abs(stale_command-native_acc) > .79
    # Even linear extrapolation from the previous native acceleration is wrong
    # when a new cooperative yield starts this step.
    previous_acc = .2
    assert abs(previous_acc+eps-native_acc) > .99


def test_continuous_guard_measures_effect_not_residual_sign():
    common = dict(checkpoint_sha256="a"*64,recordings_sha256="b"*64)
    assert authority_guard([.1]*100,executed_delta=[.02]*100,**common)["passed"]
    assert not authority_guard([-.1]*100,executed_delta=[0.]*96+[-.1]*4,**common)["passed"]


def test_infeasible_head_projection_is_shared_by_native_and_policy_rows():
    st = {"v_mps":torch.tensor([10.,1.]),EXEC_CAP_KEY:torch.tensor([.3,-6.])}
    projected = project_native(st,torch.tensor([2.6,-9.]))
    assert torch.equal(projected,torch.tensor([.3,-2.]))
    assert torch.equal(native_relative(st,torch.zeros(2),projected),projected)


def test_execution_log_uses_completed_same_step_base_and_records_actual_response():
    from wms.control.continuous import completed_execution_rows
    domain = SimpleNamespace(getIDList=lambda:["cav"],
        getSpeedWithoutTraCI=lambda vid:9.6,getSpeed=lambda vid:9.605)
    row = completed_execution_rows(SimpleNamespace(vehicle=domain),{"cav":10.},
                                   {"cav":.01},{"cav":.01},-3.,1.2)["cav"]
    assert row["same_step_no_traci_speed_mps"] == 9.6
    assert row["issued_acceleration_before_safe_cap_mps2"] == pytest.approx(-.79)
    assert row["realised_acceleration_mps2"] == pytest.approx(-.79)


def test_proxy_parity_report_cannot_authorise_deployment(tmp_path,monkeypatch):
    import json
    from wms.control.execution import verify_parity
    monkeypatch.setenv("C4_EXECUTION_DRIVER",str(tmp_path/"driver.json"))
    path = tmp_path/"proxy.json"
    path.write_text(json.dumps({"schema":"execparity_continuous_recorded_state/1",
                               "passed":True,"proxy_within_native_head_error":True}))
    with pytest.raises(ValueError,match="failing"):
        verify_parity(path)


def test_authority_is_bound_to_the_execution_contract(tmp_path,monkeypatch):
    import hashlib,json
    from wms.control import execution
    policy = tmp_path/"policy.pt"
    policy.write_bytes(b"checkpoint")
    hashes = {"native/episode.npz":"a"*64}
    monkeypatch.setattr(execution,"contract",lambda:{"executor":"one"})
    receipt = authority_guard([.1],executed_delta=[.01],
        checkpoint_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
        recordings_sha256=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest())
    receipt["recordings"] = hashes
    path = tmp_path/"authority.json"
    path.write_text(json.dumps(receipt))
    assert execution.verify_authority(path,policy)["passed"]
    monkeypatch.setattr(execution,"contract",lambda:{"executor":"two"})
    with pytest.raises(ValueError,match="stale"):
        execution.verify_authority(path,policy)
