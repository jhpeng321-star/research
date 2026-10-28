"""Execution algebra, gradients, sole ownership and runtime-port regressions; CPU only."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wms.control.execution import (authority_guard, executed_switch_residual as executed_residual, idm_follow_speed,
                                   verify_authority, verify_parity)
from wms.control.nativeinit import NativeResidualCommand, sumo_residual_commands
from wms.model.cav import EXEC_CAP_KEY
from traffic_shepherd.ownership import CommandOwnership


def test_positive_residual_has_no_effect_or_gradient_and_braking_retains_gradient():
    r = torch.tensor([[.04, .07, -.1, 0.]], requires_grad=True)
    st = {EXEC_CAP_KEY: torch.tensor([[.3, -.4, .3, .3]]),
          "v_mps": torch.full((1,4), 10.)}
    native = torch.tensor([[.1, .1, .1, -.2]])
    out = executed_residual(st, r, native)
    torch.testing.assert_close(out, torch.tensor([[.3, -.4, .2, -.2]]))
    out.sum().backward()
    assert torch.equal(r.grad, torch.tensor([[0., 0., 1., 0.]]))


def test_speedmode27_does_not_clip_emergency_cap_at_policy_decel_and_obeys_stop():
    st = {EXEC_CAP_KEY: torch.tensor([[-6., -6., 2.6]]), "v_mps": torch.tensor([[10., 1., 10.]])}
    out = executed_residual(st, torch.tensor([[.1, .1, .1]]), torch.zeros(1,3))
    torch.testing.assert_close(out, torch.tensor([[-6., -2., 1.2]]))
    with pytest.raises(ValueError, match="cap"):
        executed_residual({"v_mps": st["v_mps"]}, torch.ones(1,3), torch.zeros(1,3))


def test_sumo_idm_two_substeps_matches_independent_scalar_reference():
    # Synthetic parameters, not a driver table or any fit to evaluation outcomes.
    params = {k:torch.tensor([[v]],dtype=torch.float64) for k,v in
              dict(accel=1.1,decel=2.3,time_headway=1.4,min_gap=2.1,v0=29.).items()}
    speed, gap, leader = 11., 21., 9.
    actual = idm_follow_speed(torch.tensor([[speed]],dtype=torch.float64),
                             torch.tensor([[leader]],dtype=torch.float64),
                             torch.tensor([[gap]],dtype=torch.float64), params).item()
    import math
    for _ in range(2):
        desired = max(0.,speed*1.4+speed*(speed-leader)/(2*math.sqrt(1.1*2.3)))+2.1
        accel = 1.1*(1-(speed/29.)**4-(desired/gap)**2)
        speed = max(0.,speed+.25*accel)
        gap -= max(0.,speed-leader)*.25
    assert actual == pytest.approx(speed, abs=1e-12)


class Vehicle:
    def __init__(self):
        self.calls = []
        self.gaps = []
    def getSpeedMode(self, v): return 27
    def getSpeed(self, v): return 10.
    def getLeader(self, v): return ("leader", 20.)
    def getDecel(self, v): return 2.
    def getMinGap(self, v): return 2.
    def getAccel(self, v): return 1.2
    def getAllowedSpeed(self, v): return 30.
    def getFollowSpeed(self, v, speed, gap, pred_speed, decel, leader):
        self.gaps.append(gap)
        return 10.15
    def setSpeed(self, *args): self.calls.append(("setSpeed",*args))
    def setAcceleration(self, *args): self.calls.append(("setAcceleration",*args))
    def slowDown(self, *args): self.calls.append(("slowDown",*args))
    def getNativeResidualContract(self): return "native_relative_same_step/1"
    def setNativeAccelerationResidual(self, *args): self.calls.append(("nativeResidual",*args))


def frame(vid):
    return SimpleNamespace(vehicle_id=vid,timestamp_s=300.,speed_mps=10.,allowed_speed_mps=30.,
                           front_coverage_m=100.,rear_coverage_m=100.,front=None,rear=None)


def test_single_owner_never_overwrites_fallback_and_has_no_slowdown():
    api = SimpleNamespace(vehicle=Vehicle())
    controller = SimpleNamespace(config=SimpleNamespace(control_dt_s=1.,sensor_range_m=100.))
    own = CommandOwnership(api, {"accepted","missing"},controller,cav_speed_mode=27)
    rows = own.step({"accepted":frame("accepted")},now_s=300.,
                   live_ids={"accepted","missing"},wall_commands={"accepted":.3,"missing":.3},
                   defer_wall=True)
    assert not api.vehicle.calls
    assert {r.vehicle_id:r.owner for r in rows} == {"accepted":"wall","missing":"native"}
    assert next(r for r in rows if r.vehicle_id=="accepted").issued_target_speed_mps == 30.
    for now in (300.,300.5):
        cmd = own.execute_residual({"accepted":.04,"missing":.04},-3.,1.2,
                                   now_s=now,live_ids={"accepted","missing"})
        assert cmd == pytest.approx({"accepted":.04})
    assert [r[0] for r in api.vehicle.calls] == ["nativeResidual"]*2
    assert all(r[1] == "accepted" for r in api.vehicle.calls)
    assert own.execute_residual({"accepted":.04},-3.,1.2,now_s=301.,live_ids={"accepted"}) == {}


def test_corrected_follow_query_does_not_double_add_min_gap_and_zero_releases():
    api = SimpleNamespace(vehicle=Vehicle())
    assert sumo_residual_commands(api,{"cav":0.},-3.,1.2,parity=True) == {}
    sumo_residual_commands(api,{"cav":.04},-3.,1.2,parity=True)
    assert api.vehicle.gaps[-1] == 20.
    sumo_residual_commands(api,{"cav":.04},-3.,1.2)
    assert api.vehicle.gaps[-1] == 22.


def test_guard_threshold_empty_nonfinite_and_sha_binding(tmp_path):
    import hashlib
    p = tmp_path / "policy.pt"
    p.write_bytes(b"policy")
    hashes = {"episode.npz":"a"*64}
    common = dict(checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                  recordings_sha256=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest())
    assert authority_guard([.1]*96+[-.1]*4,**common)["passed"] is False
    receipt = authority_guard([.1]*95+[-.1]*5,executed_delta=[0.]*95+[-.1]*5,**common)
    assert receipt["passed"] is True
    receipt["recordings"] = hashes
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    assert verify_authority(path,p)["passed"]
    p.write_bytes(b"new policy")
    with pytest.raises(ValueError,match="stale"):
        verify_authority(path,p)
    for r in ([],[float("nan")]):
        with pytest.raises(ValueError): authority_guard(r,**common)


def test_residual_split_and_subset_are_typed_and_native_nonmembers():
    from traffic_shepherd.c4.branching import split_output
    command = NativeResidualCommand(np.array([[.1,-.2]]),-3.,1.2)
    out, lanes = split_output((command,None))
    assert out is command and lanes is None
    assert np.array_equal(out.subset([True,False]).residual,[[.1,0.]])


def test_deploy_population_is_explicitly_selectable():
    from wms.control.deploy import parser
    args = parser().parse_args(["--controller","native","--seed","977000","--out","unused",
                                "--population","textbook"])
    assert args.population == "textbook"


def test_bridge_continuous_positive_residual_has_authority_without_soft_safety(monkeypatch):
    from .test_nativeinit import scene, networks
    from wms.control.bridge import make_cav_controller
    from wms.model.cav import WM_LANE_TARGET_KEY
    monkeypatch.setenv("C4_EXECUTION_PARITY","1")
    p, enc = networks()
    with torch.no_grad(): p.residual_head.bias.fill_(.05)
    st = scene()
    st[EXEC_CAP_KEY] = torch.full_like(st["s_m"],.3)
    st[WM_LANE_TARGET_KEY] = st["lane"].clone()
    # Deliberately different soft safety demonstrates that acting commands do
    # not flow through it. The native reference still does.
    def safety(state,accel,**kwargs): return torch.full_like(accel,-.8)
    controller = make_cav_controller(p,enc,safety,native=lambda state:torch.zeros_like(state["s_m"]),
                                     native_lane_change=True)
    out = controller(st)
    acting = st["present"] & st["is_cav"] & (st["lane"] >= 0)
    torch.testing.assert_close(out[...,3][acting],torch.full_like(out[...,3][acting],1.2*float(np.tanh(.05))))
    out[...,3][acting].sum().backward()
    assert p.residual_head.bias.grad.item() > 0.


def test_observer_port_and_population_forwarding_without_simulator(tmp_path,monkeypatch):
    from wms.control import deploy
    from traffic_shepherd.c4 import env as env_module
    calls = []
    class FakeEnv:
        now = 300.
        def __init__(self,**kwargs): calls.append(("construct",kwargs["population"]))
        def reset(self,**kwargs):
            calls.append(("reset",kwargs["controller"]))
            return {}
        def step(self,accel,lanes=None):
            assert accel is None and lanes is None
            return {},{"time_s":301.,"done":True,"summary":{"collisions":0,"teleports":0}}
    class Observer:
        def start(self,env): calls.append(("start",env.now))
        def update(self,info): calls.append(("update",info["time_s"]))
        def finish(self):
            calls.append(("finish",))
            return {"canary":"measurement"}
    monkeypatch.setattr(env_module,"CavSumoEnv",FakeEnv)
    args = deploy.parser().parse_args(["--controller","native","--seed","977000",
        "--population","textbook","--out",str(tmp_path/"out")])
    result = deploy.run(args,observer=Observer())
    assert result["status"] == "complete" and result["measurement"] == {"canary":"measurement"}
    assert calls == [("construct","textbook"),("reset","native"),("start",300.),("update",301.),("finish",)]


def test_failing_parity_cannot_be_bypassed_by_changing_passed_boolean(tmp_path,monkeypatch):
    import hashlib
    driver = tmp_path/"driver.json"
    driver.write_text("{}")
    monkeypatch.setenv("C4_EXECUTION_DRIVER",str(driver))
    receipt = {"schema":"execparity_recorded_acceleration/1","passed":True,
        "driver_sha256":hashlib.sha256(driver.read_bytes()).hexdigest(),
        "worlds":{str(i):{} for i in range(10)},"native_arm_head_error":{"rmse":.2},
        "error":{"ours":{"rmse":.17},"mappo":{"rmse":.201}},
        "c1_residual_execution_error":{"rmse":.19}}
    path = tmp_path/"parity.json"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match="failing"):
        verify_parity(path)


def test_residual_deployment_requires_guard_before_environment_construction(tmp_path,monkeypatch):
    from wms.control import deploy,policy
    from wms.control.lanes import LaneRules
    from traffic_shepherd.c4 import env as env_module
    monkeypatch.setenv("C4_EXECUTION_PARITY","1")
    monkeypatch.setattr(policy,"load_policy_fn",lambda *a,**kw:
        SimpleNamespace(lane_rules=LaneRules(),memory=None,residual_on_native=True))
    def forbidden_constructor(**kw):
        raise AssertionError("environment constructed before authority validation")
    monkeypatch.setattr(env_module,"CavSumoEnv",forbidden_constructor)
    args = deploy.parser().parse_args(["--controller","policy","--seed","977000",
        "--population","textbook","--policy",str(tmp_path/"policy.pt"),"--out",str(tmp_path/"out")])
    with pytest.raises(ValueError,match="authority"):
        deploy.run(args)


def test_wm_parity_cap_uses_learned_params_and_refuses_missing_inference(monkeypatch):
    from .test_nativeinit import scene
    from wms.model import ModelSpec,WmSModel
    from wms.control.execution import inferred_follow_cap
    monkeypatch.setenv("C4_EXECUTION_PARITY","1")
    monkeypatch.delenv("C4_EXECUTION_DRIVER",raising=False)
    model = WmSModel(ModelSpec(rollout_mode="driver_params"))
    model.cav_exec_cap = "idm"
    st = scene()
    params = {k:torch.full_like(st["s_m"],v) for k,v in
              dict(accel=.8,decel=2.3,time_headway=1.1,min_gap=2.2,v0=29.).items()}
    torch.testing.assert_close(model._exec_cap({},st,params),inferred_follow_cap(st,params))
    with pytest.raises(ValueError,match="learned driver"):
        model._exec_cap({},st,None)


def test_checkpoint_contract_cannot_silently_fall_back_to_legacy(tmp_path,monkeypatch):
    from .test_nativeinit import networks
    from wms.control.execution import contract
    from wms.control.policy import load_policy_components
    driver = tmp_path/"driver.json"
    driver.write_text(json.dumps({"schema":"execparity_learned_idm/1","fit_controller":"direct",
        "evaluation_data_used":False,"fit_worlds":{"train":"hash"},"score_worlds":{"val":"hash"},
        "parameters":dict(accel=1.1,decel=2.3,time_headway=1.4,min_gap=2.1,v0=29.)}))
    monkeypatch.setenv("C4_EXECUTION_PARITY","1")
    monkeypatch.setenv("C4_EXECUTION_DRIVER",str(driver))
    p,enc = networks()
    path = tmp_path/"policy.pt"
    torch.save({"format":"c4_dreamer/3","policy":p.state_dict(),"group_encoder":enc.state_dict(),
                "residual_on_native":True,"lane_control":"native","execution_contract":contract()},path)
    load_policy_components(path)
    monkeypatch.setenv("C4_EXECUTION_PARITY","0")
    with pytest.raises(ValueError,match="execution contract"):
        load_policy_components(path)
    monkeypatch.setenv("C4_EXECUTION_PARITY","1")
    driver.write_text(driver.read_text()+"\n")
    with pytest.raises(ValueError,match="execution contract"):
        load_policy_components(path)


def test_driver_fit_uses_boolean_masks_and_refuses_eval_before_array_access(tmp_path):
    from tools.execparity.fit_follow import rows
    directory = tmp_path/"episode"
    directory.mkdir()
    kin = np.zeros((2,3,7),np.float32)
    kin[...,0] = [0.,30.,60.]
    kin[...,1] = 1.
    kin[...,4] = 10.
    kin[1,0,4] = 10.1
    kin[...,6] = 5.
    arrays = {"veh_kin":kin,"veh_mask":np.array([[1,1,0],[1,1,0]],np.uint8),
              "is_cav":np.array([[1,0,0],[1,0,0]],np.uint8),
              "target_mask":np.array([[1,1,0]],np.uint8),"controller":np.array([[3,0,0]],np.int8),
              "issued_acceleration_mps2":np.array([[1.2,0.,0.]],np.float32),
              "time_s":np.array([0.,.5]),"control_time_s":np.array([0.])}
    for name,value in arrays.items(): np.save(directory/(name+".npy"),value)
    data,_ = rows(tmp_path,[{"seed":963100,"dir":"episode","world":"w963100m1900p20"}])
    assert data.shape == (1,7) and float(data[0,-1]) == pytest.approx(.2,abs=1e-5)
    for seed in (964000,973000,977000):
        with pytest.raises(ValueError,match="evaluation seed"):
            rows(tmp_path,[{"seed":seed,"dir":"does-not-exist","world":"refused"}])
