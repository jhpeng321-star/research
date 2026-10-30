"""Cap-relative CPU execution parity; no simulator, external data or driver fits."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.unit.test_execparity2 import SumoDomain
from tests.unit.test_keyerr import make_trainer
from tests.unit.test_nanfix import RISKY, _batch_state
from tools.execparity import alwayscmd_adapter as adapter
from wms.control import alwayscmd, execution, sumo_idm
from wms.control.nativeinit import sumo_native_accel
from wms.control.residual import executed_from_residual


class LiveDomain(SumoDomain):
    """Live-query fixture, including the lane-end getStopSpeed query."""

    def __init__(self, vehicles, *, lane_id="1487606178_0", pos=10.):
        super().__init__(vehicles)
        self.lane_id, self.pos = lane_id, pos
        self.writes = []
        self.stop_gaps = []

    def getDecel(self, vid):
        return 2.

    def getLaneID(self, vid):
        return self.lane_id

    def getLanePosition(self, vid):
        return self.pos

    def getStopSpeed(self, vid, speed, gap):
        self.stop_gaps.append(gap)
        return sumo_idm.follow_speed_scalar(gap, speed, 0.,
            self.vehicles[vid]["allowed"], respect_min_gap=False)

    def setSpeed(self, *args):
        self.writes.append(("setSpeed", *args))

    def setAcceleration(self, *args):
        self.writes.append(("setAcceleration", *args))


def scene(*, v=20., vl=20., gap=60., has=True, ramp=False, dtype=torch.float64):
    s = -20. if ramp else 500.
    speed = torch.tensor([[v, vl]], dtype=dtype)
    state = {"v_mps": speed, "a_mps2": torch.zeros_like(speed),
             "s_m": torch.tensor([[s, s + gap + 5.]], dtype=dtype),
             "lane": torch.full((1, 2), 2 if ramp else 1),
             "present": torch.tensor([[True, has]]),
             "is_cav": torch.tensor([[True, False]]),
             "length_m": torch.full_like(speed, 5.)}
    allowed = sumo_idm.allowed_speed_scalar(2 if ramp else 1, s)
    domain = LiveDomain({"ego": dict(speed=v, leader_speed=vl, leader="LEAD" if has else None,
                                    gap=gap - 2., allowed=allowed)},
                        lane_id="11553048#1_0" if ramp else "1487606178_0",
                        pos=s + 378.66 if ramp else 10.)
    return state, SimpleNamespace(vehicle=domain)


def bits(tensor):
    return tensor.detach().contiguous().cpu().numpy().tobytes()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("spec", [dict(), dict(v=8., vl=8., gap=40.),
    dict(gap=200.), dict(has=False), dict(v=10., ramp=True, has=False),
    dict(v=25., vl=0., gap=.000001), dict(v=0., vl=0., gap=5.)])
def test_zero_is_bitwise_ctrl0_in_imagination_and_deployment(monkeypatch, dtype, spec):
    monkeypatch.setenv("C4_SUMO_LEADER_RULE", "1")
    st, api = scene(dtype=dtype, **spec)
    zero = torch.zeros_like(st["v_mps"])
    ctrl0 = torch.maximum(torch.minimum(alwayscmd.inferred_base(st), alwayscmd.inferred_cap(st)),
                          -st["v_mps"] / .5)
    assert bits(alwayscmd.executed(st, zero, residual_reference="ctrl0")) == bits(ctrl0)
    assert bits(executed_from_residual(st, zero, "ctrl0")) == bits(ctrl0)
    old = alwayscmd.commands(api, {"ego": 0.}, -3., 1.2)
    new = alwayscmd.commands(api, {"ego": 0.}, -3., 1.2, residual_reference="ctrl0")
    # Same write, so SUMO receives bit-identical CTRL0 even after speed arithmetic.
    assert np.float64(old["ego"]).tobytes() == np.float64(new["ego"]).tobytes()
    assert adapter.law_commands(api, {"ego": 0.}, -3., 1.2,
                                residual_reference="ctrl0") == new


@pytest.mark.parametrize("spec", [dict(), dict(v=8., vl=8., gap=40.),
                                 dict(gap=200.), dict(has=False)])
def test_small_braking_removes_base_cap_deadzone_and_matches_live_queries(monkeypatch, spec):
    monkeypatch.setenv("C4_SUMO_LEADER_RULE", "1")
    st, api = scene(**spec)
    b, c = alwayscmd.inferred_base(st), alwayscmd.inferred_cap(st)
    assert b[0, 0] > c[0, 0]
    assert float(b[0, 0]) == sumo_native_accel(api, "ego")
    assert float(c[0, 0]) == alwayscmd.command_cap(api, "ego")
    r = torch.zeros_like(b)
    r[0, 0] = -min(float(b[0, 0] - c[0, 0]) / 4., .0001)
    zero = alwayscmd.executed(st, torch.zeros_like(r))
    assert bits(alwayscmd.executed(st, r)) == bits(zero)  # legacy is inert here
    r.requires_grad_()
    out = alwayscmd.executed(st, r, residual_reference="ctrl0")
    assert out[0, 0] < zero[0, 0]
    assert torch.autograd.grad(out[0, 0], r)[0][0, 0] == 1.
    command = alwayscmd.commands(api, {"ego": float(r[0, 0].detach())}, -3., 1.2,
                                 residual_reference="ctrl0")["ego"]
    assert float(out[0, 0].detach()) == command
    assert adapter.law_commands(api, {"ego": float(r[0, 0].detach())}, -3., 1.2,
                                residual_reference="ctrl0")["ego"] == command
    # Base double-add and cap single-add query conventions both actually occurred.
    if spec.get("has", True):
        assert ("ego", spec.get("gap", 60.)) in api.vehicle.gaps
        assert ("ego", spec.get("gap", 60.) - 2.) in api.vehicle.gaps


def test_monotone_bounds_unit_gradient_finite_difference_and_inert_positive():
    st = {"v_mps": torch.tensor([10.], dtype=torch.float64),
          "base_mps2": torch.tensor([.25], dtype=torch.float64),
          "cap_mps2": torch.tensor([.15], dtype=torch.float64)}
    requests = [-100., -4., -2., -.2, -.0001, 0., .0001, .2, 100.]
    values = [float(executed_from_residual(st, r, "ctrl0")) for r in requests]
    assert all(a <= b for a, b in zip(values, values[1:]))
    assert values[0] == -3. and values[-1] == .15
    for r0 in (-2., -.2, -.0001):
        r = torch.tensor([r0], dtype=torch.float64, requires_grad=True)
        y = executed_from_residual(st, r, "ctrl0")
        assert torch.autograd.grad(y.sum(), r)[0].item() == 1.
        fd = (executed_from_residual(st, r0 + 1e-7, "ctrl0") -
              executed_from_residual(st, r0 - 1e-7, "ctrl0")) / 2e-7
        assert float(fd) == pytest.approx(1., abs=2e-9)
    r = torch.tensor([.2], dtype=torch.float64, requires_grad=True)
    y = executed_from_residual(st, r, "ctrl0")
    assert torch.autograd.grad(y.sum(), r)[0].item() == 0.
    # Emergency cap below lo, and the stopped nonnegative-speed floor, are retained.
    for speed, cap in ((10., -8.), (0., 0.)):
        state = dict(v_mps=speed, base_mps2=.2, cap_mps2=cap)
        assert executed_from_residual(state, -100., "ctrl0") == max(-speed / .5, cap)


def test_signed_zero_bit_identity_and_state_gradients():
    v = torch.tensor([0., 10.], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([-0., .2], dtype=torch.float64, requires_grad=True)
    c = torch.tensor([-0., .1], dtype=torch.float64, requires_grad=True)
    r = torch.zeros_like(v, requires_grad=True)
    st = dict(v_mps=v, base_mps2=b, cap_mps2=c)
    ctrl0 = torch.maximum(torch.minimum(b, c), -v / .5)
    out = executed_from_residual(st, r, "ctrl0")
    assert bits(out) == bits(ctrl0)
    grads = torch.autograd.grad(out.sum(), (v, b, c, r))
    assert all(torch.isfinite(g).all() for g in grads)
    assert grads[2][1] > 0.  # physical cap sensitivity, not a detached centre


@pytest.mark.parametrize("field", ["v_mps", "base_mps2", "cap_mps2", "residual"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_inputs_fail_before_deployment_writes(field, bad):
    st = dict(v_mps=10., base_mps2=.2, cap_mps2=.1)
    r = bad if field == "residual" else -.1
    if field != "residual":
        st[field] = bad
    with pytest.raises(ValueError, match="finite"):
        executed_from_residual(st, r, "ctrl0")
    _, api = scene()
    if field == "residual":
        with pytest.raises(ValueError, match="nonfinite"):
            alwayscmd.write_residuals(api, {"ego": bad}, -3., 1.2, residual_reference="ctrl0")
        assert not api.vehicle.writes


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("spec", list(RISKY.values()))
def test_nanfix_infinite_and_tiny_gap_gradients_survive(dtype, spec):
    v, vl, gap, has, lane, s = spec
    leaves = [torch.tensor(x, dtype=dtype, requires_grad=True) for x in (v, vl, gap, -.1)]
    speed, leader, gap_t, r = leaves
    lane_t, s_t = torch.tensor(lane), torch.tensor(s, dtype=dtype)
    a = sumo_idm.executed_accel(speed, leader, gap_t, sumo_idm.allowed_speed(lane_t, s_t),
        r, -3., 1.2, has_leader=has, lane=lane_t, s_m=s_t, residual_reference="ctrl0")
    assert torch.isfinite(a)
    a.backward()
    assert all(t.grad is None or torch.isfinite(t.grad) for t in leaves)


@pytest.mark.parametrize("rule", ["0", "1"])
def test_flag_off_is_bitwise_legacy_forward_and_backward(monkeypatch, rule):
    monkeypatch.setenv("C4_SUMO_LEADER_RULE", rule)
    st = _batch_state(torch.float32, with_grad=True)
    r = torch.linspace(-4., 2., st["v_mps"].numel()).reshape_as(st["v_mps"]).requires_grad_()
    b, c = alwayscmd.inferred_base(st), alwayscmd.inferred_cap(st)
    old = torch.maximum(torch.minimum(torch.minimum(torch.maximum(b + r,
        torch.minimum(b, torch.full_like(b, -3.))), torch.maximum(b, torch.full_like(b, 1.2))), c),
        -st["v_mps"] / .5)
    new = alwayscmd.executed(st, r, residual_reference="base")
    assert bits(new) == bits(old)
    variables = (st["s_m"], st["v_mps"], r)
    old_grad = torch.autograd.grad(old.sum(), variables, retain_graph=True)
    new_grad = torch.autograd.grad(new.sum(), variables)
    assert all(bits(a) == bits(b) for a, b in zip(old_grad, new_grad))


@pytest.mark.parametrize("memory", [False, True])
def test_real_imagination_entrants_empty_ticks_and_proxy_propagation(monkeypatch, memory):
    from wms.model.cav import EXEC_CAP_KEY
    trainer = make_trainer(monkeypatch, exits=True, memory=memory)
    trainer.policy.residual_reference = "ctrl0"
    original_rollout = trainer.world_models[0].rollout
    seen = []

    class Entrants:
        def __init__(self, tick):
            self.tick = tick

        def insert(self, state):
            new = torch.zeros_like(state["present"])
            if self.tick == 3:
                new[:, 0] = True
                state["present"][:, 0] = state["is_cav"][:, 0] = True
                state["s_m"][:, 0], state["v_mps"][:, 0], state["lane"][:, 0] = 90., 8., 1
            self.tick += 1
            return new

    def rollout(*args, k0=0, history=None, **kwargs):
        return original_rollout(*args, k0=k0, history=history, entries=Entrants(k0), **kwargs)

    original_execute = execution.executed_residual

    def observe(st, *args, **kwargs):
        assert kwargs["residual_reference"] == "ctrl0"
        assert EXEC_CAP_KEY in st and torch.isfinite(st[EXEC_CAP_KEY]).all()
        seen.append(bool((st["present"] & st["is_cav"]).any()))
        return original_execute(st, *args, **kwargs)

    monkeypatch.setattr(trainer.world_models[0], "rollout", rollout)
    monkeypatch.setattr(execution, "executed_residual", observe)
    result = trainer.imagine(trainer.start_states.batch, deterministic=True)
    assert torch.isfinite(result["rewards"]).all()
    result["returns"].sum().backward()
    assert False in seen and seen[-1] is True
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in trainer.actor_parameters)


def test_typed_policy_output_ownership_writer_and_subset(monkeypatch):
    from tests.unit.test_execparity_alwayscmd import domain
    from traffic_shepherd.ownership import CommandOwnership
    from wms.control.policy import CavPolicy, PolicyExecutor, RsuGroupEncoder
    monkeypatch.setenv("C4_EXECUTION_PARITY", "2")
    st, api = scene(dtype=torch.float32)
    policy = CavPolicy(hidden_dim=8, n_heads=2, lane_head=False, residual_on_native=True,
                       residual_reference="ctrl0")
    exe = PolicyExecutor(policy, RsuGroupEncoder(hidden_dim=8, n_heads=2))
    command = exe(st)
    assert command.residual_reference == "ctrl0"
    assert command.subset([True, False]).residual_reference == "ctrl0"
    controller = SimpleNamespace(config=SimpleNamespace(control_dt_s=1., sensor_range_m=100.))
    own = CommandOwnership(api, {"ego"}, controller, cav_speed_mode=27)
    own._owned["ego"] = "wall"
    own._last["ego"] = SimpleNamespace(timestamp_s=300.)
    issued = own.execute_residual({"ego": -.0001}, -3., 1.2, now_s=300., live_ids={"ego"},
                                  residual_reference=command.residual_reference)
    assert issued == alwayscmd.commands(api, {"ego": -.0001}, -3., 1.2,
                                        residual_reference="ctrl0")
    assert [r[0] for r in api.vehicle.writes] == ["setSpeed", "setAcceleration"]
    api.vehicle.getIDList = lambda: ["ego"]
    rows = alwayscmd.completed_rows(api, {"ego": 20.}, {"ego": -.0001}, issued)
    assert rows["ego"]["base_law_mps2"] == sumo_native_accel(api, "ego")
    assert rows["ego"]["ctrl0_mps2"] == alwayscmd.command_cap(api, "ego")
    assert rows["ego"]["residual_reference"] == "ctrl0"
    # OFF fixtures need none of the new live cap API calls.
    assert alwayscmd.commands(SimpleNamespace(vehicle=domain()), {"free": .1}, -3., 1.2)


def test_config_extends_exact_parent_and_default_payload_is_unchanged():
    from wms.control.train_cli import _read_config, policy_settings
    root = Path(__file__).resolve().parents[2]
    config = root / "configs/c4_policy14_candA_caprel.toml"
    parent = _read_config(root / "configs/c4_policy14_candA_longgrad_skip.toml")
    actual = _read_config(config)
    assert policy_settings(actual)["residual_reference"] == "ctrl0"
    actual["policy"].pop("residual_reference")
    assert actual == parent
    assert "residual_reference" not in policy_settings(parent)


def test_caprel_checkpoint_metadata_roundtrip_and_explicit_phase(monkeypatch, tmp_path):
    from wms.control.policy import load_policy_components
    trainer = make_trainer(monkeypatch, exits=False, memory=False)
    monkeypatch.setenv("C4_ALLOWED_ROOTS", str(tmp_path))
    # Standard-size components let the deployment loader exercise its usual constructor.
    from wms.control.policy import CavPolicy, RsuGroupEncoder
    from wms.control.train_cli import GroupSignal
    trainer.policy = CavPolicy(lane_head=False, residual_on_native=True, residual_reference="ctrl0")
    trainer.group_encoder = GroupSignal(RsuGroupEncoder())
    driver = tmp_path / "driver.json"
    driver.write_text(json.dumps(dict(schema="execparity_learned_idm/1", fit_controller="direct",
        evaluation_data_used=False, fit_worlds=["a"], score_worlds=["b"],
        parameters=dict(accel=1.1, decel=2., time_headway=1.2, min_gap=2., v0=28.))))
    monkeypatch.setenv("C4_EXECUTION_DRIVER", str(driver))
    path = tmp_path / "policy.pt"
    trainer.save_checkpoint(path)
    loaded, _, _, payload = load_policy_components(path)
    assert loaded.residual_reference == payload["residual_reference"] == "ctrl0"
    assert payload["execution_contract"] == execution.contract("ctrl0")
    trainer.policy.residual_reference = "base"
    with pytest.raises(ValueError, match="explicit new phase"):
        trainer.load_checkpoint(path)


def test_old_parity_receipt_cannot_qualify_caprel(tmp_path, monkeypatch):
    monkeypatch.setenv("C4_EXECUTION_PARITY", "2")
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"residual_reference": "base"}))
    with pytest.raises(ValueError, match="residual_reference"):
        execution.verify_parity(path, residual_reference="ctrl0")
    path.write_text(json.dumps({"residual_reference": "ctrl0", "caprel_source_sha256": {}}))
    with pytest.raises(ValueError, match="source binding"):
        execution.verify_parity(path, residual_reference="ctrl0")


def test_caprel_replay_rejects_old_law_logs(tmp_path, monkeypatch):
    from tests.unit.test_execparity_alwayscmd_replay import _job, _row
    from tools.execparity.alwayscmd_replay import world_rows
    job = _job(tmp_path, [20., 20.3], [_row(300., "cav1", 20., residual=-.0001)])
    with pytest.raises(ValueError, match="residual_reference"):
        world_rows(job, residual_reference="ctrl0")


def test_live_ramp_stop_cap_matches_imagination(monkeypatch):
    monkeypatch.setenv("C4_SUMO_LEADER_RULE", "1")
    st, api = scene(v=10., ramp=True, has=False)
    expected = float(alwayscmd.inferred_cap(st)[0, 0])
    assert alwayscmd.command_cap(api, "ego") == expected
    assert api.vehicle.stop_gaps == [pytest.approx(75.44 + 20.)]
    assert math.isfinite(expected)


def test_positive_deployment_commands_keep_exact_ctrl0_write(monkeypatch):
    monkeypatch.setenv("C4_SUMO_LEADER_RULE", "1")
    st, api = scene()
    for r in (.0001, .1, 100.):
        assert alwayscmd.commands(api, {"ego": r}, -3., 1.2,
                                  residual_reference="ctrl0") == alwayscmd.commands(
                                      api, {"ego": 0.}, -3., 1.2)
        zero = torch.zeros_like(st["v_mps"])
        assert bits(alwayscmd.executed(st, zero + r, residual_reference="ctrl0")) == bits(
            alwayscmd.executed(st, zero, residual_reference="ctrl0"))


def test_caprel_replay_accepts_only_matching_measured_sources(tmp_path):
    import gzip
    from tests.unit.test_execparity_alwayscmd_replay import _job, _row
    from tools.execparity.alwayscmd_replay import caprel_sources, world_rows
    r = {**_row(300., "cav1", 20., residual=-.0001), "residual_reference": "ctrl0"}
    job = _job(tmp_path, [20., 20.3], [r])
    result = job / "result.json"
    result.write_text(json.dumps({"caprel_source_sha256": {}}))
    with pytest.raises(ValueError, match="source hashes"):
        world_rows(job, residual_reference="ctrl0")
    result.write_text(json.dumps({"caprel_source_sha256": caprel_sources()}))
    rows, _ = world_rows(job, residual_reference="ctrl0")
    assert len(rows) == 1
    from wms.control.alwayscmd import scalar_executed
    assert rows[0]["imagined"] == pytest.approx(
        scalar_executed(20., 20., 55., -.0001, residual_reference="ctrl0"), abs=1e-6)
    # A log's label must agree even when the result carries matching source hashes.
    with gzip.open(job / "execution_log.jsonl.gz", "wt", encoding="utf-8") as log:
        log.write(json.dumps({**r, "residual_reference": "base"}) + "\n")
    with pytest.raises(ValueError, match="residual_reference"):
        world_rows(job, residual_reference="ctrl0")


def test_probe_adapter_exercises_negative_actions_without_changing_zero_canary():
    from wms.control.nativeinit import NativeResidualCommand
    command = NativeResidualCommand(np.array([[.1, .2]]), -3., 1.2)
    class Executor:
        residual_on_native = True

        def __call__(self, *args, **kwargs):
            return command

    probe = adapter.AlwaysCommanded(Executor(), force_zero=False, probe_residual=-.01)
    canary = adapter.AlwaysCommanded(Executor(), force_zero=True, probe_residual=-.01)
    assert (probe({}).residual == -.01).all()
    assert (canary({}).residual == 0.).all()


def test_caprel_parity_receipt_and_authority_bind_new_contract(tmp_path, monkeypatch):
    import hashlib
    from tools.execparity.alwayscmd_replay import caprel_sources
    monkeypatch.setenv("C4_EXECUTION_PARITY", "2")
    driver = tmp_path / "driver.json"
    driver.write_text(json.dumps(dict(schema="execparity_learned_idm/1", fit_controller="direct",
        evaluation_data_used=False, fit_worlds=["a"], score_worlds=["b"],
        parameters=dict(accel=1.1, decel=2., time_headway=1.2, min_gap=2., v0=28.))))
    monkeypatch.setenv("C4_EXECUTION_DRIVER", str(driver))
    good = dict(schema="execparity_always_commanded_acceleration/1",
        backend_contract="always_commanded_ctrl0/1", measured_real_sumo=True,
        zero_ctrl0_identity=True, passed=True, worlds=["977001"],
        driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
        native_arm_head_error=dict(rmse=.2), c1_residual_execution_error=dict(rmse=.1),
        error=dict(ours=dict(rmse=.1)), residual_reference="ctrl0",
        caprel_source_sha256=caprel_sources())
    path = tmp_path / "parity.json"
    path.write_text(json.dumps(good))
    assert execution.verify_parity(path, residual_reference="ctrl0")["passed"]
    with pytest.raises(ValueError, match="residual_reference"):
        execution.verify_parity(path)
    authority = execution.authority_guard([-.01], executed_delta=[-.01], capped=[False],
        checkpoint_sha256="a" * 64, recordings_sha256="b" * 64, residual_reference="ctrl0")
    assert authority["execution_contract"] == execution.contract("ctrl0")
    monkeypatch.setenv("C4_EXECUTION_PARITY", "0")
    with pytest.raises(ValueError, match="mode 2"):
        execution.contract("ctrl0")
