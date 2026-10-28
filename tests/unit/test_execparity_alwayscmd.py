"""00:20 EXACT ALWAYS-COMMANDED law: CTRL0 identity, continuity, formula identity; CPU only."""
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wms.control.alwayscmd import (BACKEND_CONTRACT, commands, completed_rows, executed,
                                   inferred_base, scalar_executed, write_residuals)
from wms.control.execution import contract, inferred_follow_cap, mode
from wms.control.nativeinit import sumo_native_accel
from wms.model.cav import EXEC_CAP_KEY


class Domain:
    """Deterministic TraCI vehicle-domain stand-in; no SUMO involved."""

    def __init__(self, vehicles):
        self.vehicles = vehicles
        self.calls = []
        self.gaps = []

    def getSpeed(self, v):
        return self.vehicles[v]["speed"]

    def getLeader(self, v):
        return None if self.vehicles[v]["leader"] is None else ("l", self.vehicles[v]["gap"])

    def getDecel(self, v):
        return self.vehicles[v]["decel"]

    def getMinGap(self, v):
        return self.vehicles[v]["min_gap"]

    def getAccel(self, v):
        return self.vehicles[v]["accel"]

    def getAllowedSpeed(self, v):
        return self.vehicles[v]["allowed"]

    def getFollowSpeed(self, v, speed, gap, pred_speed, decel, leader_id):
        self.gaps.append((v, gap))
        # deterministic SUMO-like follow target; exact shape is irrelevant to the law
        return float(np.clip(pred_speed + 0.01 * (gap - speed), 0., 40.))

    def getSpeedMode(self, v):
        return 27

    def setSpeed(self, *args):
        self.calls.append(("setSpeed", *args))

    def setAcceleration(self, *args):
        self.calls.append(("setAcceleration", *args))

    def slowDown(self, *args):
        self.calls.append(("slowDown", *args))


def domain():
    return Domain({
        "free": dict(speed=10., leader=None, gap=math.inf, decel=2., min_gap=2.,
                     accel=1.2, allowed=30.),
        "follow": dict(speed=8., leader="l", gap=12.5, decel=2.2, min_gap=2.5,
                       accel=1.2, allowed=28.),
        "tight": dict(speed=3., leader="l2", gap=4., decel=3., min_gap=1.5,
                      accel=1.2, allowed=25.5),
        "l": dict(speed=7.5, leader=None, gap=math.inf, decel=2.6, min_gap=2.5,
                  accel=1.2, allowed=28.),
        "l2": dict(speed=2.2, leader=None, gap=math.inf, decel=2.9, min_gap=1.5,
                   accel=1.2, allowed=25.5),
        "accepted": dict(speed=8., leader="l", gap=18., decel=2.2, min_gap=2.5,
                         accel=1.2, allowed=28.),
    })


def ctrl0_forced_base(api, residuals):
    """S-CTRLARM tools/ctrlarm/zero.py forced_base_commands, re-implemented verbatim."""
    return {vid: float(sumo_native_accel(api, vid, 0.5)) for vid in residuals}


def test_zero_residual_commands_are_bit_identical_to_ctrl0_forced_base():
    api = SimpleNamespace(vehicle=domain())
    residuals = {vid: 0.0 for vid in ("free", "follow", "tight", "l", "l2")}
    mine = commands(api, residuals, -3., 1.2)
    reference = ctrl0_forced_base(api, residuals)
    assert mine == reference
    assert (json.dumps(mine, sort_keys=True).encode()
            == json.dumps(reference, sort_keys=True).encode())


def test_remote_adapter_law_copy_is_bit_identical_to_the_tested_module():
    import importlib.util
    from pathlib import Path
    adapter_path = Path(__file__).resolve().parents[2] / "tools/execparity/alwayscmd_adapter.py"
    spec = importlib.util.spec_from_file_location("alwayscmd_adapter", adapter_path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    rng = np.random.default_rng(20261029)
    for trial in range(50):
        api = SimpleNamespace(vehicle=domain())
        residuals = {vid: float(choice) for vid, choice in
                     zip(("free", "follow", "tight"), rng.choice(
                         [0.0, 0.04, -0.4, 3.7, -4.1, 0.7], size=3))}
        assert adapter.law_commands(api, residuals, -3., 1.2) == commands(api, residuals, -3., 1.2)


def test_nonzero_residual_uses_the_frozen_legacy_envelope_exactly():
    api = SimpleNamespace(vehicle=domain())
    for vid in ("free", "follow", "tight"):
        base = float(sumo_native_accel(api, vid, 0.5))
        for r in (0.3, -0.5, -3.5, 5.0):
            upper = min(1.2, api.vehicle.getAccel(vid))
            reference = float(np.clip(base + r, min(-3., base), max(upper, base)))
            assert commands(api, {vid: r}, -3., 1.2)[vid] == reference
    # the frozen legacy gap convention (gap + minGap) is retained, not the corrected one
    assert any(abs(g - 12.5 - 2.5) < 1e-12 for v, g in api.vehicle.gaps if v == "follow")


def test_sole_writer_commands_every_cav_including_exact_zero():
    api = SimpleNamespace(vehicle=domain())
    issued = write_residuals(api, {"free": 0.0, "follow": 0.04, "tight": -0.4}, -3., 1.2)
    assert set(issued) == {"free", "follow", "tight"}
    assert [c[0] for c in api.vehicle.calls] == ["setSpeed", "setAcceleration"] * 3
    assert all(c[2] == -1 for c in api.vehicle.calls[0::2])
    for c, value in zip(api.vehicle.calls[1::2], issued.values()):
        assert c[3] == 0.5 and c[2] == value
    with pytest.raises(ValueError, match="nonfinite"):
        write_residuals(api, {"free": float("nan")}, -3., 1.2)


DRIVER_VALUES = dict(accel=1.15, decel=2.4, time_headway=1.35, min_gap=2.2, v0=29.)


def params_tensor(shape, dtype=torch.float64):
    return {k: torch.full(shape, v, dtype=dtype) for k, v in DRIVER_VALUES.items()}


@pytest.fixture
def explicit_geometry(monkeypatch):
    """Make leader_gap_dv return the exact tensors the scalar law is fed with."""
    from wms.control import native_cav
    holder = {}

    def fake(state):
        return (holder["gap"], holder["dv"], holder["has"])

    monkeypatch.setattr(native_cav, "leader_gap_dv", fake)
    return holder


def test_imagination_law_matches_independent_scalar_formula(explicit_geometry):
    from wms.control.contract import MERGE_S1_M
    rng = np.random.default_rng(20261029)
    n = 4000
    dtype = torch.float64
    v_np = rng.uniform(0., 30., n)
    leader_np = rng.uniform(0., 30., n)
    gap_np = rng.uniform(0.5, 200., n)
    has_np = rng.uniform(size=n) < .8
    lane_np = np.where(rng.uniform(size=n) < .3, 2, 1)
    s_np = rng.uniform(-50., 300., n)
    r_np = rng.uniform(-4., 2., n)
    state = {"v_mps": torch.tensor(v_np, dtype=dtype)[None],
             "s_m": torch.tensor(s_np, dtype=dtype)[None],
             "lane": torch.tensor(lane_np)[None],
             "present": torch.ones(1, n, dtype=torch.bool),
             "length_m": torch.full((1, n), 5., dtype=dtype)}
    explicit_geometry["gap"] = torch.where(torch.tensor(has_np), torch.tensor(gap_np, dtype=dtype),
                                           torch.zeros(n, dtype=dtype))[None]
    explicit_geometry["dv"] = (torch.tensor(v_np, dtype=dtype)
                               - torch.tensor(leader_np, dtype=dtype))[None]
    explicit_geometry["has"] = torch.tensor(has_np)[None]
    params = params_tensor((1, n), dtype)
    state[EXEC_CAP_KEY] = inferred_follow_cap(state, params)
    tensor_out = executed(state, torch.tensor(r_np, dtype=dtype)[None], params=params)[0].numpy()
    scalar_out = np.array([
        scalar_executed(v_np[i], leader_np[i] if has_np[i] else v_np[i],
                        gap_np[i] if has_np[i] else float("inf"), r_np[i], DRIVER_VALUES,
                        ramp_gap=(max(MERGE_S1_M - s_np[i], 0.) if lane_np[i] == 2 else None))
        for i in range(n)])
    assert np.max(np.abs(tensor_out - scalar_out)) < 2e-9


def test_zero_is_the_ctrl0_law_and_capped_positive_residual_has_zero_gradient(explicit_geometry):
    dtype = torch.float64
    n = 3
    v = torch.tensor([[10., 10., 10.]], dtype=dtype)
    explicit_geometry["gap"] = torch.tensor([[15., 60., 15.]], dtype=dtype)
    explicit_geometry["dv"] = torch.tensor([[2., 0., 2.]], dtype=dtype)
    explicit_geometry["has"] = torch.tensor([[True, False, True]])
    st = {"v_mps": v, "s_m": torch.tensor([[500., 500., 500.]], dtype=dtype),
          "lane": torch.tensor([[1, 1, 1]]), "present": torch.ones(1, n, dtype=torch.bool),
          "length_m": torch.full((1, n), 5., dtype=dtype)}
    params = params_tensor((1, n), dtype)
    st[EXEC_CAP_KEY] = inferred_follow_cap(st, params)
    base = inferred_base(st, params)
    zero = executed(st, torch.zeros_like(v), params=params)
    torch.testing.assert_close(zero, torch.maximum(torch.minimum(base, st[EXEC_CAP_KEY]), -v / .5))
    r = torch.tensor([[-.5, -.2, .5]], dtype=dtype, requires_grad=True)
    out = executed(st, r, params=params)
    out_value = out.detach().clone()
    out.sum().backward()
    # row 0 (leader): braking residual with headroom -> strictly below the CTRL0 law
    assert float(out_value[0, 0]) < float(zero[0, 0]) - 1e-9 and r.grad[0, 0].item() == 1.
    # row 1 (no leader): free-road braking also executes with unit gradient
    assert float(out_value[0, 1]) < float(zero[0, 1]) - 1e-9 and r.grad[0, 1].item() == 1.
    # row 2 (leader): positive residual beyond the safe-speed cap -> inert, no gradient
    assert float(out_value[0, 2]) == float(zero[0, 2]) and r.grad[0, 2].item() == 0.


def test_tiny_residual_continuity_both_sides_of_zero(explicit_geometry):
    dtype = torch.float64
    explicit_geometry["gap"] = torch.tensor([[15., 120.]], dtype=dtype)
    explicit_geometry["dv"] = torch.tensor([[1., 0.]], dtype=dtype)
    explicit_geometry["has"] = torch.tensor([[True, False]])
    st = {"v_mps": torch.tensor([[12., 5.]], dtype=dtype),
          "s_m": torch.tensor([[400., 120.]], dtype=dtype),
          "lane": torch.tensor([[1, 2]]), "present": torch.ones(1, 2, dtype=torch.bool),
          "length_m": torch.full((1, 2), 5., dtype=dtype)}
    params = params_tensor((1, 2), dtype)
    st[EXEC_CAP_KEY] = inferred_follow_cap(st, params)
    zero = executed(st, torch.zeros((1, 2), dtype=dtype), params=params)
    for eps in (1e-2, 1e-5, 1e-8):
        for sign in (1., -1.):
            tiny = executed(st, torch.full((1, 2), sign * eps, dtype=dtype), params=params)
            assert torch.max(torch.abs(tiny - zero)) <= eps * (1. + 1e-9)


def test_executor_algebra_ctrl0_identity_and_inert_positive_residual():
    # In the always-commanded law the follow cap never exceeds the legacy base law
    # (double-add base target >= single-add cap target), so CTRL0 (r=0) executes
    # max(-v/dt, cap) and every positive residual is exactly inert.
    rng = np.random.default_rng(20261029)
    n = 2000
    v = torch.tensor(rng.uniform(0., 30., (1, n)), dtype=torch.float64)
    base = torch.tensor(rng.uniform(-3., 2., (1, n)), dtype=torch.float64)
    cap = torch.minimum(base, torch.tensor(rng.uniform(-7., 1.2, (1, n)), dtype=torch.float64))
    r = torch.tensor(rng.uniform(-4., 2., (1, n)), dtype=torch.float64)
    upper = torch.maximum(base, torch.full_like(base, 1.2))
    lower = torch.minimum(base, torch.full_like(base, -3.))
    issued = torch.minimum(torch.maximum(base + r, lower), upper)
    reference = torch.maximum(torch.minimum(issued, cap), -v / .5)
    zero_reference = torch.maximum(torch.minimum(base, cap), -v / .5)
    torch.testing.assert_close(zero_reference, torch.maximum(cap, -v / .5))
    positive = r > 0.
    assert torch.equal(reference[positive], zero_reference[positive])
    assert torch.all(reference[~positive] <= zero_reference[~positive] + 1e-12)
    assert torch.all(reference >= -v / .5 - 1e-12)


def test_completed_rows_report_base_law_issued_and_realised():
    class After:
        def __init__(self):
            self.vehicle = self

        def getIDList(self):
            return ["a", "b"]

        def getSpeed(self, vid):
            return 10.6 if vid == "a" else 9.9

    rows = completed_rows(SimpleNamespace(vehicle=After()), {"a": 10., "b": 10.},
                          {"a": 0., "b": -0.2}, {"a": 0.3, "b": -0.5})
    assert rows["a"]["base_law_mps2"] == 0.3 and rows["a"]["requested_residual_mps2"] == 0.
    assert rows["b"]["base_law_mps2"] == pytest.approx(-0.3)
    assert rows["a"]["realised_acceleration_mps2"] == pytest.approx(1.2)
    assert rows["a"]["writer"] == "always_commanded"


def test_mode2_contract_binds_checkpoints_and_dispatches_execute_residual(tmp_path, monkeypatch):
    import hashlib
    from wms.control import execution
    driver = tmp_path / "driver.json"
    payload = {"schema": "execparity_learned_idm/1", "fit_controller": "direct",
               "evaluation_data_used": False, "fit_worlds": ["a"], "score_worlds": ["b"],
               "parameters": {"accel": 1.1, "decel": 2., "time_headway": 1.2,
                              "min_gap": 2., "v0": 28.}}
    driver.write_text(json.dumps(payload))
    monkeypatch.setenv("C4_EXECUTION_DRIVER", str(driver))
    monkeypatch.setenv("C4_EXECUTION_PARITY", "2")
    assert mode() == 2 and execution.enabled()
    payload = contract()
    assert payload["executor"] == BACKEND_CONTRACT
    with pytest.raises(ValueError, match="differs"):
        execution.check_contract({"execution_contract": {**payload, "executor": "other/1"}})
    state = {"v_mps": torch.tensor([[10.]]), "s_m": torch.tensor([[300.]]),
             "lane": torch.tensor([[1]]), "present": torch.ones(1, 1, dtype=torch.bool),
             "length_m": torch.full((1, 1), 5., dtype=torch.float64)}
    out = execution.executed_residual(state, torch.zeros((1, 1), dtype=torch.float64),
                                      torch.full((1, 1), 99.), -3., 1.2)
    assert float(out[0, 0]) != 99.  # the WM head is never the mode-2 base
    monkeypatch.setenv("C4_EXECUTION_PARITY", "1")
    assert contract()["executor"] == "native_relative_same_step/1"


def test_mode2_ownership_execute_residual_commands_zero_residual_cavs(monkeypatch):
    from traffic_shepherd.ownership import CommandOwnership
    monkeypatch.setenv("C4_EXECUTION_PARITY", "2")
    api = SimpleNamespace(vehicle=domain())
    controller = SimpleNamespace(config=SimpleNamespace(control_dt_s=1., sensor_range_m=100.))
    own = CommandOwnership(api, {"accepted"}, controller, cav_speed_mode=27)
    frame = SimpleNamespace(vehicle_id="accepted", timestamp_s=300., speed_mps=8.,
                            allowed_speed_mps=28., front_coverage_m=100., rear_coverage_m=100.,
                            front=None, rear=None)
    rows = own.step({"accepted": frame}, now_s=300., live_ids={"accepted"},
                    wall_commands={"accepted": .3}, defer_wall=True)
    assert rows[0].owner == "wall" and not api.vehicle.calls
    base = float(sumo_native_accel(api, "accepted", 0.5))
    for now in (300., 300.5):
        issued = own.execute_residual({"accepted": 0.0}, -3., 1.2,
                                      now_s=now, live_ids={"accepted"})
        assert issued == {"accepted": base}
    kinds = [c[0] for c in api.vehicle.calls]
    assert kinds == ["setSpeed", "setAcceleration"] * 2
    assert all(c[2] == base for c in api.vehicle.calls[1::2])


def test_mode2_verify_parity_requires_real_ctrl0_identity_receipt(tmp_path, monkeypatch):
    import hashlib
    from wms.control.execution import verify_parity
    driver = tmp_path / "driver.json"
    payload = {"schema": "execparity_learned_idm/1", "fit_controller": "direct",
               "evaluation_data_used": False, "fit_worlds": ["a"], "score_worlds": ["b"],
               "parameters": {"accel": 1.1, "decel": 2., "time_headway": 1.2,
                              "min_gap": 2., "v0": 28.}}
    driver.write_text(json.dumps(payload))
    monkeypatch.setenv("C4_EXECUTION_DRIVER", str(driver))
    monkeypatch.setenv("C4_EXECUTION_PARITY", "2")
    good = {"schema": "execparity_always_commanded_acceleration/1",
            "backend_contract": BACKEND_CONTRACT, "measured_real_sumo": True,
            "zero_ctrl0_identity": True, "passed": True, "worlds": ["977001"],
            "driver_sha256": hashlib.sha256(driver.read_bytes()).hexdigest(),
            "native_arm_head_error": {"rmse": .2},
            "error": {"ours": {"rmse": .17}}, "c1_residual_execution_error": {"rmse": .17}}
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(good))
    assert verify_parity(path)["passed"]
    for mutate in ({**good, "zero_ctrl0_identity": False},
                   {**good, "measured_real_sumo": False},
                   {**good, "schema": "execparity_continuous_acceleration/1"},
                   {**good, "error": {"ours": {"rmse": .25}}}):
        path.write_text(json.dumps(mutate))
        with pytest.raises(ValueError):
            verify_parity(path)
