"""E5: the thin emulate() API — batching, device-agnosticism, contract."""
import os

import numpy as np
import pytest
import torch

import emulators
import refload

MODEL = refload.MODEL_ONNX


def _frames_dicts(n, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        speed = float(rng.uniform(0, 33))
        has_leader = rng.random() < 0.85
        rows.append(dict(
            now_s=500.0, ts_s=500.0, speed_mps=speed,
            allowed_speed_mps=31.29,
            front_coverage_m=100.0, rear_coverage_m=50.0,
            front=None if not has_leader else dict(
                gap_m=float(rng.uniform(1, 100)),
                relative_speed_mps=float(rng.uniform(-4, 4))),
            rear=None))
    return rows


def test_emulate_contract_shapes_and_codes():
    emu = emulators.ControllerEmulators(MODEL, device=torch.device("cpu"))
    rows = _frames_dicts(8)
    batch = emulators.FrameBatch.stack(rows, device=torch.device("cpu"))
    kinds = torch.tensor([0, 1, 2, 3, 1, 2, 0, 1])
    state = emu.init_state(8)
    out = emu.emulate(kinds, state, batch,
                      wall_speed_mps=torch.full((8,), 20.0))
    assert out["target_speed_mps"].shape == (8,)
    assert out["reason_code"].shape == (8,)
    assert out["raw_command_mps"].shape == (8,)
    assert out["new_state"].pi.raw_prev.shape == (8,)
    # native slots: no command, selected_native
    for slot in (0, 6):
        assert math_isnan(float(out["target_speed_mps"][slot]))
        assert int(out["reason_code"][slot]) == \
            emulators.REASON_CODES["selected_native"]
    # wall slots: bounded command speed
    for slot in (3,):
        target = float(out["target_speed_mps"][slot])
        assert not math_isnan(target)
        lo = rows[slot]["speed_mps"] - 3.0
        assert target >= max(0.0, lo) - 1e-9


def math_isnan(value):
    return value != value


def _same(a, b):
    return (a == b) or (a != a and b != b)  # nan-aware equality


def test_emulate_batching_invariance():
    emu = emulators.ControllerEmulators(MODEL, device=torch.device("cpu"))
    rows = _frames_dicts(6, seed=3)
    kinds = torch.tensor([1, 2, 1, 2, 0, 1])
    batch = emulators.FrameBatch.stack(rows, device=torch.device("cpu"))
    # first tick from identical zero states: batched == per-row singles
    out_first = emu.emulate(kinds, emu.init_state(6), batch)
    for slot in range(6):
        one = emu.emulate(kinds[slot:slot + 1], emu.init_state(1),
                          emulators.FrameBatch.stack([rows[slot]],
                                                     device=torch.device("cpu")))
        assert _same(float(out_first["target_speed_mps"][slot]),
                     float(one["target_speed_mps"][0]))
        assert int(out_first["reason_code"][slot]) == \
            int(one["reason_code"][0])
    # second tick from the joint state: still per-slot independent
    state2 = out_first["new_state"]
    out_second = emu.emulate(kinds, state2, batch)
    for slot in range(6):
        one_state = emulators.ControllerState(
            pi=emulators.PIState(
                raw_prev=state2.pi.raw_prev[slot:slot + 1].clone(),
                last_ts=state2.pi.last_ts[slot:slot + 1].clone(),
                active=state2.pi.active[slot:slot + 1].clone()))
        one = emu.emulate(kinds[slot:slot + 1], one_state,
                          emulators.FrameBatch.stack([rows[slot]],
                                                     device=torch.device("cpu")))
        assert _same(float(out_second["target_speed_mps"][slot]),
                     float(one["target_speed_mps"][0]))
        assert int(out_second["reason_code"][slot]) == \
            int(one["reason_code"][0])


def test_device_contract():
    os.environ["WD_DEVICE"] = "cpu"
    emu = emulators.ControllerEmulators(MODEL)  # default_device from env
    assert emu.device.type == "cpu"
    del os.environ["WD_DEVICE"]
    emu2 = emulators.ControllerEmulators(MODEL)
    assert emu2.device.type == "cpu"
    with pytest.raises(ValueError):
        os.environ["WD_DEVICE"] = "tpu"
        emulators.default_device()
    del os.environ["WD_DEVICE"]


def test_cuda_requested_but_unavailable_is_loud():
    if torch.cuda.is_available():
        pytest.skip("CUDA present; loud-failure path not applicable")
    os.environ["WD_DEVICE"] = "cuda"
    try:
        with pytest.raises(RuntimeError):
            emulators.default_device()
    finally:
        del os.environ["WD_DEVICE"]


def test_unregistered_u_refused():
    with pytest.raises(ValueError):
        emulators.PIEmulator(u_mps=30.0)
