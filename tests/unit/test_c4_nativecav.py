"""V10NATIVECAV: learned native-CAV twin switch, weights and feature parity.

The default remains the legacy IDM (bit-identical imagination); ``native_cav="head"``
selects the fitted 9-32-32-1 head committed in ``configs/native_cav_head_v5.json``.
"""
from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from wms.control.counterfactual import CounterfactualConfig, idm_native_accel, native_fn
from wms.control.native_cav import (ACCEL_MIN_MPS2, HEAD_ACCEL_MAX_MPS2, HEAD_PATH,
                                    NativeCavHead, feature_matrix, head_features,
                                    head_native_accel, load_head)

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _state(k: int = 5, close_m: float = 10.0):
    s = torch.tensor([[0.0, close_m, 30.0, 60.0, 90.0]])[:, :k]
    v = torch.full((1, k), 10.0)
    return {
        "s_m": s,
        "v_mps": v,
        "a_mps2": torch.zeros(1, k),
        "lane": torch.zeros(1, k, dtype=torch.long),
        "present": torch.ones(1, k, dtype=torch.bool),
        "is_cav": torch.tensor([[False, True, True, True, True]])[:, :k],
        "length_m": torch.full((1, k), 5.0),
    }


def test_default_switch_is_the_legacy_idm():
    cfg = CounterfactualConfig()
    assert cfg.native_cav == "idm"
    assert cfg.native_cav_head_path is None
    state = _state()
    legacy = native_fn("idm", a_max=1.2)(state)
    assert torch.equal(legacy, idm_native_accel(state, a_max=1.2))
    assert asdict(cfg)["native_cav"] == "idm"


def test_v10nativecav_config_selects_the_head():
    from wms.control.counterfactual import CounterfactualConfig
    from wms.control.train_cli import _read_config

    raw = _read_config(REPO / "configs" / "c4_policy_v10nativecav_v5m.toml")
    cfg = CounterfactualConfig(**raw["counterfactual"])
    assert cfg.native_cav == "head"
    assert cfg.native_accel_max_mps2 >= HEAD_ACCEL_MAX_MPS2
    assert raw["trainer"]["entrance_zone_native"] is True


def test_head_switch_and_validation():
    cfg = CounterfactualConfig(native_cav="head")
    assert cfg.native_cav == "head"
    assert asdict(cfg)["native_cav_head_path"] is None
    with pytest.raises(ValueError, match="native_cav"):
        CounterfactualConfig(native_cav="learned")
    with pytest.raises(ValueError, match="native_cav_head_path"):
        CounterfactualConfig(native_cav="head", native_cav_head_path=3)


def test_committed_head_loads_with_the_expected_contract():
    assert HEAD_PATH.is_file()
    head = load_head(REPO / "configs" / "native_cav_head_v5.json")
    assert isinstance(head, NativeCavHead)
    assert head.accel_min == ACCEL_MIN_MPS2
    assert head.accel_max == HEAD_ACCEL_MAX_MPS2
    assert [tuple(w.shape) for w, _ in head.layers] == [(32, 9), (32, 32), (1, 32)]


def test_head_output_is_bounded_and_gap_sensitive():
    head = load_head()
    state = _state()
    out = head_native_accel(state, head)
    assert out.shape == (1, 5)
    assert torch.isfinite(out).all()
    assert float(out.min()) >= ACCEL_MIN_MPS2 - 1e-6
    assert float(out.max()) <= HEAD_ACCEL_MAX_MPS2 + 1e-6
    # Absent slots are zero; closer leaders never command more acceleration here.
    absent = {**state, "present": torch.tensor([[False, True, True, True, True]])}
    assert float(head_native_accel(absent, head)[0, 0]) == 0.0
    close = head_native_accel({**state, "s_m": torch.tensor([[0.0, 8.0, 30.0, 60.0, 90.0]])}, head)
    far = head_native_accel({**state, "s_m": torch.tensor([[0.0, 80.0, 90.0, 120.0, 150.0]])}, head)
    assert float(close[0, 0]) < float(far[0, 0])


def test_head_switch_differs_from_legacy_and_keeps_gradients():
    head = load_head()
    state = _state()
    switched = native_fn("head")(state)
    legacy = native_fn("idm", a_max=1.2)(state)
    assert not torch.equal(switched, legacy)
    s = state["s_m"].clone().requires_grad_(True)
    v = state["v_mps"].clone().requires_grad_(True)
    out = head_native_accel({**state, "s_m": s, "v_mps": v}, head)
    out.sum().backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()


def test_bridge_uses_the_head_for_non_acting_cavs():
    from wms.control.bridge import make_cav_controller

    state = _state()
    state["physics_step_index"] = 0
    native = native_fn("head")
    def policy(obs, signal, **kwargs):
        return torch.zeros(1, 5), torch.zeros(1, 5)

    def encoder(obs, positions, field=None):
        return torch.zeros(1, 5, 8)

    def safety(st, accel, **kwargs):
        return accel
    controller = make_cav_controller(policy, encoder, safety, act_mask=torch.zeros(1, 5, dtype=torch.bool),
                                     native=native, accel_max=1.2, native_accel_max=1.2)
    with torch.no_grad():
        applied = controller(state)[0, :, 3]
    expected = head_native_accel(state) * state["is_cav"]
    assert torch.allclose(applied, expected, atol=1e-6)


def test_numpy_training_features_match_the_torch_inference_features():
    state = _state(k=5, close_m=17.0)
    state["lane"] = torch.tensor([[0, 1, 2, 0, 1]])
    state["present"] = torch.tensor([[True, True, True, True, False]])
    torch_features = head_features(state).numpy()
    numpy_features = feature_matrix(
        state["s_m"].numpy(), state["v_mps"].numpy(), state["a_mps2"].numpy(),
        state["lane"].numpy(), state["present"].numpy(), state["length_m"].numpy())
    assert torch_features.shape == numpy_features.shape == (1, 5, 9)
    assert np.allclose(torch_features, numpy_features, atol=1e-6)


def test_fit_tool_refuses_restricted_paths_and_splits_are_disjoint():
    from tools.native_cav_fit import guard, world_bucket

    with pytest.raises(SystemExit, match="restricted"):
        guard(Path("E:/research-dense/holdout/v5_native"))
    buckets = [world_bucket(f"w96300{i}m1900p40t300") for i in range(20)]
    assert all(0 <= b < 10 for b in buckets)
    assert {world_bucket(f"w{i}") for i in range(40)} == set(range(10))
