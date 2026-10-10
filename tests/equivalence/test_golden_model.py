"""New model vs the golden artifacts of branch wms/golden (skipped when absent).

Reads the tracked ``tests/golden/data`` of this work tree; the merge-calibration parameters
come from the data registry (``merge_params`` -> ``configs/merge_params.json``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from wms.data.registry import resolve
from wms.model import ModelSpec, build_model, load_checkpoint
from wms.physics import PhysicsConfig

HERE = Path(__file__).resolve()
CANDIDATES = (HERE.parents[2] / "tests" / "golden" / "data",)
DATA = next((p for p in CANDIDATES if (p / "forward_c1.npz").is_file()), None)
MERGE_PARAMS = resolve("merge_params")

pytestmark = pytest.mark.skipif(DATA is None, reason="golden data not available")
ATOL = 1e-6
N_STEPS = {"u10": 20, "u60": 120}


def _batch(name: str, j: int) -> dict[str, torch.Tensor]:
    arrays = np.load(DATA / f"batches_{name}.npz")
    prefix = f"b{j}_"
    return {k[len(prefix):]: torch.from_numpy(arrays[k]) for k in arrays.files
            if k.startswith(prefix)}


def _state_from_kin(kin: torch.Tensor, present: torch.Tensor) -> dict[str, torch.Tensor]:
    lane = kin[..., 1:4].argmax(-1)
    lane = torch.where(kin[..., 1:4].sum(-1) > 0, lane, torch.full_like(lane, -1))
    return {"veh_kin": kin, "s_m": kin[..., 0], "v_mps": kin[..., 4], "a_mps2": kin[..., 5],
            "length_m": kin[..., 6], "lane": lane, "present": present}


def _rollout(model, batch: dict, n_steps: int) -> dict[str, np.ndarray]:
    states, present, is_cav = batch["states"].float(), batch["present"].bool(), \
        batch["is_cav"].bool()
    initial = _state_from_kin(states[:, 0], present[:, 0])
    initial["is_cav"] = is_cav[:, 0]
    hist, hist_mask = batch.get("hist"), batch.get("hist_mask")
    with torch.no_grad():
        roll = model.rollout(
            initial, n_steps=n_steps, k0=0,
            cav_teacher=batch["cav_teacher"].float()[:, :n_steps + 1],
            present_seq=present[:, :n_steps + 1], is_cav_seq=is_cav[:, :n_steps + 1],
            controller_id=batch.get("controller_id"), history=hist, history_mask=hist_mask,
            driver_history=batch.get("driver_hist", hist),
            driver_history_mask=batch.get("driver_hist_mask", hist_mask))
        gate = model.gate(torch.arange(n_steps))
    out = {"accel_mps2": roll["accel"], "raw_residual_mps2": roll["raw_residual_mps2"],
           "gate": gate, "lane_logits": roll["lane_logits"], "positions": roll["positions"],
           "speeds": roll["speeds"], "lanes": roll["lanes"], "masks": roll["masks"],
           "lane_change_probabilities": roll["lane_change_probabilities"]}
    return {k: v.detach().cpu().numpy() for k, v in out.items()}


def _compare(arrays: dict[str, np.ndarray], reference: str, prefix: str) -> None:
    ref = np.load(DATA / reference)
    for key, value in arrays.items():
        expected = ref[f"{prefix}_{key}"]
        assert value.shape == expected.shape, key
        if value.dtype.kind == "f":
            assert np.allclose(value, expected, atol=ATOL, rtol=0, equal_nan=True), \
                f"{prefix}_{key}: {np.nanmax(np.abs(value - expected))}"
        else:
            assert np.array_equal(value, expected), f"{prefix}_{key}"


@pytest.mark.skipif(DATA is None or not (DATA / "c1" / "wm_s.pt").is_file(),
                    reason="golden C1 checkpoint missing")
def test_forward_c1_matches_golden():
    receipt = json.loads((DATA / "c1" / "TRAIN_RECEIPT.json").read_text(encoding="utf-8"))
    loaded = load_checkpoint(DATA / "c1" / "wm_s.pt", strict=True, physics=receipt["physics"])
    for j in range(3):
        _compare(_rollout(loaded.model, _batch("u10_histoff", j), N_STEPS["u10"]),
                 "forward_c1.npz", f"u10_b{j}")
    _compare(_rollout(loaded.model, _batch("u60_histoff", 0), N_STEPS["u60"]),
             "forward_c1.npz", "u60_b0")


@pytest.mark.skipif(DATA is None or not MERGE_PARAMS.is_file(),
                    reason="R3 merge params missing")
def test_forward_r3_matches_golden_from_init_state():
    physics = PhysicsConfig.from_params_file(MERGE_PARAMS, sumo_fidelity=True, sumo_idm=True,
                                             merge_cal=True)
    model = build_model(ModelSpec(history_s=60.0, history_stride_s=1.0,
                                  residual_soft_bound_mps2=4.0), physics)
    init = np.load(DATA / "r3_init_state.npz")
    state = model.state_dict()
    assert {k.replace("/", ".") for k in init.files} == set(state)
    model.load_state_dict({k.replace("/", "."): torch.from_numpy(init[k]) for k in init.files},
                          strict=True)
    model.eval()
    for j in range(3):
        _compare(_rollout(model, _batch("u10_histon", j), N_STEPS["u10"]),
                 "forward_r3.npz", f"u10_b{j}")
    _compare(_rollout(model, _batch("u60_histon", 0), N_STEPS["u60"]),
             "forward_r3.npz", "u60_b0")


@pytest.fixture(autouse=True)
def _golden_determinism():
    """Same execution settings as the capture (2 threads, deterministic algorithms)."""
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)
    torch.set_num_threads(threads)
