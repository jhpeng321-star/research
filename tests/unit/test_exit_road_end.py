"""Plain-bug regression: every rollout removes vehicles at the road end.

The model has always gated removal on ``state["exit_s_m"]``; the evaluation predictor,
the Dreamer imagination and the WM training unrolls did not supply it, so vehicles stayed
present beyond 553.33 m.  These tests pin the shared constant, the predictor initial state
(and its removal effect), the training inputs and the ``physics.exit_at_road_end`` escape.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from wms.eval.backend import WmsBackend, default_tools
from wms.eval.predictor import WmsPredictor
from wms.model.residual_net import HdvConfig
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.physics.batched import MODEL_FRAME_LANE_LENGTHS, MODEL_FRAME_ROAD_END_M
from wms.physics.config import PhysicsConfig
from wms.train.inputs import prepare_inputs
from wms.train.settings import TrainSettings

ROAD_END_M = 553.33


def small_model(*, exit_at_road_end: bool = True) -> WmSModel:
    return WmSModel(ModelSpec(history_hidden=16, k_hidden=8),
                    hdv=HdvConfig(d_model=16, n_heads=2, n_layers=1, ffn=32),
                    physics=PhysicsConfig(exit_at_road_end=exit_at_road_end))


def record(s_m: float = ROAD_END_M - 1.33, v_mps: float = 20.0) -> dict:
    """One HDV just before the road end; no recorded future arm (no SUMO presence mask)."""
    vehicle = {"id": "v1", "s": s_m, "v": v_mps, "lane": 0, "present": True,
               "is_cav": False}
    return {"world": "w", "key": "k", "t0_s": 0.0, "source_arm": "x", "paired": False,
            "initial_state": {"time_s": 0.0, "vehicles": [vehicle]}, "recorded": {}}


def training_batch() -> dict:
    states = torch.zeros(1, 3, 1, 7)
    states[..., 0] = ROAD_END_M - 1.33
    states[..., 1] = 1.0                       # lane 0 one-hot
    states[..., 4] = 20.0
    states[..., 6] = 5.0
    return {"states": states, "present": torch.ones(1, 3, 1, dtype=torch.bool),
            "is_cav": torch.zeros(1, 3, 1, dtype=torch.bool)}


def predict(model: WmSModel, horizon_s: float = 2.0) -> list[dict]:
    backend = WmsBackend(model=model, config={}, tools=default_tools())
    return WmsPredictor(backend).predict_states(record(), {"arm": "native"}, horizon_s, 0)


def test_road_end_is_derived_from_the_lane_contract_once():
    assert MODEL_FRAME_LANE_LENGTHS[0] == pytest.approx(ROAD_END_M)
    assert MODEL_FRAME_ROAD_END_M == MODEL_FRAME_LANE_LENGTHS[0]


def test_predictor_supplies_exit_bound_and_drops_the_vehicle():
    model = small_model()
    frames = predict(model)
    assert frames[0]["vehicles"][0]["s"] == pytest.approx(ROAD_END_M - 1.33)
    assert all(not frame["vehicles"] for frame in frames[1:])


def test_escape_flag_restores_old_predictor_behaviour():
    model = small_model()
    model.physics = dataclasses.replace(model.physics, exit_at_road_end=False)
    frames = predict(model)
    assert frames[-1]["vehicles"]
    assert frames[-1]["vehicles"][0]["s"] > ROAD_END_M


def test_training_inputs_supply_exit_bound_and_drop_the_vehicle():
    model = small_model()
    batch = training_batch()
    inputs = prepare_inputs(batch, model, TrainSettings(), torch.device("cpu"), 2)
    torch.testing.assert_close(inputs.initial["exit_s_m"],
                               torch.full_like(inputs.initial["s_m"], ROAD_END_M))
    roll = model.rollout(inputs.initial, n_steps=1,
                         present_seq=torch.ones(1, 2, 1, dtype=torch.bool))
    assert not bool(roll["masks"][0, 0, 0])

    model.physics = dataclasses.replace(model.physics, exit_at_road_end=False)
    old = prepare_inputs(batch, model, TrainSettings(), torch.device("cpu"), 2)
    assert "exit_s_m" not in old.initial
    kept = model.rollout(old.initial, n_steps=1,
                         present_seq=torch.ones(1, 2, 1, dtype=torch.bool))
    assert bool(kept["masks"][0, 0, 0])


def test_recorded_sumo_exit_still_ends_the_slot_without_the_road_end():
    """The two masks AND together: a recorded exit removes a car before the road end."""
    model = small_model()
    batch = training_batch()
    batch["states"][..., 0] = 400.0
    batch["states"][..., 4] = 10.0
    inputs = prepare_inputs(batch, model, TrainSettings(), torch.device("cpu"), 2)
    present_seq = torch.ones(1, 2, 1, dtype=torch.bool)
    present_seq[:, 1] = False
    roll = model.rollout(inputs.initial, n_steps=1, present_seq=present_seq)
    assert not bool(roll["masks"][0, 0, 0])
