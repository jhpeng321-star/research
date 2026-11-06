"""V10COURTESY: learned ramp-car / merge-intent inputs and the event loss boost."""
from __future__ import annotations

import json

import pytest
import torch

from wms.config.train import TrainConfig
from wms.model import HdvConfig, ModelSpec, WmSModel
from wms.model.checkpoint import load_checkpoint
from wms.model.residual_net import (COURTESY_F, HdvResidualNet, courtesy_features,
                                    input_schema)
from wms.physics.config import PhysicsConfig
from wms.train.courtesy import courtesy_follower_boost
from wms.train.losses.base import LossContext, UnrollTargets
from wms.train.losses.trajectory import PositionLoss

SMALL = dict(d_model=16, n_heads=2, n_layers=1, ffn=32)


def _kin(rows):
    kin = torch.zeros(1, len(rows), 7)
    for i, (s, lane, v, present) in enumerate(rows):
        kin[0, i, 0] = s
        if present:
            kin[0, i, 1 + lane] = 1.0
        kin[0, i, 4] = v
        kin[0, i, 6] = 5.0
    present = torch.tensor([[row[3] for row in rows]], dtype=torch.bool)
    return kin, present


def test_courtesy_features_are_ramp_only_and_intent_is_detached():
    physics = PhysicsConfig(merge_zone_s1_m=75.44, merge_lane=2)
    kin, present = _kin([
        (55.0, 1, 12.0, True),    # lane-1 follower: no courtesy channels
        (40.0, 2, 4.0, True),     # ramp car, 35.44 m to the end, slow
        (0.0, 0, 0.0, False),     # empty slot
    ])
    merge_prob = torch.tensor([[0.0, 0.8, 0.0]], requires_grad=True)
    age = torch.tensor([[0.0, 10.0, 0.0]])       # 10 steps * 0.5 s = 5 s in lane 2
    feats = courtesy_features(kin, present, physics, lane_age=age, merge_prob=merge_prob)
    assert feats.shape == (1, 3, COURTESY_F)
    assert torch.equal(feats[0, 0], torch.zeros(COURTESY_F))
    assert torch.equal(feats[0, 2], torch.zeros(COURTESY_F))
    ramp, dend, near, speed, slow, wait, intent = feats[0, 1].tolist()
    assert ramp == 1.0 and near == 1.0
    assert dend == pytest.approx(35.44 / 50.0, rel=1e-6)
    assert speed == pytest.approx(4.0 / 20.0)
    assert slow == pytest.approx(0.5)
    assert wait == pytest.approx(0.25)
    assert intent == pytest.approx(0.8)
    assert not feats.requires_grad


def test_zero_initialised_courtesy_adapter_is_an_exact_noop():
    torch.manual_seed(0)
    kin, present = _kin([(55.0, 1, 12.0, True), (40.0, 2, 4.0, True)])
    base = HdvResidualNet(HdvConfig(**SMALL)).eval()
    on = HdvResidualNet(HdvConfig(courtesy_inputs=True, **SMALL)).eval()
    # Every baseline tensor is identical and the new adapters start at zero.
    on.load_state_dict(base.state_dict(), strict=False)
    courtesy = courtesy_features(kin, present, PhysicsConfig(merge_zone_s1_m=75.44),
                                 merge_prob=torch.tensor([[0.0, 0.9]]))
    a = base(kin, present)
    b = on(kin, present, courtesy=courtesy)
    for key in ("accel_delta_mps2", "accel_log_sigma", "lane_logit_delta"):
        assert torch.equal(a[key], b[key]), key
    assert on.courtesy_proj[-1].weight.abs().max().item() == 0.0
    assert on.blocks[0].attn.courtesy_bias.weight.abs().max().item() == 0.0


def test_input_schema_declares_courtesy_without_cav_identity():
    schema = input_schema(HdvConfig(courtesy_inputs=True))
    assert "courtesy" in schema["allowed_inputs"]
    assert schema["courtesy_f"] == COURTESY_F
    assert "is_cav" in schema["forbidden_inputs"]
    assert "cav_flag" in schema["forbidden_inputs"]
    assert input_schema(HdvConfig())["courtesy_f"] == 0


def test_config_keys_default_off_and_round_trip():
    default = TrainConfig()
    assert default.model.courtesy_inputs is False
    assert default.loss.courtesy_weight == 1.0
    flat = default.to_legacy_flat()
    assert "courtesy_inputs" not in flat and "courtesy_weight" not in flat
    cfg = TrainConfig.from_legacy_flat({"courtesy_inputs": True, "courtesy_weight": 3.0})
    assert cfg.model.courtesy_inputs is True and cfg.loss.courtesy_weight == 3.0
    with pytest.raises(ValueError):
        TrainConfig.from_legacy_flat({"courtesy_weight": 0.5})


def test_courtesy_model_loads_an_old_checkpoint_and_stays_identical(tmp_path):
    cfg = TrainConfig.from_legacy_flat({"courtesy_inputs": True})
    base = WmSModel(ModelSpec(), physics=PhysicsConfig())
    path = tmp_path / "old.pt"
    (tmp_path / "physics.json").write_text(json.dumps(base.physics.__dict__), encoding="utf-8")
    # An old checkpoint: config already asks for courtesy (the new run) but the
    # weights predate the adapter, exactly like v5m / iter0b.
    torch.save({"format": "wm_s/1", "model": base.state_dict(),
                "config": cfg.to_legacy_flat()}, path)
    loaded = load_checkpoint(path, strict=True).model
    assert loaded.hdv.courtesy_inputs is True
    kin, present = _kin([(55.0, 1, 12.0, True), (40.0, 2, 4.0, True)])
    state = {"veh_kin": kin, "present": present}
    base.eval()
    loaded.eval()
    with torch.no_grad():
        a = base.forward(state)["accel_mps2"]
        b = loaded.forward(state)["accel_mps2"]
    assert torch.equal(a, b)


def test_enable_courtesy_inputs_keeps_weights_and_outputs():
    base = WmSModel(ModelSpec(), physics=PhysicsConfig()).eval()
    kin, present = _kin([(55.0, 1, 12.0, True), (40.0, 2, 4.0, True)])
    state = {"veh_kin": kin, "present": present}
    with torch.no_grad():
        before = base.forward(state)["accel_mps2"]
    base.set_courtesy_inputs(True)
    with torch.no_grad():
        after = base.forward(state)["accel_mps2"]
    assert torch.equal(before, after)
    assert base.hdv.courtesy_inputs is True


def test_courtesy_follower_boost_marks_only_the_yield_event():
    physics = PhysicsConfig(merge_zone_s1_m=75.44, merge_lane=2)
    # [ramp at 40 m (dend 35.44), lane-1 follower 20 m behind, lane-1 car far ahead]
    kin, present = _kin([(40.0, 2, 10.0, True), (20.0, 1, 15.0, True),
                         (100.0, 1, 20.0, True)])
    states = kin.unsqueeze(1)
    boost = courtesy_follower_boost(states, present.unsqueeze(1), physics, 3.0)
    assert boost.shape == (1, 1, 3)
    assert boost[0, 0].tolist() == [1.0, 3.0, 1.0]
    assert courtesy_follower_boost(states, present.unsqueeze(1), physics, 1.0) is None


def test_position_loss_applies_the_courtesy_boost():
    b, t, k = 1, 1, 2
    targets = UnrollTargets(
        states=torch.zeros(b, t + 1, k, 7), n_steps=t, weights=torch.ones(1, t, 1),
        hdv_mask=torch.ones(b, t, k, dtype=torch.bool),
        aux_mask=torch.ones(b, t, k, dtype=torch.bool), aux_count=2,
        denom=torch.tensor(2.0), lane_target=torch.zeros(b, t, k, dtype=torch.long),
        loss_valid=torch.ones(b, t + 1, k, dtype=torch.bool),
        is_cav=torch.zeros(b, t + 1, k, dtype=torch.bool), dt_s=0.5,
        position_delta_m=5.0, speed_delta_mps=1.0, history_lane_head=False,
        courtesy_boost=torch.tensor([[[1.0, 3.0]]]))
    ctx = LossContext(pred={"positions": torch.ones(b, t, k),
                            "speeds": torch.zeros(b, t, k)},
                      current={}, targets=targets, c0=0, c1=1)
    # Huber(1, 0, delta=5) = 0.5 for each slot; (0.5*1 + 0.5*3) / 2 = 1.0.
    assert float(PositionLoss(1.0)(ctx).logged) == pytest.approx(1.0)
