from dataclasses import replace

import pytest
import torch

from tools.rbint.rebuild import rebuild_with_motion_encoder
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.physics.config import PhysicsConfig


def source_model():
    source = WmSModel(ModelSpec(rollout_mode="hybrid", driver_param_mode="physical",
                               follower_courtesy=True)).eval()
    # Distinctive non-initial values make an accidental reinitialization visible.
    with torch.no_grad():
        for i, parameter in enumerate(source.driver_encoder.parameters()):
            parameter.fill_((i + 1) / 17.)
    return source


def test_preserves_encoder_and_source_when_removing_correction_head():
    source = source_model()
    before = {key: value.clone() for key, value in source.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    spec = replace(source.spec, rollout_mode="driver_params", follower_courtesy=False)
    candidate, receipt = rebuild_with_motion_encoder(source, spec, PhysicsConfig())
    assert torch.equal(rng, torch.random.get_rng_state())
    assert receipt["encoder_exact"]
    assert any(key.startswith("follower_courtesy.") for key in receipt["unused_source_tensors"])
    for key, value in before.items():
        assert torch.equal(value, source.state_dict()[key])
        if key.startswith("driver_encoder."):
            assert torch.equal(value, candidate.state_dict()[key])


@pytest.mark.parametrize("override", [{"driver_param_mode": "legacy"}, {"history_s": 60.},
                                      {"history_features": 7}, {"rollout_mode": "residual"}])
def test_refuses_encoder_semantic_changes(override):
    source = source_model()
    with pytest.raises(ValueError):
        rebuild_with_motion_encoder(source, replace(source.spec, **override), PhysicsConfig())
