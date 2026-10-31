from dataclasses import asdict

import numpy as np
import pytest
import torch

from tools.rblane.encoder import FrozenMotionEncoder
from wms.data.history import long_history_features
from wms.model.driver_encoder import PARAMETER_NAMES
from wms.model.factory import build_model
from wms.model.spec import ModelSpec
from wms.physics.config import PhysicsConfig


def source(tmp_path):
    torch.manual_seed(901)
    physics = PhysicsConfig(sumo_fidelity=True)
    spec = ModelSpec(history_s=60., history_stride_s=1., driver_param_mode='physical',
                     rollout_mode='hybrid')
    model = build_model(spec, physics).eval()
    path = tmp_path / 'wm_s.pt'
    torch.save(dict(format='wm_s/1', config=asdict(spec), model=model.state_dict()), path)
    return model, physics, path


def inputs():
    kin = np.zeros((123, 3, 7), np.float32)
    kin[:, :, 0] = np.arange(123)[:, None] * 5 + np.array([0, 40, 60])
    kin[:, :, 2] = 1
    kin[:, :, 4], kin[:, :, 6] = 10, 5
    mask = np.ones((123, 3), np.uint8)
    mask[:10, 2] = 0
    ids = np.tile([11, 12, 13], (123, 1)).astype(np.int32)
    times = np.arange(123) * .5 + 240
    return kin, mask, ids, times


def test_frozen_extractor_is_exactly_the_source_model_encoder(tmp_path):
    model, physics, path = source(tmp_path)
    frozen = FrozenMotionEncoder(path, physics)
    kin, mask, ids, times = inputs()
    settings = frozen.prepare(kin, mask, ids, times, 300.)
    k = torch.from_numpy(kin[:121])
    m = torch.from_numpy(mask[:121]).bool()
    continuity = m[:-1] & m[1:]
    hist, hm = long_history_features(k, m, continuity, m[-1], history_s=60., history_stride_s=1.)
    state = dict(s_m=k[-1:, :, 0], v_mps=k[-1:, :, 4], a_mps2=k[-1:, :, 5],
                 lane=k[-1:, :, 1:4].argmax(-1), length_m=k[-1:, :, 6], present=m[-1:])
    with torch.no_grad():
        _, _, reference = model.encode_drivers(state, driver_history=hist[None], driver_history_mask=hm[None])
    got = settings.settings(torch.from_numpy(ids[120:121]), torch.zeros(1, 3).bool())
    for name in PARAMETER_NAMES:
        torch.testing.assert_close(got[name], reference[name].double(), rtol=0, atol=0)
    # Alter every future value: neither inference nor the persistent settings changes.
    kin[121:] = 1e5
    again = frozen.prepare(kin, mask, ids, times, 300.)
    for name in PARAMETER_NAMES:
        torch.testing.assert_close(again.settings(torch.from_numpy(ids[120:121]),
            torch.zeros(1, 3).bool())[name], got[name], rtol=0, atol=0)


def test_unknown_identity_uses_encoder_centre_and_cav_uses_original_defaults(tmp_path):
    _, physics, path = source(tmp_path)
    frozen = FrozenMotionEncoder(path, physics)
    settings = frozen.prepare(*inputs(), 300.)
    got = settings.settings(torch.tensor([[999, 11]]), torch.tensor([[False, True]]))
    for i, name in enumerate(PARAMETER_NAMES):
        assert got[name][0, 0] == frozen.population[i]
        assert got[name][0, 1] == frozen.defaults[i]


def test_missing_source_weight_and_wrong_checkpoint_identity_are_rejected(tmp_path):
    _, physics, path = source(tmp_path)
    with pytest.raises(ValueError, match='identity'):
        FrozenMotionEncoder(path, physics, required_sha='0' * 64)
    payload = torch.load(path, weights_only=True)
    payload['model'].pop('driver_encoder.head.2.bias')
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match='Missing key'):
        FrozenMotionEncoder(path, physics)
