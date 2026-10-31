"""Real frozen-weight CPU parity; synthetic observables, no traffic data read."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .encoder import FrozenMotionEncoder, REGISTERED_WM2_SHA
from .operate import RECEIPTS
from wms.data.history import long_history_features
from wms.model.driver_encoder import PARAMETER_NAMES
from wms.model.factory import build_model
from wms.model.spec import ModelSpec
from wms.physics.config import PhysicsConfig


def main():
    out = RECEIPTS / 'ENCODER_PARITY.json'
    assert out.parent.is_dir() and not out.exists()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source = json.loads((RECEIPTS / 'ENCODER_SOURCE.json').read_text())
    path = Path(source['local_path'])
    physics = PhysicsConfig(sumo_fidelity=True, open_road=True)
    frozen = FrozenMotionEncoder(path, physics, required_sha=REGISTERED_WM2_SHA)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    # The original learned-yield architecture requires its lane_learnable flag
    # at construction. No lane or acceleration forward is executed here.
    model = build_model(ModelSpec.from_mapping(payload['config']),
                        replace(physics, lane_learnable=True)).eval()
    model.load_state_dict(payload['model'], strict=True)
    kin = np.zeros((121, 4, 7), np.float32)
    kin[:, :, 0] = np.arange(121)[:, None] * np.array([8., 7., 10., 5.]) + [-1000., -960., -900., -1000.]
    kin[:, :3, 2], kin[:, 3, 1] = 1, 1
    kin[:, :, 4], kin[:, :, 6] = [16., 14., 20., 10.], 5.
    mask = np.ones((121, 4), np.uint8)
    mask[:70, 2] = 0
    ids = np.tile([11, 12, 13, 14], (121, 1))
    times = np.arange(121) * .5 + 240
    inferred = frozen.prepare(kin, mask, ids, times, 300.)
    k, m = torch.from_numpy(kin), torch.from_numpy(mask).bool()
    hist, hm = long_history_features(k, m, m[:-1] & m[1:], m[-1],
                                     history_s=60., history_stride_s=1.)
    state = dict(s_m=k[-1:, :, 0], v_mps=k[-1:, :, 4], a_mps2=k[-1:, :, 5],
        lane=k[-1:, :, 1:4].argmax(-1), length_m=k[-1:, :, 6], present=m[-1:])
    with torch.no_grad():
        _, _, expected = model.encode_drivers(state, driver_history=hist[None], driver_history_mask=hm[None])
    actual = inferred.settings(torch.from_numpy(ids[-1:]), torch.zeros(1, 4).bool())
    errors = {}
    for name in PARAMETER_NAMES:
        torch.testing.assert_close(actual[name], expected[name].double(), rtol=0, atol=0)
        errors[name] = float((actual[name] - expected[name].double()).abs().max())
    report = dict(status='PASS', checkpoint_sha256=frozen.sha256,
        source_full_model_strict_load=True, encoder_strict_load=True,
        full_model_construction_lane_learnable=True,
        parameters_bit_exact=errors, source_model_weight_keys=len(payload['model']),
        max_observation_time_s=300, history_s=60, no_training=True, no_wm_rollout=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
