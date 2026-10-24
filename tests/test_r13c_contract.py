"""R13C must preserve the calibrated core and refuse an overbroad optimizer."""
import json
from pathlib import Path

import pytest
import torch

from wms.config.load import load_train_config
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.train.round13c_gate import optimizer_errors

ROOT = Path(__file__).resolve().parents[1]


def test_config_and_actual_trainable_parameter_contract():
    cfg = load_train_config(ROOT / "configs/c4_wm_v10round13c.toml")
    assert cfg.model.driver_param_init == "checkpoint"
    assert not cfg.model.residual_accel_reinit
    assert cfg.model.follower_courtesy_init_zero
    assert cfg.curriculum.stage_steps == (40, 60)
    assert cfg.curriculum.unroll_s == (120., 300.)
    assert cfg.selection.val_max_windows == 16
    assert cfg.loss.driver_prior.weight == 0
    assert cfg.loss.follower_courtesy.absolute_weight == 4.525
    assert cfg.optim.lr_scale["residual_net"] == 0.5
    model = WmSModel(ModelSpec.from_mapping(cfg.to_legacy_flat()))
    names = [n for n, _ in model.named_parameters() if not n.startswith(cfg.optim.freeze_prefixes)]
    text = "WM-S optimizer groups: " + json.dumps({"trainable_tensor_names": names})
    assert optimizer_errors(text, names) == names
    with pytest.raises(ValueError, match="core/physics"):
        bad = [*names, "driver_encoder.centre_logits"]
        optimizer_errors("WM-S optimizer groups: " + json.dumps({"trainable_tensor_names": bad}), bad)
    kin = torch.zeros(1, 3, 7)
    kin[..., 0] = torch.tensor([0., 100., 40.])
    kin[..., 2] = 1.
    kin[..., 4] = 10.
    kin[..., 6] = 5.
    model.diag_emit_temporal = True
    with torch.no_grad():
        result = model({"veh_kin": kin, "present": torch.ones(1, 3, dtype=torch.bool)}, k_steps=1)
    assert torch.count_nonzero(result["temporal_accel_pregate_mps2"]) == 0
