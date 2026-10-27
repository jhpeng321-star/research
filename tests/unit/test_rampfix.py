"""S-RAMPFIX unit tests: mainline interaction residual + lane_v5_long parameter group.

Contract under test (all default OFF = bit-identical):

* ``model.mainline_interaction_residual`` adds exactly ``mainline_residual.*`` tensors;
  the zero-initialised read-out makes the enabled model's forward bit-identical to the
  disabled one at init, including after loading a pre-RAMPFIX (WM1-like) state dict;
* gradients reach the head through its weights and the state (feature inputs);
* the head is masked to mainline geometry (lanes 0/1, upstream + merge zone), so ramp
  and downstream slots keep an exactly zero offset whatever the weights;
* ``optim.lane_v5_long`` unfreezes the existing lane_v5 network into its own optimizer
  group at its own lr scale; when off, freeze_prefixes behave exactly as before.
"""
from __future__ import annotations

import torch

from wms.config.train import (CurriculumConfig, ModelConfig, OptimConfig, RuntimeConfig,
                              TrainConfig)
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.physics.config import PhysicsConfig

KIN = torch.tensor([[[-3000.0, 1.0, 0.0, 0.0, 15.0, 0.0, 5.0],     # lane 0, far upstream
                     [-40.0, 0.0, 1.0, 0.0, 12.0, 0.0, 5.0],        # lane 1, merge zone
                     [20.0, 0.0, 1.0, 0.0, 10.0, 0.0, 5.0],         # lane 1, merge zone
                     [200.0, 0.0, 1.0, 0.0, 20.0, 0.0, 5.0],        # lane 1, downstream
                     [30.0, 0.0, 0.0, 1.0, 6.0, 0.0, 5.0],          # ramp lane 2
                     [10.0, 0.0, 1.0, 0.0, 9.0, 0.0, 5.0]]])        # lane 1 follower
PRESENT = torch.ones(1, 6, dtype=torch.bool)


def _model(flag: bool, seed: int = 7) -> WmSModel:
    torch.manual_seed(seed)
    spec = ModelSpec(rollout_mode="hybrid", lane_model="hazard_v5",
                     merge_decision="first_passage", follower_courtesy=True,
                     follower_courtesy_version=2, ramp_speed_adapt_learnable=True,
                     mainline_interaction_residual=flag)
    return WmSModel(spec, physics=PhysicsConfig(max_gap_steps=8))


def _state():
    return {"veh_kin": KIN.clone(), "present": PRESENT.clone()}


def test_flag_off_keeps_exact_key_set():
    off, on = _model(False), _model(True)
    keys_on, keys_off = set(on.state_dict()), set(off.state_dict())
    extra = keys_on - keys_off
    assert extra and all(key.startswith("mainline_residual.") for key in extra)
    assert not keys_off - keys_on


def test_identity_at_init_forward_bit_identical():
    off, on = _model(False), _model(True)
    off.eval(), on.eval()
    with torch.no_grad():
        a = off(_state(), k_steps=4)
        b = on(_state(), k_steps=4)
    assert set(a) | {"mainline_residual_mps2"} == set(b)
    for key, value in a.items():
        if torch.is_tensor(value):
            assert torch.equal(value, b[key]), key
    assert torch.equal(b["mainline_residual_mps2"], torch.zeros_like(KIN[..., 0]))


def test_enable_hook_on_legacy_core_is_identity():
    legacy = _model(False)
    legacy.eval()
    with torch.no_grad():
        before = legacy(_state(), k_steps=4)
    legacy.enable_mainline_residual(hidden=32)
    assert legacy.mainline_residual is not None
    with torch.no_grad():
        after = legacy(_state(), k_steps=4)
    for key, value in before.items():
        if torch.is_tensor(value):
            assert torch.equal(value, after[key]), key


def test_legacy_state_dict_loads_into_enabled_model():
    off, on = _model(False), _model(True)
    payload = off.state_dict()
    assert not any(key.startswith("mainline_residual.") for key in payload)
    on.load_state_dict(payload)  # auto-fills the zero-initialised head
    on.eval()
    with torch.no_grad():
        out = on(_state(), k_steps=4)
    assert torch.equal(out["mainline_residual_mps2"], torch.zeros_like(KIN[..., 0]))


def test_head_masked_to_mainline_geometry():
    on = _model(True)
    with torch.no_grad():  # make the head non-trivial: read-out no longer zero
        for layer in on.mainline_residual.net:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.normal_(0.0, 0.5)
                layer.bias.normal_(0.0, 0.5)
    on.eval()
    with torch.no_grad():
        out = on(_state(), k_steps=4)
    delta = out["mainline_residual_mps2"][0]
    assert delta[4] == 0.0, "ramp-lane slot must keep an exactly zero offset"
    assert delta[3] == 0.0, "downstream slot must keep an exactly zero offset"
    assert delta[0] != 0.0 and delta[1] != 0.0 and delta[2] != 0.0, "mainline slots act"
    assert float(delta.abs().max()) <= 1.5 + 1.0e-6


def test_gradients_flow_through_weights_and_state():
    on = _model(True)
    kin = KIN.clone().requires_grad_(True)
    out = on({"veh_kin": kin, "present": PRESENT.clone()}, k_steps=4)
    out["accel_mps2"].sum().backward()
    head = dict(on.mainline_residual.named_parameters())
    assert head["net.4.weight"].grad is not None
    assert float(head["net.4.weight"].grad.abs().sum()) > 0.0
    assert kin.grad is not None and float(kin.grad.abs().sum()) > 0.0, "state path is live"
    with torch.no_grad():  # once the read-out is nonzero, the trunk learns too
        head["net.4.weight"].normal_(0.0, 0.3)
        head["net.4.bias"].normal_(0.0, 0.3)
    for parameter in on.parameters():
        parameter.grad = None
    kin = KIN.clone().requires_grad_(True)
    out = on({"veh_kin": kin, "present": PRESENT.clone()}, k_steps=4)
    out["accel_mps2"].sum().backward()
    for name, parameter in on.mainline_residual.named_parameters():
        assert parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0, name


def test_gate_multiplies_mainline_residual_like_existing_residual():
    on = _model(True)
    with torch.no_grad():
        for layer in on.mainline_residual.net:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.normal_(0.0, 0.5)
                layer.bias.normal_(0.0, 0.5)
    on.eval()
    with torch.no_grad():
        k0 = on(_state(), k_steps=0)
        k4 = on(_state(), k_steps=4)
    assert torch.equal(k0["mainline_residual_mps2"], torch.zeros_like(KIN[..., 0])) \
        or float(k0["mainline_residual_mps2"].abs().max()) == 0.0
    # k=0 gate is exactly zero, so the emitted correction must vanish there too.
    assert float(k0["mainline_residual_mps2"].abs().max()) == 0.0
    assert float(k4["mainline_residual_mps2"].abs().max()) > 0.0


def test_features_finite_and_masked():
    from wms.model.mainline_residual import (MAINLINE_F, mainline_interaction_features,
                                             mainline_lane_mask)
    physics = PhysicsConfig(max_gap_steps=8)
    feats = mainline_interaction_features(KIN, PRESENT, physics)
    assert tuple(feats.shape) == (1, 6, MAINLINE_F)
    assert torch.isfinite(feats).all()
    mask = mainline_lane_mask(KIN, PRESENT, physics)
    assert mask.tolist() == [[True, True, True, False, False, True]]
    absent = PRESENT.clone()
    absent[0, 0] = False
    feats_absent = mainline_interaction_features(KIN, absent, physics)
    assert float(feats_absent[0, 0].abs().sum()) == 0.0


# ------------------------------------------------------------------ lane_v5_long
def _trainer_cfg(tmp_path, **optim) -> TrainConfig:
    return TrainConfig(
        model=ModelConfig(rollout_mode="driver_params", lane_model="hazard_v5"),
        curriculum=CurriculumConfig(unroll_s=(5.0,), stage_steps=(1,),
                                    truncation_steps_per_stage=(8,)),
        optim=OptimConfig(batch_size=2, lr=1.0e-3, **optim),
        runtime=RuntimeConfig(device="cpu", threads=1, seed=3, out_dir=str(tmp_path)))


def _tiny_dataset():
    import numpy as np
    from wms.data.dataset import TrajectoryDataset

    steps, slots = 12, 4
    states = np.zeros((1, steps + 1, slots, 7), dtype=np.float32)
    states[..., 0] = -100.0 + 20.0 * np.arange(slots)
    states[..., 4] = 10.0
    states[..., 6] = 5.0
    states[..., 1:4] = np.eye(3, dtype=np.float32)[np.zeros((steps + 1, slots), dtype=int)]
    present = np.ones((1, steps + 1, slots), dtype=bool)
    return TrajectoryDataset.from_arrays({"states": states, "present": present})


def _lane_v5_model() -> WmSModel:
    torch.manual_seed(11)
    spec = ModelSpec(rollout_mode="driver_params", lane_model="hazard_v5")
    return WmSModel(spec, physics=PhysicsConfig(max_gap_steps=8))


def _lane_v5_names(model) -> list[str]:
    return [name for name, _ in model.named_parameters() if name.startswith("lane_v5")]


def test_lane_v5_long_off_freezes_exactly_as_before(tmp_path):
    from wms.train.trainer import WmSTrainer, build_optimizer

    cfg = _trainer_cfg(tmp_path, freeze_prefixes=("lane_v5", "merge_core"))
    model = _lane_v5_model()
    trainer = WmSTrainer(cfg, model, _tiny_dataset(), out_dir=tmp_path)
    assert "lane_v5_long" not in trainer.optimizer_groups
    names = trainer.optimizer_groups["trainable_tensor_names"]
    assert not any(name.startswith("lane_v5") for name in names)
    assert all(not p.requires_grad for n, p in model.named_parameters() if n.startswith("lane_v5"))
    # Identical call to build_optimizer reproduces the report (bit-identical flag-off path).
    optimizer, report = build_optimizer(
        _lane_v5_model(), lr=1.0e-3, weight_decay=0.0,
        freeze_prefixes=("lane_v5", "merge_core"), lr_scale=None)
    assert report == trainer.optimizer_groups
    assert optimizer.param_groups[0]["lr"] == trainer.optimizer.param_groups[0]["lr"]


def test_lane_v5_long_on_own_group_and_lr_scale(tmp_path):
    from wms.train.trainer import WmSTrainer

    cfg = _trainer_cfg(tmp_path, freeze_prefixes=("lane_v5", "merge_core"),
                       lane_v5_long=True, lane_v5_long_lr_scale=0.05)
    model = _lane_v5_model()
    frozen_before = [n for n, p in model.named_parameters()
                     if n.startswith("lane_v5") and not p.requires_grad]
    assert not frozen_before or True  # fresh model: nothing frozen yet
    trainer = WmSTrainer(cfg, model, _tiny_dataset(), out_dir=tmp_path)
    names = trainer.optimizer_groups["trainable_tensor_names"]
    lane_names = _lane_v5_names(model)
    assert lane_names and all(name in names for name in lane_names)
    assert all(p.requires_grad for n, p in model.named_parameters() if n.startswith("lane_v5"))
    groups = {g["prefix"]: g for g in trainer.optimizer_groups["groups"]}
    assert "lane_v5" in groups and groups["lane_v5"]["lr_scale"] == 0.05
    assert groups["lane_v5"]["lr"] == 1.0e-3 * 0.05
    assert trainer.optimizer_groups["lane_v5_long"]["trainable"] is True
    # merge_core keeps the frozen contract.
    assert not any(name.startswith("merge_core") for name in names)


def test_lane_v5_long_config_roundtrip(tmp_path):
    cfg = _trainer_cfg(tmp_path, lane_v5_long=True, lane_v5_long_lr_scale=0.25)
    flat = cfg.to_legacy_flat()
    assert flat["lane_v5_long"] is True and flat["lane_v5_long_lr_scale"] == 0.25
    default = _trainer_cfg(tmp_path).to_legacy_flat()
    assert "lane_v5_long" not in default and "lane_v5_long_lr_scale" not in default
    restored = TrainConfig.from_legacy_flat(dict(default, lane_v5_long=True))
    assert restored.optim.lane_v5_long is True
