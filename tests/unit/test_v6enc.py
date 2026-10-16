"""V6 opt-in physical driver settings; synthetic CPU prediction-error checks only."""
from __future__ import annotations

from dataclasses import asdict

import pytest
import torch
from torch import nn

from wms.cli.assemble import build_physics
from wms.config.load import load_train_config
from wms.config.train import ModelConfig, TrainConfig
from wms.core.errors import ConfigError
from wms.model.checkpoint import load_checkpoint, save_checkpoint
from wms.model.driver_encoder import (
    PARAMETER_LOWER, PARAMETER_NAMES, PARAMETER_UPPER, DriverEncoder,
)
from wms.model.factory import build_model
from wms.model.spec import ModelSpec
from wms.physics.config import PhysicsConfig
from wms.train.losses.compose import build_loss
from wms.train.settings import TrainSettings
from wms.train.unroll import UnrollRunner

from .test_wmlane import _batch


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _encoder_inputs(n=10, physics=None):
    p = physics or PhysicsConfig(sumo_fidelity=True)
    a, b, t, s0, _ = p.idm()
    defaults = torch.tensor((p.v0_mps, t, s0, a, b, 0.2, 0.1, 6.0, 1.0))
    g = torch.Generator().manual_seed(17)
    history = torch.randn(1, 2, n, 6, generator=g)
    mask = torch.ones(1, 2, n, dtype=torch.bool)
    interaction = torch.randn(1, 2, 4, generator=g)
    return history, mask, interaction, defaults.expand(1, 2, -1)


def _physical(physics=None):
    return DriverEncoder(driver_param_mode="physical",
                         physics=physics or PhysicsConfig(sumo_fidelity=True))


def test_default_encoder_is_bit_identical_to_legacy_arithmetic():
    """Check the fixed-batch legacy map independently of the new physical map."""
    torch.manual_seed(0)
    default = DriverEncoder()
    torch.manual_seed(0)
    explicit = DriverEncoder(driver_param_mode="legacy")
    assert default.state_dict().keys() == explicit.state_dict().keys()
    assert not hasattr(default, "centre_logits")
    assert all(torch.equal(v, explicit.state_dict()[k]) for k, v in default.state_dict().items())
    inputs = _encoder_inputs()
    inputs[1][0, 0, :3] = False
    out, other = default(*inputs), explicit(*inputs)
    assert all(torch.equal(out[k], other[k]) for k in out)
    raw = default.head(out["embedding"]).reshape(1, 2, 2, 9)
    count = inputs[1].sum(-1, keepdim=True).float()
    # Match the original operation grouping exactly (the delta is formed first).
    legacy = inputs[3] + count / (count + 8) * (torch.tanh(raw[..., 0, :]) * default.bounds)
    legacy = torch.maximum(torch.minimum(legacy, legacy.new_tensor(PARAMETER_UPPER)),
                           legacy.new_tensor(PARAMETER_LOWER))
    assert torch.equal(out["values"], legacy)
    out["values"].sum().backward()
    other["values"].sum().backward()
    for left, right in zip(default.parameters(), explicit.parameters()):
        assert torch.equal(left.grad, right.grad)


def test_physical_ranges_represent_textbook_values_with_nonzero_jacobian():
    enc = _physical()
    # Desired speeds are ratios of the road limit, not the legacy v0 default.
    scale = enc.physical_lower.new_tensor((31.29, 1, 1, 1, 1))
    assert torch.allclose(enc.physical_lower / scale, torch.tensor((0.6, 0.4, 0.5, 0.3, 0.8)))
    assert torch.allclose(enc.physical_upper / scale, torch.tensor((1.4, 3.0, 5.0, 4.0, 6.0)))
    span = enc.physical_upper - enc.physical_lower
    # Grid probes the whole interior; textbook a/b/T and high a are explicit cases.
    targets = [enc.physical_lower + f * span for f in (0.01, 0.1, 0.5, 0.9, 0.99)]
    targets += [torch.tensor((31.29, 1.0, 2.0, a, 1.5)) for a in (1.0, 3.5)]
    for n in (1, 10, 61):
        shrink = torch.tensor(n / (n + 8))
        for target in targets:
            raw = ((torch.logit((target - enc.physical_lower) / span)
                    - enc.centre_logits.detach()) / shrink).requires_grad_(True)
            got = enc._physical_values(raw, shrink)
            assert torch.allclose(got, target, atol=2e-6)
            jac = torch.autograd.functional.jacobian(lambda x: enc._physical_values(x, shrink),
                                                      raw)
            assert torch.all(torch.diag(jac) > 1e-4)
            assert torch.isfinite(jac).all()
    # Reach the textbook targets through the actual head, not just the map helper.
    history, mask, interaction, defaults = _encoder_inputs()
    target = torch.tensor((31.29, 1.0, 2.0, 1.0, 1.5))
    with torch.no_grad():
        enc.head[-1].weight.zero_()
        enc.head[-1].bias[:5].copy_(
            (torch.logit((target - enc.physical_lower) / span) - enc.centre_logits) / (10 / 18))
    values = enc(history, mask, interaction, defaults)["values"]
    assert torch.allclose(values[..., :5], target.expand(1, 2, -1), atol=2e-6)
    values[..., :5].sum().backward()
    assert (enc.head[-1].bias.grad[:5].abs() > 0).all()


@pytest.mark.parametrize("fidelity", [False, True])
def test_learned_centre_initialization_shrinkage_and_lane_identity(fidelity):
    physics = PhysicsConfig(sumo_fidelity=fidelity, speed_limit_mps=25.0)
    torch.manual_seed(2)
    enc = _physical(physics)
    torch.manual_seed(2)
    legacy = DriverEncoder()
    inputs = _encoder_inputs(physics=physics)
    inputs[1].zero_()
    assert isinstance(enc.centre_logits, nn.Parameter) and enc.centre_logits.requires_grad
    out = enc(*inputs)
    assert torch.allclose(out["values"][..., :5], inputs[3][..., :5], atol=2e-6)
    before = out["values"].detach().clone()
    with torch.no_grad():
        enc.centre_logits.add_(0.2)
    after = enc(*inputs)["values"]
    assert (after[..., :5] > before[..., :5]).all()
    after[..., :5].sum().backward()
    assert (enc.centre_logits.grad.abs() > 0).all()
    assert torch.count_nonzero(enc.head[-1].bias.grad[:5]) == 0
    # Low observation counts pull logits closer to the learned centre.
    inputs[1][:, 0, :1] = True
    inputs[1][:, 1, :] = True
    with torch.no_grad():
        enc.head[-1].weight.zero_()
        enc.head[-1].bias[:5].fill_(1.0)
    result = enc(*inputs)
    centre = enc._physical_values(torch.zeros(5), torch.tensor(0.0))
    assert torch.all((result["values"][0, 0, :5] - centre).abs()
                     < (result["values"][0, 1, :5] - centre).abs())
    assert torch.equal(result["shrinkage"], torch.tensor([[1 / 9, 10 / 18]]))
    # Copy the same trunk/head so lane outputs can be compared bit for bit.
    legacy.load_state_dict({k: v for k, v in enc.state_dict().items()
                            if k in legacy.state_dict()}, strict=True)
    old = legacy(*inputs)
    assert torch.equal(result["values"][..., 5:], old["values"][..., 5:])
    for key in ("embedding", "attention", "uncertainty"):
        assert torch.equal(result[key], old[key])


def test_initial_centre_is_clipped_inside_physical_range():
    enc = _physical(PhysicsConfig(speed_factor=3.0))
    centre = enc._physical_values(torch.zeros(5), torch.tensor(0.0))
    assert torch.isfinite(enc.centre_logits).all()
    assert torch.all(centre > enc.physical_lower) and torch.all(centre < enc.physical_upper)
    assert centre[0].item() == pytest.approx(enc.physical_upper[0].item(), rel=2e-6)


def test_ema_off_keeps_v0_and_headway_but_allows_embedding_refinement():
    models = []
    for enabled in (True, False):
        torch.manual_seed(0)
        models.append(build_model(ModelSpec(rollout_mode="hybrid", driver_update="ema",
                                             driver_ema_update=enabled)))
    kin = torch.zeros(1, 2, 7)
    kin[..., 0] = torch.tensor([0.0, 30.0])
    kin[..., 2], kin[..., 4], kin[..., 6] = 1.0, 10.0, 5.0
    state = {"veh_kin": kin, "present": torch.ones(1, 2, dtype=torch.bool)}
    defaults = models[0]._default_driver_parameters(kin, 1, 2)
    params = {name: defaults[..., i] for i, name in enumerate(PARAMETER_NAMES)}
    observed = state["present"]
    embedding = torch.zeros(1, 2, models[0].driver_embed_dim)
    on, _ = models[0].update_drivers(params, embedding, state, observed)
    assert not torch.equal(on["v0"], params["v0"])
    assert not torch.equal(on["time_headway"], params["time_headway"])
    with torch.no_grad():
        models[1].driver_updater.embed_proj.bias.fill_(0.2)
    off, refined = models[1].update_drivers(params, embedding, state, observed)
    assert all(off[name] is params[name] for name in PARAMETER_NAMES)
    assert not torch.equal(refined, embedding)
    off_no_embed, no_embed = models[1].update_drivers(params, None, state, observed)
    assert no_embed is None and all(off_no_embed[n] is params[n] for n in PARAMETER_NAMES)


def test_config_defaults_roundtrip_and_v6_recipe():
    for cfg in (TrainConfig(), load_train_config("configs/c4_wm_v5.toml")):
        flat = cfg.to_legacy_flat()
        assert "driver_param_mode" not in flat and "driver_ema_update" not in flat
        settings, spec = TrainSettings.from_flat(flat), ModelSpec.from_mapping(flat)
        assert settings.driver_param_mode == spec.driver_param_mode == "legacy"
        assert settings.driver_ema_update is spec.driver_ema_update is True
    cfg = load_train_config("configs/c4_wm_v6.toml")
    cfg.validate_for_training()
    flat = cfg.to_legacy_flat()
    assert flat["driver_param_mode"] == "physical" and flat["driver_ema_update"] is False
    assert TrainConfig.from_legacy_flat(flat).to_legacy_flat() == flat
    assert TrainSettings.from_flat(flat).driver_param_mode == "physical"
    assert not TrainSettings.from_flat(flat).driver_ema_update
    assert cfg.loss.driver_prior.weight == 0 and cfg.runtime.out_dir == "runs/active/c4_wm_v6"


@pytest.mark.parametrize("factory", [ModelSpec, ModelConfig, TrainSettings])
@pytest.mark.parametrize("kw", [{"driver_param_mode": "wide"}, {"driver_ema_update": "false"},
                                {"driver_ema_update": 0}, {"driver_ema_update": None}])
def test_invalid_switches(factory, kw):
    with pytest.raises((ValueError, ConfigError)):
        factory(**kw)


@pytest.mark.parametrize("mode,ema", [("legacy", True), ("physical", False), ("physical", True)])
def test_checkpoint_roundtrip(tmp_path, mode, ema):
    spec = ModelSpec(rollout_mode="hybrid", driver_update="ema",
                     driver_param_mode=mode, driver_ema_update=ema)
    physics = PhysicsConfig(sumo_fidelity=True, speed_limit_mps=25.0)
    torch.manual_seed(13)
    model = build_model(spec, physics).eval()
    if mode == "physical":
        with torch.no_grad():
            model.driver_encoder.centre_logits.add_(0.3)
    path = tmp_path / "wm_s.pt"
    # These new keys must be saved even when the caller supplies an old-style config.
    save_checkpoint(path, model, {"rollout_mode": "hybrid", "driver_update": "ema",
                                  "physics": asdict(physics)})
    loaded = load_checkpoint(path)
    assert loaded.model.spec.driver_param_mode == mode
    assert loaded.model.driver_updater.driver_ema_update == ema
    if mode == "legacy":
        assert "driver_param_mode" not in loaded.legacy_config
        assert "driver_ema_update" not in loaded.legacy_config
    else:
        assert loaded.legacy_config["driver_param_mode"] == mode
    state, restored = model.state_dict(), loaded.model.state_dict()
    assert state.keys() == restored.keys()
    assert all(torch.equal(state[k], restored[k]) for k in state)
    inputs = _encoder_inputs(physics=physics)
    before, after = model.driver_encoder(*inputs), loaded.model.driver_encoder(*inputs)
    assert all(torch.equal(before[k], after[k]) for k in before)


def _synthetic_run(overrides=()):
    cfg = load_train_config("configs/c4_wm_v6.toml", ["runtime.device=cpu", "runtime.threads=1",
                                                     *overrides])
    flat = cfg.to_legacy_flat()
    torch.manual_seed(5)
    model = build_model(ModelSpec.from_mapping(flat), build_physics(cfg))
    settings = TrainSettings.from_flat(flat)
    runner = UnrollRunner(model, settings, build_loss(settings, physics=model.physics),
                          projection=None, device=torch.device("cpu"))
    record = runner.run(_batch(), unroll_steps=2, backward=True, truncation_steps=1)
    return model, record


def test_v6_recipe_synthetic_two_step_forward_backward():
    model, record = _synthetic_run()
    assert torch.isfinite(torch.tensor(record["loss"]))
    assert "driver_prior" not in record
    grad = model.driver_encoder.centre_logits.grad
    assert grad is not None and torch.isfinite(grad).all() and (grad.abs() > 0).all()
    head_grad = model.driver_encoder.head[-1].weight.grad[:5]
    assert torch.isfinite(head_grad).all() and head_grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_explicit_off_fixed_batch_values_and_gradients_are_bit_identical():
    base = load_train_config("configs/c4_wm_v5.toml",
                             ["runtime.device=cpu", "runtime.threads=1"])
    explicit = load_train_config("configs/c4_wm_v5.toml", ["runtime.device=cpu", "runtime.threads=1",
                                 'model.driver_param_mode="legacy"', "model.driver_ema_update=true"])
    results = []
    for cfg in (base, explicit):
        flat = cfg.to_legacy_flat()
        torch.manual_seed(11)
        model = build_model(ModelSpec.from_mapping(flat), build_physics(cfg))
        settings = TrainSettings.from_flat(flat)
        runner = UnrollRunner(model, settings, build_loss(settings, physics=model.physics),
                              projection=None, device=torch.device("cpu"))
        rec = runner.run(_batch(), unroll_steps=2, backward=True, truncation_steps=1)
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        results.append((rec, grads, model.state_dict()))
    assert results[0][0] == results[1][0]
    for i in (1, 2):
        assert results[0][i].keys() == results[1][i].keys()
        assert all(torch.equal(value, results[1][i][name]) for name, value in results[0][i].items())
