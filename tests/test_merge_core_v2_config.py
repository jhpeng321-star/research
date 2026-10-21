"""R8 merge-core v2 config plumbing (agent C).

* the five new fields reach ``loss.merge_core`` -> flat -> ``TrainSettings`` -> composer;
* every existing config keeps its exact legacy flat form (round 7b hash) and new defaults;
* the round-8 freeze prefixes keep ``merge_core.temperature_raw`` bitwise unchanged after an
  optimizer step and the margin residual is exactly 0 at the fitted zero-effect init;
* ``lr_scale`` longest-prefix resolution gives 100x for s0/T/b/tau and 20x for alpha/sigma.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path

import pytest
import torch

from wms.config.load import load_train_config
from wms.config.train import TrainConfig
from wms.model.merge_core import MergeCoreNet
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.physics.config import PhysicsConfig
from wms.train.settings import TrainSettings
from wms.train.trainer import build_optimizer

REPO = Path(__file__).resolve().parents[1]
ROUND8 = REPO / "configs" / "c4_wm_v10round8.toml"
ROUND7B = REPO / "configs" / "c4_wm_v10round7b.toml"

NEW_FLAT = ("merge_core_version", "merge_core_z_tf", "merge_core_z_on",
            "merge_core_sigma_reg_weight", "merge_core_sigma_fit")
#: SHA-256 of ``json.dumps(round7b.to_legacy_flat(), sort_keys=True, default=str)``
#: captured before this change: the defaults must keep the legacy flat form bit-identical.
ROUND7B_FLAT_SHA256 = "db0d9376e013e22febbdd44ebd75516fc9585384c0f48192d0897b9e300ef011"


def _r8_settings() -> TrainSettings:
    return TrainSettings.from_config(load_train_config(ROUND8))


def _legacy_flat_hash(path: Path) -> str:
    flat = load_train_config(path).to_legacy_flat()
    blob = json.dumps(flat, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def test_round8_config_reaches_flat_and_settings():
    cfg = load_train_config(ROUND8)
    core = cfg.loss.merge_core
    assert core.version == 2
    assert core.weight == pytest.approx(0.4)
    assert core.tf_weight == pytest.approx(0.6)
    assert core.yield_weight == pytest.approx(0.4)
    assert core.n_quad == 8 and core.metrics is True
    assert (core.pos_tol_m, core.speed_tol_mps) == (pytest.approx(50.0), pytest.approx(8.0))
    assert (core.gap_tol_m, core.gap_speed_tol_mps) == (pytest.approx(40.0),
                                                       pytest.approx(8.0))
    assert core.location_reg_weight == pytest.approx(1.0e-3)
    assert core.sigma_reg_weight == pytest.approx(1.0e-3)
    assert core.z_tf == pytest.approx(1.0) and core.z_on == pytest.approx(1.0)
    assert core.sigma_fit == pytest.approx(0.0)      # 0 -> fitted alpha_log_sigma

    flat = cfg.to_legacy_flat()
    assert flat["merge_core_version"] == 2
    assert flat["merge_core_sigma_reg_weight"] == pytest.approx(1.0e-3)
    # z_tf / z_on / sigma_fit equal their defaults here, so the legacy flat form omits them.
    assert "merge_core_z_tf" not in flat and "merge_core_z_on" not in flat
    assert "merge_core_sigma_fit" not in flat
    overridden = load_train_config(
        ROUND8, ["loss.merge_core.z_tf=2.0", "loss.merge_core.z_on=3.0",
                 "loss.merge_core.sigma_fit=0.125"])
    flat = overridden.to_legacy_flat()
    assert flat["merge_core_z_tf"] == pytest.approx(2.0)
    assert flat["merge_core_z_on"] == pytest.approx(3.0)
    assert flat["merge_core_sigma_fit"] == pytest.approx(0.125)

    settings = TrainSettings.from_config(cfg)
    assert settings.merge_core_version == 2
    assert settings.merge_core_z_tf == pytest.approx(1.0)
    assert settings.merge_core_z_on == pytest.approx(1.0)
    assert settings.merge_core_sigma_reg_weight == pytest.approx(1.0e-3)
    assert settings.merge_core_sigma_fit == pytest.approx(0.0)
    assert settings.merge_core_on
    assert settings.freeze_prefixes == (
        "merge_core.temperature_raw", "merge_core.residual_head",
        "merge_core.residual_scale_raw")
    assert settings.lr_scale["merge_core"] == pytest.approx(20.0)
    for name in ("s0_raw", "t_raw", "b_raw", "tau_raw"):
        assert settings.lr_scale[f"merge_core.{name}"] == pytest.approx(100.0)
    for name in ("alpha_bias", "alpha_log_sigma_raw", "yield_scale_raw"):
        assert settings.lr_scale[f"merge_core.{name}"] == pytest.approx(20.0)


def test_existing_configs_keep_defaults_and_round7b_flat_is_identical():
    assert _legacy_flat_hash(ROUND7B) == ROUND7B_FLAT_SHA256
    flat = load_train_config(ROUND7B).to_legacy_flat()
    assert not set(NEW_FLAT) & set(flat)
    core = load_train_config(ROUND7B).loss.merge_core
    assert (core.version, core.z_tf, core.z_on) == (1, 1.0, 1.0)
    assert (core.sigma_reg_weight, core.sigma_fit) == (0.0, 0.0)
    settings = TrainSettings.from_config(load_train_config(ROUND7B))
    assert settings.merge_core_version == 1
    assert settings.merge_core_z_tf == 1.0 and settings.merge_core_z_on == 1.0
    assert settings.merge_core_sigma_reg_weight == 0.0
    assert settings.merge_core_sigma_fit == 0.0
    # A default TrainConfig keeps the same legacy flat form as well (new fields omitted).
    assert not set(NEW_FLAT) & set(TrainConfig().to_legacy_flat())
    defaults = TrainSettings()
    assert (defaults.merge_core_version, defaults.merge_core_z_tf, defaults.merge_core_z_on,
            defaults.merge_core_sigma_reg_weight, defaults.merge_core_sigma_fit) == (
                1, 1.0, 1.0, 0.0, 0.0)


def test_validation_rejects_bad_round8_fields():
    with pytest.raises(ValueError):
        TrainConfig.from_dict({"loss": {"merge_core": {"version": 3}}})
    with pytest.raises(ValueError):
        TrainConfig.from_dict({"loss": {"merge_core": {"z_tf": 0.0}}})
    with pytest.raises(ValueError):
        TrainConfig.from_dict({"loss": {"merge_core": {"z_on": float("inf")}}})
    with pytest.raises(ValueError):
        TrainSettings(merge_core_version=0)
    with pytest.raises(ValueError):
        TrainSettings(merge_core_z_tf=0.0)
    with pytest.raises(ValueError):
        TrainSettings(merge_core_sigma_reg_weight=-1.0)


_HAS_R8_KWARGS = {"version", "z_tf", "z_on", "sigma_reg_weight", "sigma_fit"} <= {
    f.name for f in fields(__import__("wms.train.losses.merge_core",
                                      fromlist=["MergeCoreConfig"]).MergeCoreConfig)}


@pytest.mark.xfail(not _HAS_R8_KWARGS, reason=(
    "merge_core.py on the C branch predates agent A's R8 kwargs; the coordinator merge "
    "supplies them"), strict=False)
def test_compose_passes_the_r8_fields():
    from wms.train.losses.compose import build_loss
    composer = build_loss(_r8_settings(), physics=None)
    term = next(t for t in composer.chunk_terms if t.name == "merge_core")
    cfg = term.cfg
    assert cfg.version == 2
    assert cfg.z_tf == pytest.approx(1.0) and cfg.z_on == pytest.approx(1.0)
    assert cfg.sigma_reg_weight == pytest.approx(1.0e-3)
    assert cfg.sigma_fit == pytest.approx(0.0)


class _CoreModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.merge_core = MergeCoreNet(input_dim=8, hidden=8, residual_hidden=8)


class _FullModel(torch.nn.Module):
    """The real secure-gap model (all round-8 lr_scale prefixes exist)."""

    def __init__(self) -> None:
        super().__init__()
        spec = ModelSpec.from_mapping(load_train_config(ROUND8).to_legacy_flat())
        self.model = WmSModel(spec, physics=PhysicsConfig())

    def named_parameters(self, *args, **kwargs):
        # Expose the inner model's names unchanged (no "model." prefix).
        return self.model.named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self.model.parameters(*args, **kwargs)


def _residual(core: MergeCoreNet) -> torch.Tensor:
    zero = torch.zeros(2, 3)
    ten = torch.full((2, 3), 10.0)
    return core.margin_residual(ten, ten, ten, ten, ten, zero, zero, None)


def test_freeze_keeps_temperature_bitwise_and_residual_exactly_zero():
    settings = _r8_settings()
    model = _CoreModel()
    core = model.merge_core
    residual_before = _residual(core)
    assert torch.equal(residual_before, torch.zeros_like(residual_before))
    temperature_before = core.temperature_raw.detach().clone()
    scale_before = core.residual_scale_raw.detach().clone()
    s0_before = core.s0_raw.detach().clone()

    opt, report = build_optimizer(model, lr=1.0e-2, weight_decay=0.0,
                                  freeze_prefixes=settings.freeze_prefixes,
                                  lr_scale=settings.lr_scale)
    assert report["unmatched_freeze_prefixes"] == []
    assert not core.temperature_raw.requires_grad
    assert not any(p.requires_grad for p in core.residual_head.parameters())
    assert not core.residual_scale_raw.requires_grad

    loss = sum(p.sum() for p in model.parameters() if p.requires_grad)
    loss.backward()
    opt.step()

    assert torch.equal(core.temperature_raw.detach(), temperature_before)
    assert torch.equal(core.residual_scale_raw.detach(), scale_before)
    assert torch.equal(core.s0_raw.detach(), s0_before) is False   # the step did move it
    residual_after = _residual(core)
    assert torch.equal(residual_after, torch.zeros_like(residual_after))


def test_lr_scale_longest_prefix_on_the_real_model():
    settings = _r8_settings()
    model = _FullModel()
    _, report = build_optimizer(model, lr=1.0, weight_decay=0.0,
                                freeze_prefixes=settings.freeze_prefixes,
                                lr_scale=settings.lr_scale)
    assert report["unmatched_freeze_prefixes"] == []
    assert report["unmatched_lr_scale_prefixes"] == []
    factors = {g["prefix"]: g["lr_scale"] for g in report["groups"]}
    for name in ("s0_raw", "t_raw", "b_raw", "tau_raw"):
        assert factors[f"merge_core.{name}"] == pytest.approx(100.0)
    for name in ("alpha_bias", "alpha_log_sigma_raw", "yield_scale_raw"):
        assert factors[f"merge_core.{name}"] == pytest.approx(20.0)
    assert factors["merge_core"] == pytest.approx(20.0)
    # Frozen temperature/residual tensors are not in any optimizer group.
    frozen = {"merge_core.temperature_raw", "merge_core.residual_scale_raw"}
    frozen |= {f"merge_core.residual_head.{i}.weight" for i in (0, 2)}
    frozen |= {f"merge_core.residual_head.{i}.bias" for i in (0, 2)}
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert not frozen & trainable
