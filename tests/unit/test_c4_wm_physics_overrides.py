"""V9NOYIELD: WM ``mainline_merge_yield`` physics override inside Dreamer imagination (CPU only).

Three guarantees: (1) an empty mapping leaves every PhysicsConfig holder bit-identical;
(2) a set override reaches EVERY holder found by the model walk; (3) the shipped policy config
loads, resolves and carries exactly the one switch.  No GPU, SUMO or checkpoint is touched.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from wms.control.contract import RewardWeights
from wms.control.dreamer import (DreamerConfig, DreamerTrainer, _physics_holders,
                                 apply_wm_physics_overrides)
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.physics.config import PhysicsConfig


def _model(*, mainline=True):
    return WmSModel(ModelSpec(), physics=PhysicsConfig(mainline_merge_yield=mainline))


def test_empty_override_is_bit_identical():
    model = _model(mainline=True)
    base = model.physics
    holders = _physics_holders(model)
    assert len(holders) >= 1                      # at least the model's own physics
    result = apply_wm_physics_overrides(model, {})
    assert result == {"overrides": {}, "holders": len(holders)}
    assert model.physics is base                  # same object, nothing replaced
    assert model.physics.mainline_merge_yield is True
    for owner, name in _physics_holders(model):
        assert getattr(owner, name) is base


def test_override_reaches_every_holder():
    model = _model(mainline=True)
    holders = _physics_holders(model)
    result = apply_wm_physics_overrides(model, {"mainline_merge_yield": False})
    assert result["overrides"] == {"mainline_merge_yield": False}
    assert result["holders"] == len(holders)
    fresh = _physics_holders(model)
    assert fresh                                  # every reference still resolves
    assert all(getattr(owner, name).mainline_merge_yield is False for owner, name in fresh)
    # the swap shares one PhysicsConfig instance across the holders
    assert len({id(getattr(owner, name)) for owner, name in fresh}) == 1


def test_unknown_override_field_rejected():
    with pytest.raises(ValueError):
        apply_wm_physics_overrides(_model(), {"no_such_switch": True})


def test_default_config_has_no_override():
    assert DreamerConfig().wm_physics_overrides == {}
    with pytest.raises(ValueError):
        DreamerConfig(wm_physics_overrides=[("mainline_merge_yield", False)])


class _HolderWorldModel(nn.Module):
    """Minimal WM stub with a real PhysicsConfig at the top level and in a child module."""

    def __init__(self, physics):
        super().__init__()
        self.gain = nn.Parameter(torch.zeros(1))
        self.physics = physics
        self.child = nn.Module()
        self.child.physics = physics


def test_trainer_applies_override_at_init():
    from tests.unit.test_c4_dreamer import (FakeCritic, FakeGroupEncoder, FakePolicy, FakeSampler,
                                            observations, safety, speed_reward)
    model = _HolderWorldModel(PhysicsConfig(mainline_merge_yield=True))
    assert len(_physics_holders(model)) == 2
    cfg = DreamerConfig(batch_size=2, horizon=4, actor_lr=1e-3, critic_lr=1e-3,
                        entropy_weight=0.0, uncertainty_weight=0.0, ttc_budget=0.1,
                        checkpoint_every=0, validation_every=0,
                        wm_physics_overrides={"mainline_merge_yield": False})
    trainer = DreamerTrainer([model], FakePolicy(), FakeGroupEncoder(), FakeCritic(), FakeSampler(),
                             observation_builder=observations, reward_fn=speed_reward, safety=safety,
                             config=cfg, reward_weights=RewardWeights(), device="cpu",
                             output_dir=None)
    assert trainer.wm_physics_overrides[0]["overrides"] == {"mainline_merge_yield": False}
    fresh = _physics_holders(trainer.world_models[0])
    assert len(fresh) == 2
    assert all(getattr(owner, name).mainline_merge_yield is False for owner, name in fresh)


def test_policy_config_loads_and_resolves():
    from wms.control.train_cli import _read_config
    root = Path(__file__).resolve().parents[2]
    raw = _read_config(root / "configs" / "c4_policy_v9noyield_v5m.toml")
    trainer_section = raw["trainer"]
    assert trainer_section["wm_physics_overrides"] == {"mainline_merge_yield": False}
    assert trainer_section["entry_queue_state"] is True
    cfg = DreamerConfig(**trainer_section)
    assert cfg.wm_physics_overrides == {"mainline_merge_yield": False}
    assert raw["data"]["wm_config"] == "configs/c4_wm_v5m_refresh_q.toml"
