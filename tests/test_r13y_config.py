"""R13Y: default-off learnable mainline merge-yield config and legacy round trips."""
import pytest

from wms.config.train import ModelConfig, TrainConfig
from wms.core.errors import ConfigError
from wms.model.spec import ModelSpec


def test_default_is_false_and_omitted_from_legacy_flat():
    cfg = TrainConfig()
    assert ModelConfig().mainline_merge_yield_learnable is False
    assert cfg.model.mainline_merge_yield_learnable is False
    assert ModelSpec().mainline_merge_yield_learnable is False
    flat = cfg.to_legacy_flat()
    assert "mainline_merge_yield_learnable" not in flat
    assert TrainConfig.from_legacy_flat(flat) == cfg
    assert TrainConfig.from_legacy_flat({}) == cfg
    assert TrainConfig.from_dict({}) == cfg
    assert ModelSpec.from_mapping(flat).mainline_merge_yield_learnable is False


@pytest.mark.parametrize("enabled", [False, True])
def test_nested_and_legacy_round_trip_preserve_switch(enabled):
    cfg = TrainConfig.from_dict({"model": {"mainline_merge_yield_learnable": enabled}})
    nested = cfg.to_dict()
    assert nested["model"]["mainline_merge_yield_learnable"] is enabled
    assert TrainConfig.from_dict(nested) == cfg

    flat = cfg.to_legacy_flat()
    if enabled:
        assert flat["mainline_merge_yield_learnable"] is True
    else:
        assert "mainline_merge_yield_learnable" not in flat
    assert TrainConfig.from_legacy_flat(flat) == cfg
    assert TrainConfig.from_legacy_flat({"mainline_merge_yield_learnable": enabled}) == cfg
    assert ModelSpec.from_mapping(flat).mainline_merge_yield_learnable is enabled


@pytest.mark.parametrize("value", [0, 1, 1.0, "false", "true", None, []])
def test_non_boolean_switch_is_rejected(value):
    with pytest.raises(ConfigError, match="mainline_merge_yield_learnable.*boolean"):
        ModelConfig(mainline_merge_yield_learnable=value)
    with pytest.raises(ConfigError, match="mainline_merge_yield_learnable.*bool"):
        TrainConfig.from_dict({"model": {"mainline_merge_yield_learnable": value}})
    with pytest.raises(ConfigError, match="mainline_merge_yield_learnable.*bool"):
        TrainConfig.from_legacy_flat({"mainline_merge_yield_learnable": value})
