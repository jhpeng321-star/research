"""Config contract tests: defaults, JSON/YAML load, save, model mapping."""
from __future__ import annotations

import json

import pytest

from train.config import (ConfigError, DEFAULT_SEEDS, MEMBER_ORDER,
                          N_MEMBERS, TrainConfig)


def test_defaults_match_contract():
    cfg = TrainConfig()
    assert cfg.seeds == DEFAULT_SEEDS
    assert cfg.n_members == N_MEMBERS == len(DEFAULT_SEEDS)
    assert MEMBER_ORDER == (1, 2, 3, 4, 5)
    assert cfg.lambda_contrast == 1.0
    assert cfg.calibrate_on_cal is False
    assert cfg.epochs == 15 and cfg.steps_per_epoch == 2000
    assert cfg.device == "cuda" and cfg.out_dir == "runs/wm_n1"


def test_save_load_json_round_trip(tmp_path):
    cfg = TrainConfig(epochs=2, seeds=(1, 2, 3, 4, 5), out_dir="x")
    path = cfg.save(tmp_path / "cfg.json")
    back = TrainConfig.load(path)
    assert back.to_dict() == cfg.to_dict()
    assert back.seeds == (1, 2, 3, 4, 5)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["seeds"] == [1, 2, 3, 4, 5]


def test_load_yaml_subset_keeps_defaults(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("epochs: 3\nbatch_size: 4\n", encoding="utf-8")
    cfg = TrainConfig.load(path)
    assert cfg.epochs == 3 and cfg.batch_size == 4
    assert cfg.steps_per_epoch == 2000        # default kept


def test_unknown_key_refused(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"nope": 1}), encoding="utf-8")
    with pytest.raises(ConfigError):
        TrainConfig.load(path)


def test_model_config_matches_architecture_fields():
    cfg = TrainConfig(d_model=48, n_heads=4, n_layers=1, ffn=96,
                      bias_hidden=8, counter_channels=8)
    model_cfg = cfg.model_config()
    assert model_cfg.d_model == 48 and model_cfg.n_layers == 1
    assert model_cfg.ffn == 96 and model_cfg.counter_channels == 8


def test_member_seeds_are_independent():
    cfg = TrainConfig()
    seeds = [cfg.seed_for_member(m) for m in MEMBER_ORDER]
    assert seeds == list(DEFAULT_SEEDS)
    assert len(set(seeds)) == N_MEMBERS
    with pytest.raises(ConfigError):
        cfg.seed_for_member(6)


def test_loss_weights_keys():
    w = TrainConfig().loss_weights()
    assert set(w) == {"step", "lane", "arrival", "outcome", "collision",
                      "contrast"}
    assert w["contrast"] == 1.0
