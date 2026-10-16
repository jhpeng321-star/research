"""Assembly of datasets/physics/model from a TrainConfig (WP-I glue), without real data."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from wms.cli import assemble
from wms.config.load import load_train_config
from wms.core.errors import ConfigError

ROOT = Path(__file__).resolve().parents[3]


def test_declared_names_cover_every_dataset_of_r3():
    names = assemble.declared_data_names(load_train_config("r3"))
    assert names["pack[pack_v3]"] == "pack_v3"
    assert names["identities[pack_v3]"] == "sidecars_packv3_v2/identities"
    assert names["overlay[damping]"] == "overlay_damping"
    assert names["outcomes"] == "sidecars_packv3_v2/outcomes"
    assert names["windows.exclude[0]"] == "lpwo_v3/excluded_runs_PI_wall_v3.json"
    assert names["physics_params"] == "merge_params"
    assert names["switching.zero_shot_heldout"] == "zero_shot_heldout"
    assert "switching.windows_val" not in names               # optional, unset in the preset


def test_packs_only_preset_has_no_switching_names():
    names = assemble.declared_data_names(load_train_config("r3_packs_only"))
    assert not any(label.startswith("switching.") for label in names)
    assert load_train_config("r3").to_dict()["data"]["packs"] == \
        load_train_config("r3_packs_only").to_dict()["data"]["packs"]


def test_resolve_all_reports_unknown_names_instead_of_raising(tmp_path):
    cfg = load_train_config("r3_packs_only", ['data.windows.train="no_such_dataset/x.json"'])
    paths, errors = assemble.resolve_all(cfg)
    assert "windows.train" in errors and "no_such_dataset" in errors["windows.train"]
    assert "pack[pack_v3]" in paths and "windows.train" not in paths


def test_restricted_tokens_are_refused_at_resolution():
    cfg = load_train_config("r3_packs_only", ['data.windows.train="lpwo_v3/holdout_list.json"'])
    _, errors = assemble.resolve_all(cfg)
    assert "restricted" in errors["windows.train"]


def test_physics_follows_merge_cal_flags():
    r3 = assemble.build_physics(load_train_config("r3"))
    assert r3.merge_cal and r3.sumo_fidelity and r3.sumo_idm
    fitted = json.loads((ROOT / "configs" / "merge_params.json").read_text(encoding="utf-8"))
    key = next(k for k, v in fitted.items()
               if k in r3.__dataclass_fields__ and k != "merge_cal" and isinstance(v, float))
    assert getattr(r3, key) == fitted[key]                    # fitted params reach the config
    c1 = assemble.build_physics(load_train_config("c1fast"))
    assert not c1.merge_cal and c1.sumo_fidelity
    geometry = assemble.build_physics(load_train_config("c1fast", ["model.geometry_only=true"]))
    assert not geometry.sumo_idm


def test_model_follows_the_config():
    cfg = load_train_config("golden_r3")
    model = assemble.build_model(cfg, assemble.build_physics(cfg))
    assert model.residual_soft_bound_mps2 == 4.0 and model.history_s == 60.0
    assert model.rollout_mode == "residual"
    golden = json.loads((ROOT / "tests" / "golden" / "data" / "forward_r3.json").read_text())
    n_values = sum(v.numel() for v in model.state_dict().values())
    assert n_values == golden["state_checksum"]["n_params"]


def test_blackbox_kind_builds_the_legacy_baseline():
    cfg = load_train_config("c1fast", ['model.kind="blackbox_traj"'])
    model = assemble.build_model(cfg, assemble.build_physics(cfg))
    assert type(model).__name__ == "BlackBoxTrajModel"


def test_overlays_must_cover_exactly_the_extra_packs(tmp_path):
    cfg = load_train_config("r3_packs_only")
    raw = cfg.to_dict()
    del raw["data"]["sidecars"]["overlays"]["damping"]
    broken = type(cfg).from_dict(raw)
    with pytest.raises(ConfigError, match="overlays must name exactly"):
        assemble.build_datasets(broken, tmp_path, resolver=lambda name: tmp_path / name)


def test_missing_data_fails_closed_before_any_read(tmp_path):
    from wms.data.registry import require_present

    cfg = load_train_config("smoke_local")
    with pytest.raises(FileNotFoundError, match="not found"):
        assemble.build_datasets(cfg, tmp_path,
                                resolver=lambda name: require_present(str(tmp_path / "absent" / name)))
