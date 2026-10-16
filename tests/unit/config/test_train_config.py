"""TrainConfig: validation, legacy flat format, fingerprint, parity with WmSTrainConfig."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from wms.config.load import load_train_config
from wms.config.schema import from_mapping
from wms.config.train import ModelConfig, TrainConfig
from wms.core.errors import ConfigError, TrainDataRefusal

from .old_code import run_old

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "c1fast_receipt_subset.json"


def build(**tree) -> TrainConfig:
    return from_mapping(TrainConfig, tree)


def test_defaults_match_the_legacy_trainer_defaults():
    old = run_old("import json; from wms.train import WmSTrainConfig as C; "
                  "print(json.dumps(C().to_dict()))")
    new = TrainConfig().to_legacy_flat()
    old.pop("out_dir")
    new.pop("out_dir")
    assert new == old


def test_every_field_combination_matches_legacy_to_dict():
    flat = {"unroll_curriculum_s": [10.0, 30.0], "stage_steps": [3, 4], "num_workers": 0,
            "batch_size": 4, "lr": 5e-4, "history_s": 30.0, "history_stride_s": 1.0,
            "rollout_mode": "hybrid", "lane_head": "history", "residual_magnitude_weight": 0.01,
            "asym_speed_alpha": 2.0, "decel_weight": 5.0, "propagation_weight": 1.0,
            "prop_max_gap_m": 40.0, "residual_soft_bound_mps2": 1.5,
            "residual_soft_bound_neg_mps2": 2.5, "residual_mag_weight": 0.05,
            "lane_rate_weight": 0.1, "select_by_long_horizon": True, "val_every": 50,
            "cav_drive": "closed_loop", "traffic_loss": True, "traffic_ramp_start_s": 0.0,
            "traffic_ramp_end_s": 1.0, "projection_enabled": True, "stability_weight": 0.2,
            "conservation_enabled": True, "window_list": "w.json", "val_window_list": "v.json",
            "exclude_window_list": ["a.json", "b.json"], "driver_sidecar": "d.npz",
            "entries_v2": True, "black_box": False, "switch_window_fraction": 0.25,
            "out_dir": "x/run"}
    old = run_old("import json; from wms.train import WmSTrainConfig as C; "
                  f"print(json.dumps(C.from_dict({flat!r}).to_dict()))")
    new = TrainConfig.from_legacy_flat(flat).to_legacy_flat()
    assert new == old


def test_legacy_flat_roundtrip_of_c1fast_receipt():
    receipt = json.loads(FIXTURE.read_text(encoding="utf-8"))
    flat = receipt["config"]
    cfg = TrainConfig.from_legacy_flat(flat)
    again = cfg.to_legacy_flat()
    assert {k: again[k] for k in flat} == flat
    defaults = TrainConfig().to_legacy_flat()
    assert {k: v for k, v in again.items() if k not in flat} == \
        {k: defaults[k] for k in again if k not in flat}
    assert cfg.model.n_controllers == 6 and cfg.runtime.num_workers == 5
    assert cfg.curriculum.stage_steps == (2500, 2500, 1875, 1250)


def test_c1fast_preset_reproduces_the_receipt():
    receipt = json.loads(FIXTURE.read_text(encoding="utf-8"))
    flat = receipt["config"]
    names = {
        "lpo_c1/windows_train_noPI.json": flat["window_list"],
        "lpo_c1/windows_val.json": flat["val_window_list"],
        "lpo_c1/pi_excluded_runs.json": flat["exclude_window_list"][0],
    }
    cfg = load_train_config("c1fast", [f"runtime.out_dir={json.dumps(flat['out_dir'])}"])
    again = cfg.to_legacy_flat(resolve=names.__getitem__)
    assert {k: again[k] for k in flat} == flat
    assert cfg.data.start_shift_prob == receipt["data"]["start_shift_prob"] == 0.75
    assert len(cfg.data.packs) == len(receipt["data"]["packs"]) == 3
    assert len(cfg.data.sidecars.identities) == len(receipt["data"]["identity_sidecars"]) == 3
    assert len(cfg.data.sidecars.overlays) == len(receipt["data"]["overlays"]) == 2
    assert cfg.model.merge_cal is receipt["physics"]["merge_cal"] is False
    assert cfg.model.sumo_fidelity is receipt["physics"]["sumo_fidelity"]
    assert cfg.model.geometry_only is (not receipt["physics"]["sumo_idm"])


def test_unknown_flat_key_is_rejected():
    with pytest.raises(ConfigError, match="unknown WM-S config keys"):
        TrainConfig.from_legacy_flat({"not_a_field": 1})


def test_unknown_nested_key_is_rejected_with_valid_names():
    with pytest.raises(ConfigError, match="unknown keys"):
        build(optim={"learning_rate": 1})


def test_nested_dict_roundtrip_and_fingerprint():
    cfg = load_train_config("r3")
    assert TrainConfig.from_dict(cfg.to_dict()) == cfg
    assert cfg.fingerprint() == TrainConfig.from_dict(cfg.to_dict()).fingerprint()
    assert cfg.fingerprint() != load_train_config("r3", ["optim.lr=0.002"]).fingerprint()
    assert len(cfg.fingerprint()) == 64


@pytest.mark.parametrize("tree, message", [
    ({"curriculum": {"stage_steps": [1, 2]}}, "stage_steps"),
    ({"curriculum": {"unroll_s": []}}, "unroll_s"),
    ({"curriculum": {"unroll_s": [10, -1], "stage_steps": [1, 1]}}, "unroll_s"),
    ({"model": {"lane_head": "history"}}, "requires rollout_mode"),
    ({"model": {"kind": "blackbox_traj", "rollout_mode": "hybrid"}}, "blackbox_traj"),
    ({"model": {"residual_soft_bound_neg_mps2": 1.0}}, "requires residual_soft_bound_mps2"),
    ({"model": {"residual_soft_bound_mps2": 1.0, "residual_cap_mps2": 2.0}}, "cannot be combined"),
    ({"model": {"lane_logit_cap": -1.0}}, "lane_logit_cap"),
    ({"loss": {"braking": {"asym_speed_alpha": 0.5}}}, "asym_speed_alpha"),
    ({"loss": {"braking": {"decel_threshold_mps2": 0.0}}}, "decel_threshold_mps2"),
    ({"loss": {"traffic": {"ramp_start_s": 5.0, "ramp_end_s": 5.0}}}, "ramp_end_s"),
    ({"loss": {"lane_rate": {"target": 1.5}}}, "target"),
    ({"loss": {"residual": {"residual_mag_weight": -1.0}}}, "residual_mag_weight"),
    ({"data": {"history_s": 30.0, "history_stride_s": 0.7}}, "multiple"),
    ({"data": {"start_shift_prob": 1.5}}, "start_shift_prob"),
    ({"data": {"switching": {"pack": "p"}}}, "requires"),
    ({"data": {"switching": {"windows_train": "w"}}}, "without"),
    ({"data": {"windows": {"val_holdout_frac": 0.1}}}, "requires data.windows.train"),
    ({"data": {"packs": {"a": "x"}}}, "identities missing"),
    ({"data": {"sidecars": {"overlays": {"zz": "x"}}}}, "unknown pack roles"),
    ({"optim": {"batch_size": 0}}, "batch_size"),
    ({"schema_version": 1}, "schema_version"),
    ({"model": {"kind": "blackbox_id"}, "data": {"history_s": 30.0}}, "long history"),
    ({"model": {"kind": "blackbox_id"}, "loss": {"braking": {"decel_weight": 5.0}}}, "structured"),
    ({"optim": {"lr": "fast"}}, "expected float"),
    ({"runtime": {"threads": True}}, "expected int"),
])
def test_invalid_configs_are_refused(tree, message):
    with pytest.raises(ConfigError, match=message):
        build(**tree)


def test_hard_cap_warns_on_load_and_refuses_for_training():
    with pytest.warns(UserWarning, match="REJECTED"):
        cfg = build(model={"residual_cap_mps2": 3.0})
    with pytest.raises(ConfigError, match="REJECTED"):
        cfg.validate_for_training()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        allowed = build(model={"residual_cap_mps2": 3.0, "allow_rejected_hard_cap": True})
    allowed.validate_for_training()


def test_training_requires_zero_workers_and_validation_for_selection():
    with pytest.raises(ConfigError, match="num_workers"):
        build(runtime={"num_workers": 2}).validate_for_training()
    with pytest.raises(ConfigError, match="validation window list"):
        build(selection={"select_by_long_horizon": True}).validate_for_training()
    build(selection={"select_by_long_horizon": True},
          data={"windows": {"train": "t", "val": "v"}}).validate_for_training()


def test_training_refuses_restricted_paths():
    sealed = "seal" + "ed"
    cfg = build(data={"windows": {"train": f"{sealed}/windows.json"}})
    with pytest.raises(TrainDataRefusal):
        cfg.validate_for_training()


def test_required_log_terms_follow_enabled_terms():
    assert TrainConfig().required_log_terms() == ("position", "speed", "lane")
    cfg = build(loss={"traffic": {"enabled": True}, "stability_weight": 0.1,
                      "conservation_weight": 0.2,
                      "braking": {"asym_speed_alpha": 2.0, "decel_weight": 5.0,
                                  "propagation_weight": 1.0}})
    assert cfg.required_log_terms() == ("position", "speed", "lane", "traffic", "asym_speed",
                                        "decel_weight", "propagation", "stability", "conservation")


def test_effective_start_shift_is_zero_when_pi_wall_windows_are_left_out():
    cfg = build(data={"start_shift_prob": 0.75, "windows": {"leave_pi_wall_out": True}})
    assert cfg.data.start_shift_prob == 0.75 and cfg.data.effective_start_shift_prob == 0.0


def test_model_config_alone_has_no_cap():
    assert not ModelConfig().has_hard_cap
