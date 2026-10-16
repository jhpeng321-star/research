"""Preset loading: extends, overrides, schema helpers, shipped presets."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal, Optional

import pytest

from wms.config import schema
from wms.config.eval import EvalConfig, EvalSetConfig, RolloutFlags
from wms.config.load import (apply_overrides, load_eval_config, load_raw, load_train_config,
                             parse_override, write_resolved)
from wms.config.presets import list_presets, preset_path
from wms.core.errors import ConfigError


def test_all_shipped_train_presets_load_and_validate():
    for name in ("c1fast", "r3", "r3_ctrl", "smoke"):
        cfg = load_train_config(name)
        assert cfg.schema_version == 2 and cfg.name.startswith("WM-S")
    assert set(list_presets()) >= {"c1fast", "r3", "r3_ctrl", "smoke", "eval_unseen"}


def test_r3_inherits_c1fast_and_overrides_what_the_chain_changes():
    c1, r3 = load_train_config("c1fast"), load_train_config("r3")
    assert r3.curriculum.stage_steps == c1.curriculum.stage_steps == (2500, 2500, 1875, 1250)
    assert r3.curriculum.unroll_s == (10.0, 30.0, 60.0, 120.0)
    assert (c1.model.n_controllers, c1.runtime.num_workers) == (6, 5)
    assert (r3.model.n_controllers, r3.runtime.num_workers) == (7, 0)
    assert r3.model.merge_cal and not c1.model.merge_cal
    assert r3.data.switching.mix_ratio == 0.3 and r3.data.switching.event_oversample == 1.0
    assert r3.loss.braking.asym_speed_alpha == 2.0 and r3.loss.braking.decel_weight == 5.0
    assert r3.loss.braking.propagation_weight == 1.0
    assert not r3.model.has_hard_cap and not r3.model.projection.enabled
    assert not r3.loss.traffic.enabled and r3.loss.stability_weight == 0.0
    assert r3.data.jam.oversample == 1.0 and not r3.data.entries_v2
    r3.validate_for_training()


def test_r3_ctrl_only_turns_braking_off():
    r3, ctrl = load_train_config("r3"), load_train_config("r3_ctrl")
    assert not ctrl.loss.braking.any_on and r3.loss.braking.any_on
    assert ctrl.data == r3.data and ctrl.model == r3.model and ctrl.curriculum == r3.curriculum


def test_smoke_has_six_steps_per_stage():
    assert load_train_config("smoke").curriculum.stage_steps == (6, 6, 6, 6)


def test_eval_preset():
    cfg = load_eval_config("eval_unseen")
    assert cfg.sets == ("wall_same", "wall_pilot", "seen_controls")
    assert cfg.previous_best == "ckpt_c1fast" and cfg.horizons_s == (10.0, 30.0, 60.0, 120.0)
    assert cfg.flags == RolloutFlags()


def test_driver_params_scoring_is_refused():
    with pytest.raises(ConfigError, match="driver_params"):
        load_eval_config("eval_unseen", ['flags.rollout_mode="driver_params"'])


def test_eval_set_specs_must_be_listed():
    with pytest.raises(ConfigError, match="set_specs"):
        EvalConfig(sets=("wall_same",), set_specs={"other": EvalSetConfig()})


@pytest.mark.parametrize("text, expected", [
    ("a.b=3", ("a.b", 3)), ("x=1e-3", ("x", 0.001)), ("x=true", ("x", True)),
    ("x=[1, 2]", ("x", [1, 2])), ('x="s"', ("x", "s")), ("x=bare_word", ("x", "bare_word")),
    ("x = 5", ("x", 5)), ("p=a/b.json", ("p", "a/b.json")),
])
def test_parse_override(text, expected):
    assert parse_override(text) == expected


@pytest.mark.parametrize("bad", ["novalue", "=3", ""])
def test_parse_override_rejects_malformed(bad):
    with pytest.raises(ConfigError):
        parse_override(bad)


def test_overrides_apply_in_order_and_deep_merge():
    raw = {"optim": {"lr": 1, "batch_size": 2}}
    out = apply_overrides(raw, ["optim.lr=3", "optim.lr=4", "runtime.seed=9"])
    assert out == {"optim": {"lr": 4, "batch_size": 2}, "runtime": {"seed": 9}}
    assert raw == {"optim": {"lr": 1, "batch_size": 2}}


def test_set_override_changes_config_and_unknown_key_is_refused():
    assert load_train_config("r3", ["optim.lr=5e-4"]).optim.lr == 5e-4
    assert load_train_config("r3", ["data.history_s=30"]).data.history_s == 30.0
    with pytest.raises(ConfigError, match="unknown keys"):
        load_train_config("r3", ["optim.learning_rate=1"])


def test_extends_chain_and_cycle_detection(tmp_path):
    (tmp_path / "base.toml").write_text('schema_version = 2\nname = "b"\n[optim]\nlr = 0.1\n'
                                        "batch_size = 2\n", encoding="utf-8")
    (tmp_path / "child.toml").write_text('extends = "base.toml"\nname = "c"\n[optim]\nlr = 0.2\n',
                                         encoding="utf-8")
    cfg = load_train_config(tmp_path / "child.toml")
    assert (cfg.name, cfg.optim.lr, cfg.optim.batch_size) == ("c", 0.2, 2)
    (tmp_path / "a.toml").write_text('extends = "b.toml"\n', encoding="utf-8")
    (tmp_path / "b.toml").write_text('extends = "a.toml"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="cycle"):
        load_raw(tmp_path / "a.toml")
    (tmp_path / "bad.toml").write_text("extends = 3\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="extends"):
        load_raw(tmp_path / "bad.toml")


def test_missing_preset_is_a_config_error():
    with pytest.raises(ConfigError, match="no such config"):
        preset_path("does_not_exist")


def test_write_resolved_records_fingerprint(tmp_path):
    cfg = load_train_config("smoke")
    target = write_resolved(cfg, tmp_path / "run")
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["fingerprint"] == cfg.fingerprint()
    assert payload["config"]["curriculum"]["stage_steps"] == [6, 6, 6, 6]


@dataclass(frozen=True)
class _Inner:
    x: int = 1


@dataclass(frozen=True)
class _Outer:
    inner: _Inner = field(default_factory=_Inner)
    mode: Literal["a", "b"] = "a"
    values: tuple[float, ...] = ()
    maybe: Optional[str] = None
    table: dict[str, str] = field(default_factory=dict)


def test_schema_coercion_rules():
    out = schema.from_mapping(_Outer, {"inner": {"x": 2}, "values": [1, 2.5], "maybe": "m",
                                       "table": {"k": "v"}})
    assert out.inner.x == 2 and out.values == (1.0, 2.5) and isinstance(out.values[0], float)
    for bad in ({"mode": "c"}, {"values": "no"}, {"inner": {"x": 1.5}}, {"maybe": 3},
                {"table": {"k": 1}}, {"inner": 5}, {"zzz": 1}):
        with pytest.raises(ConfigError):
            schema.from_mapping(_Outer, bad)
    assert schema.to_mapping(out)["values"] == [1.0, 2.5]


def test_deep_merge_does_not_alias_and_fingerprint_is_key_order_independent():
    a = {"t": {"x": 1}}
    merged = schema.deep_merge(a, {"t": {"y": 2}})
    assert merged == {"t": {"x": 1, "y": 2}} and a == {"t": {"x": 1}}
    assert schema.fingerprint({"a": 1, "b": 2}) == schema.fingerprint({"b": 2, "a": 1})
