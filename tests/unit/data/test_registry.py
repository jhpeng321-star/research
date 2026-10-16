"""wms.data.registry: resolution order, sub-paths, restricted-token refusal, fail-closed presence."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from wms.core.errors import ConfigError, TrainDataRefusal
from wms.data import registry

SEALED = "seal" + "ed"


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Point the registry at temporary toml/json files and clear WMS_DATA_* variables."""
    toml_dir = tmp_path / "configs"
    tools_dir = tmp_path / "tools"
    toml_dir.mkdir()
    tools_dir.mkdir()
    monkeypatch.setattr(registry, "config_dir", lambda: toml_dir)
    monkeypatch.setattr(registry, "repo_root", lambda: tmp_path)
    for key in [k for k in __import__("os").environ if k.startswith(registry.ENV_PREFIX)]:
        monkeypatch.delenv(key)
    return tmp_path


def write_toml(root: Path, body: str) -> None:
    (root / "configs" / "data_registry.toml").write_text(body, encoding="utf-8")


def write_json(root: Path, datasets: dict) -> None:
    (root / "tools" / "data_registry.json").write_text(
        json.dumps({"schema_version": 1, "datasets": datasets}), encoding="utf-8")


def test_env_var_name():
    assert registry.env_var_name("pack_v3") == "WMS_DATA_PACK_V3"
    assert registry.env_var_name("a-b.c") == "WMS_DATA_A_B_C"
    assert registry.split_name("lpwo_v3/sub/windows.json") == ("lpwo_v3", "sub/windows.json")
    assert registry.split_name("pack_v3") == ("pack_v3", "")


def test_environment_wins_over_toml_and_json(isolated, monkeypatch):
    write_toml(isolated, '[datasets.pack_v3]\npath = "from_toml"\n')
    write_json(isolated, {"pack_v3": {"local_path": "data_cache/pack_v3"}})
    monkeypatch.setenv("WMS_DATA_PACK_V3", "/box/pack")
    assert registry.resolve("pack_v3") == Path("/box/pack")
    assert registry.resolve("pack_v3/file.json") == Path("/box/pack/file.json")


def test_custom_env_override_name(isolated, monkeypatch):
    monkeypatch.setenv("MY_BOX_PACK", "/x/y")
    assert registry.resolve("pack_v3", env_override="MY_BOX_PACK") == Path("/x/y")


def test_toml_beats_json_and_relative_paths_are_rooted(isolated):
    write_toml(isolated, '[datasets.merge_params]\npath = "configs/merge_params.json"\n'
                         'sha_manifest = "abc"\n')
    write_json(isolated, {"merge_params": {"local_path": "data_cache/elsewhere"}})
    item = registry.lookup("merge_params")
    assert item.path == isolated / "configs" / "merge_params.json" and item.sha_manifest == "abc"


def test_json_registry_uses_data_cache_or_override(isolated):
    write_json(isolated, {"pack_v3": {"local_path": "data_cache/pack_v3"}})
    assert registry.resolve("pack_v3") == isolated / "data_cache" / "pack_v3"
    assert registry.resolve("pack_v3/a/b", cache_dir=isolated / "cache2") == \
        isolated / "cache2" / "pack_v3" / "a" / "b"


def test_absolute_names_pass_through(isolated, tmp_path):
    assert registry.resolve(str(tmp_path / "x")) == tmp_path / "x"
    assert registry.resolve("D:/data/pack") == Path("D:/data/pack")


def test_unknown_dataset_message_names_the_remedies(isolated):
    with pytest.raises(ConfigError, match="WMS_DATA_NOPE"):
        registry.resolve("nope/x")


def test_toml_entry_without_path_is_an_error(isolated):
    write_toml(isolated, "[datasets.bad]\nnote = 1\n")
    with pytest.raises(ConfigError, match="no 'path'"):
        registry.resolve("bad")


def test_restricted_names_and_paths_are_refused(isolated, monkeypatch):
    with pytest.raises(TrainDataRefusal):
        registry.resolve(f"{SEALED}_windows/x")
    monkeypatch.setenv("WMS_DATA_PACK_V3", f"/data/{SEALED}/pack")
    with pytest.raises(TrainDataRefusal):
        registry.resolve("pack_v3")
    with pytest.raises(TrainDataRefusal):
        registry.resolve(f"/abs/{SEALED}/path")


def test_require_present_is_fail_closed_with_fetch_hint(isolated, monkeypatch, tmp_path):
    monkeypatch.setenv("WMS_DATA_PACK_V3", str(tmp_path / "absent"))
    with pytest.raises(FileNotFoundError, match="fetch_data.py get pack_v3"):
        registry.require_present("pack_v3")
    (tmp_path / "present").mkdir()
    monkeypatch.setenv("WMS_DATA_PACK_V3", str(tmp_path / "present"))
    assert registry.require_present("pack_v3") == tmp_path / "present"


def test_shipped_registry_maps_merge_params_to_a_real_file():
    # Uses the real repository files (no isolation): the R3 preset relies on this entry.
    path = registry.require_present("merge_params")
    assert path.name == "merge_params.json" and "fitted merge-cal" in path.read_text(encoding="utf-8")
