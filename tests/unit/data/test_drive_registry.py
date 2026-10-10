"""tools/data_registry.json: the generated Google Drive registry (offline, no network).

Covers: the file loads and every entry passes ``tools/fetch_data.py`` validation, the logical
names the research code relies on exist, nothing restricted is registered, the entry layout is
internally consistent, and ``configs/data_registry.toml`` / ``WMS_DATA_<NAME>`` keep precedence.
"""
from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath

import pytest

from tools import build_drive_registry as builder
from wms.cli.fetch import load_fetch_tool
from wms.core import guard, toml
from wms.data import registry

ROOT = Path(__file__).resolve().parents[3]
REGISTRY_JSON = ROOT / "tools" / "data_registry.json"
REGISTRY_TOML = ROOT / "configs" / "data_registry.toml"

# logical name -> (Drive path suffix below gdrive:research-archive/, original local path)
REQUIRED = {
    "shards_v3": ("data/staging/rd-staging/shards_v3/shards_v3.tar.zst",
                  "D:/rd-staging/wmn1_feat/shards_v3"),
    "raw_chunk1_p40": ("repo-evidence/runs/raw_and_superseded/raw_chunk1_p40.tar.zst",
                       "D:/rd-staging/wmn1_feat/raw_chunk1_p40"),
    "campaign_stage1_chunk3": (
        "repo-evidence/runs/campaign_stage1_chunk3/campaign_stage1_chunk3.tar.zst",
        "E:/research-dense/runs/campaign_stage1_chunk3"),
    "campaign_stage1_chunk4": (
        "repo-evidence/runs/campaign_stage1_chunk4/campaign_stage1_chunk4.tar.zst",
        "E:/research-dense/runs/campaign_stage1_chunk4"),
    "runs_active": ("repo-evidence/runs/active/active.tar.zst", "E:/research-dense/runs/active"),
    "pack_v3": ("data/packs/training_pack_v3_k192", "D:/rd-staging/wmn1_feat/pack_v3"),
}
MD5 = re.compile(r"^[0-9a-f]{32}$")


@pytest.fixture(scope="module")
def fetch_tool():
    return load_fetch_tool()


@pytest.fixture(scope="module")
def datasets(fetch_tool):
    # _load_registry applies the tool's complete schema validation to every entry.
    return fetch_tool._load_registry(REGISTRY_JSON)


@pytest.fixture
def clean_env(monkeypatch):
    for key in [k for k in __import__("os").environ if k.startswith(registry.ENV_PREFIX)]:
        monkeypatch.delenv(key)


def test_registry_loads_and_every_entry_validates(datasets):
    raw = json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 1 and set(raw) == {"schema_version", "description", "datasets"}
    assert len(datasets) == len(raw["datasets"]) >= 600


def test_fetch_list_prints_every_entry(fetch_tool, datasets, capsys):
    assert fetch_tool.main(["--registry", str(REGISTRY_JSON), "list"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [line.split("\t")[0] for line in lines] == sorted(datasets)


def test_required_logical_names_exist_and_point_at_the_expected_drive_objects(datasets):
    for name, (drive_suffix, original) in REQUIRED.items():
        entry = datasets[name]
        assert entry["drive_path"] == "gdrive:research-archive/" + drive_suffix, name
        assert entry["local_path"] == f"data_cache/{name}"
        assert entry["original_path"] == original
        assert entry["size_bytes"] > 0


def test_pack_v3_is_the_verified_multi_part_directory(datasets):
    entry = datasets["pack_v3"]
    manifest = entry["md5_manifest"]
    parts = [k for k in manifest if ".tar.zst.part" in k]
    assert entry["format"] == "tar.zst.parts" and entry["parts"] == sorted(parts)
    assert len(parts) == 4 and {"SHA256SUMS", "UNPACK.md", "PARTS.json"} <= set(manifest)
    assert entry["size_bytes"] > 2 * 1024**3 * 3  # three full 2 GiB parts plus the tail


def test_entry_layout_is_consistent(datasets):
    local_paths = [e["local_path"] for e in datasets.values()]
    assert len(set(local_paths)) == len(local_paths)  # no two datasets share a cache directory
    for name, entry in datasets.items():
        assert entry["local_path"] == f"data_cache/{name}", name  # flat: never nested
        assert all(MD5.fullmatch(v) for v in entry["md5_manifest"].values()), name
        assert entry["receipt"]["source"] and entry["receipt"]["id"], name
        assert entry["class"] and entry["kind"] and "original_path" in entry, name
        fmt = entry["format"]
        assert fmt in {"tar.zst", "files", "tar.zst.parts"}, name
        if fmt == "tar.zst":
            # the verified object is the archive: one member named like the Drive file
            assert list(entry["md5_manifest"]) == [PurePosixPath(entry["drive_path"]).name], name
            assert entry["drive_path"].endswith(".tar.zst") and entry["size_bytes"] > 0, name
            assert entry["unpacked_bytes"] >= 0 and entry["unpacked_files"] >= 0, name
        if fmt == "files":
            assert entry["size_bytes"] == entry["unpacked_bytes"], name
            assert len(entry["md5_manifest"]) == entry["unpacked_files"], name
        for parent in entry.get("covered_by", []):
            assert parent in datasets and parent != name, name


def _strings(value):
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)
    elif value:
        yield str(value)


def test_no_restricted_names_are_registered(datasets):
    for name, entry in datasets.items():
        texts = [name, entry["drive_path"], entry["local_path"], entry["original_path"],
                 entry["receipt"]["id"], entry["receipt"]["file"], entry.get("branch"),
                 list(entry["md5_manifest"]), entry.get("parts")]
        for item in _strings(texts):
            assert builder.sensitive_kind(item) is None, f"{name}: restricted token"
            assert guard.find_restricted_token(item) is None, f"{name}: restricted token"
        registry.refuse_restricted(name, what="dataset name")  # the resolver would accept it


def test_generator_sensitivity_matches_the_repo_guard_and_the_owner_list():
    for token in ("holdout", "cal_split", "CAL_SPLIT", "chunk5", "chunk-5", "j4b", "inc1", "sealed"):
        assert builder.sensitive_kind(f"x_{token}_y") == "substring"
    assert builder.sensitive_kind("state_calibration_v1") == "guard"
    assert guard.find_restricted_token("state_calibration_v1") == "calibration"
    assert builder.sensitive_kind("runs_active_WM-N1-SCORE-CALTUNE-20261004") is None
    assert guard.find_restricted_token("runs_active_WM-N1-SCORE-CALTUNE-20261004") is None
    assert builder.sensitive_kind("campaign_stage1_chunk4") is None
    assert builder._GUARD.pattern == guard._RESTRICTED.pattern  # the copy has not drifted


def test_toml_names_take_precedence_over_the_json_registry(datasets, clean_env, tmp_path, monkeypatch):
    toml_names = set(toml.load(REGISTRY_TOML)["datasets"])
    overlap = sorted(toml_names & set(datasets))
    assert {"pack_v3", "identities_dspeed", "overlay_dspeed", "sidecars_packv3_v2"} <= set(overlap)
    for name in overlap:  # the real repo: both registries agree on where the data lives
        from_toml = toml.load(REGISTRY_TOML)["datasets"][name]["path"]
        assert registry.lookup(name).path == ROOT / from_toml
        assert datasets[name]["local_path"] == from_toml
    # a JSON-only name resolves through the JSON registry (and honours a cache override)
    assert registry.lookup("shards_v3").path == ROOT / "data_cache" / "shards_v3"
    other = tmp_path / "cache"
    assert registry.resolve("shards_v3/x", cache_dir=other) == other / "shards_v3" / "x"
    # strict precedence on an isolated copy: a toml path that differs from the JSON one wins
    (tmp_path / "configs").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "data_registry.json").write_text(
        REGISTRY_JSON.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "configs" / "data_registry.toml").write_text(
        '[datasets.pack_v3]\npath = "elsewhere/pack_v3"\n', encoding="utf-8")
    monkeypatch.setattr(registry, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(registry, "config_dir", lambda: tmp_path / "configs")
    assert registry.lookup("pack_v3").path == tmp_path / "elsewhere" / "pack_v3"
    assert registry.lookup("runs_active").path == tmp_path / "data_cache" / "runs_active"
    monkeypatch.setenv("WMS_DATA_PACK_V3", "/box/pack")
    assert registry.resolve("pack_v3/a.npy") == Path("/box/pack/a.npy")


@pytest.mark.skipif(not builder.DEFAULT_RECEIPTS.is_dir(), reason="data_ops/receipts not present")
def test_committed_registry_matches_the_generator_output(datasets):
    # Offline regeneration from the in-repo receipts; the member-manifest annotations
    # (covered_by, restricted_member_names) are carried over from the committed file.
    expected = builder.render(builder.Builder(builder.DEFAULT_RECEIPTS, previous=datasets).build())
    assert REGISTRY_JSON.read_text(encoding="utf-8") == expected
