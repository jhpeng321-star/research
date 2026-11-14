"""RC-10 O5: chunk-5 seal path resolution in tools/holdout_guard.py.

The seal path is an explicit parameter/env (``CHUNK5_SEAL_PATH``) with the
default ``<ROOT>/protocols/chunk5_test_seal.json`` (ROOT = env
``RESEARCH_DENSE_ROOT`` or its recorded default), and a MISSING seal file
fails CLOSED loudly (a named ``HoldoutExclusionError``), never a silent
pass.  These tests exec the REAL ``tools/holdout_guard.py`` with env
overrides and fixture seals only; no outcome file is touched.

Lives in its own module because ``tests/planner/test_holdout_guard.py``
import-or-skips wholesale when the planner-worktree report modules are
absent from this workspace; the seal resolution must stay tested here.
"""

import importlib.util
import json
from pathlib import Path

import pytest

_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def tools_guard():
    path = _WORKSPACE_ROOT / "tools" / "holdout_guard.py"
    spec = importlib.util.spec_from_file_location(
        "rc10_tools_holdout_guard_seal", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_seal_fails_closed_loudly(tools_guard, tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_DENSE_ROOT", str(tmp_path))
    monkeypatch.delenv(tools_guard.CHUNK5_SEAL_ENV, raising=False)
    with pytest.raises(tools_guard.HoldoutExclusionError, match="missing"):
        tools_guard.load_chunk5_seal()
    with pytest.raises(tools_guard.HoldoutExclusionError, match="missing"):
        tools_guard.chunk5_sealed_roots()
    with pytest.raises(tools_guard.HoldoutExclusionError, match="missing"):
        tools_guard.assert_chunk5_not_sealed(
            ["wx_world__base"], reader="ANALYSIS"
        )
    with pytest.raises(tools_guard.HoldoutExclusionError, match="missing"):
        tools_guard.filter_chunk5_sealed(
            ["wx_world__base"], reader="ANALYSIS"
        )


def test_seal_env_override_is_honored(tools_guard, tmp_path, monkeypatch):
    seal = tmp_path / "elsewhere" / "chunk5_test_seal.json"
    seal.parent.mkdir(parents=True)
    seal.write_text(
        json.dumps(
            {
                "schema": "chunk5_test_seal/1",
                "allowed_reader": "WM-N1_FINAL_TEST",
                "sealed_roots": ["wx_fixture_sealed"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(tools_guard.CHUNK5_SEAL_ENV, str(seal))
    monkeypatch.setenv("RESEARCH_DENSE_ROOT", str(tmp_path / "no_root"))
    assert tools_guard.chunk5_seal_path() == seal
    data = tools_guard.load_chunk5_seal()
    assert tools_guard.chunk5_sealed_roots(data) == {"wx_fixture_sealed"}
    with pytest.raises(tools_guard.HoldoutExclusionError):
        tools_guard.assert_chunk5_not_sealed(
            ["wx_fixture_sealed__base"], reader="ANALYSIS"
        )
    tools_guard.assert_chunk5_not_sealed(
        ["wx_fixture_sealed__base"], reader="WM-N1_FINAL_TEST"
    )
    assert tools_guard.filter_chunk5_sealed(
        ["wx_fixture_sealed__base", "wx_open__base"], reader="ANALYSIS"
    ) == ["wx_open__base"]


def test_default_path_is_root_protocols_seal(tools_guard, tmp_path,
                                             monkeypatch):
    monkeypatch.delenv(tools_guard.CHUNK5_SEAL_ENV, raising=False)
    monkeypatch.setenv("RESEARCH_DENSE_ROOT", str(tmp_path))
    assert tools_guard.chunk5_seal_path() == (
        tmp_path / "protocols" / "chunk5_test_seal.json"
    )
    assert tools_guard.chunk5_seal_path(root=tmp_path / "alt") == (
        tmp_path / "alt" / "protocols" / "chunk5_test_seal.json"
    )


def test_real_rc10_seal_loads_via_explicit_path(tools_guard):
    seal_path = (_WORKSPACE_ROOT / "protocols" / "chunk5_test_seal.json")
    data = tools_guard.load_chunk5_seal(seal_path)
    assert data["allowed_reader"] == "WM-N1_FINAL_TEST"
    assert len(tools_guard.chunk5_sealed_roots(data)) == 30
