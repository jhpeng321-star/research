"""P11 torch >=2.7,<2.13 compatibility: static scan + version-gated op test."""
from __future__ import annotations

from fakes import clean_torch_tree, dirty_torch_tree
from preflight import torch_compat


def test_scan_clean_tree(tmp_path):
    scan = torch_compat.scan_sources(clean_torch_tree(tmp_path))
    assert scan["ok"] is True and scan["forbidden_hits"] == []
    assert scan["n_files"] >= 1


def test_scan_finds_forbidden_api(tmp_path):
    scan = torch_compat.scan_sources(dirty_torch_tree(tmp_path))
    assert scan["ok"] is False
    assert any(h["api"] == "torch.compile" for h in scan["forbidden_hits"])


def test_scan_skips_torch_compat_itself(tmp_path):
    scan = torch_compat.scan_sources(tmp_path)   # the empty tmp dir
    assert scan["ok"] is True


def test_version_gate_runs_or_skips_cleanly():
    gate = torch_compat.version_gate()
    assert gate["ok"] is True
    assert gate["status"] in ("ran", "skipped")
    if gate["status"] == "skipped":
        assert "2.7" in gate["reason"]


def test_supports_window():
    assert torch_compat.supports("2.7.0+cu128") is True
    assert torch_compat.supports("2.12.1+cpu") is True
    assert torch_compat.supports("2.6.0") is False
    assert torch_compat.supports("2.13.0") is False


def test_scan_and_check_clean(tmp_path):
    out = torch_compat.scan_and_check(clean_torch_tree(tmp_path))
    assert out["ok"] is True
    assert out["version_gate"]["ok"] is True


def test_scan_and_check_dirty_fails(tmp_path):
    out = torch_compat.scan_and_check(dirty_torch_tree(tmp_path))
    assert out["ok"] is False
