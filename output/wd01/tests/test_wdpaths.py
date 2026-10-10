"""Unit tests for wdpaths.remap() and the forbid-D audit hook.

Run:  python -m pytest output/wd01/tests -q
With WD_FORBID_D=1 the canary test additionally proves the conftest
installed the hook before this module was imported.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]      # output/wd01
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import forbid_d_hook  # noqa: E402
import wdpaths  # noqa: E402


# ------------------------------------------------------- wdpaths.remap ----

def test_roots_and_defaults():
    assert wdpaths.RESEARCH_DENSE_ROOT == os.environ.get(
        "RESEARCH_DENSE_ROOT", "").strip() or "E:/research-dense"
    assert wdpaths.RD_WT_ROOT == os.environ.get(
        "RD_WT_ROOT", "").strip() or "E:/rd-wt"
    assert wdpaths.BACKUPS_ROOT == os.environ.get(
        "RESEARCH_DENSE_BACKUPS_ROOT", "").strip() \
        or "E:/research-dense-backups"


def test_remap_forward_slashes():
    assert wdpaths.remap("D:/research-dense") == "E:/research-dense"
    assert wdpaths.remap("D:/research-dense/runs/x.json") == \
        "E:/research-dense/runs/x.json"
    assert wdpaths.remap("D:/rd-wt/wd01/output/wd01") == \
        "E:/rd-wt/wd01/output/wd01"
    assert wdpaths.remap("D:/research-dense-backups/wd01/part1") == \
        "E:/research-dense-backups/wd01/part1"


def test_remap_backslashes():
    assert wdpaths.remap("D:\\research-dense\\runs\\cv17") == \
        "E:/research-dense\\runs\\cv17"
    assert wdpaths.remap("D:\\rd-wt\\wd01n\\output\\wd01\\partB_v0data") == \
        "E:/rd-wt\\wd01n\\output\\wd01\\partB_v0data"
    assert wdpaths.remap("D:\\research-dense-backups\\wd01") == \
        "E:/research-dense-backups\\wd01"


def test_remap_case_insensitive_drive_and_segment():
    assert wdpaths.remap("d:/research-dense") == "E:/research-dense"
    assert wdpaths.remap("D:/RESEARCH-DENSE/runs") == \
        "E:/research-dense/runs"
    assert wdpaths.remap("d:\\RD-WT\\x") == "E:/rd-wt\\x"


def test_remap_longest_prefix_wins():
    # research-dense-backups must not be mangled by the research-dense rule
    assert wdpaths.remap("D:/research-dense-backups/wd01/part3/run3") == \
        "E:/research-dense-backups/wd01/part3/run3"


def test_remap_bare_prefix():
    assert wdpaths.remap("D:/research-dense") == "E:/research-dense"
    assert wdpaths.remap("D:/rd-wt") == "E:/rd-wt"
    assert wdpaths.remap("D:\\research-dense") == "E:/research-dense"


def test_remap_path_objects():
    p = wdpaths.remap(Path("D:/rd-wt/wd01/output/wd01/partA_data"))
    assert isinstance(p, Path)
    assert p == Path("E:/rd-wt/wd01/output/wd01/partA_data")
    s = wdpaths.remap("D:/rd-wt/x")
    assert isinstance(s, str) and s == "E:/rd-wt/x"


def test_remap_extended_length_prefix():
    assert wdpaths.remap("\\\\?\\D:\\research-dense\\x.npz") == \
        "\\\\?\\E:/research-dense\\x.npz"


def test_remap_non_d_paths_untouched():
    assert wdpaths.remap("E:/research-dense") == "E:/research-dense"
    assert wdpaths.remap("C:/Users/jpeng") == "C:/Users/jpeng"
    assert wdpaths.remap("relative/path/x.json") == "relative/path/x.json"
    assert wdpaths.remap("D:/other/tree") == "D:/other/tree"
    assert wdpaths.remap("D:/research-denseX/y") == "D:/research-denseX/y"
    assert wdpaths.remap("D:") == "D:"
    assert wdpaths.remap(Path("E:/rd-wt")) == Path("E:/rd-wt")


def test_remap_rejects_non_paths():
    with pytest.raises(TypeError):
        wdpaths.remap(42)
    with pytest.raises(TypeError):
        wdpaths.remap(None)


def test_remap_env_override(monkeypatch):
    monkeypatch.setenv("RD_WT_ROOT", "W:/wt")
    monkeypatch.setenv("RESEARCH_DENSE_ROOT", "W:/rd")
    monkeypatch.setenv("RESEARCH_DENSE_BACKUPS_ROOT", "W:/bk")
    fresh = importlib.reload(wdpaths)
    try:
        assert fresh.RD_WT_ROOT == "W:/wt"
        assert fresh.RESEARCH_DENSE_ROOT == "W:/rd"
        assert fresh.BACKUPS_ROOT == "W:/bk"
        assert fresh.remap("D:/rd-wt/wd01") == "W:/wt/wd01"
        assert fresh.remap("D:/research-dense/runs") == "W:/rd/runs"
        assert fresh.remap("D:/research-dense-backups/b") == "W:/bk/b"
    finally:
        monkeypatch.undo()
        importlib.reload(wdpaths)


# --------------------------------------------------------- forbid-D hook ----

def test_is_d_path_forms():
    assert forbid_d_hook.is_d_path("D:/x")
    assert forbid_d_hook.is_d_path("d:/x")
    assert forbid_d_hook.is_d_path("D:\\x")
    assert forbid_d_hook.is_d_path("\\\\?\\D:\\x")
    assert forbid_d_hook.is_d_path(Path("D:/x"))
    assert not forbid_d_hook.is_d_path("E:/x")
    assert not forbid_d_hook.is_d_path("C:/x")
    assert not forbid_d_hook.is_d_path("relative/x")
    assert not forbid_d_hook.is_d_path("\\\\?\\C:\\x")
    assert not forbid_d_hook.is_d_path(3)          # fd
    assert not forbid_d_hook.is_d_path(b"D:/x")    # bytes unsupported


def test_hook_installed():
    # conftest installs it when WD_FORBID_D=1; install() is idempotent
    forbid_d_hook.install()
    assert forbid_d_hook.installed()


def test_hook_fires_on_d_open():
    with pytest.raises(forbid_d_hook.ForbiddenDriveDError):
        open("D:/research-dense/__forbid_d_test_canary__", "rb")


def test_hook_fires_on_extended_prefix_open():
    with pytest.raises(forbid_d_hook.ForbiddenDriveDError):
        open("\\\\?\\D:\\research-dense\\__forbid_d_test_canary__", "rb")


def test_hook_fires_on_d_listdir():
    with pytest.raises(forbid_d_hook.ForbiddenDriveDError):
        os.listdir("D:/")


def test_hook_quiet_on_e_paths(tmp_path):
    p = tmp_path / "ok.txt"
    p.write_text("ok", encoding="utf-8")
    assert p.read_text(encoding="utf-8") == "ok"
    os.listdir(tmp_path)


def test_conftest_canary():
    """When WD_FORBID_D=1, the hook must ALREADY be installed by
    output/wd01/conftest.py before this test module was imported."""
    if os.environ.get(forbid_d_hook.WD_FORBID_D_ENV, "").strip() != "1":
        pytest.skip("WD_FORBID_D not set (hook activation not under test)")
    assert forbid_d_hook.installed()
    with pytest.raises(forbid_d_hook.ForbiddenDriveDError):
        open("D:/research-dense/__conftest_canary__", "rb")
