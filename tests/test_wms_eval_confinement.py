import importlib
import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "output" / "wd01" / "wm_n1"
sys.path.insert(0, str(PKG))
from wms_eval.cli import _inside_workspace  # noqa: E402


def _make_dir_link(link: Path, target: Path) -> None:
    """Create a directory symlink or junction; skip if the OS forbids it."""
    if os.name == "nt":
        try:
            import _winapi

            _winapi.CreateJunction(str(target), str(link))
            return
        except (ImportError, OSError):
            pass
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass
    pytest.skip("OS cannot create a symlink or junction for this test")


def test_inside_workspace_rejects_a_link_that_escapes(tmp_path, monkeypatch):
    cli = importlib.import_module("wms_eval.cli")
    workspace = (tmp_path / "workspace").resolve()
    outside = (tmp_path / "outside").resolve()
    workspace.mkdir()
    outside.mkdir()
    (outside / "secret.json").write_text("{}", encoding="utf-8")

    link = workspace / "escape"
    _make_dir_link(link, outside)
    monkeypatch.setattr(cli, "WORKSPACE", workspace)

    with pytest.raises(ValueError):
        _inside_workspace(link / "secret.json")
    with pytest.raises(ValueError):
        _inside_workspace(outside / "secret.json")


def test_inside_workspace_allows_an_uncreated_nested_output(tmp_path, monkeypatch):
    cli = importlib.import_module("wms_eval.cli")
    workspace = (tmp_path / "workspace").resolve()
    workspace.mkdir()
    monkeypatch.setattr(cli, "WORKSPACE", workspace)

    target = workspace / "new_run" / "EVAL.json"
    assert _inside_workspace(target) == Path(os.path.realpath(str(target)))


def test_inside_workspace_allows_an_internal_link(tmp_path, monkeypatch):
    cli = importlib.import_module("wms_eval.cli")
    workspace = (tmp_path / "workspace").resolve()
    real = workspace / "real"
    real.mkdir(parents=True)

    link = workspace / "alias"
    _make_dir_link(link, real)
    monkeypatch.setattr(cli, "WORKSPACE", workspace)

    assert _inside_workspace(link / "out.json") == Path(
        os.path.realpath(str(real / "out.json"))
    )
