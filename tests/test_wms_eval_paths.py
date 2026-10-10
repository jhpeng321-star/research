import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "output" / "wd01" / "wm_n1"
sys.path.insert(0, str(PKG))
from wms_eval.cli import _inside_workspace  # noqa: E402


def test_relative_arguments_are_anchored_and_escapes_refused():
    inside = _inside_workspace(Path("_out") / "wms_eval_paths_absent")
    assert inside.is_absolute()
    with pytest.raises(ValueError):
        _inside_workspace(Path("..") / "outside")


def test_absolute_outside_workspace_is_refused():
    with pytest.raises(ValueError):
        _inside_workspace(Path("C:/Windows"))


def _make_dir_link(link: Path, target: Path) -> str:
    """Create a directory link; return "symlink" or "junction" (Windows)."""
    try:
        link.symlink_to(target, target_is_directory=True)
        return "symlink"
    except (OSError, NotImplementedError):
        pass
    if os.name == "nt":
        done = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True, text=True)
        if done.returncode == 0 and link.exists():
            return "junction"
    pytest.skip("directory links are unavailable in this environment")


def test_link_escaping_the_workspace_is_refused(tmp_path, monkeypatch):
    root = Path(os.path.realpath(str(_inside_workspace(Path("_out")))))
    link = root / f"escape_{os.getpid()}"
    kind = _make_dir_link(link, tmp_path)
    try:
        # Workspace-confined --basetemp makes tmp_path a legitimate internal
        # link target. Model an external resolved target without reading or
        # creating anything outside the checkout, retaining the real link test.
        realpath = os.path.realpath
        resolved_payload = realpath(str(tmp_path / "payload"))
        if Path(resolved_payload).is_relative_to(ROOT):
            assert _inside_workspace(Path("_out") / link.name / "payload") == Path(
                resolved_payload)

            def escaped_realpath(value, *args, **kwargs):
                resolved = realpath(value, *args, **kwargs)
                if resolved == resolved_payload:
                    return str(ROOT.parent / "outside_fixture" / "payload")
                return resolved

            monkeypatch.setattr(os.path, "realpath", escaped_realpath)
        with pytest.raises(ValueError):
            _inside_workspace(Path("_out") / link.name / "payload")
    finally:
        if kind == "junction":
            subprocess.run(["cmd", "/c", "rmdir", str(link)], capture_output=True)
        else:
            link.unlink()
