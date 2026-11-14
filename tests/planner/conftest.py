"""PL-13 guard: fail loudly if the planner suite ever touches the D: drive.

With ``PLANNER_FORBID_D=1`` in the environment, an autouse fixture patches
``builtins.open`` / ``io.open``, ``os.stat`` and ``pathlib.Path.open`` /
``Path.exists`` / ``Path.is_file`` so ANY path starting with ``D:`` (either
slash, case-insensitive) raises. The D: junctions are scheduled for removal;
the proof run is::

    PLANNER_FORBID_D=1 python -m pytest tests/planner -q -p no:cacheprovider

Inactive (plain pass-through) unless the env var is set, so normal runs are
unaffected.
"""

import builtins
import io
import os
from pathlib import Path

import pytest

_ENV_VAR = "PLANNER_FORBID_D"


def _is_d_drive(path_like) -> bool:
    """True for any path-like whose text starts with 'D:' (case-insensitive)."""
    try:
        text = os.fspath(path_like)
    except TypeError:
        return False
    if isinstance(text, bytes):
        try:
            text = os.fsdecode(text)
        except UnicodeDecodeError:
            return False
    return len(text) >= 2 and text[0] in "dD" and text[1] == ":"


def _raise(path_like) -> None:
    raise PermissionError(
        f"{_ENV_VAR}=1: access to the doomed D: drive is forbidden "
        f"(path: {path_like!r}); resolve roots via "
        f"traffic_shepherd.planner.paths instead"
    )


@pytest.fixture(autouse=True)
def _forbid_d_drive(monkeypatch):
    if os.environ.get(_ENV_VAR) != "1":
        yield
        return

    real_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        if _is_d_drive(file):
            _raise(file)
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(io, "open", guarded_open, raising=False)

    real_stat = os.stat

    def guarded_stat(path, *args, **kwargs):
        if _is_d_drive(path):
            _raise(path)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", guarded_stat)

    for method_name in ("open", "exists", "is_file"):
        real_method = getattr(Path, method_name)

        def _make_guarded(real):
            def guarded(self, *args, **kwargs):
                if _is_d_drive(self):
                    _raise(self)
                return real(self, *args, **kwargs)

            guarded.__name__ = f"guarded_{method_name}"
            return guarded

        monkeypatch.setattr(Path, method_name, _make_guarded(real_method))

    yield


def _ensure_controller_verification_namespace() -> None:
    """PL-04k: the real PI session imports ``output.controller_verification_*``
    (untracked campaign modules that a git worktree does not carry). When
    they are not importable from this worktree, extend the ``output``
    namespace package with the main repo's ``output/`` dir (E: only,
    read-only import; nothing is written or modified there)."""
    import importlib
    import sys

    try:
        importlib.import_module("output.controller_verification_v7.pi_owner_adapter")
        return
    except ModuleNotFoundError:
        pass
    main = Path(os.environ.get("PLANNER_MAIN_REPO", "E:/research-dense"))
    if not (main / "output").is_dir():
        return
    # APPENDED (never inserted): this worktree's own packages keep precedence,
    # and a sys.path entry (unlike an appended ``output.__path__`` item)
    # survives the namespace package's path recalculation.
    if str(main) not in sys.path:
        sys.path.append(str(main))


_ensure_controller_verification_namespace()
