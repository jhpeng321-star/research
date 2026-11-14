"""Explicit loader for this directory's conftest helpers.

A bare ``from conftest import ...`` binds to whichever tests directory's
conftest pytest imported first; loading by absolute path keeps the
partB_v0* test trees runnable in one pytest invocation.
"""
import importlib.util
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parent / "conftest.py"
    spec = importlib.util.spec_from_file_location(
        "_partb_v0data_tests_conftest", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_c = _load()

HoldoutAccessWatch = _c.HoldoutAccessWatch
make_mini_v02_root = _c.make_mini_v02_root
rewrite_arm_to_speed_mode = _c.rewrite_arm_to_speed_mode
