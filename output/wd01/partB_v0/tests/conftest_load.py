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
        "_partb_v0_tests_conftest", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_c = _load()

CARD_MANIFEST_SHA = _c.CARD_MANIFEST_SHA
CARD_NATIVE_T570_TRAIN = _c.CARD_NATIVE_T570_TRAIN
CARD_ROWS = _c.CARD_ROWS
CARD_SOURCES = _c.CARD_SOURCES
CARD_TD = _c.CARD_TD
make_tiny_dataset = _c.make_tiny_dataset
