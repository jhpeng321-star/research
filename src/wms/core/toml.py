"""TOML loading that works on Python 3.10 (no ``tomllib``) without extra installs.

Order of preference: stdlib ``tomllib`` (3.11+), then ``tomli`` if importable, then
the strict vendored subset reader in :mod:`wms.core.toml_subset`.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Union

from .toml_subset import TomlError
from .toml_subset import loads as _subset_loads


def _select_loads() -> Callable[[str], dict[str, Any]]:
    """Return the best available ``loads`` implementation."""
    try:
        import tomllib
        return tomllib.loads
    except ImportError:
        pass
    try:
        import tomli
        return tomli.loads
    except ImportError:
        return _subset_loads


_loads = _select_loads()


def loads(text: str) -> dict[str, Any]:
    """Parse TOML text into nested dicts; raises :class:`TomlError` on bad input."""
    try:
        return _loads(text)
    except TomlError:
        raise
    except ValueError as exc:  # tomllib / tomli decode errors
        raise TomlError(str(exc)) from exc


def load(path: Union[str, "os.PathLike[str]"]) -> dict[str, Any]:
    """Read and parse a UTF-8 TOML file."""
    with open(path, "r", encoding="utf-8") as stream:
        return loads(stream.read())


__all__ = ["TomlError", "load", "loads"]
