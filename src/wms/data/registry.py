"""Logical dataset names -> local paths.  The only place that knows where data lives.

Resolution order for a name ``<dataset>[/<relative path>]``:

1. environment variable ``WMS_DATA_<DATASET>`` (non-alphanumerics become ``_``),
   or the variable named by ``env_override`` (box-specific locations);
2. ``configs/data_registry.toml``  (``[datasets.<dataset>] path = "..."``);
3. ``tools/data_registry.json``    (``local_path`` below ``data_cache/``).

An absolute path passed as a name is taken as already resolved.  Every result is
checked against the restricted-token guard.  This module never downloads
anything: ``python tools/fetch_data.py get <name>`` does that.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from ..config.presets import config_dir, repo_root
from ..core import toml
from ..core.errors import ConfigError
from ..core.guard import refuse_restricted
from ..core.io import read_json

ENV_PREFIX = "WMS_DATA_"
REGISTRY_TOML = "data_registry.toml"
REGISTRY_JSON = "data_registry.json"


@dataclass(frozen=True)
class DataItem:
    """A resolved dataset: logical name, local path and optional manifest digest."""

    name: str
    path: Path
    sha_manifest: Optional[str] = None


def env_var_name(dataset: str) -> str:
    """Environment variable overriding ``dataset`` (e.g. ``pack_v3`` -> ``WMS_DATA_PACK_V3``)."""
    return ENV_PREFIX + re.sub(r"[^A-Za-z0-9]", "_", dataset).upper()


def split_name(name: str) -> tuple[str, str]:
    """Split ``"lpwo_v3/windows.json"`` into ``("lpwo_v3", "windows.json")``."""
    dataset, _, rest = name.replace("\\", "/").partition("/")
    return dataset, rest


def _toml_entries() -> Mapping[str, Mapping[str, Any]]:
    path = config_dir() / REGISTRY_TOML
    if not path.is_file():
        return {}
    return toml.load(path).get("datasets", {})


def _json_entries() -> Mapping[str, Mapping[str, Any]]:
    path = repo_root() / "tools" / REGISTRY_JSON
    if not path.is_file():
        return {}
    return read_json(path).get("datasets", {})


def _under_root(text: str, base: Path) -> Path:
    path = Path(text)
    return path if path.is_absolute() else base / path


def lookup(dataset: str, *, cache_dir: Optional[Path] = None,
           env_override: Optional[str] = None) -> DataItem:
    """Find the root path of ``dataset`` (no relative sub-path)."""
    env_value = os.environ.get(env_override or env_var_name(dataset))
    if env_value:
        return DataItem(dataset, Path(env_value))
    entry = _toml_entries().get(dataset)
    if entry is not None:
        if "path" not in entry:
            raise ConfigError(f"{REGISTRY_TOML}: dataset {dataset!r} has no 'path'")
        return DataItem(dataset, _under_root(entry["path"], repo_root()), entry.get("sha_manifest"))
    entry = _json_entries().get(dataset)
    if entry is not None:
        local = Path(entry["local_path"])
        if cache_dir is not None and local.parts[:1] == ("data_cache",):
            return DataItem(dataset, cache_dir.joinpath(*local.parts[1:]))
        return DataItem(dataset, repo_root() / local)
    raise ConfigError(
        f"unknown dataset {dataset!r}: set {env_var_name(dataset)}, add it to "
        f"configs/{REGISTRY_TOML}, or register it in tools/{REGISTRY_JSON}")


def resolve(name: str, *, cache_dir: Optional[Path] = None,
            env_override: Optional[str] = None) -> Path:
    """Local path of logical ``name``; raises ``TrainDataRefusal`` for restricted material."""
    refuse_restricted(name, what="dataset name")
    if os.path.isabs(name) or re.match(r"^[A-Za-z]:[\\/]", name):
        path = Path(name)
    else:
        dataset, rest = split_name(name)
        root = lookup(dataset, cache_dir=cache_dir, env_override=env_override).path
        path = root / rest if rest else root
    refuse_restricted(str(path), what="resolved path")
    return path


def require_present(name: str, *, cache_dir: Optional[Path] = None) -> Path:
    """Like :func:`resolve` but fail closed when the data is not on disk."""
    path = resolve(name, cache_dir=cache_dir)
    if not path.exists():
        dataset, _ = split_name(name)
        raise FileNotFoundError(
            f"data {name!r} not found at {path}; fetch it with "
            f"`python tools/fetch_data.py get {dataset}` or set {env_var_name(dataset)}")
    return path
