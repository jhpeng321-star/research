"""Locating the preset files shipped in ``configs/``."""
from __future__ import annotations

from pathlib import Path
from typing import Union

from ..core.errors import ConfigError

PathLike = Union[str, Path]


def repo_root() -> Path:
    """Repository root (``src/wms/config`` sits three levels below it)."""
    return Path(__file__).resolve().parents[3]


def config_dir() -> Path:
    """Directory holding the ``*.toml`` presets."""
    return repo_root() / "configs"


def preset_path(name_or_path: PathLike) -> Path:
    """Resolve a preset name ("r3"), "r3.toml" or an explicit path to an existing file."""
    candidate = Path(name_or_path)
    if candidate.is_file():
        return candidate
    stem = candidate.name if candidate.suffix == ".toml" else f"{candidate.name}.toml"
    preset = config_dir() / stem
    if preset.is_file():
        return preset
    raise ConfigError(f"no such config or preset: {name_or_path!s} (looked in {config_dir()})")


def list_presets() -> list[str]:
    """Names of the shipped presets, sorted."""
    return sorted(p.stem for p in config_dir().glob("*.toml") if p.stem != "data_registry")
