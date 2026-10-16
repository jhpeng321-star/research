"""Loading TOML presets with ``extends`` inheritance and ``--set a.b=c`` overrides."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, Type, TypeVar

from ..core import toml
from ..core.errors import ConfigError
from ..core.io import write_json_atomic
from .eval import EvalConfig
from .presets import PathLike, preset_path
from .schema import deep_merge, fingerprint, from_mapping, to_mapping
from .train import TrainConfig

T = TypeVar("T")
_MAX_EXTENDS_DEPTH = 16


def load_raw(path: PathLike, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Read a preset and deep-merge its ``extends`` chain (child keys win)."""
    file = preset_path(path).resolve()
    if file in _seen:
        raise ConfigError(f"config 'extends' cycle through {file}")
    if len(_seen) >= _MAX_EXTENDS_DEPTH:
        raise ConfigError("config 'extends' chain is too deep")
    raw = toml.load(file)
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    if not isinstance(parent, str):
        raise ConfigError(f"{file}: 'extends' must be a file name")
    base = load_raw(file.parent / parent, (*_seen, file))
    return deep_merge(base, raw)


def parse_override(text: str) -> tuple[str, Any]:
    """Split ``"a.b=c"``; ``c`` is parsed as a TOML value, else kept as a plain string."""
    key, sep, value = text.partition("=")
    key = key.strip()
    if not sep or not key:
        raise ConfigError(f"override {text!r} must look like section.key=value")
    try:
        parsed = toml.loads(f"v = {value.strip()}")["v"]
    except toml.TomlError:
        parsed = value.strip()
    return key, parsed


def apply_overrides(raw: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Return ``raw`` with every ``a.b=c`` override applied (later ones win)."""
    result = dict(raw)
    for text in overrides:
        key, value = parse_override(text)
        nested: Any = value
        for part in reversed(key.split(".")):
            nested = {part: nested}
        result = deep_merge(result, nested)
    return result


def _load(cls: Type[T], path: PathLike, overrides: Sequence[str]) -> T:
    return from_mapping(cls, apply_overrides(load_raw(path), overrides))


def load_train_config(path: PathLike, overrides: Sequence[str] = ()) -> TrainConfig:
    """Load a training preset plus ``--set`` overrides into a validated ``TrainConfig``."""
    return _load(TrainConfig, path, overrides)


def load_eval_config(path: PathLike, overrides: Sequence[str] = ()) -> EvalConfig:
    """Load an evaluation preset plus overrides into a validated ``EvalConfig``."""
    return _load(EvalConfig, path, overrides)


def write_resolved(cfg: Any, out_dir: PathLike) -> Path:
    """Write ``<out_dir>/config.resolved.json`` (nested config plus its fingerprint)."""
    target = Path(out_dir) / "config.resolved.json"
    write_json_atomic(target, {"fingerprint": fingerprint(cfg), "config": to_mapping(cfg)})
    return target
