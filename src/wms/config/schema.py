"""Generic helpers for typed, strictly validated configuration dataclasses.

Config classes are plain ``@dataclass(frozen=True)`` types whose field annotations
drive :func:`from_mapping` (nested dicts -> dataclasses, with unknown-key and type
checks) and :func:`to_mapping` (the inverse, JSON/TOML friendly).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import typing
from typing import Any, Mapping, TypeVar

from ..core.errors import ConfigError

T = TypeVar("T")
_NoneType = type(None)


def config_dataclass(cls: type[T]) -> type[T]:
    """Class decorator: frozen, slotted dataclass."""
    return dataclasses.dataclass(frozen=True, slots=True)(cls)


def from_mapping(cls: type[T], data: Mapping[str, Any], path: str = "") -> T:
    """Build dataclass ``cls`` from nested ``data``; unknown keys are an error."""
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path or cls.__name__}: expected a table, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ConfigError(f"{path or cls.__name__}: unknown keys {unknown}; valid: {sorted(names)}")
    kwargs = {name: _coerce(hints[name], value, f"{path}.{name}".lstrip("."))
              for name, value in data.items()}
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"{path or cls.__name__}: {exc}") from exc


def to_mapping(obj: Any) -> Any:
    """Recursively convert dataclasses/tuples to plain dicts/lists."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_mapping(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {str(k): to_mapping(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_mapping(v) for v in obj]
    return obj


def canonical_json(obj: Any) -> str:
    """Deterministic JSON text of ``to_mapping(obj)`` (sorted keys, no spaces)."""
    return json.dumps(to_mapping(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(obj: Any) -> str:
    """SHA-256 of the canonical JSON of a config."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge nested tables; values in ``override`` win, tables merge recursively."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def require(condition: bool, message: str) -> None:
    """Raise :class:`ConfigError` unless ``condition`` holds."""
    if not condition:
        raise ConfigError(message)


def is_finite(value: Any) -> bool:
    """True for real numbers (not bool) that are finite."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _coerce(tp: Any, value: Any, path: str) -> Any:
    """Convert ``value`` to annotation ``tp`` or raise :class:`ConfigError`."""
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin is typing.Union:
        if value is None and _NoneType in args:
            return None
        errors = []
        for option in args:
            if option is _NoneType:
                continue
            try:
                return _coerce(option, value, path)
            except ConfigError as exc:
                errors.append(str(exc))
        raise ConfigError("; ".join(errors))
    if origin is typing.Literal:
        if value not in args:
            raise ConfigError(f"{path}: {value!r} not one of {list(args)}")
        return value
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected an array")
        item = args[0] if args else Any
        return tuple(_coerce(item, v, f"{path}[{i}]") for i, v in enumerate(value))
    if origin in (dict, Mapping) or tp is dict:
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path}: expected a table")
        key_tp, val_tp = (args if args else (str, Any))
        return {_coerce(key_tp, k, path): _coerce(val_tp, v, f"{path}.{k}") for k, v in value.items()}
    if dataclasses.is_dataclass(tp):
        if isinstance(value, tp):
            return value
        return from_mapping(tp, value, path)
    return _coerce_scalar(tp, value, path)


def _coerce_scalar(tp: Any, value: Any, path: str) -> Any:
    """Strict scalar conversion: ints widen to float, nothing else is guessed."""
    if tp is Any:
        return value
    if tp is bool:
        if isinstance(value, bool):
            return value
    elif tp is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif tp is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    elif tp is str:
        if isinstance(value, str):
            return value
    else:  # pragma: no cover - guards against unsupported annotations in new configs
        raise ConfigError(f"{path}: unsupported annotation {tp!r}")
    raise ConfigError(f"{path}: expected {tp.__name__}, got {value!r}")
