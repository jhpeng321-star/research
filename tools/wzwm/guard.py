"""Fail-closed guards of the wz3to2 fine-tune: no driver truth as an input, no restricted paths.

Driver settings are learned by the driver encoder from prediction error only.  The guards refuse
(a) a resolved training config that would feed per-vehicle driver truth or SUMO vType parameters
(driver sidecar, switching driver_params, physics_params file, ``driver_params`` rollout mode),
(b) any configured path / data source naming truth or restricted material, and
(c) a training batch carrying a truth-like key (checked on real batches by ``probe_batch``).
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

RESTRICTED = re.compile(
    r"(?:^|[^a-z0-9])(sealed|holdout|heldout|held_out|cal_split[a-z0-9_]*|calibration|chunk[-_]?5|j4b|inc1)"
    r"(?:$|[^a-z0-9])")
TRUTH = re.compile(r"(hetdata|truth|vtype|v_type|driver_class|type_id|vehicle_params|driver_params"
                   r"|driver_type|true_type|oracle)")


class GuardError(RuntimeError):
    """A forbidden input (truth label or restricted path) was configured or observed."""


def path_problems(text: str) -> list[str]:
    low = str(text).replace("\\", "/").lower()
    out = []
    m = RESTRICTED.search(low)
    if m:
        out.append(f"restricted token {m.group(1)!r} in {text!r}")
    m = TRUTH.search(low)
    if m:
        out.append(f"driver-truth token {m.group(1)!r} in {text!r}")
    return out


def check_path(text: str) -> str:
    p = path_problems(text)
    if p:
        raise GuardError("; ".join(p))
    return str(text)


def _strings(obj: Any, where: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            yield from _strings(v, f"{where}.{k}" if where else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _strings(v, f"{where}[{i}]")
    elif isinstance(obj, str):
        yield where, obj


def config_problems(cfg: Mapping) -> list[str]:
    """Problems in a resolved TrainConfig mapping (``to_mapping`` / config.resolved.json 'config')."""
    model, data = cfg.get("model", {}), cfg.get("data", {})
    out = []
    if model.get("physics_params"):
        out.append("model.physics_params set (per-type physics parameters are not learned)")
    if model.get("rollout_mode") == "driver_params":
        out.append("model.rollout_mode = driver_params")
    if model.get("driver_param_init", "checkpoint") not in ("checkpoint", "centre"):
        out.append(f"model.driver_param_init = {model.get('driver_param_init')!r}")
    if (data.get("sidecars") or {}).get("driver_sidecar"):
        out.append("data.sidecars.driver_sidecar set (driver truth sidecar)")
    if (data.get("switching") or {}).get("driver_params"):
        out.append("data.switching.driver_params set (driver truth)")
    prior = (cfg.get("loss") or {}).get("driver_prior") or {}
    if prior.get("weight", 0) and str(prior.get("target", "")).lower() not in ("textbook", "population"):
        out.append(f"loss.driver_prior target {prior.get('target')!r} with nonzero weight")
    for where, text in _strings(cfg):
        if where.startswith(("name",)):
            continue
        out.extend(f"{where}: {p}" for p in path_problems(text))
    return out


def check_config(cfg: Mapping) -> None:
    p = config_problems(cfg)
    if p:
        raise GuardError("fine-tune config refused: " + "; ".join(p))


def batch_problems(keys: Iterable[str]) -> list[str]:
    """Truth-like keys among the record/batch keys handed to the model."""
    return [f"batch key {k!r} looks like driver truth" for k in keys if TRUTH.search(str(k).lower())]


def check_batch(record: Mapping) -> None:
    keys = list(record)
    for k, v in record.items():
        if isinstance(v, Mapping):
            keys.extend(f"{k}.{kk}" for kk in v)
    p = batch_problems(keys)
    if p:
        raise GuardError("; ".join(p))
