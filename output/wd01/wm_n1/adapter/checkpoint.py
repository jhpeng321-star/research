"""WM-N1 ensemble checkpoint I/O (plain ``state_dict`` + JSON config).

The adapter is coded to the checkpoint format the ``train`` piece is frozen
to (INTERFACES.md "Torch-2.5 notes"): a plain ``state_dict`` per member plus a
JSON config, never a pickled module.  ``torch.load`` is always called with an
**explicit** ``weights_only=...`` so every supported Torch (>=2.7, where the
default is ``True``) agrees; the safe value ``True`` is used everywhere
(only tensors and primitive config values are stored).

Two on-disk layouts are accepted by :func:`load_ensemble`:

* **directory** ``ckpt/`` with ``member_<k>.pt`` (``k = 1..n_members``) and
  the config either in ``config.json`` (this module's
  :func:`save_ensemble`) or, as ``train.Trainer`` writes it, in the
  ``member_<k>.json`` sidecar / member payload (flat ``TrainConfig``); and
* **single file** ``ckpt.pt`` holding ``{"config": <json str>, "members":
  [<state_dict>, ...]}`` saved with :func:`save_ensemble_file`.

The ``train.Trainer`` member payload nests the tensor weights under
``"state_dict"`` (next to ``optimizer``/``epoch``/``seed`` for resume); this
loader accepts both a nested payload and a plain ``state_dict``.

The config carries the architecture (``WmN1Config`` fields under ``"arch"``,
or flat at the top level for a ``TrainConfig``), the member seeds, and optional
OOF bookkeeping used by the export guards: ``"train_worlds"`` (every member saw
these worlds) and ``"member_train_worlds"`` (``{member_index_1based:
[world, ...]}``).
"""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import torch

from model.ensemble import DEFAULT_SEEDS, WmN1Ensemble
from model.net import WmN1Config

#: ``WmN1Config`` fields, in dataclass order.
ARCH_KEYS = tuple(f.name for f in fields(WmN1Config))

MEMBER_GLOB = "member_*.pt"


def ensemble_config(ensemble: WmN1Ensemble, **extra) -> dict:
    """Build the JSON config for ``ensemble`` (arch + seeds + extras)."""
    cfg = ensemble.cfg
    conf = {
        "arch": {k: getattr(cfg, k) for k in ARCH_KEYS},
        "seeds": [int(s) for s in ensemble.seeds],
        "n_members": len(ensemble.seeds),
    }
    conf.update(extra)
    return conf


def _state_only(module) -> dict:
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def save_ensemble(path, ensemble: WmN1Ensemble, **config_extra) -> dict:
    """Write the directory layout; return a receipt dict."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    conf = ensemble_config(ensemble, **config_extra)
    (path / "config.json").write_text(
        json.dumps(conf, indent=1), encoding="utf-8")
    for i, member in enumerate(ensemble.members, start=1):
        torch.save(_state_only(member), path / f"member_{i}.pt")
    return dict(path=str(path), layout="directory", config=conf,
                n_members=len(ensemble.members))


def save_ensemble_file(path, ensemble: WmN1Ensemble, **config_extra) -> dict:
    """Write the single-file layout; return a receipt dict."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conf = ensemble_config(ensemble, **config_extra)
    torch.save({"config": json.dumps(conf),
                "members": [_state_only(m) for m in ensemble.members]}, path)
    return dict(path=str(path), layout="file", config=conf,
                n_members=len(ensemble.members))


def _normalise_config(conf) -> dict:
    """Wrap a flat ``train.TrainConfig`` config under the adapter ``"arch"``.

    The adapter's own layouts store the architecture under ``"arch"``; a
    ``train.Trainer`` sidecar/payload stores the flat ``TrainConfig.to_dict()``
    (architecture fields at the top level).  Both are accepted.
    """
    if not isinstance(conf, dict):
        raise ValueError(
            f"checkpoint config must be a mapping, got {type(conf).__name__}")
    if "arch" in conf:
        return conf
    arch = {k: conf[k] for k in ARCH_KEYS if k in conf}
    if not arch:
        raise ValueError(
            "checkpoint config has neither 'arch' nor WmN1Config fields")
    out = dict(conf)
    out["arch"] = arch
    if "n_members" not in out:
        out["n_members"] = len(out.get("seeds", DEFAULT_SEEDS))
    return out


def _member_state(obj):
    """Return the plain ``state_dict`` of a train payload or a state_dict."""
    if isinstance(obj, dict) and isinstance(obj.get("state_dict"), dict):
        return obj["state_dict"]
    return obj


def _config_from_obj(conf) -> dict:
    conf = json.loads(conf) if isinstance(conf, str) else dict(conf)
    return _normalise_config(conf)


def _trainer_dir_config(path: Path) -> dict:
    """Read a ``train.Trainer`` member config from the directory sidecars."""
    sidecar = path / "member_1.json"
    if sidecar.is_file():
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "config" in data:
            return _config_from_obj(data["config"])
    files = sorted(path.glob(MEMBER_GLOB),
                   key=lambda p: int(p.stem.split("_")[1]))
    for member_file in files:
        blob = torch.load(member_file, map_location="cpu", weights_only=True)
        if isinstance(blob, dict) and "config" in blob:
            return _config_from_obj(blob["config"])
        break
    raise ValueError(
        f"checkpoint directory {path} has no config.json, member_*.json or "
        "member payload config")


def _read_config(path: Path) -> dict:
    if path.is_dir():
        cfg_path = path / "config.json"
        if cfg_path.is_file():
            return _normalise_config(
                json.loads(cfg_path.read_text(encoding="utf-8")))
        return _trainer_dir_config(path)
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(blob, dict) and "config" in blob:
        return _config_from_obj(blob["config"])
    raise ValueError(f"checkpoint {path} has no config")


def _load_states(path: Path) -> list:
    if path.is_dir():
        files = sorted(
            path.glob(MEMBER_GLOB),
            key=lambda p: int(p.stem.split("_")[1]))
        return [_member_state(torch.load(f, map_location="cpu",
                                         weights_only=True))
                for f in files]
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(blob, dict) and blob.get("members"):
        return list(blob["members"])
    if isinstance(blob, dict) and "state_dict" in blob:
        return [blob["state_dict"]]
    raise ValueError(f"checkpoint {path} has no member state_dict(s)")


def build_ensemble(config: dict) -> WmN1Ensemble:
    """Build an (untrained) ensemble from a config dict."""
    arch = config.get("arch", {})
    cfg = WmN1Config(**{k: arch[k] for k in ARCH_KEYS if k in arch})
    seeds = tuple(int(s) for s in config.get("seeds", DEFAULT_SEEDS))
    return WmN1Ensemble(cfg=cfg, seeds=seeds)


def load_ensemble(path, *, device: str = "cpu"):
    """Load ``(ensemble, config)`` from the directory or single-file layout."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    conf = _read_config(path)
    ensemble = build_ensemble(conf)
    states = _load_states(path)
    if len(states) != len(ensemble.members):
        raise ValueError(
            f"checkpoint {path}: {len(states)} member state_dict(s) vs "
            f"{len(ensemble.members)} seeds")
    for member, state in zip(ensemble.members, states):
        member.load_state_dict(state, strict=True)
    ensemble.to(device)
    ensemble.eval()
    return ensemble, conf
