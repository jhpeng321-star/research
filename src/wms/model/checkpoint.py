"""Checkpoint load/save in the frozen ``wm_s/1`` format.

The payload is ``{"format": "wm_s/1", "model": state_dict, "config": flat training
config, ...}`` and is read with ``weights_only=True``.  State-dict key names are
frozen; no key is renamed on load.
"""
from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import torch

from wms.physics.config import PhysicsConfig
from wms.physics.projection import FeasibilityProjection, ProjectionConfig

from .factory import MODEL_KINDS, build_model
from .spec import ModelSpec
from .wms import WmSModel

CHECKPOINT_FORMAT = "wm_s/1"
#: Weights that checkpoints written before the corresponding feature legitimately lack.
_OPTIONAL_PREFIXES = ("driver_encoder.", "command_proj.",
                      "residual_net.courtesy_proj.", "merge_negotiation.",
                      "residual_net.residual_head_log_scale")
#: V10COURTESY: per-block attention bias keys (``residual_net.blocks.N.attn.``).
_OPTIONAL_CONTAINS = (".courtesy_bias.",)
#: LANERATECAL: closed-loop per-head rate calibration stored next to a checkpoint
#: (``tools/lane_rate_calib.py``).  Absent file = no-op (bit-identical load).
LANE_CALIB_FILE = "lane_calib.json"
LANE_CALIB_FORMAT = "lane_rate_calib/1"
LANE_CALIB_HEADS = ("overtake", "return", "merge")


class CheckpointError(RuntimeError):
    """The file is not a loadable ``wm_s/1`` checkpoint for the requested model."""


@dataclass
class LoadedWms:
    """A loaded model with the metadata of its checkpoint.

    ``train_cfg`` is ``config_factory(legacy_config)`` when a factory was given
    (typically ``TrainConfig.from_legacy_flat``), else ``None``; ``legacy_config`` is
    the flat config stored in the checkpoint.  ``projection`` is set when the run was
    trained with the feasibility projection.
    """

    model: WmSModel
    train_cfg: Any
    payload_meta: dict
    sha256: str
    legacy_config: dict = field(default_factory=dict)
    projection: Optional[FeasibilityProjection] = None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_payload(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        found = payload.get("format") if isinstance(payload, dict) else type(payload).__name__
        raise CheckpointError(f"unsupported checkpoint format: {found}")
    if "model" not in payload:
        raise CheckpointError("checkpoint has no model state_dict")
    return payload


def _physics_config(path: Path, config: Mapping[str, Any],
                    override: Optional[Mapping[str, Any]]) -> PhysicsConfig:
    """Physics the model was trained with: override, embedded block, sibling files.

    A run's checkpoints do not carry their physics switches (``merge_cal`` ...); the trainer
    writes ``physics.json`` next to them and the TRAIN receipt of the same run holds the same
    block, which is how the legacy C1/R3 runs are found.
    """
    raw = override or config.get("physics")
    if raw is None:
        sibling = path.with_name("physics.json")
        if sibling.is_file():
            raw = json.loads(sibling.read_text(encoding="utf-8"))
    if raw is None:
        receipt = path.with_name("TRAIN_RECEIPT.json")
        if receipt.is_file():
            raw = json.loads(receipt.read_text(encoding="utf-8")).get("physics")
    if not raw:
        # A checkpoint copied without its run's physics.json / TRAIN_RECEIPT.json silently scored with
        # default physics (MOBIL lanes, no SUMO fidelity) on 2026-10-13; make the fallback loud.
        warnings.warn(f"{path}: no physics block, physics.json or TRAIN_RECEIPT.json found; using "
                      "default PhysicsConfig() (lane_rule=mobil). Copy the run's physics.json next to "
                      "the checkpoint.", RuntimeWarning, stacklevel=2)
        return PhysicsConfig()
    allowed = set(PhysicsConfig.__dataclass_fields__)
    return PhysicsConfig(**{key: value for key, value in raw.items() if key in allowed})


def _model_spec(config: Mapping[str, Any], kind: str) -> ModelSpec:
    spec = ModelSpec.from_mapping(config)
    if kind == "blackbox_id":
        return ModelSpec(history_features=spec.history_features, black_box=True,
                         n_controllers=int(config.get("n_controllers", 6)),
                         controller_embed_dim=spec.controller_embed_dim)
    if kind == "blackbox_traj":
        return ModelSpec(history_features=spec.history_features)
    if config.get("history_s") is not None and kind != "structured":
        raise CheckpointError("long history is supported only for structured checkpoints")
    # The structured model ignores any black-box flag of the stored config.
    return ModelSpec(**{**spec.__dict__, "black_box": False})


def read_lane_rate_calib(path: str | Path) -> tuple[dict, tuple[float, float, float]]:
    """Read and validate a ``lane_calib.json`` (format :data:`LANE_CALIB_FORMAT`).

    ``offsets`` may be a length-3 list (overtake, return, merge) or a mapping by head; the
    canonical three-value vector is returned with the document.
    """
    path = Path(path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CheckpointError(f"{path}: unreadable lane calibration: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("format") != LANE_CALIB_FORMAT:
        raise CheckpointError(f"{path}: not a {LANE_CALIB_FORMAT} file")
    raw = doc.get("offset_vector")
    if raw is None:
        raw = doc.get("offsets")
        if isinstance(raw, Mapping):
            raw = [raw.get(name, 0.0) for name in LANE_CALIB_HEADS]
    if not isinstance(raw, (list, tuple)) or len(raw) != len(LANE_CALIB_HEADS):
        raise CheckpointError(f"{path}: calibration needs {len(LANE_CALIB_HEADS)} offsets")
    try:
        vector = tuple(float(x) for x in raw)
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"{path}: non-numeric calibration offset") from exc
    if not all(math.isfinite(x) for x in vector):
        raise CheckpointError(f"{path}: calibration offsets must be finite")
    return doc, vector  # type: ignore[return-value]


def apply_lane_rate_calib(model: WmSModel, path: str | Path,
                          sha256: str | None = None) -> dict | None:
    """Apply ``lane_calib.json`` next to ``path`` (no file = no-op, returns ``None``).

    The calibrated offsets are additive on top of the config's ``lane_logit_offsets``.  A
    stored ``checkpoint_sha256`` is enforced so a calibration never transfers to another
    checkpoint that happens to share the directory (LANE_REDESIGN: calibrate per checkpoint).
    """
    calib_path = Path(path).with_name(LANE_CALIB_FILE)
    if not calib_path.is_file():
        return None
    doc, vector = read_lane_rate_calib(calib_path)
    expected = doc.get("checkpoint_sha256")
    if expected:
        digest = sha256 or sha256_file(path)
        if str(expected).lower() != digest.lower():
            raise CheckpointError(
                f"{calib_path} was fitted for checkpoint sha256 {expected}, not {digest}")
    base = tuple(float(x) for x in getattr(model, "lane_logit_offsets", (0.0, 0.0, 0.0)))
    if len(base) != len(vector):
        base = (0.0,) * len(vector)
    model.lane_logit_offsets = tuple(a + b for a, b in zip(base, vector))
    model.lane_rate_calib = {**doc, "path": str(calib_path),
                             "offset_vector": list(vector)}
    return model.lane_rate_calib


def _check_keys(result, config: Mapping[str, Any], strict: bool) -> None:
    uses_encoder = (str(config.get("rollout_mode", "residual")) != "residual"
                    or str(config.get("lane_head", "off")) != "off"
                    or config.get("history_s") is not None)
    optional = tuple(p for p in _OPTIONAL_PREFIXES
                     if not (uses_encoder and p == "driver_encoder."))
    missing = [key for key in result.missing_keys
               if not key.startswith(optional)
               and not any(part in key for part in _OPTIONAL_CONTAINS)]
    if missing:
        raise CheckpointError(f"checkpoint is missing network weights: {missing[:5]}")
    if strict and result.unexpected_keys:
        raise CheckpointError(f"checkpoint has unexpected weights: {result.unexpected_keys[:5]}")


def load_checkpoint(path: str | Path, *, device: str = "cpu", strict: bool = True,
                    model_kind: str = "structured",
                    physics: Optional[Mapping[str, Any]] = None,
                    config_factory: Optional[Callable[[Mapping[str, Any]], Any]] = None
                    ) -> LoadedWms:
    """Load a ``wm_s/1`` checkpoint into an evaluation-mode model.

    Every network weight must be present except those of features newer than the
    checkpoint (``driver_encoder.*`` for residual models, ``command_proj.*``); with
    ``strict`` unexpected keys are errors too.  ``physics`` overrides the physics
    stored with the run.
    """
    if model_kind not in MODEL_KINDS:
        raise CheckpointError(
            f"unknown model kind {model_kind!r}; choose from {', '.join(MODEL_KINDS)}")
    path = Path(path)
    payload = _read_payload(path)
    config = dict(payload.get("config") or {})
    weights = payload["model"]
    model = build_model(_model_spec(config, model_kind), _physics_config(path, config, physics),
                        kind=model_kind,
                        action_conditioned="controller_embed.weight" in weights)
    _check_keys(model.load_state_dict(weights, strict=False), config, strict)
    calib = payload.get("lane_v5_merge_calib")
    if calib is not None:
        # MERGECAL (tools/merge_calib.py): per-position merge-head offsets of a calibrated copy.
        from . import lane_v5 as lv5
        if getattr(model, "lane_v5", None) is None:
            raise CheckpointError("lane_v5_merge_calib needs a hazard_v5 lane model")
        model.lane_v5_merge_calib = lv5.check_merge_calib(
            (calib["edges_m"], calib["offsets"]))
    digest = None
    if path.with_name(LANE_CALIB_FILE).is_file():
        # LANERATECAL (tools/lane_rate_calib.py): closed-loop per-head rate offsets.  This is
        # a no-op when the file is absent; the calibrated checkpoint is the same .pt file.
        digest = sha256_file(path)
        apply_lane_rate_calib(model, path, digest)
    model.to(device)
    model.eval()
    projection = None
    if config.get("projection_enabled"):
        projection = FeasibilityProjection(
            ProjectionConfig(min_gap_m=float(config.get("projection_min_gap_m", 2.0))))
    meta = {key: value for key, value in payload.items() if key not in ("model", "config")}
    return LoadedWms(model=model,
                     train_cfg=None if config_factory is None else config_factory(config),
                     payload_meta=meta, sha256=digest or sha256_file(path),
                     legacy_config=config, projection=projection)


def save_checkpoint(path: str | Path, model: WmSModel, cfg: Any, *,
                    history: Optional[list] = None, val_by_stage: Optional[list] = None,
                    extra: Optional[Mapping[str, Any]] = None) -> None:
    """Write a ``wm_s/1`` checkpoint.

    ``cfg`` is a flat dict or an object with ``to_legacy_flat()``/``to_dict()``.
    ``extra`` adds stage metadata (``stage_s``, ``step``, ``val``) to the payload.
    """
    if isinstance(cfg, Mapping):
        flat = dict(cfg)
    elif hasattr(cfg, "to_legacy_flat"):
        flat = cfg.to_legacy_flat()
    else:
        flat = cfg.to_dict()
    # Persist the actual encoder/update semantics even with a minimal caller config.
    # Default keys stay absent, preserving the legacy checkpoint config layout.
    for key, default in (("driver_param_mode", "legacy"), ("driver_ema_update", True)):
        value = getattr(getattr(model, "spec", None), key, default)
        if value != default:
            flat[key] = value
        else:
            flat.pop(key, None)
    payload: dict[str, Any] = {"format": CHECKPOINT_FORMAT, "model": model.state_dict(),
                               "config": flat}
    if history is not None:
        payload["history"] = history
    if val_by_stage is not None:
        payload["val_by_stage"] = val_by_stage
    payload.update(extra or {})
    torch.save(payload, Path(path))
