"""The model-side dependencies of a rollout, bundled so the predictor stays model-agnostic.

:class:`WmsPredictor` needs a loaded network and a handful of pure helpers that live in
other sub-packages (history featurisation, junction-carry presence, action timelines,
entry processes).  They are collected in :class:`RolloutTools` and passed in, so the
evaluation package never imports training code and can be driven by any object that
honours the network contract:

``model.rollout(state, n_steps=..., k0=0, cav_teacher=..., present_seq=..., junction_seq=...,
cav_controller=..., entries=..., driver_history=..., driver_history_mask=...,
action_timeline=..., projection=...) -> {"positions","speeds","lanes","masks","state"}``,
plus ``model.hold_teacher`` and (optional) ``history_len`` / ``driver_history_len`` /
``history_s`` / ``history_stride_s``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class RolloutTools:
    """Pure helpers the rollout input builders call (see :func:`default_tools`)."""

    junction_carry_presence: Callable[[Any, Any], tuple[np.ndarray, np.ndarray]]
    history_spec: Callable[..., tuple[int, int, int]]
    history_step_features: Callable[[Any, Any], Any]
    long_history_features: Callable[..., tuple[Any, Any]]
    action_timeline: type
    controllers: Sequence[str]
    #: module-like object exposing ``EntryConfig``, ``TorchEntryProcess``,
    #: ``TorchEntryProcessV2``, ``observed_rates``, ``build_entrance_history`` and
    #: ``read_insertion_history``; ``None`` makes entries unavailable.
    entries: ModuleType | Any | None = None
    #: ``(base_run_dir, t0_s) -> float32 [2, 30]`` counter reader for states that carry
    #: ``base_run_dir`` instead of an inline ``counter``.
    counter_reader: Callable[[Path, float], np.ndarray] | None = None


@dataclass(frozen=True)
class WmsBackend:
    """A loaded network, its flat training config and the rollout helpers."""

    model: Any
    config: Mapping[str, Any]
    tools: RolloutTools
    projection: Any | None = None


def default_tools() -> RolloutTools:
    """Resolve the helpers from the clean package layout (``data`` and ``model``)."""
    from wms.data import history as history_mod
    from wms.data.action_timeline import CONTROLLERS, ActionTimeline
    from wms.data.presence import junction_carry_presence
    from wms.model import entries

    return RolloutTools(
        junction_carry_presence=junction_carry_presence,
        history_spec=history_mod.history_spec,
        history_step_features=history_mod.history_step_features,
        long_history_features=history_mod.long_history_features,
        action_timeline=ActionTimeline,
        controllers=CONTROLLERS,
        entries=entries,
    )


def feasibility_projection(config: Mapping[str, Any]) -> Any | None:
    """The run-2 feasibility projection a checkpoint was trained with, or ``None``."""
    if not config.get("projection_enabled"):
        return None
    from wms.model.stability import FeasibilityProjection, ProjectionConfig

    return FeasibilityProjection(
        ProjectionConfig(min_gap_m=float(config.get("projection_min_gap_m", 2.0))))


def load_backend(checkpoint: str | Path, *, device: str = "cpu",
                 physics: Mapping[str, Any] | None = None,
                 cav_exec_cap: str | None = None) -> WmsBackend:
    """Load a ``wm_s/1`` checkpoint through ``wms.model.checkpoint`` (strict).

    The stored flat config is parsed with ``TrainConfig.from_legacy_flat`` (so a checkpoint
    whose config the schema cannot read is refused here, not mid-rollout) and kept verbatim
    as ``WmsBackend.config``.  ``physics`` overrides the physics switches normally read from
    ``physics.json`` / ``TRAIN_RECEIPT.json`` next to the checkpoint.

    ``cav_exec_cap`` sets the model's runtime execution-cap switch (EXECCAP_NOTES.md); ``None``
    reads the process default from the ``WMS_CAV_EXEC_CAP`` environment variable ("off" if unset),
    so every imagination tool that loads its WM here follows one flag.
    """
    import os

    from wms.config.train import TrainConfig
    from wms.model.cav import EXEC_CAP_ENV, check_exec_cap_mode, set_exec_cap
    from wms.model.checkpoint import load_checkpoint

    mode = check_exec_cap_mode(cav_exec_cap if cav_exec_cap is not None
                               else os.environ.get(EXEC_CAP_ENV, "off"))
    loaded = load_checkpoint(checkpoint, device=device, strict=True, physics=physics,
                             config_factory=TrainConfig.from_legacy_flat)
    set_exec_cap(loaded.model, mode)
    config = dict(loaded.legacy_config)
    return WmsBackend(model=loaded.model.eval(), config=config, tools=default_tools(),
                      projection=loaded.projection or feasibility_projection(config))
