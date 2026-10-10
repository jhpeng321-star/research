"""Model construction for the supported model kinds."""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

from wms.physics.config import PhysicsConfig

from .residual_net import HdvConfig
from .spec import ModelSpec
from .wms import WmSModel

MODEL_KINDS = ("structured", "blackbox_traj", "blackbox_id")


def build_model(spec: Optional[ModelSpec] = None, physics: Optional[PhysicsConfig] = None,
                *, kind: str = "structured", hdv: Optional[HdvConfig] = None,
                action_conditioned: bool = True) -> WmSModel:
    """Build the WM-S model of the requested ``kind``.

    ``structured`` is the physics-informed model; ``blackbox_id`` the same class with
    controller-id-conditioned learned CAVs; ``blackbox_traj`` the physics-free legacy
    baseline (``action_conditioned=False`` rebuilds its pre-switch layout).
    """
    spec = spec or ModelSpec()
    if kind == "structured":
        return WmSModel(spec, physics=physics, hdv=hdv)
    if kind == "blackbox_id":
        return WmSModel(replace(spec, black_box=True), physics=physics, hdv=hdv)
    if kind == "blackbox_traj":
        if spec.long_history:
            raise ValueError("long history requires structured WM-S")
        from wms.legacy.blackbox.traj import BlackBoxTrajModel
        return BlackBoxTrajModel(
            physics=physics, hdv=hdv, history_features=spec.history_features,
            history_len=spec.history_len, history_hidden=spec.history_hidden,
            k_hidden=spec.k_hidden, action_conditioned=action_conditioned)
    raise ValueError(f"unknown model kind {kind!r}; choose from {', '.join(MODEL_KINDS)}")
