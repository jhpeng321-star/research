"""Validated per-CAV, per-control-step actions for WMS rollouts.

Schema ``wms_action_timeline/1`` stores a categorical controller id for every
1 s step and optional issued acceleration / target speed arrays.  Missing
timeline data is represented by a constant timeline when adapting legacy packs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

CONTROLLERS = ("native", "damping", "pi-external-u", "direct")
SCHEMA = "wms_action_timeline/1"


@dataclass(frozen=True)
class ActionTimeline:
    controller: np.ndarray
    acceleration_mps2: np.ndarray
    target_speed_mps: np.ndarray
    control_dt_s: float = 1.0

    def __post_init__(self):
        ids = np.asarray(self.controller)
        accel = np.asarray(self.acceleration_mps2)
        target = np.asarray(self.target_speed_mps)
        if ids.ndim != 2 or accel.shape != ids.shape or target.shape != ids.shape:
            raise ValueError("timeline fields must share [control_steps, CAVs] shape")
        if ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= len(CONTROLLERS)):
            raise ValueError("controller ids must index the supported controller vocabulary")
        if not np.isfinite(self.control_dt_s) or self.control_dt_s <= 0:
            raise ValueError("control_dt_s must be finite and positive")
        if not np.isfinite(accel).all() or not np.isfinite(target).all():
            raise ValueError("issued commands must be finite; use 0 for unused commands")
        direct = ids == CONTROLLERS.index("direct")
        if direct.any() and not np.isfinite(accel[direct] + target[direct]).all():
            raise ValueError("direct actions require finite acceleration and target speed")

    @property
    def has_switch(self) -> bool:
        return bool(np.any(self.controller[1:] != self.controller[:-1]))

    @classmethod
    def constant(cls, controller: str | int, steps: int, cavs: int, *,
                 control_dt_s: float = 1.0) -> "ActionTimeline":
        index = CONTROLLERS.index(controller) if isinstance(controller, str) else int(controller)
        if index < 0 or index >= len(CONTROLLERS) or steps < 1 or cavs < 1:
            raise ValueError("invalid constant timeline dimensions/controller")
        return cls(np.full((steps, cavs), index, dtype=np.int64),
                   np.zeros((steps, cavs), dtype=np.float32),
                   np.zeros((steps, cavs), dtype=np.float32), control_dt_s)

    @classmethod
    def from_dict(cls, data: Mapping) -> "ActionTimeline":
        if data.get("schema") != SCHEMA:
            raise ValueError(f"unsupported action timeline schema: {data.get('schema')!r}")
        labels = np.asarray(data["controller"])
        if labels.dtype.kind in "USO":
            labels = np.vectorize(CONTROLLERS.index, otypes=[np.int64])(labels)
        ids = np.asarray(labels, dtype=np.int64)
        if np.any(ids == CONTROLLERS.index("direct")) and not {
                "issued_acceleration_mps2", "issued_target_speed_mps"}.issubset(data):
            raise ValueError("direct actions require both issued command fields")
        return cls(ids,
                   np.asarray(data.get("issued_acceleration_mps2", 0), dtype=np.float32)
                   + np.zeros_like(labels, dtype=np.float32),
                   np.asarray(data.get("issued_target_speed_mps", 0), dtype=np.float32)
                   + np.zeros_like(labels, dtype=np.float32),
                   float(data.get("control_dt_s", 1.0)))

    def to_dict(self) -> dict:
        return {"schema": SCHEMA, "control_dt_s": self.control_dt_s,
                "controller": np.asarray(CONTROLLERS, dtype=object)[self.controller].tolist(),
                "issued_acceleration_mps2": self.acceleration_mps2.tolist(),
                "issued_target_speed_mps": self.target_speed_mps.tolist()}
