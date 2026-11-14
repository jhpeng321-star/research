"""Validated per-CAV, per-control-step actions for WMS rollouts.

Schema ``wms_action_timeline/1`` stores a categorical controller id for every
1 s step and optional issued acceleration / target speed arrays.  Missing
timeline data is represented by a constant timeline when adapting legacy packs.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
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


def load_timeline(schedule_path: str | Path, command_log_path: str | Path | None = None,
                  *, cav_ids: list[str] | None = None) -> ActionTimeline:
    """Load switch schedule + per-step issued-command JSON/JSONL files.

    Schedule entries are ``{step, vehicle_id, controller}``; command rows use
    ``{step, vehicle_id, issued_acceleration_mps2, issued_target_speed_mps}``.
    The schedule declares ``vehicle_ids`` in stable slot order.
    """
    schedule = json.loads(Path(schedule_path).read_text(encoding="utf-8"))
    if schedule.get("schema") not in (SCHEMA, "wms_switch_schedule/1"):
        raise ValueError("unsupported switch schedule schema")
    roster = list(cav_ids or schedule["vehicle_ids"])
    n_steps = int(schedule["control_steps"])
    if n_steps < 1 or len(set(roster)) != len(roster) or not roster:
        raise ValueError("schedule needs positive control_steps and unique vehicle ids")
    ids = np.full((n_steps, len(roster)), CONTROLLERS.index("native"), dtype=np.int64)
    for item in schedule.get("initial", []):
        try:
            slot = roster.index(str(item["vehicle_id"]))
            action = CONTROLLERS.index(str(item["controller"]))
        except ValueError as exc:
            raise ValueError("initial action references an unknown vehicle/controller") from exc
        ids[:, slot] = action
    for event in schedule.get("switches", []):
        step = int(event["step"])
        if not 0 <= step < n_steps:
            raise ValueError("switch step is outside the timeline")
        try:
            slot = roster.index(str(event["vehicle_id"]))
            action = CONTROLLERS.index(str(event["controller"]))
        except ValueError as exc:
            raise ValueError("switch references an unknown vehicle/controller") from exc
        ids[step:, slot] = action
    accel = np.zeros_like(ids, dtype=np.float32)
    target = np.zeros_like(ids, dtype=np.float32)
    supplied = np.zeros_like(ids, dtype=bool)
    if command_log_path is not None:
        path = Path(command_log_path)
        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
        else:
            rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            try:
                step, slot = int(row["step"]), roster.index(str(row["vehicle_id"]))
            except (KeyError, ValueError) as exc:
                raise ValueError("command row references an unknown vehicle or lacks a step") from exc
            if not 0 <= step < n_steps or ids[step, slot] != CONTROLLERS.index("direct"):
                raise ValueError("command row must align with a direct timeline action")
            if supplied[step, slot]:
                raise ValueError("duplicate direct command row")
            try:
                accel[step, slot] = float(row["issued_acceleration_mps2"])
                target[step, slot] = float(row["issued_target_speed_mps"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("direct command row requires numeric acceleration and target speed") from exc
            supplied[step, slot] = True
    direct = ids == CONTROLLERS.index("direct")
    if np.any(direct & ~supplied):
        raise ValueError("every direct timeline action requires one issued-command row")
    return ActionTimeline(ids, accel, target, float(schedule.get("control_dt_s", 1.0)))


class ControllerStateSemantics:
    """Shared lifecycle contract: damping resumes; PI is reset on re-entry.

    The simulator creates one damping controller per CAV for the episode.  PI's
    PISession invalidates its state on native handback and resets on activation.
    This small tracker exposes those transitions without owning plugin state.
    """

    def __init__(self):
        self._previous: dict[str, str] = {}
        self.damping_instances: dict[str, object] = {}

    def transition(self, vehicle_id: str, controller: str, *, damping_factory=None) -> dict:
        if controller not in CONTROLLERS:
            raise ValueError(f"unsupported controller: {controller}")
        previous = self._previous.get(vehicle_id)
        events = {"controller": controller, "damping_resume": False,
                  "pi_reset": False, "native_release": controller == "native"}
        if controller == "damping":
            if vehicle_id not in self.damping_instances:
                self.damping_instances[vehicle_id] = (
                    damping_factory(vehicle_id) if damping_factory else object())
            events["damping_resume"] = previous not in (None, "damping")
        if controller == "pi-external-u" and previous != controller:
            events["pi_reset"] = True
        self._previous[vehicle_id] = controller
        return events


def sample_window_indices(timelines: list[ActionTimeline], *, count: int,
                          switch_fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Sample timeline indices with a configurable minimum switch-window share."""
    if not 0.0 <= switch_fraction <= 1.0 or count < 1 or not timelines:
        raise ValueError("invalid sampler arguments")
    switched = np.asarray([i for i, t in enumerate(timelines) if t.has_switch], dtype=np.int64)
    fixed = np.asarray([i for i, t in enumerate(timelines) if not t.has_switch], dtype=np.int64)
    n_switch = min(count, int(round(count * switch_fraction)))
    if n_switch and not switched.size:
        raise ValueError("switch oversampling requested but no switched windows exist")
    if n_switch < count and not fixed.size:
        raise ValueError("sampler needs fixed windows for the requested switch fraction")
    return np.concatenate((rng.choice(switched, n_switch, replace=n_switch > len(switched)),
                           rng.choice(fixed, count - n_switch,
                                      replace=count - n_switch > len(fixed))))
