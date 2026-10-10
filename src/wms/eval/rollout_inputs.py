"""Exogenous rollout inputs derived from a state record.

* the t0 vehicle table (sorted ids, ``kin`` rows, CAV flags),
* the recorded-arm CAV *teacher* (positions/lanes/speeds the CAVs are forced to follow)
  with a monotone presence sequence,
* the per-CAV action timeline aligned to the sorted slots.

Recorded labels are used only as exogenous CAV motion; HDV motion is always predicted.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from wms.core.constants import DEFAULT_LENGTH_M

from .backend import RolloutTools
from .errors import DataError, PredictorError
from .normalize import DT_S, initial_frame, recorded_for

N_LANES = 3
_CONTROLLER_ALIASES = {"pi": "pi-external-u", "fleet-pi": "pi-external-u"}


def nearest_frame(frame_by_round: Mapping[int, dict], index: int) -> dict | None:
    """The frame at tick ``index`` or, failing that, the one closest in time."""
    if not frame_by_round:
        return None
    if index in frame_by_round:
        return frame_by_round[index]
    return frame_by_round[min(frame_by_round, key=lambda key: abs(key - index))]


def t0_vehicles(state: Mapping[str, Any]):
    """Sorted modelled-lane vehicles of the t0 frame: ``(t0_s, ids, kin[N,7], is_cav[N])``."""
    init = initial_frame(dict(state))
    t0_s = float(init["time_s"])
    rows = [row for row in init["vehicles"]
            if row.get("present", True) and int(row.get("lane", -1)) >= 0]
    rows.sort(key=lambda row: str(row["id"]))
    if not rows:
        raise PredictorError("state has no vehicles on the modelled road")
    kin = np.zeros((len(rows), 7), dtype=np.float32)
    is_cav = np.zeros(len(rows), dtype=bool)
    for index, row in enumerate(rows):
        lane = int(row.get("lane", -1))
        kin[index, 0] = float(row.get("s", 0.0))
        if 0 <= lane < N_LANES:
            kin[index, 1 + lane] = 1.0
        kin[index, 4] = float(row.get("v", 0.0))
        kin[index, 6] = DEFAULT_LENGTH_M
        is_cav[index] = bool(row.get("is_cav", False))
    return t0_s, [str(row["id"]) for row in rows], kin, is_cav


def recorded_teacher(state: Mapping[str, Any], t0_s: float, ids: Sequence[str],
                     kin: np.ndarray, n_steps: int, arm: str, tools: RolloutTools):
    """Recorded-arm teacher ``[1,T+1,N,4]`` (s, lane, v, a), presence and junction ``[1,T+1,N]``.

    A vehicle on a junction internal lane (``lane < 0``) is carried with its last known
    state and kept present rather than dropped; only a genuinely absent recorded row (a
    true exit) ends the slot.  Returns ``(None, None, None)`` when the arm has no
    recorded trajectory.
    """
    recorded_present = np.zeros((n_steps + 1, len(ids)), dtype=bool)
    junction = np.zeros((n_steps + 1, len(ids)), dtype=bool)
    teacher = np.zeros((n_steps + 1, len(ids), 4), dtype=np.float32)
    teacher[:, :, 0] = kin[:, 0]
    teacher[:, :, 1] = kin[:, 1:4].argmax(-1)
    teacher[:, :, 2] = kin[:, 4]
    try:
        recorded = recorded_for(dict(state), arm)
    except DataError:
        recorded = None
    if not recorded:
        return None, None, None
    by_round = {int(round((float(frame["time_s"]) - t0_s) / DT_S)): frame
                for frame in recorded}
    previous = {vid: (float(kin[k, 0]), int(kin[k, 1:4].argmax(-1)), float(kin[k, 4]))
                for k, vid in enumerate(ids)}
    speeds: dict[str, tuple[int, float]] = {}
    for index in range(n_steps + 1):
        frame = nearest_frame(by_round, index)
        rows = {}
        if frame is not None:
            for row in frame["vehicles"]:
                if row.get("present", True):
                    rows[str(row["id"])] = row
        for k, vid in enumerate(ids):
            row = rows.get(vid)
            if row is None:
                recorded_present[index, k] = index == 0       # a real exit
                s, lane, v = previous[vid]
            elif int(row.get("lane", -1)) < 0:
                junction[index, k] = True                      # transit, carried forward
                s, lane, v = previous[vid]
            else:
                lane, v, s = int(row["lane"]), float(row["v"]), float(row["s"])
                recorded_present[index, k] = True
                previous[vid] = (s, lane, v)
                speeds[vid] = (index, v)
            teacher[index, k] = (s, lane, v, 0.0)
        if index > 0:
            for k, vid in enumerate(ids):
                hit = speeds.get(vid)
                if hit is not None and hit[0] == index:
                    teacher[index, k, 3] = (teacher[index, k, 2] - teacher[index - 1, k, 2]) / DT_S
    recorded_present[0] = True            # t0 is a modelled-lane snapshot by construction
    junction[0] = False
    present, junction_step = tools.junction_carry_presence(recorded_present, junction)
    return (torch.as_tensor(teacher[None], dtype=torch.float32),
            torch.as_tensor(present[None]), torch.as_tensor(junction_step[None]))


def _canonical_controller(name: str) -> str:
    return _CONTROLLER_ALIASES.get(name, name)


def action_timeline_inputs(state: Mapping[str, Any], controllers: Mapping[str, Any],
                           ids: Sequence[str], n_pad: int, tools: RolloutTools):
    """Align a validated per-CAV action timeline to the sorted t0 and entrant slots."""
    raw = controllers.get("action_timeline", state.get("action_timeline"))
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise PredictorError("action_timeline must be a schema-declared mapping")
    names = list(tools.controllers)
    try:
        timeline = tools.action_timeline.from_dict(raw)
        roster = [str(vid) for vid in raw.get("vehicle_ids", ids)]
        if len(roster) != timeline.controller.shape[1] or len(set(roster)) != len(roster):
            raise ValueError("timeline needs unique vehicle_ids matching its columns")
        if set(roster) - set(ids):
            raise ValueError("timeline vehicle_ids do not match t0 identities")
        entrant = _canonical_controller(str(controllers.get("default_entrant_controller", "native")))
        shape = (timeline.controller.shape[0], len(ids) + n_pad)
        ctrl = np.full(shape, names.index(entrant), dtype=np.int64)
        accel = np.zeros(shape, dtype=np.float32)
        target = np.zeros(shape, dtype=np.float32)
        modes = controllers.get("vehicle_controllers", {})
        arm = str(controllers.get("arm", "native"))
        for col, vid in enumerate(ids):
            ctrl[:, col] = names.index(_canonical_controller(str(modes.get(vid, arm))))
        for source, vid in enumerate(roster):
            col = list(ids).index(vid)
            ctrl[:, col] = timeline.controller[:, source]
            accel[:, col] = timeline.acceleration_mps2[:, source]
            target[:, col] = timeline.target_speed_mps[:, source]
    except (ValueError, KeyError, TypeError) as exc:
        raise PredictorError(f"invalid action timeline: {exc}") from exc
    return {"controller": torch.as_tensor(ctrl[None]),
            "acceleration_mps2": torch.as_tensor(accel[None]),
            "target_speed_mps": torch.as_tensor(target[None]),
            "control_dt_s": timeline.control_dt_s}
