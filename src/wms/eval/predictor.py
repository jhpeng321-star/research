"""Roll a world-model checkpoint forward from an evaluation state.

``WmsPredictor.predict_states`` is the one entry point every scorer uses: given a state
record, the CAV/arm description, a horizon and a seed, it returns predicted frames
(``[{"time_s", "vehicles": [...]}, ...]``) at the model's 0.5 s physics interval.

CAVs are teacher-forced from the recorded arm (their motion is exogenous); only HDVs are
predicted.  Entries (vehicles appearing after t0) are OFF by default.  The closed-loop CAV
mode of the historical harness is not part of this predictor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .backend import WmsBackend, load_backend
from .entries_input import entry_process_v1, entry_process_v2
from .errors import PredictorError
from .history_inputs import HistoryInputs
from .normalize import DT_S
from .rollout_inputs import (DEFAULT_LENGTH_M, action_timeline_inputs, recorded_teacher,
                             t0_vehicles)

CAV_MODES = ("teacher",)
FORBIDDEN_ROLLOUT_MODE = "driver_params"


@dataclass(frozen=True)
class RolloutFlags:
    """Rollout switches that are not part of the checkpoint (all default OFF).

    The ``diag_*`` switches are round-5 post-run attribution diagnostics: they
    neutralise one learned component at a time at inference time and are never
    active in training or scoring by default.
    """

    entries: bool = False
    entries_v2: bool = False
    diag_lane_gate_shared: bool = False
    diag_lane_cap_off: bool = False
    diag_encoder_population: bool = False
    diag_recorded_lanes: bool = False
    #: LANEFIX: seed of the lane draws when the model samples its lane decisions
    #: (``lane_decision='sample'``; ignored for argmax models) - every state is scored with
    #: the same, reproducible draws.
    lane_seed: int = 0

    def __post_init__(self) -> None:
        if self.entries_v2 and not self.entries:
            raise ValueError("entries_v2 requires entries=True")


def _pad_slots(initial: dict, n_pad: int) -> dict:
    """Append ``n_pad`` absent entrant slots to the t0 state tensors."""
    def pad(name, value, dtype):
        extra = torch.full((1, n_pad), value, dtype=dtype)
        return torch.cat((initial[name], extra), dim=1)

    return {
        "s_m": pad("s_m", 0.0, torch.float32),
        "v_mps": pad("v_mps", 0.0, torch.float32),
        "a_mps2": pad("a_mps2", 0.0, torch.float32),
        "lane": pad("lane", 0, torch.long),
        "length_m": pad("length_m", DEFAULT_LENGTH_M, torch.float32),
        "present": pad("present", False, torch.bool),
        "is_cav": pad("is_cav", False, torch.bool),
    }


class WmsPredictor:
    """Predict next states of a loaded ``wm_s`` network with CAVs teacher-forced."""

    def __init__(self, backend: WmsBackend, flags: RolloutFlags = RolloutFlags()) -> None:
        if str(backend.config.get("rollout_mode", "residual")) == FORBIDDEN_ROLLOUT_MODE:
            raise PredictorError(
                "driver_params rollout scoring is refused (WM_CHANGE_LOG #15)")
        self.backend = backend
        self.flags = flags
        self.model = backend.model
        self._history = HistoryInputs(backend)
        self.entries_summary: Mapping[str, Any] | None = None
        self._apply_diag_flags(flags)

    def _apply_diag_flags(self, flags: RolloutFlags) -> None:
        """Round-5 attribution switches: runtime model attributes, default no-op."""
        if flags.diag_lane_gate_shared:
            self.model.lane_gate_kind = "shared"
        if flags.diag_lane_cap_off:
            self.model.lane_logit_cap = None
        if flags.diag_encoder_population:
            self.model.diag_encoder_population = True

    @classmethod
    def from_checkpoint(cls, path: str | Path, flags: RolloutFlags = RolloutFlags(), *,
                        device: str = "cpu",
                        physics: Mapping[str, Any] | None = None) -> "WmsPredictor":
        return cls(load_backend(path, device=device, physics=physics), flags)

    # ------------------------------------------------------------------ rollout
    def predict_states(self, state: dict, plan_or_controllers: Mapping[str, Any] | None,
                       horizon_s: float, seed: int) -> list[dict]:
        controllers = dict(plan_or_controllers or {})
        mode = str(controllers.get("cav_mode", "teacher"))
        if mode not in CAV_MODES:
            raise PredictorError(f"unsupported CAV evaluation mode: {mode}")
        arm = str(controllers.get("arm", "native"))
        tools = self.backend.tools
        t0_s, ids, kin, is_cav = t0_vehicles(state)
        driver_history, driver_history_mask = self._history.build(state, ids)
        n_steps = max(1, int(round(float(horizon_s) / DT_S)))
        proc, n_pad = None, 0
        if self.flags.entries:
            if self.flags.entries_v2:
                proc, n_pad = entry_process_v2(state, t0_s, horizon_s, seed, kin, tools)
            else:
                proc, n_pad = entry_process_v1(state, t0_s, horizon_s, seed, len(ids), tools)
        entries_on = proc is not None
        action_timeline = action_timeline_inputs(state, controllers, ids, n_pad, tools)
        initial = {
            "s_m": torch.as_tensor(kin[None, :, 0]),
            "v_mps": torch.as_tensor(kin[None, :, 4]),
            "a_mps2": torch.zeros((1, len(ids))),
            "lane": torch.as_tensor(kin[None, :, 1:4].argmax(-1)),
            "length_m": torch.as_tensor(kin[None, :, 6]),
            "present": torch.ones((1, len(ids)), dtype=torch.bool),
            "is_cav": torch.as_tensor(is_cav[None]),
        }
        teacher, present_seq, junction_seq = recorded_teacher(
            state, t0_s, ids, kin, n_steps, arm, tools)
        if entries_on:
            initial = _pad_slots(initial, n_pad)
            pad_shape = (1, n_steps + 1, n_pad)
            if present_seq is not None:
                present_seq = torch.cat(
                    (present_seq, torch.ones(*pad_shape, dtype=torch.bool)), dim=2)
            if teacher is not None:
                teacher = torch.cat((teacher, torch.zeros(1, n_steps + 1, n_pad, 4)), dim=2)
            if junction_seq is not None:
                # Padding slots are never junction transits; pad the mask to the slot count.
                junction_seq = torch.cat(
                    (junction_seq, torch.zeros(*pad_shape, dtype=torch.bool)), dim=2)
            if driver_history is not None:
                # Entrants carry no observed history: the encoder falls back to the
                # physics defaults for them (mask False).
                driver_history = torch.cat((driver_history, torch.zeros(
                    1, n_pad, driver_history.shape[2], driver_history.shape[3],
                    dtype=driver_history.dtype)), dim=1)
                driver_history_mask = torch.cat((driver_history_mask, torch.zeros(
                    1, n_pad, driver_history_mask.shape[2], dtype=torch.bool)), dim=1)
        lane_kwargs = ({"lane_seed": int(self.flags.lane_seed)}
                       if getattr(self.model, "_samples_lanes", False) else {})
        with torch.no_grad():
            lane_teacher = None
            if self.flags.diag_recorded_lanes:
                if teacher is None:
                    raise PredictorError(
                        "diag_recorded_lanes requires the state's recorded arm")
                lane_teacher = teacher[..., 1].long()
            if teacher is None:
                roll = self.model.rollout(
                    initial, n_steps=n_steps, k0=0,
                    cav_controller=(self.model.hold_teacher
                                    if (bool(is_cav.any()) or entries_on) else None),
                    present_seq=present_seq, entries=proc,
                    driver_history=driver_history, driver_history_mask=driver_history_mask,
                    action_timeline=action_timeline, controller_id=None,
                    projection=self.backend.projection, lane_teacher=lane_teacher,
                    **lane_kwargs)
            else:
                roll = self.model.rollout(
                    initial, n_steps=n_steps, k0=0, cav_teacher=teacher,
                    present_seq=present_seq, junction_seq=junction_seq,
                    cav_controller=self.model.hold_teacher if entries_on else None,
                    entries=proc, driver_history=driver_history,
                    driver_history_mask=driver_history_mask, controller_id=None,
                    action_timeline=action_timeline, projection=self.backend.projection,
                    lane_teacher=lane_teacher, **lane_kwargs)
        if proc is not None:
            self.entries_summary = proc.summary()
        return _frames_from_rollout(roll, t0_s, ids, kin, is_cav, n_pad, n_steps)


def _frames_from_rollout(roll: Mapping[str, Any], t0_s: float, ids: list[str],
                         kin: np.ndarray, is_cav: np.ndarray, n_pad: int,
                         n_steps: int) -> list[dict]:
    positions = roll["positions"][0].numpy()
    speeds = roll["speeds"][0].numpy()
    lanes = roll["lanes"][0].numpy()
    masks = roll["masks"][0].numpy()
    n_slots = len(ids) + n_pad
    slot_vid = list(ids) + [f"__entry_{i}" for i in range(n_pad)]
    final_cav = roll["state"]["is_cav"][0].numpy()
    cav_slot = [bool(is_cav[k]) if k < len(ids) else bool(final_cav[k]) for k in range(n_slots)]
    frames = [{"time_s": t0_s, "vehicles": [
        {"id": slot_vid[k], "s": float(kin[k, 0]), "lane": int(kin[k, 1:4].argmax(-1)),
         "v": float(kin[k, 4]), "present": True, "is_cav": bool(is_cav[k])}
        for k in range(len(ids))]}]
    for index in range(n_steps):
        vehicles = [{"id": slot_vid[k], "s": float(positions[index, k]),
                     "lane": int(lanes[index, k]), "v": float(speeds[index, k]),
                     "present": True, "is_cav": cav_slot[k]}
                    for k in range(n_slots) if bool(masks[index, k])]
        frames.append({"time_s": t0_s + (index + 1) * DT_S, "vehicles": vehicles})
    return frames
