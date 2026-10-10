"""CAV motion inside a WM-S step: action-timeline commands and teacher overrides.

The structured model never learns CAV motion.  A CAV follows (in priority order)
a ``direct`` timeline command, the recorded teacher trajectory, or - for slots that
entered after t0 - the supplied controller callback or native hold-speed motion.
"""
from __future__ import annotations

from typing import Mapping, Optional

import torch
from torch import Tensor

from .controllers import TIMELINE_TO_VOCAB


def timeline_controls(timeline: Mapping[str, Tensor], step_index: int, dt_s: float,
                      timeline_dt_s: float, reference: Tensor, dtype: torch.dtype
                      ) -> tuple[Tensor, Tensor]:
    """Controller ids ``[B, K]`` and commands ``[B, K, 2]`` active at ``step_index``.

    ``timeline`` holds ``controller [B, control_steps, K]`` (compact action-contract
    ids native/damping/PI/direct), ``acceleration_mps2`` and ``target_speed_mps``
    (both ``[B, control_steps, ...]``).  The ids are mapped to the controller
    vocabulary; ``reference`` is the ``[B, K]`` presence tensor providing the
    shape and device.
    """
    controller = torch.as_tensor(timeline["controller"], device=reference.device,
                                 dtype=torch.long)
    if controller.ndim != 3 or controller.shape[0] != reference.shape[0] \
            or controller.shape[2] != reference.shape[1]:
        raise ValueError("action timeline controller must be [B,control_steps,K]")
    control_index = min(int((step_index * float(dt_s)) // timeline_dt_s),
                        controller.shape[1] - 1)
    action_id = controller[:, control_index]
    action_id = action_id.new_tensor(TIMELINE_TO_VOCAB)[action_id.clamp(0, 3)]
    action_id = action_id.expand_as(reference)
    command = torch.stack((
        torch.as_tensor(timeline["acceleration_mps2"], device=reference.device,
                        dtype=dtype)[:, control_index],
        torch.as_tensor(timeline["target_speed_mps"], device=reference.device,
                        dtype=dtype)[:, control_index]), dim=-1)
    return action_id, command


def timeline_period(timeline: Optional[Mapping[str, Tensor]]) -> float:
    """Control period (s) of an action timeline; 1 s without a timeline."""
    period = float(timeline.get("control_dt_s", 1.0)) if timeline is not None else 1.0
    if period <= 0:
        raise ValueError("action timeline control_dt_s must be positive")
    return period


def direct_command_motion(s: Tensor, v: Tensor, command: Tensor, dt_s: float,
                          accel_min: float, accel_max: float
                          ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Feasible one-step motion of CAVs following an issued acceleration/target speed.

    ``command`` is ``[B, K, 2]`` = (requested acceleration, target speed).  The
    acceleration is limited by the physical bounds, the stop condition and the
    target speed.  Returns ``(next_s, next_v, accel, feasible)`` for every slot.
    """
    if command.shape[-1] != 2 or command.shape[:2] != v.shape:
        raise ValueError("action_command must be [B,K,2]")
    requested_a, target_v = command[..., 0], command[..., 1]
    dt = float(dt_s)
    lo = torch.maximum(torch.full_like(v, accel_min), -v / dt)
    hi = torch.minimum(torch.full_like(v, accel_max), (target_v - v) / dt)
    feasible = hi >= lo
    accel = torch.minimum(torch.maximum(requested_a, lo), hi)
    next_v = (v + accel * dt).clamp_min(0.0)
    next_s = s + 0.5 * (v + next_v) * dt
    return next_s, next_v, accel, feasible


def hold_speed_teacher(s: Tensor, lane: Tensor, v: Tensor, dt_s: float) -> Tensor:
    """Native hold-speed CAV motion ``[B, K, 4] = s, lane, v, a``."""
    return torch.stack((s + v * dt_s, lane.float(), v, torch.zeros_like(v)), dim=-1)


def check_teacher(teacher: Tensor, reference: Tensor) -> None:
    if teacher.shape[-1] != 4 or teacher.shape[:2] != reference.shape:
        raise ValueError("CAV teacher must be [B,K,4] = s,lane,v,a")
