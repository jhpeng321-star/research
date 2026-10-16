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


# ---------------------------------------------------------------- execution cap (EXECCAP_NOTES.md)
#: Allowed values of ``WmSModel.cav_exec_cap``: ``"off"`` (default, bit-identical legacy
#: behaviour), ``"idm"`` (SUMO speedMode-27 consistent execution of commanded CAVs with the
#: WM-inferred settings of the slot) or ``"idm_normal[:<table>]"`` (INFRAV5: the same ceiling
#: with the fixed, operator-known CAV vType settings of :data:`CAV_SETTINGS_TABLES`; the CAV
#: type is known to the operator -- design rule 2026-10-14: every CAV has one vType).
#: ``"idm_normal"`` alone = :data:`DEFAULT_CAV_SETTINGS_TABLE`.
#: SUMO ``normal_car`` vType (scenarios/idm/driver_routes.rou.xml; generated scenarios use
#: speedDev 0, so v0 = speedFactor x speed limit): the legacy population's CAV type.
NORMAL_CAR_SETTINGS = {"accel": 2.6, "decel": 4.5, "time_headway": 1.1, "min_gap": 2.5,
                       "speed_factor": 1.0}
#: Textbook population CAV type ``cav_typical`` (c4/popfix, 2026-10-14).
#: TODO(c4/popfix): import ``TEXTBOOK_CAV_PARAMS`` from ``traffic_shepherd/c4/population.py``
#: once merged (keys there: speed_factor, tau_s, min_gap_m, accel_mps2, decel_mps2); the values
#: below are a copy and tests/unit/test_infrav5.py checks them against it when importable.
TEXTBOOK_CAV_SETTINGS = {"accel": 1.2, "decel": 2.0, "time_headway": 1.2, "min_gap": 2.0,
                         "speed_factor": 1.0}
#: CAV settings tables of the ``idm_normal`` execution cap (name -> settings).
CAV_SETTINGS_TABLES = {"textbook": TEXTBOOK_CAV_SETTINGS, "legacy": NORMAL_CAR_SETTINGS}
DEFAULT_CAV_SETTINGS_TABLE = "textbook"
EXEC_CAP_MODES = ("off", "idm", "idm_normal",
                  *(f"idm_normal:{name}" for name in CAV_SETTINGS_TABLES))
#: Process-wide default read by :func:`wms.eval.backend.load_backend` (and nothing else).
EXEC_CAP_ENV = "WMS_CAV_EXEC_CAP"
#: Key under which the model hands the per-slot ceiling ``[B, K]`` to its CAV callback.
EXEC_CAP_KEY = "cav_exec_cap_mps2"


def check_exec_cap_mode(mode) -> str:
    mode = "off" if mode is None else str(mode).strip().lower()
    if mode not in EXEC_CAP_MODES:
        raise ValueError(f"cav_exec_cap must be one of {EXEC_CAP_MODES}, got {mode!r}")
    return mode


def exec_cap_settings(mode) -> dict:
    """Fixed CAV settings of an ``idm_normal[:<table>]`` mode."""
    mode = check_exec_cap_mode(mode)
    if not mode.startswith("idm_normal"):
        raise ValueError(f"{mode!r} has no fixed CAV settings table")
    table = mode.partition(":")[2] or DEFAULT_CAV_SETTINGS_TABLE
    return dict(CAV_SETTINGS_TABLES[table])


def set_exec_cap(model, mode) -> str:
    """Set the runtime execution-cap switch on ``model`` (no-op for models without it)."""
    mode = check_exec_cap_mode(mode)
    if hasattr(model, "cav_exec_cap"):
        model.cav_exec_cap = mode
    return mode


def exec_accel_cap(physics_accel: Tensor, v: Tensor, dt_s: float, v0, accel_limit) -> Tensor:
    """Ceiling ``[B, K]`` of the acceleration a commanded CAV can realise in SUMO.

    In SUMO (speedMode 27, safe speed regarded) a commanded CAV realises
    ``min(command, own vType car-following acceleration)``; the car-following term is the
    WM physics core's IDM acceleration of the slot with its inferred driver settings
    (incl. the accel-lane-end obstacle), further bounded by the approach to the slot's
    desired speed ``(v0 - v) / dt`` and its vType-like ``accel`` limit.
    """
    cap = torch.minimum(physics_accel, (torch.as_tensor(v0, dtype=v.dtype, device=v.device) - v)
                        / float(dt_s))
    return torch.minimum(cap, torch.as_tensor(accel_limit, dtype=v.dtype, device=v.device)
                         .expand_as(cap))


def apply_exec_cap(state: Mapping[str, Tensor], accel: Tensor,
                   mask: Optional[Tensor] = None) -> Tensor:
    """``min(accel, state[EXEC_CAP_KEY])`` on ``mask`` (all slots if None); identity without a cap.

    Used by every COMMAND-driven CAV executor (bridge, coop-imagine callable, trainer) on its
    post-safety command; recorded teachers never pass through it.  Gradient: ``torch.minimum``
    routes the gradient to the binding argument -- a braking (non-binding) command keeps its
    full gradient; where the SUMO ceiling binds, d a_exec / d command = 0 (a larger command
    changes nothing physically) and the gradient flows through the ceiling to the state.
    """
    cap = state.get(EXEC_CAP_KEY) if hasattr(state, "get") else None
    if cap is None:
        return accel
    capped = torch.minimum(accel, cap.to(dtype=accel.dtype, device=accel.device))
    return capped if mask is None else torch.where(mask.bool(), capped, accel)
