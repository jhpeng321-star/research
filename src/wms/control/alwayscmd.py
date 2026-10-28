"""S-EXECPARITY 00:20 EXACT ALWAYS-COMMANDED law (owner decision, REGISTRATION.md 00:20).

Every designated CAV is commanded at every control/physics tick through the
single CommandOwnership writer - the S-CTRLARM CTRL0 path with the policy's
residual kept instead of forced to zero. There is no native release at
residual 0: the deterministic base law (SUMO's own car-following command,
``sumo_native_accel``) is issued plus the residual, and SUMO then applies its
speedMode-27 safe-speed cap and the 1.2 m/s^2 acceleration bound.

S-EXECPARITY2 (02:40): imagination executes commanded CAVs through the SAME
function as the deployment executor. ``inferred_base``/``executed`` call the
shared exact law of :mod:`wms.control.sumo_idm` (SUMO 1.27.1
``MSCFModel_IDM::_v`` on the designed ``cav_typical`` vType parameters, the
frozen legacy gap convention and lane speed limits) with the WM state of the
CAV and its leader, then the same safe-speed cap formula and bounds. The
TRAIN-learned IDM driver mirror is retired from this path.

Residual 0 is bit-identical to CTRL0 by construction: the composer returns the
unclipped base law exactly as S-CTRLARM ``forced_base_commands`` did. The
policy can reintroduce yielding/holding only through its own executed braking
residuals; a positive residual beyond the safe-speed cap has zero effect and
zero gradient.
"""
from __future__ import annotations

import math

BACKEND_CONTRACT = "always_commanded_ctrl0/1"


def commands(api, residuals, lo, hi, *, dt=0.5, parity=None):
    """CTRL0-compatible always-commanded composer; single source of the base law.

    ``parity`` is accepted and ignored: the always-commanded law has one frozen
    base convention (the legacy ``getFollowSpeed`` gap handling CTRL0 ran with).
    Residual exactly zero returns the unclipped base law bit-identically to
    S-CTRLARM ``forced_base_commands``; nonzero residuals use the frozen legacy
    envelope ``clip(base + r, min(lo, base), max(min(hi, getAccel), base))``.
    """
    import numpy as np
    from .nativeinit import sumo_native_accel
    composed = {}
    for vid, residual in residuals.items():
        if not np.isfinite(residual):
            raise ValueError("nonfinite residual for a designated CAV")
        baseline = float(sumo_native_accel(api, vid, dt))
        if residual == 0.0:
            composed[vid] = baseline
        else:
            upper = min(float(hi), float(api.vehicle.getAccel(vid)))
            composed[vid] = float(np.clip(baseline + float(residual),
                                          min(float(lo), baseline),
                                          max(upper, baseline)))
    return composed


def write_residuals(api, residuals, lo, hi, *, dt=0.5):
    """The sole longitudinal writer: setSpeed(-1) then setAcceleration, every tick.

    Identical TraCI call sequence to the frozen CTRL0/``execute_sumo_residual``
    physics path, extended to zero-residual CAVs. SUMO's speedMode 27 remains
    the final safety executor; no slowDown, stale cached speed or absolute
    speed target may precede or compete with this write.
    """
    composed = commands(api, residuals, lo, hi, dt=dt)
    if not all(math.isfinite(c) for c in composed.values()):
        raise ValueError("nonfinite always-commanded acceleration")
    for vid, command in composed.items():
        api.vehicle.setSpeed(vid, -1)
        api.vehicle.setAcceleration(vid, command, dt)
    return composed


def _wm_geometry(state):
    """The WM frame's own speed, leader bumper gap, leader speed and presence."""
    from .native_cav import leader_gap_dv
    v = state["v_mps"]
    gap, dv, has = leader_gap_dv(state)
    return v, gap, v - dv, has


def inferred_base(state, params=None, *, dt=0.5):
    """EXACT executor base law for a commanded CAV (S-EXECPARITY2, 02:40).

    Calls the SAME shared function as the deployment executor's law
    (:func:`wms.control.nativeinit.sumo_native_accel` -> SUMO 1.27.1
    ``MSCFModel_IDM::_v`` on the designed ``cav_typical`` vType parameters):
    :func:`wms.control.sumo_idm.base_accel` with the WM state of the CAV and
    its leader, the frozen legacy double-minGap gap convention and the lane
    ``allowed_speed``. ``params`` (learned driver artifact values) is accepted
    for signature compatibility and deliberately ignored - the base law is the
    executor's own function and parameters, never a learned mirror.
    """
    from . import sumo_idm
    v, gap, leader_speed, has = _wm_geometry(state)
    allowed = sumo_idm.allowed_speed(state["lane"], state["s_m"])
    return sumo_idm.base_accel(v, leader_speed, gap, allowed,
                               has_leader=has, dt=dt)


def inferred_cap(state, *, dt=0.5):
    """EXACT speedMode-27 executed-speed ceiling from the WM frame."""
    from . import sumo_idm
    v, gap, leader_speed, has = _wm_geometry(state)
    allowed = sumo_idm.allowed_speed(state["lane"], state["s_m"])
    return sumo_idm.safe_speed_cap(v, leader_speed, gap, allowed,
                                   has_leader=has, lane=state["lane"],
                                   s_m=state["s_m"], dt=dt)


def executed(state, residual, lo=-3., hi=1.2, *, dt=0.5, params=None):
    """Imagination mirror of the always-commanded executed acceleration.

    ``issued = clip(base + r, min(lo, base), max(min(hi, 1.2), base))`` with
    the EXACT shared executor base law, then the SUMO speedMode-27 outcome
    ``max(-v/dt, min(issued, cap))`` with the exact shared safe-speed cap
    formula and bounds (single-minGap follow speed, lane-2 dead-end stop,
    accel/allowedSpeed ceiling): the single shared composed law
    :func:`wms.control.sumo_idm.executed_accel` evaluated on the WM state of
    the CAV and its leader. The WM head is NOT the base and
    ``state[EXEC_CAP_KEY]`` is NOT consulted: imagination executes commanded
    CAVs through the executor's own function. Residual 0 yields the executed
    CTRL0 law; a positive residual beyond the cap keeps exactly zero gradient.
    """
    from . import sumo_idm
    v, gap, leader_speed, has = _wm_geometry(state)
    allowed = sumo_idm.allowed_speed(state["lane"], state["s_m"])
    return sumo_idm.executed_accel(v, leader_speed, gap, allowed, residual,
                                   float(lo), float(hi), has_leader=has,
                                   lane=state["lane"], s_m=state["s_m"], dt=dt)


def scalar_executed(speed, leader_speed, gap, residual, params=None, lo=-3., hi=1.2, *,
                    dt=0.5, has_leader=True, ramp_gap=None, allowed_speed=None,
                    prev_lane=None, lane=None, s_m=None):
    """Independent scalar mirror of :func:`executed` for fixtures and offline replay.

    ``gap`` is the WM-frame bumper gap of the CAV's leader; ``params`` is
    accepted and ignored (the law is the executor's own). ``allowed_speed``
    defaults to the lane/position rule (or ``prev_lane`` for junction rows);
    ``ramp_gap`` (if finite) applies the lane-2 dead-end stop exactly like the
    tensor path. Evaluates the SAME shared composed law through
    :func:`wms.control.sumo_idm.executed_accel_scalar`, so it is bitwise
    identical to the tensor path by construction.
    """
    from . import sumo_idm
    if allowed_speed is None:
        if lane is not None:
            allowed = sumo_idm.allowed_speed_scalar(lane, s_m if s_m is not None else 0.,
                                                    prev_lane=prev_lane)
        else:
            from .contract import SPEED_LIMIT_MPS
            allowed = float(SPEED_LIMIT_MPS)
    else:
        allowed = float(allowed_speed)
    return sumo_idm.executed_accel_scalar(speed, leader_speed, gap, allowed,
                                          residual, float(lo), float(hi),
                                          has_leader=has_leader, ramp_gap=ramp_gap,
                                          dt=dt)


def completed_rows(api, before, residuals, issued, *, dt=0.5):
    """Per-tick execution ledger rows, read only after the step that realised them."""
    live = set(api.vehicle.getIDList())
    rows = {}
    for vid in sorted(set(before) & live):
        v = before[vid]
        r = float(residuals.get(vid, 0.))
        command = issued.get(vid)
        rows[vid] = {"requested_residual_mps2": r,
                     "base_law_mps2": (command if command is None or r == 0.
                                       else command - r),
                     "issued_acceleration_mps2": None if command is None else float(command),
                     "speed_before_mps": v,
                     "realised_acceleration_mps2": (float(api.vehicle.getSpeed(vid)) - v) / dt,
                     "writer": "always_commanded" if command is not None else "native"}
    return rows
