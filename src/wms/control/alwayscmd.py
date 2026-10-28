"""S-EXECPARITY 00:20 EXACT ALWAYS-COMMANDED law (owner decision, REGISTRATION.md 00:20).

Every designated CAV is commanded at every control/physics tick through the
single CommandOwnership writer - the S-CTRLARM CTRL0 path with the policy's
residual kept instead of forced to zero. There is no native release at
residual 0: the deterministic base law (SUMO's own car-following command,
``sumo_native_accel``) is issued plus the residual, and SUMO then applies its
speedMode-27 safe-speed cap and the 1.2 m/s^2 acceleration bound. Imagination
mirrors the same law with the identical envelope algebra, an inferred base law
and an inferred follow cap from the TRAIN-learned IDM driver artifact.

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


def _driver_params(state, params):
    """Same parameter source as ``execution.inferred_follow_cap``."""
    import os
    import torch
    fitted = os.environ.get("C4_EXECUTION_DRIVER")
    if fitted:
        from .execution import load_learned_driver
        values = load_learned_driver(fitted)
        v = state["v_mps"]
        return {key: torch.full_like(v, value) for key, value in values.items()}
    if params is None:
        raise ValueError("always-commanded imagination needs learned driver parameters")
    return params


def inferred_base(state, params=None, *, dt=0.5):
    """Mirror of ``sumo_native_accel`` under the learned IDM driver.

    The frozen base law passes the bumper gap plus minGap to ``getFollowSpeed``
    while the IDM adds minGap internally; that legacy convention is mirrored
    here by adding the learned min_gap once more before the single-add
    ``idm_follow_speed``. ``getAllowedSpeed`` is mirrored by the learned v0.
    """
    import torch
    from .execution import idm_follow_speed
    from .native_cav import leader_gap_dv
    params = _driver_params(state, params)
    v = state["v_mps"]
    gap, dv, has = leader_gap_dv(state)
    gap = torch.where(has, gap, torch.full_like(gap, float("inf")))
    legacy_gap = gap + params["min_gap"].to(v)
    target = idm_follow_speed(v, v - dv, legacy_gap, params, dt=dt)
    target = torch.minimum(target, params["v0"].to(v))
    return (target.clamp_min(0.) - v) / dt


def executed(state, residual, lo=-3., hi=1.2, *, dt=0.5, params=None):
    """Imagination mirror of the always-commanded executed acceleration.

    ``issued = clip(base + r, min(lo, base), max(min(hi, 1.2), base))`` with the
    inferred base law, then the speedMode-27 outcome ``min(issued, follow cap)``
    floored by the nonnegative-speed bound. Residual 0 yields the executed
    CTRL0 law ``max(-v/dt, min(base, cap))``; the WM head is NOT the base. A
    positive residual beyond the cap keeps exactly zero gradient.
    """
    import torch
    from wms.model.cav import EXEC_CAP_KEY
    v = state["v_mps"]
    base = inferred_base(state, params, dt=dt)
    if EXEC_CAP_KEY in state:
        cap = state[EXEC_CAP_KEY].to(residual)
    else:
        from .execution import inferred_follow_cap
        cap = inferred_follow_cap(state, _driver_params(state, params), dt=dt)
    upper = torch.maximum(base, torch.full_like(base, min(float(hi), 1.2)))
    lower = torch.minimum(base, torch.full_like(base, float(lo)))
    issued = torch.minimum(torch.maximum(base + residual, lower), upper)
    return torch.maximum(torch.minimum(issued, cap), -v / dt)


def scalar_executed(speed, leader_speed, gap, residual, params, lo=-3., hi=1.2, *,
                    dt=0.5, has_leader=True, ramp_gap=None, allowed_speed=None):
    """Independent scalar mirror of :func:`executed` for fixtures and offline replay.

    ``params`` maps accel/decel/time_headway/min_gap/v0 to plain floats. The
    follow cap uses the single-minGap convention plus the lane-end stop and the
    learned v0/accel/1.2 bounds, exactly like ``inferred_follow_cap``; the base
    law uses the legacy double-add convention plus the v0 clamp.
    """
    values = [speed, leader_speed, residual, lo, hi, dt]
    if not all(math.isfinite(float(x)) for x in values) or dt <= 0:
        raise ValueError("finite always-commanded execution inputs required")
    a, b, tau, gap0, v0 = (float(params[k]) for k in
                           ("accel", "decel", "time_headway", "min_gap", "v0"))
    if allowed_speed is not None:
        v0 = min(v0, float(allowed_speed))
    # matches torch.finfo(float64).eps used by execution.idm_follow_speed's clamp_min
    eps = 2.220446049250313e-16

    def idm_speed(v, vl, g, respect_min_gap):
        # Branch-by-branch mirror of execution.idm_follow_speed (dt=.5 -> 2 substeps).
        iterations = max(1, int(float(dt) / .25 + .5))
        step = float(dt) / iterations
        out, gap_now = v, g
        for _ in range(iterations):
            desired = max(0., out * tau + out * (out - vl) / (2. * math.sqrt(max(a * b, eps))))
            if respect_min_gap:
                desired += gap0
            accel = a * (1. - (out / max(v0, eps)) ** 4
                         - (desired / max(gap_now, eps)) ** 2)
            out = max(0., out + accel * step)
            gap_now = gap_now - max(0., out - vl) * step
        return out

    if has_leader:
        cap_target = idm_speed(speed, float(leader_speed), float(gap), True)
        base_target = idm_speed(speed, float(leader_speed), float(gap) + gap0, True)
    else:
        cap_target = idm_speed(speed, float(speed), float("inf"), True)
        base_target = idm_speed(speed, float(speed), float("inf"), True)
    if ramp_gap is not None and math.isfinite(float(ramp_gap)):
        stop = idm_speed(speed, 0., float(ramp_gap), False)
        cap_target = min(cap_target, stop, float(ramp_gap) / dt)
    cap_target = min(cap_target, v0)
    cap = min((cap_target - speed) / dt, a, 1.2)
    base = (max(0., min(base_target, v0)) - speed) / dt
    upper = max(base, min(float(hi), 1.2))
    lower = min(base, float(lo))
    issued = min(max(base + float(residual), lower), upper)
    return max(-speed / dt, min(issued, cap))


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
