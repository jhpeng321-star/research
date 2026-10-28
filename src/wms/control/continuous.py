"""Combined native-relative law and fail-closed same-step backend boundary.

Stock TraCI getSpeedWithoutTraCI is a *completed-step* observation. It cannot
provide the next native speed before simulationStep. A native-speed callback in
the simulator is required; never substitute that stale observation here.
"""
from __future__ import annotations

import math
import torch

BACKEND_CONTRACT = "native_relative_same_step/1"


class _IdentityAtZero(torch.autograd.Function):
    @staticmethod
    def forward(ctx, native, executed, residual):
        return torch.where(residual == 0., native, executed)

    @staticmethod
    def backward(ctx, gradient):
        # The physical clipped law provides the derivative, including at r=0.
        return None, gradient, None


def project_native(state, native, *, dt=.5):
    """Constrain the WM prediction once, including the native reference row.

    SUMO's actual same-step native speed already satisfies these physical bounds.
    A learned head may not; projection is explicit model error, not residual credit.
    """
    from wms.model.cav import EXEC_CAP_KEY
    cap = state[EXEC_CAP_KEY].to(native)
    return torch.maximum(torch.minimum(torch.minimum(native, cap),
                         torch.full_like(native, 1.2)), -state["v_mps"] / dt)


def native_relative(state, residual, native, lo=-3., hi=1.2, *, dt=.5):
    """Shared algebra: native + residual, physical cap, no soft safe_accel.

    `native` is the *same-step* acceleration, including LC speed adjustments.
    Zero returns the native tensor bit-exactly. Its derivative remains useful at
    zero (unlike a torch.where zero-release branch). Emergency native braking is
    preserved; speedMode27 disables the comfortable-deceleration bound.
    """
    from wms.model.cav import EXEC_CAP_KEY
    cap = state[EXEC_CAP_KEY].to(residual)
    upper = torch.minimum(cap, torch.full_like(cap, min(float(hi), 1.2)))
    lower = torch.minimum(native, torch.full_like(native, float(lo)))
    # Express clipping in residual coordinates: exact identity at residual=0
    # and no cancellation from subtracting nearly equal absolute speeds.
    delta = torch.minimum(torch.maximum(residual, lower-native), upper-native)
    out = native + delta
    out = torch.maximum(out, -state["v_mps"] / dt)
    return _IdentityAtZero.apply(native, out, residual)


def scalar_native_relative(speed, native_speed, cap_accel, residual, lo=-3., hi=1.2, dt=.5):
    """Independent scalar definition for backend fixtures/offline diagnostics."""
    values = (speed, native_speed, cap_accel, residual, lo, hi, dt)
    if not all(math.isfinite(float(x)) for x in values) or dt <= 0:
        raise ValueError("finite native-relative execution inputs required")
    native = (native_speed-speed)/dt
    if residual == 0.:
        return native
    return max(-speed/dt, min(cap_accel, hi, 1.2,
                             max(min(lo, native), native+residual)))


def require_same_step_backend(api):
    """No built-in SUMO/TraCI version currently supplies this extension.

    Explicit extension boundary, NOT a claim that stock SUMO has this method.
    The setter must apply r after native LC/CF finalizeSpeed, before mode27
    safety, for exactly one physics interval. A cached speed getter is forbidden.
    """
    domain = api.vehicle
    contract = getattr(domain, "getNativeResidualContract", None)
    setter = getattr(domain, "setNativeAccelerationResidual", None)
    if not callable(contract) or not callable(setter) or contract() != BACKEND_CONTRACT:
        raise ValueError("continuous execution blocked: same-step native residual backend missing; "
                         "stock getSpeedWithoutTraCI is a completed-step cache")


def write_residuals(api, residuals, lo, hi, *, dt=.5):
    """The sole writer sends residuals, never an independently composed speed.

    Exactly zero is true native: no getter, setter, ownership, or speed advice.
    Nonzero commands fail before any write on the frozen stock-SUMO backend.
    """
    from .execution import SPEED_MODE
    commands = {v: float(r) for v,r in residuals.items() if float(r) != 0.}
    if not all(math.isfinite(r) for r in commands.values()):
        raise ValueError("nonfinite residual for a designated CAV")
    if not commands:
        return {}
    require_same_step_backend(api)
    for vid in commands:
        if api.vehicle.getSpeedMode(vid) != SPEED_MODE:
            raise ValueError("continuous execution requires speedMode27")
    for vid, r in commands.items():
        api.vehicle.setNativeAccelerationResidual(vid, r, min(float(hi), 1.2), float(lo), dt)
    return commands


def completed_execution_rows(api, before, residuals, accepted, lo, hi, *, dt=.5):
    """Read no-TraCI speed AFTER the step that generated it, never as an oracle."""
    live = set(api.vehicle.getIDList())
    rows = {}
    for vid in sorted(set(before) & live):
        v = before[vid]
        native_speed = float(api.vehicle.getSpeedWithoutTraCI(vid))
        native = (native_speed-v)/dt
        r = float(residuals[vid])
        issued = max(-v/dt, min(hi, 1.2, max(min(lo,native), native+r))) if vid in accepted else None
        rows[vid] = {"requested_residual_mps2":r, "speed_before_mps":v,
            "same_step_no_traci_speed_mps":native_speed,
            "issued_acceleration_before_safe_cap_mps2":issued,
            "realised_acceleration_mps2":(float(api.vehicle.getSpeed(vid))-v)/dt,
            "accepted":vid in accepted,
            "writer":"native_relative_same_step" if vid in accepted else "native"}
    return rows
