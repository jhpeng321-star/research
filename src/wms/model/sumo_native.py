"""Opt-in native-CAV delegation to the existing physics plant, no learned head.

SUMO's native action is absence of an external command: its car follows CF/LC
physics. A learned acceleration surrogate can instead create an artificial
standstill before the ramp's legal merge zone. This callback consumes the SAME
physics execution ceiling already supplied by WmSModel, with the registered
native CAV executor's acceleration bound (continuous.project_native), and lets
the physics lane/merge streams decide lanes. External controller paths are not
changed; callers must select this callback explicitly for the native arm.
"""
import torch

from .cav import EXEC_CAP_KEY, WM_LANE_TARGET_KEY, WM_MERGE_KEY


def native_physics_accel(state, *, dt_s=0.5):
    """Drop-in ``native(state)`` for native-relative controller prediction.

    Uses the same inferred/population CF/merge plant as the native callback;
    zero-residual plans therefore share their physical native reference.
    """
    from wms.control.continuous import project_native
    mask = state['present'].bool() & state['is_cav'].bool()
    cap = state[EXEC_CAP_KEY]
    if not torch.isfinite(cap[mask]).all():
        raise ValueError('nonfinite physics native acceleration')
    return project_native(state, cap, dt=dt_s)


def native_physics_controller(*, dt_s=0.5):
    dt = float(dt_s)
    if dt <= 0:
        raise ValueError('positive native execution step required')

    def callback(state):
        mask = state['present'].bool() & state['is_cav'].bool()
        acceleration = native_physics_accel(state, dt_s=dt)
        speed = (state['v_mps'] + dt*acceleration).clamp_min(0.)
        lane = torch.where(state['lane'] != 2,
                           state.get(WM_LANE_TARGET_KEY, state['lane']), state['lane'])
        state[WM_MERGE_KEY] = mask & (state['lane'] == 2)
        return torch.stack((state['s_m'] + dt*speed, lane, speed,
                            (speed-state['v_mps'])/dt), -1)

    callback.native_lane_mask = lambda s: s['present'].bool() & s['is_cav'].bool()
    return callback
