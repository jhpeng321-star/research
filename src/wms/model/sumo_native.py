"""Opt-in native-CAV delegation to the existing physics plant, no learned head.

SUMO's native action is absence of an external command: its car follows CF/LC
physics. A learned acceleration surrogate can instead create an artificial
standstill before the ramp's legal merge zone. The compatibility callback uses
the registered externally commanded execution ceiling. Explicit uncommanded
delegation instead leaves native CAV CF/LC motion with the SUMO physical stream,
without applying that external ceiling. Both paths retain dispatcher lifecycle
calls and physical lane/merge decisions; externally commanded paths are unchanged.
"""
import torch

from .cav import EXEC_CAP_KEY, WM_LANE_TARGET_KEY, WM_MERGE_KEY

NATIVE_UNCOMMANDED_KEY = 'wm_sumo_uncommanded_mask'


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


def native_physics_controller(*, dt_s=0.5, uncommanded=False):
    """Native delegation; uncommanded payload is ignored by the SUMO step adapter."""
    dt = float(dt_s)
    if dt <= 0:
        raise ValueError('positive native execution step required')

    def callback(state):
        mask = state['present'].bool() & state['is_cav'].bool()
        acceleration = (torch.zeros_like(state['v_mps']) if uncommanded else
                        native_physics_accel(state, dt_s=dt))
        speed = (state['v_mps'] + dt*acceleration).clamp_min(0.)
        lane = torch.where(state['lane'] != 2,
                           state.get(WM_LANE_TARGET_KEY, state['lane']), state['lane'])
        state[WM_MERGE_KEY] = mask & (state['lane'] == 2)
        return torch.stack((state['s_m'] + dt*speed, lane, speed,
                            (speed-state['v_mps'])/dt), -1)

    callback.native_lane_mask = lambda s: s['present'].bool() & s['is_cav'].bool()
    if uncommanded:
        callback.native_uncommanded_mask = callback.native_lane_mask
    return callback
