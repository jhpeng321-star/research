"""Default-off WM adapter; retains checkpoint tensors but bypasses learned motion."""
from dataclasses import replace

import torch

from wms.physics.accel import baseline_accel
from wms.physics.junction_merge import apply_merge, merge_speed_advice
from wms.physics.mobil import mobil_merge_logits
from wms.physics.state import state_kin, state_present
from . import cav as cav_motion
from .driver_encoder import PARAMETER_NAMES


def clean_physics(physics):
    return replace(physics, merge_cal=False, mainline_merge_yield=False,
                   merge_end_wait=False, merge_impatience=False, lane_learnable=False,
                   lane_change_secure_gap=False, secure_gap_veto=False,
                   entry_learned=False)


def forward(model, state, learned, lane_deciders, encoded, driver_params,
            driver_embedding, baseline_lane_logits=None):
    kin = state_kin(state).float()
    present = state_present(state, kin)
    physics = clean_physics(model.physics)
    # Direct physics kernel bypasses every learned acceleration, courtesy,
    # merge, temporal, noise and lane-correction emission.
    accel = baseline_accel(kin, present, physics, driver_params, emergency_envelope=False)
    accel, geometry = merge_speed_advice(kin, present, accel, physics, driver_params)
    if physics.emergency_braking:
        from wms.physics.accel import apply_emergency_envelope
        accel = apply_emergency_envelope(kin, present, accel, physics, driver_params)
    logits = (mobil_merge_logits(kin, present, physics, driver_params)
              if baseline_lane_logits is None else baseline_lane_logits.clone())
    lane = kin[..., 1:4].argmax(-1)
    ramp_logits = torch.full_like(logits, -1e9)
    ramp_logits.scatter_(-1, lane.unsqueeze(-1), 0)
    logits = torch.where(geometry['ramp'].unsqueeze(-1), ramp_logits, logits)
    zero = torch.zeros_like(accel)
    out = dict(physics_accel_mps2=accel, accel_mps2=accel,
               raw_residual_mps2=zero, accel_delta_mps2=zero,
               lane_logits=logits, gate=zero, residual_log_sigma=zero,
               lane_probabilities=logits.softmax(-1),
               hdv_mask=learned, lane_decider_mask=lane_deciders,
               sumo_merge_safe=geometry['safe'])
    if encoded is not None:
        out.update(driver_parameters=encoded['values'], driver_uncertainty=encoded['uncertainty'],
                   driver_shrinkage=encoded['shrinkage'])
    if driver_embedding is not None:
        out['driver_embedding'] = driver_embedding
    if model.lane_model == 'lc2013':
        from .lc2013_adapter import forward_output
        forward_output(model, state, kin, present, lane_deciders, driver_params, out)
        # Ramp acceptance remains exclusively owned by apply_merge in step.
        out['lane_logits'] = torch.where(geometry['ramp'].unsqueeze(-1),
                                          ramp_logits, out['lane_logits'])
    out['lane_probabilities'] = out['lane_logits'].softmax(-1)
    return out


def step(model, state, out, k_steps, driver_params, action_id, action_command,
         cav_teacher, cav_controller, projection):
    from .controllers import DIRECT_ID
    kin = state_kin(state)
    present = state_present(state, kin)
    cav = state.get('is_cav', torch.zeros_like(present)).bool() & present
    if driver_params is None and 'driver_parameters' in out:
        driver_params = {name: out['driver_parameters'][..., i]
                         for i, name in enumerate(PARAMETER_NAMES)}
    dt = model.physics.dt_s
    v = kin[..., 4]
    accel = out['accel_mps2']
    next_v = (v + accel * dt).clamp_min(0)
    next_s = kin[..., 0] + .5 * (v + next_v) * dt
    if getattr(model.physics, 'cf_model', 'legacy') == 'sumo_idm':
        # SUMO default --step-method.ballistic=false: advance with end-step speed.
        # CAV commands below retain the registered executor's own position rule.
        next_s = kin[..., 0] + next_v * dt
    elif model.physics.stop_time_integration:
        distance = v.square() / (2 * (-accel).clamp_min(1e-9))
        next_s = torch.where(v + accel * dt < 0, kin[..., 0] + distance, next_s)
    lane = kin[..., 1:4].argmax(-1)
    deciders = out['lane_decider_mask']
    next_lane = torch.where(deciders, out['lane_logits'].argmax(-1), lane)
    direct = (torch.zeros_like(cav) if action_id is None or not model.apply_direct_commands
              else (action_id == DIRECT_ID) & cav)
    cap = (model._exec_cap(out, state, driver_params)
           if (cav.any() or cav_controller is not None) and model.cav_exec_cap != 'off' else None)
    if direct.any():
        direct, next_s, next_v = model._apply_direct_commands(
            out, direct, kin[..., 0], v, next_s, next_v, action_command, exec_cap=cap)
    if not model.black_box and (cav.any() or cav_controller is not None):
        next_s, next_v, next_lane = model._apply_cav_overrides(
            out, state, cav, direct, present, k_steps, action_id, action_command,
            cav_teacher, cav_controller, next_s, next_v, next_lane,
            exec_cap=cap, driver_params=driver_params)
    if model.lane_model == 'lc2013':
        from .lc2013_adapter import step_output
        next_lane = step_output(model, state, kin, present, out, next_s, next_v,
                                next_lane, driver_params, ramp_owned_by_merge=True)
    moved = kin.clone()
    moved[..., 0], moved[..., 4] = next_s, next_v
    ramp = present & (lane == model.physics.merge_lane)
    moved[..., 0] = torch.where(ramp, next_s.clamp_max(model.physics.merge_zone_s1_m), next_s)
    deciders = deciders | out.get(cav_motion.WM_MERGE_KEY, torch.zeros_like(present))
    next_lane = apply_merge(moved, present, next_lane, deciders, model.physics, driver_params)
    staying = ramp & (next_lane == model.physics.merge_lane)
    overrun = staying & (next_s >= model.physics.merge_zone_s1_m)
    next_s = torch.where(overrun, torch.full_like(next_s, model.physics.merge_zone_s1_m), next_s)
    next_v = torch.where(overrun, torch.zeros_like(next_v), next_v)
    next_present = present.clone()
    if state.get('exit_s_m') is not None:
        next_present = next_present & (next_s <= state['exit_s_m'])
    if projection is not None:
        raise ValueError('SUMO merge uses exact footprints; padded projection must be off')
    out.update(next_s_m=next_s, next_v_mps=next_v, next_lane=next_lane,
               next_present=next_present, sumo_merge_cross=ramp & ~staying,
               lane_probabilities=torch.nn.functional.one_hot(next_lane.clamp(0, 2), 3).to(kin.dtype))
    return out
