"""Explicit inference-only multi-step SUMO move/LC/emit adapter.

Uses MERGE's tested one-step phase. The frozen evaluator's cost observer must
retain the OLD road cohort (with newborn rows blended in), not the post-motion
snapshot seen by insertion. No shared model defaults or source draws change.
"""
import torch

from wms.model.sumo_departure import step_after_movement


TAPES = ('positions', 'speeds', 'lanes', 'masks', 'accel', 'lane_logits',
         'raw_residual_mps2', 'lane_change_probabilities', 'lane_probabilities',
         'entrant_seq', 'sections', 'physics_accel_mps2')


def rollout_after_movement(model, state, *, n_steps, entries, **kwargs):
    if torch.is_grad_enabled() or model.training:
        raise ValueError('departure phase adapter is frozen inference only')
    if int(n_steps) <= 0:
        raise ValueError('nonempty rollout required')
    allowed = {'k0', 'history', 'history_mask', 'driver_history', 'driver_history_mask',
               'driver_defaults', 'driver_params', 'driver_embedding',
               'entrant_driver_sampler', 'cav_controller'}
    if set(kwargs) - allowed:
        raise ValueError('unsupported phased rollout inputs: '+str(sorted(set(kwargs)-allowed)))
    # No silent truncation of future observations, teachers, stochastic tapes or
    # controller timelines. Registered candidates use fixed settings and callbacks.
    kw = dict(kwargs)
    k0 = int(kw.pop('k0', 0))
    current = state
    tapes = {name: [] for name in TAPES}
    for tick in range(int(n_steps)):
        result = step_after_movement(model, current, entries, k0=k0+tick, **kw)
        if hasattr(entries, 'observer'):
            # Entries.insert records an insertion-time cohort. Reported-boundary
            # costs instead use old speed on exit and no road charge at birth.
            entries.observer.states[-1].update(result['source_before'])
        for name in TAPES:
            if name in result:
                tapes[name].append(result[name])
        current = result['state']
        kw.update(history=result['history'], history_mask=result['history_mask'],
                  driver_params=result['driver_params_after_insertion'])
        for name in ('driver_history', 'driver_history_mask', 'driver_defaults'):
            kw.pop(name, None)
    result.update({name: torch.cat(values, 1) for name, values in tapes.items() if values})
    result['source_phase'] = 'after_movement_and_lanes'
    return result
