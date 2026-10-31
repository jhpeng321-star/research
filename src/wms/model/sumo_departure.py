"""Opt-in one-step SUMO departure phase, AFTER motion and lane changing.

SUMO v1_27_1 MSNet::simulationStep calls executeMovements/changeLanes before
emitVehicles. New departures therefore appear at basePos with departSpeed,
without another same-step movement/controller/LC evaluation. This adapter
keeps existing arrival draws, FIFO/ordinals and SourceLedger. Callers consume
source_before/source_birth_mask/source_exit_mask for boundary cost accounting;
an entry observer's post-insertion snapshot is not the pre-motion cohort.

No default rollout or shared config changes: explicit one-step API only.
"""
import numpy as np
import torch

from .entries import section_ids
from .sumo_insertion import configure_entries


def step_after_movement(model, state, entries, *, driver_params=None,
                        entrant_driver_sampler=None, **rollout_kwargs):
    if model.physics.merge_model != 'sumo':
        raise ValueError('post-movement departure requires merge_model=sumo')
    if entrant_driver_sampler is not None and driver_params is None:
        raise ValueError('post-movement departure needs pre-bound initial driver settings')
    configure_entries(entries, model.physics, entrant_driver_sampler)
    process = getattr(entries, 'process', entries)
    # Seed virgin-slot bookkeeping from the true initial cohort, not after an
    # initial vehicle has already exited on the first movement step.
    if callable(getattr(process, '_observe', None)):
        process._observe(state)
    before = {name: state[name].clone() for name in
              ('s_m', 'v_mps', 'a_mps2', 'length_m', 'lane', 'present', 'is_cav')}
    result = model.rollout(state, n_steps=1, driver_params=driver_params, **rollout_kwargs)
    current = {key: value.clone() if torch.is_tensor(value) else value
               for key, value in result['state'].items()}
    exits = before['present'].bool() & ~current['present'].bool()
    # A numerical slot cannot represent both an old exit and a new birth in
    # the same logged frame. Reuse exited slots from the NEXT step. This is a
    # representation constraint, not lost demand; insufficient capacity is
    # reported by the existing no_slot statistic/conserved source queue.
    original_pool = process.pool
    eligible = ~before['present'].detach().cpu().numpy()
    if original_pool is not None:
        eligible &= np.broadcast_to(np.asarray(original_pool, dtype=bool), eligible.shape)
    process.pool = eligible
    try:
        newborn = entries.insert(current)
    finally:
        process.pool = original_pool
    if (newborn & before['present'].bool()).any():
        raise AssertionError('same-tick exit/birth recycled one logged slot')
    current['a_mps2'] = torch.where(newborn, torch.zeros_like(current['a_mps2']), current['a_mps2'])
    current['entrant'] = result['entrant'].bool() | newborn
    if model.lane_model == 'lc2013':
        from .lc2013_adapter import reset_newborn
        reset_newborn(current, newborn)
    after_params = driver_params
    if entrant_driver_sampler is not None:
        ordinals = getattr(entries, 'last_birth_ordinals', None)
        after_params = (entrant_driver_sampler(current, newborn, driver_params, birth_ordinals=ordinals)
                        if ordinals else entrant_driver_sampler(current, newborn, driver_params))

    # Boundary costs charge old road occupants, not newborns, and carry the
    # OLD speed for exiting vehicles. Blend only newborn rows into the cohort.
    cohort = {name: torch.where(newborn, current[name], value) for name, value in before.items()}
    cohort['present'] = before['present'].bool() | newborn
    result.update(state=current, source_before=cohort, source_birth_mask=newborn,
                  source_exit_mask=exits, driver_params_after_insertion=after_params,
                  source_phase='after_movement_and_lanes')
    for output, name in (('positions', 's_m'), ('speeds', 'v_mps'), ('accel', 'a_mps2'),
                         ('lanes', 'lane'), ('masks', 'present')):
        result[output] = current[name][:, None]
    result['entrant'] = current['entrant']
    result['entrant_seq'] = current['entrant'][:, None]
    result['sections'] = section_ids(result['positions'], result['masks'])
    if newborn.any():
        stay = torch.nn.functional.one_hot(current['lane'].clamp(0, 2), 3).to(current['s_m'])
        if 'lane_probabilities' in result:
            result['lane_probabilities'] = torch.where(newborn[:, None, :, None],
                                                       stay[:, None], result['lane_probabilities'])
        logits = torch.where(stay.bool(), torch.zeros_like(stay), torch.full_like(stay, -torch.inf))
        result['lane_logits'] = torch.where(newborn[:, None, :, None], logits[:, None], result['lane_logits'])
        result['lane_change_probabilities'] = torch.where(newborn[:, None],
            torch.zeros_like(result['lane_change_probabilities']), result['lane_change_probabilities'])
        # The last observation is the actual post-insertion frame. Reset reused
        # slot history before writing it, so a newborn cannot inherit another ID.
        from .history import history_step_features
        from wms.physics.state import state_kin
        from wms.physics.neighbours import lead_neighbour
        kin = state_kin(current)
        gap, _ = lead_neighbour(kin, current['present'], model.physics)
        features = history_step_features(kin, gap)
        history = torch.where(newborn[..., None, None], torch.zeros_like(result['history']), result['history']).clone()
        mask = result['history_mask'].bool() & ~newborn[..., None]
        history[:, :, -1] = features
        mask[:, :, -1] = current['present'].bool()
        result['history'], result['history_mask'] = history, mask
    return result
