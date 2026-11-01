"""Entrant-only physical population weights on an unchanged frozen joint support."""
import torch

from .fitted_hidden import IDM_ORDER, read_artifact
from wms.model.driver_encoder import PARAMETER_NAMES


def bind_entrant_population(entries, path, source_population_sha256):
    artifact, sha = read_artifact(path)
    if (artifact.get('schema') != 'wm_phys.entrant_population.v1'
            or tuple(artifact['parameter_order']) != IDM_ORDER
            or artifact['source_population_sha256'] != source_population_sha256
            or not artifact['motion_validation_improved']
            or artifact['fit_worlds'] != list(range(969420, 969432))
            or artifact['conditioning_end_s'] != 300):
        raise ValueError('unsupported or unqualified entrant physical population artifact')
    mix = entries.process.birth_mix
    weights = torch.tensor(artifact['weights'], dtype=torch.float64)
    if not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError('invalid entrant physical population weights')
    for b, pool in enumerate(mix.pools):
        expected = pool.new_tensor(artifact['support'])
        actual = torch.stack([pool[:, PARAMETER_NAMES.index(name)] for name in IDM_ORDER], -1)
        if not torch.equal(actual, expected) or len(weights) != len(pool):
            raise ValueError('entrant posterior support differs from frozen physical population')
        mix.weights[b] = weights.clone()/weights.sum()
    return dict(entrant_population_sha256=sha, entrant_population_source=str(path),
        entrant_population_fit_worlds=artifact['fit_worlds'],
        entrant_population_motion_window_s=artifact['per_driver_motion_window_s'],
        current_driver_posteriors_changed=False, arrival_process_changed=False,
        cav_parameters_changed=False, physics_rule_changed=False)
