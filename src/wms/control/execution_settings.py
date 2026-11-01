"""Opt-in consistency of native/lane/admission settings with the fixed CAV plant.

Mode2's actual executor uses the shared designed CAV law in sumo_idm, NOT the
retired learned-driver mirror used by legacy inferred_follow_cap. Admission,
native CF and lane/merge geometry must use the actual active actuator vector.
The designed CAV constants are part of the registered control law, not hidden
HDV settings or an assignment lookup. Legacy modes retain the learned fit.
No controller/executor equation, latent HDV draw or RNG changes here.
"""
from pathlib import Path
import hashlib
import inspect
import json

import torch

from wms.model.driver_encoder import PARAMETER_NAMES


def bind_execution_settings(kwargs, entries, initial, path):
    from .execution import load_learned_driver, mode
    if mode() == 2:
        from . import sumo_idm
        settings = {name:float(sumo_idm.CAV_PARAMS[name]) for name in
                    ('accel', 'decel', 'time_headway', 'min_gap')}
        settings['v0'] = float(sumo_idm.allowed_speed_scalar(0, 0.))
        provenance = dict(cav_settings_source='registered_exact_mode2_executor',
            cav_settings_sha256=hashlib.sha256(json.dumps(settings, sort_keys=True,
                separators=(',', ':')).encode()).hexdigest(),
            cav_settings_source_sha256=hashlib.sha256(Path(inspect.getfile(sumo_idm)).read_bytes()).hexdigest(),
            retired_learned_driver_used=False)
    else:
        settings = load_learned_driver(str(path))
        provenance = dict(cav_settings_source='registered_motion_fitted_execution_driver',
            cav_settings_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            retired_learned_driver_used=True)
    process = entries.process
    mix = process.birth_mix
    params = kwargs['driver_params']
    cav = initial['is_cav'].bool() & initial['present'].bool()
    for name, value in settings.items():
        params[name] = torch.where(cav, torch.full_like(params[name], value), params[name])
    for pool in mix.cav_pools:
        for name, value in settings.items():
            pool[:, PARAMETER_NAMES.index(name)] = value
    # PopulationAdmission and future SumoInsertion both hold this same mix.
    return dict(**provenance, cav_settings=settings, external_executor_changed=False,
        hdv_settings_changed=False, latent_streams_changed=False)
