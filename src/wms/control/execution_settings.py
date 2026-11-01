"""Opt-in consistency of native/lane/admission settings with the fixed CAV plant.

The registered executor uses a motion-fitted CAV artifact independently of the
HDV encoder. Admission and lane/merge geometry must use that SAME physical CAV
vector when selected, rather than an encoder population that the executor will
ignore. This does not change the controller/executor, latent HDV draws or RNG.
"""
from pathlib import Path
import hashlib

import torch

from wms.model.driver_encoder import PARAMETER_NAMES


def bind_execution_settings(kwargs, entries, initial, path):
    from .execution import load_learned_driver
    settings = load_learned_driver(str(path))
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
    return dict(cav_settings_source='registered_motion_fitted_execution_driver',
        cav_settings_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        cav_settings=settings, external_executor_changed=False,
        hdv_settings_changed=False, latent_streams_changed=False)
