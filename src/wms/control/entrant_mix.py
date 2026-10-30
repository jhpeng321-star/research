"""Opt-in connection from candidate-A imagination to the qualified ERRSRC mix."""
from __future__ import annotations

import torch


def make_sampler(model, initial, kwargs, seed, row_keys):
    from tools.errsrc.inference import LearnedMix

    observed = kwargs.get('driver_history_mask', kwargs.get('history_mask'))
    if observed is None:
        raise ValueError('learned entrant mix needs observed pre-t0 history')
    params = kwargs.get('driver_params')
    if params is None:
        names = ('driver_history', 'driver_history_mask', 'history', 'history_mask', 'driver_defaults')
        _, _, params = model.encode_drivers(initial, **{k: kwargs[k] for k in names if k in kwargs})
    if params is None:
        raise ValueError('learned entrant mix needs inferred driver parameters')
    # WMFIX: entrant CAVs draw from the row's own observed t0 CAV inferred settings
    # (inert in the legacy slot-order path; the learned admission consumes it).
    cav_rows = initial['is_cav'].detach().bool() & initial['present'].detach().bool()
    cav_pools = [torch.stack([params[n][b, cav_rows[b]] for n in params], -1).detach()
                 for b in range(cav_rows.shape[0])]
    return LearnedMix(params, observed, seed, row_keys, cav_pools=cav_pools)
