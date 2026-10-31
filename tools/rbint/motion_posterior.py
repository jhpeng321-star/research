"""Condition the fitted physical population on a new causal observed history.

Reuse FIT's committed interval selector and faithful profile step verbatim;
SciPy is needed by FIT's estimator, not by these two pure NumPy functions.
No population fitting, saved-driver-ID lookup or future observation is used.
"""
import ast
from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np
import torch

@lru_cache(maxsize=1)
def fit_kernels():
    source = Path(__file__).resolve().parents[1]/'rbfit/idm_fit.py'
    raw = source.read_bytes()
    tree = ast.parse(raw, filename=str(source))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('extract_rows', 'next_speed')]
    constants = [n for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'EPS' for t in n.targets)]
    if len(functions) != 2 or len(constants) != 1:
        raise ValueError('unexpected FIT pure-motion source contract')
    namespace = dict(np=np)
    exec(compile(ast.Module(body=constants+functions, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['extract_rows'], namespace['next_speed'], hashlib.sha256(raw).hexdigest()


def condition_population(base, motion_fit, policy_path, start, history, *, kernels=None):
    """Return a fresh fitted-state reader in this export's own observable ID space."""
    if start['t0'] != 300.:
        raise ValueError('registered conditioning time is300s')
    times = np.asarray(history['time_s'], dtype=float)
    kin = np.asarray(history['kin'])
    present = np.asarray(history['mask'], dtype=bool)
    vids = np.asarray(history['vid_index'])
    if (kin.shape[:2] != present.shape or vids.shape != present.shape
            or times.shape != (len(kin),) or not np.isfinite(times).all()
            or times[-1] != 300. or (times > 300.).any() or (np.diff(times) <= 0).any()):
        raise ValueError('motion conditioning needs causal, identity-aligned pre-t0 frames')
    selected = start['state']['present'][0].cpu().numpy().astype(bool)
    cav = start['state']['is_cav'][0].cpu().numpy().astype(bool)
    hdv_ids = vids[-1, selected & ~cav]
    if (hdv_ids < 0).any() or len(np.unique(hdv_ids)) != len(hdv_ids):
        raise ValueError('invalid observed HDV identities')
    # Ownership is observed and constant by ID. Unknown past egos are excluded;
    # they remain visible leaders, without inventing their driver/CAV settings.
    arrays = dict(time_s=times, veh_kin=kin, veh_mask=present, vid=vids,
                  is_cav=~np.isin(vids, hdv_ids))
    extract, predict, source_sha = fit_kernels() if kernels is None else kernels
    rows, _, _ = extract(arrays, start=240., stop=300.)
    if len(rows) and ((rows[:, 0] < 240).any() or (rows[:, 0]+rows[:, 9] > 300).any()):
        raise ValueError('FIT selector returned noncausal intervals')
    support = np.asarray(base.population['support'], dtype=float)
    drivers, counts = {}, {}
    for vid in hdv_ids:
        observed = rows[rows[:, 1] == vid]
        counts[str(int(vid))] = len(observed)
        if not len(observed):
            continue  # exact learned-population fallback, never a fake posterior
        dt = observed[:, 9]
        if not np.equal(dt, dt[0]).all():
            raise ValueError('registered profile step requires uniform recorded dt')
        predictions = predict(observed[:, 3, None], observed[:, 5, None],
                              observed[:, 6, None], support, float(dt[0]))
        tensor = lambda x: torch.as_tensor(x, dtype=torch.float64)
        driver = motion_fit.condition_motion(tensor(predictions), tensor(observed[:, 3]),
            tensor(observed[:, 4]), tensor(dt), tensor(observed[:, 0]+dt),
            t0_s=300., policy_path=policy_path)
        drivers[int(start['seed']), int(vid)] = driver
    result = base.with_motion_posteriors(drivers, t0_s=300.)
    result.provenance.update(runtime_conditioning=True,
        saved_driver_id_lookup=False, conditioning_interval_s=[240., 300.],
        motion_kernel_source_sha256=source_sha)
    return result, dict(runtime_motion_conditioning=True, saved_driver_id_lookup=False,
        posterior_source='runtime_motion_conditioned_fixed_population',
        motion_rows_by_observed_id=counts, conditioned_drivers=len(drivers),
        population_fallbacks=len(hdv_ids)-len(drivers), motion_kernel_source_sha256=source_sha)
