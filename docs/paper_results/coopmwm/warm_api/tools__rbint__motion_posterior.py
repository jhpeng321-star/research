"""Condition the fitted physical population on a new causal observed history.

Reuse FIT's committed interval selector and the ACTIVE SUMO CF profile step;
SciPy is needed by FIT's estimator, not by causal runtime conditioning.
No population fitting, saved-driver-ID lookup or future observation is used.
"""
import ast
from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np
import torch


def profile_next_speed(before, gap, leader, support, dt):
    """Evaluate exactly the deployed SUMO core, including its lane-cap chain."""
    from dataclasses import replace
    from wms.control.fitted_hidden import IDM_ORDER
    from wms.physics.idm_sumo import SumoIDMParams, idm_iterations, sumo_idm_step
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float64)
    observed_speed = tensor(before)
    observed_gap = tensor(gap)
    has_leader = observed_gap.isfinite()
    # Same no-leader contract as physics.neighbours (SUMO free-road argument).
    observed_gap = torch.where(has_leader, observed_gap, torch.full_like(observed_gap, 1e6))
    leader_speed = torch.where(has_leader, tensor(leader), observed_speed)
    profiles = tensor(support)
    params = SumoIDMParams.from_driver_params({n:profiles[:, i] for i, n in enumerate(IDM_ORDER)})
    params = replace(params, emergency_decel=9.)  # identity-bound WM-PHYS recipe
    result, _ = sumo_idm_step(observed_speed, observed_gap, leader_speed,
        params.des_speed, params, dt_s=float(dt), iterations=idm_iterations(float(dt)))
    return result.numpy()

@lru_cache(maxsize=1)
def fit_kernels():
    source = Path(__file__).resolve().parents[1]/'rbfit/idm_fit.py'
    raw = source.read_bytes()
    tree = ast.parse(raw, filename=str(source))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'extract_rows']
    if len(functions) != 1:
        raise ValueError('unexpected FIT pure-motion source contract')
    namespace = dict(np=np)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['extract_rows'], profile_next_speed, hashlib.sha256(raw).hexdigest()


def chain_gaps(rows, arrays):
    """Use the deployed CF geometry for stable observed leaders across links."""
    from wms.physics.lane_chain import chain_s
    result = rows.copy()
    kin, vids, present = arrays['veh_kin'], arrays['vid'], arrays['veh_mask']
    lane = np.asarray(kin[..., 1:4]).argmax(-1)
    chain = chain_s(torch.as_tensor(lane), torch.as_tensor(kin[..., 0])).double().numpy()
    maps = [{int(v):i for i, v in enumerate(vids[t]) if present[t, i] and v >= 0}
            for t in range(len(kin))]
    for row in result:
        if row[7] < 0:
            continue
        tick = int(np.searchsorted(arrays['time_s'], row[0]))
        if tick >= len(kin) or arrays['time_s'][tick] != row[0]:
            raise ValueError('observed interval clock differs')
        ego, lead = maps[tick][int(row[1])], maps[tick][int(row[7])]
        row[5] = chain[tick, lead]-chain[tick, ego]-kin[tick, lead, 6]
    return result


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
    rows = chain_gaps(rows, arrays)
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
        motion_kernel_source_sha256=source_sha, motion_profile_step='active_sumo_idm_step',
        motion_gap_geometry='registered_lane_chain')
    return result, dict(runtime_motion_conditioning=True, saved_driver_id_lookup=False,
        posterior_source='runtime_motion_conditioned_fixed_population',
        motion_rows_by_observed_id=counts, conditioned_drivers=len(drivers),
        population_fallbacks=len(hdv_ids)-len(drivers), motion_kernel_source_sha256=source_sha)
