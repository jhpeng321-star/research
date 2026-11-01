"""Two FIT-start native closed-cohort free rollouts of the real fitted-LC recipe.

This is an engineering gate, not the integrated arrival/action distributional
gate. No recorded future states, masks, lanes or driver IDs enter the rollout.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
    C4_ALLOWED_ROOTS='/root/autodl-tmp/c4:/root/wmcalib:/root/rbeval20261031',
    C4_EXECUTION_PARITY='2', C4_SUMO_LEADER_RULE='1',
    C4_EXECUTION_DRIVER=str(ROOT/'configs/execparity_cav_idm_v1.json'), WMPHYS_VARIANT='FULL')

import numpy as np
import torch

from tools.rbint.audit import audit_closed_cohort, forbid_learned_corrections
from tools.rbint.observations import load_motion_history, fitted_inputs
from tools.rbint.recipe import load_candidate, native_controller, fitted_state_at_start
from tools.rbeval.core import digest, reference, save_new
from tools.rblane.runtime_history import lane_arrays
from wms.control.hidden_state import plan_hidden_state_futures
from wms.physics.batched import road_end_exit_bound
from wms.physics.lane_continuation import warm_lane_memory


def lane_counts(state, rollout, dt):
    lanes = torch.cat((state['lane'][:, None], rollout['lanes']), 1)
    pos = torch.cat((state['s_m'][:, None], rollout['positions']), 1)
    present = torch.cat((state['present'][:, None], rollout['masks']), 1).bool()
    good = present[:, :-1] & present[:, 1:]
    change = good & (lanes[:, :-1] != lanes[:, 1:])
    result = {}
    for source in (0, 1, 2):
        for segment in ('upstream', 'merge', 'downstream', 'ramp'):
            region = (lanes[:, :-1]==2) if segment=='ramp' else ((lanes[:, :-1]<2) & (
                (pos[:, :-1]<0) if segment=='upstream' else
                (pos[:, :-1]>75.44) if segment=='downstream' else
                ((pos[:, :-1]>=0) & (pos[:, :-1]<=75.44))))
            selected = good & (lanes[:, :-1]==source) & region
            n, exposure = int((change & selected).sum()), float(selected.sum())*dt
            if exposure:
                result[f'{segment}/{source}'] = dict(count=n, exposure_veh_s=exposure, rate_per_veh_h=3600*n/exposure)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists() or not args.out.parent.is_dir() or os.getloadavg()[0] >= 20:
        raise ValueError('new output and spare CPU required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    begin = time.time()
    recipe = ROOT/'configs/wm_phys_rb2_lane.toml'
    models, construction = load_candidate(['/root/autodl-tmp/c4/wmcombined_wm2/wm_s.pt'], [str(recipe)], 'cpu')
    model = models[0]
    policy = torch.load('/root/autodl-tmp/c4/setupaudit20261030/policy_000190.pt', map_location='cpu', weights_only=False)
    model.cav_exec_cap = policy['config']['cav_exec_cap']
    namespace = dict(torch=torch)
    reference('worker', namespace)
    results = []
    for world in (969420, 969421):
        start, observed = load_motion_history(world)
        selected = start['state']['present'][0].nonzero().flatten()
        initial = {name:value[:, selected].clone() for name,value in start['state'].items()}
        ids, _, _ = fitted_inputs(initial, world, start['initial_state_sha256'], model._wmphys_arrival_fit)
        with np.load(Path('/root/wmcalib/data/jobs')/f's{world}_ctrl0/start_history.npz', allow_pickle=False) as f:
            history = {name:np.array(f[name], copy=True) for name in ('kin', 'mask', 'target_mask')}
        h, mask = namespace['history_tensors'](history, history_len=model.history_len)
        kwargs = dict(history=h[:, selected], history_mask=mask[:, selected])
        if model.long_history:
            h, mask = namespace['history_tensors'](history, model.history_s, model.history_stride_s)
            kwargs.update(driver_history=h[:, selected], driver_history_mask=mask[:, selected])
        fitted, conditioning = fitted_state_at_start(model, world)
        with torch.no_grad(), forbid_learned_corrections(model) as calls:
            futures = plan_hidden_state_futures(model, initial, kwargs, future_sampling='hidden_state',
                n_samples=2, seed=190, row_keys=[world], fitted_state=fitted, vehicle_ids=ids, t0_s=300.)
            for sample, future in enumerate(futures):
                state = copy.deepcopy(initial)
                state.update(warm_lane_memory(lane_arrays(start, observed), t0_s=300., physics=model.physics,
                    driver_params=future.kwargs['driver_params'], vehicle_ids=ids), vehicle_id=ids.clone())
                state['exit_s_m'] = torch.full_like(state['s_m'], road_end_exit_bound(model.physics))
                roll = model.rollout(copy.deepcopy(state), n_steps=120, cav_controller=native_controller(model),
                    **kwargs, **future.kwargs)
                gate = audit_closed_cohort(state['present'], roll, exit_s_m=state['exit_s_m'])
                results.append(dict(world=world, sample=sample, **gate, conditioning=conditioning,
                    future=future.receipt, lane_segment=lane_counts(state, roll, model.physics.dt_s),
                    initial_warm_wishes=int(state['lc2013_keep_right'].count_nonzero()),
                    final_lc_state_finite=all(torch.isfinite(roll['state'][key]).all().item()
                        for key in ('lc2013_left', 'lc2013_right', 'lc2013_keep_right', 'lc2013_lookahead'))))
                print(json.dumps(dict(world=world, sample=sample, conserved=gate['conserved'],
                    elapsed_s=time.time()-begin)), flush=True)
    save_new(args.out, dict(status='PASS', source='real strict-loaded checkpoint/recipe',
        recipe_sha256=digest(recipe), construction=construction, results=results,
        inference_only=True, cuda_used=False, future_observations=False, elapsed_s=time.time()-begin))


if __name__ == '__main__':
    main()
