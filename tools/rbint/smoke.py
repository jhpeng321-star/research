"""Two registered FIT starts, 60 s closed cohort, independent no-correction guard."""
import argparse
import copy
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
    C4_ALLOWED_ROOTS='/root/autodl-tmp/c4:/root/wmcalib:/root/rbeval20261031',
    C4_EXECUTION_PARITY='2', C4_SUMO_LEADER_RULE='1',
    C4_EXECUTION_DRIVER=str(ROOT/'configs/execparity_cav_idm_v1.json'))

import numpy as np
import torch

from tools.rbint.audit import audit_closed_cohort, forbid_learned_corrections
from tools.rbint.recipe import load_candidate, native_controller
from tools.rbeval.core import digest, load, reference, save_new
from wms.control.hidden_state import plan_hidden_state_futures
from wms.model.driver_encoder import PARAMETER_NAMES
from wms.model.entries import build_entrance_history, _counts_counter
from wms.physics.batched import road_end_exit_bound
from wms.physics.state import state_kin


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    out = Path(args.out)
    assert out.parent.is_dir() and not out.exists()
    expected = load('/root/wmcalib/expected_hashes.json')
    results = []
    begin = time.monotonic()
    for variant in ('A', 'B'):
        os.environ['WMPHYS_VARIANT'] = variant
        models, receipt = load_candidate(
            ['/root/autodl-tmp/c4/wmcombined_wm2/wm_s.pt'], [str(ROOT/'configs/wm_phys.toml')], 'cpu')
        model = models[0]
        policy = torch.load('/root/autodl-tmp/c4/setupaudit20261030/policy_000190.pt',
                            map_location='cpu', weights_only=False)
        model.cav_exec_cap = policy['config']['cav_exec_cap']
        assert model.cav_exec_cap != 'off'
        namespace = dict(torch=torch)
        reference('worker', namespace)
        for seed in (969420, 969421):
            data = Path(f'/root/wmcalib/data/jobs/s{seed}_ctrl0')
            for name in ('start.pt', 'start_history.npz', 'result.json'):
                assert digest(data/name) == expected[f'jobs/s{seed}_ctrl0/{name}']
            start = torch.load(data/'start.pt', map_location='cpu', weights_only=False)
            assert start['t0'] == 300
            with np.load(data/'start_history.npz', allow_pickle=False) as f:
                raw = {k: f[k] for k in ('kin', 'mask', 'target_mask')}
            state = copy.deepcopy(start['state'])
            keep = state['present'][0].nonzero().flatten()
            state = {k: v[:, keep].clone() for k, v in state.items()}
            h, m = namespace['history_tensors'](raw, history_len=model.history_len)
            kwargs = dict(history=h[:, keep], history_mask=m[:, keep])
            if model.long_history:
                h, m = namespace['history_tensors'](raw, model.history_s, model.history_stride_s)
                kwargs.update(driver_history=h[:, keep], driver_history_mask=m[:, keep])
            with torch.no_grad(), forbid_learned_corrections(model) as calls:
                if variant == 'A':
                    future = plan_hidden_state_futures(model, state, kwargs,
                        future_sampling='hidden_state', n_samples=1, seed=190, row_keys=[seed])[0]
                    kwargs.update(future.kwargs)
                else:
                    _, embedding, _ = model.encode_drivers(state, **kwargs)
                    kin = state_kin(state)
                    defaults = model._default_driver_parameters(kin, *kin.shape[:2])
                    kwargs.update(driver_embedding=embedding,
                        driver_params={n: defaults[..., i] for i, n in enumerate(PARAMETER_NAMES)})
                state['exit_s_m'] = torch.full_like(state['s_m'], road_end_exit_bound(model.physics))
                roll = model.rollout(copy.deepcopy(state), n_steps=120,
                                     cav_controller=native_controller(), **kwargs)
                gate = audit_closed_cohort(state['present'], roll, exit_s_m=state['exit_s_m'])
                assert not roll['raw_residual_mps2'].count_nonzero()
            results.append(dict(variant=variant, seed=seed, t0=300, horizon_s=60,
                source_sha256=start['initial_state_sha256'], encoder_calls=len(calls),
                encoder_exact=receipt['encoder_exact'], guard_pass=True, **gate))
    save_new(out, dict(schema='wmphys-smoke/1', passed=True, results=results,
        recipe_sha256=digest(ROOT/'configs/wm_phys.toml'),
        checkpoint_sha256=digest('/root/autodl-tmp/c4/wmcombined_wm2/wm_s.pt'),
        seconds=time.monotonic()-begin))
    print('SMOKE PASS', len(results), 'rollouts', flush=True)


if __name__ == '__main__':
    main()
