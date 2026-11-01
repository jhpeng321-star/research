"""Causal prefix-conditioned physical support check; no traffic outcomes read."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tools.rb2break.birth_population import acceleration_crps
from tools.rb2break.run import DEV
from tools.rbint.motion_posterior import chain_gaps, fit_kernels
from tools.rbint.observations import load_motion_history, load_observations
from tools.rbeval.core import digest, save_new
from wms.physics.motion_constraints import constrain_acceleration
from wms.physics.motion_fit import MotionFit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    assert args.out.parent.is_dir() and not args.out.exists()
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    directory = root/'artifacts/rbfit/shrinkage_v1'
    motion = MotionFit(directory)
    support = np.array(motion.population['support'])
    policy = directory/'idm_shrinkage.json'
    extract, predict, _ = fit_kernels()
    rows_out, audits = [], []
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float64)
    for world in DEV:
        start, history = load_motion_history(world)
        selected = start['state']['present'][0].numpy().astype(bool)
        cav = start['state']['is_cav'][0].numpy().astype(bool)
        hdv = history['vid_index'][-1, selected & ~cav]
        arrays = dict(time_s=history['time_s'], veh_kin=history['kin'],
            veh_mask=history['mask'], vid=history['vid_index'],
            is_cav=~np.isin(history['vid_index'], hdv))
        rows, _, _ = extract(arrays, start=240., stop=300.)
        rows = chain_gaps(rows, arrays)
        baseline, corrected = [], []
        for identity in hdv:
            own = rows[rows[:, 1] == identity]
            prefix = own[own[:, 0]+own[:, 9] <= 270.]
            target = own[own[:, 0] >= 270.]
            if not len(target):
                continue
            def predictions(rr):
                if not len(rr):
                    return np.empty((0, len(support)))
                assert np.all(rr[:, 9] == .5)
                return predict(rr[:, 3, None], rr[:, 5, None], rr[:, 6, None], support, .5)
            post = motion.condition_motion(tensor(predictions(prefix)), tensor(prefix[:, 3]),
                tensor(prefix[:, 4]), tensor(prefix[:, 9]), tensor(prefix[:, 0]+prefix[:, 9]),
                t0_s=270., policy_path=policy)
            weights = torch.zeros(len(support), dtype=torch.float64)
            weights[post['support_indices']] = tensor(post['weights'])
            bounded, audit = constrain_acceleration(weights, tensor(support),
                tensor(prefix[:, 3]), tensor(prefix[:, 4]), tensor(prefix[:, 9]))
            audits.append(dict(world=world, identity=int(identity), **audit))
            predicted = (predictions(target)-target[:, 3, None])/target[:, 9, None]
            observed = (target[:, 4]-target[:, 3])/target[:, 9]
            baseline.extend(acceleration_crps(predicted, observed, weights.numpy()))
            corrected.extend(acceleration_crps(predicted, observed, bounded.numpy()))
        assert baseline
        rows_out.append(dict(world=world, rows=len(baseline), baseline_crps=float(np.mean(baseline)),
                             bounded_crps=float(np.mean(corrected)), observable_hashes=load_observations(world)[2]))
        print('WORLD', world, rows_out[-1], flush=True)
    old = float(np.mean([r['baseline_crps'] for r in rows_out]))
    new = float(np.mean([r['bounded_crps'] for r in rows_out]))
    result = dict(schema='rb2break-motion-envelope/1', worlds=list(DEV),
        conditioning_interval_s=[240, 270], validation_interval_s=[270, 300],
        cohort='registered t0=300 surviving HDVs with later stable-leader motion',
        baseline_crps_mps2=old, bounded_crps_mps2=new, improved=new < old,
        per_world=rows_out, audits=audits, source_sha256=digest(__file__),
        constraint_sha256=digest(root/'src/wms/physics/motion_constraints.py'),
        population_sha256=digest(directory/'idm_population.json'), policy_sha256=digest(policy),
        traffic_outcomes_used=False, qualified=False)
    save_new(args.out, result)
    print(json.dumps({k:result[k] for k in ('baseline_crps_mps2','bounded_crps_mps2','improved')}))


if __name__ == '__main__':
    main()
