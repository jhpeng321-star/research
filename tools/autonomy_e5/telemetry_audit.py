"""Compare only observable requested/issued command ledgers for equal trajectories."""
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
os.nice(19)
import numpy as np  # noqa: E402 - thread environment is fixed before numerical imports

R = Path('/home/ubuntu/e3long')
OP = R/'operations_e5'
spec = importlib.util.spec_from_file_location('e5_analysis', OP/'analyze_shard.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
shard = int(sys.argv[1])
rows = []
keys = ('issued_acceleration_mps2', 'requested_acceleration_mps2', 'issued_target_speed_mps',
        'lane_command_target', 'lane_request_target')
for seed in range(973000+shard, 973228, 2):
    paths = [R/'main/e3long_policy'/f'shard{shard}'/'jobs'/f's{seed}_ours',
             R/'main/e3long_coopoff'/f'shard{shard}'/('retry_jobs' if seed in A.FAILED else 'jobs')/f's{seed}_ours']
    paths = [p/'deploy/episode/episode.npz' for p in paths]
    for p in paths:
        A.digest(p)
    with np.load(A.safe(paths[0]), allow_pickle=False) as on, np.load(A.safe(paths[1]), allow_pickle=False) as off:
        assert np.array_equal(on['control_time_s'], off['control_time_s'])
        assert np.array_equal(on['time_s'], off['time_s'])
        assert np.array_equal(on['vid_index'], off['vid_index'])
        assert np.array_equal(on['is_cav'], off['is_cav'])
        metrics = {}
        for key in keys:
            a, b = on[key], off[key]
            assert a.shape == b.shape
            finite = np.isfinite(a) & np.isfinite(b)
            unequal = ~((a == b) | (np.isnan(a) & np.isnan(b)))
            metrics[key] = dict(unequal_elements=int(unequal.sum()), finite_pairs=int(finite.sum()),
                finite_mask_identical=bool(np.array_equal(np.isfinite(a), np.isfinite(b))),
                max_abs_diff=float(np.abs(a[finite].astype(float)-b[finite].astype(float)).max()) if finite.any() else 0.)
        rows.append(dict(seed=seed, commands=metrics))
receipt = dict(status='PASS', shard=shard, worlds=114, rows=rows, input_sha256=A.INPUT_HASHES,
    aggregates={k: dict(worlds_different=sum(r['commands'][k]['unequal_elements'] > 0 for r in rows),
                        unequal_elements=sum(r['commands'][k]['unequal_elements'] for r in rows),
                        max_abs_diff=max(r['commands'][k]['max_abs_diff'] for r in rows)) for k in keys})
A.write(OP/'TELEMETRY_AUDIT.json', receipt)
print(json.dumps({k: v for k, v in receipt.items() if k not in ('rows', 'input_sha256')}))
