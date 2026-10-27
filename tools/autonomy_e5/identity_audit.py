"""Zero-SUMO provenance audit prompted by exactly equal ON/OFF trajectories."""
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
os.nice(19)
R = Path('/home/ubuntu/e3long')
OP = R/'operations_e5'
spec = importlib.util.spec_from_file_location('e5_analysis', OP/'analyze_shard.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
import numpy as np  # noqa: E402 - thread environment is fixed before numerical imports
import torch  # noqa: E402
torch.set_num_threads(1)
shard = int(sys.argv[1])
data = A.doc(OP/'ANALYSIS_DATA.json')
summaries = []
for seed in range(973000+shard, 973228, 2):
    paths = [R/'main/e3long_policy'/f'shard{shard}'/'jobs'/f's{seed}_ours',
             R/'main/e3long_coopoff'/f'shard{shard}'/('retry_jobs' if seed in A.FAILED else 'jobs')/f's{seed}_ours']
    docs = [A.doc(p/'deploy/DEPLOY_SUMMARY.json') for p in paths]
    for folder, summary in zip(('cycle1', 'coopoff_cycle1'), docs):
        assert summary['status'] == 'complete' and summary['controller'] == 'policy'
        assert summary['policy_ckpt'] == str(R/'inputs'/folder/'policy_latest.pt')
        assert summary['executed_control_ticks'] == {'policy': 600}
    ta, tb = [A.digest(p/'traj_base.npz') for p in paths]
    assert ta == tb
    assert docs[0]['measurement'] == docs[1]['measurement']
    summaries.append(dict(seed=seed, deployment_paths=[d['policy_ckpt'] for d in docs],
                          trajectories_byte_identical=True, trajectory_sha256=ta,
                          metrics_exactly_identical=True))
checkpoints = [A.safe(R/'inputs'/f/'policy_latest.pt') for f in ('cycle1', 'coopoff_cycle1')]
payloads = [torch.load(p, map_location='cpu', weights_only=True) for p in checkpoints]
tensor_comparison = {}
for key in ('policy', 'group_encoder'):
    x, y = [p[key] for p in payloads]
    assert set(x) == set(y)
    tensor_comparison[key] = dict(tensors=len(x), different=[k for k in x if not torch.equal(x[k], y[k])],
        max_abs_diff=max(float((x[k].double()-y[k].double()).abs().max()) for k in x))
sys.path[:0] = [str(R/'runtime/src'), str(R/'runtime')]
from wms.control.policy import load_policy_fn  # noqa: E402 - frozen runtime path above
executors = [load_policy_fn(str(p), device='cpu') for p in checkpoints]
loaded = []
for executor, payload in zip(executors, payloads):
    state = executor.policy.state_dict()
    assert all(torch.equal(state[k], v) for k, v in payload['policy'].items())
    loaded.append(dict(exact_stored_policy_tensors=True, residual_on_native=executor.residual_on_native,
                       native_lanes=executor.native_lanes, deterministic=executor.deterministic,
                       hold=executor.hold, extra_accel_noise=executor.extra_accel_noise))
# Inspect observable recording schema only; don't open scenario, truth or assignments.
sample = R/'main/e3long_policy'/f'shard{shard}'/'jobs'/f's{973000+shard}_ours'/'deploy/episode/episode.npz'
with np.load(A.safe(sample), allow_pickle=False) as z:
    episode_array_keys = z.files
policy_source = A.read(R/'runtime/src/wms/control/policy.py').decode()
deploy_source = A.read(R/'runtime/src/wms/control/deploy.py').decode()
native_source = A.read(R/'runtime/src/wms/control/nativeinit.py').decode()
receipt = dict(status='PASS', shard=shard, worlds=114, summaries=summaries,
    tensor_comparison=tensor_comparison, loaded_executors=loaded,
    checkpoint_metadata=[{k: p.get(k) for k in ('format', 'residual_on_native', 'lane_control', 'update_count')} for p in payloads],
    checkpoint_sha256=[A.digest(p) for p in checkpoints], episode_array_keys=episode_array_keys,
    input_sha256=A.INPUT_HASHES)
A.write(OP/'IDENTITY_AUDIT.json', receipt)
# Human-readable source snippets are returned for diagnosis, not model inference.
print(json.dumps(dict(receipt={k: v for k, v in receipt.items() if k not in ('summaries', 'input_sha256')},
    policy_forward=policy_source[policy_source.index('        if self.residual_on_native:', policy_source.index('    def forward')):][:1500],
    native_composition=native_source[native_source.index('def sumo_residual_commands'):][:2300],
    deploy_policy=[line for line in deploy_source.splitlines() if 'policy_fn(' in line or 'exec_fns' in line or 'exec_fn(' in line])))
