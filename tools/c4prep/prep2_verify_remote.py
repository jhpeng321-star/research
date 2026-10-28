"""Verify the immutable A preparation; write truthful fail-closed readiness receipts."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

r = Path('/root/autodl-tmp/c4/restart20261026')
out = r / 'arm/loop/c4prep'
pointer = json.loads((out / 'CURRENT_RUNTIME.json').read_text())
code = Path(pointer['runtime'])
source = json.loads((code / 'SOURCE.json').read_text())
assert source['commit'] == pointer['commit']
for name, digest in source['files'].items():
    assert hashlib.sha256((code / name).read_bytes()).hexdigest() == digest
assert 'configs/native_cav_head_v5.json' in source['files']
os.environ['C4_ALLOWED_ROOTS'] = '/root/autodl-tmp/c4'
sys.path[:0] = [str(code / 'tools/c4prep'), str(code / 'src'), str(code)]
from common import qualified_gate, read, sha, write_new
from calibrate import inventory
from launch import descriptor, run
from wms.control.counterfactual import CounterfactualConfig
from wms.control.policy14 import guarded_real_credit_weights
from wms.control.train_cli import _read_config
from wms.control.real_credit14 import load_multipliers

preflight = inventory()
preflight_path = out / 'INPUT_PREFLIGHT_PREP2_v1.json'
write_new(preflight_path, preflight)
gate_error = None
try:
    horizon, gate = qualified_gate(True)
except (AssertionError, FileNotFoundError, ValueError) as e:
    horizon = None
    gate_error = f'{type(e).__name__}: {e}'
cfg = descriptor(True)
cfg['blockers'] = list(preflight['blockers'])
if gate_error:
    cfg['blockers'].append('compatible WM3 registered horizon unavailable: ' + gate_error)
cfg['blockers'].append('empirical K8 native-anchor calibration and real A GPU receipt do not exist')
config_path = out / 'CONFIG_A_ON_PREP2_v1.json'
write_new(config_path, cfg)
refusal = None
try:
    run(config_path, dry=True)
except ValueError as e:
    refusal = str(e)
assert refusal is not None, 'an unresolved descriptor must refuse before GPU dispatch'
raw = _read_config(code / 'configs/c4_policy14_c4prep_on.toml')
cf = CounterfactualConfig(**raw['counterfactual'])
credit_path = r / 'arm/loop/sumocredit/REAL_CREDIT_c03.a1.json'
weights, weights_receipt = guarded_real_credit_weights(cf.weights(180),
    load_multipliers(credit_path), individual_floor=raw['trainer']['real_credit_individual_floor'])
assert weights['coop'] > 0 and weights['individual'] >= weights['coop']
gate_log = r / 'arm/loop/checks/c03.a2/HORIZON_GATE.log'
groups = []
for line in gate_log.read_text().splitlines():
    if line.startswith('{'):
        record = json.loads(line)
        if 'group' in record:
            groups.append(record['group'])
status_path = r / 'AUTONOMY_STATUS.json'
before = read(status_path)
receipt = {'task': 'S-C4PREP2', 'status': 'BLOCKED_C4A',
    't': datetime.now(timezone.utc).isoformat(), 'source_commit': source['commit'],
    'runtime': str(code), 'source_files_sha_verified': len(source['files']),
    'actor_shas_verified': {str(c['cycle']): c['actor_verified'] for c in preflight['cycles']},
    'preflight_sha256': sha(preflight_path),
    'cycles': [{'cycle': c['cycle'], 'pairs': c['pairs'],
                'episodes': len(c['episodes']), 'missing_clock_episodes':
                sum(bool(e['missing_observable_arrays']) for e in c['episodes'])}
               for c in preflight['cycles']],
    'trusted_horizon_s': horizon, 'gate_error': gate_error,
    'gate_groups_logged': len(groups), 'gate_log_sha256': sha(gate_log),
    'real_credit_sha256': sha(credit_path), 'real_credit_weights': weights_receipt,
    'anchor_fitted': False, 'useful_deviation_suppression': None,
    'real_gpu_updates': 0, 'dry_dispatch_refusal': refusal,
    'autonomy_status_preserved': before.get('status'),
    'new_sim_s': 0, 'full_policy_launched': False,
    'gpu_memory': subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total,memory.used',
                                          '--format=csv,noheader'], text=True).strip(),
    'blockers': cfg['blockers']}
write_new(out / 'PREP2_VERIFIED_v1.json', receipt)
# Keep the live S-C3WMFT status/phase/owner fields: report prep as a subordinate
# fact, never overwrite a running peer gate with a false READY_C4A state.
current = read(status_path)
current['c4a_preparation'] = {'status': 'BLOCKED_C4A', 'task': 'S-C4PREP2',
    'receipt': str(out / 'PREP2_VERIFIED_v1.json'), 'source_commit': source['commit'],
    'ready': False, 'blockers': cfg['blockers']}
temp = r / 'AUTONOMY_STATUS.c4prep2.tmp'
with temp.open('x') as stream:
    json.dump(current, stream, indent=1)
temp.replace(status_path)
print(json.dumps(receipt, indent=1))
