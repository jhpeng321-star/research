"""Final evidence/file identity audit, after all own numerical jobs have completed."""
import json
from pathlib import Path
import sys

root = Path('/root/autodl-tmp/c4/v5')
repo = root/'repo_r13e_8c7ff0a1'
out = root/'diag_emerg/offline_train48_workers4'
assert json.loads((out/'POSTPROCESS_COMPLETE.json').read_text())['status'] == 'complete'
sys.path.insert(0,str(repo))
from tools.r13e_checks import offline as study  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

study.cpu_limits()
tables = {}
paths = {arm:out/f'onestep_{arm}.npz' for arm in study.ARMS}
paths['r9'] = out/'round9_reference/onestep_off.npz'
for arm,path in paths.items():
    with np.load(path,allow_pickle=False) as data:
        tables[arm] = (data['rows'],list(data['cols']))
assert all(cols == tables['off'][1] for rows,cols in tables.values())
assert np.array_equal(tables['emergency'][0],tables['emergency+gs'][0],equal_nan=True)
cols = tables['off'][1]
fixed = [i for i,c in enumerate(cols) if c not in ('fin','pre','off','dacc','ddec')]
assert np.array_equal(tables['off'][0][:,fixed],tables['r9'][0][:,fixed],equal_nan=True)
names = ['report.json','base_checks.json','gs_equivalence.json','linux_tests.log','POSTPROCESS_COMPLETE.json']
report = json.loads((out/'report.json').read_text())
base = json.loads((out/'base_checks.json').read_text())
assert base['one_step_windows'] == report['one_step_windows']
assert len(base['one_step_windows']) == 48 and len(report['rollouts']['rows']) == 96
assert study.file_digest(root/'runs/c4_wm_v10_round12c/wm_s.pt') == report['checkpoint_sha256']
result = {'status':'passed','source_provenance':study.source_provenance(),
          'checkpoint_unchanged':True,'round9_same_replay_geometry':True,
          'emergency_vs_emergency_gs_one_step_arrays_exact':True,
          'base_and_rollout_windows_identical':True,'one_step_windows':48,'paired_state_seed_rows':96,
          'npz_rows':{arm:len(rows) for arm,(rows,cols) in tables.items()},
          'cuda_initialized':torch.cuda.is_initialized(),
          'evidence_sha256':{name:study.file_digest(out/name) for name in names},
          'npz_sha256':{arm:study.file_digest(path) for arm,path in paths.items()},
          'postprocess_script_sha256':study.file_digest(root/'diag_emerg/postprocess_remote.py'),
          'gs_comparator_sha256':study.file_digest(repo/'tools/r13e_checks/gs_equivalence.py')}
assert not result['cuda_initialized']
(out/'final_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
