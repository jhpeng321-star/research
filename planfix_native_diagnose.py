"""Zero-SUMO loader diagnosis; subprocess creation is forbidden."""
import json,os,subprocess,sys,traceback
from pathlib import Path
source=Path('/home/ubuntu/coopmwm20261105_clean1_src_7cc6edaa')
os.nice(19-os.nice(0))
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
os.environ['SUMO_HOME']='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo'
os.environ['PATH']=os.environ['SUMO_HOME']+'/bin:/home/ubuntu/miniconda3/bin:'+os.environ['PATH']
for path in (source/'pytools',source/'runtime_src',source/'WM_PACKAGE',Path('/home/ubuntu/e3final/runtime/src'),Path('/home/ubuntu/e3final/runtime')):
 sys.path.append(str(path))
def refuse(*args,**kwargs):
 print('SUBPROCESS_REFUSED',repr(args[0])[:1000],flush=True)
 raise RuntimeError('ZERO_SUMO_DIAGNOSIS_SUBPROCESS_REFUSED')
subprocess.Popen=refuse
os.system=refuse
cfg=json.loads(Path('/home/ubuntu/coopmwm20261105_clean1/CONFIG.json').read_text())
print('CONFIG_RUNTIME',cfg.get('runtime'),cfg.get('checkpoint'),flush=True)
try:
 from tools.leverprobe import backend as exact
 print('EXACT_BACKEND',exact.__file__,flush=True)
 exact.setup(cfg['runtime'],cfg['checkpoint'])
 from tools.coopmwm.accel_protocol import install_seed_scope,guard
 install_seed_scope();exact.guard=lambda world:guard(world,'train')
 print('EXACT_SETUP_PASS',flush=True)
 import torch
 snap_path=Path('/home/ubuntu/coopmwm20261105_clean1/plan_fix/bank_verification_v4/bank_02713af428706f46/FROZEN_SNAPSHOT.pt')
 snap=torch.load(snap_path,weights_only=False,map_location='cpu')
 import time
 sys.path.insert(0,'/home/ubuntu/coopmwm20261105_clean1/plan_fix/ops_v4_ownerstopped')
 from planfix_library_runtime import library_plan,relative_hook
 from tools.coopmwm import sumo_verify as sv,bank_v2v
 from tools.coopmwm.v2v_runtime import from_weights
 inventory=json.loads(Path('/home/ubuntu/coopmwm20261105_clean1/plan_fix/BANK_INVENTORY.json').read_text())
 item=next(row for row in inventory['rows'] if row['id']=='bank_02713af428706f46')
 candidate=library_plan(item['source_plan'],'zero_sumo_diagnostic')
 spec=sv.build_branches(snap,[candidate])[0]
 root=Path('/home/ubuntu/coopmwm20261105_clean1/plan_fix')/('diagnostic_startup_run_'+str(time.time_ns()))
 print('ZERO_SUMO_NATIVE_RUNNER_CONSTRUCTION',flush=True)
 runner=sv.LocalBranchRunner(root,controller_hook=relative_hook,policy_loader=lambda weights:from_weights(weights,enabled=True).eval())
 with bank_v2v.enabled(True):runner.run(spec,snap)
except Exception:
 traceback.print_exc()
r=Path('/home/ubuntu/coopmwm20261105_clean1/plan_fix/bank_verification_v4')
for p in r.rglob('*'):
 if p.is_file() and ('sumo' in p.name.lower() or p.name.endswith('.log') or p.name=='CLAIM.json'):
  print('OWN_ARTIFACT',str(p),p.stat().st_size,flush=True)
