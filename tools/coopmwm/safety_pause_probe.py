"""Read-only registered TRAIN safety-pause evidence; never clears or retries jobs."""
import json
import time
from . import ops
from .ls_collect_ops import ROOT
from .ls_validity_ops import remote
from .storage import write


def run():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import sys,json,time,hashlib
r=Path({ROOT!r});sys.path.insert(0,'/home/ubuntu/coopmwm20261103_exposurecpu1/pytools')
from tools.coopmwm.accel_storage import read,charged
pause=r/'PAUSE_SAFETY.json';e=read(pause);pairs=[]
worlds=e.get('worlds',e.get('evidence',{{}}).get('worlds',[]))
for seed in worlds:
 a=r/'jobs'/('warm_u006_A_'+str(seed));b=r/'jobs'/('replay0735_warm_u006_B_'+str(seed))
 A=read(a/'RESULT.json');B=read(b/'RESULT.json')
 pairs.append(dict(seed=seed,same_world=A['world']==B['world'],
  A_hard_brake=A['metrics']['900']['hard_brake'],B_hard_brake=B['metrics']['900']['hard_brake'],
  delta=B['metrics']['900']['hard_brake']-A['metrics']['900']['hard_brake'],
  B_executed_plan_steps=sum(bool(x['sample'].get('behaviour',{{}}).get('executed')) for x in B['steps']),
  A_result_sha256=hashlib.sha256((a/'RESULT.json').read_bytes()).hexdigest(),
  B_result_sha256=hashlib.sha256((b/'RESULT.json').read_bytes()).hexdigest()))
print(json.dumps(dict(unix=time.time(),pause=e,pause_sha256=hashlib.sha256(pause.read_bytes()).hexdigest(),
 pairs=pairs,unique_worlds=len({{p['seed'] for p in pairs}}),
 checkpoints={{a:[u for u in (5,6,7) if (r/(a+'_checkpoint_'+str(u).zfill(3)+'.pt')).exists()] for a in ('B','R')}},
 u6_results={{a:sum((p/'RESULT.json').exists() for p in (r/'jobs').glob('replay0735_warm_u006_'+a+'_*')) for a in ('B','R')}},
 charged_sim_s=charged(r),physics_started=False,pause_cleared=False)))''',timeout=40))
    p=ops.EVIDENCE/f'SAFETY_PAUSE_PROBE_{time.time_ns()}.json';write(p,value)
    print(json.dumps(dict(receipt=str(p),**value)))


if __name__=='__main__':run()
