import json
from pathlib import Path
import subprocess
r=Path('/root/mining20261106/wm_worker01')
out={}
for name in ('MINING_WM_DISPATCH.json','MINING_WM_OWNER.json','GPU_MODEL_READY.json','MINING_WM_FAULT.json','FRESH_STARTUP.json','FRESH_QUALIFICATION.json'):
 p=r/name
 if p.is_file():out[name]=json.loads(p.read_bytes())
out['gates']={p.name:json.loads(p.read_bytes()) for p in r.glob('STARTUP*.json')}
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True)
out['log_tail']=(r/'MINING_WM.log').read_text()[-12000:]
print(json.dumps(out,sort_keys=True))
