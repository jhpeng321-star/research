import json
from pathlib import Path
h=Path('/home/ubuntu')
out={}
for rel in ('e3final','e3final/runtime','mappospeed/production2','coopmwm20261104_launchcoop_src7'):
 p=h/rel
 out[rel]=[(q.name,q.is_dir(),q.stat().st_size) for q in p.iterdir() if q.name not in ('__pycache__',)]
p=h/'coopmwm20261104_launchcoop_src7/pytools/tools/leverprobe/backend.py'
out['backend_source']=p.read_text()
out['ENV_MANIFEST_meta']=(h/'cpuv_setup_20261105/ENV_MANIFEST.json').stat().st_size
print(json.dumps(out))
