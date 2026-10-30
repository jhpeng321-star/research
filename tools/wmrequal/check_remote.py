"""CPU-only port checks, staged outside immutable numerical runtime."""
import base64
import json
from pathlib import Path
from remote import remote

ROOT=Path(__file__).resolve().parents[2]
files={'test_mix.py':ROOT/'tools/wmrequal/test_mix.py',
       'imagination_memory.py':ROOT/'src/wms/control/imagination_memory.py'}
data=base64.b64encode(json.dumps({n:base64.b64encode(p.read_bytes()).decode() for n,p in files.items()}).encode()).decode()
remote(f'''import base64,json,pathlib,subprocess,sys,os,time
p=pathlib.Path('/root/autodl-tmp/c4/wmrequal20261030')
c=p/'checks';c.mkdir(exist_ok=True)
for n,raw in json.loads(base64.b64decode({data!r})).items():
 (c/n).write_bytes(base64.b64decode(raw))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=f'{{p}}:{{p}}/src',
 WMREQUAL_MEMORY_CHECK=str(c/'imagination_memory.py'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
r=subprocess.run([sys.executable,'-B',str(c/'test_mix.py')],cwd=p,env=env,capture_output=True,text=True,timeout=180)
(c/'result.log').write_text(r.stdout+r.stderr)
(c/'RESULT.json').write_text(json.dumps(dict(code=r.returncode,time=time.time(),cuda_visible='',no_optimizer=True)))
print(r.stdout+r.stderr);assert r.returncode==0
''',timeout=240)
