"""Install immutable parity clone into the private, initially empty home alias."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

r=Path('/root/mining20261106');home=r/'cpuv_home'
assert Path('/home/ubuntu').resolve()==home and (r/'EXPERIMENT_ID').read_text()=='S-MINING-20261106\n'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
archive=r/'cpuv_parity.tar.gz'
assert sha(archive)=='212737b1a2c63ead99632db15c40e5fdba8be40281b44e105a3716a1b64cfa4e'
assert not list(home.iterdir()),'no overwrite or repeated installation'
with tarfile.open(archive,'r:gz') as tar:
 first=tar.next();assert first.name=='PARITY_CLONE_MANIFEST.json'
 raw=tar.extractfile(first).read()
 assert hashlib.sha256(raw).hexdigest()=='f02fe78453c96be4d314dfbdaf0e05c7d40b6ab926b4c8f2ad2ac62168a45dd5'
 manifest=json.loads(raw)
 (home/first.name).write_bytes(raw)
 seen=set()
 for info in tar:
  if info.name==first.name:continue
  rel=Path(info.name)
  assert not rel.is_absolute() and '..' not in rel.parts
  assert info.name in manifest and info.name not in seen
  p=home/rel;p.parent.mkdir(parents=True,exist_ok=True)
  assert p.parent.resolve().is_relative_to(home)
  rec=manifest[info.name]
  if 'link' in rec:
   assert info.issym() and info.linkname==rec['link']
   resolved=Path(os.path.normpath(str(p.parent/info.linkname)))
   if Path(info.linkname).is_absolute():
    assert info.linkname.startswith('/home/ubuntu/')
   else:assert resolved.is_relative_to(home)
   p.symlink_to(info.linkname)
  else:
   assert info.isfile() and info.size==rec['size']
   with p.open('xb') as f:
    src=tar.extractfile(info)
    while True:
     b=src.read(4194304)
     if not b:break
     f.write(b)
   p.chmod(info.mode)
   assert sha(p)==rec['sha256'],str(rel)
  seen.add(info.name)
 assert seen==set(manifest)
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
           PYTHONDONTWRITEBYTECODE='1',LD_PRELOAD='/home/ubuntu/cpus_env/libmkl_intel_shim.so',
           SUMO_HOME='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo')
source='/home/ubuntu/coopmwm20261104_launchcoop_src7'
env['PYTHONPATH']=':'.join(source+'/'+p for p in ('pytools','runtime_src','WM_PACKAGE'))
env['PATH']=env['SUMO_HOME']+'/bin:/home/ubuntu/miniconda3/bin:'+env['PATH']
code="""import hashlib,importlib.metadata as m,json,sys
from pathlib import Path
from tools.leverprobe import backend
backend.setup('/home/ubuntu/e3final','/home/ubuntu/mappospeed/production2/policy_00169.pt')
pins={k:m.version(k) for k in ('eclipse-sumo','traci','sumolib','libsumo','numpy','scipy','pandas','torch')}
assert sys.version.startswith('3.10.16')
assert pins=={'eclipse-sumo':'1.27.1','traci':'1.27.1','sumolib':'1.27.1','libsumo':'1.27.1','numpy':'2.2.6','scipy':'1.15.3','pandas':'2.3.3','torch':'2.12.1+cpu'}
p=Path('/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo/bin/sumo')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='118bd81749f3fc0e4bd8c5840053ebe65a29770c940cb4e43e9eb561b06b7591'
print(json.dumps(dict(pins=pins,python=sys.version,backend_setup='PASS',sim_s=0)))
"""
p=subprocess.run(['/usr/bin/nice','-n','19','/home/ubuntu/miniconda3/bin/python','-B','-c',code],
                  env=env,capture_output=True,text=True,timeout=60)
record=dict(files=len(seen),archive_sha256=sha(archive),manifest_sha256=hashlib.sha256(raw).hexdigest(),
            returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,sim_s=0)
(r/'CLONE_QUALIFICATION.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
assert p.returncode==0
