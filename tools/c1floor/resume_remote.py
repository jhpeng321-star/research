"""Explicit transport for the owner-authorized fallback; preserves prior root."""
from remote import ssh, scp, HERE
from pathlib import Path
import sys

if __name__ == '__main__':
    action=sys.argv[1] if len(sys.argv)>1 else 'inspect'
    if action=='unpack':
        import tarfile
        p=HERE/'evidence'/sys.argv[2]
        with tarfile.open(p) as tar:
            for m in tar.getmembers():assert m.isfile() and '..' not in Path(m.name).parts
            tar.extractall(p.with_suffix(''),filter='data')
        raise SystemExit()
    if action=='stage':
        print(ssh("from pathlib import Path; p=Path('/root/c1floor20261031/fallback'); assert p.parent.is_dir(); p.mkdir(exist_ok=True); print(p)"))
        print(scp(HERE/'fallback.py','gpuR:/root/c1floor20261031/fallback.py'))
        print(scp(HERE/'test_fallback.py','gpuR:/root/c1floor20261031/test_fallback.py'))
        raise SystemExit()
    if action in ('a','b_canary','b_full'):
        print(ssh(f'''from pathlib import Path
import os,subprocess,json
r=Path('/root/c1floor20261031/fallback');assert r.is_dir()
env={{**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PATH':'/root/miniconda3/bin:'+os.environ['PATH']}}
with (r/{(action+'.log')!r}).open('x') as log:
 p=subprocess.Popen(['nice','-n','19','/root/miniconda3/bin/python','-B','-u','/root/c1floor20261031/fallback.py',{action!r}],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 owner=dict(pid=p.pid,start_ticks=Path(f'/proc/{{p.pid}}/stat').read_text().split()[21],phase={action!r})
 (r/{(action+'_PID.json')!r}).write_text(json.dumps(owner));print(json.dumps(owner))
'''))
        raise SystemExit()
    if action=='status':
        print(ssh('''from pathlib import Path
import json,os
r=Path('/root/c1floor20261031/fallback');print('load',os.getloadavg())
for p in r.glob('*_PID.json'):
 d=json.loads(p.read_text());q=Path('/proc')/str(d['pid']);print(p.name,d,'state',q.joinpath('stat').read_text().split()[2] if q.exists() else 'exited')
for p in sorted(Path('/root/c1floor20261031/jobs').glob('fallback_*/*.json')):
 if p.name=='result.json':
  d=json.loads(p.read_text());print(p.parent.name,dict(ok=d['ok'],actual=d['actual_sim_s'],original_bitwise=all(x['identical'] for x in d['reproduction'].values())))
 if p.name=='failure.json':print(str(p),p.read_text())
print('charged',sum(json.loads(s)['sim_s'] for p in (r/'accounting.jsonl',r.parent/'accounting.jsonl') for s in p.read_text().splitlines()))
'''))
        raise SystemExit()
    if action=='tests':
        print(ssh('''from pathlib import Path
import sys,json,hashlib,subprocess,os
r=Path('/root/c1floor20261031');p=Path('/root/autodl-tmp/c4/speedhead20261030/e3final/runtime')
sys.path.insert(0,str(r));from core import digest,DENIED
pre=json.loads((r/'PREFLIGHT_BINDING.json').read_text())
assert digest(p/'SOURCE_MANIFEST.json')==pre['source_manifest_sha256']
n=0
for name,h in json.loads((p/'SOURCE_MANIFEST.json').read_text()).items():
 if any(x in name.lower() for x in DENIED):continue
 assert digest(p/name)==h;n+=1
compile((r/'fallback.py').read_text(),str(r/'fallback.py'),'exec')
for name in ('test_core.py','test_fallback.py'):
 q=subprocess.run(['/root/miniconda3/bin/python','-B',str(r/name)],text=True,capture_output=True)
 print(q.stdout,q.stderr);assert q.returncode==0
receipt=dict(ok=True,frozen_files=n,source_sha256=digest(r/'fallback.py'),prior_actual=2400,prior_charged=9000)
import time
with (r/'fallback'/('CPU_'+str(time.time_ns())+'.json')).open('x') as f:json.dump(receipt,f,indent=1)
print(json.dumps(receipt))
'''))
        raise SystemExit()
    if action=='metadata':
        print(ssh('''from pathlib import Path
for seed in range(969400,969411):
 p=Path('/root/wmcalib/data/jobs')/f's{seed}_native'
 print(seed,{n:(p/n).exists() for n in ('start.pt','start_history.npz')})
'''))
        raise SystemExit()
    if action=='speed-api':
        print(ssh('''import traci,inspect
print(inspect.getsource(traci.vehicle.setPreviousSpeed))
'''))
        raise SystemExit()
    if action=='canary-effect':
        print(ssh('''from pathlib import Path
import numpy as np,json
r=Path('/root/c1floor20261031/jobs/fallback_b_s969400');d=json.loads((r/'result.json').read_text())
with np.load(r/'reference.npz',allow_pickle=False) as ref:
 meta=json.loads(str(ref['meta']))
 for k,name in enumerate(d['targets']):
  target=meta['vids'].index(name)
  with np.load(r/f'alt{k}.npz',allow_pickle=False) as alt:
   for at in (300.5,301.,301.5):
    a=np.flatnonzero((ref['t']==at)&(ref['vid']==target));b=np.flatnonzero((alt['t']==at)&(alt['vid']==target))
    print(k,at,{field:float(alt[field][b[0]])-float(ref[field][a[0]]) for field in ('s','v','a')})
'''))
        raise SystemExit()
    if action=='stop-amplitude':
        print(ssh('''from pathlib import Path
import os,json
p=Path('/root/c1floor20261031/fallback/STOP');assert not p.exists()
with p.open('x') as f:
 f.write('Amplitude audit: t300+0.01 speed request is not native-next-speed+0.01; stop before further branch admission.');f.flush();os.fsync(f.fileno())
print('durable STOP at next admission; no process signals')
'''))
        raise SystemExit()
    if action=='copy-starts':
        name='c1floor_start_metadata_20261031.tgz'
        print(ssh('''from pathlib import Path
import tarfile
p=Path('/tmp');assert p.is_dir()
with tarfile.open(p/'c1floor_start_metadata_20261031.tgz','x:gz') as tar:
 for seed in range(969400,969411):
  for name in ('start.pt','start_history.npz'):
   q=Path('/home/ubuntu/wmdata20261031/jobs')/f's{seed}_native'/name
   assert q.is_file();tar.add(q,arcname=f's{seed}_native/{name}',recursive=False)
print('22 original start metadata files archived; no simulation')
''',host='cpuQ',python='python3'))
        dest=HERE/'evidence'/name
        print(scp('cpuQ:/tmp/'+name,dest))
        print(scp(dest,'gpuR:/root/c1floor20261031/fallback/'+name))
        print(ssh('''from pathlib import Path
import tarfile
r=Path('/root/c1floor20261031')
with tarfile.open(r/'fallback/c1floor_start_metadata_20261031.tgz') as tar:
 for m in tar.getmembers():
  p=Path(m.name);assert m.isfile() and len(p.parts)==2 and p.name in ('start.pt','start_history.npz')
  assert not (r/'reference'/p).exists()
 tar.extractall(r/'reference',filter='data')
print('original metadata added to existing references')
'''))
        raise SystemExit()
    if action=='extract':
        print(scp(HERE/'fallback_extract.py','gpuR:/root/c1floor20261031/fallback_extract.py'))
        print(ssh('''import subprocess,os
p=subprocess.run(['nice','-n','19','/root/miniconda3/bin/python','-B','/root/c1floor20261031/fallback_extract.py'],capture_output=True,text=True,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''})
print(p.stdout,p.stderr);assert p.returncode==0
''',timeout=120))
        print(scp('gpuR:/root/c1floor20261031/fallback/LABELS.json',HERE/'evidence/FALLBACK_LABELS.json'))
        raise SystemExit()
    if action=='fetch':
        import tarfile,time
        name='fallback_'+str(time.time_ns())+'.tgz'
        print(ssh(f'''from pathlib import Path
import tarfile
r=Path('/root/c1floor20261031')
files=list((r/'fallback').glob('*.json'))+list((r/'fallback').glob('*.jsonl'))+list((r/'fallback').glob('*.log'))+list((r/'jobs').glob('fallback_*/*.json'))
with tarfile.open(r/'evidence'/{name!r},'w:gz') as tar:
 for p in files:tar.add(p,arcname=str(p.relative_to(r)),recursive=False)
print('receipts',len(files))
'''))
        dest=HERE/'evidence';assert dest.is_dir()
        print(scp('gpuR:/root/c1floor20261031/evidence/'+name,dest/name))
        with tarfile.open(dest/name) as tar:
            for m in tar.getmembers():assert m.isfile() and '..' not in Path(m.name).parts
            tar.extractall(dest/name.removesuffix('.tgz'),filter='data')
        print(name)
        raise SystemExit()
    print(ssh('''from pathlib import Path
import json
r=Path('/root/c1floor20261031')
print('ledger', (r/'accounting.jsonl').read_text())
print('blocked', (r/'BLOCKED.json').read_text())
print('coop_canary',Path('/root/coopaudit20261031/CANARY_VERIFIED.json').read_text())
for name in ('/root/autodl-tmp/c4/speedhead20261030/e3final/runtime/traffic_shepherd/c4/branching.py','/root/wmdata2/tools/ctrlarm/zero.py'):
 p=Path(name); text=p.read_text(); print(name,text[text.index('def load_state'):text.index('def run_branch')] if 'def load_state' in text else text)
'''))
