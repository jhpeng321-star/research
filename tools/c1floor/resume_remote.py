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
 if p.name in ('result.json','failure.json','IDENTITY.json'):print(str(p),p.read_text())
for p in r.glob('*.log'):print(p.name,p.read_text()[-5000:])
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
