"""Read-only remote inventory and dependency grep, excluding restricted trees."""
import base64
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def remote(host, code, data=None, timeout=120):
    encoded = base64.b64encode(code.encode()).decode()
    cmd = f"/root/miniconda3/bin/python -c \"import base64;exec(base64.b64decode('{encoded}'))\""
    p = subprocess.run(["ssh", "-o", "BatchMode=yes", host, cmd], input=data,
                       capture_output=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(p.stderr.decode() + p.stdout.decode())
    return p.stdout.decode()


INVENTORY = r'''
import os,json,pathlib,subprocess,shutil
v=pathlib.Path('/root/autodl-tmp/c4/v5')
bad=('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth','vtype')
protected=('data','main14_arm','pairft','repo_main14','repo_r14b_7adf5663','repo_r13e_65cc714e')
rows=[]
def measure(p):
 total=0; links=[]; denied=[]; files=0
 for root,ds,fs in os.walk(p,followlinks=False):
  for name in list(ds):
   x=pathlib.Path(root)/name
   if x.is_symlink(): links.append(str(x)); ds.remove(name)
   elif any(t in str(x).lower() for t in bad): denied.append(str(x)); ds.remove(name)
  for name in fs:
   x=pathlib.Path(root)/name
   if x.is_symlink(): links.append(str(x)); continue
   if any(t in str(x).lower() for t in bad) or x.suffix.lower()=='.xml': denied.append(str(x)); continue
   s=x.stat(); total+=s.st_blocks*512; files+=1
 return dict(bytes=total,files=files,symlinks=links,denied=denied)
for parent in (v,v/'runs'):
 for p in parent.iterdir():
  if not p.is_dir() or p.is_symlink() or p.name in protected: continue
  if p.name.startswith(('c4_wm_v10_round13e_','c4_wm_v10_round14_')): continue
  if any(t in str(p).lower() for t in bad): continue
  rows.append(dict(path=str(p),**measure(p)))
print(json.dumps(dict(free=shutil.disk_usage(v).free,candidates=sorted(rows,key=lambda x:-x['bytes'])),indent=1))
'''

GREP = r'''
import os,json,pathlib,sys,subprocess
req=json.load(sys.stdin); needles=req['needles']
bad=('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth','vtype')
skip={'data','runs','.git','__pycache__','.pytest_cache','.cache','miniconda3','venv','archive','archive_gpuR','boxsync'}
roots=set(); processes=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit(): continue
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if not any(x in cmd for x in ('python','bash','sh ')): continue
  if 'base64.b64decode' in cmd: continue
  cwd=(p/'cwd').resolve()
  if str(cwd).startswith(('/root/autodl-tmp/c4','/root/c4')):
   processes.append(dict(pid=int(p.name),cwd=str(cwd),cmd=cmd)); roots.add(cwd)
  for word in cmd.split():
   if word.startswith('/root/autodl-tmp/c4/') and word.endswith(('.py','.sh','.json','.toml')):
    roots.add(pathlib.Path(word).parent)
 except (OSError,PermissionError): pass
roots.add(pathlib.Path('/root/autodl-tmp/c4/v5'))
matches=[]; scanned={}; links=[]
for base in sorted(roots):
 for root,ds,fs in os.walk(base,followlinks=False):
  ds[:]=[d for d in ds if d not in skip and not any(t in str(pathlib.Path(root)/d).lower() for t in bad)]
  # Only active repo source subtrees and operational configs outside repositories.
  for d in list(ds):
   x=pathlib.Path(root)/d
   if x.is_symlink(): links.append(str(x)); ds.remove(d)
   elif root==str(pathlib.Path('/root/autodl-tmp/c4/v5')) and (d.startswith(('repo_','chk_','diag_'))):
    if not any(x==r or x in r.parents for r in roots): ds.remove(d)
  for f in fs:
   x=pathlib.Path(root)/f
   if str(x) in scanned or x.is_symlink() or any(t in str(x).lower() for t in bad): continue
   if x.suffix not in ('.py','.sh','.json','.toml','.yaml','.yml','.cfg','.ini'): continue
   if x.stat().st_size>5_000_000: continue
   text=x.read_text(errors='replace'); scanned[str(x)]=True
   for n in needles:
    for i,line in enumerate(text.splitlines(),1):
     if n in line: matches.append(dict(needle=n,path=str(x),line=i,text=line[:1000]))
print(json.dumps(dict(processes=processes,roots=list(map(str,roots)),scanned_files=len(scanned),matches=matches,symlinks=links),indent=1))
'''


def main():
    action = sys.argv[1]
    if action == 'inventory':
        out = remote('gpuR', INVENTORY)
        (ROOT / 'tools/mainstart/inventory.json').write_text(out)
        print(out)
    elif action == 'grep':
        needles = sys.argv[2:]
        for h in ('gpuR', 'gpuB'):
            out = remote(h, GREP, json.dumps({'needles': needles}).encode(), timeout=180)
            (ROOT / f'tools/mainstart/grep_{h}.json').write_text(out)
            doc = json.loads(out)
            print(h, json.dumps({'processes': doc['processes'], 'scanned_files': doc['scanned_files'],
                                 'matches': doc['matches']}))
    elif action == 'archive':
        # Only inert *work scratch outputs; preserve every RESULT/config/gate at its original path.
        relative = ['chk_r14/a/midrun_work', 'chk_r14/a/postrun_work',
                    'chk_r14/b/midrun_work', 'chk_r14/b/postrun_work',
                    'chk_r13e/a/midrun_work', 'chk_r13e/a/postrun_work']
        for host in ('gpuR', 'gpuB'):
            report = json.loads(remote(host, GREP, json.dumps({'needles': relative}).encode(), timeout=180))
            (ROOT / f'tools/mainstart/work_grep_{host}.json').write_text(json.dumps(report, indent=1))
            assert not report['matches'], (host, report['matches'])
            print(host, 'active dependency grep clean', report['scanned_files'])
        code = r'''
import pathlib,os,json,hashlib,subprocess,shutil,base64,sys
relative=json.load(sys.stdin); root=pathlib.Path('/root/autodl-tmp/c4/v5')
dest='/root/autodl-tmp/c4/archive_gpuR/S-MAINSTART-20261026'
bad=('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth','vtype')
def manifest(p):
 out={}; links=[]
 for d,ds,fs in os.walk(p,followlinks=False):
  for n in ds+fs:
   x=pathlib.Path(d)/n
   assert not any(t in str(x).lower() for t in bad), str(x)
   if x.is_symlink(): links.append(str(x))
  for n in fs:
   x=pathlib.Path(d)/n; assert x.suffix.lower()!='.xml'
   if not x.is_symlink():
    h=hashlib.sha256()
    with x.open('rb') as f:
     for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    out[x.relative_to(p).as_posix()]=h.hexdigest()
 assert not links, links
 return out,links
assert root.is_dir()
records=[]
subprocess.run(['ssh','gpuB', 'test -d /root/autodl-tmp/c4 && mkdir -p '+dest],check=True)
for rel in relative:
 p=root/rel; assert p.is_dir() and not p.is_symlink() and p.name.endswith('_work')
 files,links=manifest(p); before=shutil.disk_usage(root).free
 # Verify destination parent exists before rsync creates the leaf.
 parent=dest+'/'+str(pathlib.Path(rel).parent)
 subprocess.run(['ssh','gpuB','test -d '+dest+' && mkdir -p '+parent],check=True)
 target=dest+'/'+rel
 subprocess.run(['rsync','-a','--timeout=120',str(p)+'/', 'gpuB:'+target+'/'],check=True)
 verify="import pathlib,hashlib,json,sys; root=pathlib.Path(sys.argv[1]); files=json.load(sys.stdin); actual={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}; assert actual==files; print(len(actual))"
 subprocess.run(['ssh','gpuB','/root/miniconda3/bin/python -c '+__import__('shlex').quote(verify)+' '+target],input=json.dumps(files).encode(),check=True)
 assert manifest(p)==(files,links), 'source changed while archiving'
 # Enumerated immediately before recursive removal; no symlink traversal permitted.
 enumerated=[str(pathlib.Path(d)/n) for d,ds,fs in os.walk(p,followlinks=False) for n in ds+fs if (pathlib.Path(d)/n).is_symlink()]
 assert not enumerated
 shutil.rmtree(p)
 record=dict(source=str(p),archive='gpuB:'+target,files=len(files),manifest_sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),symlinks=enumerated,free_before=before,free_after=shutil.disk_usage(root).free)
 records.append(record); print(json.dumps(record),flush=True)
print(json.dumps({'archives':records,'free':shutil.disk_usage(root).free}))
'''
        out = remote('gpuR', code, json.dumps(relative).encode(), timeout=1200)
        (ROOT / 'tools/mainstart/ARCHIVE_RECEIPT.jsonl').write_text(out)
        print(out)


if __name__ == '__main__':
    main()
