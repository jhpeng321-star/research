"""MAIN17:52 exact-scope deletion admission, no scenario/truth content reads."""
import argparse
import json
import time
import subprocess
from .ls_validity_ops import remote
from . import ops
from .storage import write,read
from .protocol import sha,checked,REGISTRY


def audit_code():
    return '''from pathlib import Path
import os,json,time,shutil
target=Path('/home/ubuntu')/'mappospeed'/'production2'/'scenarios'
bad=('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth','vehicle_params','driver_params','vtype','assignment')
out=dict(target=str(target),exists=target.exists(),process_references=[],metadata_references=[],code_references=[],files=0,bytes=0,forbidden_named_entries=0,symlinks=0,unix=time.time())
roots=set();cfgs=set()
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  values=[]
  for field in ('cwd','exe'):
   try:values.append(os.readlink(proc/field))
   except FileNotFoundError:pass
  cmd=(proc/'cmdline').read_bytes().decode(errors='replace');env=(proc/'environ').read_bytes().decode(errors='replace');values.extend([cmd,env])
  for x in env.split(chr(0)):
   if x.startswith('PYTHONPATH='):
    roots.update(p for p in x.split('=',1)[1].split(':') if p.startswith('/home/ubuntu/coopmwm') and (p.endswith('/pytools') or p.endswith('/runtime_src')))
  args=cmd.split(chr(0))
  for x in args:
   if x.endswith('.sumocfg') and x.startswith('/home/ubuntu/') and not any(b in x.lower() for b in bad):cfgs.add(x)
  for fd in (proc/'fd').iterdir():
   try:values.append(os.readlink(fd))
   except FileNotFoundError:pass
  if any(str(target) in s for s in values):out['process_references'].append(proc.name)
 except (FileNotFoundError,ProcessLookupError):pass
for root in Path('/home/ubuntu').glob('coopmwm*'):
 if not root.is_dir() or root.is_symlink() or any(b in str(root).lower() for b in bad):continue
 if 'archive' in root.name or 'rolling' in root.name:continue
 for p in root.glob('*.json'):
  if any(b in str(p).lower() for b in bad) or p.is_symlink():continue
  if not any(k in p.name.upper() for k in ('BINDING','CONFIG','MANIFEST','RUNTIME','OWNER','DISPATCH')):continue
  if str(target) in p.read_text():out['metadata_references'].append(str(p))
for name in cfgs:
 p=Path(name)
 if p.is_file() and str(target) in p.read_text():out['metadata_references'].append(str(p))
for name in roots:
 root=Path(name)
 for directory,dirs,files in os.walk(root,followlinks=False):
  dirs[:]=[n for n in dirs if not any(b in str(Path(directory)/n).lower() for b in bad) and not (Path(directory)/n).is_symlink()]
  for name in files:
   p=Path(directory)/name
   if p.suffix!='.py' or p.is_symlink() or any(b in str(p).lower() for b in bad):continue
   if str(target) in p.read_text(errors='replace'):out['code_references'].append(str(p))
if target.is_dir():
 for directory,dirs,files in os.walk(target,followlinks=False):
  for name in dirs+files:
   p=Path(directory)/name
   if p.is_symlink():out['symlinks']+=1
   if any(b in str(p).lower() for b in bad):out['forbidden_named_entries']+=1
  dirs[:]=[n for n in dirs if not (Path(directory)/n).is_symlink() and not any(b in n.lower() for b in bad)]
  for name in files:
   p=Path(directory)/name
   if not p.is_symlink():out['files']+=1;out['bytes']+=p.stat().st_size
out['live_code_roots']=sorted(roots);out['live_sumo_configs_checked']=len(cfgs)
out['cpu_free']=shutil.disk_usage('/home/ubuntu').free
out['eligible_no_references']=not any(out[k] for k in ('process_references','metadata_references','code_references','symlinks'))
print(json.dumps(out))'''


def audit():
    code=audit_code()
    value=json.loads(remote('cpuQ',f'''import subprocess
p=subprocess.run(['sudo','-n','/home/ubuntu/miniconda3/bin/python','-B','-c',{code!r}],capture_output=True,text=True,timeout=70)
if p.returncode:raise RuntimeError(p.stderr)
print(p.stdout)''',timeout=85))
    write(ops.EVIDENCE/f'DISK1752_SCENARIO_AUDIT_{time.time_ns()}.json',value)
    print(json.dumps(value))
    return value


def register():
    paths=subprocess.run(['git','ls-files','--stage','--','mappospeed/production2/scenarios','production2/scenarios'],
        cwd=str(ops.LOCAL),capture_output=True,text=True,check=True)
    assert not paths.stdout.strip(),'generated scope must contain no Git tracked inputs'
    value=audit();assert value['eligible_no_references'] and value['exists']
    record=ops.EVIDENCE/'SCENARIO_DELETE_REGISTRATION_1752.json'
    line=('\nS-COOPMWM MAIN2026-11-03 17:52/18:00b scoped deletion: ONLY old generated '
        '/home/ubuntu/mappospeed/production2/scenarios leaf outputs. Git index audit '
        'finds no tracked paths under mappospeed/production2/scenarios or production2/scenarios; '
        'live qualified backend explicitly creates each SUMO case under its episode path/scenarios '
        '(tools/leverprobe/backend.py:113), not this old output root. Privileged all-process '
        'cwd/exe/FD/cmdline/environment, live SUMO config, public bindings/manifests/owners '
        'and live PYTHONPATH source audit required immediately before deletion. Preserve '
        'checkpoints/policy/logs and every file outside exact scope. Do not open scenario/'
        'truth/vType content; metadata-only inventory and inode/mtime/size proof. Skip '
        'symlinks, prohibited names, directories and files modified in last1800s; delete '
        'only audited matching plain-file leaves. V2V HOLD must be present and no jobs. '
        'Record exact removed paths/bytes/reference audit in append-only receipt; no new SUMO.\n')
    if not record.exists():write(record,dict(line=line,audit=value,new_sim_s=0,git_index_matches=paths.stdout,
        source_sha256=sha(__file__)))
    h=sha(record);registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(line+'amendmentSHA'+h+'\n')
    print(remote('cpuQ',f'''from pathlib import Path
import json
r=Path('/home/ubuntu/coopmwm20261103_cleanup1752');assert r.parent.is_dir();r.mkdir(exist_ok=True)
p=r/'AUTHORITY.json';v=json.loads({json.dumps(dict(read(record),amendment_sha256=h))!r})
if p.exists():assert json.loads(p.read_text())==v
else:
 with p.open('x') as f:json.dump(v,f)
print(json.dumps(dict(registered=True,sha256={h!r},new_sim_s=0)))''',timeout=40))


def delete():
    reg=read(ops.EVIDENCE/'SCENARIO_DELETE_REGISTRATION_1752.json')
    assert sha(__file__)==reg['source_sha256'],'registered deletion source changed'
    code='''from pathlib import Path
import json,time,os,stat,shutil,io,contextlib
r=Path('/home/ubuntu/coopmwm20261103_cleanup1752');p=r/'DELETION_RECEIPT.json'
if p.exists():print(p.read_text());raise SystemExit
assert (r/'AUTHORITY.json').is_file()
root=Path('/home/ubuntu/coopmwm20261103_v2v3');assert (root/'HOLD.json').exists()
assert not list((root/'jobs').iterdir()) and not (root/'V2V_STARTED.json').exists()
buf=io.StringIO()
with contextlib.redirect_stdout(buf):exec(compile(AUDIT,'<fresh-scenario-reference-audit>','exec'),{})
proof=json.loads(buf.getvalue());assert proof['eligible_no_references'] and proof['exists'],proof
target=Path(proof['target']);assert target==Path('/home/ubuntu')/'mappospeed'/'production2'/'scenarios'
assert not target.is_symlink()
forbidden=('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1')
manifest=[];now=time.time()
for directory,dirs,names in os.walk(target,followlinks=False):
 dirs[:]=[n for n in dirs if not (Path(directory)/n).is_symlink() and not any(b in n.lower() for b in forbidden)]
 for name in names:
  q=Path(directory)/name
  if any(b in str(q).lower() for b in forbidden) or q.is_symlink():continue
  st=q.stat()
  if not stat.S_ISREG(st.st_mode) or now-st.st_mtime<1800:continue
  manifest.append(dict(path=str(q),bytes=st.st_size,inode=st.st_ino,mtime_ns=st.st_mtime_ns))
with (r/'DELETE_MANIFEST.json').open('x') as f:json.dump(dict(reference_audit=proof,files=manifest),f);f.flush();os.fsync(f.fileno())
removed=[]
for e in manifest:
 q=Path(e['path']);st=q.lstat()
 if not stat.S_ISREG(st.st_mode) or (st.st_ino,st.st_mtime_ns,st.st_size)!=(e['inode'],e['mtime_ns'],e['bytes']):raise ValueError('generated file changed; stop')
 q.unlink();removed.append(e)
v=dict(unix=time.time(),target=str(target),removed=removed,removed_bytes=sum(e['bytes'] for e in removed),files=len(removed),reference_audit=proof,cpu_free=shutil.disk_usage('/home/ubuntu').free,new_sim_s=0,reason='MAIN17:52 old unreferenced regenerable scenario outputs')
with p.open('x') as f:json.dump(v,f);f.flush();os.fsync(f.fileno())
print(json.dumps({k:x for k,x in v.items() if k!='removed'}))'''
    # Send the audit through stdin to avoid self-references in audit process argv.
    payload='AUDIT='+repr(audit_code())+'\n'+code
    value=json.loads(remote('cpuQ',f'''import subprocess
p=subprocess.run(['sudo','-n','/home/ubuntu/miniconda3/bin/python','-B','-'],input={payload!r},capture_output=True,text=True,timeout=90)
if p.returncode:raise RuntimeError(p.stderr)
print(p.stdout)''',timeout=105))
    record=ops.EVIDENCE/'SCENARIO_DELETION_RECEIPT_1752.json'
    if not record.exists():write(record,value)
    h=sha(record);registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write('\nS-COOPMWM MAIN1752 deletion COMPLETE '+json.dumps({k:v for k,v in value.items() if k not in ('removed','reference_audit')})+' receiptSHA'+h+'\n')
    print(json.dumps({k:v for k,v in value.items() if k not in ('removed','reference_audit')}))


def adopt():
    code='''from pathlib import Path
import json,hashlib,shutil
r=Path('/home/ubuntu/coopmwm20261103_cleanup1752');manifest=json.loads((r/'DELETE_MANIFEST.json').read_text())
receipt=r/'DELETION_RECEIPT.json';recovered=False
if not receipt.exists():
 # The first execution's inline exec polluted p with the final scenario leaf.
 # Its own metadata JSON was written there AFTER all source leaves were removed.
 # Recover that exact output, never reread any truth or route file.
 source=Path(manifest['files'][-1]['path'])
 assert source==Path('/home/ubuntu/mappospeed/production2/scenarios/pid199686/s4139_m2600_nn_ckea6/scenario.sumocfg')
 assert source.is_file() and source.stat().st_size==4063316
 v=json.loads(source.read_text());assert v['removed']==manifest['files'] and v['files']==23319
 assert v['target']==str(Path('/home/ubuntu')/'mappospeed'/'production2'/'scenarios')
 source.replace(receipt);recovered=True
v=json.loads(receipt.read_text());assert v['removed']==manifest['files']
h=hashlib.sha256(receipt.read_bytes()).hexdigest();v.pop('removed')
v.update(full_receipt_sha256=h,full_receipt_path=str(receipt),current_cpu_free=shutil.disk_usage(r).free,metadata_receipt_recovered=recovered)
print(json.dumps(v))'''
    value=json.loads(remote('cpuQ',f'''import subprocess
p=subprocess.run(['sudo','-n','/home/ubuntu/miniconda3/bin/python','-B','-'],input={code!r},capture_output=True,text=True,timeout=30)
if p.returncode:raise RuntimeError(p.stderr)
print(p.stdout)''',timeout=45))
    record=ops.EVIDENCE/'SCENARIO_DELETION_RECEIPT_1752.json'
    if not record.exists():write(record,value)
    h=sha(record);registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write('\nS-COOPMWM MAIN1752 exact scenario cleanup COMPLETE; '
            f"files={value['files']} bytes={value['removed_bytes']}; full receipt at {value['full_receipt_path']} "
            'SHA'+value['full_receipt_sha256']+'; reporting variable-collision recovered, no truth data read; receiptSHA'+h+'\n')
    print(json.dumps({k:v for k,v in value.items() if k!='reference_audit'}))


def temp_audit():
    code='''from pathlib import Path
import os,json,shutil,time
bad=('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth','vehicle_params','driver_params','vtype','assignment')
paths=[p for p in Path('/tmp').glob('*.tgz') if not p.is_symlink() and not any(b in str(p).lower() for b in bad)]
out={str(p):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,inode=p.stat().st_ino,references=[],processes=[]) for p in paths}
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  values=[]
  for field in ('cwd','exe'):
   try:values.append(os.readlink(proc/field))
   except FileNotFoundError:pass
  values.extend([(proc/'cmdline').read_bytes().decode(errors='replace'),(proc/'environ').read_bytes().decode(errors='replace')])
  for fd in (proc/'fd').iterdir():
   try:values.append(os.readlink(fd))
   except FileNotFoundError:pass
  for p in paths:
   if any(str(p) in s for s in values):out[str(p)]['processes'].append(proc.name)
 except (FileNotFoundError,ProcessLookupError):pass
for root in Path('/home/ubuntu').glob('coopmwm*'):
 if not root.is_dir() or root.is_symlink() or any(b in str(root).lower() for b in bad):continue
 if 'archive' in root.name or 'rolling' in root.name or 'cleanup' in root.name:continue
 for p in root.glob('*.json'):
  if any(b in str(p).lower() for b in bad) or p.is_symlink():continue
  if not any(k in p.name.upper() for k in ('BINDING','CONFIG','MANIFEST','RUNTIME','OWNER','DISPATCH')):continue
  text=p.read_text()
  for q in paths:
   if str(q) in text:out[str(q)]['references'].append(str(p))
for p,v in out.items():v['eligible']=not v['processes'] and not v['references'] and time.time()-Path(p).stat().st_mtime>=1800
print(json.dumps(dict(candidates=out,eligible_bytes=sum(v['bytes'] for v in out.values() if v['eligible']),cpu_free=shutil.disk_usage('/home/ubuntu').free,tmp_same_device=Path('/tmp').stat().st_dev==Path('/home/ubuntu').stat().st_dev)))'''
    value=json.loads(remote('cpuQ',f'''import subprocess
p=subprocess.run(['sudo','-n','/home/ubuntu/miniconda3/bin/python','-B','-'],input={code!r},capture_output=True,text=True,timeout=40)
if p.returncode:raise RuntimeError(p.stderr)
print(p.stdout)''',timeout=55))
    write(ops.EVIDENCE/f'TEMP1752_AUDIT_{time.time_ns()}.json',value);print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('audit','register','delete','adopt','temp_audit'))
    globals()[p.parse_args().action]()
