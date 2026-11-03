"""MAIN17:52 completed diagnostic raw cleanup after byte-exact summary mirror."""
import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path
import time
import zlib
from . import ops
from .ls_validity_ops import remote
from .storage import read,write
from .protocol import checked,sha,REGISTRY

CONTROL='/home/ubuntu/coopmwm20261103_diagcleanup1752'
MIRROR=Path('E:/research-dense/docs/paper_results/coopmwm')


def plan():
    value=json.loads(remote('cpuQ',f'''import subprocess
code={PLAN_CODE!r}
p=subprocess.run(['sudo','-n','/home/ubuntu/miniconda3/bin/python','-B','-'],input=code,capture_output=True,text=True,timeout=60)
if p.returncode:raise RuntimeError(p.stderr)
print(p.stdout)''',timeout=75))
    assert MIRROR.is_dir()
    packed=value.pop('summaries');summaries=json.loads(zlib.decompress(base64.b64decode(packed)))
    mirrored=[]
    for name,item in summaries.items():
        assert Path(name).name==name and name.startswith('POLICY_SYNERGY_u') and name.endswith('.json')
        data=base64.b64decode(item['data']);assert hashlib.sha256(data).hexdigest()==item['sha256']
        p=MIRROR/name
        if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
        else:
            with p.open('xb') as f:f.write(data)
        mirrored.append(dict(path=str(p),sha256=item['sha256']))
    value.update(mirrored=mirrored,new_sim_s=0)
    p=ops.EVIDENCE/'DIAGNOSTIC_RAW_CLEANUP_PLAN_1752.json'
    if not p.exists():write(p,value)
    print(json.dumps({k:v for k,v in value.items() if k not in ('cases','mirrored')}))


PLAN_CODE='''from pathlib import Path
import os,json,time,base64,zlib,hashlib,shutil
root=Path('/home/ubuntu/coopmwm20261102_warm1');control=Path('/home/ubuntu/coopmwm20261103_diagcleanup1752')
assert control.parent.is_dir();control.mkdir(exist_ok=True)
out=control/'PLAN.json'
if out.exists():print(out.read_text());raise SystemExit
cases=[];summaries={};refs=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  for field in ('cwd','exe'):
   try:refs.append(os.readlink(proc/field))
   except FileNotFoundError:pass
  refs.extend([(proc/'cmdline').read_bytes().decode(errors='replace'),(proc/'environ').read_bytes().decode(errors='replace')])
  for fd in (proc/'fd').iterdir():
   try:refs.append(os.readlink(fd))
   except FileNotFoundError:pass
 except (FileNotFoundError,ProcessLookupError):pass
bad=('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1')
for parent in Path('/home/ubuntu').glob('coopmwm*'):
 if not parent.is_dir() or parent.is_symlink() or 'archive' in parent.name or 'cleanup' in parent.name or 'rolling' in parent.name:continue
 for p in parent.glob('*.json'):
  if any(b in str(p).lower() for b in bad) or p.is_symlink():continue
  if any(s in p.name.upper() for s in ('BINDING','CONFIG','RUNTIME','MANIFEST','OWNER','DISPATCH')):refs.append(p.read_text())
for name in ('coopmwm20261103_rolling1647','coopmwm20261103_rolling1647_v3'):
 for p in (Path('/home/ubuntu')/name).glob('batch_*/MANIFEST.json'):
  if not (p.parent/'ARCHIVE_MOVE.json').exists():refs.append(p.read_text())
for name,suffix in (('policy_synergy_1205_v1',''),('policy_synergy_isolated_1226_v2','_ISOLATED_V2')):
 for case in (root/name).glob('u[0-9][0-9][0-9]_[BR]'):
  summary=root/('POLICY_SYNERGY_'+case.name+suffix+'.json');raw=case/'branches'
  if not summary.exists() or not raw.is_dir() or raw.is_symlink():continue
  data=summary.read_bytes();v=json.loads(data)
  if v.get('status')!='complete' or time.time()-summary.stat().st_mtime<1800:continue
  if any(str(case) in s for s in refs):continue
  entries=[];blocked=False
  for directory,dirs,names in os.walk(raw,followlinks=False):
   if any(any(b in n.lower() for b in bad) or (Path(directory)/n).is_symlink() for n in dirs+names):blocked=True;break
   for n in names:
    p=Path(directory)/n;st=p.stat();entries.append(dict(path=str(p),bytes=st.st_size,inode=st.st_ino,mtime_ns=st.st_mtime_ns))
  if blocked:continue
  h=hashlib.sha256(data).hexdigest();summaries[summary.name]=dict(data=base64.b64encode(data).decode(),sha256=h)
  cases.append(dict(raw=str(raw),summary=str(summary),summary_sha256=h,entries=entries,bytes=sum(e['bytes'] for e in entries)))
v=dict(cases=cases,files=sum(len(c['entries']) for c in cases),bytes=sum(c['bytes'] for c in cases),case_count=len(cases),cpu_free=shutil.disk_usage(root).free,
 summaries=base64.b64encode(zlib.compress(json.dumps(summaries).encode())).decode(),unix=time.time())
with out.open('x') as f:json.dump(v,f)
print(json.dumps(v))'''


def register():
    value=read(ops.EVIDENCE/'DIAGNOSTIC_RAW_CLEANUP_PLAN_1752.json')
    for item in value['mirrored']:assert sha(item['path'])==item['sha256']
    tracked=subprocess.run(['git','ls-files','--stage','--','policy_synergy_1205_v1','policy_synergy_isolated_1226_v2'],cwd=str(ops.LOCAL),capture_output=True,text=True,check=True)
    assert not tracked.stdout.strip(),'raw generated scopes must not contain Git tracked inputs'
    record=ops.EVIDENCE/'DIAGNOSTIC_RAW_CLEANUP_REGISTRATION_1752.json'
    line=('\nS-COOPMWM MAIN17:52 raw-diagnostic deletion: completed TRAIN original/isolated '
        'synergy branches only, summary status=complete/age>=1800s. Every summary '
        'byte-exact mirrored+SHA-verified in E:/research-dense/docs/paper_results/coopmwm '
        'BEFORE any deletion; root reports/claims/RESULT receipts outside raw branches kept. '
        'Exclude any live process cwd/exe/FD/argv/environment or public binding/config/'
        'owner/manifest reference and every unfinished rolling archive manifest. '
        'Exact metadata inventory/inode/mtime/size and fresh FD/reference/summary-SHA '
        'proof before removing approved raw trees. Quiesce ONLY owned rolling mover '
        '4116186 after boot/start/argv/no-child proof using SIGSTOP; recheck no children, '
        'and SIGCONT in finally before returning, so no concurrent batch races. '
        'No source/checkpoint/replay_bank/'
        'grade/final_receipt path. These generated output trees have no Git tracked inputs. '
        f"Quote {value['case_count']}complete cases/{value['bytes']}bytes; no SUMO.\n")
    if not record.exists():write(record,dict(line=line,plan=value,new_sim_s=0))
    h=sha(record);registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(line+'amendmentSHA'+h+'\n')
    payload=dict(read(record),amendment_sha256=h)
    payload['plan']={k:v for k,v in payload['plan'].items() if k!='cases'}
    code=f'''from pathlib import Path
import json
p=Path({CONTROL!r})/'AUTHORITY.json';v=json.loads({json.dumps(payload)!r})
v['plan']['cases']=json.loads((p.parent/'PLAN.json').read_text())['cases']
if p.exists():assert json.loads(p.read_text())==v
else:
 with p.open('x') as f:json.dump(v,f)
print(json.dumps(dict(registered=True,sha256={h!r},new_sim_s=0)))'''
    print(remote('cpuQ',f'''import subprocess
p=subprocess.run(['sudo','-n','/home/ubuntu/miniconda3/bin/python','-B','-'],input={code!r},capture_output=True,text=True,timeout=30)
if p.returncode:raise RuntimeError(p.stderr)
print(p.stdout)''',timeout=45))


def delete():
    import textwrap
    authority=read(ops.EVIDENCE/'DIAGNOSTIC_RAW_CLEANUP_REGISTRATION_1752.json')
    for item in authority['plan']['mirrored']:assert sha(item['path'])==item['sha256']
    fresh=PLAN_CODE.split("for name,suffix in")[0].replace("if out.exists():print(out.read_text());raise SystemExit",'')
    code=f'''from pathlib import Path
import json,os,time,shutil,hashlib,re
control=Path({CONTROL!r});receipt=control/'DELETION_RECEIPT.json'
if receipt.exists():print(receipt.read_text());raise SystemExit
authority=json.loads((control/'AUTHORITY.json').read_text());plan=authority['plan']
ns={{}};exec(compile({fresh!r},'<fresh-diagnostic-reference-audit>','exec'),ns);refs=ns['refs']
root=Path('/home/ubuntu/coopmwm20261102_warm1');removed=[];skipped=[]
for case in plan['cases']:
 raw=Path(case['raw']);summary=Path(case['summary'])
 assert raw.name=='branches' and re.fullmatch('u[0-9]{{3}}_[BR]',raw.parent.name)
 assert raw.parent.parent in (root/'policy_synergy_1205_v1',root/'policy_synergy_isolated_1226_v2')
 if not raw.exists():skipped.append(dict(raw=str(raw),reason='already retired by archive'));continue
 if any(str(raw.parent) in s for s in refs):skipped.append(dict(raw=str(raw),reason='fresh live/manifest reference'));continue
 assert hashlib.sha256(summary.read_bytes()).hexdigest()==case['summary_sha256']
 assert json.loads(summary.read_text())['status']=='complete'
 expected={{e['path']:e for e in case['entries']}};present={{}}
 for directory,dirs,names in os.walk(raw,followlinks=False):
  assert not any((Path(directory)/n).is_symlink() for n in dirs+names)
  for n in names:
   p=Path(directory)/n;st=p.stat();present[str(p)]=(st.st_ino,st.st_mtime_ns,st.st_size)
 if set(present)!=set(expected):skipped.append(dict(raw=str(raw),reason='inventory changed since plan'));continue
 assert all(present[p]==(e['inode'],e['mtime_ns'],e['bytes']) for p,e in expected.items())
 shutil.rmtree(raw);removed.append(dict(raw=str(raw),bytes=case['bytes'],summary_sha256=case['summary_sha256']))
v=dict(unix=time.time(),removed=removed,skipped=skipped,removed_bytes=sum(e['bytes'] for e in removed),cpu_free=shutil.disk_usage(root).free,new_sim_s=0)
with receipt.open('x') as f:json.dump(v,f);f.flush();os.fsync(f.fileno())
print(json.dumps(v))'''
    start=code.index("for case in plan['cases']:");end=code.index('v=dict(unix=time.time()',start)
    body=code[start:end]
    pause='''import signal
signal.signal(signal.SIGHUP,signal.SIG_IGN)
owner=json.loads(Path('/home/ubuntu/coopmwm20261103_rolling1647_v3/OWNER.json').read_text())
assert owner['pid']==4116186 and owner['start_ticks']=='71133854'
proc=Path('/proc',str(owner['pid']))
assert proc.exists() and (proc/'stat').read_text().split()[21]==owner['start_ticks']
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==owner['boot_id']
assert b'/home/ubuntu/coopmwm20261103_rolling1647_v3/rolling_archive.py' in (proc/'cmdline').read_bytes()
assert not (proc/'task'/str(owner['pid'])/'children').read_text().strip(),'mover busy: retry after current batch'
os.kill(owner['pid'],signal.SIGSTOP)
'''
    code=code[:start]+pause+'try:\n'+textwrap.indent(
        "assert not (proc/'task'/str(owner['pid'])/'children').read_text().strip()\n"+
        "ns2={};exec(compile("+repr(fresh)+",'<quiesced-reference-audit>','exec'),ns2);refs=ns2['refs']\n"+body,' ')+'''finally:
 if proc.exists() and (proc/'stat').read_text().split()[21]==owner['start_ticks']:
  os.kill(owner['pid'],signal.SIGCONT)
'''+code[end:]
    compile(code,'<diagnostic-cleanup-worker>','exec')
    value=json.loads(remote('cpuQ',f'''import subprocess
p=subprocess.run(['sudo','-n','/home/ubuntu/miniconda3/bin/python','-B','-'],input={code!r},capture_output=True,text=True,timeout=75)
if p.returncode:raise RuntimeError(p.stderr)
print(p.stdout)''',timeout=90))
    p=ops.EVIDENCE/'DIAGNOSTIC_RAW_DELETION_RECEIPT_1752.json'
    if not p.exists():write(p,value)
    h=sha(p);registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write('\nS-COOPMWM completed diagnostic raw cleanup '
            f"cases={len(value['removed'])} bytes={value['removed_bytes']} new_sim_s=0 receiptSHA{h}\n")
    print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('plan','register','delete'))
    globals()[p.parse_args().action]()
