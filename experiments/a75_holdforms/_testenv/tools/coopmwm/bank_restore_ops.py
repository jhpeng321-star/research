"""Restore only certified archived bank tapes/envelopes, with exact SHA proof."""
import argparse
import json
import time
from . import ops
from .ls_validity_ops import remote
from .storage import read,write
from .protocol import checked,sha,REGISTRY

CONTROL='/home/ubuntu/coopmwm20261103_restore_bank1900'


def plan():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,sys,hashlib
sys.path.insert(0,'/home/ubuntu/coopmwm20261103_compositereleasecpu1/pytools')
from tools.coopmwm import replay_bank as rb
root=Path('/home/ubuntu/coopmwm20261102_warm1');ls=Path('/home/ubuntu/coopmwm20261102_lscollect1')
control=Path({CONTROL!r});assert control.parent.is_dir();control.mkdir(exist_ok=True)
p=control/'PLAN.json'
if p.exists():print(p.read_text());raise SystemExit
conf=rb.confirmations(root,'B');wanted=set();index={{}}
for r in rb.bank_decisions(root,'B',sources=[str(ls)]):
 if not conf.get(r['verification_id'],{{}}).get('admitted'):continue
 for branch in (r['branches'][0],r['branches'][r['chosen']+1]):
  d=Path(r['_branches'])/r['verification_id']/branch['branch']
  for n in ('RESULT.json','RESULT.json.gz'):
   q=d/n
   if not q.exists():wanted.add(str(q))
for label in (ls/'labels').glob('*.json'):
 r=json.loads(label.read_text());v=r['verification']['verification_id']
 c=conf.get(v,{{}})
 if not ((c.get('cooperation') or {{}}).get('cooperation_gain_veh_s') or 0)>0:continue
 q=ls/'pending'/(r['search_id']+'.json.gz')
 if not q.exists():wanted.add(str(q))
home=Path('/home/ubuntu')
for name in ('coopmwm20261103_archive1450_v2','coopmwm20261103_archive1450_v3'):
 base=home/name
 for receipt in base.glob('ARCHIVE_MOVE_batch*.json'):
  v=json.loads(receipt.read_text());verified={{e['relative']:e for e in v['verified']}}
  for rel in v['deleted_sources']:
   e=verified[rel];source=str(home/rel)
   if source in wanted:index[source]=dict(source=source,archive=v['destination']+'/'+rel,sha256=e['sha256'],bytes=e['bytes'],proof=str(receipt))
for name in ('coopmwm20261103_rolling1647','coopmwm20261103_rolling1647_v3'):
 for receipt in (home/name).glob('batch_*/ARCHIVE_MOVE.json'):
  v=json.loads(receipt.read_text());m=receipt.parent/'MANIFEST.json';proof=json.loads((receipt.parent/'REMOTE_VERIFIED.json').read_text())
  assert hashlib.sha256(m.read_bytes()).hexdigest()==v['manifest_sha256']==proof['manifest_sha256']
  entries={{e['relative']:e for e in json.loads(m.read_text())['entries']}}
  for rel in v['deleted_sources']:
   source=str(home/rel)
   if source not in wanted:continue
   e=entries[rel];assert proof['hashes'][rel]==e['sha256']
   index[source]=dict(source=source,archive=v['destination']+'/'+rel,sha256=e['sha256'],bytes=e['bytes'],proof=str(receipt))
missing=sorted(wanted-set(index));assert not missing,missing
v=dict(entries=[index[k] for k in sorted(index)],files=len(index),bytes=sum(e['bytes'] for e in index.values()),new_sim_s=0,
 rule='restore exact previously moved policy/chosen RESULT tapes for legacy admitted B entries and missing cooperative snapshot envelopes; keep archive copies')
with p.open('x') as f:json.dump(v,f)
print(json.dumps(v))''',timeout=60))
    p=ops.EVIDENCE/'BANK_RESTORE_PLAN_1900.json'
    if not p.exists():write(p,value)
    print(json.dumps(value))


def register():
    value=read(ops.EVIDENCE/'BANK_RESTORE_PLAN_1900.json')
    record=ops.EVIDENCE/'BANK_RESTORE_REGISTRATION_1900.json'
    line=('\nS-COOPMWM MAIN19:00 bank boot engineering data repair: restore ONLY exact '
        'SHA-certified archived policy/chosen RESULT.json(+gzip) tapes for legacy-admitted '
        'B states and missing cooperative pending envelopes, from original SHA-move '
        'receipts. No simulation rerun, no outcome edits, archive copies retained. '
        'Explicit manifest/target path gate, source archive checksum, local restored '
        'checksum/size before install; existing unequal files cause STOP. Restore '
        'protection binding excludes these new gate inputs from rolling archive. '
        f"Exact {value['files']}files/{value['bytes']}bytes. Zero SUMO/inference.\n")
    if not record.exists():write(record,dict(line=line,plan=value,plan_sha256=sha(ops.EVIDENCE/'BANK_RESTORE_PLAN_1900.json'),new_sim_s=0))
    h=sha(record);registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(line+'amendmentSHA'+h+'\n')
    print(remote('cpuQ',f'''from pathlib import Path
import json
r=Path({CONTROL!r});v=json.loads({json.dumps(dict(read(record),amendment_sha256=h))!r});p=r/'AUTHORITY.json'
if p.exists():assert json.loads(p.read_text())==v
else:
 with p.open('x') as f:json.dump(v,f)
print(json.dumps(dict(registered=True,sha256={h!r},new_sim_s=0)))''',timeout=35))


def restore():
    code=f'''from pathlib import Path
import sys,json,hashlib,subprocess,shlex,os,time
control=Path({CONTROL!r});assert (control/'AUTHORITY.json').exists();done=control/'COMPLETE.json'
if done.exists():print(done.read_text());raise SystemExit
sys.path.insert(0,'/home/ubuntu/coopmwm20261103_rolling1647_v3');import rolling_archive as m
v=json.loads((control/'PLAN.json').read_text());restored=[]
binding=Path('/home/ubuntu/coopmwm20261102_lscollect1/RESTORED_BANK_BINDING_1900.json')
protection=dict(paths=[e['source'] for e in v['entries']],authority_sha256=m.base.digest_file(control/'AUTHORITY.json'))
if not binding.exists():
 with binding.open('x') as f:json.dump(protection,f)
for i,e in enumerate(v['entries']):
 p=Path(e['source']);assert str(p).startswith('/home/ubuntu/coopmwm20261102_lscollect1/')
 assert not m.base.BAD.search(str(p));assert p.name in ('RESULT.json','RESULT.json.gz') or (p.parent.name=='pending' and p.name.endswith('.json.gz'))
 assert e['archive'].startswith('/root/cpuq_archive_20261103/')
 if p.exists():assert p.stat().st_size==e['bytes'] and m.base.digest_file(p)==e['sha256'];restored.append(e['source']);continue
 proof=m.remote("from pathlib import Path\\nimport hashlib,json\\np=Path("+repr(e['archive'])+");print(json.dumps(dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)))")
 assert proof['sha256']==e['sha256'] and proof['bytes']==e['bytes']
 temp=control/('restore_'+str(i));command,env=m.ssh()
 x=subprocess.run(['rsync','-a','--checksum','--no-owner','--no-group','-e',' '.join(shlex.quote(k) for k in command[:-1]),command[-1]+':'+e['archive'],str(temp)],env=env,capture_output=True,text=True,timeout=180)
 assert x.returncode==0,x.stderr
 assert temp.stat().st_size==e['bytes'] and m.base.digest_file(temp)==e['sha256']
 assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True);temp.replace(p);restored.append(e['source'])
 with (control/'RESTORED.jsonl').open('a') as f:f.write(json.dumps(dict(path=str(p),sha256=e['sha256'],unix=time.time()))+'\\n')
out=dict(unix=time.time(),files=len(restored),bytes=v['bytes'],new_sim_s=0,protected_binding=str(binding))
with done.open('x') as f:json.dump(out,f)
print(json.dumps(out))'''
    value=json.loads(remote('cpuQ',f'''import subprocess
p=subprocess.run(['sudo','-n','/home/ubuntu/miniconda3/bin/python','-B','-'],input={code!r},capture_output=True,text=True,timeout=180)
if p.returncode:raise RuntimeError(p.stderr)
print(p.stdout)''',timeout=200))
    p=ops.EVIDENCE/'BANK_RESTORE_COMPLETE_1900.json'
    if not p.exists():write(p,value)
    print(json.dumps(value))


def protect_pending():
    line=('\nS-COOPMWM MAIN19:03 archive protection amendment: all currently present '
        'LS pending inputs are explicitly listed in a public PENDING_PROTECTION_BINDING '
        'consumed by the existing mover metadata-reference guard. No pending input '
        'may be newly archived; restoration bindings also protect recovered inputs. '
        'No simulation/inference, no mover/source restart.\n')
    registry=checked(REGISTRY)
    if line not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(line)
    value=json.loads(remote('cpuQ','''from pathlib import Path
import os,json,time,hashlib
root=Path('/home/ubuntu/coopmwm20261102_lscollect1');p=root/'PENDING_PROTECTION_BINDING_1903.json'
if p.exists():print(json.dumps(dict(adopt=True,files=len(json.loads(p.read_text())['paths']),sha256=hashlib.sha256(p.read_bytes()).hexdigest())));raise SystemExit
bad=('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth','vehicle_params','driver_params','vtype','assignment')
paths=[]
for directory,dirs,names in os.walk(root/'pending',followlinks=False):
 dirs[:]=[n for n in dirs if not any(b in str(Path(directory)/n).lower() for b in bad) and not (Path(directory)/n).is_symlink()]
 for name in names:
  q=Path(directory)/name
  if not any(b in str(q).lower() for b in bad) and not q.is_symlink():paths.append(str(q))
with p.open('x') as f:json.dump(dict(paths=sorted(paths),unix=time.time(),authority='MAIN19:03; pending inputs protected from archive'),f)
print(json.dumps(dict(files=len(paths),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),new_sim_s=0)))''',timeout=35))
    p=ops.EVIDENCE/'PENDING_ARCHIVE_PROTECTION_1903.json'
    if not p.exists():write(p,value)
    print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('plan','register','restore','protect_pending'))
    globals()[p.parse_args().action]()
