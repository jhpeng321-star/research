"""MAIN17:27/17:31 qualification, metadata-only health and scoped storage audit."""
import argparse
import json
import time
from . import ops
from .ls_validity_ops import remote
from .storage import write,read
from .protocol import sha,checked,REGISTRY

CONTROL='/home/ubuntu/coopmwm20261103_amend1727'


def audit():
    code='''from pathlib import Path
import json,time,os,shutil,subprocess
home=Path('/home/ubuntu');run=home/'coopmwm20261102_warm1'
names=['e3final','coopfree20261101_owner1511.tar.gz','c1switch20261031.tar.gz']
out={n:dict(exists=(home/n).exists(),references=[],processes=[]) for n in names}
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  texts=[]
  for field in ('cwd','exe'):
   try:texts.append(os.readlink(proc/field))
   except FileNotFoundError:pass
  texts += [(proc/'cmdline').read_bytes().decode(errors='replace'),(proc/'environ').read_bytes().decode(errors='replace')]
  for fd in (proc/'fd').iterdir():
   try:texts.append(os.readlink(fd))
   except FileNotFoundError:pass
  for n in names:
   if any(str(home/n) in text for text in texts):out[n]['processes'].append(proc.name)
 except (FileNotFoundError,ProcessLookupError):pass
for root in home.glob('coopmwm*'):
 if not root.is_dir() or root.is_symlink() or any(k in root.name for k in ('archive','rolling')):continue
 for p in root.glob('*.json'):
  if any(k in str(p).lower() for k in ('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1')):continue
  if not any(k in p.name.upper() for k in ('BINDING','CONFIG','RUNTIME','MANIFEST')):continue
  text=p.read_text()
  for n in names:
   if str(home/n) in text:out[n]['references'].append(str(p))
for n,v in out.items():
 p=home/n
 v['eligible']=v['exists'] and not v['references'] and not v['processes']
 v['bytes']=p.stat().st_size if p.is_file() else None
for arm in ('B','R'):
 bank=run/'replay_bank'/arm
 for name in ('BANK_QUALITY','BANK_STATUS'):
  p=bank/(name+'.json')
  if p.exists():out[arm+'_'+name]=dict(mtime=p.stat().st_mtime,record=json.loads(p.read_text()))
out['unix']=time.time();out['cpu_free']=shutil.disk_usage(home).free
print(json.dumps(out))'''
    # Root scan is needed to prove absence of references in every process.
    value=json.loads(remote('cpuQ',f'''import subprocess
p=subprocess.run(['sudo','-n','/home/ubuntu/miniconda3/bin/python','-B','-c',{code!r}],capture_output=True,text=True,timeout=50)
if p.returncode:raise RuntimeError(p.stderr)
print(p.stdout)''',timeout=60))
    write(ops.EVIDENCE/f'PRE_V2V_AUDIT_{time.time_ns()}.json',value)
    # Avoid dumping large bank rows; exact originals are retained in the receipt.
    print(json.dumps({k:({a:b for a,b in v.items() if a!='record'} if isinstance(v,dict) else v) for k,v in value.items()}))


def register():
    cpu=read(ops.EVIDENCE/'V2V_LINUX_CPU_RESULT_V8.json');assert cpu['pass_'] and cpu['sim_s']==0 and cpu['skips']==0
    record=ops.EVIDENCE/'V2V_AMENDMENT_1727_REGISTRATION.json'
    line=('\nS-COOPMWM MAIN2026-11-03 17:27: V2V B/R only, separate imitation KL .045 '
        'relative to post-PPO/pre-imitation policy, five requested complete bank passes; '
        'transactional full-pass LR backtracking, restore policy+Adam for rejected passes, '
        'require at least one accepted complete pass for nonempty bank. No partial-pass '
        'policy publishes. AWR targets/temperature/clip unchanged. Bank reader keeps '
        'only legacy-safe entries with two-independent-CRN-seed Student-t two-sided95% '
        'mean-advantage CI lower>0 (df1 t=.975=12.7062047364); extended window is NOT '
        'an independent seed. Cooperation requires corresponding replicated seed-level '
        'synergy CI lower>0; missing replicated solos are unqualified, never imputed '
        'or treated as independent members. Original bank/confirmation/lesson receipts '
        'remain immutable; V2V reader overlays preserve legacy verdicts. Original '
        'non-V2V run unchanged. No new SUMO in qualification. Same cap17471840, '
        'phases, horizons, pacing, u0, endpoints, safety and tests. Source v2vcpu8 '
        'full Linux CPU qualification required before replacement of idle resource watcher.\n')
    if not record.exists():write(record,dict(line=line,cpu_result_sha256=sha(ops.EVIDENCE/'V2V_LINUX_CPU_RESULT_V8.json'),
        cpu=cpu,new_sim_s=0,sources={n:sha(ops.LOCAL/'tools/coopmwm'/n) for n in
        ('bank_imitation.py','bank_statistics.py','test_bank_amendment_cpu.py','v2v_live.py')}))
    h=sha(record);registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(line+'amendmentSHA'+h+'\n')
    print(remote('cpuQ',f'''from pathlib import Path
import json
r=Path({CONTROL!r});assert r.parent.is_dir();r.mkdir(exist_ok=True)
p=r/'AUTHORITY.json';v=json.loads({json.dumps(dict(read(record),amendment_sha256=h))!r})
if p.exists():assert json.loads(p.read_text())==v
else:
 with p.open('x') as f:json.dump(v,f)
print(json.dumps(dict(registered=True,sha256={h!r},new_sim_s=0)))''',timeout=40))


def preflight():
    value=json.loads(remote('cpuQ','''from pathlib import Path
import sys,json,time
source=Path('/home/ubuntu/coopmwm20261103_v2vcpu8')
sys.path[:0]=[str(source/'pytools'),str(source/'runtime_src'),'/home/ubuntu/e3final/runtime']
from tools.coopmwm import replay_bank as rb,bank_statistics as bs
r=Path('/home/ubuntu/coopmwm20261103_v2v2');v=json.loads((r/'V2V_RUNTIME.json').read_text())
old={a:rb.confirmations(r,a) for a in ('B','R')};bs.install(rb);out={}
for arm in ('B','R'):
 pairs,summary=rb.bank_pairs(r,arm,20261101,sources=v['config']['replay_sources'][arm],v2v=True)
 out[arm]=dict(pairs=len(pairs),counts=summary['counts'],entries=summary['entries'],
  old_admitted=sum(bool(c.get('admitted')) for c in old[arm].values()),
  new_admitted=sum(bs.view(c)['admitted'] for c in old[arm].values()),
  missing_replicated_synergy=sum(c.get('admitted',False) and not c.get('cooperation',{}).get('replication_cooperation_gain_veh_s') for c in old[arm].values()))
print(json.dumps(dict(pass_=True,arms=out,new_sim_s=0,new_optimizer_steps=0,unix=time.time())))''',timeout=75))
    write(ops.EVIDENCE/f'V2V_STATISTICAL_PREFLIGHT_{time.time_ns()}.json',value)
    print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('audit','register','preflight'))
    globals()[p.parse_args().action]()
