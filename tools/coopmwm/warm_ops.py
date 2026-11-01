"""OWNER22:08 recovery: identity-checked hold and prospective warm-memory release."""
import argparse
import json
from . import ops
from .storage import write, read
from .protocol import REGISTRY, checked, sha


def hold():
    value=json.loads(ops.ssh.remote('cpuQ', '''import pathlib,json,os,signal,time
r=pathlib.Path('/home/ubuntu/coopmwm20261101_accel1');assert r.is_dir()
target=r/'HOLD_WARM2208.json'
if target.exists():
 print(target.read_text())
else:
 d=json.loads((r/'TRAIN_DISPATCH.json').read_text());boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 assert boot==d['boot_id'];p=pathlib.Path('/proc')/str(d['pid'])
 s=(p/'stat').read_text().split();assert s[21]==d['start_ticks']
 assert 'tools.coopmwm.accel_runner' in (p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
 os.kill(d['pid'],signal.SIGSTOP)
 # The shared supervisor cannot advance either optimizer. Already-running A
 # episodes drain; pause only B workers and their verified descendant trees.
 table={}
 for q in pathlib.Path('/proc').iterdir():
  if not q.name.isdigit():continue
  try:
   st=(q/'stat').read_text().split();table[int(q.name)]=dict(parent=int(st[3]),start_ticks=st[21],state=st[2])
  except (OSError,PermissionError):pass
 def descendant(pid,parent):
  seen=set()
  while pid in table and pid not in seen:
   if pid==parent:return True
   seen.add(pid);pid=table[pid]['parent']
  return False
 stopped=[];draining_A=[]
 for c in [json.loads(x) for x in (r/'CHARGES.jsonl').read_text().splitlines()]:
  j=r/'jobs'/c['id'];claim=j/'CLAIM.json'
  if not claim.exists() or (j/'RESULT.json').exists():continue
  owner=json.loads(claim.read_text())['pid']
  if owner not in table:continue
  assert descendant(owner,d['pid']),'job PID no longer belongs to supervisor'
  if '_A_' in c['id']:draining_A.append(dict(id=c['id'],pid=owner));continue
  assert '_B_' in c['id'] or '_R_' in c['id']
  for pid,ident in table.items():
   if not descendant(pid,owner):continue
   q=pathlib.Path('/proc')/str(pid)
   if not (q/'stat').exists():continue
   assert (q/'stat').read_text().split()[21]==ident['start_ticks']
   os.kill(pid,signal.SIGSTOP);stopped.append(dict(pid=pid,**ident))
 charges=[json.loads(x) for x in (r/'CHARGES.jsonl').read_text().splitlines()]
 v=dict(unix=time.time(),owner=d,supervisor_state=(p/'stat').read_text().split()[2],
  held_arms=['B','R'],shared_optimizer_supervisor_held=True,A_episodes_draining=draining_A,
  stopped_owned_processes=stopped,charged_sim_s=811640+sum(c['sim_s'] for c in charges),
  jobs=len(charges),closed=sum((r/'jobs'/c['id']/'CLOSE.json').exists() for c in charges),
  updates=sorted(x.name for x in r.glob('*_UPDATE_*.json')),
  reason='OWNER22:08 observable per-driver warm memory; B/R cold engineering runs superseded; restart u0 only after qualification',new_sumo_sim_s=0)
 with target.open('x') as f:json.dump(v,f,sort_keys=True)
 print(json.dumps(v))
''',timeout=30))
    write(ops.EVIDENCE/'HOLD_WARM2208.json',value)
    print(json.dumps(value))


def register():
    held=read(ops.EVIDENCE/'HOLD_WARM2208.json')
    value=dict(owner='2026-11-01 22:08',hold=held,history_s=60,
        observation_contract='C1 genuine sensed history only; positions/speeds/lanes, sensed identity/CAV flag, entrance counts; no hidden settings',
        model='FULL-v4 22cc32f6',posterior='frozen learned history encoder/shrinkage; per-driver posterior CRN across candidates; unseen population prior',
        release_gates=['CPU observable whitelist/history/empty equivalence/warm path',
            'both GPU batched-single warm parity','cold/warm decision diagnostic on registered exact-labelled E2 states; timing'],
        restart='B/R MAPPO u0, isolated immutable namespace; A retained; no new sealed access',
        budget='all old reservations retained; exact revised quote before restart; cap6000000',
        new_sumo_sim_s=0)
    write(ops.EVIDENCE/'WARM2208_AMENDMENT.json',value);h=sha(ops.EVIDENCE/'WARM2208_AMENDMENT.json')
    line=('\nS-COOPMWM OWNER22:08 WARM MEMORY ENGINEERING HOLD REGISTERED: '
        f'amendmentSHA{h}; owned supervisor{held["owner"]["pid"]} identity-verified/held, B/R workers held, '
        'A in-flight episodes may drain; A optimizer continuation retained. Cold B/R receipts superseded, all charges retained '
        f'(current conservative{held["charged_sim_s"]}sim-s). Before B/R MAPPOu0 restart: observable-only rolling60s C1 sensed history, '
        'FULL-v4 learned per-driver posterior/shrinkage, unseen population prior, CRN shared across candidate plans; '
        'CPU whitelist/warm/empty equivalence, both-GPU batch parity, exact-labelled registered E2 cold/warm diagnostic and query timing required. '
        'SUMO-only reward, medium CEM, safety/ownership/trust/R matching unchanged; exact revised cap6000000 quote before any new physics. '
        'No outcome-dependent tuning, no new sealed access, newSUMO0.\n')
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    write(ops.EVIDENCE/'WARM2208_REGISTRATION.json',dict(line=line,amendment_sha256=h));print(line.strip())


def inspect_frozen():
    value=json.loads(ops.ssh.remote('gpuA', '''import pathlib,json,hashlib
r=pathlib.Path('/root/rbeval20261031/phys_code_22cc32f65a80')
names=['src/wms/control/hidden_state.py','src/wms/control/fitted_hidden.py',
 'src/wms/model/history.py','tools/rbint/recipe.py','tools/rbint/motion_posterior.py',
  'src/wms/eval/rollout_inputs.py','src/wms/model/driver_inference.py',
  'src/wms/model/driver_encoder.py','src/wms/data/history.py',
  'src/wms/physics/motion_fit.py']
out={}
for n in names:
 p=r/n
 if p.is_file():out[n]=dict(source=p.read_text(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
print(json.dumps(out))
''',timeout=30))
    target=ops.EVIDENCE/'WARM2208_FROZEN_API.json'
    if target.exists():
        old=read(target);assert all(value[k]==v for k,v in old.items())
        if value!=old:
            target=ops.EVIDENCE/'WARM2208_FROZEN_API3.json'
            if target.exists():assert read(target)==value
            else:write(target,value)
    else:write(target,value)
    for name,item in value.items():
        target=ops.EVIDENCE/'warm_api'/name.replace('/','__')
        target.parent.mkdir(exist_ok=True)
        if target.exists():assert target.read_text(encoding='utf-8')==item['source']
        else:
            with target.open('x',encoding='utf-8') as f:f.write(item['source'])
    print(json.dumps({k:dict(sha256=v['sha256'],lines=len(v['source'].splitlines())) for k,v in value.items()}))


def recover():
    value=json.loads(ops.ssh.remote('cpuQ', '''import pathlib,json,time
r=pathlib.Path('/home/ubuntu/coopmwm20261101_accel1');assert r.is_dir()
d=json.loads((r/'TRAIN_DISPATCH.json').read_text());p=pathlib.Path('/proc')/str(d['pid'])
boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
s=(p/'stat').read_text().split() if (p/'stat').exists() else None
assert not s or (boot==d['boot_id'] and s[21]==d['start_ticks'] and s[2] in ('T','Z'))
held=json.loads((r/'HOLD_WARM2208.json').read_text());children=[]
for ident in held['stopped_owned_processes']:
 q=pathlib.Path('/proc')/str(ident['pid']);st=(q/'stat').read_text().split() if (q/'stat').exists() else None
 assert not st or (st[21]==ident['start_ticks'] and st[2] in ('T','Z'))
 children.append(dict(pid=ident['pid'],state=st[2] if st else 'exited'))
charges=[json.loads(x) for x in (r/'CHARGES.jsonl').read_text().splitlines()]
jobs=[dict(id=c['id'],claimed=(r/'jobs'/c['id']/'CLAIM.json').exists(),
 complete=(r/'jobs'/c['id']/'RESULT.json').exists(),closed=(r/'jobs'/c['id']/'CLOSE.json').exists()) for c in charges]
print(json.dumps(dict(unix=time.time(),owner=d,state=s[2] if s else 'exited',children=children,jobs=jobs,
 charged_sim_s=811640+sum(c['sim_s'] for c in charges),updates=sorted(x.name for x in r.glob('*_UPDATE_*.json')),new_sumo_sim_s=0)))
''',timeout=25))
    write(ops.EVIDENCE/'WARM2222_RECOVERY.json',value)
    print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('hold','register','inspect-frozen','recover'));a=p.parse_args()
    {'hold':hold,'register':register,'inspect-frozen':inspect_frozen,'recover':recover}[a.mode]()
