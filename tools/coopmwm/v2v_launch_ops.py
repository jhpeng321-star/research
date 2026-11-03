"""Register/prepare one durable V2V watcher; resource waits never spend SUMO."""
import argparse
import json
import time
from . import ops
from .ls_validity_ops import remote
from .protocol import sha,checked,REGISTRY
from .storage import read,write

SOURCE='/home/ubuntu/coopmwm20261103_v2vcpu7'
ROOT='/home/ubuntu/coopmwm20261103_v2v2'
LEGACY='/home/ubuntu/coopmwm20261102_warm1'
RECORD='V2V_RUNTIME_REGISTRATION_1640_V2.json'


def register():
    cpu=read(ops.EVIDENCE/'V2V_LINUX_CPU_RESULT_V7.json')
    assert cpu['pass_'] and not cpu['sim_s'] and not cpu['skips']
    gate=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,hashlib
source=Path({SOURCE!r});root=Path({LEGACY!r})
cpu=json.loads((source/'LS_CPU_RESULT.json').read_text());assert cpu['pass_']
start=root/'B_checkpoint_000.pt';assert start.is_file()
print(json.dumps(dict(common_u0_sha256=hashlib.sha256(start.read_bytes()).hexdigest(),
 cpu_result_sha256=hashlib.sha256((source/'LS_CPU_RESULT.json').read_bytes()).hexdigest(),
 legacy_health='validated by resource watcher before dispatch')))''',timeout=40))
    record=ops.EVIDENCE/RECORD
    if not record.exists():
        line=('\nS-COOPMWM MAIN13:45/14:06/15:46 V2V RUNTIME ENGINEERING REGISTRATION: '
            'ALL A/B/R-v2v common original MAPPOu0; source '+SOURCE+'; Linux release qualified; '
            'late-rehearsal fallback uses real one-update smokes of ALL3, retained as training '
            'update1/no duplicate physics, then exact Adam/RNG continuation u2..24 with1800s '
            'minimum start spacing. Durable remote watcher waits for >=10GiB disk and health; '
            'no simulation charges before resource gate. Shared CHARGES.jsonl AND ACCOUNTING.claim '
            'symlinks to authoritative legacy ledger; cap17471840/registered restricted V2V phases. '
            'Bank inputs read-only same-arm views of current B/R confirmed bank; first R fair cap '
            'is initial B admitted target count (metadata-only B_UPDATE_000), then lag1 counts. '
            'Own PPO episodes have no plans, WM or collection wait. Live sampling/scoring/checkpoints '
            'use widened V2V; causal bank/runtime features use sensed speed differences and previous '
            'actual post-safety issued commands, never current imitation targets. Composite safety '
            'vs same-update A-v2v every update; all24 pairs mandatory. B lesson+isolated6/12/18/24; '
            'R isolated12/24; A no extra. Isolated all/solo/background and fresh original-arm-u3 '
            'reference fully precharged; no V2V final paired branches. Fixedu8/u24 episode tests '
            'auto-run in separate children, hash-only until registered analysis; existing MAPPO/native '
            'controls reused at analysis, never rerun. Existing training/GPUs/builders untouched. '
            'Failed/interrupted children alert and retain claims; no automatic physics retry.\n')
        line+=('Two assignment-named inherited implementation paths in the CPU manifest '
            'are not reopened by the startup fingerprint; their original qualification '
            'metadata is retained. V1 preparation failed before physics on the strict '
            'path guard; its root and registration remain immutable. New root v2v2.\n')
        write(record,dict(line=line,authority='MAIN2026-11-03 13:45/14:06/15:46',source_root=SOURCE,
            root=ROOT,cpu=cpu,gate=gate,funding_amendment_sha256=sha(ops.EVIDENCE/'V2V_CAP_REGISTRATION_1406.json'),
            new_sim_s=0,new_inference=0))
    h=sha(record);value=read(record);registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(value['line']+'amendmentSHA'+h+'\n')
    print(json.dumps(dict(registered=True,sha256=h,source_root=SOURCE,new_sim_s=0)))


def prepare():
    v=read(ops.EVIDENCE/RECORD);h=sha(ops.EVIDENCE/RECORD)
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import sys,json,shutil,hashlib,time
source=Path({SOURCE!r});legacy=Path({LEGACY!r});root=Path({ROOT!r})
sys.path[:0]=[str(source/'pytools'),str(source/'runtime_src'),'/home/ubuntu/e3final/runtime']
from tools.coopmwm.storage import read,write
from tools.coopmwm.protocol import sha,digest
from tools.coopmwm.v2v_runner import prepare
from tools.coopmwm.replay_start import binding
from tools.coopmwm import replay_bank as rb
reg=json.loads({json.dumps(v)!r});cpu=read(source/'LS_CPU_RESULT.json')
assert cpu==reg['cpu'] and cpu['pass_'] and not cpu['skips'] and not cpu['sim_s']
assert sha(source/'LS_CPU_RESULT.json')==reg['gate']['cpu_result_sha256']
fund=read(legacy/'V2V_CAP_1406.json');assert fund['amendment_sha256']==reg['funding_amendment_sha256']
assert root.parent.is_dir();root.mkdir(exist_ok=True)
if (root/'V2V_RUNTIME.json').exists():
 print(json.dumps(dict(adopt=True,root=str(root),physics_dispatched=(root/'WATCH_DISPATCH.json').exists())));raise SystemExit
base=binding(legacy,read(legacy/'B_REPLAY_EXPOSURE_V2.json'),'B')
cfg=prepare(base,root=root,legacy_root=legacy,source_root=source,
 common_checkpoint_sha256=reg['gate']['common_u0_sha256'],v2v_dim=24)
cfg.update(budget=fund['budget'],workers=2,health_guards=True,N_exec=0,
 launch_authorized=True,phase_funding_authorized=True,launch_amendment_sha256={h!r},
 funding_amendment_sha256=fund['amendment_sha256'],runtime_receipt_suffix='_V2V_LIVE')
release=dict(qualified=True,linux_cpu_pass=True,zero_init_parity=True,
 default_off_byte_identical=True,stub=False,v2v_dim=24,rehearsal_pass=False,real_update_smokes_pass=False)
for name in ('ACCOUNTING.claim','CHARGES.jsonl'):
 target=legacy/name;assert target.exists();p=root/name
 if p.is_symlink():assert p.resolve()==target.resolve()
 else:assert not p.exists();p.symlink_to(target)
for arm in ('A','B','R'):
 src=legacy/'B_checkpoint_000.pt';p=root/f'{{arm}}_checkpoint_000.pt'
 if not p.exists():shutil.copy2(src,p)
 assert sha(p)==cfg['common_checkpoint_sha256']
 if arm=='A':continue
 bank=root/'replay_bank'/arm;bank.mkdir(parents=True,exist_ok=True)
 original=legacy/'replay_bank'/arm
 for name in ('decisions','snapshots','confirmations','confirmations_safety0426'):
  target=original/name
  if not target.exists():continue
  p=bank/name
  if not p.exists():p.symlink_to(target,target_is_directory=True)
  assert p.is_symlink() and p.resolve()==target.resolve()
 verification=bank/'sumo_verification';verification.mkdir(exist_ok=True)
 for name in ('verifications','branches'):
  target=original/'sumo_verification'/name;assert target.is_dir();p=verification/name
  if not p.exists():p.symlink_to(target,target_is_directory=True)
  assert p.is_symlink() and p.resolve()==target.resolve()
 lesson=original/f'LESSON_LIST_{{arm}}.json'
 if lesson.exists() and not (bank/lesson.name).exists():shutil.copy2(lesson,bank/lesson.name)
 if not (bank/'lesson_check').exists():(bank/'lesson_check').mkdir()
(root/'jobs').mkdir(exist_ok=True);(root/'final_receipts').mkdir(exist_ok=True)
with (root/'AUTHORITY.md').open('x') as f:f.write(reg['line']+'amendmentSHA'+{h!r}+'\\n'+fund['line']+'amendmentSHA'+fund['amendment_sha256']+'\\n')
rule=read(legacy/'SAFETY_RULE_0957.json');write(root/'SAFETY_RULE_0957.json',rule)
with (root/'AUTHORITY.md').open('a') as f:f.write('composite safety authority '+rule['amendment_sha256']+'\\n')
pairs,summary=rb.bank_pairs(root,'B',20261101,sources=cfg['replay_sources']['B'],v2v=True)
assert summary['counts']['used_admitted']>=8 and pairs
write(root/'B_UPDATE_000.json',dict(update=0,optimizer_updates_applied=0,
 exact_imitation=summary,rule='initial same-arm B bank fairness cap only; no PPO update'))
write(root/'V2V_PREFLIGHT_BANK.json',dict(B_summary=summary,B_pairs=len(pairs),new_sim_s=0))
runtime=dict(config=cfg,release=release,amendment_sha256={h!r},cpu_result_sha256=reg['gate']['cpu_result_sha256'])
write(root/'V2V_RUNTIME.json',runtime)
print(json.dumps(dict(prepared=True,root=str(root),B_pairs=len(pairs),B_v2v=summary['v2v'],
 resource_free_bytes=shutil.disk_usage(root).free,new_sim_s=0,physics_dispatched=False)))''',timeout=75))
    p=ops.EVIDENCE/'V2V_RUNTIME_PREPARED_1640_V2.json'
    if not p.exists():write(p,value)
    print(json.dumps(value))


def launch():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,os,subprocess,time,sys
r=Path({ROOT!r});source=Path({SOURCE!r});p=r/'WATCH_DISPATCH.json'
if p.exists():print(p.read_text());raise SystemExit
assert (r/'V2V_RUNTIME.json').is_file()
env=dict(os.environ,PYTHONPATH=str(source/'pytools')+':'+str(source/'runtime_src')+':/home/ubuntu/e3final/runtime',
 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',
 SUMO_HOME='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo',
 PATH='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo/bin:'+os.environ['PATH'])
command=['/home/ubuntu/miniconda3/bin/python','-B','-u','-m','tools.coopmwm.v2v_live',str(r),'watch']
with (r/'WATCH.log').open('x') as log:
 child=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(pid=child.pid,start_ticks=Path('/proc',str(child.pid),'stat').read_text().split()[21],
 boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),argv=command,
 source_root=str(source),root=str(r),resource_gate_bytes=10*1024**3)
with p.open('x') as f:json.dump(d,f)
print(json.dumps(d))''',timeout=45))
    p=ops.EVIDENCE/'V2V_WATCH_DISPATCH_1640_V2.json'
    if not p.exists():write(p,value)
    print(json.dumps(value))


def preflight():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import sys,json,copy
source=Path({SOURCE!r});root=Path({ROOT!r})
sys.path[:0]=[str(source/'pytools'),str(source/'runtime_src'),'/home/ubuntu/e3final/runtime']
from tools.coopmwm.v2v_live import configuration,hooks,resource_gate
from tools.coopmwm.v2v_runner import admit
from tools.coopmwm.v2v_runtime import load_checkpoint
from tools.coopmwm import replay_bank as rb
v=configuration(root);cfg=copy.deepcopy(v['config'])
cfg.update(qualification_stage='one_update_smoke',smoke_authority_sha256=v['amendment_sha256'],
 rehearsal_late_60min=True,stop_after_update=1,A_stop_after_update=1)
admit(cfg,v['release'],hooks(root,cfg))
summaries={{}}
for arm in ('B','R'):
 pairs,summary=rb.bank_pairs(root,arm,20261101,sources=cfg['replay_sources'][arm],v2v=True)
 assert pairs and not summary['v2v']['stub']
 policy,opt,rng,payload=load_checkpoint(root/f'{{arm}}_checkpoint_000.pt',config=cfg)
 mean,value=policy(pairs[0][0]['context']);assert mean.shape[1]==1
 summaries[arm]=dict(pairs=len(pairs),admitted=summary['counts']['used_admitted'],
  v2v=summary['v2v'],forward_members=len(mean),checkpoint_update=payload['update'])
print(json.dumps(dict(pass_=True,bank_preflight=summaries,resource_gate=resource_gate(root,cfg),
 new_sim_s=0,new_optimizer_steps=0,new_inference=0)))''',timeout=75))
    p=ops.EVIDENCE/'V2V_REAL_BANK_PREFLIGHT_1640_V2.json'
    if not p.exists():write(p,value)
    print(json.dumps(value))


def status():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,time,shutil
r=Path({ROOT!r});out=dict(unix=time.time(),root=str(r),disk_free_bytes=shutil.disk_usage('/home/ubuntu').free)
for name in ('WATCH_DISPATCH','V2V_WATCH_STATUS','watch_FAILED','V2V_STARTED','V2V_REAL_SMOKES_PASS','WATCH_COMPLETE'):
 p=r/(name+'.json');out[name]=json.loads(p.read_text()) if p.exists() else None
d=out['WATCH_DISPATCH']
if d:
 p=Path('/proc',str(d['pid']));s=(p/'stat').read_text().split() if p.exists() else None
 out['watcher_alive']=bool(s and s[21]==d['start_ticks'] and s[2]!='Z')
print(json.dumps(out))''',timeout=40))
    write(ops.EVIDENCE/f'V2V_WATCH_STATUS_{time.time_ns()}.json',value);print(json.dumps(value))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('register','prepare','preflight','launch','status'))
    globals()[parser.parse_args().action]()
