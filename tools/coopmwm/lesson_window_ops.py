"""Qualify and register a lesson-only repair; continue idle trainers exactly at u4."""
import argparse
import json
import time
from . import ops, replay_start_ops as start
from .ls_collect_ops import ROOT
from .ls_validity_ops import remote
from .protocol import checked, sha, REGISTRY
from .storage import read, write

SOURCE='/home/ubuntu/coopmwm20261103_replaylesson_cpu3'
CONTROL=ROOT+'/replay_lesson_window_v1'
NAME='B_REPLAY_LESSON_WINDOW_V1.json'
RECEIPT='B_REPLAY_CONTINUED_LESSONWINDOW.json'
REG='B_REPLAY_LESSON_WINDOW_REGISTRATION_V1.json'


def qualify():
    start.BASE='/home/ubuntu/coopmwm20261103_replaylesson_cpu2'
    names=('replay_start','replay_bank','test_replay_bank_lesson_cpu','lesson_window_ops',
           'r_bank_recover','test_r_bank_recover_cpu','test_replay_start_cpu','lesson_window_correct')
    files={n:checked(ops.LOCAL/f'tools/coopmwm/{n}.py').read_text(encoding='utf-8') for n in names}
    for n,text in files.items():compile(text,n+'.py','exec')
    print(remote('cpuQ',start.qualification_payload(files,SOURCE),timeout=60))


def cpu_result():
    start.SOURCE=SOURCE
    start.cpu_result()


def boundary_pause():
    """Freeze only verified idle arm parents while a qualified repair is prepared."""
    p=ops.EVIDENCE/'B_REPLAY_IDLE_REPAIR_BOUNDARY_U5_REGISTRATION.json'
    if not p.exists():
        write(p,dict(unix=time.time(),line='S-COOPMWM engineering idle-boundary freeze: verified B/R trainers only, completed u5/checkpoint5/no u6 jobs/no physics children; preserve supervisor, graders, GPU workers and all receipts; resume exact checkpoint after CPU-qualified lesson/R-admission repair. Prior u4 freeze refused before any signal because B-u5 had started. Zero physics/reservations; no outcome-based selection.'))
    h=sha(p);registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write('\n'+read(p)['line']+' amendmentSHA'+h+'\n')
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import os,json,signal,time
r=Path({ROOT!r});c=r/'replay_start_0835_v4';target=c/'IDLE_REPAIR_FREEZE_U5.json'
if target.exists():print(target.read_text());raise SystemExit
start=json.loads((r/'B_REPLAY_STARTED.json').read_text());arms=[d for d in start['children'] if d['component']=='arm']
for d in arms:
 a=d['arm'];p=Path('/proc',str(d['pid']))
 assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
 assert (p/'stat').read_text().split()[21]==d['start_ticks']
 assert 'tools.coopmwm.replay_start' in (p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
 assert (r/(a+'_UPDATE_005.json')).exists() and not (r/(a+'_checkpoint_006.pt')).exists()
 assert not list((r/'jobs').glob('replay0735_warm_u006_'+a+'_*'))
 for child in (p/'task'/str(d['pid'])/'children').read_text().split():
  assert 'multiprocessing.resource_tracker' in (Path('/proc',child)/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
for d in arms:os.kill(d['pid'],signal.SIGSTOP)
v=dict(unix=time.time(),owners=arms,amendment_sha256={h!r},new_sim_s=0)
with target.open('x') as f:json.dump(v,f)
print(json.dumps(v))''',timeout=40))
    p=ops.EVIDENCE/'B_REPLAY_IDLE_REPAIR_BOUNDARY_U5_FREEZE.json'
    if not p.exists():write(p,value)
    print(json.dumps(value))


def register():
    path=ops.EVIDENCE/REG
    if not path.exists():
        cpu=read(ops.EVIDENCE/('B_REPLAY_START_CPU_'+SOURCE.rsplit('/',1)[1]+'.json'))
        assert cpu['pass_'] and cpu['skips']==cpu['sim_s']==0
        live=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,time,hashlib,sys
r=Path({ROOT!r});s=Path({SOURCE!r});sys.path[:0]=[str(s/'pytools'),str(s/'WM_PACKAGE'),'/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
from tools.coopmwm.warm_health import assert_running
from tools.coopmwm.accel_storage import ledger,charged
v=json.loads((r/'B_REPLAY_START_0835_V4.json').read_text())
from tools.coopmwm.nullfix_geometry import install
install(str(s/'FROZEN_GEOMETRY_SOURCE.py'),str(s/'FROZEN_NETWORK.net.xml.gz'))
from tools.coopmwm.replay_start import binding
from tools.coopmwm.r_bank_recover import prepare
import torch
probe=prepare(r,r/'jobs/replay0735_warm_u004_R_986045',binding(r,v,'R'))
assert probe is not None,'no qualified saved-R state; do not publish any reconstructed data'
snap,record,meta=probe
R_probe=dict(state_id=record['state_id'],time_s=record['time_s'],policy_update=meta['policy_update'],
 history_frames=len(record['history']),members=record['group'],RNG_actions_bitwise=meta['RNG_actions_bitwise'],new_episode_sim_s=0)
for a in ('B','R'):
 assert_running(r,a)
 assert (r/(a+'_UPDATE_005.json')).exists()
 assert not (r/(a+'_checkpoint_006.pt')).exists()
 assert not list((r/'jobs').glob('replay0735_warm_u006_'+a+'_*'))
budget=v['budget'];entries=ledger(r)
phase='bank_verification_ext';remaining=budget['phases'][phase]-sum(x['sim_s'] for x in entries if x['phase']==phase)
assert remaining>=580260,'report phase shortfall before continuation'
remaining_verification=budget['phases']['sumo_verification']-sum(x['sim_s'] for x in entries if x['phase']=='sumo_verification')
assert remaining_verification>=259200,'report R bank phase shortfall before collection'
print(json.dumps(dict(unix=time.time(),previous_binding=v,remaining_lesson_phase_sim_s=remaining,
 R_saved_tape_probe=R_probe,
 remaining_verification_phase_sim_s=remaining_verification,
 charged_sim_s=charged(r),checkpoints={{a:hashlib.sha256((r/(a+'_checkpoint_005.pt')).read_bytes()).hexdigest() for a in ('B','R')}},
 next_not_before_s={{a:min(p.stat().st_mtime for p in (r/'jobs').glob('replay0735_warm_u005_'+a+'_*/CLAIM.json'))+v['pace_s'] for a in ('B','R')}})))''',timeout=40))
        v=live['previous_binding']
        line=('\nS-COOPMWM lesson-window engineering amendment: read-only probe found6/8 '
            'B-u4 lesson branches shorter than their original bank reference; exclude '
            'the invalid aggregate from learning evidence, preserve all original receipts '
            'and charges. Future B/R lesson policy branches retain EXACT original slate '
            'start/end/tail/metric, full-prefix CRN and source-inclusive J; reseal branch '
            'spec and charge its full original end. No horizon reduction, reward, PPO, '
            'bank admission, AWR, policy/search or sealed-evaluation change. MAIN09:06: '
            'R-u4 had no bank because B occupied the shared eight-pending admission cap; '
            'use independent equal bounded per-arm caps8. Restore missed R-u4 snapshots '
            'from saved observable tapes only after bitwise regeneration of every own '
            'action and torch RNG; no episode rerun or extra episode charge. Tag these '
            'same-arm states policyu3 honestly; fresh future PPO captures currentpolicy. '
            'Existing external B LS receipts also supply available-k matching metadata '
            '(counts only, no B actions/library/scoring to R); preserve R imitation cap. '
            'R centre+8random/exact/replication/safety/synergy pipeline shares two branch '
            'workers with B equally. Quote24 recovered states x9 x1200<=259200 in '
            'existing sumo_verification, charges before each list. Report delta_veh_s '
            'mean/bootstrapCI and positive share as primary; recapture median/CI only '
            'when abs(first gain)>abs(first minus replication gain). Correct B-u4 into '
            'a NEW receipt: adopt its2 matching-window branches, only6 malformed ones '
            'may run fresh matched-window branches (4260sim-s upper) in bank_verification_ext; '
            'all original outputs/charges preserved. Continue both idle trainers exactly '
            'from u5 checkpoint/Adam/RNG in new immutable CPU-qualified source and '
            'receipt namespace. Keep replay0735 fresh future job IDs, lag1 and original '
            'minimum1800s pacing anchored to u4 CLAIM timestamps. Retire only verified '
            'idle trainers, supervisor and idle R grader, no physics child; adopt B grader, '
            'GPU broker/workers/rollout and collection. All gates/14.5M cap unchanged. '
            'Future lessons already use bank_verification_ext: upper576000 for remaining '
            '40 arm-updates x12 x1200; existing phase has enough room, no new allocation '
            'or precharge here. Prior u4 freeze refused safely before signals as u5 '
            'had already started; preserve u5 outputs and flag its legacy lesson '
            'ratios as uncorrected diagnostics. Resume u6 automatically, fixed '
            'finalu24 and interimu8.\n')
        write(path,dict(line=line,authority='registered engineering correction under MAIN unattended continuation',
            source_root=SOURCE,control_root=CONTROL,resume_update=5,
            runtime_receipt_suffix='_REPLAY_LESSON1',start_receipt_name=RECEIPT,
            cpu=cpu,checkpoints=live['checkpoints'],next_not_before_s=live['next_not_before_s'],
            R_saved_tape_probe=live['R_saved_tape_probe'],
            old_endpoints={'B':None,'R':None},budget=v['budget'],collection_root=v['collection_root'],
            start_admitted_min=8,gpu_rollout_required=False,gpu_deadline=v['gpu_deadline'],
            pace_s=v['pace_s'],demand_phase_end_s=v['demand_phase_end_s'],
            interim_quote_sim_s=v['interim_quote_sim_s'],quote=dict(new_reserved_sim_s=0,
                remaining_lesson_upper_sim_s=580260,available_phase_sim_s=live['remaining_lesson_phase_sim_s'],
                R_recovered_states=24,R_verification_upper_sim_s=259200,
                R_verification_available_sim_s=live['remaining_verification_phase_sim_s'],new_episode_sim_s=0),
            original_start_amendment_sha256=v['amendment_sha256'],charged_sim_s=live['charged_sim_s']))
    value=read(path);h=sha(path);registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(value['line']+'amendmentSHA'+h+'\n')
    binding=dict(value,amendment_sha256=h)
    print(remote('cpuQ',f'''from pathlib import Path
import json
r=Path({ROOT!r});p=r/{NAME!r};v=json.loads({json.dumps(binding)!r})
assert r.is_dir()
if p.exists():assert json.loads(p.read_text())==v
else:
 with p.open('x') as f:json.dump(v,f)
a=r/'AUTHORITY.md'
if {h!r} not in a.read_text():
 with a.open('a') as f:f.write(v['line']+'amendmentSHA'+{h!r}+chr(10))
print(json.dumps(dict(registered=True,amendment_sha256={h!r},new_reserved_sim_s=0)))''',timeout=40))


def launch():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,sys,os,time,signal,subprocess,fcntl,hashlib
r=Path({ROOT!r});s=Path({SOURCE!r});c=Path({CONTROL!r});assert c.parent.is_dir();c.mkdir(exist_ok=True)
lock=(c/'DISPATCH.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (c/'DISPATCH.json').exists():print((c/'DISPATCH.json').read_text());raise SystemExit
v=json.loads((r/{NAME!r}).read_text());assert v['amendment_sha256'] in (r/'AUTHORITY.md').read_text()
cpu=json.loads((s/'LS_CPU_RESULT.json').read_text());assert cpu['pass_'] and cpu['sim_s']==cpu['skips']==0
for n,h in cpu['source_hashes'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h,n
oldc=r/'replay_start_0835_v4';old=json.loads((oldc/'DISPATCH.json').read_text())
previous=json.loads((r/'B_REPLAY_STARTED.json').read_text())
def verified(d):
 assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
 p=Path('/proc',str(d['pid']));st=(p/'stat').read_text().split() if p.exists() else None
 assert st is not None and st[21]==d['start_ticks'] and st[2]!='Z'
 argv=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
 assert 'tools.coopmwm.replay_start' in argv and str(r) in argv
 return p,argv
verified(old)
arms=[d for d in previous['children'] if d['component']=='arm']
for d in arms:
 a=d['arm'];p,argv=verified(d);assert argv.rstrip().endswith('arm --arm '+a)
 assert (r/(a+'_UPDATE_005.json')).exists() and not (r/(a+'_checkpoint_006.pt')).exists()
 assert hashlib.sha256((r/(a+'_checkpoint_005.pt')).read_bytes()).hexdigest()==v['checkpoints'][a]
 assert not list((r/'jobs').glob('replay0735_warm_u006_'+a+'_*'))
 for child in (p/'task'/str(d['pid'])/'children').read_text().split():
  argv=(Path('/proc',child)/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
  assert 'multiprocessing.resource_tracker' in argv,'not idle: live physics/optimizer child'
for d in previous['children']:
 if d['component']=='grade':verified(d)
rg=next(d for d in previous['children'] if d['component']=='grade' and d['arm']=='R')
rp,argv=verified(rg)
assert not (rp/'task'/str(rg['pid'])/'children').read_text().split(),'R grader has active branches'
with (c/'PRIOR_OWNERS.json').open('x') as f:json.dump(dict(supervisor=old,children=previous['children']),f)
for d in [old]+arms+[rg]:
 p,argv=verified(d);os.kill(d['pid'],signal.SIGTERM)
 for _ in range(100):
  if not p.exists() or (p/'stat').read_text().split()[2]=='Z':break
  time.sleep(.1)
 else:raise ValueError('idle owned process did not exit')
for d in previous['children']:
 if d['component']=='grade' and d['arm']=='B':
  with (c/('grade_'+d['arm']+'_DISPATCH.json')).open('x') as f:json.dump(d,f)
env=dict(os.environ,PYTHONPATH=str(s/'pytools')+':'+str(s/'WM_PACKAGE')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SUMO_HOME='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo')
env['PATH']=env['SUMO_HOME']+'/bin:'+env['PATH']
assert Path(env['SUMO_HOME'],'bin/sumo').is_file()
with (c/'WATCH.log').open('x') as log:
 p=subprocess.Popen(['/home/ubuntu/miniconda3/bin/python','-B','-u','-m','tools.coopmwm.replay_start',str(r),'watch'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(pid=p.pid,start_ticks=Path('/proc',str(p.pid),'stat').read_text().split()[21],boot_id=old['boot_id'],unix=time.time(),source_root=str(s),control_root=str(c),amendment_sha256=v['amendment_sha256'])
with (c/'DISPATCH.json').open('x') as f:json.dump(d,f)
print(json.dumps(d))''',timeout=60))
    path=ops.EVIDENCE/'B_REPLAY_LESSON_WINDOW_DISPATCH_V1.json'
    if not path.exists():write(path,value)
    print(json.dumps(value))


def auxiliary():
    """Start one durable saved-R collector and one u4 lesson correction, once."""
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import os,json,time,subprocess,fcntl
r=Path({ROOT!r});s=Path({SOURCE!r});c=Path({CONTROL!r})
assert (r/{RECEIPT!r}).exists(),'continuation must be live first'
env=dict(os.environ,PYTHONPATH=str(s/'pytools')+':'+str(s/'WM_PACKAGE')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SUMO_HOME='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo')
env['PATH']=env['SUMO_HOME']+'/bin:'+env['PATH']
lock=(c/'AUXILIARY.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
out=[]
for name in ('lesson_window_correct',):
 target=c/(name+'_DISPATCH.json')
 if target.exists():out.append(json.loads(target.read_text()));continue
 code='from tools.coopmwm.nullfix_geometry import install;install('+repr(str(s/'FROZEN_GEOMETRY_SOURCE.py'))+','+repr(str(s/'FROZEN_NETWORK.net.xml.gz'))+');from tools.coopmwm.'+name+' import run;run('+repr(str(r))+','+repr(str(c))+')'
 with (c/(name+'.log')).open('x') as log:
  p=subprocess.Popen(['nice','-n','19','/home/ubuntu/miniconda3/bin/python','-B','-u','-c',code],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 d=dict(pid=p.pid,start_ticks=Path('/proc',str(p.pid),'stat').read_text().split()[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),component=name)
 with target.open('x') as f:json.dump(d,f)
 out.append(d)
print(json.dumps(out))''',timeout=60))
    p=ops.EVIDENCE/'B_REPLAY_LESSON_R_COLLECT_AUXILIARY_V1.json'
    if not p.exists():write(p,value)
    print(json.dumps(value))


def r_collect():
    """Release saved-R snapshot collection independently of the trainer boundary."""
    cpu=read(ops.EVIDENCE/('B_REPLAY_START_CPU_'+SOURCE.rsplit('/',1)[1]+'.json'))
    assert cpu['pass_'] and cpu['sim_s']==cpu['skips']==0
    control=ROOT+'/r_bank_recover_0906_v1'
    target=ops.EVIDENCE/'R_BANK_COLLECT_REGISTRATION_0906.json'
    if not target.exists():
        line=('\nS-COOPMWM MAIN09:06 R bank collection engineering recovery: R-u4 had '
            'no bank because shared B pending slots starved R. Restore up to24 missed '
            'R-u4 snapshots from saved RSU/CAV observable tapes only; regenerate each '
            'policyu3 own action and torch generator exactly before publishing, preserve '
            'same-arm policyversion3 tag, no episode rerun/new episode charge. Same '
            'warm60s history, groups2-6, centre-only inference +8 random candidates, '
            'same exact verification/confirmation/safety/synergy and R imitation fairness '
            'cap. Quote<=259200 sumo_verification for24*(8+policy)*1200, existing '
            'phase/14.5M total only; exact branches precharged by existing R grader. '
            'Current-policy future R PPO capture gets independent equal bounded slots '
            'under the qualified idle-boundary release. No B scoring/library to R.\n')
        write(target,dict(line=line,cpu=cpu,source_root=SOURCE,control_root=control,
            verification_quote_sim_s=259200,new_episode_sim_s=0))
    h=sha(target);registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(read(target)['line']+'amendmentSHA'+h+'\n')
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import os,json,time,sys,subprocess,hashlib,fcntl
r=Path({ROOT!r});s=Path({SOURCE!r});c=Path({control!r});assert c.parent.is_dir();c.mkdir(exist_ok=True)
lock=(c/'DISPATCH.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (c/'DISPATCH.json').exists():print((c/'DISPATCH.json').read_text());raise SystemExit
cpu=json.loads((s/'LS_CPU_RESULT.json').read_text());assert cpu['pass_'] and cpu['skips']==cpu['sim_s']==0
for n,h in cpu['source_hashes'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h,n
sys.path[:0]=[str(s/'pytools'),str(s/'WM_PACKAGE'),'/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
from tools.coopmwm.accel_storage import ledger
from tools.coopmwm.nullfix_geometry import install
install(str(s/'FROZEN_GEOMETRY_SOURCE.py'),str(s/'FROZEN_NETWORK.net.xml.gz'))
from tools.coopmwm.replay_start import binding
from tools.coopmwm.r_bank_recover import prepare
import torch
v=json.loads((r/'B_REPLAY_START_0835_V4.json').read_text());v.update(source_root=str(s),amendment_sha256={h!r})
phase='sumo_verification';assert v['budget']['phases'][phase]-sum(x['sim_s'] for x in ledger(r) if x['phase']==phase)>=259200
probe=prepare(r,r/'jobs/replay0735_warm_u004_R_986045',binding(r,v,'R'));assert probe is not None
with (c/'SAVED_R_PREFLIGHT.json').open('x') as f:json.dump(probe[2],f)
with (c/'R_RECOVER_BINDING.json').open('x') as f:json.dump(v,f)
with (r/'AUTHORITY.md').open('a') as f:f.write({read(target)['line']!r}+'amendmentSHA'+{h!r}+chr(10))
env=dict(os.environ,PYTHONPATH=str(s/'pytools')+':'+str(s/'WM_PACKAGE')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
code='from tools.coopmwm.nullfix_geometry import install;install('+repr(str(s/'FROZEN_GEOMETRY_SOURCE.py'))+','+repr(str(s/'FROZEN_NETWORK.net.xml.gz'))+');from tools.coopmwm.r_bank_recover import run;run('+repr(str(r))+','+repr(str(c))+')'
with (c/'RECOVER.log').open('x') as log:
 p=subprocess.Popen(['nice','-n','19','/home/ubuntu/miniconda3/bin/python','-B','-u','-c',code],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(pid=p.pid,start_ticks=Path('/proc',str(p.pid),'stat').read_text().split()[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),source_root=str(s),control_root=str(c),amendment_sha256={h!r},preflight=probe[2])
with (c/'DISPATCH.json').open('x') as f:json.dump(d,f)
print(json.dumps(d))''',timeout=60))
    p=ops.EVIDENCE/'R_BANK_COLLECT_DISPATCH_0906.json'
    if not p.exists():write(p,value)
    print(json.dumps(value))


def r_status():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,time
r=Path({ROOT!r});c=r/'r_bank_recover_0906_v1';d=json.loads((c/'DISPATCH.json').read_text())
p=Path('/proc',str(d['pid']));st=(p/'stat').read_text().split() if p.exists() else None
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
assert st is None or st[21]==d['start_ticks']
bank=r/'replay_bank/R';keys={{p.stem for p in (bank/'snapshots').glob('*.json')}}
print(json.dumps(dict(unix=time.time(),owner=d,state=st[2] if st else 'ABSENT',
 status=json.loads((c/'STATUS.json').read_text()) if (c/'STATUS.json').exists() else None,
 complete=(c/'COMPLETE.json').exists(),log=(c/'RECOVER.log').read_text()[-3000:],
 snapshots=len(keys),centres_answered=sum((r/'centre_responses'/(k+'.json')).exists() for k in keys),
 decisions=len(list((bank/'decisions').glob('*.json'))),
 quality=json.loads((bank/'BANK_QUALITY.json').read_text()) if (bank/'BANK_QUALITY.json').exists() else None)))''',timeout=40))
    p=ops.EVIDENCE/f'R_BANK_COLLECT_STATUS_{time.time_ns()}.json';write(p,value)
    print(json.dumps(dict(receipt=str(p),**value)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('qualify','cpu_result','register','launch','auxiliary','boundary_pause','r_collect','r_status'))
    globals()[p.parse_args().action]()
