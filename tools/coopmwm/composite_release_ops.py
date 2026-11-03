"""Conditional MAIN09:57 release plus MAIN09:36 exposure, exact checkpoint5 resume."""
import argparse
import json
import time
from . import ops,replay_start_ops as start
from .ls_collect_ops import ROOT
from .ls_validity_ops import remote
from .protocol import checked,sha,REGISTRY
from .storage import read,write

SOURCE='/home/ubuntu/coopmwm20261103_compositereleasecpu1'
CONTROL=ROOT+'/replay_composite_0957'
NAME='B_REPLAY_EXPOSURE_V2.json'
RECEIPT='B_REPLAY_COMPOSITE_STARTED_0957.json'
REG='COMPOSITE_SAFETY_RELEASE_REGISTRATION_0957.json'


def preregister():
    p=ops.EVIDENCE/'COMPOSITE_RELEASE_IMPLEMENTATION_0957.json'
    if not p.exists():
        line=('\nS-COOPMWM MAIN09:57 conditional composite release implementation: '
            'use preregistered1870328f rule/diagnosis, exact SHA-bound old safety pause '
            'only; keep original and every other/new pause active. MAIN09:36 full-bank '
            '5post-PPO AWR passes, existing pre-updateKL.045 stop and weight/beta/'
            'temperature/cooperative exposure rules. ActualB/R checkpoint5, u6 not '
            'applied; resume u6 after eligible release. Earlier draft u7 assumption '
            'superseded by durable pause evidence. Reuse24 completed B-u6 own-policy '
            'RESULTs exactly, no episode replay/recharge; R-u6 fresh. Every future '
            'B/R update runs the same paired composite guard before PPO, including '
            'R6; collisions/teleports remain hard constraints. Live collision guard '
            'retained, new safety evidence gets a NEW pause receipt and cannot be '
            'hidden by old release. R centre envelope repair accepts only exact '
            'mode-stripped hash/candidate_scores0, rescues pre-physics identity skips '
            'without new WM inference or deleting originals. R fairness cap uses '
            'B actual admitted/imitation entries (legacy75 total lists are not8 '
            'targets); apply cap AFTER confirmed-positive filtering, never pad '
            'from B. This bookkeeping correction is before any R exact grade/'
            'imitation outcome. Preserve all old counters/receipts and expose new '
            'available_admitted/used_admitted counts. 1800s pacing, fixed endpoints24 '
            'and sealed plan unchanged. New fresh PPO quote1065600 within train, '
            'no new allocation; total14.5M unchanged. No dispatch until CPU suite '
            'and registered diagnosis eligibility both pass.\n')
        write(p,dict(unix=time.time(),line=line,first_update=6,resume_update=5,
            imitation_epochs=5,new_allocation_sim_s=0,remaining_fresh_PPO_sim_s=1065600))
    h=sha(p);registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(read(p)['line']+'amendmentSHA'+h+'\n')
    print(json.dumps(dict(preregistered=True,amendment_sha256=h,first_update=6)))


def qualify():
    start.BASE='/home/ubuntu/coopmwm20261103_exposurecpu1'
    names=('warm_health','composite_health','safety_diagnosis','test_composite_health_cpu',
        'test_safety_diagnosis_cpu','replay_bank','warm_runner','test_replay_bank_confirm_cpu',
        'test_replay_bank_lesson_cpu')
    files={n:checked(ops.LOCAL/f'tools/coopmwm/{n}.py').read_text(encoding='utf-8') for n in names}
    print(remote('cpuQ',start.qualification_payload(files,SOURCE),timeout=60))


def cpu_result():
    start.SOURCE=SOURCE;start.cpu_result()


def register():
    path=ops.EVIDENCE/REG
    if not path.exists():
        cpu=read(ops.EVIDENCE/('B_REPLAY_START_CPU_'+SOURCE.rsplit('/',1)[1]+'.json'))
        assert cpu['pass_'] and cpu['sim_s']==cpu['skips']==0
        d=read(ops.EVIDENCE/'SAFETY_DIAGNOSIS_u6.json')
        from .safety_diagnosis import comparison
        assert d['release_eligible'] and not any(d['missing'].values())
        assert all(comparison(d['paired_world_rows'][a])['release'] for a in ('B','R'))
        live=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,hashlib,time,sys
r=Path({ROOT!r});sys.path.insert(0,{SOURCE!r}+'/pytools')
from tools.coopmwm.accel_storage import ledger,charged
v=json.loads((r/'B_REPLAY_LESSON_WINDOW_V1.json').read_text())
for a in ('B','R'):
 assert (r/(a+'_UPDATE_005.json')).exists() and not (r/(a+'_checkpoint_006.pt')).exists()
 assert not (r/(a+'_UPDATE_006.json')).exists()
assert sum((p/'RESULT.json').exists() for p in (r/'jobs').glob('replay0735_warm_u006_B_*'))==24
assert not list((r/'jobs').glob('replay0735_warm_u006_R_*/CLAIM.json'))
entries=ledger(r);remaining=v['budget']['phases']['train']-sum(x['sim_s'] for x in entries if x['phase']=='train')
assert remaining>=1065600
print(json.dumps(dict(unix=time.time(),previous_binding=v,charged_sim_s=charged(r),remaining_train=remaining,
 pause_sha256=hashlib.sha256((r/'PAUSE_SAFETY.json').read_bytes()).hexdigest(),
 checkpoints={{a:hashlib.sha256((r/(a+'_checkpoint_005.pt')).read_bytes()).hexdigest() for a in ('B','R')}},
 next_not_before_s={{a:min(p.stat().st_mtime for p in (r/'jobs').glob('replay0735_warm_u005_'+a+'_*/CLAIM.json'))+v['pace_s'] for a in ('B','R')}})))''',timeout=40))
        assert live['pause_sha256']==d['original_pause_sha256']
        v=dict(live.pop('previous_binding'));v.pop('amendment_sha256',None)
        implementation=ops.EVIDENCE/'COMPOSITE_RELEASE_IMPLEMENTATION_0957.json'
        v.update(source_root=SOURCE,control_root=CONTROL,resume_update=5,checkpoints=live['checkpoints'],
            next_not_before_s=live['next_not_before_s'],runtime_receipt_suffix='_REPLAY_COMPOSITE0957',
            start_receipt_name=RECEIPT,imitation_epochs=5,recover_centre_identity=True,cpu=cpu,
            old_endpoints={'B':None,'R':None},line=read(implementation)['line'],
            implementation_sha256=sha(implementation),diagnosis_sha256=sha(ops.EVIDENCE/'SAFETY_DIAGNOSIS_u6.json'),
            original_pause_sha256=live['pause_sha256'],live_boundary=live,
            quote=dict(new_reserved_sim_s=0,remaining_fresh_PPO_sim_s=1065600,B_u6_reused_episodes=24))
        write(path,v)
    v=read(path);h=sha(path);registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(v['line']+'activationBindingSHA'+h+'\n')
    rule=read(ops.EVIDENCE/'SAFETY_DIAGNOSIS_RULE_0957.json')
    rule=dict(rule,amendment_sha256=sha(ops.EVIDENCE/'SAFETY_DIAGNOSIS_RULE_0957.json'))
    diagnosis=read(ops.EVIDENCE/'SAFETY_DIAGNOSIS_u6.json')
    binding=dict(v,amendment_sha256=h)
    print(remote('cpuQ',f'''from pathlib import Path
import json,hashlib,sys
r=Path({ROOT!r});v=json.loads({json.dumps(binding)!r});rule=json.loads({json.dumps(rule)!r});d=json.loads({json.dumps(diagnosis)!r})
def put(p,value):
 if p.exists():assert json.loads(p.read_text())==value
 else:
  with p.open('x') as f:json.dump(value,f)
put(r/{NAME!r},v);put(r/'SAFETY_RULE_0957.json',rule);put(r/'SAFETY_DIAGNOSIS_u6.json',d)
with (r/'AUTHORITY.md').open('a') as f:f.write(v['line']+'activationBindingSHA'+{h!r}+chr(10))
release=dict(authority='MAIN2026-11-03 09:57',released=True,original_pause_sha256=v['original_pause_sha256'],
 diagnosis_sha256=hashlib.sha256((r/'SAFETY_DIAGNOSIS_u6.json').read_bytes()).hexdigest(),
 rule_sha256=hashlib.sha256((r/'SAFETY_RULE_0957.json').read_bytes()).hexdigest(),amendment_sha256={h!r})
put(r/'SAFETY_RELEASE_0957.json',release)
sys.path.insert(0,{SOURCE!r}+'/pytools')
from tools.coopmwm.warm_health import assert_running
assert_running(r,'B');assert_running(r,'R')
print(json.dumps(dict(registered=True,amendment_sha256={h!r},released_original_only=True,release=release)))''',timeout=60))


def launch():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,os,time,sys,subprocess,fcntl,hashlib
r=Path({ROOT!r});s=Path({SOURCE!r});c=Path({CONTROL!r});assert c.parent.is_dir();c.mkdir(exist_ok=True)
lock=(c/'DISPATCH.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (c/'DISPATCH.json').exists():print((c/'DISPATCH.json').read_text());raise SystemExit
v=json.loads((r/{NAME!r}).read_text());assert v['amendment_sha256'] in (r/'AUTHORITY.md').read_text()
cpu=json.loads((s/'LS_CPU_RESULT.json').read_text());assert cpu['pass_'] and cpu['sim_s']==cpu['skips']==0
for n,h in cpu['source_hashes'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h,n
sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.warm_health import assert_running
assert_running(r,'B');assert_running(r,'R')
old=json.loads((r/'replay_lesson_window_v1/DISPATCH.json').read_text())
prior=json.loads((r/'B_REPLAY_CONTINUED_LESSONWINDOW.json').read_text())
for d in [old]+prior['children']:
 assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
 p=Path('/proc',str(d['pid']));st=(p/'stat').read_text().split() if p.exists() else None
 assert st is None or (st[21]==d['start_ticks'] and st[2]=='Z'),'adopt live/reused owner, never duplicate'
for a in ('B','R'):
 assert hashlib.sha256((r/(a+'_checkpoint_005.pt')).read_bytes()).hexdigest()==v['checkpoints'][a]
 assert not (r/(a+'_checkpoint_006.pt')).exists()
with (c/'PRIOR_HEALTH_STOPPED_OWNERS.json').open('x') as f:json.dump(dict(supervisor=old,children=prior['children']),f)
env=dict(os.environ,PYTHONPATH=str(s/'pytools')+':'+str(s/'WM_PACKAGE')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SUMO_HOME='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo')
env['PATH']=env['SUMO_HOME']+'/bin:'+env['PATH'];assert Path(env['SUMO_HOME'],'bin/sumo').is_file()
with (c/'WATCH.log').open('x') as log:
 p=subprocess.Popen(['/home/ubuntu/miniconda3/bin/python','-B','-u','-m','tools.coopmwm.replay_start',str(r),'watch'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(pid=p.pid,start_ticks=Path('/proc',str(p.pid),'stat').read_text().split()[21],boot_id=old['boot_id'],unix=time.time(),source_root=str(s),control_root=str(c),amendment_sha256=v['amendment_sha256'])
with (c/'DISPATCH.json').open('x') as f:json.dump(d,f)
print(json.dumps(d))''',timeout=60))
    p=ops.EVIDENCE/'COMPOSITE_SAFETY_RESUME_DISPATCH_0957.json'
    if not p.exists():write(p,value)
    print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('preregister','qualify','cpu_result','register','launch'))
    globals()[p.parse_args().action]()
