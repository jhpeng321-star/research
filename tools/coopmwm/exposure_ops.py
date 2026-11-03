"""MAIN09:36 full-bank exposure and centre-envelope repair at a common idle boundary."""
import argparse
import json
import time
from . import ops,replay_start_ops as start
from .ls_collect_ops import ROOT
from .ls_validity_ops import remote
from .protocol import checked,sha,REGISTRY
from .storage import read,write

SOURCE='/home/ubuntu/coopmwm20261103_exposurecpu1'
CONTROL=ROOT+'/replay_exposure_v2'
NAME='B_REPLAY_EXPOSURE_V2.json'
RECEIPT='B_REPLAY_EXPOSURE_STARTED_V2.json'
REG='B_REPLAY_EXPOSURE_REGISTRATION_V2.json'


def preregister():
    p=ops.EVIDENCE/'B_REPLAY_EXPOSURE_PREREG_0936.json'
    if not p.exists():
        line=('\nS-COOPMWM MAIN09:36 exposure correction preregistration: B/R post-PPO '
            'AWR at least5 FULL passes over every admitted bank state/member action, '
            'existing temperature/weightclip/cooperative50% oversampling x existing '
            'beta unchanged. Stop ONLY for existing KL>.045 versus frozen PRE-UPDATE '
            'policy; report requested/started/completed epochs, pair/member coverage, '
            'action error before/after, and explanation if not decreased. Critic and '
            'own-policy PPO unchanged; plan ratios never computed. Keep 1800s pacing. '
            'TIMING DEVIATION: MAIN09:36 lines687-693 read in this agent at~09:56 '
            'after u6 was already collecting/applying under previous binding; no u6 '
            'outcome values used for this correction. Preserve every u6 episode/'
            'checkpoint/charge and disclose it; earliest common safe activation is '
            'u7 from exact u6 checkpoints, same index B/R, no completed PPO replay '
            'or retrospective claim. Fixed5/KL.045 before reading further outcomes. '
            'Also repair proven centre identity mismatch: service strips ONLY '
            'mode=centre before hashing; accept iff full immutable request hash equals '
            'saved metadata AND response hash equals that exact mode-stripped request, '
            'candidate_scores0. Preserve raw responses/skips; re-admit only these '
            'pre-physics identity declines, never rerun WM inference/graded branches. '
            'CPU-qualify/register final binding before u7. Cap14.5M and all horizons/'
            'arms/final checkpoints/evaluation unchanged.\n')
        write(p,dict(unix=time.time(),line=line,imitation_epochs=5,KL_limit=.045,
            first_new_update=7,pre_update_reference=True,coefficient_rule='inherited AWR x beta',
            extra_sim_s=0,late_instruction_deviation=True))
    h=sha(p);registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(read(p)['line']+'amendmentSHA'+h+'\n')
    print(json.dumps(dict(preregistered=True,amendment_sha256=h,first_new_update=7)))


def qualify():
    start.BASE='/home/ubuntu/coopmwm20261103_replaylesson_cpu3'
    names=('replay_start','replay_bank','warm_runner','exact_grade','bank_imitation',
           'test_bank_imitation_cpu','test_centre_identity_cpu','test_replay_bank_lesson_cpu','test_replay_start_cpu')
    files={n:checked(ops.LOCAL/f'tools/coopmwm/{n}.py').read_text(encoding='utf-8') for n in names}
    print(remote('cpuQ',start.qualification_payload(files,SOURCE),timeout=60))


def cpu_result():
    start.SOURCE=SOURCE;start.cpu_result()


def register():
    path=ops.EVIDENCE/REG
    if not path.exists():
        cpu=read(ops.EVIDENCE/('B_REPLAY_START_CPU_'+SOURCE.rsplit('/',1)[1]+'.json'))
        assert cpu['pass_'] and cpu['sim_s']==cpu['skips']==0
        live=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,time,hashlib,sys
r=Path({ROOT!r});s=Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.warm_health import assert_running
from tools.coopmwm.accel_storage import ledger,charged
v=json.loads((r/'B_REPLAY_LESSON_WINDOW_V1.json').read_text())
for a in ('B','R'):
 assert_running(r,a)
 assert (r/(a+'_UPDATE_006.json')).exists(),'wait for current update6; never replay'
 assert not (r/(a+'_checkpoint_007.pt')).exists()
 assert not list((r/'jobs').glob('replay0735_warm_u007_'+a+'_*'))
entries=ledger(r);remaining=v['budget']['phases']['train']-sum(x['sim_s'] for x in entries if x['phase']=='train')
assert remaining>=1036800,'report exact train phase shortfall'
print(json.dumps(dict(unix=time.time(),previous_binding=v,charged_sim_s=charged(r),train_remaining=remaining,
 checkpoints={{a:hashlib.sha256((r/(a+'_checkpoint_006.pt')).read_bytes()).hexdigest() for a in ('B','R')}},
 next_not_before_s={{a:min(p.stat().st_mtime for p in (r/'jobs').glob('replay0735_warm_u006_'+a+'_*/CLAIM.json'))+v['pace_s'] for a in ('B','R')}})))''',timeout=40))
        previous=live.pop('previous_binding');v=dict(previous)
        v.pop('amendment_sha256',None)
        prereg=ops.EVIDENCE/'B_REPLAY_EXPOSURE_PREREG_0936.json'
        v.update(source_root=SOURCE,control_root=CONTROL,resume_update=6,checkpoints=live['checkpoints'],
            next_not_before_s=live['next_not_before_s'],runtime_receipt_suffix='_REPLAY_EXPOSURE2',
            start_receipt_name=RECEIPT,imitation_epochs=5,recover_centre_identity=True,
            old_endpoints={'B':None,'R':None},cpu=cpu,activation_update=7,
            line=read(prereg)['line'],preregistration_sha256=sha(prereg),live_boundary=live,
            quote=dict(new_reserved_sim_s=0,remaining_PPO_sim_s=1036800))
        write(path,v)
    v=read(path);h=sha(path);registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(v['line']+'activationBindingSHA'+h+'\n')
    binding=dict(v,amendment_sha256=h)
    print(remote('cpuQ',f'''from pathlib import Path
import json
r=Path({ROOT!r});p=r/{NAME!r};v=json.loads({json.dumps(binding)!r})
if p.exists():assert json.loads(p.read_text())==v
else:
 with p.open('x') as f:json.dump(v,f)
with (r/'AUTHORITY.md').open('a') as f:f.write(v['line']+'activationBindingSHA'+{h!r}+chr(10))
print(json.dumps(dict(registered=True,amendment_sha256={h!r},first_update=7,new_reserved_sim_s=0)))''',timeout=40))


def launch():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import os,json,time,signal,subprocess,fcntl,hashlib
r=Path({ROOT!r});s=Path({SOURCE!r});c=Path({CONTROL!r});assert c.parent.is_dir();c.mkdir(exist_ok=True)
lock=(c/'DISPATCH.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (c/'DISPATCH.json').exists():print((c/'DISPATCH.json').read_text());raise SystemExit
v=json.loads((r/{NAME!r}).read_text());assert v['amendment_sha256'] in (r/'AUTHORITY.md').read_text()
cpu=json.loads((s/'LS_CPU_RESULT.json').read_text());assert cpu['pass_'] and cpu['sim_s']==cpu['skips']==0
for n,h in cpu['source_hashes'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h,n
oldc=r/'replay_lesson_window_v1';old=json.loads((oldc/'DISPATCH.json').read_text())
prior=json.loads((r/'B_REPLAY_CONTINUED_LESSONWINDOW.json').read_text())
def verified(d):
 assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
 p=Path('/proc',str(d['pid']));st=(p/'stat').read_text().split() if p.exists() else None
 assert st is not None and st[21]==d['start_ticks'] and st[2]!='Z'
 argv=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
 assert 'tools.coopmwm.replay_start' in argv and str(r) in argv
 return p
verified(old)
arms=[d for d in prior['children'] if d['component']=='arm']
for d in arms:
 a=d['arm'];p=verified(d)
 assert (r/(a+'_UPDATE_006.json')).exists() and not (r/(a+'_checkpoint_007.pt')).exists()
 assert hashlib.sha256((r/(a+'_checkpoint_006.pt')).read_bytes()).hexdigest()==v['checkpoints'][a]
 assert not list((r/'jobs').glob('replay0735_warm_u007_'+a+'_*'))
 for child in (p/'task'/str(d['pid'])/'children').read_text().split():
  assert 'multiprocessing.resource_tracker' in (Path('/proc',child)/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(),'not idle'
rg=next(d for d in prior['children'] if d['component']=='grade' and d['arm']=='R')
rp=verified(rg);assert not (rp/'task'/str(rg['pid'])/'children').read_text().split(),'R grade active; wait'
with (c/'PRIOR_OWNERS.json').open('x') as f:json.dump(dict(supervisor=old,children=prior['children']),f)
for d in [old]+arms+[rg]:
 p=verified(d);stopped=(p/'stat').read_text().split()[2]=='T';os.kill(d['pid'],signal.SIGTERM)
 if stopped:os.kill(d['pid'],signal.SIGCONT)
 for _ in range(100):
  if not p.exists() or (p/'stat').read_text().split()[2]=='Z':break
  time.sleep(.1)
 else:raise ValueError('owned idle process did not exit')
bg=next(d for d in prior['children'] if d['component']=='grade' and d['arm']=='B');verified(bg)
bg=dict(bg,completion_root=str(r/'replay_start_0835_v4'))
with (c/'grade_B_DISPATCH.json').open('x') as f:json.dump(bg,f)
env=dict(os.environ,PYTHONPATH=str(s/'pytools')+':'+str(s/'WM_PACKAGE')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SUMO_HOME='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo')
env['PATH']=env['SUMO_HOME']+'/bin:'+env['PATH'];assert Path(env['SUMO_HOME'],'bin/sumo').is_file()
with (c/'WATCH.log').open('x') as log:
 proc=subprocess.Popen(['/home/ubuntu/miniconda3/bin/python','-B','-u','-m','tools.coopmwm.replay_start',str(r),'watch'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(pid=proc.pid,start_ticks=Path('/proc',str(proc.pid),'stat').read_text().split()[21],boot_id=old['boot_id'],unix=time.time(),source_root=str(s),control_root=str(c),amendment_sha256=v['amendment_sha256'])
with (c/'DISPATCH.json').open('x') as f:json.dump(d,f)
print(json.dumps(d))''',timeout=60))
    p=ops.EVIDENCE/'B_REPLAY_EXPOSURE_DISPATCH_V2.json'
    if not p.exists():write(p,value)
    print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('preregister','qualify','cpu_result','register','launch'))
    globals()[p.parse_args().action]()
