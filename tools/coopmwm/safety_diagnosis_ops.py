"""Register, qualify and launch MAIN09:57 read-only diagnosis; never auto-clear a pause."""
import argparse
import json
import time
from . import ops,replay_start_ops as start
from .ls_collect_ops import ROOT
from .ls_validity_ops import remote
from .protocol import checked,sha,REGISTRY
from .storage import read,write

SOURCE='/home/ubuntu/coopmwm20261103_safetydiagcpu1'
CONTROL=ROOT+'/safety_diagnosis_0957'
REG='SAFETY_DIAGNOSIS_RULE_0957.json'


def preregister():
    p=ops.EVIDENCE/REG
    if not p.exists():
        line=('\nS-COOPMWM MAIN2026-11-03 09:57 safety diagnosis/release rule preregistered '
            'BEFORE new diagnosis numbers: same existing TRAIN episodes only, zero SUMO. '
            'B6 vsA6 and last completedR5 vsA5: inherited hard-brake veh-seconds + '
            'trace-based closingTTC<1.5 vehicle-seconds at existing1s cadence. '
            '10000 paired whole-world bootstrap, RNG20261101, two-sided95CI. '
            'RELEASE eligible iff no paired-world collision/teleport increase AND '
            'critical-total mean 95CI lower<=0. Same rule B/R; new R6 must pass '
            'the same composite check before PPO. Other health pauses never released. '
            'Compare A5/A6/A7 hard-braking variability, all matched-world intersections '
            'and full-panel means disclosed; B4/B5/B6-A braking curves + actual imitation '
            'exposure. Situations: rank worlds by extra braking, inherited10s windows '
            'of continuous negative speed variation/positive issued CAV acceleration '
            'and nearest admitted-bank requested-acceleration mean, descriptive only '
            'with no new behavioural threshold or causal imitation claim. Observable '
            'traces only, no final/dev/test/sealed access. Qualification and on-disk '
            'owner required; publishing diagnosis alone never clears the safety pause.\n')
        write(p,dict(unix=time.time(),line=line,authority='MAIN2026-11-03 09:57',
            TTC_threshold=1.5,bootstrap=10000,bootstrap_seed=20261101,alpha=.05,
            collision_rule='no paired-world increase',critical_rule='paired CI lower <= 0',new_sim_s=0))
    h=sha(p);registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(read(p)['line']+'amendmentSHA'+h+'\n')
    print(json.dumps(dict(preregistered=True,amendment_sha256=h,new_sim_s=0)))


def qualify():
    start.BASE='/home/ubuntu/coopmwm20261103_exposurecpu1'
    files={n:checked(ops.LOCAL/f'tools/coopmwm/{n}.py').read_text(encoding='utf-8')
           for n in ('safety_diagnosis','test_safety_diagnosis_cpu')}
    print(remote('cpuQ',start.qualification_payload(files,SOURCE,
        selection=['test_safety_diagnosis_cpu','test_replay_bank_confirm_cpu']),timeout=60))


def cpu_result():
    start.SOURCE=SOURCE;start.cpu_result()


def launch():
    reg=read(ops.EVIDENCE/REG);h=sha(ops.EVIDENCE/REG)
    cpu=read(ops.EVIDENCE/('B_REPLAY_START_CPU_'+SOURCE.rsplit('/',1)[1]+'.json'))
    assert cpu['pass_'] and cpu['sim_s']==cpu['skips']==0
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import os,json,time,sys,subprocess,fcntl,hashlib
r=Path({ROOT!r});s=Path({SOURCE!r});c=Path({CONTROL!r});assert c.parent.is_dir();c.mkdir(exist_ok=True)
lock=(c/'DISPATCH.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (c/'DISPATCH.json').exists():print((c/'DISPATCH.json').read_text());raise SystemExit
cpu=json.loads((s/'LS_CPU_RESULT.json').read_text());assert cpu['pass_'] and cpu['sim_s']==cpu['skips']==0
for n,h in cpu['source_hashes'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h,n
assert (r/'PAUSE_SAFETY.json').exists()
with (c/'RULE.json').open('x') as f:json.dump({dict(reg,amendment_sha256=h)!r},f)
with (r/'AUTHORITY.md').open('a') as f:f.write({reg['line']!r}+'amendmentSHA'+{h!r}+chr(10))
env=dict(os.environ,PYTHONPATH=str(s/'pytools')+':'+str(s/'WM_PACKAGE')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
code='from tools.coopmwm.nullfix_geometry import install;install('+repr(str(s/'FROZEN_GEOMETRY_SOURCE.py'))+','+repr(str(s/'FROZEN_NETWORK.net.xml.gz'))+');from tools.coopmwm.safety_diagnosis import diagnose;diagnose('+repr(str(r))+','+repr(str(c))+')'
with (c/'DIAGNOSIS.log').open('x') as log:
 p=subprocess.Popen(['nice','-n','19','/home/ubuntu/miniconda3/bin/python','-B','-u','-c',code],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(pid=p.pid,start_ticks=Path('/proc',str(p.pid),'stat').read_text().split()[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),source_root=str(s),control_root=str(c),amendment_sha256={h!r},new_sim_s=0)
with (c/'DISPATCH.json').open('x') as f:json.dump(d,f)
print(json.dumps(d))''',timeout=60))
    p=ops.EVIDENCE/'SAFETY_DIAGNOSIS_DISPATCH_0957.json'
    if not p.exists():write(p,value)
    print(json.dumps(value))


def status():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,time
c=Path({CONTROL!r});d=json.loads((c/'DISPATCH.json').read_text());p=Path('/proc',str(d['pid']))
s=(p/'stat').read_text().split() if p.exists() else None
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
assert s is None or s[21]==d['start_ticks']
done=c/'SAFETY_DIAGNOSIS_u6.json'
print(json.dumps(dict(unix=time.time(),owner=d,state=s[2] if s else 'ABSENT',
 diagnosis=json.loads(done.read_text()) if done.exists() else None,
 log=(c/'DIAGNOSIS.log').read_text()[-4000:])))''',timeout=60))
    p=ops.EVIDENCE/f'SAFETY_DIAGNOSIS_STATUS_{time.time_ns()}.json';write(p,value)
    diagnosis=value.pop('diagnosis')
    if diagnosis:
        final=ops.EVIDENCE/'SAFETY_DIAGNOSIS_u6.json'
        if not final.exists():write(final,diagnosis)
        summary={k:diagnosis[k] for k in ('release_eligible','B_u6_vs_A_u6','R_u5_vs_A_u5','A_update_changes','imitation_exposure')}
        summary['A_variability']={u:{k:v for k,v in x.items() if k!='by_world'} for u,x in diagnosis['A_variability'].items()}
        summary['acceleration_braking_proxy']=diagnosis['acceleration_braking_proxy']
        summary['braking_history']=diagnosis['braking_history']
        summary['top_worlds']=[dict(seed=x['seed'],world=x['world'],hard_brake_delta=x['hard_brake_delta']) for x in diagnosis['situations'][:3]]
    else:summary=None
    print(json.dumps(dict(receipt=str(p),**value,summary=summary)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('preregister','qualify','cpu_result','launch','status'))
    globals()[p.parse_args().action]()
