"""One-use reporting-only u4 recovery; leave all training/service owners running."""
import argparse
import json
import time
from . import ops,replay_start_ops as start
from .ls_collect_ops import ROOT
from .ls_validity_ops import remote
from .protocol import checked,sha,REGISTRY
from .storage import read,write

SOURCE='/home/ubuntu/coopmwm20261103_lessonfixcpu2'
CONTROL=ROOT+'/lesson_correction_v3'
REG='B_LESSON_CORRECTION_REGISTRATION_V3.json'


def qualify():
    start.BASE='/home/ubuntu/coopmwm20261103_replaylesson_cpu3'
    files={n:checked(ops.LOCAL/f'tools/coopmwm/{n}.py').read_text(encoding='utf-8')
           for n in ('replay_bank','lesson_window_correct','test_replay_bank_lesson_cpu')}
    print(remote('cpuQ',start.qualification_payload(files,SOURCE,
        selection=['test_replay_bank_lesson_cpu']),timeout=60))


def cpu_result():
    start.SOURCE=SOURCE;start.cpu_result()


def register():
    p=ops.EVIDENCE/REG
    if not p.exists():
        cpu=read(ops.EVIDENCE/('B_REPLAY_START_CPU_'+SOURCE.rsplit('/',1)[1]+'.json'))
        assert cpu['pass_'] and cpu['sim_s']==cpu['skips']==0
        line=('\nS-COOPMWM u4 lesson correction V3 engineering recovery: V1 stopped '
            'before any new branch on missing copy import; V2 adopted2 valid branches '
            'but all6 missing branches were blocked before physics because the old '
            'lesson_u004 charge namespace was reused. Retain ALL originals/V1/V2 '
            'outputs/logs/charges. Use NEW lesson_u004matched_v3 charge/work namespace '
            'and adoption/changed-window charge CPU regressions. Add copy import and CPU '
            'regression proving adopted branch runs zero physics/charges and preserves '
            'original bytes. Independent one-use correction owner/new binding/root; '
            'no training/grader/GPU restart. B fixedu4 policy only; adopt2 existing '
            'same-window branches, run only6 originally mismatched windows<=4260sim-s '
            'in existing bank_verification_ext. Verify no matched-window charge exists '
            'before launch; never repeat a completed corrected branch. New output '
            'LESSON_CHECK_u004_B_MATCHED_WINDOW_V3.json, original and failedV1/V2 retained. '
            'Primary absolute paired delta veh-s/mean/bootstrapCI/positive share; '
            'replication-spread-qualified recapture median/CI secondary. No reward/'
            'PPO/AWR/search/horizon/bank-admission/evaluation change. This is the '
            'already-authorized offline u4 diagnostic only; current SAFETY training '
            'pause remains intact, no optimizer or live grader is restarted. '
            'Cap14.5M unchanged.\n')
        write(p,dict(line=line,cpu=cpu,source_root=SOURCE,control_root=CONTROL,
            base_cpu_sha256=sha(ops.EVIDENCE/'B_REPLAY_START_CPU_coopmwm20261103_replaylesson_cpu3.json'),
            correction_sim_s_upper=4260))
    v=read(p);h=sha(p);registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(v['line']+'amendmentSHA'+h+'\n')
    print(json.dumps(dict(registered=True,amendment_sha256=h,tests=v['cpu']['tests'],extra_upper_sim_s=4260)))


def launch():
    reg=read(ops.EVIDENCE/REG);h=sha(ops.EVIDENCE/REG)
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,sys,time,os,hashlib,subprocess,fcntl
r=Path({ROOT!r});s=Path({SOURCE!r});c=Path({CONTROL!r});assert c.parent.is_dir();c.mkdir(exist_ok=True)
lock=(c/'DISPATCH.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (c/'DISPATCH.json').exists():print((c/'DISPATCH.json').read_text());raise SystemExit
cpu=json.loads((s/'LS_CPU_RESULT.json').read_text());assert cpu['pass_'] and cpu['sim_s']==cpu['skips']==0
for n,h in cpu['source_hashes'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h,n
entries=[json.loads(x) for x in (r/'CHARGES.jsonl').read_text().splitlines()]
assert not [x for x in entries if x['id'].startswith('lesson_u004matched_v3_')],'adopt existing corrected branches; never replay'
v=json.loads((r/'B_REPLAY_LESSON_WINDOW_V1.json').read_text())
assert v['budget']['phases']['bank_verification_ext']-sum(x['sim_s'] for x in entries if x['phase']=='bank_verification_ext')>=4260
v.update(source_root=str(s),control_root=str(c),amendment_sha256={h!r},correction_receipt_suffix='_MATCHED_WINDOW_V3',correction_attempt_kind='lesson_u004matched_v3')
with (c/'CORRECTION_BINDING.json').open('x') as f:json.dump(v,f)
with (r/'AUTHORITY.md').open('a') as f:f.write({reg['line']!r}+'amendmentSHA'+{h!r}+chr(10))
env=dict(os.environ,PYTHONPATH=str(s/'pytools')+':'+str(s/'WM_PACKAGE')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SUMO_HOME='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo')
env['PATH']=env['SUMO_HOME']+'/bin:'+env['PATH'];assert Path(env['SUMO_HOME'],'bin/sumo').is_file()
code='from tools.coopmwm.nullfix_geometry import install;install('+repr(str(s/'FROZEN_GEOMETRY_SOURCE.py'))+','+repr(str(s/'FROZEN_NETWORK.net.xml.gz'))+');from tools.coopmwm.lesson_window_correct import run;run('+repr(str(r))+','+repr(str(c))+')'
with (c/'CORRECTION.log').open('x') as log:
 proc=subprocess.Popen(['nice','-n','19','/home/ubuntu/miniconda3/bin/python','-B','-u','-c',code],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(pid=proc.pid,start_ticks=Path('/proc',str(proc.pid),'stat').read_text().split()[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),source_root=str(s),control_root=str(c),amendment_sha256={h!r})
with (c/'DISPATCH.json').open('x') as f:json.dump(d,f)
print(json.dumps(d))''',timeout=60))
    p=ops.EVIDENCE/'B_LESSON_CORRECTION_DISPATCH_V3.json'
    if not p.exists():write(p,value)
    print(json.dumps(value))


def status():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,time
c=Path({CONTROL!r});d=json.loads((c/'DISPATCH.json').read_text());p=Path('/proc',str(d['pid']))
s=(p/'stat').read_text().split() if p.exists() else None
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
assert s is None or s[21]==d['start_ticks']
done=c/'CORRECTION_COMPLETE.json'
result=Path({ROOT!r})/'LESSON_CHECK_u004_B_MATCHED_WINDOW_V3.json'
entries=json.loads(result.read_text()).get('entries',[]) if result.exists() else []
print(json.dumps(dict(unix=time.time(),owner=d,state=s[2] if s else 'ABSENT',
 complete=json.loads(done.read_text()) if done.exists() else None,
 entry_statuses=[dict(id=e['verification_id'],status=e.get('status'),reason=e.get('reason') or e.get('new_policy_branch',{{}}).get('reason')) for e in entries],
 log=(c/'CORRECTION.log').read_text()[-3000:])))''',timeout=40))
    p=ops.EVIDENCE/f'B_LESSON_CORRECTION_V3_STATUS_{time.time_ns()}.json';write(p,value)
    print(json.dumps(dict(receipt=str(p),**value)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('qualify','cpu_result','register','launch','status'))
    globals()[p.parse_args().action]()
