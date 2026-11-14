"""Qualified, append-only policy-synergy sidecar; never hands over training."""
import argparse
import json
import time
from . import ops, replay_start_ops
from .ls_collect_ops import ROOT, COLLECTION
from .ls_validity_ops import remote
from .protocol import REGISTRY, checked, sha
from .storage import read, write

SOURCE='/home/ubuntu/coopmwm20261103_policysynergycpu2'
BASE='/home/ubuntu/coopmwm20261103_compositereleasecpu1'
CONTROL=ROOT+'/policy_synergy_1205_v1'
REG='POLICY_SYNERGY_REGISTRATION_1205.json'
BINDING='POLICY_SYNERGY_BINDING_1205.json'
DISPATCH='POLICY_SYNERGY_DISPATCH_1205.json'
REPORT_PATTERN='POLICY_SYNERGY_u[0-9][0-9][0-9]_[BR].json'
STATUS_PREFIX='POLICY_SYNERGY_STATUS'
COMBINED_PATTERN=''


def qualify():
    names=('policy_synergy','policy_synergy_ops','test_policy_synergy_cpu')
    files={n:checked(ops.LOCAL/f'tools/coopmwm/{n}.py').read_text(encoding='utf-8') for n in names}
    for n,t in files.items():compile(t,n+'.py','exec')
    replay_start_ops.BASE=BASE
    print(remote('cpuQ',replay_start_ops.qualification_payload(files,source=SOURCE,
        selection=['test_policy_synergy_cpu','test_replay_bank_lesson_cpu','test_sumo_verify_cpu']),timeout=50))


def cpu_result():
    replay_start_ops.SOURCE=SOURCE
    replay_start_ops.cpu_result()


def register():
    p=ops.EVIDENCE/REG
    if not p.exists():
        cpu=read(ops.EVIDENCE/('B_REPLAY_START_CPU_'+SOURCE.rsplit('/',1)[1]+'.json'))
        assert cpu['pass_'] and cpu['sim_s']==cpu['skips']==0
        live=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,sys
r=Path({ROOT!r});sys.path.insert(0,{SOURCE!r}+'/pytools')
b=json.loads((r/'B_REPLAY_EXPOSURE_V2.json').read_text())
print(json.dumps(dict(budget=b['budget'],gpu_deadline=b['gpu_deadline'])))''',timeout=40))
        assert live['budget']['total']==14500000
        line=('\nS-COOPMWM MAIN2026-11-03 12:05 POLICY COOPERATION TEST: '
            'independent durable cpuQ sidecar, no training/source/optimizer handover. '
            'After every B/R update8..24 freeze ALL then-admitted cooperative own-arm bank states. '
            'Same state/prefix/window/CRN as lesson, all group members NEW policy versus '
            'each member alone NEW and other group members u3; non-group NEW as lesson. '
            'Common captured t0 action retained as lesson; intervention t0+1 onward. '
            'Delta vs u3, gain=delta(all)-max(delta(solo)); report per-state plan gain alongside. '
            'Reuse only sealed exact checkpoint/state/window all-new lessons; no completed branch rerun. '
            'u3 reference reused only when source weights match own u3, else fresh u3 branch. '
            'Complete n+1 branches (+u3 if needed) precharged atomically before physics in '
            'bank_verification_ext; sum<=14.5M and phase cap; no implicit transfers. '
            'Per-state upper(n+2)*original_window_end<=9600sim-s for n<=6; '
            'dynamic state count, stop/alert before physics on phase shortage, no cap increase. '
            'One branch at a time/nice19/CPU-only; no GPU gates needed; health rules active. '
            'One-use claims; failures/mismatch recorded, never replayed/imputed; no sealed reads.\n')
        write(p,dict(line=line,authority='MAIN2026-11-03 12:05',source_root=SOURCE,
            control_root=CONTROL,collection_root=COLLECTION,cpu=cpu,**live,
            first_update=8,final_update=24,phase='bank_verification_ext',new_reserved_sim_s=0))
    v=read(p);h=sha(p);registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(v['line']+'amendmentSHA'+h+'\n')
    value=dict(v,amendment_sha256=h)
    print(remote('cpuQ',f'''from pathlib import Path
import json
r=Path({ROOT!r});v=json.loads({json.dumps(value)!r});p=r/{BINDING!r}
if p.exists():assert json.loads(p.read_text())==v
else:
 with p.open('x') as f:json.dump(v,f)
a=r/'AUTHORITY.md'
if {h!r} not in a.read_text():
 with a.open('a') as f:f.write(v['line']+'amendmentSHA'+{h!r}+chr(10))
print(json.dumps(dict(registered=True,sha256={h!r},new_reserved_sim_s=0)))''',timeout=40))


def launch():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,sys,subprocess,os,fcntl,hashlib
r=Path({ROOT!r});s=Path({SOURCE!r});c=Path({CONTROL!r});assert c.parent.is_dir();c.mkdir(exist_ok=True)
lock=(c/'DISPATCH.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (c/'DISPATCH.json').exists():print((c/'DISPATCH.json').read_text());raise SystemExit
v=json.loads((r/{BINDING!r}).read_text());assert v['amendment_sha256'] in (r/'AUTHORITY.md').read_text()
cpu=json.loads((s/'LS_CPU_RESULT.json').read_text());assert cpu['pass_'] and cpu['sim_s']==cpu['skips']==0
for n,h in cpu['source_hashes'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h,n
sys.path.insert(0,str(s/'pytools'));from tools.coopmwm.replay_start import owner
env=dict(os.environ,PYTHONPATH=str(s/'pytools')+':'+str(s/'WM_PACKAGE')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SUMO_HOME='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo')
env['PATH']=env['SUMO_HOME']+'/bin:'+env['PATH'];assert Path(env['SUMO_HOME'],'bin/sumo').is_file()
with (c/'WATCH.log').open('x') as log:
 p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-m','tools.coopmwm.policy_synergy','watch','--root',str(r),'--binding',str(r/{BINDING!r})],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(owner(p.pid),source_root=str(s),control_root=str(c),amendment_sha256=v['amendment_sha256'])
with (c/'DISPATCH.json').open('x') as f:json.dump(d,f)
print(json.dumps(d))''',timeout=50))
    p=ops.EVIDENCE/DISPATCH
    if not p.exists():write(p,value)
    print(json.dumps(value))


def status():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,sys,time
c=Path({CONTROL!r});r=Path({ROOT!r});sys.path.insert(0,{SOURCE!r}+'/pytools')
from tools.coopmwm.replay_start import alive
d=json.loads((c/'DISPATCH.json').read_text()) if (c/'DISPATCH.json').exists() else None
reports={{p.name:json.loads(p.read_text()) for p in r.glob({REPORT_PATTERN!r})}}
combined={{p.name:json.loads(p.read_text()) for p in r.glob({COMBINED_PATTERN!r})}} if {bool(COMBINED_PATTERN)!r} else {{}}
entries={{}}
for base in c.glob('u[0-9][0-9][0-9]_[BR]'):
 entries[base.name]={{p.name:json.loads(p.read_text()) for p in base.glob('sv_*.json') if not p.name.endswith(('.QUOTE.json','.ALERT.json'))}}
print(json.dumps(dict(unix=time.time(),owner=d,alive=alive(d) if d else False,
status=json.loads((c/'STATUS.json').read_text()) if (c/'STATUS.json').exists() else None,
 log=(c/'WATCH.log').read_text()[-8000:] if (c/'WATCH.log').exists() else '',reports=reports,combined_reports=combined,completed_entries=entries)))''',timeout=45))
    p=ops.EVIDENCE/f'{STATUS_PREFIX}_{time.time_ns()}.json';write(p,value)
    for n,v in value['reports'].items():
        if not (ops.EVIDENCE/n).exists():write(ops.EVIDENCE/n,v)
    for n,v in value['combined_reports'].items():
        if not (ops.EVIDENCE/n).exists():write(ops.EVIDENCE/n,v)
    summary={}
    for name,report in value['reports'].items():
        summary[name]=dict(status=report['status'],charged_sim_s=sum(e.get('charged_sim_s',0) for e in report['entries']),
            adopted_branches=sum('adopted_lesson_sha256' in b for e in report['entries'] for b in e.get('results',[])),
            new_branches=sum(sum('adopted_lesson_sha256' not in b for b in e.get('results',[]))
                +int(e.get('background_only_result') is not None)
                +int(e.get('fresh_u3_reference') is not None) for e in report['entries']),
            entries=[{k:e.get(k) for k in
            ('verification_id','status','reason','policy_cooperation_gain_veh_s','plan_cooperation_gain_veh_s')}
            for e in report['entries']])
    partial={name:[{k:e.get(k) for k in ('verification_id','status','reason','policy_cooperation_gain_veh_s')}
                   for e in entries.values()] for name,entries in value['completed_entries'].items()}
    print(json.dumps(dict(receipt=str(p),owner=value['owner'],alive=value['alive'],
                         status=value['status'],reports=summary,completed_entries=partial)))
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cmd',choices=('qualify','cpu_result','register','launch','status'))
    globals()[p.parse_args().cmd]()
