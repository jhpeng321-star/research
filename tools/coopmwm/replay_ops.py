"""Outcome-blind Linux qualification and registration for MAIN20:33 B-REPLAY."""
import argparse
import base64
import json
import zlib
from . import ops
from .protocol import checked
from .storage import read,write
from .ls_recover import RELEASE as BASE

SOURCE='/home/ubuntu/coopmwm20261103_replayreleasecpu1'


def release_status():
    """No jobs: verify delegated readiness and that its commit was integrated."""
    import subprocess
    p=ops.LOCAL.parent/'c4-harden-int/HARDEN_INT_READY.json'
    if not p.exists():
        value=dict(ready=False,reason='delegated HARDEN_INT_READY.json absent',
            qualification_dispatched=False,new_sumo_sim_s=0)
    else:
        ready=read(p)
        commit=ready['commit']
        integrated=subprocess.run(['git','merge-base','--is-ancestor',commit,'HEAD'],
            cwd=ops.LOCAL,check=False).returncode==0
        value=dict(ready=True,release=ready,integrated=integrated,
            qualification_dispatched=False,new_sumo_sim_s=0)
    return value


def query_diagnosis():
    from .ls_collect_ops import COLLECTION,ROOT
    print(ops.ssh.remote('cpuQ',f'''import pathlib,json,time
c=pathlib.Path({COLLECTION!r});r=pathlib.Path({ROOT!r});out=[]
for p in (c/'broker_assignments').glob('*.json'):
 a=json.loads(p.read_text());q=json.loads((c/'requests'/p.name).read_text())
 out.append(dict(key=p.stem,slot=a['slot'],world_id=q['world_id'],time_s=q['time_s'],group_size=len(q['group']),has_response=(r/'responses'/p.name).exists()))
print(json.dumps(dict(unix=time.time(),assigned=out)))''',timeout=30))


def qualify():
    release=release_status()
    if not release['ready'] or not release['integrated']:
        print(json.dumps(release));return
    names=('accel_plant','accel_runner','capacity_broker','centre_client','conftest','exact_runtime',
        'fcntl_compat','hotadd','incident_recovery','mppi_runtime','parallel_groups','plan_client',
        'plan_service','plans','query_errors','rehearsal_cpu','replay_bank','sumo_verify',
        'test_capacity_broker_cpu','test_harden1_cpu','test_harden2_main_cpu','test_harden2_resume_cpu',
        'test_harden3_cpu','test_incident_cpu','test_parallel_groups_cpu',
        'test_replay_bank_confirm_cpu','test_replay_bank_lesson_cpu','verification_beta_ops',
        'verification_grades','warm_broker','warm_rpc','warm_runner','warm_spool')
    files={n:checked(ops.LOCAL/'tools/coopmwm'/(n+'.py')).read_text(encoding='utf-8') for n in names}
    payload=base64.b64encode(zlib.compress(json.dumps(files).encode())).decode()
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,os,time,shutil,hashlib,subprocess,zlib,base64
r=pathlib.Path({SOURCE!r});base=pathlib.Path({BASE!r});assert r.parent.is_dir()
files=json.loads(zlib.decompress(base64.b64decode({payload!r})))
if not r.exists():
 shutil.copytree(base,r,ignore=shutil.ignore_patterns('LS_CPU*','LS_OVERLAY*','__pycache__'))
 for n,text in files.items():(r/'pytools/tools/coopmwm'/(n+'.py')).write_text(text)
 hashes={{str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'pytools').rglob('*.py')}}
 with (r/'LS_SOURCE_HASHES.json').open('w') as f:json.dump(hashes,f)
 tests=json.loads((base/'LS_RELEASE_TESTS.json').read_text())
 tests+=['test_harden1_cpu','test_harden2_main_cpu','test_harden2_resume_cpu','test_harden3_cpu','test_mppi_runtime_cpu','test_incident_recovery_cpu','test_hotadd_cpu','test_r_matching_cpu','test_replay_bank_confirm_cpu','test_replay_bank_lesson_cpu']
 with (r/'LS_RELEASE_TESTS.json').open('w') as f:json.dump(sorted(set(tests)),f)
 with (r/'DELEGATED_RELEASE.json').open('x') as f:json.dump({release!r},f)
for n,text in files.items():assert (r/'pytools/tools/coopmwm'/(n+'.py')).read_text()==text,n
assert json.loads((r/'DELEGATED_RELEASE.json').read_text())=={release!r}
done=r/'LS_CPU_RESULT.json';dispatch=r/'LS_CPU_DISPATCH.json'
if done.exists():print(done.read_text())
elif dispatch.exists():print(dispatch.read_text())
else:
 env=dict(os.environ,PYTHONPATH=str(r/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime')
 with (r/'LS_CPU.log').open('x') as log:
  proc=subprocess.Popen(['/home/ubuntu/miniconda3/bin/python','-B','-u','-m','tools.coopmwm.ls_cpu_test',str(r)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 d=dict(pid=proc.pid,start_ticks=pathlib.Path('/proc',str(proc.pid),'stat').read_text().split()[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time())
 with dispatch.open('x') as f:json.dump(d,f)
 print(json.dumps(d))''',timeout=40)
    print(out)


def cpu_result():
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json
r=pathlib.Path({SOURCE!r});p=r/'LS_CPU_RESULT.json'
if p.exists():print(p.read_text())
else:print(json.dumps(dict(pending=True,log=(r/'LS_CPU.log').read_text()[-5000:])))''',timeout=30)
    v=json.loads(out)
    if not v.get('pending'):
        p=ops.EVIDENCE/('REPLAY_CPU_'+SOURCE.rsplit('/',1)[1]+'.json')
        if not p.exists():write(p,v)
        print(json.dumps({k:v[k] for k in ('pass_','tests','skips','sim_s','wall_s','root')}))
        if not v['pass_']:print(ops.ssh.remote('cpuQ',f'from pathlib import Path\nprint(Path({SOURCE!r},"LS_CPU.log").read_text()[-16000:])',timeout=30))
    else:print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['qualify','cpu_result','query_diagnosis','release_status'])
    a=p.parse_args()
    if a.action=='release_status':print(json.dumps(release_status()))
    else:globals()[a.action]()
