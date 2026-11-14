"""One-time old-wave admission drain for the already-registered tail handover."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
from . import ops
from .storage import read,write
from .protocol import checked,REGISTRY,sha
from .warm_deploy import ROOT

SOURCE='/home/ubuntu/coopmwm20261102_taildraincpu2'
CONFIG='TAIL_DRAIN_BROKER_CONFIG_V2.json'


def broker_state(owner,verify_command=True):
    """Verify the exact serve argv and bound PYTHONPATH, plus boot/start identity.

    This entry point has no root argv. Only PYTHONPATH is inspected from its
    environment, and no environment values are logged.
    """
    assert owner['root']==ROOT and owner['source']=='/home/ubuntu/coopmwm20261102_centrefaircpu1'
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==owner['boot_id']
    proc=Path('/proc')/str(owner['pid'])
    if not (proc/'stat').exists():return 'exited'
    fields=(proc/'stat').read_text().split();assert fields[21]==owner['start_ticks'],'broker PID reused'
    if fields[2]=='Z':return 'exited'
    if not verify_command:return fields[2]
    raw=(proc/'cmdline').read_bytes()
    if not raw:return 'exiting'
    argv=raw.rstrip(bytes([0])).decode().split(chr(0))
    assert argv==['/home/ubuntu/miniconda3/bin/python','-B','-u','-m','tools.coopmwm.centre_fair_ops','serve'],'broker argv changed'
    paths=[v.split(b'=',1)[1].decode() for v in (proc/'environ').read_bytes().split(bytes([0])) if v.startswith(b'PYTHONPATH=')]
    assert len(paths)==1 and paths[0].split(':')[0]==owner['source']+'/pytools','broker source environment changed'
    return fields[2]


def retire_broker(owner):
    status=broker_state(owner)
    if status=='exited':return
    if status!='exiting':
        os.kill(owner['pid'],signal.SIGTERM)
        if status=='T':os.kill(owner['pid'],signal.SIGCONT)
    deadline=time.monotonic()+15
    while broker_state(owner,verify_command=False)!='exited':
        if time.monotonic()>deadline:raise ValueError('owned broker did not exit; no replacement')
        time.sleep(.1)


def qualify():
    prior=read(ops.EVIDENCE/'CPU_coopmwm20261102_centrefaircpu1.json')
    files=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,hashlib
r=pathlib.Path({prior['root']!r});files={{}}
for name,h in json.loads({json.dumps(prior['source_hashes'])!r}).items():
 p=r/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==h;files[name.removeprefix('pytools/')]=p.read_text()
print(json.dumps(files))''',timeout=40))
    for name in ('capacity_broker','tail_drain_ops','test_tail_drain_cpu'):
        files['tools/coopmwm/'+name+'.py']=checked(ops.LOCAL/'tools/coopmwm'/(name+'.py')).read_text(encoding='utf-8')
    stage=json.loads(ops.stage('cpuQ',SOURCE,files))
    out=ops.ssh.remote('cpuQ',f'''import sys,os,json,unittest,time
from unittest.mock import patch
sys.path[:0]=[{SOURCE!r}+'/pytools','/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime'];os.nice(19)
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName('tools.coopmwm.'+n) for n in ('test_tail_drain_cpu','test_centre_fair_cpu','test_capacity_broker_cpu','test_quota_admission_cpu','test_warm_broker_cpu','test_warm_transport_cpu'));start=time.monotonic()
with patch('subprocess.Popen',side_effect=AssertionError('no physics/inference in qualification')):result=unittest.TextTestRunner(stream=sys.stdout).run(suite)
print(json.dumps(dict(pass_=result.wasSuccessful(),tests=result.testsRun,skips=len(result.skipped),sim_s=0,root={SOURCE!r},wall_s=time.monotonic()-start)))''',timeout=60)
    print(out);v=json.loads(out.strip().splitlines()[-1]);v['source_hashes']=stage['hashes'];write(ops.EVIDENCE/'CPU_coopmwm20261102_taildraincpu2.json',v)
    assert v['pass_'] and v['sim_s']==v['skips']==0


def deploy():
    cpu=read(ops.EVIDENCE/'CPU_coopmwm20261102_taildraincpu2.json');assert cpu['pass_'] and cpu['sim_s']==cpu['skips']==0
    v=dict(source_root=SOURCE,cpu=cpu,new_sumo_sim_s=0,
        rule='one-time pending TAIL handover: only after B-u2 executed quota>=10, finish every assigned search once, decline only NEW unassigned old-wave plan admissions; preserve R-centres/diagnostics; automatically restore original full admission at TAIL_HANDOVER_COMPLETE before B-u3 starts',
        reason='at1793579645 old22/24closed episodes stillt350/t340 and opening further free-slot searches; finite infrastructure boundary required for qualified fan-out deployment; no outcome/plan-gain selection',
        identity_repair='first launcher refused before any signal because serve has no root argv; retain original config/amendment; require exact serve argv, boot/start and bound source PYTHONPATH instead; original broker3706352 must still match',
        invariants='no existing assigned query/SUMO cancellation/replay; same four GPU workers/search/CRN/horizons/grades/trust/arms/budget/deadline; per-slot centre fairness retained; replace only verified broker3706352')
    target=ops.EVIDENCE/'TAIL_BOUNDARY_DRAIN_AMENDMENT_V2.json';assert not target.exists();write(target,v);h=sha(target)
    line='\nS-COOPMWM QUOTA-COMPLETE ONE-TIME TAIL HANDOVER DRAIN REGISTERED amendmentSHA'+h+': '+json.dumps({k:x for k,x in v.items() if k!='cpu'},sort_keys=True)+'\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,os,subprocess,time,hashlib
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.warm_health import assert_running
from tools.coopmwm.quota_transition import state
from tools.coopmwm.tail_drain_ops import retire_broker
from tools.coopmwm.quota_admission import state as quota_state
assert_running(r,'B');assert not (r/'TAIL_DRAIN_BROKER_DISPATCH.json').exists(),'adopt existing broker'
frozen=json.loads((r/'TAIL_PARENT_FROZEN.json').read_text());assert frozen['adopt_update']==2
assert state(frozen['owner'],'tools.coopmwm.quota_runtime')=='T'
assert quota_state(r,2)['executed']>=10,'never drain below executed quota'
for name,h in json.loads({json.dumps(cpu['source_hashes'])!r}).items():assert hashlib.sha256((s/name).read_bytes()).hexdigest()==h
with (r/'AUTHORITY.md').open('a') as f:f.write({line!r})
old=json.loads((r/'CENTRE_FAIR_BROKER_DISPATCH.json').read_text());assert old['pid']==3706352 and old['start_ticks']=='58959212'
config=json.loads((r/'CENTRE_FAIR_CONFIG.json').read_text());config.update(source_root=str(s),source_hashes=json.loads({json.dumps(cpu['source_hashes'])!r}),amendment_sha256={h!r},receipt_tag='_TD2',boundary_drain_until='TAIL_HANDOVER_COMPLETE.json',boundary_drain_update=2,boundary_drain_minimum=10)
immutable(r/{CONFIG!r},config)
pending={{p.stem:json.loads(p.read_text()) for p in (r/'broker_assignments').glob('*.json') if not (r/('responses' if json.loads(p.read_text())['kind']=='plan' else 'centre_responses')/p.name).exists()}}
retire_broker(old)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(s/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime')
with (r/'TAIL_DRAIN_BROKER.log').open('x') as log:
 p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-m','tools.coopmwm.tail_drain_ops','serve'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
stat=pathlib.Path('/proc',str(p.pid),'stat').read_text().split();value=dict(pid=p.pid,start_ticks=stat[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),root=str(r),source=str(s),component='broker',unix=time.time())
immutable(r/'TAIL_DRAIN_BROKER_DISPATCH.json',value)
immutable(r/'TAIL_DRAIN_BROKER_HANDOVER.json',dict(previous=old,current=value,assigned_adopted=pending,inference_replay=False,new_sumo_sim_s=0))
print(json.dumps(value))''',timeout=45)
    value=json.loads(out);write(ops.EVIDENCE/'TAIL_DRAIN_BROKER_DISPATCH.json',value);print(out)


def serve():
    from pathlib import Path
    from .capacity_broker import run
    r=Path(ROOT);c=json.loads((r/CONFIG).read_text())
    for name,h in c['source_hashes'].items():assert sha(Path(c['source_root'])/name)==h,name
    run(r,r/CONFIG)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('qualify','deploy','serve'))
    globals()[p.parse_args().mode]()
