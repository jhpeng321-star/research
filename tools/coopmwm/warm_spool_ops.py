"""One-use CPU-only detached spool qualification; never deploys a GPU/SUMO job."""
import argparse
import json
from . import ops
from .protocol import checked,sha,REGISTRY
from .storage import read,write

ROOT='/home/ubuntu/coopmwm20261102_spoolqual1'


def dispatch():
    target=ops.EVIDENCE/'WARM_SPOOL_PROBE_DISPATCH.json'
    assert not target.exists(),'adopt existing probe; never duplicate'
    files={'tools/coopmwm/'+name:checked(ops.LOCAL/'tools/coopmwm'/name).read_text(encoding='utf-8')
        for name in ('warm_spool.py','test_warm_spool_cpu.py')}
    staged=json.loads(ops.stage('cpuQ',ROOT,files))
    # The detached child emulates the model boundary only, with no torch/WM/SUMO
    # import. This checks Linux process/SSH lifecycle, not numerical WM parity.
    child='''import pathlib,sys,json,os,time,signal,fcntl
from tools.coopmwm.warm_spool import SpoolInput,immutable
r=pathlib.Path(sys.argv[1]);claim=(r/'PROBE.claim').open('a');fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
signal.signal(signal.SIGHUP,signal.SIG_IGN)
immutable(r/'PROBE_OWNER.json',dict(pid=os.getpid(),start_ticks=pathlib.Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
for line in SpoolInput(r,time.time()+300):
 record=json.loads(line);key=record['state_id']
 immutable(r/'forecasts'/(key+'.json'),dict(key=key,result=dict(cpu_fixture=True),new_sumo_sim_s=0))
immutable(r/'PROBE_COMPLETE.json',dict(pass_=True,new_sumo_sim_s=0))
'''
    value=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,subprocess,os,time,sys,unittest,io
r=pathlib.Path({ROOT!r});assert r.is_dir() and not (r/'PROBE_DISPATCH.json').exists()
sys.path.insert(0,str(r/'pytools'))
start=time.monotonic();stream=io.StringIO()
suite=unittest.defaultTestLoader.loadTestsFromName('tools.coopmwm.test_warm_spool_cpu')
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
cpu=dict(pass_=result.wasSuccessful(),tests=result.testsRun,skips=len(result.skipped),wall_s=time.monotonic()-start,log=stream.getvalue(),new_sumo_sim_s=0)
assert cpu['pass_'] and not cpu['skips']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(r/'pytools'))
with (r/'PROBE.log').open('x') as log:
 p=subprocess.Popen([{ops.ssh.PYTHON['cpuQ']!r},'-B','-u','-c',{child!r},str(r)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
s=(pathlib.Path('/proc')/str(p.pid)/'stat').read_text().split()
v=dict(pid=p.pid,start_ticks=s[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),cpu=cpu,new_sumo_sim_s=0)
with (r/'PROBE_DISPATCH.json').open('x') as f:json.dump(v,f)
print(json.dumps(v))''',timeout=30))
    write(target,dict(value,root=ROOT,hashes=staged['hashes']))
    print(json.dumps(value))


def collect():
    target=ops.EVIDENCE/'WARM_SPOOL_QUALIFICATION.json'
    assert not target.exists(),'adopt completed qualification'
    dispatch=read(ops.EVIDENCE/'WARM_SPOOL_PROBE_DISPATCH.json')
    value=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,time
r=pathlib.Path({ROOT!r});sys.path.insert(0,str(r/'pytools'))
from tools.coopmwm.warm_spool import immutable,submit
d=json.loads((r/'PROBE_DISPATCH.json').read_text());owner=json.loads((r/'PROBE_OWNER.json').read_text())
assert all(owner[k]==d[k] for k in ('pid','start_ticks','boot_id'))
p=pathlib.Path('/proc')/str(d['pid']);s=(p/'stat').read_text().split()
assert s[21]==d['start_ticks'] and s[2] in ('R','S','D')
assert pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
immutable(r/'GPU_BINDING.json',dict(cpu_fixture_only=True))
record=dict(state_id='a'*64,rows=[],history=[],commands={{}})
submit(r,record);submit(r,record)
deadline=time.monotonic()+15;dest=r/'forecasts'/('a'*64+'.json')
while not dest.exists() and time.monotonic()<deadline:time.sleep(.1)
assert dest.exists(),'detached worker did not process request after SSH session ended'
submit(r,record)
immutable(r/'SPOOL_STOP.json',dict(cpu_probe_complete=True))
while not (r/'PROBE_COMPLETE.json').exists() and time.monotonic()<deadline:time.sleep(.1)
assert (r/'PROBE_COMPLETE.json').exists()
claims=list((r/'inference_claims').glob('*.json'));assert len(claims)==1
print(json.dumps(dict(pass_=True,owner=owner,previous_ssh_session_ended=True,stdin_devnull=True,
 requests_delivered=3,inference_claims=1,forecast=json.loads(dest.read_text()),
 complete=json.loads((r/'PROBE_COMPLETE.json').read_text()),new_sumo_sim_s=0)))''',timeout=25))
    qualification=dict(value,cpu=dispatch['cpu'],source_hashes=dispatch['hashes'],root=ROOT,
        contract='detached disk-spool transport only; frozen warm WM/search untouched; no activation or physics',
        gpu_startup_parity_required=True)
    write(target,qualification)
    line=('\nS-COOPMWM MAIN0133 REMOTE WORKER SPOOL ENGINEERING QUALIFIED evidenceSHA'+sha(target)+
        ': '+qualification['contract']+'; 9CPU PASS/0skips, detached Linux probe survives ended SSH/stdinDEVNULL; '
        'three identical deliveries -> one inference claim/result. Interrupted inference is fatal/no replay; '
        'GPU worker activation remains source-bound startup-parity and budget gated. Zero SUMO.\n')
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    write(ops.EVIDENCE/'WARM_SPOOL_REGISTRATION.json',dict(line=line,evidence_sha256=sha(target),new_sumo_sim_s=0))
    print(json.dumps(qualification))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('dispatch','collect'))
    globals()[parser.parse_args().mode]()
