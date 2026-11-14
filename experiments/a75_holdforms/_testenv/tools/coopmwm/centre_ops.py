"""Register R's corrected shared policy-centre calculation before any B/R outcome."""
import argparse
import json
import subprocess
import sys
import time
from . import ops
from .storage import read,write
from .protocol import checked,sha,REGISTRY
from .warm_deploy import ROOT
from .centre_bridge import GPU_ROOT

CPU_R='/home/ubuntu/coopmwm20261102_mppicpu7'


def prepare():
    target=ops.EVIDENCE/'WARM_R_CENTRE_REGISTRATION.json';assert not target.exists()
    cpu=read(ops.EVIDENCE/('CPU_'+CPU_R.rsplit('/',1)[1]+'.json'))
    assert cpu['pass_'] and not cpu['skips'] and not cpu['sim_s']
    files=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,hashlib
r=pathlib.Path({CPU_R!r});hashes=json.loads({json.dumps(cpu['source_hashes'])!r});out={{}}
for n,h in hashes.items():
 p=r/n;assert hashlib.sha256(p.read_bytes()).hexdigest()==h;out[n.removeprefix('pytools/')]=p.read_text()
print(json.dumps(out))''',timeout=35))
    staged=json.loads(ops.stage('gpuA',GPU_ROOT,files))
    core=[n for n in cpu['source_hashes'] if n.startswith('pytools/tools/wmteach/')]
    core+=['pytools/tools/coopmwm/'+n+'.py' for n in ('wm','plan_wm','speed2_runtime','speed2_transfers','mppi_search','mppi_runtime','policy_wm','warm_runtime','speed2_memory','fused','plans')]
    audit=json.loads(ops.ssh.remote('gpuA',f'''import pathlib,json,hashlib
r=pathlib.Path({GPU_ROOT!r});b=pathlib.Path('/root/coopmwm20261102_warmservice1_w0');matches={{}}
for n in json.loads({json.dumps(core)!r}):
 h=hashlib.sha256((r/n).read_bytes()).hexdigest();assert h==hashlib.sha256((b/n).read_bytes()).hexdigest(),n;matches[n]=h
print(json.dumps(dict(pass_=True,matches=matches,new_sumo_sim_s=0)))''',timeout=30))
    b=read(ops.EVIDENCE/'WARM_A48_BINDING.json')
    amendment=dict(cpu=cpu,source_root=CPU_R,hashes=cpu['source_hashes'],core_gpu_audit=audit,
        contract='R uses exactly B imagined_centres(current R policy,warm memory,K1,H300); candidates never WM-scored; random pick, no library/covariance learning',
        schedule='baseline-only GPU helper admitted after B-u4 pause, so it cannot delay B search',
        added_sumo_sim_s=0,extra_gpu_cost='one K1/H300 two-comparator rollout per sampled matched R plan',
        superseded='held-current-command R centre; no R training outcome has used it')
    write(ops.EVIDENCE/'WARM_R_CENTRE_AMENDMENT.json',amendment);h=sha(ops.EVIDENCE/'WARM_R_CENTRE_AMENDMENT.json')
    line=('\nS-COOPMWM R SHARED-CENTRE ENGINEERING CORRECTION REGISTERED amendmentSHA'+h+
        ': before any B/R outcome, R uses SAME warm K1/H300 current-policy trajectory extraction as B (current R weights), '
        'then generic random perturbations; ZERO candidate WM scores/rankings, no library or learned covariance. '
        'This supersedes the held-current-command proxy which would confound B-R. Baseline-only helper on gpuA starts '
        'after B-u4 pause (primary priority), same original deadline and sim-s allocation; CPU/source/hash-audited. '
        'No R physics has used the old proxy; no new SUMO for qualification.\n')
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    runtime=dict(source_root=CPU_R,hashes=cpu['source_hashes'],cpu=cpu,amendment_sha256=h)
    gpu=dict(hashes=staged['hashes'],cpu=cpu,core_gpu_audit=audit,deadline_unix=b['gpu_deadline'])
    ops.stage('gpuA',GPU_ROOT,{'CENTRE_BINDING.json':json.dumps(gpu,sort_keys=True)})
    value=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json
r=pathlib.Path({ROOT!r});assert r.is_dir()
for n in ('centre_requests','centre_responses'):(r/n).mkdir(exist_ok=True)
with (r/'WARM_R_RUNTIME_BINDING.json').open('x') as f:json.dump(json.loads({json.dumps(runtime)!r}),f,sort_keys=True)
with (r/'AUTHORITY.md').open('a') as f:f.write({line!r})
print(json.dumps(dict(source_root={CPU_R!r},registered=True,new_sumo_sim_s=0)))''',timeout=30))
    write(target,dict(line=line,amendment_sha256=h,runtime=runtime,gpu=gpu))
    print(json.dumps(value))


def bridge():
    assert (ops.EVIDENCE/'WARM_R_CENTRE_REGISTRATION.json').exists()
    target=ops.EVIDENCE/'WARM_CENTRE_BRIDGE_DISPATCH.json';assert not target.exists()
    with (ops.EVIDENCE/'WARM_CENTRE_BRIDGE.log').open('x',encoding='utf-8') as log:
        p=subprocess.Popen([sys.executable,'-X','utf8','-m','tools.coopmwm.centre_bridge'],cwd=ops.LOCAL,stdout=log,stderr=subprocess.STDOUT)
    value=dict(pid=p.pid,unix=time.time(),purpose='R baseline centre helper waits for B-u4 pause; no candidate scoring')
    write(target,value);print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','bridge'));a=p.parse_args();globals()[a.mode]()
