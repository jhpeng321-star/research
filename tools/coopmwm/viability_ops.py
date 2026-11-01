"""One-use CPU-only exact-evidence publisher deployment. No simulator dispatch."""
import json
from . import ops
from .storage import read,write
from .protocol import checked,REGISTRY,sha
from .warm_deploy import ROOT

SOURCE='/home/ubuntu/coopmwm20261102_viabilityreport1'


def deploy():
    target=ops.EVIDENCE/'EXACT_VIABILITY_REPORT_DISPATCH.json';assert not target.exists()
    cpu=read(ops.EVIDENCE/'CPU_coopmwm20261102_exactcpu5.json')
    frozen=json.loads(ops.stage_frozen('cpuQ',SOURCE,cpu))
    files={'tools/coopmwm/'+n:checked(ops.LOCAL/'tools/coopmwm'/n).read_text(encoding='utf-8')
           for n in ('viability_report.py','test_viability_report_cpu.py')}
    extra=json.loads(ops.stage('cpuQ',SOURCE,files))
    hashes=dict(frozen['hashes'],**extra['hashes'])
    out=ops.ssh.remote('cpuQ',f'''import os,sys,time,json,unittest
os.nice(19);os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');sys.path.insert(0,{SOURCE!r}+'/pytools')
from unittest.mock import patch
start=time.monotonic()
with patch('subprocess.Popen',side_effect=AssertionError('physics forbidden during report tests')):
 result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('tools.coopmwm.test_viability_report_cpu'))
print(json.dumps(dict(pass_=result.wasSuccessful(),tests=result.testsRun,skips=len(result.skipped),sim_s=0,wall_s=time.monotonic()-start)))''',timeout=60)
    print(out);qualification=json.loads(out.strip().splitlines()[-1])
    assert qualification['pass_'] and qualification['skips']==qualification['sim_s']==0
    config=dict(source_root=SOURCE,source_hashes=hashes,qualification=qualification,
        role='read-only TRAIN exact-grade evidence by type/regime/rank/new-known/group-size/update; never read final results or influence training',
        estimator='mean executed-plan exact saving; 10000 whole-world cluster resamples RNG20261101; keep negative grades; <2worlds noCI; unqualified/foreign-panel grades STOP; unresolved excluded, never imputed',
        sampling='every qualified B/seed20261101 TRAIN grade; no result selection; publish via existing learning observer, 300s polls')
    p=ops.EVIDENCE/'EXACT_VIABILITY_REPORT_AMENDMENT.json';write(p,config);h=sha(p);config['amendment_sha256']=h
    line='\nS-COOPMWM EXACT TRAIN VIABILITY REPORTER REGISTERED amendmentSHA'+h+': CPU5PASS/zeroSUMO, observational only; every qualified same-B-seed TRAIN exact grade including losses, whole-world10000bootstrapCI by type/regime/predictedrank/new-known/group-size/update; never impute missing/unresolved grades, one-worldCI unavailable; byte-identical training/search/rewards/trust/library, no final outcome reads, no physics/inference submissions/newallocation; detached nice19/CPU-only 300s receipts published by existing observer.\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    value=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,sys,os,json,time,subprocess,hashlib
r=pathlib.Path({ROOT!r});source=pathlib.Path({SOURCE!r});assert r.is_dir() and source.is_dir()
sys.path.insert(0,str(source/'pytools'))
from tools.coopmwm.warm_spool import immutable
c=json.loads({json.dumps(config)!r})
for n,h in c['source_hashes'].items():assert hashlib.sha256((source/n).read_bytes()).hexdigest()==h,n
assert not (r/'EXACT_VIABILITY_REPORT_DISPATCH.json').exists()
with (r/'AUTHORITY.md').open('a') as f:f.write({line!r})
immutable(r/'EXACT_VIABILITY_REPORT_CONFIG.json',c)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(source/'pytools'))
with (r/'EXACT_VIABILITY_REPORT.log').open('x') as log:
 p=subprocess.Popen(['nice','-n','19',{ops.ssh.PYTHON['cpuQ']!r},'-B','-u','-m','tools.coopmwm.viability_report',str(r),str(r/'EXACT_VIABILITY_REPORT_CONFIG.json')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
s=pathlib.Path('/proc',str(p.pid),'stat').read_text().split()
v=dict(pid=p.pid,start_ticks=s[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),source=str(source),root=str(r),amendment_sha256=c['amendment_sha256'])
immutable(r/'EXACT_VIABILITY_REPORT_DISPATCH.json',v);print(json.dumps(v))''',timeout=40))
    write(target,dict(value,line=line,qualification=qualification));print(json.dumps(value))


if __name__=='__main__':deploy()
