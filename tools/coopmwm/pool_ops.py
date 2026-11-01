"""One-use deployment of a logging-only exporter beside existing GPU owners."""
import argparse
import json
from . import ops,exact_ops
from .protocol import checked,REGISTRY,sha
from .storage import read,write

EXPORT='/root/coopmwm20261102_poolexport1'


def stage(host):
    files={f'tools/coopmwm/{name}.py':checked(ops.LOCAL/f'tools/coopmwm/{name}.py').read_text(encoding='utf-8')
           for name in ('pool_export','test_pool_export_cpu')}
    staged=json.loads(ops.stage(host,EXPORT,files))
    result=ops.ssh.remote(host,f'''import sys,unittest,json,time
sys.path.insert(0,{EXPORT!r}+'/pytools')
start=time.monotonic();r=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('tools.coopmwm.test_pool_export_cpu'))
print(json.dumps(dict(pass_=r.wasSuccessful(),tests=r.testsRun,skips=len(r.skipped),wall_s=time.monotonic()-start,sim_s=0)))''',timeout=30)
    v=json.loads(result.strip().splitlines()[-1]);print(result)
    assert v['pass_'] and v['skips']==0
    write(ops.EVIDENCE/f'POOL_EXPORT_STAGE_{host}.json',dict(staged=staged,cpu=v))


def register():
    target=ops.EVIDENCE/'POOL_EXPORT_REGISTRATION.json';assert not target.exists()
    stages={h:read(ops.EVIDENCE/f'POOL_EXPORT_STAGE_{h}.json') for h in ('gpuA','gpuR')}
    assert all(v['cpu']['pass_'] and v['cpu']['skips']==0 for v in stages.values())
    value=dict(stages=stages,reason='MAIN0413: live RecordingEvaluator already embeds full pools; export/discovery fix only',
        contract='read completed B forecasts; export embedded full pools accepted AND rejected idempotently; SHA/byte parity before-after; no extra WM calls; no search/source edits or replay; never activate proposer',
        deadline_unix=read(ops.EVIDENCE/'WARM_A48_BINDING.json')['gpu_deadline'],new_sumo_sim_s=0)
    p=ops.EVIDENCE/'POOL_EXPORT_AMENDMENT.json';write(p,value);h=sha(p)
    line='\nS-COOPMWM MAIN0413 FULL-POOL LOGGING EXPORT REGISTERED amendmentSHA'+h+': '+value['contract']+'; same original GPU deadline; zero SUMO.\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    write(target,dict(amendment_sha256=h,line=line,**value));print(json.dumps(dict(registered=True,sha256=h)))


def dispatch(host,root=None,receipt_prefix='POOL_EXPORT',require_ready=True):
    reg=read(ops.EVIDENCE/'POOL_EXPORT_REGISTRATION.json')
    root=root or (exact_ops.GPU if host=='gpuA' else exact_ops.GPU_R)
    code=f'''import pathlib,json,sys,subprocess,os,hashlib
r=pathlib.Path({root!r});source=pathlib.Path({EXPORT!r});assert r.is_dir() and source.is_dir()
sys.path.insert(0,str(r/'pytools'))
from tools.coopmwm.warm_rpc import status
from tools.coopmwm.warm_spool import immutable
v=status(r);assert v['alive'] and (v['ready'] or not {require_ready!r}) and not v['failed']
assert not (r/'PROPOSER_EXPORT_DISPATCH.json').exists()
for name,h in {reg['stages'][host]['staged']['hashes']!r}.items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==h
cmd=['nice','-n','19',{ops.ssh.PYTHON[host]!r},'-B','-u',str(source/'pytools/tools/coopmwm/pool_export.py'),str(r),'--deadline',{str(reg['deadline_unix'])!r}]
with (r/'PROPOSER_EXPORT.log').open('x') as log:
 p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
s=pathlib.Path('/proc',str(p.pid),'stat').read_text().split()
d=dict(pid=p.pid,start_ticks=s[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),amendment_sha256={reg['amendment_sha256']!r},root=str(r),source=str(source),gpu_owner=v['receipts']['SPOOL_OWNER.json'])
immutable(r/'PROPOSER_EXPORT_DISPATCH.json',d);print(json.dumps(d))'''
    v=json.loads(ops.ssh.remote(host,code,timeout=30))
    write(ops.EVIDENCE/f'{receipt_prefix}_DISPATCH_{host}.json',v);print(json.dumps(v))


def status(host):
    root=exact_ops.GPU if host=='gpuA' else exact_ops.GPU_R
    v=json.loads(ops.ssh.remote(host,f'''import pathlib,json,time
r=pathlib.Path({root!r});d=json.loads((r/'PROPOSER_EXPORT_DISPATCH.json').read_text())
assert pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
p=pathlib.Path('/proc',str(d['pid']),'stat');s=p.read_text().split() if p.exists() else None
assert not s or s[21]==d['start_ticks']
audits=[json.loads(p.read_text()) for p in sorted((r/'proposer_export_audits').glob('*.json'))]
print(json.dumps(dict(unix=time.time(),owner=d,state=s[2] if s else 'exited',exported=len(audits),audits=audits)))''',timeout=30))
    write(ops.EVIDENCE/f'POOL_EXPORT_STATUS_{host}_{int(v["unix"])}.json',v);print(json.dumps(v))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('stage','register','dispatch','status'));p.add_argument('--host',choices=('gpuA','gpuR'))
    a=p.parse_args();register() if a.mode=='register' else globals()[a.mode](a.host)
