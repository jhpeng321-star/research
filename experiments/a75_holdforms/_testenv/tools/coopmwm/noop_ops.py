"""Registered report-only handover for MAIN07:18 per-update physical-null counts."""
import argparse
import json
from . import ops
from .storage import read,write
from .protocol import checked,REGISTRY,sha
from .warm_deploy import ROOT

SOURCE='/home/ubuntu/coopmwm20261102_nullreport1'


def qualify():
    payload=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,hashlib
r=pathlib.Path({ROOT!r});c=json.loads((r/'EXACT_VIABILITY_REPORT_CONFIG.json').read_text());s=pathlib.Path(c['source_root']);files={{}}
for name,h in c['source_hashes'].items():
 p=s/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==h;files[name.removeprefix('pytools/')]=p.read_text()
print(json.dumps(files))''',timeout=40))
    for n in ('noop_report','noop_ops','test_noop_report_cpu','viability_report','test_viability_report_cpu'):
        payload['tools/coopmwm/'+n+'.py']=checked(ops.LOCAL/'tools/coopmwm'/(n+'.py')).read_text(encoding='utf-8')
    stage=json.loads(ops.stage('cpuQ',SOURCE,payload))
    out=ops.ssh.remote('cpuQ',f'''import sys,os,json,unittest,time
from unittest.mock import patch
sys.path.insert(0,{SOURCE!r}+'/pytools');os.nice(19)
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName('tools.coopmwm.'+n) for n in ('test_noop_report_cpu','test_viability_report_cpu'))
begin=time.monotonic()
with patch('subprocess.Popen',side_effect=AssertionError('no physics/inference in report qualification')):
 result=unittest.TextTestRunner(stream=sys.stdout).run(suite)
print(json.dumps(dict(pass_=result.wasSuccessful(),tests=result.testsRun,skips=len(result.skipped),sim_s=0,wall_s=time.monotonic()-begin)))''',timeout=60)
    print(out);q=json.loads(out.strip().splitlines()[-1]);q.update(source_hashes=stage['hashes'],root=SOURCE)
    write(ops.EVIDENCE/'CPU_coopmwm20261102_nullreport1.json',q)
    assert q['pass_'] and q['skips']==q['sim_s']==0


def deploy():
    q=read(ops.EVIDENCE/'CPU_coopmwm20261102_nullreport1.json')
    assert q['pass_'] and q['skips']==q['sim_s']==0
    config=dict(source_root=SOURCE,source_hashes=q['source_hashes'],qualification=q,
        rule='all qualified same-B/seed TRAIN exact grades retained; inspect every exact-zero observable original/without-plan window; count identical whole-road sensed motion/rewards by update; distinguish saving_vs_policy from whole-road saving_vs_MAPPO',
        training_changes=False,new_sumo_sim_s=0,WM_calls=0,original_deadline_retained=True)
    p=ops.EVIDENCE/'PHYSICAL_NULL_REPORT_AMENDMENT.json'
    if p.exists():assert read(p)==config,'adopt identical registered report configuration'
    else:write(p,config)
    h=sha(p);config['amendment_sha256']=h
    line='\nS-COOPMWM MAIN0718 PHYSICAL-NULL REPORTER REGISTERED amendmentSHA'+h+': CPU9PASS/zeroSUMO/inference; report only, all exact zeros/losses retained, inspect exact-zero original/without-plan sensed tapes with complete window/clock/receipt checks, per-update null counts and both WM contrasts; no learning/trust/library/quota/search/filter change. Replace only receipt-owned reporter3357311 after PID/start/boot/cmd verification; original learning/search/grader/observers and deadline retained. Existing publisher collects immutable B_EXACT_VIABILITY reports.\n'
    registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(line)
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,os,subprocess,time,hashlib
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.warm_spool import immutable
# The older reporter snapshot predates quota_transition. Use only the already
# qualified CPU4 ownership helper; the reporting modules stay in their snapshot.
import tools.coopmwm
tools.coopmwm.__path__.append('/home/ubuntu/coopmwm20261102_quotacpu4/pytools/tools/coopmwm')
from tools.coopmwm.quota_transition import terminate
assert not (r/'PHYSICAL_NULL_REPORT_DISPATCH.json').exists(),'adopt existing reporter'
c=json.loads({json.dumps(config)!r})
for name,h in c['source_hashes'].items():assert hashlib.sha256((s/name).read_bytes()).hexdigest()==h
old=json.loads((r/'EXACT_VIABILITY_REPORT_DISPATCH.json').read_text());assert old['pid']==3357311 and old['start_ticks']=='58085285'
with (r/'AUTHORITY.md').open('a') as f:f.write({line!r})
immutable(r/'PHYSICAL_NULL_REPORT_CONFIG.json',c)
terminate(old,'tools.coopmwm.viability_report')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(s/'pytools'))
with (r/'PHYSICAL_NULL_REPORT.log').open('x') as f:
 p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-m','tools.coopmwm.viability_report',str(r),str(r/'PHYSICAL_NULL_REPORT_CONFIG.json')],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
stat=pathlib.Path('/proc',str(p.pid),'stat').read_text().split()
v=dict(pid=p.pid,start_ticks=stat[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),source=str(s),root=str(r),amendment_sha256=c['amendment_sha256'])
immutable(r/'PHYSICAL_NULL_REPORT_DISPATCH.json',v)
print(json.dumps(v))''',timeout=45)
    v=json.loads(out);write(ops.EVIDENCE/'PHYSICAL_NULL_REPORT_DISPATCH.json',v);print(out)


def status():
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,time,sys
r=pathlib.Path({ROOT!r});d=json.loads((r/'PHYSICAL_NULL_REPORT_DISPATCH.json').read_text())
assert d['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
p=pathlib.Path('/proc',str(d['pid']),'stat');stat=p.read_text().split() if p.exists() else None
assert not stat or stat[21]==d['start_ticks']
reports=[]
for p in r.glob('B_EXACT_VIABILITY_*.json'):
 v=json.loads(p.read_text());e=v.get('exact_plan_evidence',{{}})
 if 'physical_nulls_by_update' in e:reports.append(dict(name=p.name,unix=v['timestamp_utc'],overall=e['overall'],nulls=e['physical_nulls_by_update'],zero_plans=[{{k:row.get(k) for k in ('grade_id','update','world','plan_origin','predicted_saving_vs_policy','predicted_saving_vs_MAPPO','null_classification')}} for row in e['rows'] if row['exact_saving']==0]))
print(json.dumps(dict(unix=time.time(),owner=d,state=stat[2] if stat else 'exited',reports=sorted(reports,key=lambda v:v['unix']),log=(r/'PHYSICAL_NULL_REPORT.log').read_text()[-3000:],new_sumo_sim_s=0)))''',timeout=45)
    v=json.loads(out);write(ops.EVIDENCE/f'PHYSICAL_NULL_REPORT_STATUS_{int(v["unix"])}.json',v);print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('qualify','deploy','status'))
    globals()[p.parse_args().mode]()
