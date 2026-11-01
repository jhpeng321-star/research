"""One-shot phase staging, CPU certification, launch and read-only status."""
import argparse
import hashlib
import json
from pathlib import Path
from tools.coopbandit.ops import remote

ROOT='/home/ubuntu/coopfree20261101_owner1511'
OLD='/home/ubuntu/coopfree20261101_build5'
LOCAL=Path(__file__).resolve().parent
REG=Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')
SOURCES=('coopfree_continue.py','test_coopfree_continue.py','coopfree_learning_checks.py','test_coopfree_learning_checks.py')


def stage_cpu():
    sources={n:(LOCAL/n).read_text(encoding='utf-8') for n in SOURCES}
    design=(LOCAL.parent/'docs/paper_results/COOPFREE_DESIGN.md').read_text(encoding='utf-8')
    return remote('cpuQ',f'''import pathlib,json,sys,os,shutil,time,unittest,hashlib
from unittest.mock import patch
r=pathlib.Path({ROOT!r});old=pathlib.Path({OLD!r});assert r.parent.is_dir()
assert not (r/'CONTINUE_DISPATCH.json').exists(),'already dispatched'
if not r.exists():
 for receipt in ('DISPATCH.json','HOLD_DISPATCH.json','BATCH_DISPATCH.json'):
  if not (old/receipt).exists():continue
  d=json.loads((old/receipt).read_text());p=pathlib.Path('/proc/'+str(d['pid'])+'/stat')
  assert not p.exists() or p.read_text().split()[2]=='Z' or p.read_text().split()[21]!=str(d.get('start_ticks',d.get('ticks'))),'prior job alive'
 assert not any((p/'START.json').exists() and not (p/'CLOSE.json').exists() for p in (old/'jobs').iterdir())
 r.mkdir();(r/'pytools').mkdir();(r/'pytools/tools').mkdir();(r/'jobs').mkdir()
 for p in (old/'pytools').iterdir():
  if p.name not in ('tools','__pycache__'): (r/'pytools'/p.name).symlink_to(p,target_is_directory=p.is_dir())
 for p in (old/'pytools/tools').iterdir():
  if p.name!='__pycache__': (r/'pytools/tools'/p.name).symlink_to(p,target_is_directory=p.is_dir())
 for name in ('BINDING.json','CPU_APPROVED.json','APPROVAL.json','EXPOSURE.json','COOPFREE_DESIGN.md','BANDIT.json','bandit_policy.pkl','PRIOR.json','PLAN_001.json','HOLD_PLAN_001.json','HOLD_UPDATE_001.json','policy_hold_u001.pt','HOLD_ZERO_DEV.json','HOLD_DEV_001.json'):
  assert (old/name).exists();(r/name).symlink_to(old/name)
 shutil.copyfile(old/'CHARGES.jsonl',r/'CHARGES.jsonl')
 for p in (old/'jobs').iterdir(): (r/'jobs'/p.name).symlink_to(p,target_is_directory=True)
 (r/'INHERITED.json').write_text(json.dumps(dict(root=str(old),ledger_sha256=hashlib.sha256((r/'CHARGES.jsonl').read_bytes()).hexdigest(),no_open_attempts=True)))
for name,text in json.loads({json.dumps(sources)!r}).items():
 p=r/'pytools/tools'/name
 if p.exists() and p.read_text()!=text:
  assert not (r/'CONTINUE_CPU.json').exists() and not p.is_symlink(),'immutable continuation source'
  p.rename(p.with_name(p.name+'.precert_'+str(time.time_ns())))
 if not p.exists():p.write_text(text)
p=r/'CONTINUE_DESIGN.md';text={design!a}
if p.exists() and p.read_text()!=text:
 assert not (r/'CONTINUE_CPU.json').exists()
 p.rename(p.with_name(p.name+'.precert_'+str(time.time_ns())))
if not p.exists():p.write_text(text)
os.nice(19);os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.path.insert(0,str(r/'pytools'))
from tools.leverprobe.backend import setup
setup('/home/ubuntu/e3final','/home/ubuntu/mappospeed/production2/policy_latest.pt')
os.environ['COOPFREE_CPU_CHECKPOINT']='/home/ubuntu/mappospeed/production2/policy_latest.pt'
from tools.coopfree.storage import write
from tools.coopfree.prior import fit
exam=pathlib.Path('/home/ubuntu/coopbandit20261031_aggregate_v3/snapshots/full/exam')
manifest=json.loads((exam/'MANIFEST.json').read_text())
for n in ('states.jsonl','labels.csv'):assert hashlib.sha256((exam/n).read_bytes()).hexdigest()==manifest[n]
if not (r/'CONTINUE_PRIOR.json').exists():write(r/'CONTINUE_PRIOR.json',fit(exam))
begin=time.monotonic()
suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromName(n) for n in ('tools.coopfree.test_cpu','tools.test_coopfree_hold','tools.test_coopfree_continue')])
from tools.test_coopfree_learning_checks import LearningChecks
# The old Windows publisher compile test is independent of this Linux production phase.
for name in unittest.defaultTestLoader.getTestCaseNames(LearningChecks):
 if name!='test_observer_remote_source_compiles_without_dispatch':suite.addTest(LearningChecks(name))
with patch('subprocess.Popen',side_effect=AssertionError('SUMO forbidden in CPU gate')):
 result=unittest.TextTestRunner(verbosity=2).run(suite)
cert=dict(pass_=result.wasSuccessful(),tests=result.testsRun,skips=len(result.skipped),sim_s=0,wall_s=time.monotonic()-begin,sources={{n:hashlib.sha256((r/'pytools/tools'/n).read_bytes()).hexdigest() for n in {list(SOURCES)!r}}},design_sha256=hashlib.sha256((r/'CONTINUE_DESIGN.md').read_bytes()).hexdigest(),prior_sha256=hashlib.sha256((r/'CONTINUE_PRIOR.json').read_bytes()).hexdigest())
if cert['pass_'] and not cert['skips'] and not (r/'CONTINUE_CPU.json').exists():write(r/'CONTINUE_CPU.json',cert)
print(json.dumps(cert));assert cert['pass_'] and not cert['skips']
''',timeout=45)


def launch():
    text=REG.read_text(encoding='utf-8')
    approval=next(s for s in text.splitlines() if s.startswith('- 15:11 S-COOPFREE BUDGET EXTENSION APPROVED'))
    line=next(s for s in text.splitlines() if s.startswith('- S-COOPFREE OWNER15:11 CONTINUATION REGISTERED'))
    expected={n:hashlib.sha256((LOCAL/n).read_text(encoding='utf-8').encode()).hexdigest() for n in SOURCES}
    return remote('cpuQ',f'''import pathlib,json,os,sys,subprocess
r=pathlib.Path({ROOT!r});assert r.is_dir()
assert not (r/'CONTINUE_DISPATCH.json').exists() and not (r/'STOP.json').exists(),'inspect existing dispatch; no duplicate'
sys.path.insert(0,str(r/'pytools'))
from tools.coopfree.storage import read,write
from tools.coopfree_continue import CAPS,verify,configure
cert=read(r/'CONTINUE_CPU.json');assert cert['sources']==json.loads({json.dumps(expected)!r})
write(r/'CONTINUE_AUTHORITY.json',dict(total_cap=6000000,caps=CAPS,approval_line={approval!a},registration_line={line!a},design_sha256=cert['design_sha256']))
configure();verify(r)
import sumo
home=str(pathlib.Path(sumo.__file__).parent)
env=dict(os.environ,PYTHONPATH=str(r/'pytools'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SUMO_HOME=home,PATH=home+'/bin:'+str(pathlib.Path(sys.executable).parent)+':'+os.environ.get('PATH',''))
log=(r/'CONTINUE_SUPERVISOR.log').open('x')
p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-m','tools.coopfree_continue'],cwd=r,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=p.pid,start_ticks=pathlib.Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],root=str(r),workers=4,nice=19,max_load=12,total_cap=6000000,cpu=cert)
write(r/'CONTINUE_DISPATCH.json',receipt);print(json.dumps(receipt))
''',timeout=25)


def status():
    return remote('cpuQ',f'''import pathlib,json,os,shutil,sys
r=pathlib.Path({ROOT!r});assert r.is_dir();sys.path.insert(0,str(r/'pytools'))
from tools.coopfree_continue import configure
from tools.coopfree.storage import read,status,ledger
configure();out=dict(root=str(r),load=os.getloadavg(),disk_free=shutil.disk_usage(r).free)
for n in ('CONTINUE_DISPATCH.json','STOP.json','CONTINUE_BUDGET_STOP.json','CONTINUE_FIX_REQUIRED.json','CONTINUE_FIX_AUTHORITY.json','CONTINUE_COMPLETE.json'):
 if (r/n).exists():out[n]=read(r/n)
if 'CONTINUE_DISPATCH.json' in out:
 d=out['CONTINUE_DISPATCH.json'];p=pathlib.Path('/proc/'+str(d['pid'])+'/stat')
 out['alive']=p.exists() and p.read_text().split()[21]==d['start_ticks'] and p.read_text().split()[2]!='Z'
out['accounting']=status(r);out['updates']=[int(p.stem.split('_')[-1]) for p in sorted(r.glob('HOLD_UPDATE_*.json'))]
out['plans']=[dict(update=read(p)['update'],tasks=len(read(p)['tasks']),complete=sum((r/'jobs'/t['id']/'CLOSE.json').exists() for t in read(p)['tasks']),charge=read(p)['charge'],cached_holds=read(p)['cached_holds']) for p in sorted(r.glob('CONTINUE_PLAN_*.json'))]
out['active_owned']=[dict(id=p.name,start=read(p/'START.json')) for p in (r/'jobs').iterdir() if not p.is_symlink() and (p/'START.json').exists() and not (p/'CLOSE.json').exists()]
print(json.dumps(out))
''',timeout=25)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('stage-cpu','launch','status'));a=p.parse_args()
    output={'stage-cpu':stage_cpu,'launch':launch,'status':status}[a.mode]()
    print(output)
    evidence=LOCAL/'coopfree/evidence'
    if a.mode in ('launch','status','stage-cpu'):
        import time
        (evidence/f'CONTINUE_{a.mode.upper()}_{time.time_ns()}.json').write_text(output,encoding='utf-8')
