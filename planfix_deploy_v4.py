"""Prepare/qualify/dispatch once; prior dispatch or partial physics always refuses."""
from pathlib import Path
from planfix_ops import remote

wrapper=Path(__file__).with_name('planfix_bank_v4.py').read_text()
code=r'''import ast,hashlib,json,os,subprocess,time
from pathlib import Path
r=Path('/home/ubuntu/coopmwm20261105_clean1');w=r/'plan_fix'
assert not (w/'BANK_DISPATCH_v4.json').exists(),'adopt prior dispatch, never repeat'
prior=json.loads((w/'BANK_DISPATCH_v3.json').read_text())
p=Path('/proc')/str(prior['pid'])/'stat'
if p.exists():
 fields=p.read_text().split();assert fields[21]!=prior['start_ticks'] or fields[2]=='Z'
assert 'PLANFIX own-wave hold' in (w/'BANK_JOB_v3.log').read_text()
charges=[json.loads(line) for line in (r/'CHARGES.jsonl').read_text().splitlines()]
assert sum(row['sim_s'] for row in charges if row['phase']=='plan_fix')==2400
assert len(list((w/'bank_verification_v3').glob('*/STARTED.json')))==1
assert not list((w/'bank_verification_v3').rglob('RESULT.json*'))
assert not (w/'bank_verification_v4').exists()
old=w/'ops_4ea920bda5d1';ops=w/'ops_v4_ownerstopped';ops.mkdir(exist_ok=True)
manifest={}
for name,sha in prior['source_manifest'].items():
 data=(old/name).read_bytes();assert hashlib.sha256(data).hexdigest()==sha
 target=ops/name
 if target.exists():assert target.read_bytes()==data
 else:
  with target.open('xb') as out:out.write(data)
 manifest[name]=sha
text=__WRAPPER__;ast.parse(text)
target=ops/'planfix_bank_v4.py'
if target.exists():assert target.read_text()==text
else:
 with target.open('x') as out:out.write(text)
manifest[target.name]=hashlib.sha256(target.read_bytes()).hexdigest()
source='/home/ubuntu/coopmwm20261105_clean1_src_7cc6edaa'
env=dict(os.environ,PYTHONPATH=str(ops)+':'+source+'/pytools:'+source+'/runtime_src:'+source+'/WM_PACKAGE:/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime',
 CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 LD_PRELOAD='/home/ubuntu/cpus_env/libmkl_intel_shim.so',PYTHONDONTWRITEBYTECODE='1')
env['SUMO_HOME']='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo'
env['PATH']=env['SUMO_HOME']+'/bin:/home/ubuntu/miniconda3/bin:'+env['PATH']
python='/home/ubuntu/miniconda3/bin/python'
test=subprocess.run([python,'-B','-m','pytest','-q',str(ops/'test_planfix_library_runtime.py'),'-p','no:cacheprovider'],
 env=env,cwd=ops,capture_output=True,text=True,timeout=20)
print('LINUX_REUSED_EXECUTOR_TESTS',test.returncode,test.stdout,test.stderr,flush=True)
assert test.returncode==0
check=subprocess.run([python,'-B','-c','from planfix_bank_v4 import inactive_training; from pathlib import Path; assert inactive_training(Path('+repr(str(r))+')); print("OWNER_STOPPED_PACE_GUARD_PASS")'],
 env=env,cwd=ops,capture_output=True,text=True,timeout=10)
print(check.stdout,check.stderr,flush=True);assert check.returncode==0
assert '1.27.1' in subprocess.check_output([env['SUMO_HOME']+'/bin/sumo','--version'],env=env,text=True)
with (r/'AUTHORITY.md').open('a') as out:
 out.write('\nPLANFIX v4 prospective owner-stopped remainder continuation: v3 exited on pace comparison spanning B safety/owner pause BEFORE reservations or physics. Rechecked original v3 owner exited, phase2400, zero v3 RESULTs, one untouched STARTED. All v2 failed input2400 charges and v3 artifacts retained; failed986052 never replayed. New bank_verification_v4 namespace for same seven untouched inputs. While exact MAIN owner restart STOP is present, all original A/B/R/monitor identities exited and clean2 root absent, no active-training pace exists; own eight nice19 one-thread workers may proceed. If training/clean2 changes, stop own next wave for a new pace registration. No training control/bank/config changes, unchanged u0/full1200s/native all8 criterion, sole ledger/cap600000. Local REGISTRATION.md appended before this dispatch. Source '+json.dumps(manifest,sort_keys=True)+'\n')
 out.flush();os.fsync(out.fileno())
with (w/'BANK_JOB_v4.log').open('x') as log:
 proc=subprocess.Popen(['nice','-n','19','ionice','-c','3',python,'-B','-u',str(target),'--execute','--workers','8'],env=env,cwd=ops,
  stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=proc.pid,start_ticks=Path('/proc/%s/stat'%proc.pid).read_text().split()[21],
 boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),source_manifest=manifest,
 workers=8,phase='plan_fix',ledger=str(r/'CHARGES.jsonl'),previous_physics_sim_s=2400)
with (w/'BANK_DISPATCH_v4.json').open('x') as out:json.dump(receipt,out,sort_keys=True)
print('BANK_DISPATCH_V4',json.dumps(receipt),flush=True)
'''.replace('__WRAPPER__',repr(wrapper))
remote('cpuV',code,timeout=35)
