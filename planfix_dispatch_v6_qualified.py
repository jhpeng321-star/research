"""Finish the qualified receipt-adoption step after lost transport, never replay."""
from planfix_ops import remote
code=r'''import hashlib,json,os,subprocess,time
from pathlib import Path
r=Path('/home/ubuntu/coopmwm20261105_clean1');w=r/'plan_fix';ops=w/'ops_v6_receipt_adoption'
assert not (w/'BANK_DISPATCH_v6.json').exists() and not (w/'BANK_JOB_v6.log').exists()
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:cmd=(proc/'cmdline').read_bytes()
 except (FileNotFoundError,PermissionError):continue
 assert b'ADOPTION_NO_PHYSICS_PASS' not in cmd,'prior owned adoption command still present; inspect, never duplicate'
prior=json.loads((w/'BANK_DISPATCH_v5.json').read_text());p=Path('/proc')/str(prior['pid'])/'stat'
if p.exists():
 f=p.read_text().split();assert f[21]!=prior['start_ticks'] or f[2]=='Z'
charges=[json.loads(line) for line in (r/'CHARGES.jsonl').read_text().splitlines()]
assert sum(row['sim_s'] for row in charges if row['phase']=='plan_fix')==9600
line=[line for line in (r/'AUTHORITY.md').read_text().splitlines() if line.startswith('PLANFIX v6 prospective receipt-only adoption:')][-1]
manifest=json.loads(line.split(' Source ',1)[1])
for name,sha in manifest.items():assert hashlib.sha256((ops/name).read_bytes()).hexdigest()==sha
proof=json.loads((w/'ADOPTION_V6.json').read_text())
for name,sha in proof['files'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha
result=json.loads((w/'bank_verification_v5/bank_606ea498f1c6082c/PLAN_RESULT.json').read_text())
assert result['charged_sim_s']==4800 and all(c['adopted_without_replay'] for c in result['calls'])
assert all(row['keep'] is False for row in result['rows'])
source='/home/ubuntu/coopmwm20261105_clean1_src_7cc6edaa'
env=dict(os.environ,PYTHONPATH=str(ops)+':'+source+'/pytools:/home/ubuntu/e3final/runtime:/home/ubuntu/e3final/runtime/src:'+source+'/runtime_src:'+source+'/WM_PACKAGE',
 CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LD_PRELOAD='/home/ubuntu/cpus_env/libmkl_intel_shim.so',PYTHONDONTWRITEBYTECODE='1')
env['SUMO_HOME']='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo';env['PATH']=env['SUMO_HOME']+'/bin:/home/ubuntu/miniconda3/bin:'+env['PATH']
python='/home/ubuntu/miniconda3/bin/python'
check=subprocess.run([python,'-B','-c','from planfix_bank_v4 import inactive_training; from planfix_run_bank import ROOT; assert inactive_training(ROOT); print("OWNER_STOP_GUARD_PASS")'],env=env,cwd=ops,capture_output=True,text=True,timeout=10)
print(check.stdout,check.stderr,flush=True);assert check.returncode==0
with (r/'AUTHORITY.md').open('a') as out:
 out.write('\nPLANFIX v6 qualified dispatch after interrupted transport: CPU11tests passed2.21s before stream loss. Independently confirmed cached-only native all8 publication, all adopted original measurement SHA unchanged and phase9600/no new reservations. PLAN_RESULT confirmation '+result['confirmation_sha256']+'; both native CRNs translated gain0/lower0, correctly rejected. No prior v6 log/dispatch or old v5 owner; all exact source hashes match prior authority. Continue only remaining five untouched inputs, reusing this completed result without replay.\n');out.flush();os.fsync(out.fileno())
with (w/'BANK_JOB_v6.log').open('x') as log:
 proc=subprocess.Popen(['nice','-n','19','ionice','-c','3',python,'-B','-u',str(ops/'planfix_bank_v6.py'),'--execute','--workers','8'],env=env,cwd=ops,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=proc.pid,start_ticks=Path('/proc/%s/stat'%proc.pid).read_text().split()[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),source_manifest=manifest,workers=8,phase='plan_fix',ledger=str(r/'CHARGES.jsonl'))
with (w/'BANK_DISPATCH_v6.json').open('x') as out:json.dump(receipt,out,sort_keys=True)
print('BANK_DISPATCH_V6',json.dumps(receipt),flush=True)
'''
remote('cpuV',code,timeout=25)
