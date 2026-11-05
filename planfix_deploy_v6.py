"""Adopt measured v5 receipts without physics, fix all8 charge schema, continue."""
from pathlib import Path
from planfix_ops import remote

code=r'''import hashlib,json,os,subprocess,time
from pathlib import Path
r=Path('/home/ubuntu/coopmwm20261105_clean1');w=r/'plan_fix';old=w/'ops_v5_qualified_runtime';ops=w/'ops_v6_receipt_adoption'
assert not (w/'BANK_DISPATCH_v6.json').exists()
prior=json.loads((w/'BANK_DISPATCH_v5.json').read_text());p=Path('/proc')/str(prior['pid'])/'stat'
if p.exists():
 fields=p.read_text().split();assert fields[21]!=prior['start_ticks'] or fields[2]=='Z'
assert "KeyError: 'actual_sim_s'" in (w/'BANK_JOB_v5.log').read_text()
rows=[json.loads(line) for line in (r/'CHARGES.jsonl').read_text().splitlines()]
assert sum(row['sim_s'] for row in rows if row['phase']=='plan_fix')==9600
starts=list((w/'bank_verification_v5').glob('*/STARTED.json'));assert len(starts)==1
base=starts[0].parent;receipts=list(base.glob('replay_bank/B/*/verifications/*.json'))
results=list(base.glob('replay_bank/B/*/branches/*/*/RESULT.json'))
assert len(receipts)==2 and len(results)==4
files=[starts[0],base/'FROZEN_SNAPSHOT.pt',*receipts,*results]
proof=dict(input_sha256=json.loads(starts[0].read_text())['input_sha256'],
 files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},charged_sim_s=4800,
 phase_total_sim_s=9600,reason='actual_sim_s omitted in operator confirmation charge rows; adopt native measurements, no replay')
path=w/'ADOPTION_V6.json'
if path.exists():assert json.loads(path.read_text())==proof
else:
 with path.open('x') as out:json.dump(proof,out,sort_keys=True)
ops.mkdir(exist_ok=True);manifest={}
for name,sha in prior['source_manifest'].items():
 data=(old/name).read_bytes();assert hashlib.sha256(data).hexdigest()==sha
 if name=='planfix_run_bank.py':
  text=data.decode()
  before="    if (result_root/'STARTED.json').exists():\n        raise ValueError('incomplete attempt retained; inspect before continuation, never replay')"
  after="    existing_start=(result_root/'STARTED.json').exists()\n    if existing_start:\n        proof=lt.read(work/'ADOPTION_V6.json')\n        assert proof['input_sha256']==digest(item)\n        for path,sha in proof['files'].items():\n            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha"
  assert text.count(before)==1;text=text.replace(before,after)
  text=text.replace("    lt.write(result_root/'STARTED.json',dict(","    if not existing_start:lt.write(result_root/'STARTED.json',dict(")
  text=text.replace("    torch.save(snap,result_root/'FROZEN_SNAPSHOT.pt')","    if existing_start:\n        assert sv.snapshot_sha(torch.load(result_root/'FROZEN_SNAPSHOT.pt',weights_only=False,map_location='cpu'))==sv.snapshot_sha(snap)\n    else:torch.save(snap,result_root/'FROZEN_SNAPSHOT.pt')")
  text=text.replace("charged_sim_s=receipt['charged_sim_s'],status=receipt['status']))","charged_sim_s=receipt['charged_sim_s'],actual_sim_s=receipt.get('actual_sim_s'),status=receipt['status']))")
  before="        return verify(snapshot,candidates,Runner(),charge,proposer,workers=1,\n                      root=directory,tail_s=max(tail_s,required_tail))"
  after="""        receipt=verify(snapshot,candidates,Runner(),charge,proposer,workers=1,
                      root=directory,tail_s=max(tail_s,required_tail))
        if not any(c['directory']==str(directory) for c in calls):
            specs=build(snapshot,candidates,proposer,max(tail_s,required_tail))
            assert receipt['window']==specs[0]['window']
            assert receipt['branch_specs_sha256']==[s['branch_sha256'] for s in specs]
            jobs=[dict(id='planfix_'+digest(dict(directory=str(directory),snapshot=s['snapshot_sha256'],window=s['window']))[:20]+'_'+s['verification_id']+'_'+s['branch'],sim_s=s['sim_s']) for s in specs]
            ledger={row['id']:row for row in lt.read_lines(ROOT/'CHARGES.jsonl')} if hasattr(lt,'read_lines') else {row['id']:row for row in [json.loads(line) for line in (ROOT/'CHARGES.jsonl').read_text().splitlines()]}
            for job in jobs:
                assert ledger[job['id']]['phase']=='plan_fix' and ledger[job['id']]['sim_s']==job['sim_s']
            calls.append(dict(directory=str(directory),charge_ids=[j['id'] for j in jobs],charged_sim_s=sum(j['sim_s'] for j in jobs),adopted_without_replay=True))
        return receipt"""
  assert text.count(before)==1;text=text.replace(before,after)
  data=text.encode()
 target=ops/name
 if target.exists():assert target.read_bytes()==data
 else:
  with target.open('xb') as out:out.write(data)
 manifest[name]=hashlib.sha256(data).hexdigest()
wrapper="import planfix_run_bank as bank\nfrom planfix_bank_v4 import pace_while_owner_stopped\nfrom planfix_bank_v5 import remaining_only\nif __name__=='__main__':\n bank.ARTIFACTS='bank_verification_v5'\n bank.check_pace=pace_while_owner_stopped\n bank.verify_plan=remaining_only\n bank.main()\n"
target=ops/'planfix_bank_v6.py'
if target.exists():assert target.read_text()==wrapper
else:
 with target.open('x') as out:out.write(wrapper)
manifest[target.name]=hashlib.sha256(target.read_bytes()).hexdigest()
source='/home/ubuntu/coopmwm20261105_clean1_src_7cc6edaa'
env=dict(os.environ,PYTHONPATH=str(ops)+':'+source+'/pytools:/home/ubuntu/e3final/runtime:/home/ubuntu/e3final/runtime/src:'+source+'/runtime_src:'+source+'/WM_PACKAGE',
 CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LD_PRELOAD='/home/ubuntu/cpus_env/libmkl_intel_shim.so',PYTHONDONTWRITEBYTECODE='1')
env['SUMO_HOME']='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo';env['PATH']=env['SUMO_HOME']+'/bin:/home/ubuntu/miniconda3/bin:'+env['PATH']
python='/home/ubuntu/miniconda3/bin/python'
test=subprocess.run([python,'-B','-m','pytest','-q',str(ops/'test_planfix_library_runtime.py'),'-p','no:cacheprovider'],env=env,cwd=ops,capture_output=True,text=True,timeout=20)
print('V6_LINUX_TESTS',test.stdout,test.stderr,flush=True);assert test.returncode==0
with (r/'AUTHORITY.md').open('a') as out:
 out.write('\nPLANFIX v6 prospective receipt-only adoption: original v5 PID exited on operator confirmation KeyError actual_sim_s AFTER four full native branches/4800sim-s. Phase9600 retained. Bind ADOPTION_V6.json '+hashlib.sha256(path.read_bytes()).hexdigest()+' over both exact native verification receipts, four RESULTs, snapshot and STARTED. Fix only missing actual_sim_s in all8 charge metadata. Original native verify cache verifies identities; explicit cached branch/window/SHA/parent-ledger IDs checked. Existing snapshots/tapes/receipts/charges are never rewritten or replayed. First perform a zero-reservation/no-Popen cached all8 publication test; only after PASS one coordinator may adopt that PLAN_RESULT and process five untouched inputs. Same failed-input exclusions, original artifact namespace, physical-runtime/u0/full1200s/all8/safety/solo, own-stop/owner-stopped guard and phase600000/global5M caps. Local REGISTRATION.md prospectively registered. Source '+json.dumps(manifest,sort_keys=True)+'\n');out.flush();os.fsync(out.fileno())
dry="""import json
from pathlib import Path
import planfix_run_bank as bank
from planfix_bank_v4 import inactive_training
assert inactive_training(bank.ROOT)
def refuse(*args,**kwargs):raise RuntimeError('ADOPTION MUST NOT RESERVE OR RUN PHYSICS')
bank.reserve_planfix=refuse;bank.ProcessPoolExecutor=refuse;bank.ARTIFACTS='bank_verification_v5'
w=bank.ROOT/'plan_fix';proof=bank.lt.read(w/'ADOPTION_V6.json')
cfg=bank.lt.checked_config(bank.ROOT/'CONFIG.json');native=bank.prod.backend_config(cfg)
item=next(row for row in bank.lt.read(w/'BANK_INVENTORY.json')['rows'] if bank.digest(row)==proof['input_sha256'])
result=bank.verify_plan(item,cfg,native,bank.lt.read(w/'REGISTRATION.json'),8)
for p,sha in proof['files'].items():assert bank.hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha
assert result['charged_sim_s']==4800
print('ADOPTION_NO_PHYSICS_PASS',json.dumps(dict(charged_sim_s=result['charged_sim_s'],rows=[{k:v for k,v in row.items() if k!='example'} for row in result['rows']])),flush=True)
"""
test=subprocess.run(['nice','-n','19',python,'-B','-c',dry],env=env,cwd=ops,capture_output=True,text=True,timeout=25)
print(test.stdout,test.stderr,flush=True);assert test.returncode==0
assert sum(json.loads(line)['sim_s'] for line in (r/'CHARGES.jsonl').read_text().splitlines() if json.loads(line)['phase']=='plan_fix')==9600
with (w/'BANK_JOB_v6.log').open('x') as log:
 proc=subprocess.Popen(['nice','-n','19','ionice','-c','3',python,'-B','-u',str(target),'--execute','--workers','8'],env=env,cwd=ops,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=proc.pid,start_ticks=Path('/proc/%s/stat'%proc.pid).read_text().split()[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),source_manifest=manifest,workers=8,phase='plan_fix',ledger=str(r/'CHARGES.jsonl'))
with (w/'BANK_DISPATCH_v6.json').open('x') as out:json.dump(receipt,out,sort_keys=True)
print('BANK_DISPATCH_V6',json.dumps(receipt),flush=True)
'''
remote('cpuV',code,timeout=45)
