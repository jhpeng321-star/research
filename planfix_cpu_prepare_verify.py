"""Prepare a SHA-bound native-library verification slate from the ONE WM search.

Data preparation only: no simulator, reservation, process pool or bank switch.
An incomplete gate/search produces PENDING and does not publish a slate.
"""
import json
from pathlib import Path
from planfix_ops import remote,note

CODE=r'''
import json,os,subprocess
from pathlib import Path
worker=Path('/home/ubuntu/coopmwm20261105_clean1/plan_fix/cpu_wm_v6')
if not (worker/'SPEED_SEARCH_RESULT.json').exists():
 print('VERIFY_PREPARE_PENDING',json.dumps(dict(reason='original8gates and measured WM speed search not complete',worker=str(worker),new_sim_s=0)),flush=True)
else:
 code="""
import hashlib,json,sys
from pathlib import Path
from planfix_library_runtime import library_plan,locally_executable
from tools.coopmwm.moves import Move
from tools.coopmwm.protocol import digest
from tools.coopmwm.bank_v2v import trace_frames
worker=Path('/home/ubuntu/coopmwm20261105_clean1/plan_fix/cpu_wm_v6');w=worker.parent;root=w.parent
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
q=read(worker/'FRESH_QUALIFICATION.json');speed=read(worker/'SPEED_SEARCH_RESULT.json');query=read(worker/'SPEED_QUERY.json')
assert q['pass_'] and len(q['receipts'])==8 and q['pid']==speed['owner']['pid']
assert all(g['pid']==q['pid'] and g['pass_'] and g['max_cost_error']==0 for g in q['receipts'].values())
assert speed['all_eight_gates_same_pid'] and speed['new_sim_s']==0 and speed['sumo_verified_plans']==0
assert speed['source_binding_sha256']==sha(worker/'GPU_BINDING.json') and speed['record_sha256']==digest(query['record'])
assert query['record']['world_id']==986026 and 986000<=query['record']['world_id']<=986071
result=speed['result'];plans=result['collection_top8'];assert len(plans)==8 and result['available_k']==8
old=read(w/'BANK_INVENTORY.json');parent=next(x for x in old['rows'] if x['world_seed']==986026)
assert sha(parent['snapshot_path'])==parent['snapshot_sha256']
assert query['snapshot_sha256']==sha(query['snapshot_path'])
frame=trace_frames(parent['trace'])[int(parent['t0'])]
prior=read(w/'BANK_DISPATCH_v6.json');ops=w/'ops_v6_receipt_adoption'
for name,h in prior['source_manifest'].items():assert sha(ops/name)==h
seen={};rows=[];skipped=[]
for previous in old['rows']:
 if previous['snapshot_sha256']!=parent['snapshot_sha256']:continue
 translated=library_plan(previous['source_plan'],'prior_form_only')
 signature=digest(dict(snapshot_sha256=previous['snapshot_sha256'],move=translated['library_move'],duration_s=translated['duration_s']))
 seen[signature]='prior bank pass '+previous['id']+' (not an admission for this WM proposal)'
for index,p in enumerate(plans):
 ident='wm_cpu_'+digest(dict(query=sha(worker/'SPEED_QUERY.json'),plan=p))[:16]
 translated=library_plan(p,'planfix_'+ident);move=Move.from_dict(translated['library_move'])
 signature=digest(dict(snapshot_sha256=parent['snapshot_sha256'],move=move.to_dict(),duration_s=translated['duration_s']))
 if move.family=='none':
  skipped.append(dict(id=ident,wm_candidate=p['id'],family=move.family,reason='no-op library translation; unverified, not admitted'));continue
 if not locally_executable(move,frame):
  skipped.append(dict(id=ident,wm_candidate=p['id'],family=move.family,reason='not acquirable in a production local slate; unverified, not admitted'));continue
 if signature in seen:
  skipped.append(dict(id=ident,wm_candidate=p['id'],family=move.family,reason='duplicate exact translated execution form',first=seen[signature]));continue
 seen[signature]=ident
 item=dict(parent,id=ident,example_id=ident,source_example=dict(example_id=ident,seed_index=1),
  source_plan=p,source_receipt={},move=move.to_dict(),family=move.family,
  original_gain_veh_s=None,original_cooperative_gain_veh_s=None,source_seed_index=1,
  wm_provenance=dict(query_sha256=sha(worker/'SPEED_QUERY.json'),speed_sha256=sha(worker/'SPEED_SEARCH_RESULT.json'),
   qualification_sha256=sha(worker/'FRESH_QUALIFICATION.json'),candidate_id=p['id'],candidate_rank=index,
   saving_vs_mappo=p.get('saving_vs_mappo'),saving_vs_policy=p.get('saving_vs_policy'),
   proposal_family=p.get('variety_family'),translated_family=move.family),
  translated_execution=translated,translated_signature=signature,
  worst_case_sim_s=(10+2*len(move.members))*1200)
 rows.append(item)
ledger=[json.loads(x) for x in (root/'CHARGES.jsonl').read_text().splitlines()]
phase=sum(x['sim_s'] for x in ledger if x['phase']=='plan_fix')
other=sum(x['sim_s'] for x in ledger if x['phase']!='plan_fix')+960
launch=read('/home/ubuntu/coopmwm20261105_clean2/LAUNCHED.json')
assert other+600000+launch['remaining_cap_sim_s']<=5000000
quote=sum(x['worst_case_sim_s'] for x in rows)
assert phase+quote<=600000
manifest=dict(schema='planfix.cpu.library-verification-prepared.v1',rows=rows,skipped=skipped,
 worker=str(worker),wm_speed_wall_s=speed['wall_s'],source_qualification_sha256=sha(worker/'FRESH_QUALIFICATION.json'),
 worst_case_sim_s=quote,phase_spent_before=phase,phase_allocation_sim_s=600000,
 global_allocation_bound_sim_s=other+600000+launch['remaining_cap_sim_s'],
 new_sim_s=0,admitted_examples=0,execution_authorized=False,bank_switch_applied=False,
 required_before_execution='separate exact CPU native-verification/source/resource/accounting registration in AUTHORITY and central REGISTRATION')
destination=w/'cpu_variety_prepared';destination.mkdir(exist_ok=True);p=destination/'VERIFY_SLATE.json'
if p.exists():assert read(p)==manifest
else:
 with p.open('x') as out:json.dump(manifest,out,sort_keys=True)
print('VERIFY_PREPARED',json.dumps(dict(path=str(p),sha256=sha(p),candidates=len(rows),skipped=len(skipped),worst_case_sim_s=quote,
 by_family={f:sum(r['family']==f for r in rows) for f in ('ease_off','gap_open','wall','ramp_assist')},new_sim_s=0,execution_authorized=False)),flush=True)
"""
 source='/home/ubuntu/coopmwm20261105_clean1_src_7cc6edaa';ops=worker.parent/'ops_v6_receipt_adoption'
 env=dict(os.environ,PYTHONPATH=str(ops)+':'+source+'/pytools:/home/ubuntu/e3final/runtime:/home/ubuntu/e3final/runtime/src:'+source+'/runtime_src:'+source+'/WM_PACKAGE',
  CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LD_PRELOAD='/home/ubuntu/cpus_env/libmkl_intel_shim.so',PYTHONDONTWRITEBYTECODE='1')
 subprocess.run(['nice','-n','19','/home/ubuntu/miniconda3/bin/python','-B','-c',code],env=env,check=True)
'''

if __name__=='__main__':
    text=remote('cpuV',CODE,timeout=30)
    if 'VERIFY_PREPARED ' in text:
        value=json.loads(next(x[len('VERIFY_PREPARED '):] for x in text.splitlines() if x.startswith('VERIFY_PREPARED ')))
        with Path(__file__).with_name('PLANFIX_CPU_VERIFY_PREPARED.json').open('x') as out:json.dump(value,out,indent=2,sort_keys=True)
        note('cpuV','WM shortlist translated and quoted for native library-form verification; preparation only, no SUMO or bank admission')
