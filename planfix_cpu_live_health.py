"""Read-only named CPU worker/training receipt, pace and sole-ledger audit."""
import json
from pathlib import Path
from planfix_ops import remote,note

CODE=r'''
import hashlib,json,time
from pathlib import Path
r=Path('/home/ubuntu/coopmwm20261105_clean1');c=Path('/home/ubuntu/coopmwm20261105_clean2');w=r/'plan_fix/cpu_wm_v6'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
baseline=read(w/'CLEAN2_BASELINE.json');launch=read(c/'LAUNCHED.json')
assert sha(c/'LAUNCHED.json')==baseline['launch_file_sha256'] and sha(c/'CONFIG.json')==launch['config_file_sha256']
assert read(c/'SMOKE_PASS.json')['receipt_sha256']==launch['smoke_sha256']
assert read(c/'GATE_PASS.json')['receipt_sha256']==launch['gate_sha256']
owners={}
for role,d in launch['owners'].items():
 p=Path('/proc',str(d['pid']),'stat');f=p.read_text().split() if p.exists() else None
 owners[role]=dict(d,alive_same_identity=bool(f and f[21]==d['start_ticks'] and f[2]!='Z'))
pace={}
for role,b in baseline['pace'].items():
 values=[read(p) for p in sorted(c.glob(role+'_UPDATE_*.json'))[-4:]]
 latest=values[-1];previous=values[-2];dt=latest['finished_unix']-previous['finished_unix']
 pace[role]=dict(last_update=latest['update'],last_interval_s=dt,baseline_max_s=b['ceiling_s'],
  new_interval_over_baseline=latest['update']>b['last_update'] and dt>b['ceiling_s'])
oldrows=[json.loads(x) for x in (r/'CHARGES.jsonl').read_text().splitlines()]
newrows=[json.loads(x) for x in (c/'CHARGES.jsonl').read_text().splitlines()]
spent=sum(x['sim_s'] for x in oldrows if x['phase']=='plan_fix')
other=sum(x['sim_s'] for x in oldrows if x['phase']!='plan_fix')+960
actual=sum(x['sim_s'] for x in oldrows)+sum(x['sim_s'] for x in newrows)+960
assert spent<=600000 and actual<=5000000 and other+600000+launch['remaining_cap_sim_s']<=5000000
assert sha(w/'GPU_BINDING.json')=='452e7ae008aa9a92bdc68a2f4ecda4251a0ad0c4b6946fddbc549c9ab0eb8621'
out=dict(unix=time.time(),owners=owners,clean2_stop=(c/'STOP.json').exists(),clean2_fault=(c/'FAULT.json').exists(),
 pace=pace,own_next_evaluation_will_hold=any(p['new_interval_over_baseline'] for p in pace.values()),
 phase_spent_sim_s=spent,phase_reserved_remaining_sim_s=600000-spent,global_charged_sim_s=actual,
 global_allocation_bound_sim_s=other+600000+launch['remaining_cap_sim_s'],
 cpu_worker=read(w/'CPU_OWNER.json'),worker_failure=(w/'CPU_FAILED.json').exists(),
 completed_gates=sum(read(p).get('pass_') is True for p in w.glob('STARTUP_*.json')),
 speed_completed=(w/'SPEED_SEARCH_RESULT.json').exists(),new_sim_s_by_audit=0,
 note='Inter-update intervals are telemetry, not a causal attribution to this one-core worker; baseline is not changed.')
print('CPU_LIVE_HEALTH',json.dumps(out),flush=True)
'''

if __name__=='__main__':
    import time
    text=remote('cpuV',CODE,timeout=25)
    value=json.loads(next(x[len('CPU_LIVE_HEALTH '):] for x in text.splitlines() if x.startswith('CPU_LIVE_HEALTH ')))
    path=Path(__file__).with_name(f'PLANFIX_CPU_LIVE_HEALTH_{time.time_ns()}.json')
    with path.open('x') as out:json.dump(value,out,indent=2,sort_keys=True)
    note('cpuV',f"live health: clean2 seals checked, latest updates={ {k:v['last_update'] for k,v in value['pace'].items()} }, next-eval pace hold={value['own_next_evaluation_will_hold']}, phase={value['phase_spent_sim_s']}/600000 global_charged={value['global_charged_sim_s']}/5000000; no training control change")
