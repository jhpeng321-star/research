import hashlib
import json
from pathlib import Path
import time
h=Path('/home/ubuntu')
out=dict(unix=time.time(),ledgers={})
for root in sorted(h.iterdir()):
 if not root.name.startswith('coopmwm') or not any(k in root.name for k in ('clean1','clean2','clean3','s_mining')):continue
 if root.is_symlink() or not root.is_dir():continue
 p=root/'CHARGES.jsonl'
 if not p.is_file():continue
 rows=[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
 phases={}
 for row in rows:phases[row['phase']]=phases.get(row['phase'],0)+row['sim_s']
 out['ledgers'][root.name]=dict(rows=len(rows),sim_s=sum(x['sim_s'] for x in rows),phases=phases,
                              sha256=hashlib.sha256(p.read_bytes()).hexdigest())
out['historical_parity_sim_s']=960
assert out['ledgers']['coopmwm20261106_s_mining']['sim_s']==0
c1=out['ledgers']['coopmwm20261105_clean1']
c2=out['ledgers']['coopmwm20261105_clean2']
assert c1['phases'].get('plan_fix',0)<=600000
out['protected_plan_fix_remaining_sim_s']=600000-c1['phases'].get('plan_fix',0)
out['protected_clean2_cap_sim_s']=json.loads((h/'coopmwm20261105_clean2/LAUNCHED.json').read_bytes())['remaining_cap_sim_s']
assert c2['sim_s']<=out['protected_clean2_cap_sim_s']
out['mining_cap_sim_s']=500000
out['total_cap_sim_s']=6500000
out['bound_sim_s']=c1['sim_s']+960+out['protected_plan_fix_remaining_sim_s']+out['protected_clean2_cap_sim_s']+500000
extra=[name for name in out['ledgers'] if name not in ('coopmwm20261105_clean1','coopmwm20261105_clean2','coopmwm20261106_s_mining')]
out['other_ledgers']=extra
assert not extra,'new experiment ledger requires explicit reconciliation'
assert out['bound_sim_s']<=6500000
out['note']='Protect complete clean2 cap and remaining plan_fix. No clean3 ledger/allocation exists; prior full clean3 proposal was BLOCKED_BUDGET, not a reservation. Mining approved separately. No reservation cancelled.'
print(json.dumps(out,sort_keys=True))
