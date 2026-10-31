"""Independently verify owned canary receipts before any fixed-panel dispatch."""
import json
from pathlib import Path

root=Path('/root/coopaudit20261031')
path=root/'jobs/canary_s969699_attempt2/result.json'
r=json.loads(path.read_text())
assert r['ok'] and r['mode']=='canary'
assert r['frozen']['world']['seed']==969699
h=r['trajectory_sha256']
assert h['native_LIVE']==h['native_B']==h['native_B_repeat']
for name in ('wall','mappo'):
    assert h[name+'_GROUP']==h[name+'_GROUP_repeat']
    assert r['metrics'][name+'_GROUP']==r['metrics'][name+'_GROUP_repeat']
assert r['metrics']['native_LIVE']==r['metrics']['native_B']==r['metrics']['native_B_repeat']
assert r['effects']['wall_GROUP']['wall_issued_commands']>0
ledger=[json.loads(s) for s in (root/'accounting.jsonl').read_text().splitlines()]
charged=sum(x['sim_s'] for x in ledger)
assert charged<=500000 and r['actual_sim_s']<=charged<=20000
summary=dict(ok=True,actual_sim_s=r['actual_sim_s'],charged_sim_s=charged,
             group=r['frozen']['group'],window_s=r['window_s'],
             hashes=h,wall_issued_commands=r['effects']['wall_GROUP']['wall_issued_commands'],
             wall_events=r['effects']['wall_GROUP']['wall_events'],
             traces=str(path.parent/'traces'),native_live_load_bitwise=True,
             controlled_repeats_bitwise=True)
summary['credits']={}
for c in ('wall','mappo'):
    for outcome in ('180','local'):
        def cost(key):
            return (r['effects'][key]['local_cohort_delay_vs'] if outcome=='local' else
                    r['metrics'][key][outcome]['team_delay_vs'])
        b=cost('native_B')
        dg=b-cost(c+'_GROUP')
        solo=sum(b-cost(c+'_ONLY:'+str(i)) for i in range(len(r['frozen']['group'])))
        summary['credits'][c+'_'+outcome]=dict(D_G=dg,sum_D_i=solo,C_G=dg-solo)
summary['safety_totals']={k:sum(m['180'].get(k,0) for m in r['metrics'].values()) for k in
                         ('collisions','teleports','dropped_vehicle_steps')}
from hashlib import sha256
summary['result_sha256']=sha256(path.read_bytes()).hexdigest()
(root/'CANARY_VERIFIED.json').write_text(json.dumps(summary,indent=1)+'\n')
print(json.dumps(summary,indent=1))
