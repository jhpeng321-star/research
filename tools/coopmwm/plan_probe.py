"""No-SUMO parity, sensitivity and full medium-search timing on registered geometry."""
import json
from pathlib import Path
import sys
import time
import torch
from .storage import read,write


def run(root):
    from tools.wmteach.runtime import bootstrap
    from .wm import cost_adapter
    from .plans import search,family_seeds
    from .plan_wm import evaluate
    rt=bootstrap();torch.cuda.set_per_process_memory_fraction(.65)
    rt.frozen.costs=cost_adapter(rt.frozen.costs,read(root/'FROZEN_COSTS.json')['code'])
    source=read(root/'INPUTS.json')['records'][0]
    commands={r['vid']:0. for r in source['rows'] if r['is_cav']}
    record=dict(state_id=source['state_id'],world_id=source['world_id'],rows=source['rows'],
        group=source['group'],time_s=source['time_s'],commands=commands,policy_residuals=commands)
    bounds=(float(rt.mp.policy.accel_min),float(rt.mp.policy.accel_max))
    seeds=family_seeds(record['rows'],record['group'],commands,bounds)
    assert seeds,'qualification must use available family geometry'
    acts=[dict(id='policy',window=60,residuals=commands),dict(id='MAPPO',window=0),*seeds[:2]]
    start=time.monotonic()
    reference=evaluate(rt,record,acts,8,300,chunk_actions=1,trim=False,pad_actions=True)
    write(root/'REFERENCE.json',reference)
    batched=evaluate(rt,record,acts,8,300,chunk_actions=4,trim=False,pad_actions=True)
    write(root/'BATCHED.json',batched)
    errors={a:max(abs(x['team_delay']-y['team_delay']) for x,y in zip(reference[a],batched[a])) for a in reference}
    sensitivity=max(abs(x['team_delay']-y['team_delay']) for a in acts[2:] for x,y in zip(batched[a['id']],batched['MAPPO']))
    parity=dict(pass_=max(errors.values())<=1e-3,max_cost_error=max(errors.values()),errors=errors,
        sensitivity_max_veh_s=sensitivity,bounds=bounds,wall_s=time.monotonic()-start,new_sumo_sim_s=0)
    write(root/'PARITY.json',parity);print(json.dumps(parity),flush=True)
    assert parity['pass_'] and sensitivity>0,'acceleration forecast parity or sensitivity failed'
    start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    result=search(record,bounds,lambda a,k,h:evaluate(rt,record,a,k,h,chunk_actions=4,trim=False,pad_actions=True))
    torch.cuda.synchronize();wall=time.monotonic()-start
    write(root/'SEARCH.json',result)
    timing=dict(pass_=True,wall_s=wall,plans_per_hour=3600/wall,
        imagined_futures_per_hour=result['imagined_futures']*3600/wall,
        candidates_searched=result['candidates_searched'],iterations=result['iterations'],
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),new_sumo_sim_s=0)
    write(root/'COMPLETE.json',timing);print(json.dumps(timing),flush=True)


if __name__=='__main__':
    root=Path(sys.argv[1])
    try:run(root)
    except BaseException:
        import traceback
        write(root/'FAILED.json',dict(traceback=traceback.format_exc(),new_sumo_sim_s=0));raise
