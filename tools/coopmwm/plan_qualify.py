"""Final source-bound full-population GPU batches; medium search is unchanged."""
import json
from pathlib import Path
import sys
import time
import torch
from .storage import read,write


def run(root):
    from tools.wmteach.runtime import bootstrap
    from .wm import cost_adapter
    from .plans import search,family_seeds,baseline_actions
    from .plan_wm import evaluate
    rt=bootstrap();torch.cuda.set_per_process_memory_fraction(.65)
    rt.frozen.costs=cost_adapter(rt.frozen.costs,read(root/'FROZEN_COSTS.json')['code'])
    source=read(root/'INPUTS.json')['records'][0]
    commands={r['vid']:0. for r in source['rows'] if r['is_cav']}
    record=dict(state_id=source['state_id'],world_id=source['world_id'],rows=source['rows'],
        group=source['group'],time_s=source['time_s'],commands=commands,
        policy_residuals={v:-.25 for v in commands})
    bounds=(float(rt.mp.policy.accel_min),float(rt.mp.policy.accel_max))
    seeds=family_seeds(record['rows'],record['group'],commands,bounds)
    acts=baseline_actions(record)+seeds[:2]
    parities=[]
    for K,H,width in ((4,120,34),(8,300,10)):
        start=time.monotonic();torch.cuda.reset_peak_memory_stats()
        ref=evaluate(rt,record,acts,K,H,chunk_actions=1,trim=False,pad_actions=width)
        batch=evaluate(rt,record,acts,K,H,chunk_actions=width,trim=False,pad_actions=width)
        error=max(abs(a['team_delay']-b['team_delay']) for key in ref for a,b in zip(ref[key],batch[key]))
        item=dict(K=K,H=H,width=width,max_cost_error=error,pass_=error<=1e-3,
            wall_s=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            contract='single candidate padded to fixed width vs heterogeneous batch; full1600/source/controller; nonzero nongroup residual')
        write(root/f'PARITY_K{K}_H{H}.json',item);print(json.dumps(item),flush=True)
        assert item['pass_'];parities.append(item)
    write(root/'PARITY.json',dict(pass_=True,parities=parities,bounds=bounds,new_sumo_sim_s=0))
    start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    result=search(record,bounds,lambda a,k,h:evaluate(rt,record,a,k,h,chunk_actions=34 if k==4 else 10,trim=False,pad_actions=34 if k==4 else 10))
    torch.cuda.synchronize();wall=time.monotonic()-start
    write(root/'SEARCH.json',result)
    receipt=dict(pass_=True,wall_s=wall,plans_per_hour=3600/wall,
        imagined_futures_per_hour=result['imagined_futures']*3600/wall,
        candidates_searched=result['candidates_searched'],iterations=result['iterations'],
        imagined_futures=result['imagined_futures'],peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        screen_width=34,final_width=10,new_sumo_sim_s=0)
    write(root/'COMPLETE.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    root=Path(sys.argv[1])
    try:run(root)
    except BaseException:
        import traceback
        write(root/'FAILED.json',dict(traceback=traceback.format_exc(),new_sumo_sim_s=0));raise
