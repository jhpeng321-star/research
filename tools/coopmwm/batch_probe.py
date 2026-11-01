"""Registered infrastructure-only K8/H300 batching qualification; no SUMO."""
import json
from pathlib import Path
import sys
import time
from .storage import read,write


def run(root):
    import torch
    from tools.wmteach import runtime
    from .wm import adapted_futures,proposal_candidates,executor_adapter,cost_adapter
    rt=runtime.bootstrap()
    rt.frozen.costs=cost_adapter(rt.frozen.costs,read(root/'FROZEN_COSTS.json')['code'])
    record=read(root/'INPUT.json');old=read(root/'REFERENCE.json')
    ns=adapted_futures((300,));ns['_actions']=proposal_candidates(record)
    torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    with executor_adapter():
        result=ns['futures'](rt,dict(record),indices=tuple(range(8)))
    torch.cuda.synchronize();wall=time.monotonic()-start
    fs=result['futures']['300']
    assert result['population_audit']['finite'] and result['population_audit']['conserved']
    assert set(fs)==set(old['futures'])
    assert all(len(v)==8 for v in fs.values())
    assert all(x['draw_id']==y['draw_id'] for a in fs for x,y in zip(fs[a],old['futures'][a]))
    error=max(abs(x['team_delay']-y['team_delay']) for a in fs for x,y in zip(fs[a],old['futures'][a]))
    value=dict(pass_=error<=1e-3,max_team_delay_error=error,wall_s=wall,K=8,horizon_s=300,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        new_sumo_sim_s=0,result=result)
    write(root/'RESULT.json',value)
    print(json.dumps({k:v for k,v in value.items() if k!='result'}),flush=True)


if __name__=='__main__':run(Path(sys.argv[1]))
