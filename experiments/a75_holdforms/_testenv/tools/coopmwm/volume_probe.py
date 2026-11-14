"""Bounded infrastructure qualification on already-observed TRAIN geometry."""
import json
from pathlib import Path
import sys
import time
from .storage import read,write


def run(root):
    import torch
    from tools.wmteach.runtime import bootstrap
    from .wm import cost_adapter,adapted_futures,proposal_candidates,executor_adapter
    from .fused import futures
    rt=bootstrap()
    torch.cuda.set_per_process_memory_fraction(.65)
    rt.frozen.costs=cost_adapter(rt.frozen.costs,read(root/'FROZEN_COSTS.json')['code'])
    records=read(root/'INPUTS.json')['records']
    results=[]
    for count,K in ((2,8),(4,4),(4,8)):
        torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        values=futures(rt,records[:count],tuple(range(K)),trim_absent=True)
        torch.cuda.synchronize();wall=time.monotonic()-start
        write(root/f'FUSED_{count}_K{K}.json',values)
        assert all(v['population_audit']['finite'] and v['population_audit']['conserved'] for v in values)
        parity=[]
        if count==2 and K==8:
            previous=read(root/'CAPACITY_REFERENCE.json')
            capacity_error=max(abs(x['team_delay']-y['team_delay']) for a,b in zip(values,previous)
                for name,fs in a['futures']['300'].items() for x,y in zip(fs,b['futures']['300'][name]))
            assert capacity_error<=1e-3,capacity_error
            parity.append(capacity_error)
            # Once per worker, full K8 and indexed single, actual heterogeneous windows.
            for r,v in zip(records[:count],values):
                ns=adapted_futures((300,));ns['_actions']=proposal_candidates(r)
                with executor_adapter():single=ns['futures'](rt,dict(r),indices=(0,))
                err=max(abs(fs[0]['team_delay']-single['futures']['300'][a][0]['team_delay']) for a,fs in v['futures']['300'].items())
                parity.append(err)
            assert max(parity)<=1e-3,parity
        if count==4:
            previous=read(root/'FUSED_2_K8.json')
            err=max(abs(x['team_delay']-y['team_delay']) for a,b in zip(values,previous)
                    for name,fs in a['futures']['300'].items() for x,y in zip(fs,b['futures']['300'][name]))
            assert err<=1e-3,err
            parity.append(err)
        item=dict(windows=count,K=K,wall_s=wall,queries_per_hour=count/wall*3600,
            max_cost_error=max(parity),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),new_sumo_sim_s=0)
        results.append(item);write(root/f'TIMING_{count}_K{K}.json',item);print(json.dumps(item),flush=True)
    def winner(v):
        fs=v['futures']['300'];means={a:sum(x['team_delay'] for x in f)/len(f) for a,f in fs.items()}
        best=min((a for a in means if a not in ('policy','MAPPO')),key=lambda a:(means[a],a))
        return best,min(means['policy'],means['MAPPO'])>means[best]
    k8=read(root/'FUSED_4_K8.json');k4=read(root/'FUSED_4_K4.json')
    error=max(abs(x['team_delay']-y['team_delay']) for a,b in zip(k8,k4)
                for name,fs in a['futures']['300'].items() for x,y in zip(fs,b['futures']['300'][name]))
    assert error<=1e-3,error
    agreement=[winner(a)==winner(b) for a,b in zip(k8,k4)]
    write(root/'COMPLETE.json',dict(pass_=True,timings=results,K8_K4_choice_acceptance_agreement=agreement,
        qualification_windows=[r['state_id'] for r in records[:4]],new_sumo_sim_s=0))


if __name__=='__main__':
    root=Path(sys.argv[1])
    try:run(root)
    except BaseException:
        import traceback
        write(root/'FAILED.json',dict(traceback=traceback.format_exc(),new_sumo_sim_s=0));raise
