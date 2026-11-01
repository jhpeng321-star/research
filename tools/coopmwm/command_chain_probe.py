"""Replay the actual WM CAV callback on the two named observable TRAIN tapes.

No WM fitting, future sampling, SUMO launch or truth access. Other vehicles use
their recorded observable rows; the two half-step positions are linearly
interpolated where no half-second tape exists. Report errors explicitly rather
than calling this a bitwise full-road WM/SUMO parity test.
"""
import json
import os
from pathlib import Path
import sys
import time


def run(fixture_path):
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',C4_SUMO_LEADER_RULE='1')
    phys='/root/rbeval20261031/phys_code_22cc32f65a80'
    sys.path[:0]=[phys+'/src',phys]
    import torch
    torch.set_num_threads(1)
    from tools.wmteach import runtime
    from wms.control.mappo.eval_bridge import load_policy_fn
    from wms.model.cav import WM_LANE_TARGET_KEY
    from .plan_wm import plan_executor
    from .policy_wm import load_actor
    from .mppi_search import comparators
    binding=json.loads((runtime.BASE/'speedhead20261030/EXPORT_BOUND.json').read_text())
    mp=load_policy_fn(binding['path'],device='cpu')
    cases=json.loads(Path(fixture_path).read_text())['cases'];output=[]
    begin=time.monotonic()
    for case in cases:
        record=case['record'];ids=[r['vid'] for r in record['rows']]
        group=case['plan']['members'];slots=[ids.index(v) for v in group]
        acts=comparators(record,600)+[case['plan']]
        frames={int(f['time_s']):{r['vid']:r for r in f['vehicles']} for f in case['frames'][0]}
        policy=load_actor(record['policy_state'],'cpu') if record.get('policy_state') is not None else None
        for final_chain in (False,True):
            initial=record['rows'];bs=len(acts)
            def tensor(name,dtype=torch.float32):
                return torch.tensor([[r[name] for r in initial]]*bs,dtype=dtype)
            state=dict(s_m=tensor('s_m'),v_mps=tensor('v_mps'),lane=tensor('lane',torch.long),
                present=torch.ones(bs,len(ids),dtype=torch.bool),is_cav=tensor('is_cav',torch.bool),
                length_m=torch.tensor([[r.get('length_m',5.) for r in initial]]*bs),a_mps2=torch.zeros(bs,len(ids)))
            trace=[];errors=[];step_differences=0
            with plan_executor(6,policy,final_chain=final_chain):
                cb=runtime.option_controller(mp,acts,initial)
                for tick in range(12):
                    t=300+tick//2
                    # Real observed background; retain the group callback's own
                    # propagated state. Departures use the observed roster.
                    current=frames[t];nxt=frames.get(t+1,{})
                    for slot,vid in enumerate(ids):
                        present=vid in current
                        state['present'][:,slot]=present
                        if not present:continue
                        row=current[vid]
                        if tick%2==0 or vid not in group:
                            f=.5 if tick%2 and vid in nxt else 0.
                            for k in ('s_m','v_mps'):
                                state[k][:,slot]=row[k]+f*(nxt.get(vid,row)[k]-row[k])
                            state['lane'][:,slot]=row['lane']
                    state[WM_LANE_TARGET_KEY]=state['lane'].clone()
                    state['physics_step_index']=tick
                    result=cb(state)
                    active=state['present'][0,slots]
                    step_differences+=int(not torch.equal(result[0,slots][active],result[2,slots][active]))
                    trace.append(dict(tick=tick,policy=result[0,slots].tolist(),plan=result[2,slots].tolist(),present=active.tolist()))
                    if tick%2:
                        for j,vid in enumerate(group):
                            if vid not in nxt:continue
                            for coordinate,k in ((0,'s_m'),(2,'v_mps')):
                                errors.append(dict(time_s=t+1,vid=vid,field=k,
                                    error=float(result[2,slots[j],coordinate])-nxt[vid][k]))
                    state['s_m']=result[...,0];state['v_mps']=result[...,2];state['a_mps2']=result[...,3]
            output.append(dict(world=case['world'],final_chain=final_chain,trace=trace,errors=errors,
                nonidentical_plan_policy_steps=step_differences,
                max_position_error=max([abs(e['error']) for e in errors if e['field']=='s_m'] or [0.]),
                max_speed_error=max([abs(e['error']) for e in errors if e['field']=='v_mps'] or [0.])))
    return dict(cases=output,wall_s=time.monotonic()-begin,new_sumo_sim_s=0,
        contract='actual WM callback; observed backgrounds, interpolated half-step neighbours; numeric errors reported, not full-road exact parity')


if __name__=='__main__':
    print(json.dumps(run(sys.argv[1])))
