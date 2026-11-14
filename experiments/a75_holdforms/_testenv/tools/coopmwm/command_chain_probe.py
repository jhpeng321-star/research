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


def observable_midpoint(current,following):
    """Invert two semi-implicit Euler half-steps using observable endpoints.

    s1-s0 = .5*(v_half+v1). This is a law-derived reconstruction, NOT a
    half-second sensor record. Entries/departures and lane changes are explicitly
    unresolved; callers must not turn this check into a full-road parity claim.
    """
    result={};unresolved={}
    for vid in sorted(current.keys()|following.keys()):
        if vid not in current or vid not in following:
            unresolved[vid]='roster_change';continue
        a,b=current[vid],following[vid]
        if a['lane']!=b['lane']:
            unresolved[vid]='lane_change';continue
        speed=2.*(b['s_m']-a['s_m'])-b['v_mps']
        if speed<0:
            unresolved[vid]='negative_reconstructed_speed';continue
        result[vid]=dict(a,s_m=a['s_m']+.5*speed,v_mps=speed)
    return result,unresolved


def run(fixture_path,*,kinematic=False,continuous=False,device='cpu'):
    os.environ.update(CUDA_VISIBLE_DEVICES='0' if device=='cuda' else '',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',C4_SUMO_LEADER_RULE='1')
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
    mp=load_policy_fn(binding['path'],device=device)
    cases=json.loads(Path(fixture_path).read_text())['cases'];output=[]
    begin=time.monotonic()
    for case in cases:
        record=case['record'];ids=[r['vid'] for r in record['rows']]
        group=case['plan']['members'];slots=[ids.index(v) for v in group]
        acts=comparators(record,600)+[case['plan']]
        frames={int(f['time_s']):{r['vid']:r for r in f['vehicles']} for f in case['frames'][0]}
        policy=load_actor(record['policy_state'],device) if record.get('policy_state') is not None else None
        for final_chain in (False,True):
            initial=record['rows'];bs=len(acts)
            def tensor(name,dtype=torch.float32):
                return torch.tensor([[r[name] for r in initial]]*bs,dtype=dtype,device=device)
            state=dict(s_m=tensor('s_m'),v_mps=tensor('v_mps'),lane=tensor('lane',torch.long),
                present=torch.ones(bs,len(ids),dtype=torch.bool,device=device),is_cav=tensor('is_cav',torch.bool),
                length_m=torch.tensor([[r.get('length_m',5.) for r in initial]]*bs,device=device),a_mps2=torch.zeros(bs,len(ids),device=device))
            trace=[];errors=[];step_differences=0;unresolved_steps=[]
            with plan_executor(6,policy,final_chain=final_chain):
                cb=runtime.option_controller(mp,acts,initial)
                for tick in range(12):
                    t=300+tick//2
                    # Real observed background; retain the group callback's own
                    # propagated state. Departures use the observed roster.
                    current=frames[t];nxt=frames.get(t+1,{})
                    half,unresolved=observable_midpoint(current,nxt) if kinematic and tick%2 else ({},{})
                    if unresolved:unresolved_steps.append(dict(tick=tick,vehicles=unresolved))
                    for slot,vid in enumerate(ids):
                        present=vid in current
                        state['present'][:,slot]=present
                        if not present:continue
                        row=current[vid]
                        if (not continuous and tick%2==0) or tick==0 or vid not in group:
                            f=.5 if tick%2 and vid in nxt else 0.
                            observed=half.get(vid) if kinematic and tick%2 else None
                            for k in ('s_m','v_mps'):
                                state[k][:,slot]=observed[k] if observed is not None else row[k]+f*(nxt.get(vid,row)[k]-row[k])
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
                unresolved_half_steps=unresolved_steps,
                nonidentical_plan_policy_steps=step_differences,
                max_position_error=max([abs(e['error']) for e in errors if e['field']=='s_m'] or [0.]),
                max_speed_error=max([abs(e['error']) for e in errors if e['field']=='v_mps'] or [0.])))
    return dict(cases=output,wall_s=time.monotonic()-begin,new_sumo_sim_s=0,device=device,
        kinematic_midpoints=kinematic,continuous_group_motion=continuous,
        contract='actual WM callback; observable endpoints only; '+
            ('law-reconstructed semi-implicit half-step backgrounds with unresolved roster/lane changes retained as interpolation fallbacks; ' if kinematic else 'interpolated half-step neighbours; ')+
            ('group propagated continuously while present; ' if continuous else 'group reset at1s decisions; ')+
            'numeric errors reported, not genuine half-step sensed/full-road/GPU parity; no numerical activation')


if __name__=='__main__':
    print(json.dumps(run(sys.argv[1])))
