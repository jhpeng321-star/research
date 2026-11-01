"""Replay ONE step to identify non-LC initial internal→normal transitions.

This is scoring validation of the frozen native pilot, not new population fitting
or a repeated300s forecast. Uses identical300s padding, causal parameters and RNG.
"""
import argparse
import copy
import json
from pathlib import Path
import time

# The frozen pilot module sets the exact CPU-only inference environment.
from .round2_free_pilot import (ROOT,fit_starts,load_candidate,load_motion_history,
    with_entry_slots,candidate_future,fitted_inputs,warm_lane_memory,lane_arrays,
    native_controller,rollout_after_movement,forbid_learned_corrections,
    road_end_exit_bound,save_new,torch,os)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--worlds',type=int,nargs='+',required=True)
    ap.add_argument('--samples',type=int,default=2)
    ap.add_argument('--horizon',type=int,default=300)
    args=ap.parse_args()
    if args.horizon!=300 or args.out.exists() or not args.out.parent.is_dir() or os.getloadavg()[0]>=20:
        raise ValueError('new prefix audit with300s padding and spare CPU required')
    args.out.mkdir();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    models,construction=load_candidate(['/root/autodl-tmp/c4/wmcombined_wm2/wm_s.pt'],
        [str(ROOT/'configs/wm_phys_rb2_lane.toml')],'cpu')
    model=models[0]
    policy=torch.load('/root/autodl-tmp/c4/setupaudit20261030/policy_000190.pt',map_location='cpu',weights_only=False)
    model.cav_exec_cap=policy['config']['cav_exec_cap']
    batches,frozen,starts=fit_starts(model,args.worlds)
    save_new(args.out/'IDENTITY.json',dict(n_steps=1,padding_horizon=300,starts=starts,
        purpose='exclude internal→normal initialization from LC scoring only',construction=construction))
    continuation=model._wmphys_lane_continuation;caches={v:{} for v in ('source_default','fitted_lc')}
    begin=time.time();records=0
    for source in batches:
        world=int(source.metadata[0]['seed']);observed,raw=load_motion_history(world)
        for sample in range(args.samples):
            for variant in ('source_default','fitted_lc'):
                model._wmphys_lane_continuation=continuation if variant=='fitted_lc' else None
                model._wmphys_conditioned_cache=caches[variant]
                batch=with_entry_slots(copy.deepcopy(source),300,512,
                    torch.Generator().manual_seed(1_000_003*191),queue_state=True)
                state=copy.deepcopy(batch.state)
                state['exit_s_m']=torch.full_like(state['s_m'],road_end_exit_bound(model.physics))
                with torch.no_grad(),forbid_learned_corrections(model):
                    kwargs,entries,receipt=candidate_future(model=model,batch=batch,initial=state,
                        future=sample,seed=world,frozen=frozen)
                    ids,_,_=fitted_inputs(state,world,batch.metadata[0]['initial_state_sha256'],model._wmphys_arrival_fit)
                    if variant=='source_default':
                        state.update(warm_lane_memory(lane_arrays(observed,raw),t0_s=300.,physics=model.physics,
                            driver_params=kwargs['driver_params'],vehicle_ids=ids),vehicle_id=ids.clone())
                    roll=rollout_after_movement(model,state,n_steps=1,cav_controller=native_controller(model),**kwargs)
                    audit=entries.audit_population(state,roll)
                original=state['lane'][0];new=roll['lanes'][0,0];present=roll['masks'][0,0].bool()
                invalid=state['present'][0].bool()&((original<0)|(original>2))
                assert not (present&((new<0)|(new>2))).any(), 'later nonnormal frames require a fuller scorer audit'
                rows=[];cells={}
                for j in invalid.nonzero().flatten().tolist():
                    s=float(roll['positions'][0,0,j]);region='upstream' if s<0 else 'downstream' if s>75.44 else 'merge'
                    counted=bool(present[j] and original[j]<2 and new[j]<2 and original[j]!=new[j])
                    if counted:
                        key=f'{region}/mainline/0';cells[key]=cells.get(key,0)+1
                    rows.append(dict(slot=j,source=int(original[j]),destination=int(new[j]),present=bool(present[j]),s_m=s,counted_as_LC=counted))
                save_new(args.out/f'w{world}_k{sample}_{variant}.json',dict(world=world,sample=sample,variant=variant,
                    invalid_initial_rows=rows,excluded_LC_counts=cells,audit=audit,receipt=receipt))
                records+=1
                print(json.dumps(dict(world=world,sample=sample,variant=variant,excluded=sum(cells.values()),
                    elapsed_s=time.time()-begin)),flush=True)
    save_new(args.out/'COMPLETE.json',dict(status='PASS',n_steps=1,padding_horizon=300,worlds=args.worlds,
        samples=args.samples,records=records,elapsed_s=time.time()-begin))


if __name__=='__main__':main()
