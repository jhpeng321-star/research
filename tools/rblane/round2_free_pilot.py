"""Paired native FIT pilot: persistent fitted LC vs source defaults, entrants ON.

Both variants use identical fitted IDM/arrival streams and causal wish warmup.
No future observation/tape/target is loaded by this inference entrypoint.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

ROOT = Path(__file__).resolve().parents[2]
os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
    C4_ALLOWED_ROOTS='/root/autodl-tmp/c4:/root/wmcalib:/root/rbeval20261031',
    C4_EXECUTION_PARITY='2', C4_SUMO_LEADER_RULE='1', WMPHYS_VARIANT='FULL',
    C4_EXECUTION_DRIVER=str(ROOT/'configs/execparity_cav_idm_v1.json'))

import numpy as np
import torch

from tools.rbint.audit import forbid_learned_corrections
from tools.rbint.departure import rollout_after_movement
from tools.rbint.observations import load_motion_history, fitted_inputs
from tools.rbint.recipe import candidate_future, load_candidate, native_controller
from tools.rbeval.core import digest, reference, save_new
from tools.rblane.runtime_history import lane_arrays
from tools.rblane.fit_guard import guarded_start_source
from wms.control.dreamer import with_entry_slots
from wms.control.start_states import StartBatch
from wms.model.entries import build_entrance_history, _counts_counter
from wms.physics.batched import road_end_exit_bound
from wms.physics.lane_continuation import warm_lane_memory


def fit_starts(model, worlds):
    """Use the registered exporter with an explicit pre-read FIT loop guard."""
    if not set(worlds)<=set(range(969420,969432)):
        raise ValueError('development12 starts only')
    namespace = dict(torch=torch,np=np,json=json,Path=Path,ACTORS=[],
        ROOT=Path('/root/wmcalib'),DATA=Path('/root/wmcalib/data'),StartBatch=StartBatch,
        build_entrance_history=build_entrance_history,_counts_counter=_counts_counter)
    for name in ('worker','costs'):
        reference(name,namespace)
    namespace['digest']=digest
    namespace['frozen']=SimpleNamespace(**namespace)
    refs=json.loads((ROOT/'tools/rbeval/references.json').read_text())
    code=refs['starts']['code']
    assert hashlib.sha256(code.encode()).hexdigest()==refs['starts']['code_sha256']
    tree,guarded=guarded_start_source(code,worlds)
    exec(compile(tree,'[own registered FIT-only starts]','exec'),namespace)
    batches,provenance=namespace['make_starts_wmdata'](model)
    assert {b.metadata[0]['seed'] for b in batches}==set(worlds)
    return batches,namespace['frozen'],dict(reference_sha256=refs['starts']['code_sha256'],
        fit_guarded_source_sha256=hashlib.sha256(guarded.encode()).hexdigest(),
        world_guard_applied_before_any_start_reads=True,worlds=worlds,provenance=provenance)


def lane_cells(initial,rollout,dt):
    """Exactly target-style arrival-frame event positions and half-open bins."""
    lane=rollout['lanes'][0].long()
    pos=rollout['positions'][0]
    present=rollout['masks'][0].bool()
    previous_lane=torch.cat((initial['lane'][0,None],lane[:-1]),0)
    previous_present=torch.cat((initial['present'][0,None],present[:-1]),0).bool()
    changed=present & previous_present & (lane!=previous_lane)
    times=300.+np.arange(1,len(lane)+1)*dt
    cells={}
    for index in range(int(len(lane)*dt/10)):
        window=torch.as_tensor((times>=300+10*index)&(times<310+10*index))[:,None]
        for region in ('upstream','merge','downstream'):
            spatial=(pos<0) if region=='upstream' else (pos>75.44) if region=='downstream' else ((pos>=0)&(pos<=75.44))
            for kind in ('mainline','ramp_merge'):
                # The raw target denominator uses measured current lane/time;
                # an event is classified by its source lane and arrival position.
                traffic=(lane<2) if kind=='mainline' else (lane==2)
                transitions=(previous_lane<2)&(lane<2) if kind=='mainline' else ((previous_lane==2)&(lane==1))
                exposure=float((present & window & spatial & traffic).sum())*dt
                count=int((changed & window & spatial & transitions).sum())
                cells[f'{region}/{kind}/{index}']=dict(count=count,veh_time_s=exposure,
                    rate_per_veh_h=3600*count/exposure if exposure else None)
    return cells


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--worlds',type=int,nargs='+',required=True)
    ap.add_argument('--samples',type=int,default=2)
    ap.add_argument('--horizon',type=int,choices=(60,300),default=300)
    args=ap.parse_args()
    if (not set(args.worlds)<=set(range(969420,969432)) or args.out.exists()
            or not args.out.parent.is_dir() or os.getloadavg()[0]>=20 or not 1<=args.samples<=8):
        raise ValueError('new immutable development pilot and spare CPU required')
    args.out.mkdir()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    begin=time.time()
    models,construction=load_candidate(['/root/autodl-tmp/c4/wmcombined_wm2/wm_s.pt'],
        [str(ROOT/'configs/wm_phys_rb2_lane.toml')],'cpu')
    model=models[0]
    policy=torch.load('/root/autodl-tmp/c4/setupaudit20261030/policy_000190.pt',map_location='cpu',weights_only=False)
    model.cav_exec_cap=policy['config']['cav_exec_cap']
    batches,frozen,start_receipt=fit_starts(model,args.worlds)
    save_new(args.out/'IDENTITY.json',dict(source_commit=json.loads((ROOT/'INT_STAGE.json').read_text())['commit'],
        construction=construction,starts=start_receipt,samples=args.samples,horizon=args.horizon,
        comparison='same causal IDM and arrivals; both LC variants warm; fitted LC only factor',cuda=False))
    continuation=model._wmphys_lane_continuation
    caches={variant:{} for variant in ('source_default','fitted_lc')}
    for source in batches:
        world=int(source.metadata[0]['seed'])
        observed_start,raw=load_motion_history(world)
        for sample in range(args.samples):
            paired=[]
            for variant in ('source_default','fitted_lc'):
                model._wmphys_lane_continuation=continuation if variant=='fitted_lc' else None
                model._wmphys_conditioned_cache=caches[variant]
                batch=with_entry_slots(copy.deepcopy(source),args.horizon,512,
                    torch.Generator().manual_seed(1_000_003*191),queue_state=True)
                initial=copy.deepcopy(batch.state)
                initial['exit_s_m']=torch.full_like(initial['s_m'],road_end_exit_bound(model.physics))
                with torch.no_grad(),forbid_learned_corrections(model):
                    kwargs,entries,receipt=candidate_future(model=model,batch=batch,initial=initial,
                        future=sample,seed=world,frozen=frozen)
                    ids,_,_=fitted_inputs(initial,world,batch.metadata[0]['initial_state_sha256'],model._wmphys_arrival_fit)
                    if variant=='source_default':
                        initial.update(warm_lane_memory(lane_arrays(observed_start,raw),t0_s=300.,physics=model.physics,
                            driver_params=kwargs['driver_params'],vehicle_ids=ids),vehicle_id=ids.clone())
                    if variant=='source_default':
                        baseline_rates=copy.deepcopy(receipt['demand_rate_draws'])
                        baseline_idm={name:kwargs['driver_params'][name].clone()
                            for name in ('v0','time_headway','accel','decel','min_gap')}
                    else:
                        assert baseline_rates==receipt['demand_rate_draws']
                        assert all(torch.equal(value,kwargs['driver_params'][name]) for name,value in baseline_idm.items())
                    roll=rollout_after_movement(model,initial,n_steps=int(args.horizon/model.physics.dt_s),
                        cav_controller=native_controller(model),**kwargs)
                    audit=entries.audit_population(initial,roll)
                record=dict(world=world,sample=sample,variant=variant,cells=lane_cells(initial,roll,model.physics.dt_s),
                    audit=audit,receipt=receipt,source_phase=roll['source_phase'],
                    no_slot=[s.no_slot for s in entries.process.stats])
                paired.append(record)
                save_new(args.out/f'w{world}_k{sample}_{variant}.json',record)
                print(json.dumps(dict(world=world,sample=sample,variant=variant,conserved=audit['conserved'],
                    count=sum(c['count'] for c in record['cells'].values()),elapsed_s=time.time()-begin)),flush=True)
            assert paired[0]['audit']['due']==paired[1]['audit']['due']
    save_new(args.out/'COMPLETE.json',dict(status='PASS',worlds=args.worlds,samples=args.samples,
        horizon=args.horizon,paired_draws=True,records=len(batches)*args.samples*2,elapsed_s=time.time()-begin))


if __name__=='__main__':main()
