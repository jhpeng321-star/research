"""Recorded qualification pulses: supplemental, explicitly outside E3 verdict worlds."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path.home()/'lrdiag'
sys.path[:0]=[str(ROOT),str(ROOT/'runtime'),str(ROOT/'runtime/src')]
from replay import safe,dump,yielding,WM_SHA


def run(seed,draw):
    import hashlib
    import torch
    from dataclasses import replace
    from tools.r10_attribution import load_model
    from wms.data.c4_episodes import build_c4_datasets
    from wms.data.collate import collate
    from wms.train.inputs import prepare_inputs
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert 969824<=seed<=969835
    jobs=Path.home()/'c4/runs/pulse/jobs'/f's{seed}_m2400_t300'
    source=Path.home()/'c4/runs/pulse/src/all'
    rec=json.loads(safe(jobs/'record.json').read_text())
    ck=ROOT/'wm2/wm_s.pt';assert hashlib.sha256(ck.read_bytes()).hexdigest()==WM_SHA
    cfg,settings,model,report=load_model(str(ROOT/'runtime/configs/c4_wm_loop14_r16_ft.toml'),str(ck),
        ('data.entries_v2=true','loss.paired_diff.couple_randomness=true'))
    assert not report['new'] and not report['unused']
    settings=replace(settings,seed=140014+draw*1000003)
    ds=build_c4_datasets(source,horizon_steps=60,history_s=60.,history_stride_s=1.,
        entries_v2=True,stride_s=1.,junction_fix='ffill',command_known=True)['train']
    entries=ds._loader.entries
    ix=next(i for i,(ep,tick) in enumerate(ds.positions)
        if int(entries[ep]['seed'])==seed
        and abs(float(ds._loader.episode(ep).array('time_s')[tick])-300)<1e-6)
    base=ds[ix]
    names=['nopulse']+[f'{p}__t{i}' for i in range(len(rec['targets'])) for p in ('pulse_m','pulse_p')]
    records=[]
    for name in names:
        r=copy.deepcopy(base);r['pair_group']=seed;r['pair_ref']=0
        r['meta']['pair_key']=f'pulse:{seed}'
        records.append(r)
    inp=prepare_inputs(collate(records),model,settings,torch.device('cpu'),60)
    pr=inp.paired_random;assert pr is not None
    inp.initial['merge_pivot_rng_seed']=torch.tensor((pr._keys&np.uint64((1<<52)-1)).astype(np.int64))
    native=inp.cav_controller;assert native is not None
    steps=[0]
    def pulse_controller(state):
        nxt=native(state)
        if steps[0]<6:
            for b,name in enumerate(names[1:],1):
                target=int(name.rsplit('t',1)[1]);slot=rec['targets'][target]['target_slot']
                val=-1.5 if name.startswith('pulse_m') else 1.
                a=torch.as_tensor(val,dtype=nxt.dtype)
                v=(state['v_mps'][b,slot]+.5*a).clamp_min(0)
                s=state['s_m'][b,slot]+.25*(state['v_mps'][b,slot]+v)
                nxt[b,slot,0]=s;nxt[b,slot,2]=v;nxt[b,slot,3]=a
        steps[0]+=1
        if steps[0]%20==0:yielding()
        return nxt
    # Keep the production native-lane delegation contract on the wrapper.
    if hasattr(native,'native_lane_mask'):
        pulse_controller.native_lane_mask=native.native_lane_mask
    torch.manual_seed(140014+seed+draw*1000003)
    yielding();start=time.monotonic()
    with torch.no_grad():
        roll=model.rollout(inp.initial,n_steps=60,cav_teacher=None,
            present_seq=inp.present_seq[:,:61],is_cav_seq=inp.is_cav[:,:61],
            history=inp.hist,history_mask=inp.hist_mask,controller_id=inp.controller_id,
            action_timeline=inp.action_timeline,cav_controller=pulse_controller,entries=inp.entries,
            driver_history=inp.observed_hist,driver_history_mask=inp.observed_hist_mask,
            **({'lane_uniform':pr.uniform('lane',range(60))} if model._samples_lanes else {}))
    arr=lambda a:a.detach().cpu().numpy()
    v=np.concatenate([arr(inp.initial['v_mps'])[:,None],arr(roll['speeds'])],1)[:,::2,:192]
    m=np.concatenate([arr(inp.initial['present'])[:,None],arr(roll['masks'])],1)[:,::2,:192].astype(bool)
    lane=np.concatenate([arr(inp.initial['lane'])[:,None],arr(roll['lanes'])],1)[:,::2,:192]
    s=np.concatenate([arr(inp.initial['s_m'])[:,None],arr(roll['positions'])],1)[:,::2,:192]
    tapes={name:np.load(safe(jobs/f'traj__{name}.npz'),allow_pickle=False) for name in names}
    initial=tapes['nopulse'];present=initial['present'][0]
    # Explicitly verify slot alignment instead of assuming compacted index equivalence.
    valid_initial=present&(initial['lane'][0]>=0)
    assert np.allclose(s[0,0,valid_initial],initial['s'][0,valid_initial],atol=1e-3)
    assert np.allclose(v[0,0,valid_initial],initial['v'][0,valid_initial],atol=1e-4)
    rows=[];signals=[]
    for b,name in enumerate(names[1:],1):
        ti=int(name.rsplit('t',1)[1]);target=rec['targets'][ti]['target_slot']
        target_lane=int(initial['lane'][0,target]);target_s=float(initial['s'][0,target])
        followers=np.flatnonzero(present&(initial['lane'][0]==target_lane)&(initial['s'][0]<target_s))
        followers=followers[np.argsort(-initial['s'][0,followers])]
        chain=np.r_[target,followers]
        amplitudes={}
        for source_label in ('real','wm'):
            vv=tapes[name]['v'] if source_label=='real' else v[b]
            nv=initial['v'] if source_label=='real' else v[0]
            mm=tapes[name]['present'] if source_label=='real' else m[b]
            nm=initial['present'] if source_label=='real' else m[0]
            ll=tapes[name]['lane'] if source_label=='real' else lane[b]
            nl=initial['lane'] if source_label=='real' else lane[0]
            amps=[]
            for slot in chain:
                common=mm[:,slot]&nm[:,slot]
                amps.append(float(np.max(np.abs((vv-nv)[common,slot]))) if common.any() else np.nan)
            amplitudes[source_label]=amps
            for rank,(slot,amp) in enumerate(zip(chain,amps)):
                previous=amps[rank-1] if rank else np.nan
                distance=target_s-float(initial['s'][0,slot])
                local_distance=float(initial['s'][0,chain[rank-1]]-initial['s'][0,slot]) if rank else np.nan
                gain=amp/previous if previous>0 else None
                gain100=np.exp(100*np.log(gain)/local_distance) if gain is not None and gain>0 and local_distance>0 else (0. if gain==0 else None)
                if gain100 is not None and not np.isfinite(gain100):gain100=None
                common=mm[:,slot]&nm[:,slot]
                same_lane=common&(ll[:,slot]==target_lane)&(nl[:,slot]==target_lane)
                rows.append({'scope':'supplemental_non_verdict','wm_sha256':WM_SHA,'seed':seed,
                    'draw':draw,'pulse':name,'source':source_label,'target_lane':target_lane,
                    'rank':rank,'slot':int(slot),'distance_m':distance,'amplitude_dv_mps':amp,
                    'amplitude_vs_target':amp/amps[0] if amps[0]>0 else None,
                    'gain_per_vehicle':gain,'gain_per_100m':gain100,
                    'fraction_common_same_lane':float(same_lane.sum()/max(common.sum(),1))})
            chosen=np.zeros(192,bool);chosen[followers]=True
            for t in range(1,31):
                keep=chosen&mm[t]&nm[t]
                signals.append({'scope':'supplemental_non_verdict','seed':seed,'draw':draw,'pulse':name,
                    'source':source_label,'time_s':t,'n_followers':int(keep.sum()),
                    'rms_dv_mps':float(np.sqrt(np.mean((vv[t,keep]-nv[t,keep])**2))) if keep.any() else None})
    doc={'scope':'supplemental_non_verdict','seed':seed,'draw':draw,'role':'qualification_train',
         'wm_sha256':WM_SHA,'horizon_s':30,'seconds':time.monotonic()-start,'rows':rows,'signals':signals,
         'caveat':'Initial ordered following platoon; lane residence reported, no post-treatment filter. '
                  'WM2 native plant and first-passage lanes; pulse command for 3 s; target-normalized gain.'}
    dump(ROOT/'pulse_results'/f's{seed}_k{draw}.json',doc)
    print(json.dumps({k:v for k,v in doc.items() if k not in ('rows','signals')}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--draw',type=int,required=True);a=ap.parse_args();run(a.seed,a.draw)
