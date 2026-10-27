"""CPU-only matched-tape diagnostic. No simulator imports or executable calls."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import types
import numpy as np

HOME=Path.home()/'lrdiag'
BAD=('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth')
ARMS=('native','mappo','ours_c1')
VARIANTS=('baseline','residual_off','population','courtesy_off','yield_off','ramp_adapt_off')
WM_SHA='b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436'


def safe(p):
    p=Path(p)
    assert not any(x in str(p).lower() for x in BAD),p
    return p


def dump(p,obj):
    p=safe(p); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,allow_nan=False,default=lambda x:x.item()))
    tmp.replace(p)


def yielding():
    """Yield at step boundaries without signalling/altering any other process."""
    while True:
        busy=(HOME/'YIELD').exists()
        for p in Path('/proc').iterdir():
            if not p.name.isdecimal(): continue
            try:
                if (p/'comm').read_text().strip() in ('sumo','sumo-gui'): busy=True
            except (FileNotFoundError,PermissionError,ProcessLookupError): pass
        if not busy: return
        time.sleep(15)


def regions(s,lane):
    return {'upstream':(lane<2)&(s<0),
            'merge':(lane<2)&(s>=0)&(s<=75.44),
            'ramp':lane==2,
            'downstream':(lane<2)&(s>75.44)}


def speed_bins(v,mask,pos,lane,cav):
    """Half-open 10 s bins, excluding the duplicated endpoint."""
    out={}
    for r,region in regions(pos,lane).items():
        for cls,sel in [('all',np.ones_like(mask)),('CAV',cav),('HDV',~cav)]:
            keep=mask&region&sel
            vals=[]; counts=[]
            for t in range(0,len(v)-1,20):
                k=keep[t:t+20]; n=int(k.sum()); counts.append(n)
                vals.append(float(v[t:t+20][k].mean()) if n else None)
            out[r+'/'+cls]={'speed':vals,'count':counts}
    return out


def breakdown(values):
    """BIC-selected downward step + trend vs best constant/linear model."""
    y=np.array([np.nan if x is None else x for x in values],float)
    ok=np.isfinite(y); t=np.arange(len(y)); n=int(ok.sum())
    if n<8: return {'event':None,'onset_s':None,'reason':'insufficient_bins'}
    x=t[ok].astype(float); y=y[ok]
    def fit(X,k):
        b=np.linalg.lstsq(X,y,rcond=None)[0]
        sse=max(float(((y-X@b)**2).sum()),n*np.finfo(float).eps*max(float(np.mean(y*y)),1.))
        return n*np.log(sse/n)+k*np.log(n),b
    c,_=fit(np.ones((n,1)),1)
    lin,_=fit(np.column_stack([np.ones(n),x]),2)
    null=min(c,lin); choices=[]
    for cp in range(3,len(values)-2):
        step=(x>=cp).astype(float)
        if step.sum()<3 or (1-step).sum()<3: continue
        bic,b=fit(np.column_stack([np.ones(n),x,step]),4)
        if b[2]<0: choices.append((bic,cp,float(b[2])))
    if not choices: return {'event':False,'onset_s':None,'bic_improvement':0.}
    bic,cp,drop=min(choices)
    chosen=bic<null; onset=float((cp-6)*10) if chosen else None
    return {'event':bool(chosen and onset>=0),'onset_s':onset,
            'preexisting':bool(chosen and onset<0),'drop_mps':drop,
            'bic_improvement':float(null-bic)}


def build_source(root,panel='e3'):
    from wms.data.c4_pairs import PairedBranchSource
    manifest=json.loads(safe(root/'C4_EPISODES_MANIFEST.json').read_text())
    if panel=='e3':
        split=json.loads(safe(root/'SPLIT.json').read_text())
        assert len(manifest['episodes'])==108
        assert all(split['worlds'][str(e['seed'])]=='validation' and not 964000<=e['seed']<=964199
                   for e in manifest['episodes'])
    else:
        assert all(not 964000<=e['seed']<=964199 for e in manifest['episodes'])
    source=PairedBranchSource(root,horizon_steps=600,history_s=60.,history_stride_s=1.,
        entries_v2=True,junction_fix='ffill',command_known=True,
        arm_kinds=('B','POLICY') if panel=='e3' else ('B','SEED'),
        role='train',role_of={e['world']:'train' for e in manifest['episodes']},name='lrdiag_'+panel)
    assert source.receipt['n_groups']==(36 if panel=='e3' else 48)
    if panel=='e3':assert source.receipt['n_windows']==108
    return source,manifest


def configure(model,variant):
    model.residual_accel_off=variant=='residual_off'
    model.follower_courtesy_off=variant in ('courtesy_off','residual_off')
    model.mainline_merge_yield_learnable=variant not in ('yield_off','residual_off')
    # The learned yield path replaces the physical rule; disable both for OFF.
    from dataclasses import replace
    if variant in ('yield_off','residual_off'): model.physics=replace(model.physics,mainline_merge_yield=False)
    model.ramp_speed_adapt_learnable=variant not in ('ramp_adapt_off','residual_off')
    if variant=='population':
        original=model._encode_drivers
        def default_params(self,kin,present,*args,**kwargs):
            encoded,embedding,params=original(kin,present,*args,**kwargs)
            defaults=self._default_driver_parameters(kin,*kin.shape[:2])
            enc=self.driver_encoder
            if enc.driver_param_mode=='physical':
                defaults[...,:5]=enc.physical_lower+(enc.physical_upper-enc.physical_lower)*enc.centre_logits.sigmoid()
            from wms.model.driver_encoder import PARAMETER_NAMES
            return encoded,embedding,{name:defaults[...,i] for i,name in enumerate(PARAMETER_NAMES)}
        model._encode_drivers=types.MethodType(default_params,model)


def run(gi,draw,variant,steps=600,panel='e3'):
    import sys
    sys.path[:0]=[str(HOME/'runtime'),str(HOME/'runtime/src')]
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    from tools.r10_attribution import load_model
    from wms.data.collate import collate
    from wms.train.inputs import prepare_inputs
    from tools.wm_eval_common import delay_proxy, SPEED_LIMIT_MPS
    os.environ['CUDA_VISIBLE_DEVICES']=''
    config=HOME/'runtime/configs/c4_wm_loop14_r16_ft.toml'
    ck=HOME/'wm2/wm_s.pt'
    assert hashlib.sha256(ck.read_bytes()).hexdigest()==WM_SHA
    cfg,settings,model,report=load_model(str(config),str(ck),
        ('data.entries_v2=true','loss.paired_diff.couple_randomness=true'))
    assert not report['unused'] and not report['new'],report
    configure(model,variant)
    from dataclasses import replace
    settings=replace(settings,seed=140014+draw*1000003)
    data_root=HOME/('e3data' if panel=='e3' else 'pairs300')
    source,manifest=build_source(data_root,panel)
    g=source.groups[gi]; indexes=g['rows']
    def name(row):
        ep=source.dataset.positions[int(row)][0]
        return source.dataset._loader.entries[ep]['arm']
    if panel=='e3':indexes=sorted(indexes,key=lambda r:ARMS.index(name(r)))
    else:
        ref=g['rows'][0]
        indexes=[ref]+sorted([i for i in indexes if i!=ref],key=name)
    arms=tuple(name(i) for i in indexes)
    records=source.group_records(gi,indexes,base_row=0)
    batch=collate(records)
    inp=prepare_inputs(batch,model,settings,torch.device('cpu'),steps)
    assert inp.paired_random is not None
    # Reset global RNG per world/draw/variant. Identity-keyed merge streams remove
    # the documented arbitrary batch-row coupling limitation of earlier scorers.
    torch.manual_seed(140014+7919*gi+1000003*draw)
    keys=inp.paired_random._keys & np.uint64((1<<52)-1)
    inp.initial['merge_pivot_rng_seed']=torch.tensor(keys.astype(np.int64))
    kwargs={}
    if model._samples_lanes: kwargs['lane_uniform']=inp.paired_random.uniform('lane',range(steps))
    # Lightweight yielding hook once each model step, scanning /proc every 10 s.
    original=model.forward; last=[0.]; calls=[0]
    def forward(*a,**k):
        if time.monotonic()-last[0]>10:
            yielding(); last[0]=time.monotonic()
        out=original(*a,**k); calls[0]+=1
        if calls[0]%100==0:print(json.dumps({'forward_steps':calls[0],'elapsed_s':time.monotonic()-start}),flush=True)
        return out
    model.forward=forward
    start=time.monotonic()
    with torch.no_grad():
        roll=model.rollout(inp.initial,n_steps=steps,k0=0,
            cav_teacher=None if inp.teacher is None else inp.teacher[:,:steps+1],
            present_seq=inp.present_seq[:,:steps+1],is_cav_seq=inp.is_cav[:,:steps+1],
            history=inp.hist,history_mask=inp.hist_mask,controller_id=inp.controller_id,
            action_timeline=inp.action_timeline,cav_controller=inp.cav_controller,
            entries=inp.entries,driver_history=inp.observed_hist,
            driver_history_mask=inp.observed_hist_mask,**kwargs)
    arr=lambda x:x.detach().cpu().numpy()
    m0=arr(inp.present[:,0]).astype(bool); cav0=arr(inp.is_cav[:,0]).astype(bool)
    speed=np.concatenate([arr(inp.initial['v_mps'])[:,None],arr(roll['speeds'])],1)
    mask=np.concatenate([m0[:,None],arr(roll['masks']).astype(bool)],1)
    pos=np.concatenate([arr(inp.initial['s_m'])[:,None],arr(roll['positions'])],1)
    lane=np.concatenate([arr(inp.initial['lane'])[:,None],arr(roll['lanes'])],1)
    state=arr(inp.states[:,:steps+1]); real_v=state[...,4]
    real_lane=state[...,1:4].argmax(-1)
    real_mask=arr(inp.present_seq[:,:steps+1]).astype(bool)
    # Actual class membership follows entrant state, not fixed slot labels.
    cav=np.concatenate([cav0[:,None],arr(roll['is_cav_seq']).astype(bool)],1) if 'is_cav_seq' in roll else None
    # Stable original slots have recorded class membership; post-t0 synthetic
    # entries cannot be identity-matched to recorded vehicles. Timeline uses
    # original t0 cohort only, with t0 regions/classes fixed, avoiding that bias.
    out={'world':g['pair_key'].split('@')[0],'pair_key':g['pair_key'],'draw':draw,'group_index':gi,
         'variant':variant,'steps':steps,'panel':panel,
         'role':'validation' if panel=='e3' else 'diagnostic','seconds':time.monotonic()-start,
         'load_report':report,'arms':{},'source_receipt':source.receipt}
    for b,arm in enumerate(arms):
        sel=m0[b]&~cav0[b]
        delay={str(h):[delay_proxy(speed[b],mask[b],sel,h),
                       delay_proxy(real_v[b],real_mask[b],sel,h)]
               for h in (120,300) if h<=steps*.5}
        wm_bins=speed_bins(speed[b],mask[b],pos[b],lane[b],np.broadcast_to(cav0[b],mask[b].shape))
        real_bins=speed_bins(real_v[b],real_mask[b],state[b,...,0],real_lane[b],
                             arr(inp.is_cav[b,:steps+1]).astype(bool))
        entry=next(e for e in manifest['episodes'] if e['world']==out['world'] and e['arm']==arm)
        raw=data_root/entry['dir']
        warm=np.load(safe(raw/'veh_kin.npy'),mmap_mode='r')[:121]
        warm_mask=np.load(safe(raw/'veh_mask.npy'),mmap_mode='r')[:121].astype(bool)
        warm_cav=np.load(safe(raw/'is_cav.npy'),mmap_mode='r')[:121].astype(bool)
        warm_bins=speed_bins(warm[...,4],warm_mask,warm[...,0],warm[...,1:4].argmax(-1),warm_cav)
        collapse={}
        if steps==600:
            for region in ('upstream','merge'):
                k=region+'/all'; prefix=warm_bins[k]['speed']
                collapse[region]={'real':breakdown(prefix+real_bins[k]['speed']),
                                  'wm':breakdown(prefix+wm_bins[k]['speed'])}
        timeline=[]
        wm_regions=regions(pos[b],lane[b]); real_regions=regions(state[b,...,0],real_lane[b])
        for region in wm_regions:
            for cls,c in [('CAV',cav0[b]),('HDV',~cav0[b])]:
                chosen=m0[b]&c
                for t in range(0,steps,20):
                    # Horizon-gate delay convention: absent t0 vehicles are at
                    # free speed, so exited vehicles contribute zero delay.
                    w=np.maximum(0,1-speed[b,t+1:t+21]/SPEED_LIMIT_MPS)*mask[b,t+1:t+21]*wm_regions[region][t+1:t+21]
                    r=np.maximum(0,1-real_v[b,t+1:t+21]/SPEED_LIMIT_MPS)*real_mask[b,t+1:t+21]*real_regions[region][t+1:t+21]
                    timeline.append([region,cls,(t//20+1)*10,
                                     float(w[:,chosen].sum()*.5),float(r[:,chosen].sum()*.5),int(chosen.sum())])
        # Common original identities, continuous per-vehicle paired-speed signal.
        # Full traces are retained only for baseline, to limit disk use.
        out['arms'][arm]={'delay':delay,'breakdown':collapse,'timeline':timeline,
                         'speed_bins_real':real_bins,'speed_bins_wm':wm_bins}
    if variant=='baseline' and panel=='e3':
        path=HOME/'traces'/f'g{gi:02d}_k{draw}.npz'; path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,speed=speed[:,:,:192],mask=mask[:,:,:192],
            pos=pos[:,:,:192],lane=lane[:,:,:192],real_speed=real_v[:,:,:192],
            real_mask=real_mask[:,:,:192],real_pos=state[:,:,:192,0],
            real_lane=real_lane[:,:,:192],cav0=cav0[:,:192],m0=m0[:,:192])
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--group',type=int,required=True)
    ap.add_argument('--draw',type=int,default=0); ap.add_argument('--variant',choices=VARIANTS,default='baseline')
    ap.add_argument('--steps',type=int,default=600)
    ap.add_argument('--panel',choices=('e3','pairs300'),default='e3'); a=ap.parse_args()
    out=run(a.group,a.draw,a.variant,a.steps,a.panel)
    prefix='' if a.panel=='e3' else 'p300_'
    p=HOME/'results'/f'{prefix}g{a.group:02d}_{a.variant}_k{a.draw}_s{a.steps}.json'
    dump(p,out)
    print(json.dumps({'result':str(p),'seconds':out['seconds']}),flush=True)


if __name__=='__main__': main()
