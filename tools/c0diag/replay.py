"""Offline same-state cohort replay. No SUMO, teachers, or future recorded controls."""
import os
os.environ['C4_ALLOWED_ROOTS']='/root/main14_loop:/root/c0diag_s1'
import json
import hashlib
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from wms.control.policy14_cli import load_members
from wms.control.policy import load_policy_components, CavPolicy, RsuGroupEncoder, PolicyExecutor
from wms.control.policy14 import Policy14Trainer, make_controller_factory
from wms.control.dreamer import DreamerConfig
from wms.control.train_cli import GroupSignal
from wms.control.real_returns14 import ScaledTeamCritic
from wms.control.objective14 import authoritative_reward
from wms.control.obs import build_observations
from wms.control.safety import safe_accel
from wms.control.contract import RewardWeights
from wms.control.counterfactual import CounterfactualConfig, build_variants
from wms.control.start_states import StartBatch
from wms.data.c4_episodes import C4WindowLoader

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(.40)
DEV='cuda'
ROOT=Path('/root/main14_loop'); OUT=Path('/root/c0diag_s1'); DATA=ROOT/'data/pairs_c00.a1'
ck=ROOT/'policy/c00.a2/policy_latest.pt'
p=torch.load(ck,map_location='cpu',weights_only=True)
models,_=load_members([str(ROOT/'mainr16_base/wm_s_best.pt')],[str(ROOT/'mainr16_base/config.resolved.json')],DEV)
m=models[0]
policy,encoder,rules,_=load_policy_components(ck,device=DEV)
critic=ScaledTeamCritic(entry_backlog=True)
critic.load_state_dict(p['critic'])
cfg=replace(DreamerConfig(**p['config']),imagine_entries=False,real_return_mix=0.,future_samples=1,
            checkpoint_imagination=False,actor_gradient_audit=True)
switches={k:v for k,v in p['wm_controller_kwargs'].items() if k in ('policy_ramp_merge','native_ramp_merge','native_lane_change')}
trainer=Policy14Trainer(models,policy,GroupSignal(encoder),critic,None,
    observation_builder=build_observations,reward_fn=authoritative_reward,safety=safe_accel,
    controller_factory=make_controller_factory(**switches),config=cfg,
    reward_weights=RewardWeights(**p['reward_weights']),device=DEV,lane_rules=rules,
    counterfactual=CounterfactualConfig(**p['counterfactual']))
trainer.target_critic.load_state_dict(p['target_critic'])
entries=json.loads((DATA/'C4_EPISODES_MANIFEST.json').read_text())['episodes']
loader=C4WindowLoader(DATA,entries,[(i,120) for i in range(len(entries))],horizon_steps=20,
    history_s=m.history_s if m.long_history else None,history_stride_s=m.history_stride_s,
    entries_v2=False)
def load(n,e):return np.load(DATA/e['dir']/(n+'.npy'),mmap_mode='r',allow_pickle=False)
def batch_for(i):
    rec=loader(i)
    # The generic window decoder clamps junction lane -1 to lane 0 when ffill is off.
    # Initial deployment state must instead retain the observed invalid-lane mask.
    kin=torch.as_tensor(load('veh_kin',entries[i])[120].copy(),device=DEV)
    present=torch.as_tensor(load('veh_mask',entries[i])[120].copy(),device=DEV).bool()
    lane=torch.where(present&(kin[:,1:4].sum(-1)>0),kin[:,1:4].argmax(-1),-1)
    state={k:v[None] for k,v in dict(s_m=kin[:,0],v_mps=kin[:,4],a_mps2=kin[:,5],
        length_m=kin[:,6],lane=lane,present=present,is_cav=torch.as_tensor(rec['is_cav'][0].copy(),device=DEV).bool()).items()}
    hist={dest:torch.as_tensor(rec[src].copy(),device=DEV)[None] for src,dest in
          [('hist','history'),('hist_mask','history_mask'),('driver_hist','driver_history'),('driver_hist_mask','driver_history_mask')] if src in rec}
    return StartBatch(state,rollout_kwargs=hist,metadata=[rec['meta']])
def cohort_sumo(e,sec):
    kin=load('veh_kin',e); pre=load('veh_mask',e).astype(bool); ids=load('vid',e)
    sl=slice(121,121+2*sec)
    alive=pre[sl]&(ids[sl]==ids[120])&pre[120]&(ids[120]>=0)
    return float((.5*np.where(alive,np.maximum(0,1-kin[sl,:,4]/31.29),0)).sum())
mode=os.environ.get('C0DIAG_MODE','original')
rows=[]
for i,e in enumerate(entries):
    if e['branch']['kind']!='POLICY':continue
    if mode in ('enriched','ownership') and e['seed'] not in (968129,968132):continue
    if mode in ('enriched','ownership') and e['warmup_s']!=360:continue
    assert 968129<=e['seed']<=968141
    b=batch_for(i)
    # Initial memory is empty in both native-start branches and deployment executor.
    with torch.no_grad():
        _,_,params=m.encode_drivers(b.state,**b.rollout_kwargs)
        from wms.model.driver_encoder import PARAMETER_NAMES
        aug={**b.state,'inferred_driver_parameters':torch.stack([params[n].float() for n in PARAMETER_NAMES],-1),
             'inferred_idm_delta':float(m.physics.idm()[4])}
        exe=PolicyExecutor(policy,encoder,lane_rules=rules,device=DEV)
        out,ln=exe(aug,with_lanes=True)
        present=b.state['present']&b.state['is_cav']&(b.state['lane']>=0)
        mask=present.cpu().numpy()[0]
        req=load('requested_acceleration_mps2',e)[60]
        ctrl=load('controller',e)[60]
        cm=mask&(ctrl==3)
        initial_err=float(np.max(np.abs(out[0,cm]-req[cm]))) if cm.any() else 0
        first_lane=load('lane_request_target',e)[60]
        lane_err=int(np.sum(ln[0,cm]!=first_lane[cm])) if ln is not None else 0
    native=next(x for x in entries if x['pair_key']==e['pair_key'] and x['branch']['kind']=='B')
    horizons=[10,60]
    if e['seed'] in (968129,968132) and e['warmup_s']==360:horizons.append(300)
    if mode=='ownership':
        if e['seed']!=968132:continue
        trainer.controller_factory=make_controller_factory(**{**switches,'policy_ramp_merge':True})
        horizons=[300]
    for sec in horizons:
        trainer.config=replace(cfg,horizon=sec)
        variants=build_variants(b.state,trainer.counterfactual,torch.Generator().manual_seed(965149))
        with torch.no_grad():
            torch.manual_seed(965149)
            im=trainer.imagine(b,deterministic=True,variants=variants,future_seed=965149)
        base=variants.base_row[0]
        road=im['components']['delay'].sum(1).mean(0)
        states=im['states'][0][1:]
        speeds=torch.cat([s['v_mps'][0][s['present'][0]&s['is_cav'][0]&(s['lane'][0]>=0)] for s in states])
        cmd=im['action_audit']['audit_action'][0,:,0][im['action_audit']['audit_acting'][0,:,0]]
        effective=im['actions'][0,:,0][im['action_mask'][0,:,0]]
        physical_merges=0
        automatic_merges=0
        prior=im['states'][0][0]
        for ti,nxt in enumerate(states):
            merged=(prior['lane'][0]==2)&(nxt['lane'][0]==1)&prior['is_cav'][0]&prior['present'][0]&nxt['present'][0]
            physical_merges+=int(merged.sum())
            automatic_merges+=int((merged&~im['lanes']['lane_changed'][0,ti,0]).sum())
            prior=nxt
        actual_sec=int(im['imagination_horizon_s'])
        r={'seed':e['seed'],'t':e['warmup_s'],'horizon':actual_sec,'requested_horizon':sec,'population':'t0-cohort, physical exits, entries OFF; no SUMO future-mask censoring',
           'wm_policy_road':float(road[0]),'wm_native_road':float(road[base]),
           'sumo_policy_road':cohort_sumo(e,actual_sec),'sumo_native_road':cohort_sumo(native,actual_sec),
           'wm_cav_speed':float(speeds.mean()),'wm_cav_stop':float((speeds<.1).float().mean()),
           'raw_action_mean':float(cmd.mean()),'raw_action_std':float(cmd.std()),
           'raw_negative_fraction':float((cmd<0).float().mean()),'wm_physical_ramp_merges':physical_merges,
           'wm_automatic_ramp_merges':automatic_merges,
           'wm_return_gain':float(im['returns'][0,0,0]-im['returns'][0,0,base]),
           'wm_native_terminal_value':float(im['values'][0,-1,base]),
           'executed_action_mean':float(effective.mean()),'first_executor_requested_maxerr':initial_err,
           'first_lane_mismatches':lane_err,
           'initial_imagined_action_mean':float(im['actions'][0,0,0][present[0]].mean()),
           'online_value':float(critic(b.state)[0]),'target_initial_value':float(im['values'][0,0,0]),
           'target_terminal_value':float(im['values'][0,-1,0]),'wm_lane_changes':int(im['lanes']['lane_changed'][0,:,0].sum())}
        rows.append(r); print(json.dumps(r),flush=True)
        (OUT/({'enriched':'replay_enriched.json','ownership':'ownership.json'}.get(mode,'replay.json'))).write_text(json.dumps(rows,indent=2))
print('MAX_GPU_BYTES',torch.cuda.max_memory_allocated())
