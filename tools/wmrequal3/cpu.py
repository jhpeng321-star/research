"""CPU regression, actual validation census and hash-bound lane-contract audit."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import sys
import json
import time
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from run import bootstrap, OWN, OLD, RT


def main():
    begin = time.monotonic()
    bad=('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth','vtype')
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            n=os.fsdecode(args[0]).lower()
            if any(x in n for x in bad) or n.endswith('.xml'):raise RuntimeError('prohibited input')
    sys.addaudithook(guard)
    score=bootstrap()
    import torch
    import numpy as np
    torch.set_num_threads(2)
    assert not torch.cuda.is_available()
    from wms.train.cav_drive import TimelineControllers
    from wms.control.entrant import EntrantControllers, feedback_from_specs
    from wms.control.policy import load_policy_components
    from wms.model.cav import WM_MERGE_KEY,WM_LANE_TARGET_KEY
    from wms.physics.config import PhysicsConfig
    from tools.wmrequal.native_lanes import repair, NativeTimeline
    names=['native','mappo','ours_c1','coopoff_c1']
    design=json.loads((score.BASE/'overnight20261028/W5_COMBINED/DESIGN.json').read_text())
    specs={a:design['controllers'][a] for a in names[1:]}
    contracts={}
    for a,spec in specs.items():
        p=Path(spec['checkpoint']);assert hashlib.sha256(p.read_bytes()).hexdigest()==spec['sha256']
        policy,enc,rules,payload=load_policy_components(p)
        contracts[a]=dict(**spec,format=payload['format'],lane_control=payload.get('lane_control'),
            has_lane_head=bool(policy.has_lane_head),residual_on_native=bool(policy.residual_on_native))
    assert all(contracts[a]['lane_control']=='native' and not contracts[a]['has_lane_head'] for a in names[2:])
    assert contracts['mappo']['has_lane_head']
    plant=SimpleNamespace(physics=PhysicsConfig(),physics_accel=lambda st:torch.zeros_like(st['s_m']))
    e3=score.bootstrap_runtime()
    real=json.loads((OWN/'REAL.json').read_text())['panels']['validation']
    source,indexes=score.source_for('validation',real)
    exposure=[]
    for gi in indexes:
        key,ref,arms=e3.group_rows(source,gi)
        recs=source.group_records(gi,[ref]+[r for _,r in arms],base_row=0)
        for arm,rec in zip(['native']+[a for a,_ in arms],recs):
            states,present,cav=(np.asarray(rec[k]) for k in ('states','present','is_cav'))
            lanes=states[...,1:4].argmax(-1)
            ramp=present[0].astype(bool)&cav[0].astype(bool)&(lanes[0]==2)
            merged=ramp&(present.astype(bool)&(lanes==1)).any(0)
            st=dict(s_m=torch.from_numpy(states[0,:,0].copy())[None],
                v_mps=torch.from_numpy(states[0,:,4].copy())[None],
                a_mps2=torch.from_numpy(states[0,:,5].copy())[None],lane=torch.from_numpy(lanes[0].copy())[None],
                length_m=torch.from_numpy(states[0,:,6].copy())[None],
                present=torch.from_numpy(present[0].astype(bool))[None],is_cav=torch.from_numpy(cav[0].astype(bool))[None])
            st[WM_LANE_TARGET_KEY]=st['lane'].clone()
            tl={k:torch.as_tensor(v)[None] for k,v in rec['action_timeline'].items()}
            old=TimelineControllers(plant,tl)
            assert not hasattr(old,'native_lane_mask')
            old_motion=old(dict(st))
            fixed=repair(TimelineControllers(plant,tl))
            motion=fixed(st)
            np.testing.assert_array_equal(st[WM_MERGE_KEY][0].numpy(),ramp)
            assert torch.equal(motion[..., [0,2,3]],old_motion[..., [0,2,3]])
            exposure.append(dict(pair_key=key,arm=arm,original_ramp_cavs=int(ramp.sum()),
                recorded_merged=int(merged.sum()),corrected_delegated=int(st[WM_MERGE_KEY].sum())))
    totals={a:{k:sum(r[k] for r in exposure if r['arm']==a) for k in
        ('original_ramp_cavs','recorded_merged','corrected_delegated')} for a in names}
    assert all(v==dict(original_ramp_cavs=44,recorded_merged=44,corrected_delegated=44) for v in totals.values())
    checks=[]
    timeline=dict(controller=torch.zeros(4,4,3,dtype=torch.long),
        acceleration_mps2=torch.zeros(4,4,3),target_speed_mps=torch.zeros(4,4,3))
    for entrant in (False,True):
        st=dict(s_m=torch.tensor([[-20.,150.,250.]]).expand(4,-1).clone(),
            v_mps=torch.full((4,3),10.),a_mps2=torch.zeros(4,3),
            lane=torch.tensor([[2,1,0]]).expand(4,-1).clone(),length_m=torch.full((4,3),5.),
            present=torch.ones(4,3,dtype=torch.bool),is_cav=torch.ones(4,3,dtype=torch.bool),
            entrant=torch.full((4,3),entrant,dtype=torch.bool),
            inferred_driver_parameters=torch.tensor([31.29,1.2,2.,1.2,2.,.3,.1,6.,1.]).expand(4,3,-1).clone(),
            inferred_idm_delta=4.,secure_gap_T_s=torch.full((4,3),1.2),secure_gap_b_mps2=torch.full((4,3),2.))
        st[WM_LANE_TARGET_KEY]=st['lane'].clone()
        st[WM_LANE_TARGET_KEY][:,1]=0
        c=repair(EntrantControllers(TimelineControllers(plant,timeline),
            feedback_from_specs(specs,names,plant,timeline,3,'cpu')))
        mask=c.native_lane_mask(st)
        with torch.no_grad():motion=c(st)
        expected=torch.tensor([True,not entrant,True,True])
        assert torch.equal(mask[:,0],expected)
        assert st[WM_MERGE_KEY][:,0].tolist()==[True]*4
        assert motion[[0,2,3],1,1].tolist()==[0.]*3
        if not entrant:assert motion[:,1,1].tolist()==[0.]*4
        checks.append(dict(cohort='entrant' if entrant else 'original',native_lane_mask=mask[:,0].tolist(),
            ramp_delegation=st[WM_MERGE_KEY][:,0].tolist(),mainline_target=motion[:,1,1].tolist()))
    # Explicit policy lane crossings are not wrapped or changed. They own their
    # output; only the original and native fallback timelines acquire ownership.
    class Crossing:
        def native_lane_mask(self,st):return torch.zeros_like(st['present'])
        def __call__(self,st):
            st[WM_MERGE_KEY]=torch.zeros_like(st['present'])
            return torch.stack((st['s_m'],torch.ones_like(st['s_m']),st['v_mps'],st['a_mps2']),-1)
    policy=Crossing()
    c=repair(EntrantControllers(TimelineControllers(plant,timeline),policy))
    assert c.feedback is policy
    assert not c.native_lane_mask(st).any()
    assert torch.equal(c(st)[...,1],torch.ones_like(st['s_m']))
    assert not st[WM_MERGE_KEY].any()
    # Integration: an original ramp CAV gets the very same first-passage
    # opportunities and committed crossing as the HDV twin, not an auto-merge.
    from wms.model import ModelSpec,WmSModel
    from wms.model.driver_encoder import PARAMETER_NAMES
    torch.manual_seed(17)
    m=WmSModel(ModelSpec(rollout_mode='driver_params',lane_model='hazard_v5',
        merge_decision='first_passage',lane_decision='sample'),physics=PhysicsConfig()).double().eval()
    def scene(cav):
        s=torch.tensor([[30.,5.,90.]],dtype=torch.float64)
        return dict(s_m=s,v_mps=s.new_tensor([[5.,2.,8.]]),a_mps2=torch.zeros_like(s),
            length_m=torch.full_like(s,5.),lane=torch.tensor([[2,1,1]]),
            present=torch.ones_like(s,dtype=torch.bool),is_cav=torch.tensor([[cav,False,False]]))
    def forward(st,**kw):
        s,lane=st['s_m'],st['lane']
        logits=s.new_full((*s.shape,3),-20.);logits.scatter_(-1,lane[...,None],20.)
        vals=dict(v0=31.29,time_headway=1.2,min_gap=2.,accel=1.2,decel=2.)
        return dict(accel_mps2=torch.zeros_like(s),lane_logits=logits,lane_feasible=torch.ones_like(logits,dtype=torch.bool),
            hdv_mask=st['present']&~st['is_cav'],lane_decider_mask=st['present'],
            driver_parameters=torch.stack([torch.full_like(s,vals.get(n,1.)) for n in PARAMETER_NAMES],-1),
            lane_age=torch.zeros_like(lane),lane_sample=torch.zeros_like(st['present']))
    one={k:v[:1] for k,v in timeline.items()}
    with patch.object(m,'forward',side_effect=forward),torch.no_grad():
        a=m.step(scene(True),cav_controller=repair(TimelineControllers(m,one)),lane_uniform=torch.full((1,3),1e-6,dtype=torch.float64))
        b=m.step(scene(False),lane_uniform=torch.full((1,3),1e-6,dtype=torch.float64))
    for key in ('next_lane','merge_pivot_eligible','merge_pivot_committed','merge_pivot_c'):
        assert torch.equal(a[key],b[key]),key
    assert a['next_lane'][0,0]==1
    # Frozen scorer OFF vs the new scorer OFF/default, full callback+rollout path.
    spec=importlib.util.spec_from_file_location('frozen_score',OLD/'tools/wmrequal/score.py')
    frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
    _,settings,model=e3.load_for_scoring(RT/'configs/c4_wm_loop14_r16_ft.toml',score.BASE/'restart20261026/arm/loop/wm/c02/wm_s.pt')
    from dataclasses import replace
    settings=replace(settings,seed=140014,entrant_control=True,entrant_controllers=specs,
        entry_queue_state=True,entry_pad_auto=True,entry_pad_max=1024,entry_insert_rule='sumo_like_v2')
    model.eval().requires_grad_(False);model.cav_exec_cap='idm'
    score.HORIZONS=frozen.HORIZONS=(2,)
    args=(model,settings,e3,source,indexes[:1],real,torch.device('cpu'),4,True)
    x=frozen.score_batch(*args)
    y=score.score_batch(*args)
    z=score.score_batch(*args,native_lanes=False)
    assert x==y==z
    result=dict(schema='wmrequal3_cpu/1',seconds=time.monotonic()-begin,cuda_visible='',threads=2,
        exact_off_frozen_rows=len(x),exact_off=True,first_passage_hdv_twin_equal=True,
        explicit_policy_crossing_preserved=True,contracts=contracts,totals=totals,
        ownership_checks=checks,exposure=exposure)
    with (OWN/'CPU.json').open('x') as f:json.dump(result,f,indent=1,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!='exposure'},indent=1))


if __name__=='__main__':main()
