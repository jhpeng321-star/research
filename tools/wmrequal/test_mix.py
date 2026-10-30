"""CPU-only WM forwards: birth persistence, default parity and replay continuation."""
import copy
import importlib.util
import os
from pathlib import Path
import unittest

import torch
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.physics.config import PhysicsConfig
from tools.errsrc.inference import LearnedMix


class Births:
    def __init__(self):
        self.tick=0

    def insert(self,state):
        new=torch.zeros_like(state['present'])
        if self.tick==1:
            new[:,3]=True
            state['present'][:,3]=True
            state['s_m'][:,3]=-50.
            state['v_mps'][:,3]=6.
            state['lane'][:,3]=0
        self.tick+=1
        return new


def scene():
    torch.manual_seed(701)
    model=WmSModel(ModelSpec(history_hidden=8,k_hidden=8,rollout_mode='hybrid',
        lane_model='hazard_v5',merge_decision='first_passage',lane_decision='sample',
        merge_passage_law='hybrid'),physics=PhysicsConfig(exit_at_road_end=False)).eval()
    state=dict(s_m=torch.tensor([[100.,50.,0.,-50.]]),v_mps=torch.full((1,4),6.),
        a_mps2=torch.zeros(1,4),lane=torch.tensor([[0,1,0,0]]),
        present=torch.tensor([[True,True,True,False]]),is_cav=torch.zeros(1,4,dtype=torch.bool),
        length_m=torch.full((1,4),5.))
    _,embedding,params=model.encode_drivers(state)
    observed=torch.tensor([[[1],[1],[0],[0]]],dtype=torch.bool)
    params={k:v.clone() for k,v in params.items()}
    for p in params.values():
        p[:,0]*=.9
        p[:,1]*=1.1
    return model,state,embedding,params,observed


class MixTest(unittest.TestCase):
    def test_birth_parameters_persist_and_pair_keys_ignore_birth_tick(self):
        model,state,embedding,params,observed=scene()
        mix=LearnedMix(params,observed,71,[0])
        seen=[]
        original=model.step
        def step(state,**kw):
            seen.append({k:v.clone() for k,v in kw['driver_params'].items()})
            return original(state,**kw)
        model.step=step
        with torch.no_grad():
            model.rollout(state,n_steps=4,driver_params=params,driver_embedding=embedding,
                          entries=Births(),entrant_driver_sampler=mix,lane_seed=9)
        self.assertEqual(mix.assigned,[1])
        for name in params:
            self.assertTrue(torch.equal(seen[1][name][:,3],seen[3][name][:,3]))
            self.assertIn(float(seen[1][name][0,3]),params[name][0,:2].tolist())
        p={'v0':torch.tensor([[20.,30.,1.],[20.,30.,1.]])}
        o=torch.tensor([[[1],[1],[0]],[[1],[1],[0]]],dtype=torch.bool)
        sampler=LearnedMix(p,o,31,[0,0])
        rng=torch.get_rng_state()
        st={'lane':torch.zeros(2,3,dtype=torch.long)}
        first=sampler(st,torch.tensor([[0,0,1],[0,0,0]],dtype=torch.bool),p)
        second=sampler(st,torch.tensor([[0,0,0],[0,0,1]],dtype=torch.bool),first)
        self.assertEqual(first['v0'][0,2],second['v0'][1,2])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_default_off_bitwise_rng_parity(self):
        model,state,embedding,params,_=scene()
        results=[];rngs=[]
        for kw in ({},{'entrant_driver_sampler':None}):
            torch.manual_seed(99)
            with torch.no_grad():
                results.append(model.rollout(copy.deepcopy(state),n_steps=4,driver_params=params,
                    driver_embedding=embedding,entries=Births(),**kw))
            rngs.append(torch.get_rng_state())
        for key in ('positions','speeds','lanes','masks','accel'):
            self.assertTrue(torch.equal(results[0][key],results[1][key]),key)
        self.assertTrue(torch.equal(*rngs))

    def test_segmented_forward_preserves_birth_state(self):
        path=Path(os.environ['WMREQUAL_MEMORY_CHECK'])
        spec=importlib.util.spec_from_file_location('wms.control.requal_memory',path)
        mod=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        outputs=[];counts=[]
        for segment in (0,2):
            model,state,embedding,params,observed=scene()
            mix=LearnedMix(params,observed,71,[0])
            kw=dict(driver_params=params,driver_embedding=embedding,entries=Births(),
                    entrant_driver_sampler=mix,lane_uniform=torch.full((1,4,4),.5))
            with torch.no_grad():
                out=mod.segmented_rollout(model,state,4,segment,None,kw) if segment else model.rollout(state,n_steps=4,**kw)
            outputs.append(out['speeds']);counts.append(mix.assigned)
        self.assertTrue(torch.equal(*outputs))
        self.assertEqual(counts,[[1],[1]])


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
