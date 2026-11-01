"""CPU gates: real parity boundary, correlated likelihood and committed profiles."""
import copy
import unittest
import numpy as np
import torch
from pathlib import Path
import tempfile
from tools.coopnative.test_cpu import rows
from .acceleration import Policy, Controller, requests, encode, PHI
from .plans import acceleration_at, validate, thompson, population
from .plan_wm import profile_table, profile_apply


class AccelerationCPU(unittest.TestCase):
    def test_acceleration_modules_import_and_long_credit(self):
        from . import accel_runner,accel_plant,plan_bridge,plan_service,plan_client
        from .accel_learner import GAMMA,LAMBDA
        self.assertGreaterEqual(-1/np.log(GAMMA*LAMBDA),300.)

    def test_checkpoint_restores_acceleration_optimizer_and_AR_generator(self):
        from .accel_learner import save,load,update
        p=Policy();opt=torch.optim.Adam(p.parameters(),lr=3e-4);g=torch.Generator().manual_seed(1)
        sample,_=p.sample(encode(rows()),g)
        ep=dict(steps=[dict(sample=sample,reward=-123.)])
        update(p,opt,[ep],g,epochs=1)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'accel.pt';save(path,p,opt,g,1,811640)
            restored,other,rg,_=load(path)
            update(p,opt,[ep],g,epochs=1);update(restored,other,[ep],rg,epochs=1)
            for a,b in zip(p.parameters(),restored.parameters()):self.assertTrue(torch.equal(a,b))

    def test_singular_plan_ratio_never_invents_gaussian_mass(self):
        from .accel_learner import ratio,imitation
        p=Policy();s,_=p.sample(encode(rows()),torch.Generator().manual_seed(9))
        lp,_=p.score(s);s['behaviour']=dict(route='wm',executed=True,beta=1.,proposal_members=s['context']['members'])
        self.assertEqual(float(ratio(lp,s)),0.)
        s['action']=[[1.] for _ in s['action']]
        loss=imitation(p,s);loss.backward()
        self.assertGreater(float(p.actor[-1].bias.grad.abs().sum()),0.)
        s['behaviour']=dict(route='policy',sample_probability=.25)
        self.assertAlmostEqual(float(ratio(lp,s)),1/.75,places=5)

    def test_real_frozen_mappo_acceleration_executor_parity(self):
        import os
        from wms.control.policy import load_policy_fn
        from tools.coopnative.test_cpu import state,env
        from tools.coopfree.execution import inject
        a,b=[load_policy_fn(os.environ['COOPMWM_MAPPO'],device='cpu') for _ in range(2)]
        a.reset();b.reset();sensor=rows();e=env(sensor)
        s=state(sensor);s['a_mps2']=torch.zeros_like(s['v_mps'])
        controller=Controller(Policy(),torch.Generator().manual_seed(1),True)
        for t in range(300,310):
            expected,_=a(s,with_lanes=True)
            with inject(b,e,sensor,controller,t,[]):actual,_=b(s,with_lanes=True)
            self.assertEqual(expected.tobytes(),actual.tobytes())
        self.assertEqual(len(controller.events),10)

    def test_96_panel_roles_dual_seed_guards_and_full_budget(self):
        from .accel_protocol import worlds,budget,guard,install_seed_scope
        panel=worlds('final_test');self.assertEqual(len(panel),96)
        self.assertEqual(len({w['seed'] for w in panel}),96)
        for c in range(12):self.assertEqual(sum(w['cell']==c for w in panel),8)
        install_seed_scope()
        for w in panel:guard(w,'final_test')
        with self.assertRaises(ValueError):guard(panel[-1],'train')
        for seed in (964000,962001,977001):
            with self.assertRaises(ValueError):guard(dict(panel[0],seed=seed),'final_test')
        b=budget();self.assertEqual(b['allocated']+b['unallocated'],6000000)
        self.assertGreater(b['with_seed2_total'],6000000)

    def test_zero_update_is_exact_mappo_before_unchanged_safety(self):
        sensor=rows();commands={r['vid']:2.59707 for r in sensor if r['is_cav']}
        p=Policy();s,_=p.sample(encode(sensor),torch.Generator().manual_seed(1),True)
        self.assertEqual(requests(s,commands),{})
        self.assertTrue(torch.equal(p.broadcast[-1].weight,torch.zeros_like(p.broadcast[-1].weight)))

    def test_conditional_ar_density_and_no_leakage(self):
        p=Policy();g=torch.Generator().manual_seed(20);context=encode(rows())
        first,m=p.sample(context,g);second,_=p.sample(context,g,memory=m)
        self.assertAlmostEqual(float(p.score(second)[0]),second['logp'],places=5)
        self.assertTrue(any(x[0]!=0 for x in second['previous_noise']))
        with self.assertRaises(ValueError):encode([dict(rows()[0],world_id=1)])
        self.assertGreater(1/(-np.log(PHI)),4.99)

    def test_profiles_change_futures_and_release_at_duration(self):
        plan=dict(id='x',members=['a','b'],duration_s=60,knots=[[-2.]*7,[1.]*7])
        validate(plan,(-3.,2.6));table=profile_table([plan],['a','b'],'cpu')
        x=torch.zeros((1,2));v=torch.tensor([[20.,20.]]);baseline=v.clone();position=v*0
        for tick in range(120):
            a=profile_apply(x,tick,table,(-3.,2.6));v=(v+.5*a).clamp_min(0);position+=.5*v
        self.assertGreater(float((v-baseline).abs().max()),1.)
        self.assertNotEqual(float(position[0,0]),float(position[0,1]))
        self.assertEqual(acceleration_at(plan,60),{})
        self.assertTrue(torch.equal(profile_apply(x,120,table,(-3.,2.6)),x))

    def test_policy_residual_and_plan_absolute_raw_are_distinct(self):
        acts=[dict(id='policy',window=60,residuals={'a':.5}),dict(id='MAPPO',window=0),
            dict(id='p',members=['a','b'],knots=[[-1.]*7,[2.]*7],duration_s=20)]
        out=profile_apply(torch.full((3,2),.7),0,profile_table(acts,['a','b'],'cpu'),(-3.,2.6))
        self.assertTrue(torch.allclose(out,torch.tensor([[1.2,.7],[.7,.7],[-1.,2.]])))

    def test_thompson_uses_same_crn_indices_and_beats_both_comparators(self):
        plans=[dict(id='p',family='O1',origin='sampled')]
        fs={a:[dict(team_delay=x) for x in values] for a,values in
            dict(policy=[20.]*8,MAPPO=[10.]*8,p=[11.]*8).items()}
        r=thompson({},plans,fs,np.random.default_rng(1));self.assertFalse(r['accepted'])
        fs['p']=[dict(team_delay=0.)]*8
        r=thompson({},plans,fs,np.random.default_rng(1));self.assertTrue(r['accepted'])
        self.assertEqual(r['predicted_rank'],1);self.assertEqual(len(r['bootstrap_indices']),8)

    def test_plan_is_committed_and_replay_is_immutable(self):
        sensor=rows();members=[r['vid'] for r in sensor if r['is_cav']][:2]
        self.assertEqual(len(members),2)
        plan=dict(id='p',members=members,duration_s=20,knots=[[-1.]*7,[1.]*7],family='O1')
        calls=[]
        def guide(rs,cmd,t,s):
            calls.append(t)
            return dict(accepted=True,plan=plan,option='O1',behaviour=dict(route='wm',regime='ordinary',executed=False))
        c=Controller(Policy(),torch.Generator().manual_seed(2),guide=guide)
        cmd={r['vid']:0. for r in sensor if r['is_cav']}
        for t in range(300,320):
            result=c.apply(sensor,cmd,t)
            self.assertEqual(result[members[0]]['requested_raw'],-1.)
        self.assertEqual(calls,[300])
        saved=copy.deepcopy(c.events[0]['sample']);snapshot=copy.deepcopy(saved)
        r=Controller(Policy(),None,replay={'300':saved})
        r.apply(sensor,cmd,300);self.assertEqual(saved,snapshot)

if __name__=='__main__':unittest.main()
