"""CPU gates: real parity boundary, correlated likelihood and committed profiles."""
import copy
import unittest
import numpy as np
import torch
from tools.coopnative.test_cpu import rows
from .acceleration import Policy, Controller, requests, encode, PHI
from .plans import acceleration_at, validate, thompson, population
from .plan_wm import profile_table, profile_apply


class AccelerationCPU(unittest.TestCase):
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
