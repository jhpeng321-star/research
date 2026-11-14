"""MAIN02:01 learning semantics: no scoring of committed plan actions."""
import copy
import unittest
from unittest.mock import patch
import torch
from tools.coopnative.test_cpu import rows
from .acceleration import Policy,encode
from . import accel_learner as learner


class AWRCPU(unittest.TestCase):
    def fixture(self,rewards=(1.,1.,-1.)):
        p=Policy();g=torch.Generator().manual_seed(19)
        sample,_=p.sample(encode(rows()),g);steps=[]
        for i,reward in enumerate(rewards):
            s=copy.deepcopy(sample);s['value']=0.
            if i<2:
                s['behaviour']=dict(executed=True,plan_start=i==0,plan_id='p',duration_s=2,
                    proposal_members=s['context']['members'],route='wm' if i==0 else 'plan',beta=0.)
                s['action']=[[1.] for _ in s['action']]
            else:s['behaviour']=dict(route='wm',executed=False,sample_probability=.95)
            steps.append(dict(time_s=300+i,reward=reward*learner.UNIT,sample=s))
        return p,g,dict(steps=steps)

    def config(self):return dict(temperature=1.,weight_clip=20.,coefficient=1.)

    def test_awr_weight_positive_exponential_and_clip(self):
        self.assertEqual(learner.awr_weight(-1.,self.config()),0.)
        self.assertEqual(learner.awr_weight(0.,self.config()),0.)
        self.assertAlmostEqual(learner.awr_weight(1.,self.config()),2.718281828)
        self.assertAlmostEqual(learner.awr_weight(1000.,self.config()),20.)

    def test_completed_plan_targets_use_actual_sumo_and_frozen_critic(self):
        _,_,ep=self.fixture();ep['steps'][0]['sample']['value']=.5;ep['steps'][2]['sample']['value']=.25
        targets=learner.plan_targets(ep,self.config())
        self.assertAlmostEqual(targets[0]['advantage'],1.+learner.GAMMA+learner.GAMMA**2*.25-.5)
        self.assertEqual(targets[0],targets[1]);self.assertNotIn(2,targets)

    def test_negative_and_incomplete_plans_never_imitated(self):
        p,_,ep=self.fixture((-1.,-1.,0.));targets=learner.plan_targets(ep,self.config())
        self.assertEqual(targets[0]['weight'],0.)
        self.assertEqual(float(learner.imitation(p,ep['steps'][0]['sample'],targets[0]['weight'])),0.)
        ep['steps']=ep['steps'][:1];self.assertEqual(learner.plan_targets(ep,self.config()),{})

    def test_no_plan_scores_or_ratios_critic_all_steps_rejected_query_is_own(self):
        p,g,ep=self.fixture();original=p.score;calls=[]
        def score(s):
            self.assertTrue(learner.own_step(s));calls.append(s);return original(s)
        with patch.object(p,'score',side_effect=score):
            m=learner.update(p,torch.optim.Adam(p.parameters(),lr=3e-4),[ep],g,epochs=1,awr=self.config())
        self.assertEqual(len(calls),1);self.assertEqual(m[0]['own_policy_steps'],1)
        self.assertEqual(m[0]['critic_steps'],3);self.assertEqual(m[0]['awr_positive_steps'],2)
        self.assertNotEqual(m[0]['imitation'],0.)

    def test_all_plan_negative_batch_updates_critic_without_actor_scoring(self):
        p,g,ep=self.fixture((-1.,-1.));before=p.critic[-1].weight.detach().clone()
        with patch.object(p,'score',side_effect=AssertionError('plan score forbidden')):
            m=learner.update(p,torch.optim.Adam(p.parameters(),lr=3e-4),[ep],g,epochs=1,awr=self.config())
        self.assertEqual(m[0]['own_policy_steps'],0);self.assertEqual(m[0]['critic_steps'],2)
        self.assertEqual(m[0]['imitation'],0.);self.assertFalse(torch.equal(before,p.critic[-1].weight))

    def test_ratio_ignores_query_mass_and_plan_ratio_rejected(self):
        _,_,ep=self.fixture();own=ep['steps'][2]['sample']
        self.assertEqual(float(learner.ratio(torch.tensor(own['logp']),own)),1.)
        with self.assertRaises(ValueError):learner.ratio(torch.tensor(0.),ep['steps'][0]['sample'])

    def test_temperature_only_baseline_interval_data(self):
        records=[dict(world=dict(seed=986072+i),metrics={'600':dict(team_delay_vs=3.),'900':dict(team_delay_vs=x+3.)}) for i,x in enumerate([3.,4.])]
        self.assertAlmostEqual(learner.awr_config(records)['temperature'],(12.5**.5)/learner.UNIT)


if __name__=='__main__':unittest.main()
