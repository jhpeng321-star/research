"""MAIN02:43 regression: disjoint plans persist in plant and every WM candidate."""
import unittest
from unittest.mock import patch
import torch
from .acceleration import Controller,Policy
from .committed import background,validate
from .plan_wm import profile_table,profile_apply
from .speed2_transfers import vector_table
from .test_acceleration_cpu import rows


def plan(pid,vs,a=-1.,duration=30):
    return dict(id=pid,members=vs,duration_s=duration,knots=[[a]*4 for _ in vs],family='MPPI')


class CommittedTests(unittest.TestCase):
    def test_other_plan_remaining_profile_overrides_all_comparators_then_releases(self):
        p=plan('other',['c','d'],duration=30)
        cs=[dict(plan=p,elapsed_s=20)]
        acts=[dict(id='policy',window=300,residuals={'c':.3,'d':.4}),dict(id='MAPPO',window=0),
              dict(plan('new',['a','b'],a=.5),background_residuals={'c':.3,'d':.4})]
        acts=[dict(a,committed_plans=cs) for a in acts]
        for dynamic in (False,True):
            ref=profile_table(acts,['a','b','c','d'],'cpu',horizon=420,dynamic_policy=dynamic)
            fast=vector_table(acts,['a','b','c','d'],'cpu',horizon=420,dynamic_policy=dynamic)
            for x,y in zip(ref,fast):self.assertTrue(torch.equal(x,y))
            raw=torch.ones(3,4)*.7
            self.assertTrue(torch.equal(profile_apply(raw,0,ref,(-3.,2.6))[:,2:],torch.ones(3,2)*-1))
            # Remaining 10 seconds, not a fresh 30-second commitment.
            self.assertFalse(ref[1][:,10:,2:].any() if dynamic else ref[2][:,0:10,2:].any())
            self.assertEqual(background(cs,10),{})
            if dynamic:self.assertTrue(torch.equal(profile_apply(raw,20,ref,(-3.,2.6))[:,2:],raw[:,2:]))

    def test_overlap_rejected(self):
        with self.assertRaises(ValueError):validate([dict(plan=plan('p',['a','b']),elapsed_s=1)],['b','c'])

    def test_two_disjoint_plans_execute_and_expire_independently(self):
        base=[dict(r,s_m=-100.-20.*i) for i,r in enumerate([r for r in rows() if r['is_cav']][:2])]
        sensor=list(base)
        self.assertEqual(len(base),2)
        other=[]
        for r in base:other.append(dict(r,vid='far_'+r['vid'],s_m=r['s_m']-500))
        sensor+=other;cmd={r['vid']:0. for r in sensor if r['is_cav']}
        calls=[]
        def guide(rs,commands,t,s):
            from .mppi_search import group_components
            from .committed import members
            if t%10:return None
            gs=group_components(rs,exclude=members(s['committed_plans']))
            if not gs:return None
            calls.append((t,len(s['committed_plans'])))
            p=plan(str(t),gs[0]['members'],a=-1. if t==300 else -.5)
            return dict(accepted=True,plan=p,option='MPPI',behaviour=dict(route='wm',regime='ordinary',executed=False))
        c=Controller(Policy(),torch.Generator().manual_seed(2),guide=guide)
        for t in range(300,341):
            c.apply(sensor,cmd,t)
            if t==310:
                self.assertEqual(c.events[-1]['simultaneously_active_plans'],2)
                self.assertEqual(len(c.events[-1]['sample']['absolute_accels']),4)
                self.assertEqual(len(c.events[-1]['sample']['plan_executions']),2)
            if t==330:self.assertNotIn('300',[x['plan']['id'] for x in c.active_plans])
        self.assertEqual(calls[:2],[(300,0),(310,1)])


if __name__=='__main__':unittest.main()
