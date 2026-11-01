"""Integration checks for long MPPI execution and the exact transfer hot-swap."""
import unittest
from unittest.mock import patch
import torch
from .plan_wm import profile_table
from .speed2_transfers import vector_table


class LongProfileTests(unittest.TestCase):
    def test_transfer_table_exact_for_long_profiles_and_dynamic_policy(self):
        actions=[dict(id='p',members=['a','b'],duration_s=300,
            profile=[[.25]*300,[-.5]*300],background_residuals={'a':.1,'b':.2,'c':-.25}),
            dict(id='policy',window=600,residuals={'a':.3}),
            dict(id='MAPPO',window=0,residuals={})]
        for dynamic in (False,True):
            old=profile_table(actions,['a','b','c'],'cpu',420,dynamic)
            new=vector_table(actions,['a','b','c'],'cpu',420,dynamic)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(old,new)))
            self.assertEqual(float(new[0][0,299,0]),.25)
            self.assertEqual(bool(new[1][0,300,0]),not dynamic)
            if not dynamic:self.assertEqual(float(new[0][0,419,0]),float(torch.tensor(.1)))

    def test_member_reversion_preserves_profile_and_background(self):
        from .accel_diagnostic import evaluate
        plan=dict(members=['a','b','c'],duration_s=300,profile=[[.1]*300,[.2]*300,[.3]*300])
        record=dict(plan=plan,policy_residuals={},rows=[])
        captured=[]
        def futures(rt,rec,actions,*args,**kw):
            captured.extend(actions)
            return {}
        with patch('tools.coopmwm.plan_wm.evaluate',futures):evaluate(None,record)
        self.assertEqual(len(captured),5)
        self.assertEqual(captured[2]['members'],['a','c'])
        self.assertEqual(captured[2]['profile'],[plan['profile'][0],plan['profile'][2]])
        self.assertEqual(captured[-1]['profile'],[])


if __name__=='__main__':unittest.main()
