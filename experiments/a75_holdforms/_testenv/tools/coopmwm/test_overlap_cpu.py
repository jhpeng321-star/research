"""Overlap choice and executed-portion bookkeeping without a simulator."""
import unittest
from unittest.mock import patch
import torch
from .overlap import recheck
from .acceleration import Controller,Policy
from tools.coopnative.test_cpu import rows


class OverlapTests(unittest.TestCase):
    def test_all_active_groups_share_one_K8_batch_and_one_bootstrap(self):
        active=[dict(plan=dict(id='old',members=['x','y']),elapsed_s=20)]
        record=dict(state_id='state',committed_plans=active,policy_residuals={'a':.1})
        result=dict(accepted=True,plan=dict(id='new',members=['a','b']),final=dict(H=420))
        def score(rec,acts,k,h):
            self.assertIs(rec,record);self.assertEqual((k,h),(8,420))
            self.assertEqual(acts[0]['committed_plans'],active)
            self.assertEqual(acts[1]['committed_plans'],active)
            self.assertEqual(acts[2]['committed_plans'],[])
            return {a['id']:[dict(team_delay=v)]*8 for a,v in zip(acts,[30.,20.,10.])}
        out=recheck(record,result,score)
        self.assertTrue(out['accepted'])
        self.assertEqual(out['overlap_recheck']['truncated_plan_ids'],['old'])
        self.assertEqual(out['overlap_recheck']['predicted_gain_vs_keep_all'],20.)

    def test_dropping_new_plan_never_truncates_existing_plans(self):
        record=dict(state_id='state',committed_plans=[dict(plan=dict(id='old'),elapsed_s=0)],policy_residuals={})
        result=dict(accepted=True,plan=dict(id='new'),final=dict(H=300))
        out=recheck(record,result,lambda r,a,k,h:{p['id']:[dict(team_delay=v)]*8 for p,v in zip(a,[20.,0.,10.])})
        self.assertFalse(out['accepted']);self.assertEqual(out['overlap_recheck']['truncated_plan_ids'],[])

    def test_controller_truncates_only_after_execution_and_records_executed_portion(self):
        observed=rows();ids=[r['vid'] for r in observed if r['is_cav']]
        self.assertGreaterEqual(len(ids),2)
        old=dict(id='old',members=[ids[0]],family='MPPI',duration_s=60,knots=[[-1.,-1.]])
        new=dict(id='new',members=[ids[1]],family='MPPI',duration_s=60,knots=[[.2,.2]])
        proposal=dict(accepted=True,plan=new,option='MPPI',
            behaviour=dict(route='wm',regime='ordinary',sample_probability=1.),
            overlap_recheck=dict(truncated_plan_ids=['old'],predicted_gain_vs_keep_all=10.))
        controller=Controller(Policy(),torch.Generator().manual_seed(1),guide=lambda *a:proposal)
        controller.active_plans=[dict(plan=old,start_s=300,behaviour={})]
        with patch('tools.coopmwm.mppi_search.group_components',return_value=[dict(members=[ids[1]],size=2)]):
            controller.apply(observed,{v:0. for v in ids},320)
        sample=controller.events[-1]['sample']
        self.assertEqual(sample['plan_truncations'][0]['executed_duration_s'],20)
        self.assertEqual([c['plan']['id'] for c in controller.active_plans],['new'])
        self.assertNotIn(ids[0],sample['absolute_accels'])


if __name__=='__main__':unittest.main()
