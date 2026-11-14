import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from . import final_discovery as f


class DiscoveryTests(unittest.TestCase):
    def test_actual_component_rule_keeps_cross_lane_ramp_members(self):
        rows=[dict(vid=v,s_m=-900+30*i,v_mps=15.,lane=i%3,is_cav=True,length_m=5.)
              for i,v in enumerate(('a','b','c','d'))]
        group=f.group_at(dict(group=['b']),rows)
        self.assertEqual(set(group['members']),{'a','b','c','d'})
        self.assertIsNone(f.group_at(dict(group=['unsensed']),rows))

    def test_whole_group_precharge_scales_through_six_members(self):
        self.assertEqual(f.whole_group_cost(400,2),4000)
        self.assertEqual(f.whole_group_cost(400,6),8000)
        with self.assertRaises(AssertionError):f.whole_group_cost(400,7)

    def test_larger_groups_use_detector_members_not_legacy_pair_gate(self):
        ep=dict(steps=[dict(time_s=400,rows=[])])
        group=dict(members=['a','b','c','d'],size=4)
        with patch.object(f,'group_at',return_value=group):
            t,plan=f.observed_plan(ep,dict(start_s=400,end_s=541),12)
        self.assertEqual(t,400)
        self.assertEqual(plan['members'],group['members'])
        self.assertEqual(plan['duration_s'],150)

    def test_noneligible_time_does_not_open_group_geometry(self):
        with patch.object(f,'group_at',side_effect=AssertionError('not eligible')):
            self.assertIsNone(f.observed_plan(dict(steps=[]),dict(start_s=601),12))

    def test_recorded_raw_actions_are_replayed_and_original_tape_preserved(self):
        ep=dict(steps=[dict(time_s=t,requests=dict(a=dict(requested_raw=-1.2)),
                           commands=dict(a=.4,b=.7),sample=dict(own_action='unchanged')) for t in (399,400,401,430)])
        plan=dict(members=['a','b'],duration_s=30)
        twin,step=f.twin_episode(ep,400,plan)
        self.assertEqual(step['sample']['absolute_accels'],dict(a=-1.2,b=.7))
        self.assertNotIn('absolute_accels',twin['steps'][0]['sample'])
        self.assertNotIn('absolute_accels',twin['steps'][-1]['sample'])
        self.assertTrue(all('absolute_accels' not in s['sample'] for s in ep['steps']))

    def test_waiting_gate_needs_all_endpoints_and_completed_final_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch('tools.coopmwm.final_runtime.endpoints_ready',return_value=False):
                self.assertFalse(f.opening_ready(root))
            with patch('tools.coopmwm.final_runtime.endpoints_ready',return_value=True):
                self.assertFalse(f.opening_ready(root))
                (root/'FINAL_OUTCOMES_OPENED.json').touch()
                self.assertFalse(f.opening_ready(root))
                (root/'FINAL_REGISTERED_EVALUATION_COMPLETE.json').touch()
                self.assertTrue(f.opening_ready(root))


if __name__=='__main__':unittest.main()
