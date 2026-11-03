import unittest
from .v2v_launch_quote import calculate, select


class QuoteTests(unittest.TestCase):
    def test_exact_schedule_no_A_extra_or_paired(self):
        q=calculate(dict(B=[dict(end_s=720)]),
            dict(B=[dict(end_s=720,members=2)],R=[dict(end_s=780,members=2)]))
        self.assertEqual(q['phase_costs']['v2v_train'],2073600)
        self.assertEqual(q['phase_costs']['v2v_lessons'],4*720)
        self.assertEqual(q['phase_costs']['v2v_isolated'],4*5*720+2*5*780)
        self.assertEqual(q['diagnostics']['A'],dict(lessons=[],isolated=[]))
        self.assertFalse(q['final_paired_branches'])
        self.assertEqual(q['arms'],['A-v2v','B-v2v','R-v2v'])

    def test_ordered_cost_fallbacks_never_drop_an_arm(self):
        lessons=dict(B=[])
        for count in range(1,40):
            cooperative=dict(B=[dict(end_s=1200,members=6)]*count,R=[])
            r=select(lessons,cooperative)
            for attempt in r['attempts']:
                self.assertEqual(len(attempt['arms']),3)
            if len(r['attempts'])==3 and r['selected']:
                self.assertEqual(r['selected']['dropped'],['R-v2v u8 test','B-v2v u6/u18 lesson and isolated checks'])
                break
        else:self.fail('fixture did not exercise both approved fallbacks')

    def test_over_ceiling_stops_without_spend(self):
        r=select(dict(B=[]),dict(B=[dict(end_s=1200,members=6)]*100,R=[]))
        self.assertIsNone(r['selected'])
        self.assertFalse(r['launch_cost_gate'])
        self.assertGreater(r['shortfall_sim_s'],0)
        self.assertEqual(r['new_sim_s'],0)


if __name__=='__main__':unittest.main()
