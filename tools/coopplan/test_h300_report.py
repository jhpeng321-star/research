"""Meaningful wall opportunity estimand: skipped is distinct from rejected."""
import unittest
from .h300_report import wall_counts,describe_wall


class WallReportTests(unittest.TestCase):
    def test_available_evaluated_chosen_and_skipped_have_distinct_denominators(self):
        def event(t,available,delays,chosen='O0'):
            return dict(time_s=t,available_candidates=[dict(id=a) for a in available],
                        decision=dict(expected_delay=delays),executed_action=dict(id=chosen))
        records=[dict(world=dict(seed=1),history=[
            event(300,['O0','O2'],dict(O0=100.,O2=90.),'O2'),
            event(360,['O0','O2'],dict(O0=100.,O2=110.)),
            event(420,['O0','O2'],{}),
            event(480,['O0'],{})])]
        w=wall_counts(records)
        self.assertEqual({k:v for k,v in w.items() if k!='effects'},
                         dict(decisions=4,available=3,evaluated=2,budget_pruned=1,chosen=1))
        self.assertEqual([e['D_O2'] for e in w['effects']],[10.,-10.])
        d=describe_wall(w)
        self.assertEqual(d['evaluated_helpful'],1)
        self.assertEqual(d['helpful_but_not_chosen'],0)
        self.assertEqual(d['evaluated_D_O2'],dict(mean=0.,minimum=-10.,maximum=10.))

    def test_evaluation_without_geometric_availability_rejected(self):
        with self.assertRaises(AssertionError):
            wall_counts([dict(world=dict(seed=1),history=[dict(available_candidates=[],
                decision=dict(expected_delay=dict(O0=10.,O2=9.)),executed_action=dict(id='O0'))])])


if __name__=='__main__':unittest.main()
