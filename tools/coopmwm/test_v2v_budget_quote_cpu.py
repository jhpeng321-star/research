"""No SUMO, outcomes, model imports or reservation in the V2V quote."""
import unittest

from tools.coopmwm.v2v_budget_quote import quote, receipt_scenarios


class BudgetQuoteTests(unittest.TestCase):
    def test_fixed_full_prefix_costs(self):
        empty = dict(A=[], B=[], R=[])
        q = quote(lessons=empty, cooperative=empty)
        self.assertEqual(q['phase_costs']['train'], 2073600)
        self.assertEqual(q['phase_costs']['interim_u8_test'], 345600)
        self.assertEqual(q['phase_costs']['final_test'], 345600)
        self.assertEqual(q['phase_costs']['paired_branch'], 0)
        self.assertEqual(q['fixed_sim_s'], 2764800)
        self.assertEqual(q['first_update'], 1)
        self.assertTrue(q['within_additional_ceiling'])
        self.assertEqual(q['new_sim_s'], 0)
        self.assertFalse(q['launch_authorized'])

    def test_conservative_branch_sets_and_no_double_reference(self):
        # n=2: isolated all/solo0/solo1/background/reference, five branches.
        lessons = dict(A=[], B=[dict(end_s=720, members=2)], R=[])
        coop = dict(A=[], B=[dict(end_s=720, members=2)], R=[])
        q = quote(lessons=lessons, cooperative=coop)
        self.assertEqual(q['phase_costs']['lessons']['B'], 17280)
        self.assertEqual(q['phase_costs']['policy_synergy']['B'], 86400)
        self.assertEqual(q['frozen_slate_total_sim_s'], 2868480)
        self.assertFalse(q['is_full_run_upper_bound'])

    def test_six_member_groups_and_episode_end(self):
        empty = dict(A=[], B=[], R=[])
        coop = dict(A=[], B=[], R=[dict(end_s=1200, members=6)])
        q = quote(lessons=empty, cooperative=coop)
        self.assertEqual(q['phase_costs']['policy_synergy']['R'], 259200)
        self.assertFalse(q['within_additional_ceiling'])

    def test_authorized_two_arm_fallback_and_optional_branches(self):
        empty = dict(A=[], B=[])
        q = quote(lessons=empty, cooperative=empty, arms=('A', 'B'))
        self.assertEqual(q['fixed_sim_s'], 1843200)
        paired = quote(lessons=empty, cooperative=empty, arms=('A', 'B'), paired_arms=('A', 'B'))
        self.assertEqual(paired['fixed_sim_s'], 2534400)
        with self.assertRaises(ValueError):
            quote(lessons=empty, cooperative=empty, arms=('B', 'R'))
        with self.assertRaises(ValueError):
            quote(lessons=empty, cooperative=empty, arms=('A', 'B'), updates=21)

    def test_reject_missing_arm_or_impossible_window(self):
        empty = dict(A=[], B=[], R=[])
        with self.assertRaises(ValueError):
            quote(lessons=dict(B=[], R=[]), cooperative=empty)
        for row in (dict(end_s=1201, members=2), dict(end_s=720, members=7)):
            with self.assertRaises(ValueError):
                quote(lessons=dict(A=[], B=[row], R=[]), cooperative=empty)

    def test_public_receipts_project_metadata_only(self):
        reports = {}
        for a in ('B', 'R'):
            reports['lesson_'+a] = dict(arm=a, entries=[dict(status='graded',
                new_policy_branch=dict(window=dict(end_s=720)))])
            reports['synergy_'+a] = dict(arm=a, entries=[dict(
                group=['c1', 'c2'], window=dict(end_s=720))])
        q = receipt_scenarios(reports)
        self.assertEqual(q['three_arms']['frozen_slate_total_sim_s'], 3075840)
        self.assertEqual(q['two_arm_fallback']['frozen_slate_total_sim_s'], 2050560)
        self.assertFalse(q['three_arms']['within_additional_ceiling'])
        self.assertTrue(q['two_arm_fallback']['within_additional_ceiling'])
        self.assertEqual(q['two_arm_reference_reuse']['frozen_slate_total_sim_s'], 2016000)
        self.assertTrue(q['two_arm_reference_reuse']['reference_reuse']['enabled'])
        reports['lesson_R']['entries'][0]['status'] = 'failed'
        with self.assertRaises(ValueError):
            receipt_scenarios(reports)


if __name__ == '__main__':
    unittest.main()
