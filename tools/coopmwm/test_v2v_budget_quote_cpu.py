"""No SUMO, outcomes, model imports or reservation in the V2V quote."""
import unittest

from tools.coopmwm.v2v_budget_quote import quote


class BudgetQuoteTests(unittest.TestCase):
    def test_fixed_full_prefix_costs(self):
        empty = dict(A=[], B=[], R=[])
        q = quote(lessons=empty, cooperative=empty)
        self.assertEqual(q['phase_costs']['train'], 1814400)
        self.assertEqual(q['phase_costs']['interim_u8_test'], 345600)
        self.assertEqual(q['phase_costs']['final_test'], 345600)
        self.assertEqual(q['phase_costs']['paired_branch'], 691200)
        self.assertEqual(q['fixed_sim_s'], 3196800)
        self.assertEqual(q['new_sim_s'], 0)
        self.assertFalse(q['launch_authorized'])

    def test_conservative_branch_sets_and_no_double_reference(self):
        # n=2: original all/solo0/solo1/reference plus isolated
        # all/solo0/solo1/background = eight full-prefix branches.
        lessons = dict(A=[], B=[dict(end_s=720, members=2)], R=[])
        coop = dict(A=[], B=[dict(end_s=720, members=2)], R=[])
        q = quote(lessons=lessons, cooperative=coop)
        self.assertEqual(q['phase_costs']['lessons']['B'], 15120)
        self.assertEqual(q['phase_costs']['policy_synergy']['B'], 120960)
        self.assertEqual(q['frozen_slate_total_sim_s'], 3332880)
        self.assertFalse(q['is_full_run_upper_bound'])

    def test_six_member_groups_and_episode_end(self):
        empty = dict(A=[], B=[], R=[])
        coop = dict(A=[], B=[], R=[dict(end_s=1200, members=6)])
        q = quote(lessons=empty, cooperative=coop)
        self.assertEqual(q['phase_costs']['policy_synergy']['R'], 403200)

    def test_reject_missing_arm_or_impossible_window(self):
        empty = dict(A=[], B=[], R=[])
        with self.assertRaises(ValueError):
            quote(lessons=dict(B=[], R=[]), cooperative=empty)
        for row in (dict(end_s=1201, members=2), dict(end_s=720, members=7)):
            with self.assertRaises(ValueError):
                quote(lessons=dict(A=[], B=[row], R=[]), cooperative=empty)


if __name__ == '__main__':
    unittest.main()
