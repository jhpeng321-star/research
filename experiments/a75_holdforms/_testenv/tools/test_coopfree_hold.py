"""CPU-only regression gates for the HOLD correction, including its estimand."""
import copy
import unittest
import torch
from tools.coopfree.policy import SetPointPolicy, FEATURES
from tools.coopfree.credit import credit
from tools.coopfree_hold import mean_hold, solo_hold, exact_subset, optimize


def sample(n=3):
    context = dict(members=[f'c{i}' for i in range(n)], x=[[0.]*FEATURES for _ in range(n)],
                   rsu=[0.]*12, anchors=[[12.+i, .5] for i in range(n)])
    return SetPointPolicy().sample(context, torch.Generator().manual_seed(1))


class HoldTests(unittest.TestCase):
    def test_mean_baseline_and_solo_control_all_members(self):
        s = sample()
        original = copy.deepcopy(s)
        mean = mean_hold(SetPointPolicy(), s)
        self.assertEqual(mean['action'], s['context']['anchors'])
        self.assertEqual(mean['window'], s['window'])
        for i, member in enumerate(s['context']['members']):
            solo = solo_hold(mean, s, member)
            self.assertEqual(solo['context']['members'], s['context']['members'])
            for j in range(3):
                self.assertEqual(solo['action'][j], s['action'][j] if i == j else mean['action'][j])
        self.assertEqual(s, original)

    def test_exact_subset_whole_groups_budget_and_world_order(self):
        windows = [dict(world=dict(seed=seed), t0=300, sample=sample(n))
                   for seed, n in [(1,3),(1,1),(2,2),(3,4)]]
        chosen = exact_subset(windows, 5400)
        self.assertEqual(chosen, [0,2,1])
        self.assertEqual(sum(len(windows[i]['sample']['context']['members'])*900 for i in chosen), 5400)

    def test_objective_independent_of_expensive_solo_diagnostics(self):
        for solos in ([110., 75.], [1., 999.]):
            c = credit(100., 80., solos)
            self.assertEqual(c['objective'], 40.)
        with self.assertRaises(ValueError):
            credit(100., 80., [90.], w_coop=1.1)

    def test_score_update_uses_hold_credit_and_never_scores_window(self):
        torch.manual_seed(4)
        policy = SetPointPolicy()
        initial_window = {k:v.clone() for k,v in policy.window.state_dict().items()}
        s = sample()
        records = [dict(world=dict(seed=i%12), sample=s, D_G={'600':10.},
                        execution=dict(motion_changed=True, issued_changed=True, requests=3, overrides=1))
                   for i in range(156)]
        result = optimize(policy, torch.optim.Adam(policy.parameters(), lr=.001), records)
        self.assertEqual(result['windows'], 156)
        self.assertIsNone(result['C_G'])
        self.assertEqual(result['D_G']['mean'], 10.)
        self.assertEqual(result['executed_vs_issued_share'], 1.)
        self.assertAlmostEqual(result['safety_override_share'], 1/3)
        self.assertGreater(result['gradient_norm'], 0)
        for key, value in policy.window.state_dict().items():
            self.assertTrue(torch.equal(value, initial_window[key]))
        with self.assertRaises(ValueError):
            optimize(policy, torch.optim.Adam(policy.parameters()), records[:149])


if __name__ == '__main__':
    unittest.main()
