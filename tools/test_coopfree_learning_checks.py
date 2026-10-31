import copy
import unittest
from unittest.mock import patch
from tools.coopfree_learning_checks import learning_check, paired_hold


def update(i, mean=0, executed=.9):
    r = dict(update=i, baseline='zero-update policy-mean HOLD', windows=156, worlds=24,
             executed_change_share=executed, issued_change_share=1., executed_vs_issued_share=executed,
             safety_override_share=.1, positive_C_G_share=.5, exact_credit_windows=66,
             exact_credit_worlds=24, sigma_used=[4.5, .5], sigma_next=[4.5, .5])
    r.update({k: dict(mean=mean, n=66 if k != 'D_G' else 156) for k in ('D_G', 'sum_D_i', 'C_G')})
    return r


class LearningChecks(unittest.TestCase):
    def test_observer_remote_source_compiles_without_dispatch(self):
        from tools.coopfree_hold_publish import collect
        def inspect(host, source, timeout):
            self.assertEqual(host, 'cpuQ')
            compile(source, '<read-only observer>', 'exec')
            self.assertNotIn('subprocess.Popen', source)
            self.assertNotIn('os.kill', source)
            return '{}'
        with patch('tools.coopfree_hold_publish.remote', side_effect=inspect):
            self.assertEqual(collect([]), {})

    def test_cap_blocks_future_without_counting_replays(self):
        r = learning_check([update(1)], 1649760)
        self.assertEqual(r['train_remaining_sim_s'], 240)
        self.assertTrue(r['further_update_budget_blocked'])
        self.assertIsNone(r['trend'])
        with self.assertRaises(ValueError):
            learning_check([update(1), update(1)], 1649760)

    def test_flat_effects_trigger_at_six_not_three(self):
        self.assertEqual(learning_check([update(i) for i in range(1, 4)], 0)['recommended_phase_changes'], [])
        r = learning_check([update(i) for i in range(1, 7)], 0)
        self.assertEqual(r['status'], 'phase fix required')
        self.assertEqual(len(r['recommended_phase_changes']), 4)

    def test_small_execution_triggers_even_with_upward_effects(self):
        records = [update(i, mean=i, executed=.5) for i in range(1, 7)]
        self.assertEqual(learning_check(records, 0)['status'], 'phase fix required')
        for r in records:
            r['executed_change_share'] = .9
        self.assertEqual(learning_check(records, 0)['status'], 'no registered phase-fix trigger')

    def test_missing_credit_legacy_or_incomplete_never_pass(self):
        records = [update(i) for i in range(1, 7)]
        records[-1]['C_G'] = None
        self.assertIn('missing exact C_G', learning_check(records, 0)['status'])
        for key, value in [('baseline', 'MAPPO'), ('windows', 149), ('worlds', 11)]:
            r = update(1)
            r[key] = value
            with self.assertRaises(ValueError):
                learning_check([r], 0)

    def test_paired_bootstrap_cancels_world_variance(self):
        def rows(offset):
            return [dict(world=dict(seed=s, cell=s), metrics={h: {m: s*10000+offset for m in
                ('team_delay_vs', 'collisions', 'teleports', 'ttc_below_3_vs', 'hard_brake')}
                for h in ('600', '900')}) for s in range(12)]
        a, b = rows(-7), rows(0)
        r = paired_hold(a, b)
        self.assertEqual(r['learned_minus_hold']['600_team_delay_vs'], dict(mean=-7., ci95=[-7., -7.]))
        broken = copy.deepcopy(b)
        broken[0]['world']['cell'] = 99
        with self.assertRaises(ValueError):
            paired_hold(a, broken)
        with self.assertRaises(ValueError):
            paired_hold(a, b[:-1])
        with self.assertRaises(ValueError):
            paired_hold(a, b+b[:1])


if __name__ == '__main__':
    unittest.main()
