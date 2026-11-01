"""CPU tests for the offline evaluations-saved estimate (synthetic full-pool receipts)."""
import unittest

import numpy as np

from . import learned_search_eval as le
from . import mppi_search as ms
from . import plan_proposer as pp


class Surrogate(unittest.TestCase):
    def test_fit_and_pool_requirements(self):
        r = pp.synthetic_receipts(1, 1, seed=1, sizes=(2,), population=24)[0]
        sur = le.surrogate_for(r)
        pool = [c for c in r['candidates'] if c['stage'] == 'screen']
        y = np.array([c['saving_vs_policy'] for c in pool])
        pred = sur([c['perturbation_canonical'] for c in pool], [c['duration_s'] for c in pool])
        self.assertGreater(np.corrcoef(y, pred)[0, 1], .9)
        self.assertTrue(-np.inf < sur.loo_r2 <= 1)
        self.assertIsNone(le.surrogate_for(pp.synthetic_receipts(1, 1, seed=1, sizes=(2,), topk_only=True)[0]))

    def test_replay_is_deterministic_and_counts_screen_budget(self):
        r = pp.synthetic_receipts(1, 1, seed=2, sizes=(2,), population=24)[0]
        sur = le.surrogate_for(r)
        a = le.replay(r, sur, 'plain'); b = le.replay(r, sur, 'plain')
        self.assertEqual(a, b)
        sc = ms.scaling(2, ms.canonical_knots())
        self.assertEqual(len(a), ms.REGISTERED['iterations']*sc['population'])


class Curve(unittest.TestCase):
    def test_trained_proposer_reaches_plain_quality_with_fewer_evaluations(self):
        rec = pp.synthetic_receipts(160, 16, seed=11, sizes=(2,), population=40)
        S, _ = pp.build_dataset(rec)
        tr, va, vw = pp.world_split(S, .25, 1)
        model, _ = pp.train(tr, dict(steps=300, eval_every=50, d_model=32))
        held = [r for r in rec if str(r['world_id']) in set(vw)][:8]
        res = le.evaluations_curve(model, held + pp.synthetic_receipts(1, 1, seed=3, topk_only=True),
                                   modes=('plain', 'warm', 'seeds'), cfg=dict(bootstraps=200))
        self.assertEqual(res['searches'], 8)
        self.assertEqual(len(res['skipped']), 1)                       # top-k-only receipt has no pool
        s = res['summary']
        self.assertEqual(s['plain']['minus_plain']['mean'], 0.)
        self.assertEqual(s['plain']['reached_rate'], 1.)
        self.assertLess(s['warm']['mean_evals_to_target'], s['plain']['mean_evals_to_target'])
        self.assertLess(s['warm']['minus_plain']['ci_hi'], 0)
        for mode in s.values():
            self.assertEqual(sorted(mode['quality_curve']), sorted(map(str, le.REGISTERED['curve_fracs'])))


if __name__ == '__main__':
    unittest.main()
