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
        a, za = le.replay(r, sur, 'plain'); b, zb = le.replay(r, sur, 'plain')
        self.assertEqual(a, b)
        np.testing.assert_array_equal(np.stack(za), np.stack(zb))
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
                                   modes=('plain', 'warm', 'seeds', 'warm_it1'), cfg=dict(bootstraps=200))
        self.assertEqual(res['searches'], 8)
        self.assertEqual(len(res['skipped']), 1)                       # top-k-only receipt has no pool
        s = res['summary']
        self.assertEqual(s['plain']['minus_plain']['mean'], 0.)
        self.assertEqual(s['plain']['reached_rate'], 1.)
        self.assertLess(s['warm']['mean_evals_to_target'], s['plain']['mean_evals_to_target'])
        self.assertLess(s['warm']['minus_plain']['ci_hi'], 0)
        for mode in s.values():
            self.assertEqual(sorted(mode['quality_curve']), sorted(map(str, le.REGISTERED['curve_fracs'])))
        self.assertAlmostEqual(s['warm_it1']['evaluations_fraction'], 1/3, places=6)
        self.assertEqual(s['plain']['elite_coverage'], 1.)
        self.assertGreater(s['warm']['elite_coverage'], 0)
        self.assertAlmostEqual(s['plain']['final_quality']['mean'], 1.)

    def test_reduced_iterations_keep_horizons(self):
        from .test_mppi_search_cpu import Fake, group_of
        g = group_of(2); fake = Fake(2)
        model = pp.ProposerNet(dict(pp.REGISTERED, d_model=32))
        rec = dict(state_id='h', rows=g['rows'], policy_residuals={v: 0. for v in g['members']})
        plain = ms.search(rec, g, fake.policy, (-3., 1.5), Fake(2), 'ordinary', seed=1)
        ws = pp.warm_start(rec, g, model, 'ordinary')
        f1 = Fake(2)
        one = ms.search(rec, g, fake.policy, (-3., 1.5), f1, 'ordinary', seed=1, warm_start=ws,
                        config=dict(iterations=1))
        self.assertEqual(one['screen'], plain['screen'])
        self.assertEqual(one['final']['K'], plain['final']['K'])
        self.assertEqual(one['final']['H'], ms.final_horizon(max(f['duration_s'] for f in one['finalists'])))
        self.assertEqual({(k, h) for _, k, h in f1.calls[:-1]}, {(4, plain['screen']['H'])})
        self.assertEqual(len(one['iterations']), 1)


if __name__ == '__main__':
    unittest.main()


class ReportPath(unittest.TestCase):
    """Offline (no-fetch) real-data path on a synthetic local cache."""

    def _cache(self, d):
        import json
        from pathlib import Path
        from .test_mppi_search_cpu import BOUNDS, Fake, group_of
        root = Path(d)/'cpuQ'/'home/ubuntu/coopmwm20261103_b'
        for sub in ('requests', 'responses', 'proposer_receipts'):
            (root/sub).mkdir(parents=True)
        g = group_of(2); fake = Fake(2)
        for i, world in enumerate((986001, 986002, 984000)):          # 984000 = dev: must be dropped
            req = dict(state_id=f's{i}', world_id=world, rows=g['rows'], group=g['members'],
                       search_learning=dict(entries=[]))
            res = ms.search(dict(req, policy_residuals={v: 0. for v in g['members']}), g, fake.policy, BOUNDS,
                            fake, 'ordinary', mappo_traj=fake.mappo, seed=i)
            (root/'requests'/f's{i}.json').write_text(json.dumps(req))
            (root/'responses'/f's{i}.json').write_text(json.dumps(dict(res, record_sha256=f's{i}')))
        full = pp.synthetic_receipts(60, 12, seed=5, sizes=(2,), population=16)
        for k, r in enumerate(full):
            r['world_id'] = 986010+k % 12
            (root/'proposer_receipts'/f'f{k}.json').write_text(json.dumps(r))
        bad = Path(d)/'cpuQ'/'home/ubuntu/coopmwm20261103_sealed'/'proposer_receipts'
        bad.mkdir(parents=True)
        (bad/'x.json').write_text(json.dumps(dict(full[0], search_id='SEALED', world_id=986001)))
        return root

    def test_load_cache_filters_and_pairs(self):
        import tempfile
        from . import learned_search_report as lr
        with tempfile.TemporaryDirectory() as d:
            self._cache(d)
            rec, stats = lr.load_cache(d)
            self.assertEqual(stats['pairs'], 3); self.assertEqual(stats['non_train'], 1)
            self.assertNotIn('SEALED', {r['search_id'] for r in rec})
            self.assertEqual(sum(lr.is_full_pool(r) for r in rec), 60)
            self.assertEqual(len(rec), 62)

    def test_run_writes_report_without_activation(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from . import learned_search_report as lr
        with tempfile.TemporaryDirectory() as d:
            self._cache(d)
            with patch.dict(pp.REGISTERED, steps=40, eval_every=20, d_model=32), \
                    patch.object(lr, 'MIN_FULL_POOL', 10), patch.dict(le.REGISTERED, bootstraps=50):
                res = lr.run(Path(d), Path(d)/'REPORT.md', fetch_remote=False)
            text = (Path(d)/'REPORT.md').read_text()
            self.assertIn('NOT ACTIVATED', text)
            self.assertIn('offline_check', res)
            self.assertIn('summary', res['evaluations_curve'])
            self.assertTrue((Path(d)/'REPORT.json').exists())
