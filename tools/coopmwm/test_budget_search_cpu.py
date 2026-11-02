"""CPU tests for the per-episode warm start (v2 C; synthetic receipts; no SUMO, no WM, no GPU)."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from . import budget_search_eval as be
from . import learned_search_eval as le
from . import mppi_search as ms
from . import plan_proposer as pp


class ShiftedSeeds(unittest.TestCase):
    def test_shift_is_interp_and_zero_after_end(self):
        r = pp.synthetic_receipts(1, 1, seed=3, sizes=(2,), population=24)[0]
        t = np.linspace(0., 300., ms.canonical_knots())
        seeds = be.shifted_seeds(r, 30.)
        self.assertTrue(0 < len(seeds) <= 8)
        fin = sorted((c for c in r['candidates'] if c['stage'] == 'final'), key=lambda c: -c['saving_vs_policy'])
        z = np.asarray(fin[0]['perturbation_canonical'], float)
        np.testing.assert_allclose(np.asarray(seeds[0]['perturbation_canonical'], float),
                                   np.where(t+30. <= t[-1], np.stack([np.interp(t+30., t, row) for row in z]), 0.),
                                   atol=1e-9)
        self.assertEqual([s['duration_s'] for s in seeds], [c['duration_s'] for c in fin[:len(seeds)]])
        self.assertTrue(all(int(s['duration_s']) in ms.REGISTERED['durations_s'] for s in seeds))

    def test_gap_beyond_span_zeroes_and_bad_rows_are_skipped(self):
        r = pp.synthetic_receipts(1, 1, seed=4, sizes=(2,), population=24)[0]
        self.assertTrue(all(not np.any(s['perturbation_canonical']) for s in be.shifted_seeds(r, 300.)))
        for bad in (dict(stage='final', perturbation_canonical=[[0.]*3]*2, duration_s=60, saving_vs_policy=1.),
                    dict(stage='final', perturbation_canonical=[[0.]*ms.canonical_knots()]*2, duration_s=45,
                         saving_vs_policy=1.)):
            r2 = dict(r, candidates=r['candidates']+[bad])
            self.assertTrue(all(np.shape(s['perturbation_canonical'])[1] == ms.canonical_knots()
                                and int(s['duration_s']) in ms.REGISTERED['durations_s']
                                for s in be.shifted_seeds(r2, 10.)))


class EpisodePairs(unittest.TestCase):
    def test_consecutive_same_episode_pairs(self):
        with tempfile.TemporaryDirectory() as d:
            req = Path(d)/'cpuQ/home/ubuntu/coopmwm20261102_warm1/requests'
            req.mkdir(parents=True)
            rows = [dict(vid='a', s_m=0., v_mps=10., lane=0), dict(vid='b', s_m=-20., v_mps=12., lane=1)]
            q = [dict(world_id=986000, time_s=100., state_id='late', group=['b', 'a'], rows=rows),
                 dict(world_id=986000, time_s=150., state_id='early2', group=['a', 'b'], rows=rows),
                 dict(world_id=986000, time_s=451., state_id='gap', group=['a', 'b'], rows=rows),
                 dict(world_id=986000, time_s=456., state_id='small', group=['a', 'b'], rows=rows),
                 dict(world_id=986000, time_s=200., state_id='other', group=['a', 'c'], rows=rows),
                 dict(world_id=999999, time_s=0., state_id='far', group=['a', 'b'], rows=rows)]
            for i, x in enumerate(q):
                (req/f'{i:016x}.json').write_text(json.dumps(x))
            got = {(p['early'], p['late'], p['gap_s']) for p in be.episode_pairs(d)}
            self.assertEqual(got, {('late', 'early2', 50.)})       # 5 s and 351 s gaps fail; member/world mismatches drop


class WarmStartEval(unittest.TestCase):
    def test_extra_seeds_keep_futures_and_change_the_arm(self):
        early, late = pp.synthetic_receipts(2, 2, seed=6, sizes=(2,), population=24)
        seeds = be.shifted_seeds(early, 10.)
        sur = le.surrogate_for(late)
        pool = [c for c in late['candidates'] if c['stage'] == 'screen']
        sav = np.array([c['saving_vs_policy'] for c in pool], float)
        sigma = be.ALPHA_V2*np.sqrt(max(0., 1-sur.loo_r2))*sav.std()
        c1, f1 = be.noisy_replay(late, sur, sigma, None, 'epws')
        c2, f2 = be.noisy_replay(late, sur, sigma, None, 'epws', extra_seeds=seeds)
        self.assertEqual(len(c1), len(c2))                      # seeds displace samples: futures unchanged
        self.assertEqual(len(f1), len(f2))
        res = be.evaluate_warm_start([dict(world=986000, early=early['search_id'], late=late['search_id'], gap_s=10.)],
                                     [early, late], held_worlds={'986000'})
        self.assertEqual(res['evaluated'], 1)
        self.assertEqual(res['scopes']['heldout']['searches'], 1)
        self.assertFalse(res['scopes']['heldout']['enough_data'])   # 1 pair < registered minimum
        self.assertIn('q8_warm_minus_control', res['scopes']['heldout'])

    def test_under_pairs_gate_reports_counts(self):
        early, late = pp.synthetic_receipts(2, 2, seed=7, sizes=(2,), population=24)
        res = be.evaluate_warm_start([dict(world=986001, early=early['search_id'], late=late['search_id'], gap_s=15.)],
                                     [early, late])
        self.assertFalse(res['enough_data'])                    # 1 evaluated < 20: registered gate not met
        self.assertEqual(res['pairs'], 1)


if __name__ == '__main__':
    unittest.main()
