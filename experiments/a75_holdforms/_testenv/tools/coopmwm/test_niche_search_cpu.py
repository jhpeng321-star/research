"""CPU-only tests for the multi-modal ('niching') MPPI plan search (no SUMO, no WM, no GPU)."""
import importlib.util
import json
import os
import sys
import unittest
from functools import lru_cache
from pathlib import Path
import numpy as np
from . import mppi_search as ms
from .test_mppi_search_cpu import BOUNDS, Fake, group_of, run, T

FIXTURE = Path(__file__).resolve().parent/'fixtures'/'niche1_golden.json'


@lru_cache(maxsize=1)
def golden_reference():
    """A Windows JSON golden is not a bitwise Linux BLAS oracle.

    Linux qualification supplies the immutable PRE-CHANGE Git module, evaluated
    on the same interpreter/BLAS. Compare exact serialized bytes, never a widened
    tolerance or a fixture regenerated from the new implementation.
    """
    path=os.environ.get('COOPMWM_NICHE_BASELINE')
    if not path:return json.loads(FIXTURE.read_text())
    from unittest.mock import patch
    from . import test_mppi_search_cpu as fixtures
    spec=importlib.util.spec_from_file_location('tools.coopmwm._pre_niche_reference',path)
    old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
    with patch.object(fixtures,'ms',old),patch.object(sys.modules[__name__],'ms',old):
        fake=Fake(2);fake.mappo=fake.target.copy();g=group_of(2)
        return {name:json.dumps(result,sort_keys=True) for name,result in dict(
            plain1=run(seed=1)[0],plain2=run(seed=2)[0],six1=run(seed=1,m=6)[0],
            cem1=run(seed=1,config=dict(method='cem'))[0],
            reject1=old.search(dict(policy_residuals={}),g,fake.policy,BOUNDS,fake,'ordinary',seed=1)).items()}


def niche_run(seed=1, m=2, niches=4, **cfg):
    fake = Fake(m); g = group_of(m)
    config = dict(cfg)
    if niches > 1:
        config['niches'] = niches
    res = ms.search(dict(state_id='s', policy_residuals={v: 0. for v in g['members']}), g,
                    fake.policy, BOUNDS, fake, 'ordinary', mappo_traj=fake.mappo, seed=seed, config=config)
    return res, fake


class TwoOptima(Fake):
    """Synthetic two-optimum team cost: a brake/accel pair (basin A) and its exact
    mirror (basin B). Equally deep, far apart, neutral ridge in between: a purely
    bimodal landscape for the niche discovery contract. Test fixture only; the
    production search never sees these constants."""

    def __init__(self, m=2):
        super().__init__(m)
        t = np.arange(T, dtype=float)
        shape = np.sin(np.pi*np.clip(t, 0, 150)/150)
        a = np.full((m, T), .2); a[0] += -1.4*shape; a[1] += 1.4*shape
        b = np.full((m, T), .2); b[0] += 1.4*shape; b[1] += -1.4*shape
        self.optA, self.optB = a, b

    def cost(self, path):
        return min(float(np.square(path-self.optA).sum()), float(np.square(path-self.optB).sum()))

    def basin(self, path):
        return 'A' if np.square(path-self.optA).sum() <= np.square(path-self.optB).sum() else 'B'

    def full_path(self, plan):
        out = self.policy.copy(); D = plan['duration_s']
        out[:, :D] = np.asarray(plan['profile'])
        return out

    def basins(self, res):
        return {self.basin(self.full_path(p)) for p in res['topk']}


def two_optima_run(seed, niches):
    fake = TwoOptima(2); g = group_of(2)
    cfg = dict(top_k=8, finalists_per_k=1)
    if niches > 1:
        cfg['niches'] = niches
    res = ms.search(dict(state_id='s', policy_residuals={v: 0. for v in g['members']}), g,
                    fake.policy, BOUNDS, fake, 'ordinary', mappo_traj=fake.mappo, seed=seed, config=cfg)
    return res, fake


class ByteIdentity(unittest.TestCase):
    """niches=1 (default AND explicit) must be byte-identical to the pre-change search."""

    def test_niches1_matches_the_pre_change_golden(self):
        golden = golden_reference()
        fake = Fake(2); fake.mappo = fake.target.copy(); g = group_of(2)
        scenarios = {
            'plain1': lambda: run(seed=1)[0],
            'plain2': lambda: run(seed=2)[0],
            'six1': lambda: run(seed=1, m=6)[0],
            'cem1': lambda: run(seed=1, config=dict(method='cem'))[0],
            'reject1': lambda: ms.search(dict(policy_residuals={}), g, fake.policy, BOUNDS,
                                         fake, 'ordinary', seed=1),
        }
        for name, fn in scenarios.items():
            self.assertEqual(json.dumps(fn(), sort_keys=True), golden[name], name)

    def test_explicit_niches1_matches_the_same_golden(self):
        golden = golden_reference()
        self.assertEqual(json.dumps(run(seed=1, config=dict(niches=1))[0], sort_keys=True),
                         golden['plain1'])
        self.assertEqual(json.dumps(run(seed=1)[0], sort_keys=True), golden['plain1'])


class Helpers(unittest.TestCase):
    def test_plan_distance_is_rms_in_canonical_space(self):
        a, b = np.zeros((2, 10)), np.full((2, 10), .1)
        self.assertEqual(ms.plan_distance(a, a), 0.)
        self.assertAlmostEqual(ms.plan_distance(a, b), .1)
        self.assertAlmostEqual(ms.plan_distance(a, 2*b), 2*ms.plan_distance(a, b))   # scale-linear
        self.assertEqual(ms.plan_distance(a, b), ms.plan_distance(b, a))             # symmetric
        pa = ms.build_plan(group_of(2), np.zeros((2, T)), np.zeros((2, 10)), 60, BOUNDS, 'sampled', 'x')
        self.assertEqual(ms.plan_distance(pa, pa), 0.)
        with self.assertRaises(ValueError):
            ms.plan_distance(a, np.zeros((2, 11)))

    def test_niche_sizes_split_is_exact_and_batch_multiple(self):
        self.assertEqual(ms.niche_sizes(32, 4), [8, 8, 8, 8])
        self.assertEqual(ms.niche_sizes(40, 4), [12, 12, 8, 8])
        self.assertEqual(ms.niche_sizes(120, 4), [32, 32, 28, 28])
        self.assertEqual(ms.niche_sizes(120, 3), [40, 40, 40])
        self.assertEqual(ms.niche_sizes(32, 1), [32])
        for n, M in ((40, 4), (120, 4), (120, 3), (32, 2)):
            sizes = ms.niche_sizes(n, M)
            self.assertEqual(sum(sizes), n)
            self.assertTrue(all(s % 4 == 0 for s in sizes))
        with self.assertRaises(ValueError):
            ms.niche_sizes(30, 4)
        with self.assertRaises(ValueError):
            ms.niche_sizes(32, 0)

    def test_median_pair_distance_is_the_data_driven_median(self):
        xs = [np.zeros((2, 10)), np.full((2, 10), .1), np.full((2, 10), .2)]
        self.assertAlmostEqual(ms._median_pair_distance(xs), .1)    # distances .1 .1 .2
        self.assertEqual(ms._median_pair_distance(xs[:1]), 0.)      # no pairs
        self.assertEqual(ms._median_pair_distance([]), 0.)

    def test_diversified_order_niche_bests_first_then_max_min_fill(self):
        def plan(x):
            return np.full((2, 10), float(x))
        scored = [('a1', 1.0, plan(0.)), ('a0', 3.0, plan(.25)),     # niche 0 (a1 best)
                  ('b1', 2.0, plan(1.)), ('b0', 4.0, plan(1.25)),    # niche 1 (b1 best)
                  ('c1', 2.5, plan(-1.)), ('c0', 5.0, plan(-1.25))]  # niche 2 (c1 best)
        niche_of = dict(a1=0, a0=0, b1=1, b0=1, c1=2, c0=2)
        # Niche bests first in best-score order; the fills tie exactly (binary .25
        # offsets), so the better score wins deterministically.
        self.assertEqual(ms._diversified_order(scored, niche_of, 5), ['a1', 'b1', 'c1', 'a0', 'b0'])
        self.assertEqual(ms._diversified_order(scored, niche_of, 3), ['a1', 'b1', 'c1'])
        # A far-flung extra candidate beats the near-clones in the max-min fill.
        scored.append(('z1', 6.0, plan(5.))); niche_of['z1'] = 1
        self.assertEqual(ms._diversified_order(scored, niche_of, 4), ['a1', 'b1', 'c1', 'z1'])


class Parity(unittest.TestCase):
    def test_wm_evaluation_count_matches_niches1(self):
        for m, seed in ((2, 1), (6, 3)):
            r1, f1 = run(seed=seed, m=m)
            r4, f4 = niche_run(seed=seed, m=m, niches=4)
            self.assertEqual(len(f1.calls), len(f4.calls))
            self.assertEqual([(len(ids), K, H) for ids, K, H in f1.calls],
                             [(len(ids), K, H) for ids, K, H in f4.calls])
            self.assertEqual([(K, H) for _, K, H in f1.calls[:3]], [(K, H) for _, K, H in f4.calls[:3]])
            self.assertEqual(r1['imagined_futures'], r4['imagined_futures'])
            self.assertEqual(r1['candidates_searched'], r4['candidates_searched'])
            self.assertEqual(len(r1['finalists']), len(r4['finalists']))
            self.assertEqual(r1['scaling'], r4['scaling'])
            self.assertEqual(sum(r4['niche_sizes']), r4['scaling']['population'])
            self.assertEqual(r4['screen'], r1['screen'])
            self.assertEqual(r4['final']['K'], r1['final']['K'])


class Coverage(unittest.TestCase):
    def test_niches4_top8_covers_both_optima_while_niches1_returns_one(self):
        for seed in (0, 6):                     # pinned: deterministic on this host
            res4, fake4 = two_optima_run(seed, 4)
            self.assertTrue(res4['accepted'])
            self.assertEqual(len(res4['topk']), 8)
            self.assertEqual(fake4.basins(res4), {'A', 'B'})          # both optima covered
            self.assertGreaterEqual(res4['topk_distinct_niches'], 2)
            self.assertGreater(res4['topk_mean_pairwise_distance'], 0.)
            for p in res4['topk']:
                self.assertIsInstance(p['niche'], int)
            self.assertTrue(all(isinstance(f['niche'], int) for f in res4['finalists']))
            self.assertEqual(res4['niches'], 4)
            self.assertGreater(res4['repulsion_threshold'], 0.)
            json.dumps(res4)
            res1, fake1 = two_optima_run(seed, 1)
            self.assertTrue(res1['accepted'])
            self.assertEqual(len(fake1.basins(res1)), 1)              # single idea only
            self.assertNotIn('niche', res1['topk'][0])
            self.assertNotIn('topk_distinct_niches', res1)
            self.assertNotIn('topk_mean_pairwise_distance', res1)
            self.assertNotIn('niche', res1['finalists'][0])

    def test_rejection_path_carries_niche_diagnostics_without_topk(self):
        fake = Fake(2); fake.mappo = fake.target.copy()               # MAPPO already optimal
        g = group_of(2)
        res = ms.search(dict(policy_residuals={}), g, fake.policy, BOUNDS, fake, 'ordinary',
                        seed=1, config=dict(niches=4))
        self.assertFalse(res['accepted']); self.assertIsNone(res['plan'])
        self.assertNotIn('topk', res); self.assertIsNone(res['option'])
        self.assertIsNone(res['expected_saving'])
        self.assertEqual(res['niches'], 4)
        self.assertEqual(res['niche_sizes'], [12, 12, 8, 8])
        self.assertGreaterEqual(res['repulsion_threshold'], 0.)
        json.dumps(res)


class Repulsion(unittest.TestCase):
    def test_flat_landscape_repulses_the_worse_niche_once_per_iteration(self):
        fake = Fake(2); fake.cost = lambda path: 1000.                # no gradient at all
        g = group_of(2)
        res = ms.search(dict(policy_residuals={}), g, fake.policy, BOUNDS, fake, 'ordinary',
                        seed=1, config=dict(niches=4))
        self.assertEqual(res['niches'], 4)
        threshold = res['repulsion_threshold']
        self.assertGreater(threshold, 0.)                             # data-driven, from the noise
        for e in res['iterations']:
            r = e['repulsion']
            self.assertIsNotNone(r)            # collapsed means are closer than the median
            self.assertLess(r['distance'], r['threshold'])
            self.assertEqual(r['threshold'], threshold)
            self.assertIn(r['reseeded'], r['pair'])
            self.assertIsInstance(r['reseeded'], int)
        json.dumps(res)

    def test_degenerate_initial_population_disables_repulsion(self):
        fake = Fake(2); fake.cost = lambda path: 1000.
        g = group_of(2)
        res = ms.search(dict(policy_residuals={}), g, fake.policy, (-0.5, -0.5), fake, 'ordinary',
                        seed=1, config=dict(niches=4))
        self.assertEqual(res['repulsion_threshold'], 0.)              # all initial plans identical
        self.assertTrue(all(e['repulsion'] is None for e in res['iterations']))


@unittest.skipUnless(importlib.util.find_spec('torch'), 'torch unavailable in this interpreter')
class RuntimeWrapper(unittest.TestCase):
    def test_label_list8_wrapper_passes_niches_and_labels_the_top8(self):
        from .mppi_runtime import label_list8_search
        fake = TwoOptima(2); g = group_of(2)

        def evaluate(record, actions, k, h):
            return fake(record, actions, k, h)

        out = label_list8_search(ms.search)(dict(state_id='s', policy_residuals={v: 0. for v in g['members']}),
                                            g, fake.policy, BOUNDS, evaluate, 'ordinary',
                                            mappo_traj=fake.mappo, seed=0, config=dict(niches=4))
        self.assertEqual(out['config']['niches'], 4)
        self.assertEqual(len(out['topk']), 8)
        self.assertIn('topk_distinct_niches', out)
        self.assertGreaterEqual(len({p['niche'] for p in out['topk'] if 'niche' in p}), 2)
        self.assertTrue(all('niche' in p for p in out['collection_top8'] if p['id']
                            in {q['id'] for q in out['topk'] if 'niche' in q}))
        json.dumps(out)


if __name__ == '__main__':
    unittest.main()
