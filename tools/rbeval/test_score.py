"""Scientific invariants: leakage, clustered uncertainty, ties and identity safety."""
import sys
import copy
import json
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import ARMS, HERE, digest, load, save_new
from score import WorldBootstrap, eligible_peers, rank_stats, read_predictions, scored_bootstrap
import run_wm


class ScientificInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.real = load(HERE/'real_observables.json')['episodes']

    def test_climatology_leaves_entire_world_and_reserved_labels_out(self):
        for seed in (969420, 969432):
            target = next(e for e in self.real if e['seed'] == seed and e['arm'] == 'mappo')
            for cell in (False, True):
                peers = eligible_peers(self.real, target, cell)
                self.assertTrue(peers)
                self.assertTrue(all(p['seed'] != seed and p['seed'] < 969432 for p in peers))
                if cell:
                    self.assertTrue(all((p['arm'], p['demand'], p['penetration']) ==
                        (target['arm'], target['demand'], target['penetration']) for p in peers))

    def test_custom_constructor_cannot_impersonate_wm2(self):
        with patch.object(sys, 'argv', ['run_wm.py', '--out', 'unused', '--model-name', 'wm2']):
            with self.assertRaisesRegex(AssertionError, 'explicit candidate identity'):
                run_wm.main(model_loader=lambda *args: None)

    def test_custom_futures_cannot_impersonate_wm2(self):
        with patch.object(sys, 'argv', ['run_wm.py', '--out', 'unused', '--model-name', 'wm2']):
            with self.assertRaisesRegex(AssertionError, 'explicit candidate identity'):
                run_wm.main(future_builder=lambda **kwargs: None)

    def test_world_pairing_does_not_create_fake_sample_size(self):
        boot = WorldBootstrap(self.real, B=1000)
        pairs = [(969420, 0.), (969426, 2.)]
        single, duplicate = boot.bands([pairs, pairs*5])
        self.assertEqual(single, duplicate)
        self.assertEqual(single['worlds'], 2)
        self.assertIsNone(boot.bands([[(969420, 1.)]])[0]['low'])

    def test_finite_K_uncertainty_is_included(self):
        boot = WorldBootstrap(self.real, B=1000)
        a, b = next([boot.seeds[i] for i in ix] for ix in boot.strata if len(ix) == 2)
        # Real outcomes agree across worlds, but finite-K event probability varies.
        band = boot.bands([[(a, .5, [0., 1.]*4), (b, .5, [0., 1.]*4)]])[0]
        self.assertIsNotNone(band['low'])
        self.assertLess(band['low'], .5)
        self.assertGreater(band['high'], .5)

    def test_partial_reports_cannot_borrow_unscored_world_variation(self):
        # Two distinct scored singleton cells have no within-cell sampling
        # variation. Unscored same-cell peers must not randomly remove a target.
        worlds = [969420, 969421]
        boot = scored_bootstrap(self.real, worlds, B=1000)
        self.assertEqual(boot.seeds, worlds)
        np.testing.assert_array_equal(boot.weights, np.ones((1000, 2)))
        band = boot.bands([[(969420, 0.), (969421, 100.)]])[0]
        self.assertEqual(band['mean'], 50.)
        self.assertIsNone(band['low'])
        self.assertFalse(band['uncertainty_available'])
        full = scored_bootstrap(self.real, range(969420, 969440), B=1000)
        np.testing.assert_array_equal(full.weights, WorldBootstrap(self.real, B=1000).weights)

    def test_fixed_singleton_strata_cannot_get_roundoff_or_K_only_CIs(self):
        worlds = list(range(969420, 969432))
        boot = scored_bootstrap(self.real, worlds, B=10000)
        np.testing.assert_array_equal(boot.weights, np.ones((10000, 12)))
        for samples in (False, True):
            features = [[(seed, (j+1)*7312.123456789, [(j+1)*7312.123456789-10,
                         (j+1)*7312.123456789+10]*4) if samples else
                         (seed, (j+1)*7312.123456789) for j, seed in enumerate(worlds)]]
            band = boot.bands(features)[0]
            self.assertIsNone(band['low'])
            self.assertIsNone(band['high'])
            self.assertFalse(band['uncertainty_available'])

    def test_chunking_preserves_joint_family_maxstat(self):
        boot = WorldBootstrap(self.real, B=1000)
        a, b = next([boot.seeds[i] for i in ix] for ix in boot.strata if len(ix) == 2)
        features = [[(a, v, [v-1, v+1]*4), (b, 2*v, [2*v-2, 2*v+2]*4)]
                    for v in (1., 2., 3.)]
        whole = boot.bands(features, chunk_size=10)
        chunks = boot.bands(features, chunk_size=1)
        for a, b in zip(whole, chunks):
            for key in ('mean', 'low', 'high', 'maxstat_critical'):
                self.assertAlmostEqual(a[key], b[key], places=12)

    def test_ties_randomized_without_false_range_failure(self):
        rng = np.random.default_rng(123)
        results = [rank_stats([3.]*8, 3., rng) for _ in range(9000)]
        self.assertTrue(all(r['range_coverage'] == 1 for r in results))
        self.assertAlmostEqual(np.mean([r['randomized_range_coverage'] for r in results]), 7/9, delta=.02)
        self.assertAlmostEqual(np.mean([r['randomized_pit'] for r in results]), .5, delta=.02)

    def test_partial_K_is_not_silently_scored_as_K8(self):
        root = HERE/'evidence/wm2_shard0'
        if not root.exists():
            self.skipTest('GPU canary receipt not fetched on this machine')
        pred, audit = read_predictions([root], self.real, partial=True)
        self.assertGreaterEqual(audit['tasks'], 1)
        for seed in audit['complete_worlds']:
            self.assertTrue(all(len(pred[seed, arm]) == 8 for arm in ('native', 'ctrl0', 'mappo', 'pi', 'circles')))
        # At least one world is incomplete in the shard, never interpreted as K8.
        self.assertTrue(audit['missing_tasks'])
        with self.assertRaises(AssertionError):
            read_predictions([root, root], self.real, partial=True)

    def test_external_adapter_and_gate_end_to_end(self):
        # Synthetic plumbing test, never a scientific model score. Preserve the
        # uniquely named scratch artifacts rather than removing any files.
        parent = Path('E:/Temp/opencode') if os.name == 'nt' else Path('/tmp')
        self.assertTrue(parent.is_dir())
        scratch = Path(tempfile.mkdtemp(prefix='rbeval-synthetic-', dir=parent))
        episodes = []
        for arm in ARMS:
            real = next(e for e in self.real if e['seed'] == 969420 and e['arm'] == arm)
            bins = {reg: dict(speed=real['bins'][reg]['speed'][6:],
                             post_counts=real['bins'][reg]['post_counts']) for reg in ('upstream', 'merge', 'ramp')}
            episodes.append(dict(arm=arm, initial_sha256=real['initial_sha256'],
                                 bins=bins, clocks={h: {key: value[key] for key in
                                     ('team_delay_vs', 'road_delay_vs', 'source_wait_vs')}
                                     for h, value in real['clocks'].items()}))
        manifest = dict(model='synthetic-plumbing-test', model_sha256=digest(HERE/'core.py'),
                        config_sha256=digest(HERE/'references.json'), sources={str(HERE/'core.py'): digest(HERE/'core.py')},
                        tasks=[dict(seed=969420, future=k, episodes=copy.deepcopy(episodes)) for k in range(8)])
        save_new(scratch/'manifest.json', manifest)
        for script, args in [('import_external.py', ['--manifest', str(scratch/'manifest.json'), '--out', str(scratch/'normalized')]),
                             ('score.py', ['--inputs', str(scratch/'normalized'), '--out', str(scratch/'report'), '--allow-partial'])]:
            result = subprocess.run([sys.executable, '-B', str(HERE/script), *args], capture_output=True,
                                    text=True, timeout=120, env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'))
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        gate = load(scratch/'report/gate.json')
        self.assertEqual(gate['verdict'], 'NOT_QUALIFIED')
        self.assertEqual(gate['complete_worlds'], 1)
        self.assertEqual(load(scratch/'report/audit.json')['bootstrap_worlds'], [969420])
        self.assertIn('incomplete K8 panel', gate['reasons'])
        self.assertTrue((scratch/'report/onset_cdf_bands.csv').is_file())
        # Same code/recipe/name can still select different constructor variants.
        # Their deterministic receipts must prevent pooling across shard roots.
        roots = []
        for k, variant in enumerate(('encoder', 'population')):
            root = scratch/variant
            root.mkdir()
            identity = load(scratch/'normalized/IDENTITY.json')
            identity.update(construction_loader_sha256=digest(HERE/'run_wm.py'), wm={'variant': variant})
            save_new(root/'IDENTITY.json', identity)
            task = load(scratch/f'normalized/w969420_k{k}.json')
            task['identity_sha256'] = digest(root/'IDENTITY.json')
            save_new(root/f'w969420_k{k}.json', task)
            roots.append(root)
        with self.assertRaisesRegex(AssertionError, 'mixed model contracts'):
            read_predictions(roots, self.real, partial=True)
        # A candidate receipt is evidence only if its source ledger conserves.
        candidate = scratch/'future-builder'
        candidate.mkdir()
        identity = load(scratch/'normalized/IDENTITY.json')
        identity['future_builder_sha256'] = digest(HERE/'run_wm.py')
        save_new(candidate/'IDENTITY.json', identity)
        task = load(scratch/'normalized/w969420_k0.json')
        task['identity_sha256'] = digest(candidate/'IDENTITY.json')
        task['future_sampling'] = dict(population_audit=dict(finite=True, conserved=True,
            due=[[5, 2]]*5, initial_backlog=[[1, 0]]*5,
            admitted=[[4, 1]]*5, final_pending=[[2, 1]]*5))
        save_new(candidate/'w969420_k0.json', task)
        _, audit = read_predictions([candidate], self.real, partial=True)
        self.assertEqual(audit['tasks'], 1)
        # Construct a distinct immutable failing artifact rather than rewriting.
        corrupt = scratch/'corrupt-future-ledger'
        corrupt.mkdir()
        save_new(corrupt/'IDENTITY.json', identity)
        task['future_sampling']['population_audit']['admitted'] = [[5, 1]]*5
        save_new(corrupt/'w969420_k0.json', task)
        with self.assertRaisesRegex(AssertionError, 'source ledger mismatch'):
            read_predictions([corrupt], self.real, partial=True)
        other = scratch/'different-future-builder'
        other.mkdir()
        identity['future_builder_sha256'] = digest(HERE/'core.py')
        save_new(other/'IDENTITY.json', identity)
        task = load(candidate/'w969420_k0.json')
        task['identity_sha256'] = digest(other/'IDENTITY.json')
        task['future'] = 1
        for e in task['episodes']:
            e['future'] = 1
        save_new(other/'w969420_k1.json', task)
        with self.assertRaisesRegex(AssertionError, 'mixed model contracts'):
            read_predictions([candidate, other], self.real, partial=True)


if __name__ == '__main__':
    unittest.main()
