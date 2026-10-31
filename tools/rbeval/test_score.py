"""Scientific invariants: leakage, clustered uncertainty, ties and identity safety."""
import sys
import copy
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import ARMS, HERE, digest, load, save_new
from score import WorldBootstrap, eligible_peers, rank_stats, read_predictions


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

    def test_world_pairing_does_not_create_fake_sample_size(self):
        boot = WorldBootstrap(self.real, B=1000)
        pairs = [(969420, 0.), (969426, 2.)]
        single, duplicate = boot.bands([pairs, pairs*5])
        self.assertEqual(single, duplicate)
        self.assertEqual(single['worlds'], 2)
        self.assertIsNone(boot.bands([[(969420, 1.)]])[0]['low'])

    def test_finite_K_uncertainty_is_included(self):
        boot = WorldBootstrap(self.real, B=1000)
        # Real outcomes agree across worlds, but finite-K event probability varies.
        band = boot.bands([[(969420, .5, [0., 1.]*4), (969426, .5, [0., 1.]*4)]])[0]
        self.assertIsNotNone(band['low'])
        self.assertLess(band['low'], .5)
        self.assertGreater(band['high'], .5)

    def test_chunking_preserves_joint_family_maxstat(self):
        boot = WorldBootstrap(self.real, B=1000)
        features = [[(969420, v, [v-1, v+1]*4), (969426, 2*v, [2*v-2, 2*v+2]*4)]
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
        self.assertIn('incomplete K8 panel', gate['reasons'])
        self.assertTrue((scratch/'report/onset_cdf_bands.csv').is_file())


if __name__ == '__main__':
    unittest.main()
