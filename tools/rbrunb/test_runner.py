"""Ownership, readiness, resource admission and full-panel integrity regressions."""
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import supervise
import watch


class RunnerTests(unittest.TestCase):
    def setUp(self):
        parent = Path('E:/Temp/opencode')
        assert parent.is_dir()
        self.root = parent/('rbrunb_test_'+uuid.uuid4().hex)
        self.root.mkdir()
        self.claims = self.root/'claims'
        self.claims.mkdir()
        self.patcher = patch.object(watch, 'SHARED', self.claims)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        # Own generated test fixtures are retained; no cleanup/deletions.

    def write(self, path, value):
        supervise.save(path, value)

    def test_ready_required_and_heading_boundaries(self):
        self.write(self.claims/'READY.json', {'variants': [{'variant': 'A'}, {'variant': 'B'}]})
        self.assertEqual(watch.manifests(['## Status\nNothing ready\n'])[0], [])
        records, _ = watch.manifests(['## READY FOR EVAL\n- READY.json\n## Later\nignored\n'])
        self.assertEqual([r['variant'] for r in records], ['A', 'B'])
        self.assertNotIn('ignored', watch.ready_sections('## READY FOR EVAL\nA\n## Later\nignored'))
        self.assertIn('candidate', watch.ready_sections('## READY FOR EVAL\n### B\ncandidate\n## Later\nno'))
        self.assertEqual(watch.ready_sections('## NOT READY FOR EVAL\nold.json'), '')

    def test_runa_first_and_peer_claim_never_stolen(self):
        records = [dict(variant=n, host='gpuR') for n in ('A', 'B', 'C')]
        self.assertEqual([r['variant'] for r in watch.eligible(records)], ['B', 'C'])
        self.write(self.claims/'B.claim', {'owner': 'S-RB-RUNA'})
        self.assertEqual([r['variant'] for r in watch.eligible(records)], ['C'])
        self.write(self.claims/'C.claim', {'owner': watch.OWNER})
        self.assertEqual(watch.eligible(records)[0]['variant'], 'C')
        self.write(self.claims/'C.done', {'worlds': 20})
        self.assertEqual(watch.eligible(records), [])

    def test_claim_exclusive_create(self):
        f = self.claims/'B.claim'
        watch.save_new(f, {'owner': 'S-RB-RUNA'})
        with self.assertRaises(FileExistsError):
            watch.save_new(f, {'owner': watch.OWNER})
        self.assertEqual(json.loads(f.read_text())['owner'], 'S-RB-RUNA')

    def test_failed_claim_preserved_without_blocking_next_variant(self):
        records = [dict(variant=n, host='gpuR') for n in ('A', 'B', 'C')]
        self.write(self.claims/'B.claim', {'owner': watch.OWNER})
        self.write(self.claims/'B.failed.json', {'automatic_retry': False})
        self.assertEqual([r['variant'] for r in watch.eligible(records)], ['C'])
        self.assertTrue((self.claims/'B.claim').exists())

    def test_no_dispatch_without_ready_or_peer_claim(self):
        with patch.object(watch, 'ssh', side_effect=AssertionError('No dispatch allowed')):
            self.assertEqual(watch.manifests(['# Nothing ready'])[0], [])
            self.write(self.claims/'B.claim', {'owner': 'S-RB-RUNA'})
            self.assertEqual(watch.eligible([{'variant': 'A'}, {'variant': 'B'}]), [])

    def test_resource_admission_caps_total_and_waits_for_busy_gpu(self):
        self.assertEqual(supervise.capacity(dict(load=1, utilization=0, used_mib=1)), 3)
        self.assertEqual(supervise.capacity(dict(load=1, utilization=100, used_mib=15000)), 0)
        self.assertEqual(supervise.capacity(dict(load=24, utilization=0, used_mib=1)), 0)
        for used in range(0, 24564, 23):
            n = supervise.capacity(dict(load=1, utilization=0, used_mib=used))
            self.assertLessEqual(n, 3)
            if n:
                self.assertLessEqual(used+n*5120+1024, 20000)

    def test_prohibited_paths_rejected_before_access(self):
        for word in ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1',
                     'hetdata_truth', 'vehicle_params', 'driver_params', 'vType_ASSIGNMENT'):
            with self.assertRaises(ValueError):
                supervise.path_ok('/root/'+word+'/file')

    def test_full_panel_hashes_and_mutation(self):
        for i in range(3):
            d = self.root/f'shard{i}'
            d.mkdir()
            self.write(d/'IDENTITY.json', {'shard': i, 'K': 8})
            identity = supervise.digest(d/'IDENTITY.json')
            hashes = {}
            for j, seed in enumerate(range(969420, 969440)):
                if j % 3 != i:
                    continue
                for k in range(8):
                    name = f'w{seed}_k{k}.json'
                    self.write(d/name, dict(seed=seed, future=k, identity_sha256=identity))
                    hashes[name] = supervise.digest(d/name)
            self.write(d/'COMPLETE.json', dict(tasks=len(hashes), hashes=hashes, identity_sha256=identity))
        self.assertEqual(supervise.verify_panel(self.root, 3), 160)
        path = self.root/'shard0'/'w969420_k0.json'
        with path.open('a') as f:
            f.write(' ')
        with self.assertRaises(AssertionError):
            supervise.verify_panel(self.root, 3)

    def test_changed_candidate_claim_stops_before_ssh(self):
        record = dict(variant='B', host='gpuR', model_name='B', entrypoint='/root/candidate.py',
                      runtime_root='/root/runtime', wm='/root/model.pt', wm_config='/root/config.toml',
                      smoke_receipt='/root/smoke.json', smoke_sha256='0'*64)
        original = dict(record, model_name='different')
        self.write(self.claims/'B.claim', dict(owner=watch.OWNER, manifest=original))
        with patch.object(watch, 'ssh', side_effect=AssertionError('SSH should not run')):
            with self.assertRaisesRegex(AssertionError, 'candidate changed'):
                watch.launch_or_status(record, self.root, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
