import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from . import archive_completed as a


class ArchiveSafety(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name).resolve()
        self.warm = self.home/'coopmwm20261102_warm1'
        self.ctx = patch.multiple(a, HOME=self.home, WARM=self.warm)
        self.ctx.start(); self.addCleanup(self.ctx.stop)
        self.refs = patch.multiple(a, open_references=lambda paths: [], binding_references=lambda paths: [])
        self.refs.start(); self.addCleanup(self.refs.stop)
        self.paths = []
        for n in ('interim008_A_1', 'interim008_B_1'):
            d = self.warm/'jobs'/n; d.mkdir(parents=True)
            (d/'RESULT.json').write_text('keep the outcome opaque')
            p = d/'observable.jsonl.gz'; p.write_bytes(n.encode()); self.paths.append(p)
        self.manifest = [dict(path=str(p.relative_to(self.home)), bytes=p.stat().st_size,
                              sha256=a.sha(p)) for p in self.paths]
        self.verified = dict(verified_files=2, verified_bytes=sum(x['bytes'] for x in self.manifest))

    def test_tampering_any_source_prevents_all_deletions(self):
        self.paths[1].write_bytes(b'changed')
        with self.assertRaises(AssertionError): a.remove_verified(self.manifest, self.verified)
        self.assertTrue(all(p.exists() for p in self.paths))

    def test_active_reader_blocks_removal(self):
        with patch.object(a, 'open_references', return_value=[{'pid': 1}]):
            with self.assertRaises(AssertionError): a.remove_verified(self.manifest, self.verified)
        self.assertTrue(all(p.exists() for p in self.paths))

    def test_verified_removal_keeps_results_and_resumes_partial_cleanup(self):
        self.paths[0].unlink()  # simulate cleanup interrupted after remote certificate
        a.remove_verified(self.manifest, self.verified)
        self.assertTrue(all(not p.exists() for p in self.paths))
        self.assertTrue(all((p.parent/'RESULT.json').exists() for p in self.paths))

    def test_bad_remote_count_blocks_removal(self):
        with self.assertRaises(AssertionError):
            a.remove_verified(self.manifest, dict(self.verified, verified_files=1))
        self.assertTrue(all(p.exists() for p in self.paths))

    def test_result_and_live_episode_are_never_candidates(self):
        with self.assertRaises(AssertionError): a.checked_tape(self.paths[0].parent/'RESULT.json')
        live = self.warm/'jobs'/'replay_u012_B_1'/'observable.jsonl.gz'
        with self.assertRaises(AssertionError): a.checked_tape(live)

    def test_incomplete_episode_blocks_removal(self):
        (self.paths[0].parent/'RESULT.json').unlink()
        with self.assertRaises(AssertionError): a.remove_verified(self.manifest, self.verified)
        self.assertTrue(all(p.exists() for p in self.paths))


if __name__ == '__main__': unittest.main()
