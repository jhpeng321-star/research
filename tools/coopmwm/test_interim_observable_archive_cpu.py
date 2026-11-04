import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from tools.coopmwm import interim_observable_archive as a


class InterimObservableArchive(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.job = self.root/'jobs'/'v2vtest008_B_989600'
        self.job.mkdir(parents=True)
        (self.root/'final_receipts').mkdir()
        self.tape = self.job/'observable.jsonl.gz'
        self.tape.write_bytes(b'opaque tape')
        self.result = self.job/'RESULT.json'
        self.result.write_bytes(b'OUTCOME MUST NEVER BE DECODED')
        self.close = self.job/'CLOSE.json'
        self.close.write_text('{}')
        self.complete = self.root/'test8_COMPLETE.json'
        self.complete.write_text('{}')
        self.public = self.root/'final_receipts'/(self.job.name+'.json')
        self.public.write_text(json.dumps(dict(id=self.job.name, arm='B', world=dict(seed=989600),
            checkpoint_update=8, kind='episode', outcomes_closed=True,
            files={'observable.jsonl.gz': dict(sha256=self.digest(self.tape))})))
        self.now = 10000
        for p in (self.tape, self.close, self.complete, self.public):
            os.utime(p, (1, 1))

    @staticmethod
    def digest(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    def select(self, **kw):
        v = dict(root=self.root, opened=set(), files=set(), trees=set(), refs=set(),
                 now=self.now, digest=self.digest)
        v.update(kw)
        return a.select(**v)

    def test_only_completed_interim_tape_selected_without_outcome_decode(self):
        self.assertEqual(list(self.select()), [str(self.tape)])
        self.assertEqual(self.result.read_bytes(), b'OUTCOME MUST NEVER BE DECODED')
        for p in (self.result, self.job/'SNAPSHOT.pt', self.root/'jobs'/'v2vtest024_B_989600'/'observable.jsonl.gz',
                  self.root/'jobs'/'v2vtest008_B_986000'/'observable.jsonl.gz'):
            self.assertIsNone(a.tape_info(self.root, p))

    def test_incomplete_or_failed_or_young_test_excluded(self):
        os.utime(self.complete, (self.now, self.now))
        self.assertFalse(self.select())
        os.utime(self.complete, (1, 1))
        (self.root/'test8_FAILED.json').write_text('{}')
        self.assertFalse(self.select())
        (self.root/'test8_FAILED.json').unlink()
        self.close.unlink()
        self.assertFalse(self.select())

    def test_live_and_bank_references_excluded(self):
        for kw in (dict(opened={str(self.tape)}), dict(files={str(self.tape)}),
                   dict(trees={str(self.job)}), dict(refs={str(self.job)}),
                   dict(refs={str(self.tape)})):
            self.assertFalse(self.select(**kw))
        self.assertTrue(self.select(refs={str(self.root), str(self.root/'jobs')}))

    def test_original_fingerprint_and_identity_fail_closed(self):
        self.tape.write_bytes(b'changed')
        os.utime(self.tape, (1, 1))
        with self.assertRaisesRegex(ValueError, 'original completion'):
            self.select()
        self.tape.write_bytes(b'opaque tape')
        os.utime(self.tape, (1, 1))
        v = json.loads(self.public.read_text())
        v['checkpoint_update'] = 24
        self.public.write_text(json.dumps(v))
        os.utime(self.public, (1, 1))
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            self.select()

    def test_fresh_reference_or_age_change_blocks_second_check(self):
        self.assertTrue(self.select())
        self.assertFalse(self.select(refs={str(self.job)}))
        os.utime(self.tape, (self.now, self.now))
        self.assertFalse(self.select())

    def test_other_roots_and_traversal_never_selected(self):
        self.assertIsNone(a.tape_info(self.root, self.root/'other'/'jobs'/self.job.name/self.tape.name))
        self.assertIsNone(a.tape_info(self.root, self.job/'..'/self.job.name/self.tape.name))
        self.assertEqual(a.tape_info(self.root, self.root/'jobs'/'v2vtest008_R_990659'/self.tape.name), ('R', 990659))


if __name__ == '__main__':
    unittest.main()
