import hashlib
import unittest

from tools.coopmwm.legacy_tail_archive import build_worker, in_scope


class LegacyTailArchiveTests(unittest.TestCase):
    def path(self, update=19, arm='B', seed=986000, leaf='RESULT.json'):
        return f'/home/ubuntu/coopmwm20261102_warm1/jobs/replay0735_warm_u{update:03}_{arm}_{seed}/{leaf}'

    def test_only_five_completed_updates(self):
        for arm in 'ABR':
            for u in range(1, 49):
                for leaf in ('RESULT.json', 'RESULT.json.gz', 'observable.jsonl.gz', 'SNAPSHOT.pt'):
                    expected = arm in 'BR' and 19 <= u <= (21 if arm == 'B' else 20) and leaf.startswith('RESULT.')
                    self.assertEqual(in_scope(self.path(u, arm, leaf=leaf)), expected)

    def test_rejects_nontrain_roots_and_traversal(self):
        for seed in (985999, 986072, 989600, 964000, 962001, 977001):
            self.assertFalse(in_scope(self.path(seed=seed)))
        self.assertFalse(in_scope(self.path().replace('warm1', 'v2v_coop3')))
        self.assertFalse(in_scope(self.path().replace('/RESULT', '/../RESULT')))
        self.assertFalse(in_scope(self.path().replace('/RESULT.json', '/nested/RESULT.json')))

    def test_transfer_body_preserved_and_guards_inserted(self):
        anchor = "sys.path.insert(0,'/home/ubuntu/coopmwm20261104_archiveflatcpu1');import rolling_archive as m"
        source = 'def in_scope(name):\n return False\n' + anchor + '\n# unchanged SHA transfer body\n'
        worker = build_worker(source, hashlib.sha256(source.encode()).hexdigest())
        self.assertTrue(worker.endswith(anchor + '\n# unchanged SHA transfer body\n'))
        self.assertIn("authority['permanent_stop_sha256']", worker)
        self.assertIn('fcntl.LOCK_EX|fcntl.LOCK_NB', worker)
        self.assertIn("'adopt existing tail archive owner'", worker)
        namespace = {}
        exec(worker.split('import fcntl')[0].split(anchor)[0], namespace)
        self.assertTrue(namespace['in_scope'](self.path(21)))
        self.assertFalse(namespace['in_scope'](self.path(21, 'R')))

    def test_refuses_source_or_abi_drift(self):
        with self.assertRaises(ValueError):
            build_worker('def in_scope(name):\n return False\n', 'wrong')
        source = 'def in_scope(name):\n return False\n'
        with self.assertRaises(ValueError):
            build_worker(source, hashlib.sha256(source.encode()).hexdigest())


if __name__ == '__main__':
    unittest.main()
