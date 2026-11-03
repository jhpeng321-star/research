import importlib.util
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

HERE=Path(__file__).parent
sys.path.insert(0,str(HERE))
import rolling_archive as m


class RollingArchiveSafety(unittest.TestCase):
    def test_diagnostic_scope_excludes_bank(self):
        root=m.base.RUN/'policy_synergy_isolated_1226_v2/u010_B/branches/vid/branch'
        self.assertEqual(m.diagnostic_kind(root/'SNAPSHOT.pt'),(10,'B'))
        self.assertEqual(m.diagnostic_kind(root/'observable.jsonl.gz'),(10,'B'))
        self.assertEqual(m.diagnostic_kind(root/'RESULT.json'),(10,'B'))
        self.assertIsNone(m.diagnostic_kind(m.base.RUN/'replay_bank/B/SNAPSHOT.pt'))

    def test_scope_never_accepts_outcomes_or_final_detector_inputs(self):
        root=m.base.RUN/'jobs'
        self.assertIsNone(m.job_kind(root/'fixedfinal_B_989600'/'observable.jsonl.gz'))
        self.assertIsNone(m.job_kind(root/'warm_u005_B_986000'/'RESULT.json'))
        self.assertEqual(m.job_kind(root/'replay0735_warm_u005_B_986000'/'observable.jsonl.gz'),(5,'B'))
        self.assertIsNone(m.job_kind(root/'warm_u005_B_964000'/'observable.jsonl.gz'))

    def test_destination_floor_includes_whole_pending_batch(self):
        self.assertFalse(m.capacity(m.FLOOR-1,0))
        self.assertFalse(m.capacity(m.FLOOR+100,101))
        self.assertTrue(m.capacity(m.FLOOR+100,100))

    def test_sha_mismatch_and_changed_inode_retain_source(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'tape';p.write_bytes(b'observable');st=p.stat()
            e=dict(source=str(p),sha256=m.base.digest_file(p),inode=st.st_ino,
                bytes=st.st_size,mtime_ns=st.st_mtime_ns)
            with self.assertRaises(ValueError):m.remove_one(e,'wrong',{str(p)},set())
            self.assertTrue(p.exists())
            with patch.object(m,'safe',return_value=True),patch.object(m,'aged',return_value=True):
                with self.assertRaises(ValueError):m.remove_one(dict(e,inode=-1),e['sha256'],{str(p)},set())
            self.assertTrue(p.exists())

    def test_new_reference_or_open_fd_prevents_deletion(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'tape';p.write_bytes(b'observable')
            e=dict(source=str(p),sha256=m.base.digest_file(p))
            self.assertFalse(m.remove_one(e,e['sha256'],set(),set()))
            self.assertFalse(m.remove_one(e,e['sha256'],{str(p)},{str(p)}))
            self.assertTrue(p.exists())

    def test_only_exact_verified_aged_source_is_removed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'tape';p.write_bytes(b'observable');st=p.stat()
            e=dict(source=str(p),sha256=m.base.digest_file(p),inode=st.st_ino,
                bytes=st.st_size,mtime_ns=st.st_mtime_ns)
            with patch.object(m,'safe',return_value=True),patch.object(m,'aged',return_value=True):
                self.assertTrue(m.remove_one(e,e['sha256'],{str(p)},set()))
            self.assertFalse(p.exists())


if __name__=='__main__':unittest.main()
