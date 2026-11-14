"""Safety gates use synthetic files only; never contact a host or archive real data."""
import tempfile
import os
from pathlib import Path
import unittest
from unittest.mock import patch
if __package__:
    from . import archive_move as a
else:
    import archive_move as a


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='E:/Temp/opencode' if os.name=='nt' else '/tmp')
        self.home=Path(self.temp.name);self.run=self.home/'coopmwm20261102_warm1'
        self.ls=self.home/'coopmwm20261102_lscollect1'
        self.patch=patch.multiple(a,HOME=self.home,RUN=self.run,LS=self.ls)
        self.patch.start()

    def tearDown(self):
        self.patch.stop();self.temp.cleanup()

    def entry(self):
        p=self.ls/'label_attempts/SUMOENV1/branches/sv_fixture/policy/RESULT.json'
        p.parent.mkdir(parents=True);p.write_bytes(b'fixture-only')
        s=p.stat()
        return dict(source=str(p),sha256=a.digest_file(p),inode=s.st_ino,bytes=s.st_size),p

    def test_scope_keeps_all_forbidden_and_scenario_files(self):
        entry,p=self.entry();self.assertTrue(a.safe_file(p))
        for name in ('driver_routes.rou.xml','driver_params.json','checkpoint_003.pt','vType.json'):
            q=p.parent/name;q.write_bytes(b'never-read')
            self.assertFalse(a.safe_file(q))
        q=self.home/'outside/RESULT.json';q.parent.mkdir();q.write_bytes(b'outside')
        self.assertFalse(a.safe_file(q))
        self.assertFalse(a.allowed_source(self.run/'jobs/interim008_B_989600/RESULT.json'))
        self.assertTrue(a.allowed_source(self.run/'jobs/interim008_B_989600/observable.jsonl.gz'))

    def test_remote_mismatch_never_deletes(self):
        e,p=self.entry()
        with self.assertRaisesRegex(ValueError,'remote SHA'):
            a.delete_verified(e,'wrong',set())
        self.assertTrue(p.exists())

    def test_changed_or_open_source_never_deletes(self):
        e,p=self.entry()
        self.assertFalse(a.delete_verified(e,e['sha256'],{str(p)}))
        self.assertTrue(p.exists());p.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'source changed'):
            a.delete_verified(e,e['sha256'],set())
        self.assertTrue(p.exists())

    def test_only_both_matching_fingerprints_delete(self):
        e,p=self.entry();self.assertTrue(a.delete_verified(e,e['sha256'],set()))
        self.assertFalse(p.exists())

    def test_large_verification_payload_uses_stdin_not_an_argv_element(self):
        import types
        code='x=' + repr('x'*300000)
        with patch.object(a.subprocess,'run',return_value=types.SimpleNamespace(returncode=0,stdout='{}')) as call:
            self.assertEqual(a.remote(code),{})
        args,kwargs=call.call_args
        self.assertEqual(args[0][-1],'/root/miniconda3/bin/python -')
        self.assertEqual(kwargs['input'],code)
        self.assertLess(max(map(len,args[0])),1000)

    def test_partial_resume_requires_recorded_deletion_and_exact_manifest(self):
        e,p=self.entry();e['relative']=str(p.relative_to(self.home))
        left,done=a.resumed_entries([e],[]);self.assertEqual(left,[e]);self.assertEqual(done,[])
        p.unlink()
        with self.assertRaisesRegex(ValueError,'missing source'):
            a.resumed_entries([e],[])
        receipt=dict(verified=[dict(relative=e['relative'],sha256=e['sha256'],bytes=e['bytes'])],deleted_sources=[e['relative']])
        left,done=a.resumed_entries([e],[receipt]);self.assertEqual(left,[]);self.assertEqual(done,[e])
        receipt['verified'][0]['sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'differs from manifest'):
            a.resumed_entries([e],[receipt])


if __name__=='__main__':unittest.main()
