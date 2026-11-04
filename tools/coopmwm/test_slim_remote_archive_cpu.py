import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from tools.coopmwm.protocol import digest
from tools.coopmwm.slim_remote_archive import FLOOR, commit_branch, file_identity, validate_proof


class SlimRemoteArchiveTests(unittest.TestCase):
    def fixture(self, root):
        files=[]
        for name in ('RESULT.json.gz','RESULT.json'):
            p=root/name;p.write_bytes(name.encode())
            files.append(dict(source=str(p),relative=name,**file_identity(p)))
        body={'series':{'300':0.,'301':1.}}
        prepared=dict(entries=files,body_sha256=digest(body),archive='/remote/only-copy')
        proof=dict(hashes={e['relative']:e['sha256'] for e in files},free=FLOOR+1000)
        slim=SimpleNamespace(SLIM='SLIM.json',PARTIAL='PARTIAL.json',SCHEMA='test',SlimPartial=RuntimeError)
        return slim,body,prepared,proof

    def test_missing_or_bad_proof_never_unlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();s,b,p,proof=self.fixture(root)
            for bad in ({'hashes':{},'free':FLOOR+1000},dict(proof,free=FLOOR-1),dict(proof,hashes={k:'bad' for k in proof['hashes']})):
                with self.assertRaises(ValueError):
                    commit_branch(s,root,b,p,bad,opened=set,revalidate=lambda:None,write=lambda *a:self.fail('write before proof'))
                self.assertTrue(all(Path(e['source']).exists() for e in p['entries']))

    def test_changed_or_open_source_never_unlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();s,b,p,proof=self.fixture(root)
            with self.assertRaises(ValueError):
                commit_branch(s,root,b,p,proof,opened=lambda:{p['entries'][0]['source']},revalidate=lambda:None,write=lambda *a:self.fail('open source'))
            Path(p['entries'][0]['source']).write_bytes(b'changed')
            with self.assertRaises(ValueError):
                commit_branch(s,root,b,p,proof,opened=set,revalidate=lambda:None,write=lambda *a:self.fail('changed source'))
            self.assertTrue(all(Path(e['source']).exists() for e in p['entries']))

    def test_eligibility_or_slim_change_never_unlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();s,b,p,proof=self.fixture(root)
            with self.assertRaises(ValueError):
                commit_branch(s,root,dict(b,changed=True),p,proof,opened=set,revalidate=lambda:None,write=lambda *a:self.fail('changed SLIM'))
            def revoked():raise ValueError('no longer rejected')
            with self.assertRaises(ValueError):
                commit_branch(s,root,b,p,proof,opened=set,revalidate=revoked,write=lambda *a:self.fail('eligibility revoked'))
            self.assertTrue(all(Path(e['source']).exists() for e in p['entries']))

    def test_slim_written_before_first_unlink(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();s,b,p,proof=self.fixture(root)
            def write(path,body):
                self.assertTrue(all(Path(e['source']).exists() for e in p['entries']))
                path.write_text(json.dumps(body))
            commit_branch(s,root,b,p,proof,opened=set,revalidate=lambda:None,write=write)
            self.assertEqual(json.loads((root/'SLIM.json').read_text()),b)
            self.assertTrue(all(not Path(e['source']).exists() for e in p['entries']))

    def test_partial_move_retains_remaining_sources_and_records_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();s,b,p,proof=self.fixture(root)
            original=Path.unlink
            def unlink(path,*args,**kwargs):
                if path.name=='RESULT.json':raise OSError('injected unlink fault')
                return original(path,*args,**kwargs)
            def write(path,body):path.write_text(json.dumps(body))
            fake=SimpleNamespace(_seal=lambda body,key:dict(body,**{key:'test-seal'}))
            with patch.dict('sys.modules',{'tools.coopmwm.exact_grade':fake}),patch.object(Path,'unlink',unlink):
                with self.assertRaisesRegex(RuntimeError,'all verified archive copies retained'):
                    commit_branch(s,root,b,p,proof,opened=set,revalidate=lambda:None,write=write)
            self.assertTrue((root/'RESULT.json').is_file())
            self.assertEqual(json.loads((root/'PARTIAL.json').read_text())['unlinked'],['RESULT.json.gz'])
            # The commit boundary has no remote-delete primitive or cleanup callback.
            self.assertEqual(p['archive'],'/remote/only-copy')

    def test_rejects_incomplete_proof(self):
        with self.assertRaises(ValueError):
            validate_proof([dict(relative='one',sha256='ok')],dict(hashes={'one':'ok','extra':'ok'},free=FLOOR+100))


if __name__=='__main__':unittest.main()
