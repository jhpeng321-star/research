import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from .v2v_archive_scope import job_info,completed_updates,pending_refs,select,job_refs


class V2VArchiveScopeTests(unittest.TestCase):
    def setup_pool(self,d):
        r=Path(d);(r/'jobs').mkdir();old=time.time()-7200
        for u in (1,2):
            for a in 'ABR':
                p=r/f'{a}_UPDATE_{u:03}.json';p.write_text(json.dumps(dict(arm=a,update=u)));os.utime(p,(old,old))
                job=r/'jobs'/f'v2v1_warm_u{u:03}_{a}_986000';job.mkdir()
                for name in ('RESULT.json','RESULT.json.gz','observable.jsonl.gz','CLOSE.json','CLAIM.json'):
                    p=job/name;p.write_text('{}');os.utime(p,(old,old))
        return r,old

    def test_only_previous_fully_completed_all_arm_update_can_move(self):
        with tempfile.TemporaryDirectory() as d:
            r,_=self.setup_pool(d);latest,eligible=completed_updates(r,time.time())
            self.assertEqual(latest,dict(A=2,B=2,R=2));self.assertEqual(eligible,{1})
            paths,_=select(r,set(),set(),set(),set(),time.time())
            self.assertEqual(len(paths),9)
            self.assertTrue(all('_u001_' in p for p in paths))
            self.assertFalse(any(Path(p).name in ('CLOSE.json','CLAIM.json') for p in paths))

    def test_pending_reference_open_fd_and_bank_protection_exclude_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            r,_=self.setup_pool(d);job=r/'jobs'/'v2v1_warm_u001_B_986000'
            paths,_=select(r,{str(job/'RESULT.json')},{str(job/'RESULT.json.gz')},set(),{str(job)},time.time())
            self.assertFalse(any(str(job) in p for p in paths))
            (r/'requests').mkdir();(r/'requests/q.json').write_text(json.dumps(dict(job=str(job))))
            self.assertIn(str(job),pending_refs(r))
            # A runtime's run_root/jobs-root declaration must not turn into a
            # blanket protection of every finished episode.
            refs=job_refs(r,{str(r),str(r/'jobs'),str(job)})
            self.assertEqual(refs,{str(job)})

    def test_incomplete_recent_and_invalid_receipts_never_qualify(self):
        with tempfile.TemporaryDirectory() as d:
            r,_=self.setup_pool(d);p=r/'R_UPDATE_002.json';p.write_text('{')
            self.assertEqual(completed_updates(r,time.time())[1],set())
            p.write_text(json.dumps(dict(arm='B',update=2)))
            with self.assertRaisesRegex(ValueError,'identity'):completed_updates(r,time.time())

    def test_no_checkpoints_bank_sources_scenarios_or_unregistered_worlds(self):
        r=Path('/example/run')
        for name in ('B_checkpoint_001.pt','replay_bank/x/RESULT.json','jobs/v2v1_warm_u001_B_964000/RESULT.json',
            'jobs/v2v1_warm_u001_B_986000/scenario.xml','jobs/v2v1_warm_u001_B_986000/CLAIM.json'):
            self.assertIsNone(job_info(r,r/name))
        self.assertEqual(job_info(r,r/'jobs/v2v1_warm_u001_B_986000/RESULT.json'),(1,'B'))


if __name__=='__main__':unittest.main()
