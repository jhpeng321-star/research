import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from . import episode_progress as ep


class EpisodeProgressTests(unittest.TestCase):
    def test_actual_payload_parallel_wait_is_elapsed_not_query_age_sum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);evidence=root/'evidence';evidence.mkdir()
            (root/'requests').mkdir();(root/'centre_requests').mkdir()
            job=root/'jobs/warm_u001_B_986002';job.mkdir(parents=True)
            claim=job/'CLAIM.json';claim.write_text('{"pid":99999999}')
            os.utime(claim,(10,10))
            for key,mtime in [('old',5),('one',20),('two',25),('resolved',30)]:
                request=root/'requests'/(key+'.json')
                request.write_text(json.dumps(dict(world_id=986002,time_s=300,state_id=key)))
                os.utime(request,(mtime,mtime))
            (root/'responses').mkdir();(root/'responses/resolved.json').write_text('{}')
            batches=types.ModuleType('tools.coopmwm.warm_protocol')
            batches.batch_worlds=lambda update:[dict(seed=986002)]
            store=types.ModuleType('tools.coopmwm.accel_storage')
            store.read=lambda p:json.loads(p.read_text())
            paths=types.ModuleType('pathlib')
            paths.Path=lambda *args:root if args==(ep.ROOT,) else Path(*args)
            def remote(host,source,timeout):
                compile(source,'<actual parallel progress payload>','exec')
                out=io.StringIO();saved=sys.path[:]
                try:
                    with patch.dict(sys.modules,{'pathlib':paths,'tools.coopmwm.warm_protocol':batches,'tools.coopmwm.accel_storage':store}),patch.object(os,'getloadavg',return_value=(0,0,0),create=True),patch('time.time',return_value=100),contextlib.redirect_stdout(out):exec(source,{})
                finally:sys.path[:]=saved
                return out.getvalue()
            with patch.object(ep.ops.ssh,'remote',side_effect=remote),patch.object(ep.ops,'EVIDENCE',evidence),patch.object(ep,'write'),contextlib.redirect_stdout(io.StringIO()):
                value=ep.collect(current=True)
            episode=value['episodes'][0]
            self.assertEqual(episode['queries'],3)
            self.assertEqual(episode['pending_query_count'],2)
            self.assertEqual(episode['current_query_wait_s'],80)
            self.assertEqual(episode['pending_query_age_sum_s'],155)
            self.assertEqual(episode['observed_sim_time_s'],300)
            self.assertTrue(episode['progress_censored'])
            self.assertEqual(value['episodes'][1]['current_query_wait_s'],0)
            self.assertEqual(value['episodes'][1]['pending_query_count'],0)

    def test_actual_payload_current_and_legacy_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);evidence=root/'evidence';evidence.mkdir()
            (root/'requests').mkdir();(root/'centre_requests').mkdir()
            (root/'B_UPDATE_001.json').write_text('{}')
            (root/'B_UPDATE_001_TIMING.json').write_text('{}')
            job=root/'jobs/warm_u002_B_986002';job.mkdir(parents=True)
            (job/'CLAIM.json').write_text('{"pid":99999999}')
            (job/'RESULT.json').write_text('{}')
            batches=types.ModuleType('tools.coopmwm.warm_protocol')
            batches.batch_worlds=lambda update:[dict(seed=986002)]
            store=types.ModuleType('tools.coopmwm.accel_storage')
            store.read=lambda p:json.loads(p.read_text())
            paths=types.ModuleType('pathlib')
            paths.Path=lambda *args:root if args==(ep.ROOT,) else Path(*args)
            def remote(host,source,timeout):
                compile(source,'<actual episode progress payload>','exec')
                out=io.StringIO();saved=sys.path[:]
                try:
                    with patch.dict(sys.modules,{'pathlib':paths,'tools.coopmwm.warm_protocol':batches,'tools.coopmwm.accel_storage':store}),patch.object(os,'getloadavg',return_value=(0,0,0),create=True),contextlib.redirect_stdout(out):exec(source,{})
                finally:sys.path[:]=saved
                return out.getvalue()
            with patch.object(ep.ops.ssh,'remote',side_effect=remote),patch.object(ep.ops,'EVIDENCE',evidence),patch.object(ep,'write'),contextlib.redirect_stdout(io.StringIO()):
                current=ep.collect(current=True)
                legacy=ep.collect()
            self.assertEqual(current['collections'],dict(B=2,R=1))
            self.assertEqual(legacy['collections'],dict(B=1,R=1))
            self.assertEqual(current['episodes'][0]['observed_sim_time_s'],1200)
            self.assertTrue(current['episodes'][0]['closed'])
            self.assertFalse(current['episodes'][1]['claimed'])


if __name__=='__main__':unittest.main()
