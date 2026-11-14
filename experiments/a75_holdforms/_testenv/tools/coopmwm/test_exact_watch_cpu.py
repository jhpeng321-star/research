"""Receipt publisher resilience; no training/physics/inference dispatch."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import exact_watch
from .storage import read,write


class ObserverTests(unittest.TestCase):
    def setUp(self):
        # The observer itself runs on Windows. Exercise its publish/deadline
        # contract on Linux too without importing a Windows-only lock module.
        import os,sys
        from types import SimpleNamespace
        if os.name != 'nt':
            stub=SimpleNamespace(LK_NBLCK=2,locking=lambda *args:None)
            mocked=patch.dict(sys.modules,{'msvcrt':stub})
            mocked.start();self.addCleanup(mocked.stop)

    def test_v14_uses_current_state_without_dispatch_or_legacy_gpu_status(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp);write(root/'EXACT_OBSERVER_V14_BINDING.json',dict(deadline_unix=1))
            stack.enter_context(patch.object(exact_watch.ops,'EVIDENCE',root))
            stack.enter_context(patch.object(exact_watch,'TAG','EXACT_OBSERVER_V14'))
            stack.enter_context(patch.object(exact_watch.time,'time',side_effect=[0,2,2,2]))
            stack.enter_context(patch.object(exact_watch.time,'sleep'))
            stack.enter_context(patch.object(exact_watch,'commit'))
            stack.enter_context(patch.object(exact_watch.exact_ops,'status',side_effect=AssertionError('legacy status forbidden')))
            launch=stack.enter_context(patch.object(exact_watch,'launch_ready',side_effect=AssertionError('no launches')))
            current=stack.enter_context(patch('tools.coopmwm.warm_watch.state',return_value=dict(unix=123,owners={})))
            publish=stack.enter_context(patch('tools.coopmwm.warm_publish.publish'))
            with self.assertRaisesRegex(ValueError,'registered GPU observation deadline reached'):exact_watch.run()
            launch.assert_not_called();publish.assert_called_once();current.assert_called_once()
            self.assertEqual(read(root/'EXACT_CURRENT_STATE_123.json')['unix'],123)

    def test_recovery_observer_cannot_launch_training(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp);write(root/'EXACT_OBSERVER_V8_BINDING.json',dict(deadline_unix=1))
            stack.enter_context(patch.object(exact_watch.ops,'EVIDENCE',root))
            stack.enter_context(patch.object(exact_watch,'TAG','EXACT_OBSERVER_V8'))
            stack.enter_context(patch.object(exact_watch.time,'time',side_effect=[0,2,2,2]))
            stack.enter_context(patch.object(exact_watch.time,'sleep'))
            stack.enter_context(patch.object(exact_watch,'commit'))
            stack.enter_context(patch.object(exact_watch.exact_ops,'status',return_value=({},dict(owners={},stops={}))))
            launch=stack.enter_context(patch.object(exact_watch,'launch_ready',side_effect=AssertionError('must not launch')))
            publish=stack.enter_context(patch('tools.coopmwm.warm_publish.publish'))
            with self.assertRaisesRegex(ValueError,'registered GPU observation deadline reached'):exact_watch.run()
            launch.assert_not_called();publish.assert_called_once()

    def test_operator_staging_defers_commit_without_stopping_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(exact_watch.ops,'EVIDENCE',root),patch.object(exact_watch.ops,'LOCAL',root),patch(
                    'tools.coopmwm.warm_v3_watch.commit',side_effect=ValueError('other staged work; preserve it')):
                exact_watch.commit()
            receipts=list(root.glob('EXACT_COMMIT_DEFERRED_*.json'))
            self.assertEqual(len(receipts),1)
            self.assertTrue(read(receipts[0])['index_untouched'])

    def test_non_staging_errors_still_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(exact_watch.ops,'EVIDENCE',root),patch.object(exact_watch.ops,'LOCAL',root),patch(
                    'tools.coopmwm.warm_v3_watch.commit',side_effect=ValueError('source mismatch')):
                with self.assertRaisesRegex(ValueError,'source mismatch'):exact_watch.commit()
            self.assertEqual(list(root.iterdir()),[])


if __name__=='__main__':unittest.main()
