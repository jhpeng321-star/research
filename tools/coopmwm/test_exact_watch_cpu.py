"""Receipt publisher resilience; no training/physics/inference dispatch."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import exact_watch
from .storage import read,write


class ObserverTests(unittest.TestCase):
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
