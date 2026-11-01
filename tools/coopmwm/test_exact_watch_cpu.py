"""Receipt publisher resilience; no training/physics/inference dispatch."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import exact_watch
from .storage import read


class ObserverTests(unittest.TestCase):
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
