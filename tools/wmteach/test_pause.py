"""The durable owner hold must block dispatch before any remote operation."""
import unittest
from unittest.mock import patch

from tools.wmteach import auto_live, scale_watch


class PauseTests(unittest.TestCase):
    def test_live_hold_precedes_preparation_and_remote_launch(self):
        with patch('pathlib.Path.exists', return_value=True), \
             patch.object(auto_live, 'prepare', side_effect=AssertionError('must stay idle')), \
             patch.object(auto_live, 'remote', side_effect=AssertionError('must not dispatch')):
            self.assertEqual(auto_live.advance()['state'], 'PAUSED_BY_MAIN')
            self.assertEqual(auto_live.launch()['state'], 'PAUSED_BY_MAIN')

    def test_collector_hold_precedes_lock_receipt_and_child_creation(self):
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.open', side_effect=AssertionError('must not start collector')), \
             patch.object(scale_watch.subprocess, 'Popen', side_effect=AssertionError('must not spawn')):
            self.assertIsNone(scale_watch.main())
            self.assertEqual(scale_watch.launch()['state'], 'PAUSED_BY_MAIN')


if __name__ == '__main__':
    unittest.main()
