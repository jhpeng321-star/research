"""Transferred completion reports must retain their exact hash and gate chain."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .core import digest, save_new
from .verify_wm2 import point_delta, verify_report_hashes


class ReportTransferSafety(unittest.TestCase):
    def test_cross_platform_roundoff_does_not_hide_changed_points(self):
        self.assertLess(point_delta('248.81050505054884', '248.81050505054907'), 1e-10)
        self.assertEqual(point_delta('native', 'native'), 0.)
        for a, b in (('248.8', '248.9'), ('native', 'ctrl0'), ('nan', '2')):
            with self.assertRaises(AssertionError):
                point_delta(a, b)

    def test_changed_report_or_gate_cannot_verify(self):
        parent = Path('E:/Temp/opencode') if os.name == 'nt' else Path('/tmp')
        self.assertTrue(parent.is_dir())
        root = Path(tempfile.mkdtemp(prefix='rbeval-report-verification-', dir=parent))
        gate = dict(verdict='SYNTHETIC_TEST_ONLY')
        save_new(root/'gate.json', gate)
        save_new(root/'audit.json', dict(synthetic_test=True))
        receipt = dict(gate=gate, report_hashes={p.name: digest(p) for p in root.iterdir()})
        verify_report_hashes(receipt, root)
        with patch('tools.rbeval.verify_wm2.digest', return_value='changed'):
            with self.assertRaisesRegex(AssertionError, 'report hash mismatch'):
                verify_report_hashes(receipt, root)
        with self.assertRaisesRegex(AssertionError, 'gate receipt mismatch'):
            verify_report_hashes(dict(receipt, gate=dict(verdict='CHANGED')), root)
        save_new(root/'extra.json', dict(synthetic_test=True))
        with self.assertRaisesRegex(AssertionError, 'report file set changed'):
            verify_report_hashes(receipt, root)


if __name__ == '__main__':
    unittest.main()
