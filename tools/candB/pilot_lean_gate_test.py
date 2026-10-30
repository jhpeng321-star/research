"""CPU-only checks for four-start adapter and pending-call rescue mechanism."""
import unittest

from tools.candB.pilot_lean_gate import panel_gate


class LeanGateTests(unittest.TestCase):
    def test_all_four_noninferiority(self):
        row = panel_gate([0, 1, 2, 3], [1, 1, 2, 3], update=191, panel_sha256='prefix')
        self.assertTrue(row['passed'])
        self.assertEqual(row['delta_per_start'], [-1, 0, 0, 0])
        self.assertFalse(row['deployment_authorized'])
        self.assertEqual(row['schema'], 'EXPLORATORY-LEAN_native_panel/1')

    def test_harm_not_averaged_away(self):
        self.assertFalse(panel_gate([-10, 0, 0, 1], [0]*4, update=191)['passed'])

    def test_finite_and_length(self):
        for p in ([0]*16, [0]*3, [0, 0, 0, float('nan')], [0, 0, 0, float('inf')]):
            with self.assertRaises(ValueError):
                panel_gate(p, [0]*4, update=191)


if __name__ == '__main__':
    unittest.main()
