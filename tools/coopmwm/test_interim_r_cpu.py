import unittest
from .interim_r import supplement_specs


class InterimSupplementTests(unittest.TestCase):
    def panel(self):
        return [dict(seed=990600+i) for i in range(96)]

    def test_only_r_fixed_u8_and_restricted_phase(self):
        specs = supplement_specs(self.panel(), {'interim008_A_990600'})
        self.assertEqual(sum(s['sim_s'] for s in specs), 115200)
        self.assertTrue(all(s['arm']=='R' and s['checkpoint_update']==8
            and s['phase']=='interim_u8_test' for s in specs))

    def test_reject_charged_r_without_replay(self):
        with self.assertRaisesRegex(ValueError, 'never retry'):
            supplement_specs(self.panel(), {'interim008_R_990600'})

    def test_reject_incomplete_and_duplicate_panel(self):
        for panel in (self.panel()[:95], [self.panel()[0]]*96):
            with self.assertRaises(ValueError):supplement_specs(panel, set())
