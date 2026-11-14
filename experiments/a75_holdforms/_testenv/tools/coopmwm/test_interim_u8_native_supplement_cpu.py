import copy
import json
import unittest

try:
    from interim_u8_native_supplement import analyze
except ImportError:
    from tools.coopmwm.interim_u8_native_supplement import analyze


class NativeSupplementTests(unittest.TestCase):
    def setUp(self):
        self.panel = [dict(seed=986000, cell=0, demand=2400, penetration=.2),
                      dict(seed=986001, cell=0, demand=2400, penetration=.2),
                      dict(seed=986002, cell=1, demand=1900, penetration=.05),
                      dict(seed=986003, cell=1, demand=1900, penetration=.05)]
        self.records = [dict(arm=a, world=w, metrics={h: dict(team_delay_vs=j,
            collisions=0, teleports=0, ttc_below_3_vs=0, hard_brake=0)
            for h in ('600', '900')}) for a, j in
            (('A', 100), ('B', 90), ('R', 100), ('mappo', 100), ('native', 120))
            for w in self.panel]

    def test_native_pairing_and_zero_count_serialization(self):
        v = analyze(self.records, self.panel)
        row = v['results']['900:all:B-native:team_delay_vs']
        self.assertAlmostEqual(row['mean_diff_vs'], -30)
        self.assertEqual(v['focus_worlds'], 2)
        self.assertEqual(len(v['results']), 20)
        self.assertTrue(v['no_training_selection'])
        json.dumps(v, allow_nan=False)

    def test_reject_duplicate_or_missing_native(self):
        for records in (self.records + [self.records[-1]], self.records[:-1]):
            with self.assertRaises(ValueError):
                analyze(records, self.panel)

    def test_reject_cell_identity_change(self):
        records = copy.deepcopy(self.records)
        records[0]['world']['cell'] = 99
        with self.assertRaises(ValueError):
            analyze(records, self.panel)


if __name__ == '__main__':
    unittest.main()
