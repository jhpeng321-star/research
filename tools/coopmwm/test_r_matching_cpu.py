import unittest
from .r_matching import nearest, report


class MatchingTests(unittest.TestCase):
    def ticket(self, ident=0, size=3, regime='ordinary', duration=180):
        return dict(id=ident, group_size=size, regime=regime, duration_s=duration)

    def group(self, size):
        return dict(size=size, members=[str(i) for i in range(size)])

    def test_exact_precedes_size_and_regime_fallback(self):
        g, t = nearest([self.group(2), self.group(3)],
                       [self.ticket(0, 2, 'merge-dense'), self.ticket(1)], set(), 'ordinary')
        self.assertEqual((g['size'], t['id']), (3, 1))

    def test_nearest_size_retains_duration(self):
        g, t = nearest([self.group(4)], [self.ticket()], set(), 'ordinary')
        self.assertEqual((g['size'], t['duration_s']), (4, 180))

    def test_regime_fallback_and_used_ticket_exclusion(self):
        g, t = nearest([self.group(3)], [self.ticket(0), self.ticket(1, regime='merge-dense')], {0}, 'ordinary')
        self.assertEqual(t['id'], 1)

    def test_size_distance_over_one_never_matches(self):
        self.assertEqual(nearest([self.group(6)], [self.ticket()], set(), 'ordinary'), (None, None))

    def test_missing_stratum_recorded_without_assertion(self):
        r = report([self.ticket(), self.ticket(1, 4)], [self.ticket()])
        self.assertEqual(r['count_shortfall'], 1)
        self.assertFalse(r['exact_match'])
        self.assertEqual(sum(s['shortfall'] for s in r['strata']), 1)

    def test_fallback_reports_both_shortfall_and_excess(self):
        r = report([self.ticket()], [self.ticket(size=4, regime='merge-dense')])
        self.assertEqual(r['count_shortfall'], 0)
        self.assertEqual(sum(s['shortfall'] for s in r['strata']), 1)
        self.assertEqual(sum(s['excess'] for s in r['strata']), 1)


if __name__ == '__main__':
    unittest.main()
