import unittest

import numpy as np

from tools.rbeval.core import (breakdown, crps, load, HERE, onset_vector,
                                reduce_trajectory, wilson)


class ObservableContract(unittest.TestCase):
    def test_half_open_all_traffic_and_missing(self):
        # Birth, exit, invalid lane, endpoint: guard against survivor/endpoint bias.
        bins = reduce_trajectory(np.array([300., 309.5, 310., 900.]),
            np.full((4, 2), -10.), np.array([[10., 100.], [20., 30.], [40., 50.], [999., 999.]]),
            np.array([[0, -1], [0, 0], [0, 2], [0, 2]]),
            np.array([[1, 1], [1, 1], [0, 1], [1, 1]], bool))
        self.assertEqual(bins['upstream']['speed'][:2], [20., None])
        self.assertEqual(bins['upstream']['post_counts'][:2], [3, 0])
        self.assertEqual(bins['ramp']['speed'][:2], [None, 50.])
        self.assertEqual(sum(bins['upstream']['post_counts']), 3)

    def test_exact_detector_on_existing_receipts(self):
        real = load(HERE/'real_observables.json')['episodes']
        for e in real:
            for region, b in e['bins'].items():
                for h in (120, 300, 600):
                    got = breakdown(b['speed'][:6+h//10])
                    expected = b['events'][str(h)]
                    self.assertEqual((got['event'], got['onset_s'], got.get('preexisting')),
                                     (expected['event'], expected['onset_s'], expected.get('preexisting')))

    def test_crps_independent_cdf_integral(self):
        x = np.array([-4., 1., 3., 9.])
        y = 2.
        knots = np.sort(np.unique(np.r_[x, y]))
        mids = (knots[1:]+knots[:-1])/2
        score = np.sum(((x[:, None] <= mids).mean(0)-(y <= mids))**2*np.diff(knots))
        self.assertAlmostEqual(crps(x, y), score)
        self.assertLess(crps(x, y, fair=True), score)

    def test_censoring_and_precision(self):
        event = breakdown([25.]*66)
        vector = onset_vector(event, 600)
        self.assertEqual(vector[-1], 1.)
        self.assertTrue(np.all(vector[:-1] == 0))
        low, high = wilson(2, 2, .05/180)
        self.assertGreater((high-low)/2, .1)


if __name__ == '__main__':
    unittest.main()
