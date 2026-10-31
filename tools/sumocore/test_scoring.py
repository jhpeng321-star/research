"""Check proper-score algebra and whole-world clustering independently."""
import unittest
import numpy as np
from eval_core import crps
from score import bootstrap_family,by_world


class Scores(unittest.TestCase):
    def test_empirical_and_fair_against_double_sum(self):
        x=np.array([-10.,1.,100.]);y=6.
        pair=np.abs(x[:,None]-x[None,:]).sum()
        self.assertAlmostEqual(crps(x,y),np.abs(x-y).mean()-.5*pair/9)
        self.assertAlmostEqual(crps(x,y,True),np.abs(x-y).mean()-.5*pair/6)
        self.assertIsNone(crps([],y))

    def test_arm_rows_are_not_independent_worlds(self):
        rows=[dict(world=969421,x=0),dict(world=969421,x=2),dict(world=969422,x=3)]
        feature=by_world(rows,lambda r:r['x'])
        self.assertEqual(feature,{969421:1.,969422:3.})
        b=bootstrap_family([feature],B=10000)[0]
        self.assertEqual(b['worlds'],2)
        self.assertEqual(b['mean'],2)
        self.assertLessEqual(b['low'],1)
        self.assertGreaterEqual(b['high'],3)

    def test_singleton_uncertainty_unavailable(self):
        b=bootstrap_family([{969435:7.}],B=10000)[0]
        self.assertEqual(b['worlds'],1)
        self.assertIsNone(b['low']);self.assertIsNone(b['high'])


if __name__=='__main__':unittest.main()
