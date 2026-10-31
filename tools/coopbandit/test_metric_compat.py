import unittest
from .metric_compat import reference_sum


class MetricCompatibility(unittest.TestCase):
    def test_python310_scalar_order_is_preserved(self):
        self.assertEqual(reference_sum([1e16,1.,-1e16]),0.)
        self.assertEqual(reference_sum([.1,.1,.1]),(.1+.1)+.1)
        self.assertEqual(reference_sum([1,2,3]),6)
        self.assertEqual(reference_sum([],7),7)
        self.assertEqual(reference_sum((x for x in [1e16,1.,-1e16])),0.)


if __name__=='__main__': unittest.main()
