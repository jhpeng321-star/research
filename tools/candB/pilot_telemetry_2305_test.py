"""Synthetic paired-row credit proof; instrumentation must preserve gradients."""
import unittest
from types import SimpleNamespace


class TelemetryTest(unittest.TestCase):
    def test_group_terms_and_confident_gain_are_explicit(self):
        import torch
        from tools.candB.pilot_telemetry_2305 import group_credit
        variants = SimpleNamespace(base_row={0: 1}, individual_rows={(0, 0): 2, (0, 1): 3},
            groups=[dict(start=0, row=4, members=[0, 1], flips={})])
        returns = torch.tensor([12., 10., 11., 13., 17.], requires_grad=True)
        samples = torch.tensor([2., 0., 3., 4., 10.]).expand(8, 1, 5)
        output = group_credit(dict(variants=variants, returns=returns.reshape(1, 1, 5),
            gated300_long_difference=samples, gated300_reference='CTRL0'),
            dict(gated300_active_rows=[[False, False, False, False, True]]))
        dense, long = output['group_credit_terms']
        self.assertEqual([dense[k] for k in ('D_G', 'C_G', 'sum_D_i', 'D_all')], [7., 3., 4., 2.])
        self.assertEqual([long[k] for k in ('D_G', 'C_G', 'sum_D_i', 'D_all')], [10., 3., 7., 2.])
        self.assertEqual(output['confident_group_gain_rows'], [long])
        returns.sum().backward()
        self.assertTrue(torch.equal(returns.grad, torch.ones(5)))


if __name__ == '__main__':
    unittest.main()
