"""Synthetic paired-row credit proof; instrumentation must preserve gradients."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class TelemetryTest(unittest.TestCase):
    def test_minima_use_final_acting_accelerations_and_restore_patches(self):
        import torch
        from tools.candB.pilot_telemetry_2305 import telemetry
        from wms.control import execution, policy14, policy14_cli as cli
        frame = dict(s_m=torch.zeros(1, 2), v_mps=torch.ones(1, 2),
                     present=torch.ones(1, 2, dtype=torch.bool),
                     is_cav=torch.ones(1, 2, dtype=torch.bool), lane=torch.zeros(1, 2))
        batch = SimpleNamespace(state=frame)
        result = torch.zeros(1, 2, 4)
        result[0, :, 3] = torch.tensor([-.25, -2.])
        def fake_score(*args, **kwargs):
            callback = policy14.make_cav_controller(act_mask=torch.tensor([[True, False]]))
            self.assertTrue(torch.equal(callback(frame), result))
            return {}
        with patch.object(policy14, 'make_cav_controller', return_value=lambda frame: result), \
                patch.object(cli, 'independent_score', fake_score), \
                patch.object(execution, 'executed_residual', return_value=torch.zeros(1, 2)):
            original = cli.independent_score
            with telemetry():
                score = cli.independent_score(None, batch)
            self.assertIs(cli.independent_score, original)
        self.assertEqual(score['executed_residual_minima'], dict(min_delta_mps2=-.25,
            min_braking_magnitude_mps2=.25, min_executed_acceleration_mps2=-.25))

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
