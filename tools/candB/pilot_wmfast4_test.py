"""S4 operational contracts; existing numerical parity/restore tests remain applicable."""
from pathlib import Path
import unittest
from unittest.mock import patch

from tools.candB import pilot_wmfast, pilot_wmfast_authority
from tools.candB import pilot_wmfast4, pilot_wmfast4_authority, pilot_wmfast4_control
from tools.candB.pilot_wmfast4_launch import launch_source
from tools.candB.pilot_lean import memory_bound


class S4BoundaryTests(unittest.TestCase):
    def test_s4_budget_and_peer_refusal(self):
        total = 49140 * 1024**2
        # The next clean-boundary runtime still re-measures coordinator reservation.
        self.assertLess(memory_bound(total, 2*1024**3, 0, 1, .75), .90*total)
        self.assertLess(memory_bound(total, 2*1024**3, 0, 2, .24), .90*total)
        with self.assertRaises(ValueError):
            memory_bound(total, 2*1024**3, 6*1024**3, 1, .75)
        with self.assertRaises(ValueError):
            memory_bound(total, 7*1024**3, 0, 1, .75)

    def test_real_launcher_and_controller_specialization(self):
        source = launch_source(Path('tools/candB/pilot_lean_launch.sh').read_text())
        for expected in ('WMS_SPEED_ES_STACK_SIDES=4', 'WMS_SPEED_FORWARD_MEMORY_FRACTION=.75',
                'WMS_SPEED_FUTURE_VJP_SLOTS=2', 'PILOT_WMFAST4_AUTHORITY.json',
                'tools.candB.pilot_wmfast4 --parent', '>= 46000'):
            self.assertIn(expected, source)
        self.assertNotIn('--new-phase', source)
        fn = pilot_wmfast4_control.controller_main()
        self.assertIn('WMFAST4_CONTROL_STOP', fn.__code__.co_consts)
        self.assertIn('WMFAST4_CONTROL_OWNERSHIP.json', fn.__code__.co_consts)

    def test_real_full_resume_adapter_receives_bound_s4_recipe(self):
        original_env = pilot_wmfast_authority.ENV
        def exercise(*args, **kwargs):
            self.assertEqual(pilot_wmfast_authority.ENV, pilot_wmfast4_authority.ENV)
            self.assertIs(pilot_wmfast_authority.recipe, pilot_wmfast4_authority.recipe)
            self.assertEqual(pilot_wmfast_authority.ENV['WMS_SPEED_ES_STACK_SIDES'], '4')
            self.assertTrue(kwargs['exploratory'])
            return 0
        with patch.object(pilot_wmfast, 'run', side_effect=exercise):
            self.assertEqual(pilot_wmfast4.run('authority', 'output', 'parent', exploratory=True), 0)
        self.assertIs(pilot_wmfast_authority.ENV, original_env)


if __name__ == '__main__':
    unittest.main()
