"""Local-only tests of observable pressure and causal two-sensor coverage."""
import copy
import unittest
from .wm_tests import query
from .wm_contract import validate_query
from .collect_contexts import opportunities


class ContextTests(unittest.TestCase):
    def test_two_sensor_union_requires_explicit_consistent_packets(self):
        q = query()
        rows = q['record']['rows']
        extra = dict(vid='obstacle', s_m=75., v_mps=0., lane=1, is_cav=False, length_m=5.)
        rows.append(extra)  # 105m from main, only 95m from ramp
        with self.assertRaises(ValueError):
            validate_query(q)
        q['sensing_packets'] = {
            'main': dict(time_s=360, observed_lanes=[0, 1, 2], rows=copy.deepcopy(rows[:2])),
            'ramp': dict(time_s=360, observed_lanes=[0, 1, 2], rows=copy.deepcopy(rows))}
        validate_query(q)
        q['sensing_packets']['ramp']['time_s'] -= 1
        with self.assertRaises(ValueError):
            validate_query(q)

    def test_pressure_ignores_unobservables_and_respects_sensor_radius(self):
        rows = query()['record']['rows']
        first = opportunities(rows)
        rich = [dict(r, outcome=1e99, driver='never consumed') for r in rows]
        self.assertEqual(first, opportunities(rich))
        far = dict(vid='far', s_m=1000., v_mps=0., lane=2, is_cav=True, length_m=5.)
        self.assertEqual(first, opportunities(rows + [far]))


if __name__ == '__main__':
    unittest.main()
